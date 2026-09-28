################################################################
#   filename: pglLabJack.py
#    purpose: Device class for interfacing with LabJack T7
#             for analog and digital IO
#         by: JLG
#       date: Jan 27, 2026
################################################################

###########
# Import
##########
import io
import threading
import time
import numpy as np
from .pglTimestamp import pglTimestamp
from .pglDevice import pglDigitalIODevice, pglAnalogInputDevice, pglAnalogTraceData
import matplotlib.pyplot as plt
from .pglMessages import pglMessages
import time

class pglLabJack(pglDigitalIODevice, pglAnalogInputDevice):
    def __init__(self):
        self.h = None
        self._digitalIOLock = threading.Lock()
        self._wordPulseLock = threading.Lock()
        self.wordDIOBits = []
        self.wordDIOMask = 0
        self.wordPulseLen = None
        
        self.isReading = False
        self.acquisitionThread = None
        self.stopEvent = threading.Event()
        self.bufferLock = threading.Lock()
        self.analogBuffer = []

        self.digitalOutputConfigured = False
        self.analogInputConfigured = False
        super().__init__(deviceType="LabJack")
        
        # import library, checking for errors
        try:
            from labjack import ljm
            # keep ljm reference
            self.ljm = ljm
        except ImportError: 
            pglMessages.warning("Labjack library is not installed. Please install LJM Library to use LabJack.\n Installation is available from: https://support.labjack.com/docs/ljm-software-installer-macos-x64\nAfter downloading, install into pgl pip environment: python -m pip install labjack-ljm")
            return
        
        try:
            # open LabJack device
            self.h = ljm.openS("ANY", "USB", "ANY")
        except Exception as e:
            if getattr(e, "errorCode", None) == 1227:
                pglMessages.warning(f"(pglLabJack) No LabJack device found: {e}")
            else:
                pglMessages.warning(f"(pglLabJack) Error opening LabJack device: {e}")
            self.h = None
            return
        
        if self.h is not None:
            # get handle info
            (deviceType, connectionType, self.serialNumber, self.ipAddress, self.port, self.maxBytesPerMB)= ljm.getHandleInfo(self.h)
            
            # get device type as a string
            deviceTypeStrings = {
                ljm.constants.dtT4: "T4",
                ljm.constants.dtT7: "T7",
                ljm.constants.dtT8: "T8"
            }
            self.type = deviceTypeStrings.get(deviceType, "Unknown")
            
            # get connection types as a string
            connectionTypeStrings = {
                ljm.constants.ctUSB: "USB",
                ljm.constants.ctETHERNET: "Ethernet",
                ljm.constants.ctWIFI: "WiFi",
                ljm.constants.ctANY: "Any"
            }
            self.connectionType = connectionTypeStrings.get(connectionType, "Unknown")
            print(f"(pglLabJack) Opened {self.type} LabJack device via {self.connectionType} connection.")
            print(f"             serialNumber: {self.serialNumber} ipAddress: {self.ipAddress} port: {self.port} maxBytesPerMB: {self.maxBytesPerMB}")
    
            # set description
            self.deviceDescription = f"{self.type} LabJack via {self.connectionType}"
               
    @property
    def isActive(self):
        return True if self.h is not None else False
            
    def __repr__(self):
        if self.h is None:
            return "<pglLabJack device not connected>"
        else:
            return f"<pglLabJack deviceType={self.type} connectionType={self.connectionType} serialNumber={self.serialNumber}>"
    
    def setupDigitalOutput(self, channel=0, pulseLen=5, groupName="FIO", channelName=None, **kwargs):
        """
        Configure a digital output.

        Args:
            channel (int): Logical channel ID. Also used as the physical pin
                index when channelName is None.
            pulseLen (int): Pulse duration in milliseconds.
            groupName (str): Hardware bank used when channelName is None,
                e.g. "FIO", "EIO", "CIO", "MIO", or "DIO".
            channelName (str or None): Explicit physical pin name, e.g. "EIO4".
                Overrides groupName and the physical interpretation of channel.
        """
        if self.h is None:
            pglMessages.warning("(pglLabJack:setupDigitalOutput) LabJack device not connected.", level=1)
            self.digitalOutputConfigured = False
            return False

        if isinstance(pulseLen, bool) or not isinstance(pulseLen, (int, float)) or not np.isfinite(pulseLen) or pulseLen <= 0:
            pglMessages.warning("pulseLen must be a positive finite number")
            return False

        if not isinstance(channel, int) or channel < 0:
            pglMessages.warning("channel must be a nonnegative integer", level=2)
            return False
    
        if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
            pglMessages.warning("channel must be a nonnegative integer", level=2)
            return False

        if channelName is None:
            # Treat an unspecified group as the default hardware bank.
            groupName = "FIO" if groupName is None else groupName
            validChannelGroups = {"FIO", "EIO", "CIO", "MIO", "DIO"}

            if groupName not in validChannelGroups:
                pglMessages.warning(f"Invalid channel group: {groupName}")
                return False

            channelName = f"{groupName}{channel}"

        if not isinstance(channelName, str) or not channelName:
            pglMessages.warning("channelName must be a nonempty string", level=2)
            return False

        try:
            # For individual LJM T-series digital I/O registers,
            # writing LOW also configures the pin as an output.
            if self.type == "T7":
                address, _ = self.ljm.nameToAddress(channelName)
                dioBaseAddress, _ = self.ljm.nameToAddress("DIO0")
                if not 0 <= address - dioBaseAddress < 23:
                    pglMessages.warning(f"{channelName!r} is not an individual T7 DIO register")
                    return False

            with self._digitalIOLock:
                self.ljm.eWriteName(self.h, channelName, 0)
        except Exception as e:
            pglMessages.warning(f"(pglLabJack:setupDigitalOutput) Error setting up {channelName}: {e}")
            self.digitalOutputConfigured = False
            return False

        super().setupDigitalOutput(channel, pulseLen)
        self.digitalChannels[channel]["name"] = channelName
        self.digitalOutputConfigured = True

        pglMessages.message(f"Logical channel {channel}: {channelName} configured as output, set to LOW")
        return True

    def digitalOutput(self, channel, state):
        """Set a configured channel; return a host completion timestamp or None."""
        if self.h is None:
            pglMessages.warning("LabJack is not connected")
            return None

        config = self.digitalChannels.get(channel)
        if config is None or "name" not in config:
            pglMessages.warning(f"Digital output channel {channel!r} has not been configured")
            return None

        channelName = config["name"]

        try:
            with self._digitalIOLock:
                self.ljm.eWriteName(self.h, channelName, 1 if state else 0)
                timestamp = pglTimestamp.getSecs()
            return timestamp
        except Exception as e:
            pglMessages.warning(f"Error writing {channelName}: {e}")
            return None
          
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000, voltageRange=10.0):
        '''
        Start analog input reading from specified channels.
        
        Args:
            duration (float): Duration of recording in seconds
            channels (list): List of channel numbers or names
            scanRate (int): Sampling rate in Hz
            scansPerRead (int): Number of scans per read operation
            voltageRange (float): Voltage range for analog inputs. Options: 10.0V, 1.0V, 0.1V, 0.01V

        '''
        if self.h is None:
            print("(pglLabJack:startAnalogRead) LabJack device not connected.")
            return
        
        if self.isReading:
            pglMessages.warning("Analog acquisition is already active")
            return

        # Convert channel numbers to AIN names if needed
        channelAddresses = []
        for ch in channels:
            if isinstance(ch, int):
                channelAddresses.append(f"AIN{ch}")
            else:
                channelAddresses.append(ch)  # Already a string like "AIN0"
        
        # validate range 
        validRanges = [10.0, 1.0, 0.1, 0.01]
        if voltageRange not in validRanges:
            print(f"(pglLabJack:startAnalogRead) Invalid range {voltageRange}V. Valid options: {validRanges}")
            return
        try:
            # set each channel to the specified range
            for channel in channelAddresses:
                self.ljm.eWriteName(self.h, f"{channel}_RANGE", voltageRange)
        except Exception as e:
            print(f"(pglLabJack:startAnalogRead) Error setting range: {e}")
            return

        # save parameters
        self.channels = channelAddresses
        self.scanRate = scanRate
        self.scansPerRead = scansPerRead
        self.range = voltageRange
        self.analogStreamDuration = duration

        # derived parameters
        self.numChannels = len(channels)
        self.totalScans = int(duration * scanRate)
        self.totalReads = int(np.ceil(self.totalScans / scansPerRead))

        if self.totalScans % scansPerRead != 0:
            print(f"(pglLabJack:startAnalogRead) totalScans ({self.totalScans}) is not an integer multiple of scansPerRead ({scansPerRead}). Will collect {self.totalReads * scansPerRead} samples instead of {self.totalScans} and throw out extra samples.")
            
        # buffer and synchronization
        self.analogBuffer = []
        self.bufferLock = threading.Lock()
        self.stopEvent = threading.Event()

        # state flag
        self.isReading = True

        # start acquisition thread
        self.acquisitionThread = threading.Thread(
            target=self._analogReadThread,
            daemon=True
        )
        self.acquisitionThread.start()
           
    def _analogReadThread(self):
        """
        Thread function to read analog data from LabJack
        """
        
        # record the start time of the stream
        self.analogStartTimestamp = pglTimestamp.getSecs()
        
        # Convert channel names to addresses
        try:
            channelAddresses = self.ljm.namesToAddresses(self.numChannels, self.channels)[0]
        except Exception as e:
            print(f"(pglLabJack:analogReadThread) Error converting channel names: {e}")
            self.isReading = False
            return

        # start stream
        try:
            self.scanRate = self.ljm.eStreamStart(
                self.h,
                self.scansPerRead,
                self.numChannels,
                channelAddresses,
                self.scanRate
            )
        except Exception as e:
            print(f"(pglLabJack:analogReadThread) Error starting stream: {e}")
            self.isReading = False
            return

        # keep getting data until duration is reached or stop event is set
        try:
            while not self.stopEvent.is_set():
                if (pglTimestamp.getSecs() - self.analogStartTimestamp) >= self.analogStreamDuration:
                    break

                # read the data from labJack stream
                dataArray, deviceBacklog, ljmBacklog = self.ljm.eStreamRead(self.h)

                # copy over the data that was received
                with self.bufferLock:
                    self.analogBuffer.extend(dataArray)

        finally:
            try:
                # stop the stream
                self.ljm.eStreamStop(self.h)
            except Exception:
                pass

            self.isReading = False

    def stopAnalogRead(self, waitToFinish=False, doNotTruncate=False):
        """
        Stop the analog reading and return time and data arrays.
        
        Args:
            waitToFinish (bool): If True, waits for the acquisition thread to finish before returning data.
                                 If False, signals the thread to stop and returns immediately with whatever data has been collected so far.
            doNotTruncate (bool): If True, do not truncate the data to the exact number of samples.
                                 If False (default), truncates the data to the expected number of samples based on duration and scan rate.  
        Returns:
            data: pglAnalogTraceData which holds time and data
        """
        if self.h is None:
            pglMessages.warning("Device not initialized")
            return None

        # If acquisition is active, request stop when not waiting
        if not waitToFinish and self.isReading:
            self.stopEvent.set()

        # If acquisition thread exists, wait for it to finish
        if self.acquisitionThread is not None and self.acquisitionThread.is_alive():
            pglMessages.message("waiting for analog acquisition to end")
            self.acquisitionThread.join()

        # copy data safely
        with self.bufferLock:
            data = np.array(self.analogBuffer)

        if data.size == 0:
            pglMessages.warning("No data read")
            return None

        # Reshape data to separate channels
        # data shape will be (numSamples, numChannels)
        numSamples = len(data) // self.numChannels
        data = data[:numSamples * self.numChannels] 
        data = data.reshape(numSamples, self.numChannels)

        # truncate to exact number of samples
        if not doNotTruncate and numSamples > self.totalScans:
            data = data[:self.totalScans, :]
            numSamples = self.totalScans

        sampleTimes = np.arange(numSamples) / self.scanRate
        return pglAnalogTraceData(time=sampleTimes, data=data, channelNames=self.channels)            
              
    def __del__(self):
        """
        Clean up the labJack instance
        """
        self.close()
            
    def close(self):
        """Stop acquisition, wait for word output to finish, and close the device."""
        self.stopEvent.set()

        thread = self.acquisitionThread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join()

        with self._wordPulseLock:
            with self._digitalIOLock:
                if self.h is not None:
                    self.ljm.close(self.h)
                    self.h = None
                    self.digitalOutputConfigured = False
        
    def setupDigitalOutputWord(self, channels=None, **kwargs):
        """Configure T7 word channels, ordered from lowest to highest word bit."""
        if self.h is None:
            pglMessages.warning("LabJack is not connected")
            return False

        if self.type != "T7":
            pglMessages.warning(f"Port-based word output has only been implemented for T7; connected device is {self.type}")
            return False

        if channels is None:
            pglMessages.warning("No digital word channels supplied")
            return False

        channels = list(channels)
        if not channels:
            pglMessages.warning("At least one digital word channel is required")
            return False

        if len(channels) != len(set(channels)):
            pglMessages.warning("Digital word channels must be unique")
            return False

        missingChannels = set(channels) - self.digitalChannels.keys()
        if missingChannels:
            pglMessages.warning(f"Channels have not been configured: {sorted(missingChannels)}")
            return False

        if not self._wordPulseLock.acquire(blocking=False):
            pglMessages.warning("Cannot configure word channels while a word pulse is active")
            return False

        try:
            dioBaseAddress, _ = self.ljm.nameToAddress("DIO0")
            dioBits = []

            for channel in channels:
                channelName = self.digitalChannels[channel]["name"]
                address, _ = self.ljm.nameToAddress(channelName)
                dioBit = address - dioBaseAddress

                if not 0 <= dioBit < 23:
                    pglMessages.warning(f"{channelName!r} is not an individual T7 DIO register")
                    return False

                dioBits.append(dioBit)

            if len(dioBits) != len(set(dioBits)):
                pglMessages.warning("Multiple logical channels refer to the same physical DIO pin")
                return False

            pulseLengths = {self.digitalChannels[channel]["pulseLen"] for channel in channels}
            if len(pulseLengths) != 1:
                pglMessages.warning("All channels in a digital word must have the same pulseLen")
                return False

            pulseLen = next(iter(pulseLengths))
            pulseMicroseconds = int(round(pulseLen * 1000))
            if not 1 <= pulseMicroseconds <= 100000:
                pglMessages.warning("Hardware-timed word pulse must be between 1 microsecond and 100 milliseconds")
                return False
            
            # Verify access to the registers used by this implementation.
            with self._digitalIOLock:
                self.ljm.eReadName(self.h, "DIO_STATE")
                self.ljm.eReadName(self.h, "DIO_INHIBIT")
                self.ljm.nameToAddress("WAIT_US_BLOCKING")

            self.wordDigitalChanels = channels
            self.wordBits = len(channels)
            self.wordMaxValue = (1 << self.wordBits) - 1
            self.wordDIOBits = dioBits
            self.wordDIOMask = sum(1 << bit for bit in dioBits)
            self.wordPulseLen = pulseLen

            return True

        except Exception as e:
            pglMessages.warning(f"Error configuring digital word output: {e}")
            return False

        finally:
            self._wordPulseLock.release()
          
    def digitalOutputPulse(self, channel):
        """
        Pulse one configured T7 digital output using a device-side delay.

        Returns:
            float or None: Host timestamp immediately before submitting the pulse
                sequence, or None on error. This is not a measured hardware onset.

        Note:
            Blocks until the pulse finishes. The output is left LOW.
        """
        if self.h is None:
            pglMessages.warning("LabJack is not connected")
            return None

        if self.type != "T7":
            pglMessages.warning("Device-timed digital pulses are currently implemented only for T7")
            return None

        config = self.digitalChannels.get(channel)
        if config is None or "name" not in config:
            pglMessages.warning(f"Digital output channel {channel!r} has not been configured")
            return None

        channelName = config["name"]

        try:
            pulseLen = config["pulseLen"]
            if isinstance(pulseLen, bool) or not isinstance(pulseLen, (int, float)) or not np.isfinite(pulseLen):
                pglMessages.warning("pulseLen must be a finite number")
                return None

            pulseMicroseconds = int(round(pulseLen * 1000))
            if not 1 <= pulseMicroseconds <= 100000:
                pglMessages.warning("Device-timed pulse must be between 1 microsecond and 100 milliseconds")
                return None

            with self._digitalIOLock:
                # Recheck after acquiring the lock in case the device was closed.
                if self.h is None:
                    pglMessages.warning("LabJack is not connected")
                    return None

                try:
                    timestamp = pglTimestamp.getSecs()
                    self.ljm.eWriteNames(self.h, 3, [channelName, "WAIT_US_BLOCKING", channelName], [1, pulseMicroseconds, 0])
                except Exception:
                    # The HIGH write may have succeeded before a later failure.
                    try:
                        self.ljm.eWriteName(self.h, channelName, 0)
                    except Exception as cleanupError:
                        pglMessages.warning(f"Could not clear {channelName} after pulse failure: {cleanupError}")
                    raise

            return timestamp

        except Exception as e:
            pglMessages.warning(f"Error sending digital pulse on {channelName}: {e}")
            return None      
        
    def digitalOutputWord(self, outputWord):
        """
        Pulse a T7 digital word using a device-side blocking delay.

        The duration comes from the shared pulseLen of the configured word channels.

        Returns:
            float or None: Host timestamp immediately before submitting the pulse
                sequence, or None on error. This is not a measured hardware onset.

        Note:
            This method blocks until the pulse sequence and inhibit restoration finish.
        """
        if self.h is None or not self.wordDIOBits:
            pglMessages.warning("Digital word output has not been configured")
            return None

        if self.type != "T7":
            pglMessages.warning("Hardware-timed word output is currently implemented only for T7")
            return None

        if isinstance(outputWord, bool) or not isinstance(outputWord, int):
            pglMessages.warning("outputWord must be an integer")
            return None

        if not 0 <= outputWord <= self.wordMaxValue:
            pglMessages.warning(f"outputWord must be between 0 and {self.wordMaxValue}: {outputWord}")
            return None

        pulseMicroseconds = int(round(self.wordPulseLen * 1000))
        if not 1 <= pulseMicroseconds <= 100000:
            pglMessages.warning("Hardware-timed word pulse must be between 1 microsecond and 100 milliseconds")
            return None

        if not self._wordPulseLock.acquire(blocking=False):
            pglMessages.warning("Previous digital word pulse is still active")
            return None

        try:
            physicalState = sum(((outputWord >> iBit) & 1) << dioBit for iBit, dioBit in enumerate(self.wordDIOBits))
            inhibitMask = ((1 << 23) - 1) ^ self.wordDIOMask

            with self._digitalIOLock:
                if self.h is None:
                    pglMessages.warning("LabJack is not connected")
                    return None

                originalInhibit = int(self.ljm.eReadName(self.h, "DIO_INHIBIT"))

                try:
                    # Protect all pins outside the configured word.
                    self.ljm.eWriteName(self.h, "DIO_INHIBIT", inhibitMask)

                    # All three operations are submitted in one request.
                    timestamp = pglTimestamp.getSecs()
                    self.ljm.eWriteNames(self.h, 3, ["DIO_STATE", "WAIT_US_BLOCKING", "DIO_STATE"], [physicalState, pulseMicroseconds, 0])

                except Exception:
                    # Execution may have stopped after setting the word HIGH.
                    try:
                        self.ljm.eWriteName(self.h, "DIO_INHIBIT", inhibitMask)
                        self.ljm.eWriteName(self.h, "DIO_STATE", 0)
                    except Exception as cleanupError:
                        pglMessages.warning(f"Could not clear word pins after failure: {cleanupError}")
                    raise

                finally:
                    self.ljm.eWriteName(self.h, "DIO_INHIBIT", originalInhibit)

            return timestamp

        except Exception as e:
            pglMessages.warning(f"Error sending digital word {outputWord}: {e}")
            return None

        finally:
            self._wordPulseLock.release()
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
        Configure a T7 digital output and cache its device-timed pulse request.

        Args:
            channel (int): Logical channel ID. Also the physical pin index when
                channelName is omitted.
            pulseLen (float): Pulse duration in milliseconds.
            groupName (str or None): Hardware bank; defaults to FIO.
            channelName (str or None): Explicit physical pin name.

        Returns:
            bool: True on success, False on failure.
        """
        if self.h is None:
            pglMessages.warning("LabJack is not connected")
            return False

        if self.type != "T7":
            pglMessages.warning(f"This digital output implementation currently supports T7 only; connected device is {self.type}")
            return False

        if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
            pglMessages.warning("channel must be a nonnegative integer")
            return False

        if isinstance(pulseLen, bool) or not isinstance(pulseLen, (int, float)) or not np.isfinite(pulseLen) or not 0 < pulseLen <= 100:
            pglMessages.warning("pulseLen must be a positive finite number no greater than 100 milliseconds")
            return False

        pulseMicroseconds = int(round(pulseLen * 1000))
        if pulseMicroseconds < 1:
            pglMessages.warning("Pulse duration must round to at least 1 microsecond")
            return False

        if channelName is None:
            groupName = "FIO" if groupName is None else groupName
            if groupName not in {"FIO", "EIO", "CIO", "MIO", "DIO"}:
                pglMessages.warning(f"Invalid channel group: {groupName}")
                return False
            channelName = f"{groupName}{channel}"

        if not isinstance(channelName, str) or not channelName:
            pglMessages.warning("channelName must be a nonempty string")
            return False

        # Use the same lock order as word setup, sending, and shutdown.
        if not self._wordPulseLock.acquire(blocking=False):
            pglMessages.warning("Cannot configure digital output while word output or configuration is active")
            return False

        try:
            channelAddress, channelType = self.ljm.nameToAddress(channelName)
            dioBaseAddress, _ = self.ljm.nameToAddress("DIO0")
            dioBit = channelAddress - dioBaseAddress

            if not 0 <= dioBit < 23:
                pglMessages.warning(f"{channelName!r} is not an individual T7 DIO register")
                return False

            # Do not silently invalidate an existing word configuration.
            if channel in self.wordDigitalChanels:
                pglMessages.warning(f"Channel {channel} is already part of a configured word; reset the configuration before changing it")
                return False

            waitAddress, waitType = self.ljm.nameToAddress("WAIT_US_BLOCKING")

            with self._digitalIOLock:
                if self.h is None:
                    pglMessages.warning("LabJack is not connected")
                    return False

                # Writing the individual pin LOW also sets it as an output.
                self.ljm.eWriteAddress(self.h, channelAddress, channelType, 0)

                super().setupDigitalOutput(channel, pulseLen)
                self.digitalChannels[channel].update({
                    "name": channelName,
                    "address": channelAddress,
                    "dataType": channelType,
                    "dioBit": dioBit,
                    "pulseMicroseconds": pulseMicroseconds,
                    "pulseAddresses": [channelAddress, waitAddress, channelAddress],
                    "pulseTypes": [channelType, waitType, channelType],
                    "pulseValues": [1, pulseMicroseconds, 0],
                })
                self.digitalOutputConfigured = True

            pglMessages.message(f"Logical channel {channel}: {channelName} configured as output, set to LOW")
            return True

        except Exception as e:
            pglMessages.warning(f"Error configuring digital channel {channel}: {e}")
            return False

        finally:
            self._wordPulseLock.release()

    def digitalOutput(self, channel, state):
        """Set a configured pin; return a host completion timestamp or None."""
        with self._digitalIOLock:
            if self.h is None:
                pglMessages.warning("LabJack is not connected")
                return None

            config = self.digitalChannels.get(channel)
            if config is None:
                pglMessages.warning(f"Digital output channel {channel!r} has not been configured")
                return None

            try:
                self.ljm.eWriteAddress(self.h, config["address"], config["dataType"], 1 if state else 0)
                return pglTimestamp.getSecs()
            except Exception as e:
                pglMessages.warning(f"Error writing digital channel {channel}: {e}")
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
        """
        Configure T7 word channels in lowest-to-highest bit order.

        All selected channels must have the same configured pulseLen.
        Caches DIO_INHIBIT; external changes to that register are not supported.

        Returns:
            bool: True on success, False on failure.
        """
        if channels is None:
            pglMessages.warning("No digital word channels supplied")
            return False

        channels = list(channels)
        if not channels:
            pglMessages.warning("At least one digital word channel is required")
            return False

        if any(isinstance(channel, bool) or not isinstance(channel, int) or channel < 0 for channel in channels):
            pglMessages.warning("Word channels must be nonnegative integer logical IDs")
            return False

        if len(channels) != len(set(channels)):
            pglMessages.warning("Digital word channels must be unique")
            return False

        if not self._wordPulseLock.acquire(blocking=False):
            pglMessages.warning("Cannot configure word channels while word output or configuration is active")
            return False

        try:
            with self._digitalIOLock:
                if self.h is None:
                    pglMessages.warning("LabJack is not connected")
                    return False

                if self.type != "T7":
                    pglMessages.warning(f"Digital word output currently supports T7 only; connected device is {self.type}")
                    return False

                missingChannels = set(channels) - self.digitalChannels.keys()
                if missingChannels:
                    pglMessages.warning(f"Channels have not been configured: {sorted(missingChannels)}")
                    return False

                dioBits = [self.digitalChannels[channel]["dioBit"] for channel in channels]
                if len(dioBits) != len(set(dioBits)):
                    pglMessages.warning("Multiple logical channels refer to the same physical DIO pin")
                    return False

                pulseLengths = {self.digitalChannels[channel]["pulseLen"] for channel in channels}
                if len(pulseLengths) != 1:
                    pglMessages.warning("All channels in a digital word must have the same pulseLen")
                    return False

                pulseLen = next(iter(pulseLengths))
                pulseMicroseconds = self.digitalChannels[channels[0]]["pulseMicroseconds"]

                stateAddress, stateType = self.ljm.nameToAddress("DIO_STATE")
                inhibitAddress, inhibitType = self.ljm.nameToAddress("DIO_INHIBIT")
                waitAddress, waitType = self.ljm.nameToAddress("WAIT_US_BLOCKING")

                # Read once at configuration time, not on every send.
                originalInhibit = int(self.ljm.eReadAddress(self.h, inhibitAddress, inhibitType))
                self.ljm.eReadAddress(self.h, stateAddress, stateType)

                physicalMasks = [1 << bit for bit in dioBits]
                wordMask = sum(physicalMasks)
                inhibitMask = ((1 << 23) - 1) ^ wordMask

                self.wordDigitalChanels = channels
                self.wordBits = len(channels)
                self.wordMaxValue = (1 << self.wordBits) - 1
                self.wordDIOBits = dioBits
                self.wordDIOMask = wordMask
                self.wordPulseLen = pulseLen

                self.wordOriginalInhibit = originalInhibit
                self.wordInhibitMask = inhibitMask
                self.wordPhysicalMasks = physicalMasks
                self.wordPulseAddresses = [inhibitAddress, stateAddress, waitAddress, stateAddress, inhibitAddress]
                self.wordPulseTypes = [inhibitType, stateType, waitType, stateType, inhibitType]
                self.wordPulseValues = [inhibitMask, 0, pulseMicroseconds, 0, originalInhibit]

            return True

        except Exception as e:
            pglMessages.warning(f"Error configuring digital word output: {e}")
            return False

        finally:
            self._wordPulseLock.release()
          
    def digitalOutputPulse(self, channel):
        """
        Send a device-timed pulse and leave the pin LOW.

        Returns:
            float or None: Host submission timestamp, or None on error.

        Note:
            Blocks until the pulse finishes.
        """
        with self._digitalIOLock:
            if self.h is None:
                pglMessages.warning("LabJack is not connected")
                return None

            config = self.digitalChannels.get(channel)
            if config is None:
                pglMessages.warning(f"Digital output channel {channel!r} has not been configured")
                return None

            try:
                timestamp = pglTimestamp.getSecs()
                self.ljm.eWriteAddresses(self.h, 3, config["pulseAddresses"], config["pulseTypes"], config["pulseValues"])
                return timestamp

            except Exception as e:
                # The HIGH write may have succeeded before a later failure.
                try:
                    self.ljm.eWriteAddress(self.h, config["address"], config["dataType"], 0)
                except Exception as cleanupError:
                    pglMessages.warning(f"Could not clear digital channel {channel}: {cleanupError}")

                pglMessages.warning(f"Error pulsing digital channel {channel}: {e}")
                return None
        
    def digitalOutputWord(self, outputWord):
        """
        Send a device-timed word and leave its selected pins LOW.

        Returns:
            float or None: Host submission timestamp, or None on error.

        Note:
            Blocks until the pulse and inhibit restoration finish.
        """
        if not self._wordPulseLock.acquire(blocking=False):
            pglMessages.warning("Digital word output or configuration is currently busy")
            return None

        try:
            if not self.wordDIOBits:
                pglMessages.warning("Digital word output has not been configured")
                return None

            if isinstance(outputWord, bool) or not isinstance(outputWord, int) or not 0 <= outputWord <= self.wordMaxValue:
                pglMessages.warning(f"outputWord must be an integer between 0 and {self.wordMaxValue}")
                return None

            physicalState = 0
            remainingBits = outputWord

            for physicalMask in self.wordPhysicalMasks:
                if remainingBits & 1:
                    physicalState |= physicalMask
                remainingBits >>= 1
                if not remainingBits:
                    break

            # Reuse the request array while protected by the word lock.
            self.wordPulseValues[1] = physicalState

            with self._digitalIOLock:
                if self.h is None:
                    pglMessages.warning("LabJack is not connected")
                    return None

                try:
                    timestamp = pglTimestamp.getSecs()
                    self.ljm.eWriteAddresses(self.h, 5, self.wordPulseAddresses, self.wordPulseTypes, self.wordPulseValues)
                    return timestamp

                except Exception as e:
                    # A failed request may have executed only part of the sequence.
                    try:
                        self.ljm.eWriteAddress(self.h, self.wordPulseAddresses[0], self.wordPulseTypes[0], self.wordInhibitMask)
                        self.ljm.eWriteAddress(self.h, self.wordPulseAddresses[1], self.wordPulseTypes[1], 0)
                    except Exception as cleanupError:
                        pglMessages.warning(f"Could not clear word pins: {cleanupError}")
                    finally:
                        try:
                            self.ljm.eWriteAddress(self.h, self.wordPulseAddresses[0], self.wordPulseTypes[0], self.wordOriginalInhibit)
                        except Exception as restoreError:
                            pglMessages.warning(f"Could not restore DIO_INHIBIT: {restoreError}")

                    pglMessages.warning(f"Error sending digital word {outputWord}: {e}")
                    return None

        finally:
            self._wordPulseLock.release()
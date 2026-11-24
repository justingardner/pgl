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
from .pglEvent import pglEventDigitalInput

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
        
        # Initialize digital acquisition before any hardware operation can fail.
        self._initDigitalInput()
        
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

                # Never turn a monitored scanner input into an output.
                if dioBit in self._digitalChannelBits.values():
                    pglMessages.warning(f"{channelName} is currently reserved for digital input")
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
        """Start analog acquisition only when no digital stream owns the device."""
        with self._streamControlLock:
            thread = self._digitalThread
            if self._digitalStreamActive or (thread is not None and thread.is_alive()):
                raise RuntimeError("Cannot start analog acquisition while digital acquisition is active")
            if self.acquisitionThread is not None and self.acquisitionThread.is_alive():
                raise RuntimeError("Analog acquisition is already active")

            return self._startAnalogRead(duration, channels, scanRate, scansPerRead, voltageRange)
          
    def _startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000, voltageRange=10.0):
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
        """Attempt cleanup; explicit close() remains required for reliable shutdown."""
        try:
            if getattr(self, "h", None) is not None: self.close()
        except Exception:
            # Explicit close() reports failures. Destructors cannot reliably
            # report them, especially during interpreter shutdown.
            pass            
        
    def close(self):
        """Stop acquisition, wait for word output, and close the shared device."""
        with self._streamControlLock:
            self._digitalStopEvent.set()
            self.stopEvent.set()

            # Join without holding locks required by acquisition workers.
            for thread in (self._digitalThread, self.acquisitionThread):
                if thread is threading.current_thread():
                    raise RuntimeError("An acquisition worker cannot close its own LabJack")
                if thread is not None and thread.is_alive(): thread.join(timeout=5.0)
                if thread is not None and thread.is_alive():
                    raise TimeoutError("Acquisition worker did not stop; refusing to close its active handle")

            # Preserve the existing word-output / digital-IO lock order.
            with self._wordPulseLock:
                with self._digitalIOLock:
                    if self.h is not None:
                        self.ljm.close(self.h)
                        self.h = None

                    self._digitalStreamActive = False
                    self._digitalChannelBits = {}
                    self.digitalOutputConfigured = False
                    self.analogInputConfigured = False

            return True
                
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
    
    ################################################################
    # Convert sampled digital states into timestamped edge events
    ################################################################
    @staticmethod
    def _digitalEventsFromSamples(states, channelBits, previousState, firstSampleIndex, scanRate, streamStartTimestamp, edge="rising"):
        """
        Detect digital edges in one batch of decoded state bitmasks.

        Args:
            states: One-dimensional sequence of nonnegative integer bitmasks.
            channelBits (dict): Physical channel names mapped to DIO bit indices.
            previousState (int or None): Last state from the preceding batch.
            firstSampleIndex (int): Zero-based acquisition index of this batch.
            scanRate (float): Actual scan rate returned by the device.
            streamStartTimestamp (float): Estimated host-clock time of sample zero.
            edge (str): rising, falling, or both.

        Returns:
            tuple: (events, lastState), with events ordered by sample index.

        The first sample establishes the baseline when previousState is None.
        An input already active at acquisition start does not create an edge.
        Missing or malformed samples raise rather than inventing transitions.
        """
        # Validate timing and configuration before processing the batch.
        if edge not in {"rising", "falling", "both"}:
            raise ValueError("edge must be rising, falling, or both")
        if isinstance(firstSampleIndex, bool) or not isinstance(firstSampleIndex, (int, np.integer)) or firstSampleIndex < 0:
            raise ValueError("firstSampleIndex must be a nonnegative integer")
        if not np.isfinite(scanRate) or scanRate <= 0:
            raise ValueError("scanRate must be finite and positive")
        if not np.isfinite(streamStartTimestamp):
            raise ValueError("streamStartTimestamp must be finite")
        if not channelBits:
            raise ValueError("At least one digital input channel is required")

        for channel, bit in channelBits.items():
            if not isinstance(channel, str) or not channel:
                raise ValueError("Each channel name must be a nonempty string")
            if isinstance(bit, bool) or not isinstance(bit, (int, np.integer)) or not 0 <= bit < 23:
                raise ValueError("Each T7 DIO bit index must be an integer from 0 through 22")

        if len(set(channelBits.values())) != len(channelBits):
            raise ValueError("Multiple channel names refer to the same physical input")

        # LJM returns numeric samples; accept integral floats but reject missing-data markers.
        stateValues = np.asarray(states, dtype=np.float64)
        if stateValues.ndim != 1:
            raise ValueError("Digital states must be a one-dimensional sequence")
        if not np.all(np.isfinite(stateValues)):
            raise ValueError("Digital stream contains nonfinite samples")
        if np.any((stateValues < 0) | (stateValues > (1 << 23) - 1) | (stateValues != np.floor(stateValues))):
            raise ValueError("Digital stream contains missing or invalid state samples")

        if previousState is not None:
            if isinstance(previousState, bool) or not isinstance(previousState, (int, np.integer)) or not 0 <= previousState < (1 << 23):
                raise ValueError("previousState must be a valid DIO bitmask or None")

        if stateValues.size == 0:
            return [], previousState

        # Carry the previous batch's final sample into this batch's edge comparison.
        currentStates = stateValues.astype(np.uint32)
        previousStates = np.empty_like(currentStates)
        previousStates[0] = currentStates[0] if previousState is None else previousState
        previousStates[1:] = currentStates[:-1]

        # Compare only monitored bits; changes on other pins do not create events.
        monitoredMask = sum(1 << int(bit) for bit in channelBits.values())
        changedBits = (currentStates ^ previousStates) & np.uint32(monitoredMask)
        changedRows = np.flatnonzero(changedBits)

        # Iterate only over samples containing changes, not every acquired sample.
        events = []
        for row in changedRows:
            sampleIndex = int(firstSampleIndex) + int(row)
            timestamp = float(streamStartTimestamp + sampleIndex / scanRate)
            state = int(currentStates[row])
            changed = int(changedBits[row])

            # Simultaneous channel changes share a timestamp and follow channel order.
            for channel, bit in channelBits.items():
                mask = 1 << int(bit)
                if not changed & mask:
                    continue

                eventType = "rising" if state & mask else "falling"
                if edge != "both" and eventType != edge:
                    continue

                events.append(pglEventDigitalInput(channel=channel, eventType=eventType, timestamp=timestamp, sampleIndex=sampleIndex, scanRate=float(scanRate), deviceType="LabJack"))

        return events, int(currentStates[-1])
    
    ################################################################
    # Initialize runtime-only digital acquisition state
    ################################################################
    def _initDigitalInput(self):
        """Initialize stream lifecycle, event buffering, and diagnostics."""
        self._streamControlLock = threading.RLock()
        self._digitalEventLock = threading.Lock()
        self._digitalStopEvent = threading.Event()
        self._digitalThread = None
        self._digitalStreamActive = False
        self._digitalChannelBits = {}
        self._digitalEvents = []
        self._digitalError = None
        self._digitalStats = {}
        self._digitalUnknownReported = 0
        self._digitalMaxEvents = 10000
        
    ################################################################
    # Start digital acquisition on the existing LabJack connection
    ################################################################
    def startDigitalRead(
        self, channels=("FIO0",), edge="rising", scanRate=1000.0,
        scansPerRead=100, streamBufferSizeBytes=32768,
        verifyTriggerCounter=False,
        autoRestartOnStreamOverflow=True,
        maxStreamRestartAttempts=3,
        streamRestartDelay=0.05,
    ):
        """
        Start digital acquisition with explicit verification/recovery policy.

        Counter verification currently supports the tested CIO1/rising
        configuration. It adds two stream addresses and rejects requests
        above 100000 aggregate samples/second.

        Restart attempts are bounded per discontinuity occurrence.
        Pending events must be polled before starting another acquisition.
        """
        from uuid import uuid4
        from copy import deepcopy
        from .pglSettings import pglLabJackSettings

        for name, value in (
            ("verifyTriggerCounter", verifyTriggerCounter),
            ("autoRestartOnStreamOverflow", autoRestartOnStreamOverflow),
        ):
            if type(value) is not bool: raise TypeError(f"{name} must be Boolean.")

        if (
            isinstance(maxStreamRestartAttempts, (bool, np.bool_))
            or not isinstance(maxStreamRestartAttempts, (int, np.integer))
            or maxStreamRestartAttempts < 1
        ):
            raise ValueError("maxStreamRestartAttempts must be a positive integer.")

        if isinstance(streamRestartDelay, (bool, np.bool_)):
            raise ValueError("streamRestartDelay must be finite and nonnegative.")
        streamRestartDelay = float(streamRestartDelay)
        if not np.isfinite(streamRestartDelay) or streamRestartDelay < 0:
            raise ValueError("streamRestartDelay must be finite and nonnegative.")

        channels = [channels] if isinstance(channels, str) else list(channels)
        if not channels: raise ValueError("At least one digital input is required.")
        if edge not in {"rising", "falling", "both"}:
            raise ValueError("edge must be rising, falling, or both.")

        if isinstance(scanRate, (bool, np.bool_)):
            raise ValueError("scanRate must be finite and positive.")
        scanRate = float(scanRate)
        if not np.isfinite(scanRate) or scanRate <= 0:
            raise ValueError("scanRate must be finite and positive.")

        if (
            isinstance(scansPerRead, (bool, np.bool_))
            or not isinstance(scansPerRead, (int, np.integer))
            or scansPerRead <= 0
        ):
            raise ValueError("scansPerRead must be a positive integer.")
        scansPerRead = int(scansPerRead)

        if (
            isinstance(streamBufferSizeBytes, (bool, np.bool_))
            or not isinstance(streamBufferSizeBytes, (int, np.integer))
            or streamBufferSizeBytes not in (4096, 8192, 16384, 32768)
        ):
            raise ValueError("streamBufferSizeBytes must be 4096, 8192, 16384, or 32768.")
        streamBufferSizeBytes = int(streamBufferSizeBytes)

        channelBits = {}
        banks = set()
        for channel in channels:
            name = pglLabJackSettings.normalizeVolumeTriggerChannel(channel)
            if name in channelBits: raise ValueError(f"Duplicate physical input: {name}")
            bank = name[:3]
            channelBits[name] = int(name[3:]) + (16 if bank == "CIO" else 0)
            banks.add(bank)

        if len(banks) != 1:
            raise ValueError("Do not mix FIO and CIO inputs in one acquisition.")

        bank = next(iter(banks))
        stateRegister = f"{bank}_STATE"
        bankShift = 16 if bank == "CIO" else 0
        bankMask = 15 if bank == "CIO" else 255
        scanNames = [stateRegister]

        if verifyTriggerCounter:
            if channelBits != {"CIO1": 17} or edge != "rising":
                raise ValueError(
                    "Counter verification currently supports CIO1/rising only; "
                    "the selected pin/edge will not be changed."
                )
            scanNames += ["DIO17_EF_READ_A", "STREAM_DATA_CAPTURE_16"]

        if scanRate * len(scanNames) > 100000:
            raise ValueError(
                f"{len(scanNames)} addresses × {scanRate:g} scans/s exceeds "
                "100000 aggregate samples/s. The rate was not changed."
            )

        with self._streamControlLock:
            if self.h is None: raise RuntimeError("LabJack is not connected.")
            if self.type != "T7": raise RuntimeError("Digital acquisition supports T7 only.")
            if self._digitalThread is not None and self._digitalThread.is_alive():
                raise RuntimeError("Digital acquisition is already running.")
            if self._digitalStreamActive:
                raise RuntimeError("Previous stream did not stop; close/reopen the device.")
            if getattr(self, "_digitalCounterRestore", None) is not None:
                raise RuntimeError("Previous counter restoration is incomplete.")
            if self.isReading or (
                self.acquisitionThread is not None and self.acquisitionThread.is_alive()
            ):
                raise RuntimeError("Analog acquisition is active.")

            with self._digitalEventLock:
                if self._digitalEvents:
                    raise RuntimeError("Poll pending events before starting acquisition.")

            with self._wordPulseLock:
                with self._digitalIOLock:
                    outputBits = {config["dioBit"] for config in self.digitalChannels.values()}
                    if outputBits.intersection(channelBits.values()):
                        raise RuntimeError("Selected inputs overlap configured output pins.")

                    acquisitionId = str(uuid4())
                    with self._digitalEventLock:
                        # Preserve the previous acquisition's diagnostics.
                        if self._digitalStats:
                            if not hasattr(self, "_digitalAcquisitionHistory"):
                                self._digitalAcquisitionHistory = []
                            self._digitalAcquisitionHistory.append(deepcopy(self._digitalStats))

                        self._digitalError = None
                        self._digitalUnknownReported = 0
                        self._digitalStats = {
                            "acquisitionId": acquisitionId,
                            "requestedScanRate": scanRate,
                            "scanRate": None,
                            "scansPerRead": scansPerRead,
                            "requestedStreamBufferSizeBytes": streamBufferSizeBytes,
                            "originalStreamBufferSizeBytes": None,
                            "streamBufferSizeBytes": None,
                            "channels": list(channelBits),
                            "edge": edge,
                            "stateRegister": stateRegister,
                            "scanNames": list(scanNames),
                            "bankShift": bankShift,
                            "bankMask": bankMask,
                            "verifyTriggerCounter": verifyTriggerCounter,
                            "autoRestartOnStreamOverflow": autoRestartOnStreamOverflow,
                            "maxStreamRestartAttempts": int(maxStreamRestartAttempts),
                            "streamRestartDelay": streamRestartDelay,
                            "samplesReturned": 0,
                            "samplesProcessed": 0,
                            "eventsDetected": 0,
                            "eventsNotQueued": 0,
                            "unknownSamples": 0,
                            "firstUnknownIndices": [],
                            "failureBatch": None,
                            "sampleTimelineValid": True,
                            "continuousTimelineValid": True,
                            "firstUncertainSampleIndex": None,
                            "maxDeviceBacklog": 0,
                            "maxLJMBacklog": 0,
                            "maxReadDuration": 0.0,
                            "startCallBefore": None,
                            "startCallAfter": None,
                            "timestampMethod": "streamStartMidpointEstimate",
                            "recovering": False,
                            "restartAttempts": 0,
                            "successfulRestarts": 0,
                            "recoveryHistory": [],
                            "counterMismatchHistory": [],
                            "timingSegments": [],
                        }

                    self._digitalStopEvent.clear()
                    self._digitalChannelBits = dict(channelBits)

                    try:
                        originalBuffer = self.ljm.eReadName(self.h, "STREAM_BUFFER_SIZE_BYTES")
                        self.ljm.eWriteName(self.h, "STREAM_BUFFER_SIZE_BYTES", streamBufferSizeBytes)
                        acceptedBuffer = self.ljm.eReadName(self.h, "STREAM_BUFFER_SIZE_BYTES")
                        if acceptedBuffer != streamBufferSizeBytes:
                            raise RuntimeError("Stream buffer readback mismatch.")

                        with self._digitalEventLock:
                            self._digitalStats.update({
                                "originalStreamBufferSizeBytes": originalBuffer,
                                "streamBufferSizeBytes": int(acceptedBuffer),
                            })

                        for bit in channelBits.values(): self.ljm.eReadName(self.h, f"DIO{bit}")

                        if verifyTriggerCounter:
                            saved = self._configureDigitalCounter(channelBits, edge)
                            with self._digitalEventLock:
                                self._digitalStats["originalCounterConfiguration"] = deepcopy(saved)

                        self.ljm.eWriteName(self.h, "STREAM_TRIGGER_INDEX", 0)
                        self.ljm.eWriteName(self.h, "STREAM_CLOCK_SOURCE", 0)
                        addresses = self.ljm.namesToAddresses(len(scanNames), scanNames)[0]

                        before = pglTimestamp.getSecs()
                        actualRate = self.ljm.eStreamStart(
                            self.h, scansPerRead, len(addresses), addresses, scanRate
                        )
                        self._digitalStreamActive = True
                        after = pglTimestamp.getSecs()
                        if not np.isfinite(actualRate) or actualRate <= 0:
                            raise RuntimeError(f"Invalid actual scan rate: {actualRate}")

                        with self._digitalEventLock:
                            self._digitalStats.update({
                                "scanRate": float(actualRate),
                                "startCallBefore": before,
                                "startCallAfter": after,
                            })

                        self._digitalThread = threading.Thread(
                            target=self._digitalReadThread,
                            args=(
                                dict(channelBits), edge, float(actualRate),
                                (before + after) / 2, bankShift, bankMask,
                            ),
                            name="pglLabJackDigitalInput",
                            daemon=True,
                        )
                        self._digitalThread.start()

                    except BaseException as error:
                        self._setDigitalReadError(f"Startup failed: {type(error).__name__}: {error}")
                        try:
                            if self._digitalStreamActive:
                                self.ljm.eStreamStop(self.h)
                                self._digitalStreamActive = False
                            self._restoreDigitalCounter()
                        except Exception as cleanupError:
                            self._setDigitalReadError(f"Startup cleanup failed: {cleanupError}")
                        if not self._digitalStreamActive: self._digitalChannelBits = {}
                        raise

        pglMessages.message(
            f"LabJack digital input: {', '.join(channelBits)}, {edge}, "
            f"{actualRate:g} scans/s, {len(scanNames)} addresses, "
            f"counter verification={verifyTriggerCounter}, "
            f"automatic restart={autoRestartOnStreamOverflow}"
        )
        return float(actualRate)

    def _digitalReadThread(
        self, channelBits, edge, scanRate, streamStartTimestamp,
        bankShift=0, bankMask=255,
    ):
        """
        Acquire edges and counter diagnostics using the startup snapshot.

        Returned discontinuity markers trigger bounded stream recovery when
        enabled. Native read exceptions remain fatal in this implementation.

        Never reset/reconfigure the counter during internal stream recovery.
        """
        from copy import deepcopy
        from .pglEvent import pglEvent

        with self._digitalEventLock:
            config = deepcopy(self._digitalStats)

        acquisitionId = config["acquisitionId"]
        verifyCounter = config["verifyTriggerCounter"]
        scanNames = config["scanNames"]
        columnCount = len(scanNames)
        segmentId = 0
        firstSampleIndex = 0
        previousState = None
        currentRate = float(scanRate)
        currentStart = float(streamStartTimestamp)
        batchRecord = None

        matcher = (
            pglLabJackCounterMatcher(acquisitionId, "CIO1", 17)
            if verifyCounter else None
        )

        def addSegment(before, after):
            with self._digitalEventLock:
                self._digitalStats.update({
                    "currentSegmentId": segmentId,
                    "segmentSamplesProcessed": 0,
                    "scanRate": currentRate,
                })
                self._digitalStats["timingSegments"].append({
                    "segmentId": segmentId,
                    "scanRate": currentRate,
                    "startTimestamp": currentStart,
                    "startCallBefore": before,
                    "startCallAfter": after,
                    "timestampMethod": "streamStartMidpointEstimate",
                })

        def queueEvents(events):
            if not events: return
            with self._digitalEventLock:
                available = max(0, self._digitalMaxEvents - len(self._digitalEvents))
                queued = min(available, len(events))
                self._digitalEvents.extend(events[:queued])
                if queued != len(events):
                    self._digitalStats["eventsNotQueued"] += len(events) - queued
                    # Preserve payloads, not only a loss count.
                    self._digitalStats.setdefault("unqueuedEvents", []).extend(events[queued:])
                    raise RuntimeError("Digital event queue exhausted; unqueued payloads retained.")

        def diagnosticEvent(payload, notificationTime):
            event = pglEvent(type=payload["type"])
            for name, value in payload.items(): setattr(event, name, deepcopy(value))
            event.timestamp = notificationTime
            event.timestampMethod = "hostDiagnosticNotification"
            event.deviceType = "LabJack"
            event.deviceUUID = str(getattr(self, "uuid", ""))
            event.serialNumber = getattr(self, "serialNumber", None)
            return event

        def stopStream():
            # Caller holds _digitalIOLock.
            if self._digitalStreamActive:
                self.ljm.eStreamStop(self.h)
                with self._digitalEventLock:
                    self._digitalStreamActive = False

        def restartStream(record):
            """
            Retry only after hardware is known to be stopped.

            A failed start is followed by stop confirmation. If neither stop
            nor STREAM_ENABLE readback establishes that state, fail closed.
            """
            with self._digitalEventLock:
                self._digitalStats["recovering"] = True

            with self._digitalIOLock:
                stopStream()

            lastError = None
            for attempt in range(1, config["maxStreamRestartAttempts"] + 1):
                if self._digitalStopEvent.wait(config["streamRestartDelay"]): return None

                attemptRecord = {"attempt": attempt, "outcome": "starting"}
                with self._digitalEventLock:
                    record["attempts"].append(attemptRecord)
                    self._digitalStats["restartAttempts"] += 1

                try:
                    with self._digitalIOLock:
                        if self._digitalStopEvent.is_set(): return None

                        self.ljm.eWriteName(
                            self.h, "STREAM_BUFFER_SIZE_BYTES",
                            config["streamBufferSizeBytes"],
                        )
                        if self.ljm.eReadName(self.h, "STREAM_BUFFER_SIZE_BYTES") != config["streamBufferSizeBytes"]:
                            raise RuntimeError("Stream buffer readback mismatch during recovery.")

                        if verifyCounter:
                            # Verification only: do not disable/enable/reset EF.
                            if (
                                self.ljm.eReadName(self.h, "DIO17_EF_ENABLE") != 1
                                or self.ljm.eReadName(self.h, "DIO17_EF_INDEX") != 7
                            ):
                                raise RuntimeError("Counter configuration changed during acquisition.")

                        addresses = self.ljm.namesToAddresses(columnCount, scanNames)[0]
                        before = pglTimestamp.getSecs()
                        rate = self.ljm.eStreamStart(
                            self.h, config["scansPerRead"], columnCount,
                            addresses, config["requestedScanRate"],
                        )
                        with self._digitalEventLock:
                            self._digitalStreamActive = True
                        after = pglTimestamp.getSecs()

                        if not np.isfinite(rate) or rate <= 0:
                            raise RuntimeError(f"Invalid restart scan rate: {rate}")

                        if self._digitalStopEvent.is_set():
                            stopStream()
                            return None

                    with self._digitalEventLock:
                        attemptRecord["outcome"] = "restarted"
                    return float(rate), before, after

                except Exception as error:
                    lastError = error
                    with self._digitalEventLock:
                        attemptRecord.update({
                            "outcome": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        })

                    with self._digitalIOLock:
                        try:
                            # A failed start may have partially succeeded.
                            self.ljm.eStreamStop(self.h)
                        except Exception:
                            if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                                raise RuntimeError(
                                    "Failed restart left uncertain stream ownership."
                                ) from error
                        with self._digitalEventLock:
                            self._digitalStreamActive = False

                    # A changed counter configuration is not a retryable overflow.
                    if verifyCounter:
                        with self._digitalIOLock:
                            if (
                                self.ljm.eReadName(self.h, "DIO17_EF_ENABLE") != 1
                                or self.ljm.eReadName(self.h, "DIO17_EF_INDEX") != 7
                            ):
                                raise RuntimeError(
                                    "Counter configuration changed; recovery stopped."
                                ) from error

            raise RuntimeError(
                f"Restart failed after {config['maxStreamRestartAttempts']} attempts."
            ) from lastError

        addSegment(config["startCallBefore"], config["startCallAfter"])

        try:
            while not self._digitalStopEvent.is_set():
                batchRecord = None
                with self._digitalIOLock:
                    if self._digitalStopEvent.is_set(): break
                    beforeRead = time.perf_counter()
                    values, deviceBacklog, ljmBacklog = self.ljm.eStreamRead(self.h)
                    readDuration = time.perf_counter() - beforeRead
                    deliveredAt = pglTimestamp.getSecs()

                raw = np.asarray(values, dtype=np.float64)
                batchRecord = {
                    "segmentId": segmentId,
                    "firstReturnedSampleIndex": firstSampleIndex,
                    "readCompletedTimestamp": deliveredAt,
                    "shape": list(raw.shape),
                    "values": [
                        float(value) if np.isfinite(value) else repr(float(value))
                        for value in raw.reshape(-1)
                    ],
                }
                if raw.ndim != 1 or not raw.size or raw.size % columnCount:
                    raise RuntimeError("Malformed interleaved digital stream batch.")

                returnedScans = raw.size // columnCount
                with self._digitalEventLock:
                    stats = self._digitalStats
                    stats["samplesReturned"] += int(returnedScans)
                    stats["maxReadDuration"] = max(stats["maxReadDuration"], readDuration)
                    stats["maxDeviceBacklog"] = max(stats["maxDeviceBacklog"], int(deviceBacklog))
                    stats["maxLJMBacklog"] = max(stats["maxLJMBacklog"], int(ljmBacklog))

                if verifyCounter:
                    rows = self._decodeCounterStream(raw, bankMask)["rows"]
                else:
                    rows = []
                    for offset, value in enumerate(raw):
                        valid = (
                            np.isfinite(value) and value == np.floor(value)
                            and 0 <= value <= bankMask
                        )
                        classification = (
                            "valid" if valid else
                            "ljmDummyScan" if value == -9999 else
                            "invalidStateFFFF" if value == 65535 else
                            "malformedScan"
                        )
                        rows.append({
                            "scanOffset": offset,
                            "classification": classification,
                            "state": int(value) if valid else None,
                        })

                invalidOffset = next(
                    (index for index, row in enumerate(rows) if row["classification"] != "valid"),
                    len(rows),
                )
                prefix = rows[:invalidOffset]
                events = []

                if prefix:
                    states = np.asarray(
                        [row["state"] << bankShift for row in prefix],
                        dtype=np.uint32,
                    )
                    edges, previousState = self._digitalEventsFromSamples(
                        states, channelBits, previousState, firstSampleIndex,
                        currentRate, currentStart, edge=edge,
                    )
                    for event in edges:
                        event.acquisitionId = acquisitionId
                        event.segmentId = segmentId
                        event.sampleIndexScope = "timingSegment"
                        event.timestampMethod = "streamStartMidpointEstimate"
                        event.readCompletedTimestamp = deliveredAt
                        event.isEstimated = False
                        event.deviceUUID = str(getattr(self, "uuid", ""))
                    events.extend(edges)

                    if matcher is not None:
                        for offset, row in enumerate(prefix):
                            index = firstSampleIndex + offset
                            reports = matcher.observe(
                                state=row["state"] << bankShift,
                                counter=row["counter"],
                                segmentId=segmentId,
                                sampleIndex=index,
                                timestamp=currentStart + index / currentRate,
                            )
                            for report in reports:
                                with self._digitalEventLock:
                                    self._digitalStats["counterMismatchHistory"].append(deepcopy(report))
                                events.append(diagnosticEvent(report, deliveredAt))

                    with self._digitalEventLock:
                        self._digitalStats["eventsDetected"] += len(edges)

                queueEvents(events)
                firstSampleIndex += len(prefix)
                with self._digitalEventLock:
                    self._digitalStats["samplesProcessed"] += len(prefix)
                    self._digitalStats["segmentSamplesProcessed"] = firstSampleIndex

                if invalidOffset == len(rows):
                    batchRecord = None
                    continue

                classification = rows[invalidOffset]["classification"]
                with self._digitalEventLock:
                    stats = self._digitalStats
                    stats["failureBatch"] = deepcopy(batchRecord)
                    stats["sampleTimelineValid"] = False
                    stats["continuousTimelineValid"] = False
                    stats["unknownSamples"] += sum(
                        row["classification"] != "valid" for row in rows[invalidOffset:]
                    )
                    if stats["firstUncertainSampleIndex"] is None:
                        stats["firstUncertainSampleIndex"] = firstSampleIndex
                        stats["firstUncertainSegmentId"] = segmentId

                if classification not in ("ljmDummyScan", "invalidStateFFFF"):
                    raise RuntimeError(f"Invalid digital scan: {classification}")

                record = {
                    "reason": classification,
                    "oldSegmentId": segmentId,
                    "firstUncertainSampleIndex": firstSampleIndex,
                    "detectedTimestamp": deliveredAt,
                    "missingScans": None,
                    "discardedReturnedScans": len(rows) - invalidOffset,
                    "failureBatch": deepcopy(batchRecord),
                    "attempts": [],
                    "outcome": "detected",
                }
                with self._digitalEventLock:
                    self._digitalStats["recoveryHistory"].append(record)

                if not config["autoRestartOnStreamOverflow"]:
                    raise RuntimeError("Stream discontinuity detected; automatic restart is disabled.")

                try:
                    restarted = restartStream(record)
                except Exception as error:
                    with self._digitalEventLock:
                        record["outcome"] = "failed"
                        record["error"] = f"{type(error).__name__}: {error}"
                    pglMessages.warning(
                        f"LabJack stream discontinuity detected. Automatic restart failed: "
                        f"{error}. Diagnostics have been retained.",
                        level=2,
                    )
                    raise

                if restarted is None:
                    with self._digitalEventLock:
                        record["outcome"] = "cancelledByStop"
                    break

                currentRate, before, after = restarted
                currentStart = (before + after) / 2
                segmentId += 1
                firstSampleIndex = 0
                previousState = None

                if matcher is not None:
                    matcher.noteGap({
                        "reason": "streamRestart",
                        "oldSegmentId": segmentId - 1,
                        "newSegmentId": segmentId,
                        "missingScans": None,
                        "detectedTimestamp": deliveredAt,
                        "restartTimestamp": after,
                    }, counterContinuityVerified=True)

                with self._digitalEventLock:
                    record.update({
                        "outcome": "restarted",
                        "newSegmentId": segmentId,
                        "restartTimestamp": after,
                    })
                    self._digitalStats["successfulRestarts"] += 1
                    self._digitalStats["recovering"] = False
                addSegment(before, after)

                queueEvents([diagnosticEvent({
                    "type": "digitalInputRecovery",
                    "acquisitionId": acquisitionId,
                    "channels": list(channelBits),
                    "oldSegmentId": segmentId - 1,
                    "segmentId": segmentId,
                    "reason": classification,
                    "missingScans": None,
                    "missingTriggers": None,
                    "counterVerificationEnabled": verifyCounter,
                }, after)])

                pglMessages.warning(
                    f"LabJack stream discontinuity detected. Stream successfully "
                    f"restarted (timing segment {segmentId}); acquisition is continuing. "
                    "The interruption and failing batch have been retained.",
                    level=2,
                )
                batchRecord = None

        except Exception as error:
            with self._digitalEventLock:
                self._digitalStats["sampleTimelineValid"] = False
                if batchRecord is not None:
                    self._digitalStats["failureBatch"] = deepcopy(batchRecord)
            self._setDigitalReadError(f"{type(error).__name__}: {error}")

        finally:
            # Preserve an unsettled final window rather than falsely reporting
            # it as a settled missing-trigger count.
            if matcher is not None:
                with self._digitalEventLock:
                    self._digitalStats["counterFinalPendingWindow"] = {
                        "anchorObservation": deepcopy(matcher.anchorObservation),
                        "lastObservation": deepcopy(matcher.previousObservation),
                        "measuredObservations": deepcopy(matcher.measuredObservations),
                        "gapRecords": deepcopy(matcher.gapRecords),
                    }

            try:
                with self._digitalIOLock:
                    stopStream()
                    self._restoreDigitalCounter()
                    self._digitalChannelBits = {}
            except Exception as error:
                self._setDigitalReadError(
                    f"Digital cleanup failed: {type(error).__name__}: {error}"
                )
            finally:
                with self._digitalEventLock:
                    self._digitalStats["recovering"] = False

    ################################################################
    # Retain worker failures without retaining traceback objects
    ################################################################
    def _setDigitalReadError(self, message):
        with self._digitalEventLock:
            if self._digitalError is None: self._digitalError = str(message)
            else: self._digitalError += f"; {message}"

    ################################################################
    # Poll completed input events without touching hardware
    ################################################################
    def poll(self):
        """
        Drain queued digital events before reporting a latched failure.

        Events captured before an acquisition failure remain available.
        A subsequent empty poll raises the retained acquisition error.
        stopDigitalRead() also reports that error.
        """
        with self._digitalEventLock:
            events = self._digitalEvents
            self._digitalEvents = []
            error = self._digitalError

        if events: return events
        if error is not None: raise RuntimeError(f"LabJack digital acquisition failed: {error}")
        return []

    ################################################################
    # Snapshot acquisition diagnostics
    ################################################################
    def getDigitalReadStatus(self):
        """Return independent diagnostic values, including any failing batch."""
        from copy import deepcopy

        with self._digitalEventLock:
            status = deepcopy(self._digitalStats)
            status["pendingEvents"] = len(self._digitalEvents)
            status["error"] = self._digitalError

        thread = self._digitalThread
        status["running"] = thread is not None and thread.is_alive()
        status["streamActive"] = self._digitalStreamActive
        return status

    ################################################################
    # Stop digital acquisition while preserving queued events
    ################################################################
    def stopDigitalRead(self, timeout=5.0):
        """
        Stop the worker without closing the shared connection.

        Pending events remain available through poll().
        A timeout leaves the handle open because the worker may still use it.
        """
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")

        with self._streamControlLock:
            self._digitalStopEvent.set()
            thread = self._digitalThread

            if thread is threading.current_thread():
                raise RuntimeError("The acquisition worker cannot join itself")

            if thread is not None and thread.is_alive(): thread.join(timeout=timeout)
            if thread is not None and thread.is_alive():
                raise TimeoutError("Digital worker did not stop; the LabJack handle remains open")

            status = self.getDigitalReadStatus()
            if status["error"] is not None:
                raise RuntimeError(f"LabJack digital acquisition failed: {status['error']}")
            if self._digitalStreamActive:
                raise RuntimeError("Digital hardware stream did not stop successfully")

            return status
        
    @staticmethod
    def _decodeCounterStream(values, bankMask=15):
        """
        Decode interleaved state, counter-low, counter-high stream words.

        The scan list must be:
            bank state register
            selected DIO_EF_READ_A
            STREAM_DATA_CAPTURE_16

        Returns:
            Dictionary containing every returned scan in original order.

        Full -9999 rows are identified as LJM dummy scans. This classification
        alone does not authorize timeline recovery: the acquisition worker
        must establish the applicable LJM missing-scan accounting contract.

        Invalid rows retain their positions and raw words. No edge count,
        missing-scan count, or missing-trigger timestamp is inferred here.
        """
        from numbers import Real

        if (
            isinstance(bankMask, (bool, np.bool_))
            or not isinstance(bankMask, (int, np.integer))
            or bankMask not in (15, 255)
        ):
            raise ValueError("bankMask must be 15 for CIO or 255 for FIO.")

        # Reject malformed structure rather than truncating incomplete scans.
        raw = np.asarray(values, dtype=object)
        if raw.ndim != 1 or raw.size == 0 or raw.size % 3:
            raise ValueError(
                "Counter stream must be a nonempty flat sequence "
                "containing complete three-word scans."
            )

        def integerWord(value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                return None
            try:
                number = float(value)
                if not np.isfinite(number) or number != np.floor(number):
                    return None
                return int(number)
            except (ValueError, TypeError, OverflowError):
                return None

        rows = []
        classificationCounts = {}

        for scanOffset, rawRow in enumerate(raw.reshape(-1, 3)):
            words = [integerWord(value) for value in rawRow]
            state, lowWord, highWord = words

            # Diagnostic representation preserves numeric values and makes
            # malformed/nonfinite values printable and JSON-safe.
            rawWords = [
                word if word is not None else repr(value)
                for value, word in zip(rawRow, words)
            ]

            row = {
                "scanOffset": scanOffset,
                "rawWords": rawWords,
                "classification": None,
                "state": None,
                "counter": None,
            }

            if words == [-9999, -9999, -9999]:
                classification = "ljmDummyScan"
            elif -9999 in words:
                classification = "partialDummyScan"
            elif any(word is None for word in words):
                classification = "malformedScan"
            elif state == 65535:
                classification = "invalidStateFFFF"
            elif not (
                0 <= state <= bankMask
                and 0 <= lowWord <= 65535
                and 0 <= highWord <= 65535
            ):
                classification = "outOfRangeScan"
            else:
                classification = "valid"
                row["state"] = state
                row["counter"] = lowWord + (highWord << 16)

            row["classification"] = classification
            rows.append(row)
            classificationCounts[classification] = (
                classificationCounts.get(classification, 0) + 1
            )

        return {
            "returnedScanCount": len(rows),
            "rows": rows,
            "classificationCounts": classificationCounts,
        }
        
    @staticmethod
    def _compareTriggerCounterWindow(
        counterBefore, counterAfter, measuredEdgeCount,
        *, windowSettled=False, counterContinuityVerified=False,
        verifiedWrapCount=None,
    ):
        """
        Compare hardware increments with measured rising edges in one window.

        Requirements imposed on the caller:
            - Counter and sampled edges refer to the same physical source
              and rising-edge polarity.
            - Both observations delimit the same comparison window.
            - windowSettled=True only after state/counter observation ordering
              has been resolved, including across read-batch boundaries.
            - Counter continuity must be established independently.
            - verifiedWrapCount is supplied only when wrap accounting is
              independently established. A decreasing count alone does not
              distinguish rollover from reset.

        Returns:
            A plain diagnostic dictionary.

        No missing-trigger timestamps or scan-gap lengths are inferred.
        A negative discrepancy never deletes measured events.
        """
        from numbers import Integral

        def validInteger(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Integral)
            )

        for fieldName, value in (
            ("counterBefore", counterBefore),
            ("counterAfter", counterAfter),
        ):
            if not validInteger(value) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError(f"{fieldName} must be an unsigned 32-bit integer.")

        if not validInteger(measuredEdgeCount) or measuredEdgeCount < 0:
            raise ValueError("measuredEdgeCount must be a nonnegative integer.")

        if type(windowSettled) is not bool:
            raise TypeError("windowSettled must be Boolean.")
        if type(counterContinuityVerified) is not bool:
            raise TypeError("counterContinuityVerified must be Boolean.")

        if verifiedWrapCount is not None:
            if not validInteger(verifiedWrapCount) or verifiedWrapCount < 0:
                raise ValueError("verifiedWrapCount must be a nonnegative integer or None.")

        counterBefore = int(counterBefore)
        counterAfter = int(counterAfter)
        measuredEdgeCount = int(measuredEdgeCount)

        result = {
            "outcome": None,
            "reason": None,
            "counterBefore": counterBefore,
            "counterAfter": counterAfter,
            "measuredEdgeCount": measuredEdgeCount,
            "windowSettled": windowSettled,
            "counterContinuityVerified": counterContinuityVerified,
            "verifiedWrapCount": (
                None if verifiedWrapCount is None else int(verifiedWrapCount)
            ),
            "counterIncrement": None,
            "edgeDeficit": None,
            "missingEdgeCount": None,
        }

        def finish(outcome, reason):
            result["outcome"] = outcome
            result["reason"] = reason
            return result

        # Keep boundary discrepancies pending; do not emit settled mismatches.
        if not windowSettled:
            return finish("pending", "observationWindowNotSettled")

        if not counterContinuityVerified:
            return finish("unresolved", "counterContinuityNotVerified")

        if verifiedWrapCount is None:
            if counterAfter < counterBefore:
                return finish("unresolved", "counterRolloverOrReset")

            # The caller's continuity contract must also exclude unobserved
            # complete wraps. Increasing values alone do not prove continuity.
            counterIncrement = counterAfter - counterBefore
        else:
            counterIncrement = (
                counterAfter - counterBefore
                + int(verifiedWrapCount) * (1 << 32)
            )
            if counterIncrement < 0:
                return finish("unresolved", "inconsistentWrapAccounting")

        edgeDeficit = counterIncrement - measuredEdgeCount
        result["counterIncrement"] = counterIncrement
        result["edgeDeficit"] = edgeDeficit

        if edgeDeficit > 0:
            result["missingEdgeCount"] = edgeDeficit
            return finish("missingEdges", "counterEdgeMismatch")

        if edgeDeficit < 0:
            # Preserve observed edges; do not turn a surplus into an insertion.
            return finish("unresolved", "sampledEdgesExceedCounter")

        result["missingEdgeCount"] = 0
        return finish("matched", "counterAndSampledEdgesAgree")
    
    def _configureDigitalCounter(self, channelBits, edge):
        """
        Configure the tested CIO1 rising-edge high-speed counter.

        Call while holding _digitalIOLock, before starting the stream.
        Return saved configuration for later restoration.

        Refuse enabled EF consumers rather than disable their shared clocks.
        """
        if channelBits != {"CIO1": 17} or edge != "rising":
            raise ValueError(
                "Counter verification currently supports the tested "
                "CIO1/rising configuration only. No pin or edge was substituted."
            )

        saved = {
            "counterEnable": self.ljm.eReadName(self.h, "DIO17_EF_ENABLE"),
            "counterIndex": self.ljm.eReadName(self.h, "DIO17_EF_INDEX"),
            "clock0Enable": self.ljm.eReadName(self.h, "DIO_EF_CLOCK0_ENABLE"),
            "clock2Enable": self.ljm.eReadName(self.h, "DIO_EF_CLOCK2_ENABLE"),
        }

        if saved["counterEnable"] != 0:
            raise RuntimeError("CIO1 EF is already enabled; refusing to replace it.")

        enabledFeatures = []
        for bit in range(23):
            try:
                enabled = self.ljm.eReadName(self.h, f"DIO{bit}_EF_ENABLE")
            except Exception as error:
                # T7 EIO pins have no timer/counter features.
                if getattr(error, "errorCode", None) != 2550: raise
                continue
            if enabled != 0: enabledFeatures.append(f"DIO{bit}")

        if enabledFeatures:
            raise RuntimeError(
                "Counter configuration would share hardware with enabled EF "
                f"features: {', '.join(enabledFeatures)}. Configuration unchanged."
            )

        # Retain restoration information before writes can partially succeed.
        self._digitalCounterRestore = saved

        self.ljm.eWriteName(self.h, "DIO_EF_CLOCK0_ENABLE", 0)
        self.ljm.eWriteName(self.h, "DIO_EF_CLOCK2_ENABLE", 0)
        self.ljm.eWriteName(self.h, "DIO17_EF_INDEX", 7)
        self.ljm.eWriteName(self.h, "DIO17_EF_ENABLE", 1)

        checks = {
            "DIO_EF_CLOCK0_ENABLE": 0,
            "DIO_EF_CLOCK2_ENABLE": 0,
            "DIO17_EF_INDEX": 7,
            "DIO17_EF_ENABLE": 1,
        }
        for register, expected in checks.items():
            actual = self.ljm.eReadName(self.h, register)
            if actual != expected:
                raise RuntimeError(f"{register}: expected {expected}, read {actual}")

        return saved
    
    def _restoreDigitalCounter(self):
        """
        Restore counter/clock configuration after the stream has stopped.

        Call while holding _digitalIOLock. Retain restoration metadata on error.
        """
        saved = getattr(self, "_digitalCounterRestore", None)
        if saved is None: return
        if self._digitalStreamActive:
            raise RuntimeError("Cannot restore EF configuration while streaming.")

        # Disable and verify first. Never re-enable shared clocks over a
        # counter whose shutdown is uncertain.
        self.ljm.eWriteName(self.h, "DIO17_EF_ENABLE", 0)
        if self.ljm.eReadName(self.h, "DIO17_EF_ENABLE") != 0:
            raise RuntimeError("Could not disable CIO1 counter for restoration.")

        errors = []
        for register, value in (
            ("DIO17_EF_INDEX", saved["counterIndex"]),
            ("DIO_EF_CLOCK0_ENABLE", saved["clock0Enable"]),
            ("DIO_EF_CLOCK2_ENABLE", saved["clock2Enable"]),
        ):
            try:
                self.ljm.eWriteName(self.h, register, value)
                if self.ljm.eReadName(self.h, register) != value:
                    raise RuntimeError("Restoration readback mismatch")
            except Exception as error:
                errors.append(f"{register}: {type(error).__name__}: {error}")

        if errors: raise RuntimeError("; ".join(errors))
        self._digitalCounterRestore = None
    
class pglLabJackCounterMatcher:
    """
    Stateful comparison of sampled rising edges with a hardware counter.

    Supports one rising-edge source. Physical counter configuration and
    acquisition ownership remain the driver's responsibility.

    Settled boundaries require consecutive LOW samples with equal counts.
    The initial boundary establishes a baseline; earlier edges are not
    retrospectively assessed.

    Counter decreases are unresolved rollover/reset observations. This
    matcher never assumes that a decrease is a rollover.

    Returned diagnostics are dictionaries for the worker to wrap as events.
    They contain observation bounds, not estimated missing-edge timestamps.
    """

    def __init__(self, acquisitionId, channel, channelBit):
        if not isinstance(acquisitionId, str) or not acquisitionId:
            raise ValueError("acquisitionId must be a nonempty string.")
        if not isinstance(channel, str) or not channel:
            raise ValueError("channel must be a nonempty string.")
        if (
            isinstance(channelBit, (bool, np.bool_))
            or not isinstance(channelBit, (int, np.integer))
            or not 0 <= channelBit < 23
        ):
            raise ValueError("channelBit must be an integer from 0 through 22.")

        self.acquisitionId = acquisitionId
        self.channel = channel
        self.channelBit = int(channelBit)
        self.previousObservation = None
        self.anchorObservation = None
        self.measuredObservations = []
        self.gapRecords = []
        self.counterContinuityVerified = True
        self.reportNumber = 0

    def noteGap(self, gapInfo, *, counterContinuityVerified):
        """
        Break sampled-state continuity without inventing an edge.

        The worker must call this for every discarded/missing scan region
        and every stream restart. Preserve a counter baseline only when
        counter continuity across that interruption is established.
        """
        from copy import deepcopy

        if not isinstance(gapInfo, dict):
            raise TypeError("gapInfo must be a dictionary.")
        if type(counterContinuityVerified) is not bool:
            raise TypeError("counterContinuityVerified must be Boolean.")

        self.gapRecords.append(deepcopy(gapInfo))
        self.previousObservation = None
        self.counterContinuityVerified = (
            self.counterContinuityVerified and counterContinuityVerified
        )

    def observe(self, *, state, counter, segmentId, sampleIndex, timestamp):
        """
        Consume one valid sample and report settled counter-window evidence.

        state is a full DIO bitmask, not an unshifted CIO_STATE word.

        Matched windows identify measured edges that agree with the counter.
        They do not, by themselves, certify an interval crossing another
        window or a gap. The experiment must check interval continuity.

        Diagnostic timestamps are assigned by the worker at delivery.
        Observation timestamps below describe sampled acquisition positions.
        """
        from copy import deepcopy
        from numbers import Integral, Real

        def validInteger(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Integral)
            )

        if not validInteger(state) or not 0 <= state < (1 << 23):
            raise ValueError("state must be a valid full DIO bitmask.")
        if not validInteger(counter) or not 0 <= counter <= 0xFFFFFFFF:
            raise ValueError("counter must be an unsigned 32-bit integer.")
        if not validInteger(segmentId) or segmentId < 0:
            raise ValueError("segmentId must be a nonnegative integer.")
        if not validInteger(sampleIndex) or sampleIndex < 0:
            raise ValueError("sampleIndex must be a nonnegative integer.")
        if (
            isinstance(timestamp, (bool, np.bool_))
            or not isinstance(timestamp, Real)
            or not np.isfinite(timestamp)
        ):
            raise ValueError("timestamp must be finite.")

        observation = {
            "segmentId": int(segmentId),
            "sampleIndex": int(sampleIndex),
            "timestamp": float(timestamp),
            "counter": int(counter),
            "inputHigh": bool(int(state) & (1 << self.channelBit)),
        }
        previous = self.previousObservation

        if previous is not None:
            if (
                observation["segmentId"] != previous["segmentId"]
                or observation["sampleIndex"] != previous["sampleIndex"] + 1
            ):
                raise ValueError("Call noteGap() before a discontinuous sample.")
            if observation["timestamp"] <= previous["timestamp"]:
                raise ValueError("Within-segment timestamps must increase.")

            if observation["counter"] < previous["counter"]:
                self.counterContinuityVerified = False
                self.gapRecords.append({
                    "reason": "counterRolloverOrReset",
                    "before": deepcopy(previous),
                    "after": deepcopy(observation),
                })

            # Same first-sample baseline behavior as the edge detector.
            if not previous["inputHigh"] and observation["inputHigh"]:
                if self.anchorObservation is not None:
                    self.measuredObservations.append(deepcopy(observation))

        settled = (
            previous is not None
            and not previous["inputHigh"]
            and not observation["inputHigh"]
            and previous["counter"] == observation["counter"]
        )
        self.previousObservation = observation
        if not settled: return []

        anchor = self.anchorObservation
        if anchor is None:
            self._setAnchor(observation)
            return []

        comparison = pglLabJack._compareTriggerCounterWindow(
            anchor["counter"],
            observation["counter"],
            len(self.measuredObservations),
            windowSettled=True,
            counterContinuityVerified=self.counterContinuityVerified,
        )

        matched = comparison["outcome"] == "matched"
        hasEvidence = (
            not matched
            or bool(self.measuredObservations)
            or bool(self.gapRecords)
        )

        diagnostics = []
        if hasEvidence:
            self.reportNumber += 1
            windowId = f"{self.acquisitionId}:{self.reportNumber}"
            report = {
                "type": (
                    "digitalInputCounterWindow"
                    if matched else "digitalInputMismatch"
                ),
                "acquisitionId": self.acquisitionId,
                "channel": self.channel,
                "edge": "rising",
                "windowId": windowId,
                "comparison": comparison,
                "missingEdgeCount": comparison["missingEdgeCount"],
                "beforeObservation": deepcopy(anchor),
                "afterObservation": deepcopy(observation),
                "measuredObservations": deepcopy(self.measuredObservations),
                "gapRecords": deepcopy(self.gapRecords),
                "continuousScanTimeline": (
                    not self.gapRecords
                    and anchor["segmentId"] == observation["segmentId"]
                ),
            }
            if not matched:
                report["mismatchId"] = windowId
            diagnostics.append(report)

        # Advance even after a quiet matched window: later reports have
        # narrow bounds, and a deficit cannot be counted a second time.
        self._setAnchor(observation)
        return diagnostics

    def _setAnchor(self, observation):
        """Start a new comparison window without erasing returned diagnostics."""
        from copy import deepcopy

        self.anchorObservation = deepcopy(observation)
        self.measuredObservations = []
        self.gapRecords = []
        self.counterContinuityVerified = True
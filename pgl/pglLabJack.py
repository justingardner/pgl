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
    def __init__(self, clockCalibrationDuration=0.05):
        from uuid import uuid4

        if isinstance(clockCalibrationDuration, (bool, np.bool_)):
            raise ValueError("clockCalibrationDuration must be positive and finite.")
        clockCalibrationDuration = float(clockCalibrationDuration)
        if not np.isfinite(clockCalibrationDuration) or clockCalibrationDuration <= 0:
            raise ValueError("clockCalibrationDuration must be positive and finite.")

        self.clockCalibrationDuration = clockCalibrationDuration
        self._clock = None
        self._clockStreamStarts = []
        self._digitalCurrentClockSegment = None
        self._clockStopEvent = threading.Event()
        self.connectionId = str(uuid4())
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
            try:
                if self.type != "T7":
                    raise RuntimeError("Clock-calibrated LabJack acquisition currently supports T7 only.")

                self._clock = pglLabJackClock(
                    self.connectionId,
                    {
                        "type": self.type,
                        "serialNumber": self.serialNumber,
                        "connectionType": self.connectionType,
                    },
                )
                with self._digitalIOLock:
                    if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                        raise RuntimeError("Device already has an active stream.")
                    self._clock.collectBeginning(
                        lambda: self.ljm.eReadName(self.h, "CORE_TIMER"),
                        pglTimestamp.getSecs,
                        self.clockCalibrationDuration,
                        self._clockStopEvent.is_set,
                    )
            except BaseException as error:
                # Retain the object/evidence on the exception if construction fails.
                error.labJackDevice = self
                error.labJackClockEvents = self.getClockEvents()
                try:
                    self.close()
                except Exception as cleanupError:
                    error.labJackCleanupFailure = (
                        f"{type(cleanupError).__name__}: {cleanupError}"
                    )
                raise

               
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
          
    def startAnalogRead(
        self, duration=2, channels=(0,), scanRate=1000,
        scansPerRead=1000, voltageRange=10.0,
    ):
        """
        Start analog acquisition and return its actual scan rate when ready.

        Readiness means stream start and hardware-origin validation succeeded.
        No fixed sleep is used as a substitute for readiness.
        One uninterrupted stream; gaps/errors stop acquisition and retain data.
        """
        with self._streamControlLock:
            if self._digitalStreamActive or (
                self._digitalThread is not None and self._digitalThread.is_alive()
            ):
                raise RuntimeError("Digital acquisition owns the LabJack stream.")
            if getattr(self, "_analogStreamActive", False) or self.isReading:
                raise RuntimeError("Analog stream ownership is still active.")
            if self.acquisitionThread is not None and self.acquisitionThread.is_alive():
                raise RuntimeError("Analog acquisition worker is already active.")
            return self._startAnalogRead(
                duration, channels, scanRate, scansPerRead, voltageRange
            )
          
    def _startAnalogRead(
        self, duration=2, channels=(0,), scanRate=1000,
        scansPerRead=1000, voltageRange=10.0,
    ):
        """Configure and synchronously establish an analog stream's timing."""
        from copy import deepcopy
        from uuid import uuid4
        from numbers import Integral, Real

        if self.h is None:
            raise RuntimeError("LabJack is not connected.")
        if self.type != "T7":
            raise RuntimeError("Clock-aligned analog acquisition currently supports T7.")
        if self._clock is None or not self._clock.valid:
            raise RuntimeError("No valid connection clock mapping.")
        if self._clockStopEvent.is_set():
            raise RuntimeError("Device closure has been requested.")

        valuesToValidate = [("scanRate", scanRate)]
        if duration is not None: valuesToValidate.append(("duration", duration))
        for name, value in valuesToValidate:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite.")
        if (
            isinstance(scansPerRead, (bool, np.bool_))
            or not isinstance(scansPerRead, Integral) or scansPerRead < 1
        ):
            raise ValueError("scansPerRead must be a positive integer.")
        if (
            isinstance(voltageRange, (bool, np.bool_))
            or voltageRange not in (10.0, 1.0, 0.1, 0.01)
        ):
            raise ValueError("Unsupported analog voltage range.")

        channels = [channels] if isinstance(channels, str) else list(channels)
        if not channels:
            raise ValueError("At least one analog channel is required.")
        channelNames = []
        for channel in channels:
            if isinstance(channel, (bool, np.bool_)):
                raise ValueError("Boolean analog channel is invalid.")
            if isinstance(channel, Integral):
                if channel < 0:
                    raise ValueError("Analog channel number must be nonnegative.")
                channel = f"AIN{int(channel)}"
            if not isinstance(channel, str) or not channel:
                raise ValueError("Each analog channel must be an index or register name.")
            channelNames.append(channel)
        if len(set(channelNames)) != len(channelNames):
            raise ValueError("Duplicate analog channels.")
        if float(scanRate) * len(channelNames) > 100000:
            raise ValueError(
                "Analog request exceeds 100000 aggregate samples/s. "
                "The scan rate was not changed."
            )

        self.channels = channelNames
        self.numChannels = len(channelNames)
        self.scansPerRead = int(scansPerRead)
        self.range = float(voltageRange)
        self.analogStreamDuration = None if duration is None else float(duration)
        self.stopEvent.clear()
        self.analogBuffer = []
        self._analogStreamActive = False
        self._analogClockSegment = None

        # Retain earlier acquisition diagnostics before resetting runtime state.
        if getattr(self, "_analogStats", None):
            if not hasattr(self, "_analogAcquisitionHistory"):
                self._analogAcquisitionHistory = []
            self._analogAcquisitionHistory.append(deepcopy(self._analogStats))

        self._analogStats = {
            "acquisitionId": str(uuid4()),
            "connectionId": self.connectionId,
            "channels": list(channelNames),
            "scanOrder": list(channelNames),
            "requestedScanRate": float(scanRate),
            "scanRate": None,
            "scansPerRead": int(scansPerRead),
            "voltageRange": float(voltageRange),
            "requestedDurationSeconds": None if duration is None else float(duration),
            "samplesReturned": 0,
            "validSampleCount": 0,
            "firstInvalidSampleIndex": None,
            "timelineValid": True,
            "ready": False,
            "error": None,
            "cleanupErrors": [],
            "batchDiagnostics": [],
            "startRecord": {},
            "clockEvents": self.getClockEvents(),
            "channelApertureOffsetsSeconds": None,
            "channelApertureUncertainty": "notEstablished",
        }

        stage = "configuration"
        try:
            with self._wordPulseLock:
                with self._digitalIOLock:
                    if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                        raise RuntimeError("Device reports an existing active stream.")

                    for channel in channelNames:
                        self.ljm.eWriteName(
                            self.h, f"{channel}_RANGE", self.range
                        )
                    # Same supported stream-clock contract as calibrated digital IO.
                    self.ljm.eWriteName(self.h, "STREAM_TRIGGER_INDEX", 0)
                    self.ljm.eWriteName(self.h, "STREAM_CLOCK_SOURCE", 0)
                    addresses = self.ljm.namesToAddresses(
                        self.numChannels, channelNames
                    )[0]

                    record = self._analogStats["startRecord"]
                    stage = "clockValidation"
                    pre = self._readClockRegister(
                        "CORE_TIMER", record, "coreBeforeStart"
                    )
                    preTicks = self._clock.resolveRead(pre)

                    stage = "nativeStreamStart"
                    record["startCallBefore"] = pglTimestamp.getSecs()
                    try:
                        actualRate = self.ljm.eStreamStart(
                            self.h, self.scansPerRead, self.numChannels,
                            addresses, float(scanRate),
                        )
                    except BaseException:
                        # A native start exception can leave partial ownership.
                        self._analogStreamActive = True
                        raise
                    finally:
                        record["startCallAfter"] = pglTimestamp.getSecs()

                    self._analogStreamActive = True
                    stage = "clockValidation"
                    if not np.isfinite(actualRate) or actualRate <= 0:
                        raise ValueError(f"Invalid actual scan rate: {actualRate}")
                    self.scanRate = float(actualRate)
                    # Preserve the returned rate even if subsequent timing checks fail.
                    self._analogStats["scanRate"] = self.scanRate
                    record["scanRate"] = self.scanRate
                    self.totalScans = (
                        None if duration is None else int(np.ceil(duration * self.scanRate))
                    )
                    self.totalReads = (
                        None if self.totalScans is None else
                        int(np.ceil(self.totalScans / self.scansPerRead))
                    )

                    stamp = self._readClockRegister(
                        "STREAM_START_TIME_STAMP", record, "streamStartStamp"
                    )
                    post = self._readClockRegister(
                        "CORE_TIMER", record, "coreAfterStart"
                    )
                    postTicks = self._clock.resolveRead(post)

                    startTicks = self._validateAnalogStartStamp(
                        record, preTicks, postTicks
                    )

                    self.analogStartTimestamp = self._clock.toHost(startTicks)
                    self._analogClockSegment = {
                        "segmentId": 0,
                        "streamStartTicks": int(startTicks),
                        "rawStreamStartTicks": stamp["rawTicks"],
                        "scanRate": self.scanRate,
                        "mappingId": self._clock.mapping.eventId,
                        "nominalTicksPerSecond": 40_000_000,
                        "timestampMethod": "calibratedStreamStartSample",
                        "deviceTimestampMethod": "streamStartStampPlusSampleOffset",
                    }

            stage = "workerStart"
            self._analogStats.update({
                "scanRate": self.scanRate,
                "targetSampleCount": self.totalScans,
                "segment": deepcopy(self._analogClockSegment),
                "ready": True,
                "readyTimestamp": pglTimestamp.getSecs(),
            })
            self.isReading = True
            self.acquisitionThread = threading.Thread(
                target=self._analogReadThread,
                name="pglLabJackAnalogInput", daemon=True,
            )
            self.acquisitionThread.start()
            return self.scanRate

        except BaseException as error:
            self._analogStats["error"] = f"{type(error).__name__}: {error}"
            self._analogStats["ready"] = False
            if stage == "clockValidation":
                self._clock.valid = False

            with self._digitalIOLock:
                if self._analogStreamActive:
                    try:
                        try:
                            self.ljm.eStreamStop(self.h)
                        except Exception:
                            if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                                raise
                        self._analogStreamActive = False
                    except Exception as cleanupError:
                        self._analogStats["cleanupErrors"].append(
                            f"{type(cleanupError).__name__}: {cleanupError}"
                        )
            # Preserve a failed-ownership flag so another stream cannot start.
            self.isReading = self._analogStreamActive
            pglMessages.warning(
                f"LabJack analog acquisition could not start: "
                f"{type(error).__name__}: {error}. "
                "Settings were not changed. Diagnostics have been retained.",
                level=2,
            )
            raise
           
    def _analogReadThread(self):
        """Retain raw scans; stop at the first invalid/gap batch without compression."""
        try:
            while not self.stopEvent.is_set():
                with self.bufferLock:
                    returned = self._analogStats["samplesReturned"]
                if self.totalScans is not None and returned >= self.totalScans:
                    break

                with self._digitalIOLock:
                    if self.stopEvent.is_set():
                        break
                    before = pglTimestamp.getSecs()
                    values, deviceBacklog, ljmBacklog = self.ljm.eStreamRead(self.h)
                    after = pglTimestamp.getSecs()

                raw = np.asarray(values, dtype=np.float64)
                batch = {
                    "hostReadBefore": before,
                    "hostReadAfter": after,
                    "firstReturnedSampleIndex": returned,
                    "valueCount": int(raw.size),
                    "deviceBacklog": int(deviceBacklog),
                    "ljmBacklog": int(ljmBacklog),
                    "classification": "valid",
                }

                malformed = (
                    raw.ndim != 1 or raw.size == 0
                    or raw.size % self.numChannels != 0
                )
                invalidOffset = None
                if not malformed:
                    rows = raw.reshape(-1, self.numChannels)
                    invalidRows = np.flatnonzero(
                        np.any(~np.isfinite(rows) | (rows == -9999), axis=1)
                    )
                    if invalidRows.size:
                        invalidOffset = int(invalidRows[0])

                with self.bufferLock:
                    # Preserve every returned value, including invalid markers
                    # and incomplete trailing words.
                    self.analogBuffer.extend(raw.reshape(-1).tolist())
                    completeScans = raw.size // self.numChannels
                    self._analogStats["samplesReturned"] += int(completeScans)

                    if malformed or invalidOffset is not None:
                        batch["classification"] = (
                            "malformedBatch" if malformed else "invalidOrMissingScan"
                        )
                        batch["rawFailureValues"] = [
                            float(value) if np.isfinite(value) else repr(float(value))
                            for value in raw.reshape(-1)
                        ]
                        firstInvalid = returned + (
                            0 if malformed else invalidOffset
                        )
                        self._analogStats["timelineValid"] = False
                        self._analogStats["firstInvalidSampleIndex"] = firstInvalid
                        self._analogStats["validSampleCount"] = firstInvalid
                    else:
                        self._analogStats["validSampleCount"] += int(completeScans)

                    self._analogStats["batchDiagnostics"].append(batch)

                if malformed or invalidOffset is not None:
                    raise RuntimeError(
                        "Analog stream contains missing/invalid samples. "
                        "Raw values retained; timing after the first invalid position "
                        "is not treated as continuous."
                    )

        except Exception as error:
            with self.bufferLock:
                self._analogStats["error"] = f"{type(error).__name__}: {error}"
            pglMessages.warning(
                f"LabJack analog acquisition stopped: {error}. "
                "Partial data and diagnostics have been retained.",
                level=2,
            )

        finally:
            try:
                with self._digitalIOLock:
                    if self._analogStreamActive:
                        try:
                            self.ljm.eStreamStop(self.h)
                        except Exception:
                            if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                                raise
                        self._analogStreamActive = False
            except Exception as error:
                with self.bufferLock:
                    self._analogStats["cleanupErrors"].append(
                        f"{type(error).__name__}: {error}"
                    )
                pglMessages.warning(
                    f"LabJack analog stream cleanup failed: {error}. "
                    "Ownership and diagnostics are retained.",
                    level=2,
                )
            finally:
                # Block another acquisition if hardware ownership is unresolved.
                self.isReading = self._analogStreamActive

    def stopAnalogRead(self, waitToFinish=False, doNotTruncate=False):
        """
        Stop/join owned acquisition and return its retained trace, including partial data.

        time remains device-relative seconds. Hardware-derived origin and
        calibration records are in trace.timing.

        Worker data errors are exposed in trace.timing/status rather than erasing
        partial data. An unresolved worker/stream ownership failure raises.
        """
        from copy import deepcopy

        with self._streamControlLock:
            if waitToFinish and getattr(self, 'totalScans', None) is None:
                raise ValueError('Stop-controlled acquisition requires waitToFinish=False.')
            if not waitToFinish:
                self.stopEvent.set()
            thread = self.acquisitionThread
            if thread is threading.current_thread():
                raise RuntimeError("Analog worker cannot join itself.")
            if thread is not None and thread.is_alive():
                # Preserve waitToFinish behavior for intentional duration waits.
                thread.join(timeout=None if waitToFinish else 5.0)
                if thread.is_alive():
                    raise TimeoutError("Analog worker did not stop; handle retained.")

            status = self.getAnalogReadStatus()
            if status.get("streamActive"):
                raise RuntimeError("Analog stream ownership is unresolved.")

            with self.bufferLock:
                raw = np.asarray(self.analogBuffer, dtype=np.float64).copy()

            segment = getattr(self, "_analogClockSegment", None)
            if segment is None:
                if raw.size:
                    raise RuntimeError("Raw analog data have no validated timing segment.")
                return None

            completeCount = raw.size // self.numChannels
            data = raw[:completeCount * self.numChannels].reshape(
                completeCount, self.numChannels
            )
            returnedCount = completeCount
            if not doNotTruncate and self.totalScans is not None:
                data = data[:self.totalScans]
            sampleCount = len(data)

            timing = {
                "acquisitionId": status["acquisitionId"],
                "connectionId": self.connectionId,
                "segment": deepcopy(segment),
                "liveMappingId": segment["mappingId"],
                "mappingId": segment["mappingId"],
                "validSampleCount": min(
                    sampleCount, status["validSampleCount"]
                ),
                "timelineValid": status["timelineValid"],
                "status": status,
                "clockEvents": self.getClockEvents(),
                "returnedCompleteSampleCount": returnedCount,
                "truncatedSampleCount": returnedCount - sampleCount,
                "incompleteTrailingValues": [
                    float(value) if np.isfinite(value) else repr(float(value))
                    for value in raw[completeCount * self.numChannels:]
                ],
                "channelApertureOffsetsSeconds": None,
                "channelApertureUncertainty": "notEstablished",
            }
            return pglAnalogTraceData(
                time=np.arange(sampleCount) / self.scanRate,
                data=data,
                channelNames=self.channels,
                scanRate=self.scanRate,
                timing=timing,
            )
              
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
        self._clockStopEvent.set()
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
                        if getattr(self, "_analogStreamActive", False):
                            try:
                                self.ljm.eStreamStop(self.h)
                            except Exception:
                                if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                                    raise
                            self._analogStreamActive = False
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
                            "timestampMethod": "calibratedStreamStartSample",
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

                        actualRate, clockSegment = self._startClockedDigitalStream(
                            scansPerRead, addresses, scanRate
                        )
                        before = clockSegment["startCallBefore"]
                        after = clockSegment["startCallAfter"]

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
                                clockSegment["startTimestamp"], bankShift, bankMask,
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
                        self._warnDigitalStreamFailure("startup", error, self._digitalStats)
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
                    "timestampMethod": "calibratedStreamStartSample",
                    "clock": deepcopy(self._digitalCurrentClockSegment),
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
            event.labJackTimestamp = None
            event.clockMappingId = None
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
                        rate, clockSegment = self._startClockedDigitalStream(
                            config["scansPerRead"], addresses,
                            config["requestedScanRate"],
                        )
                        before = clockSegment["startCallBefore"]
                        after = clockSegment["startCallAfter"]

                        if self._digitalStopEvent.is_set():
                            stopStream()
                            return None

                    with self._digitalEventLock:
                        attemptRecord["outcome"] = "restarted"
                    return float(rate), before, after, clockSegment["startTimestamp"]

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

                    if self._clock is None or not self._clock.valid:
                        raise RuntimeError(
                            "Clock validation failed; recovery cannot reuse this mapping."
                        ) from error

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
                        event.eventId = (
                            f"{acquisitionId}:{segmentId}:{event.sampleIndex}:"
                            f"{event.channel}:{event.eventType}"
                        )
                        event.segmentId = segmentId
                        event.sampleIndexScope = "timingSegment"
                        event.timestampMethod = "calibratedStreamStartSample"
                        clockSegment = self._digitalCurrentClockSegment
                        event.labJackTimestamp = (
                            clockSegment["streamStartTicks"] / pglLabJackClock.nominalTicksPerSecond
                            + event.sampleTime
                        )
                        event.clockMappingId = clockSegment["mappingId"]
                        event.deviceTimestampMethod = "streamStartStampPlusSampleOffset"
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
                    self._warnDigitalStreamFailure("recovery", error, config)
                    raise

                if restarted is None:
                    with self._digitalEventLock:
                        record["outcome"] = "cancelledByStop"
                    break

                currentRate, before, after, currentStart = restarted
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
        status["clockEvents"] = self.getClockEvents()
        status["clockStreamStarts"] = deepcopy(self._clockStreamStarts)
        status["clockValid"] = self._clock is not None and self._clock.valid
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
    

    def getClockEvents(self):
        """Independent retained calibration/mapping snapshots; no hardware access."""
        from copy import deepcopy
        return deepcopy(self._clock.events if self._clock is not None else [])

    def _readClockRegister(self, name, record, fieldName):
        """Caller owns digital-IO lock. Retain brackets even if reading fails."""
        row = {
            "register": name,
            "hostBefore": pglTimestamp.getSecs(),
            "rawTicks": None,
            "failure": None,
        }
        record[fieldName] = row
        try:
            value = self.ljm.eReadName(self.h, name)
            row["rawValueRepresentation"] = repr(value)
            row["rawTicks"] = pglLabJackClock._rawTicks(value)
        except BaseException as error:
            row["failure"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["hostAfter"] = pglTimestamp.getSecs()
            row["roundTripSeconds"] = row["hostAfter"] - row["hostBefore"]
        return row

    def _startClockedDigitalStream(self, scansPerRead, addresses, requestedRate):
        """
        Start a stream and validate its hardware-derived origin.

        Caller holds _digitalIOLock.

        A native start exception leaves stream ownership potentially active.
        The caller must confirm stopped ownership before another attempt.
        Such an exception alone does not invalidate the clock mapping.

        Invalid timing evidence invalidates the mapping. Never fall back to
        host midpoint origins.
        """
        from copy import deepcopy

        record = {
            "outcome": "starting",
            "mappingId": (
                self._clock.mapping.eventId
                if self._clock is not None and self._clock.mapping is not None
                else None
            ),
            "failure": None,
            "failureStage": None,
            "requiresStopConfirmation": False,
        }
        self._clockStreamStarts.append(record)
        stage = "preStartClockValidation"

        try:
            if self._clock is None or not self._clock.valid:
                raise RuntimeError("No valid connection clock mapping.")

            pre = self._readClockRegister("CORE_TIMER", record, "coreBeforeStart")
            preTicks = self._clock.resolveRead(pre)

            stage = "nativeStreamStart"
            record["startCallBefore"] = pglTimestamp.getSecs()
            try:
                rate = self.ljm.eStreamStart(
                    self.h, scansPerRead, len(addresses), addresses, requestedRate
                )
            except BaseException:
                # A native exception does not establish whether the device
                # partially started. Make cleanup own that uncertainty.
                self._digitalStreamActive = True
                record["requiresStopConfirmation"] = True
                raise
            finally:
                record["startCallAfter"] = pglTimestamp.getSecs()

            # Ownership is active before rate or stamp validation can fail.
            self._digitalStreamActive = True

            stage = "scanRateValidation"
            if not np.isfinite(rate) or rate <= 0:
                raise RuntimeError(f"Invalid actual scan rate: {rate}")
            record["scanRate"] = float(rate)

            stage = "streamClockValidation"
            stamp = self._readClockRegister(
                "STREAM_START_TIME_STAMP", record, "streamStartStamp"
            )
            post = self._readClockRegister("CORE_TIMER", record, "coreAfterStart")
            postTicks = self._clock.resolveRead(post)

            if not 0 <= postTicks - preTicks < pglLabJackClock.timerModulus:
                raise RuntimeError(
                    "Stream-start timer interval cannot uniquely locate the stamp."
                )

            stampTicks = preTicks + (
                (stamp["rawTicks"] - pre["rawTicks"])
                % pglLabJackClock.timerModulus
            )
            if not preTicks <= stampTicks <= postTicks:
                raise RuntimeError("Stale or inconsistent stream-start timestamp.")

            stamp["unwrappedTicks"] = int(stampTicks)
            record.update({
                "streamStartTicks": int(stampTicks),
                "startTimestamp": self._clock.toHost(stampTicks),
                "timestampMethod": "calibratedStreamStartSample",
                "deviceTimestampMethod": "streamStartStampPlusSampleOffset",
                "outcome": "validated",
            })
            self._digitalCurrentClockSegment = deepcopy(record)
            return float(rate), deepcopy(record)

        except BaseException as error:
            record.update({
                "outcome": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "failureStage": stage,
            })
            if stage != "nativeStreamStart" and self._clock is not None:
                self._clock.valid = False
            raise

    def _warnDigitalStreamFailure(self, stage, error, config):
        """
        Warn using the failing acquisition's settings, not today's preset.

        No error-code taxonomy is assumed. Buffer/rate advice is conditional
        and is not offered as a remedy for an invalid clock mapping.
        """
        scanRate = config.get("requestedScanRate")
        bufferBytes = config.get("streamBufferSizeBytes")
        if bufferBytes is None:
            bufferBytes = config.get("requestedStreamBufferSizeBytes")
        scanNames = config.get("scanNames")

        details = []
        if scanRate is not None:
            details.append(f"requested scan rate={scanRate:g} scans/s")
        if bufferBytes is not None:
            details.append(f"stream buffer={bufferBytes} bytes")
        if isinstance(scanNames, (list, tuple)):
            details.append(f"stream addresses={len(scanNames)}")

        headline = (
            "LabJack acquisition could not start."
            if stage == "startup"
            else "LabJack stream recovery failed; acquisition cannot continue."
        )
        message = f"{headline} {type(error).__name__}: {error}."
        if details:
            message += " Acquisition settings: " + ", ".join(details) + "."

        clockInvalid = self._clock is None or not self._clock.valid
        if clockInvalid:
            message += (
                " Clock validation is unavailable or failed. Resolve the "
                "reported clock/device problem before starting a new acquisition; "
                "changing buffer size or scan rate is not a clock-calibration remedy."
            )
        else:
            if bufferBytes is not None and bufferBytes < 32768:
                advice = (
                    "If the reported cause is buffering or throughput, try a "
                    "larger supported stream buffer (up to 32768 bytes) or a "
                    "lower scan rate in the LabJack preset, then run again."
                )
            elif bufferBytes == 32768:
                advice = (
                    "If the reported cause is buffering or throughput, the "
                    "stream buffer is already at its supported maximum of "
                    "32768 bytes. Try a lower scan rate in the LabJack preset, "
                    "then run again."
                )
            else:
                advice = (
                    "If the reported cause is buffering or throughput, review "
                    "the stream buffer and scan rate in the LabJack preset "
                    "before retrying."
                )
            message += (
                " " + advice
                + " For other causes, address the reported device/configuration "
                  "error before retrying."
            )

        message += (
            " Settings were not changed automatically. Failure and any cleanup "
            "diagnostics have been retained."
        )
        pglMessages.warning(message, level=2)

    def calibrateClockEnding(self, calibrationEventIds=None):
        """
        Collect an ending calibration and fit it with explicitly selected records.

        Default: use the calibration IDs supporting the frozen live mapping.
        Explicit IDs: use exactly those earlier collections plus the new ending
        collection. Do not automatically include other connection history.

        All collection and fitting evidence is retained by the clock object.
        Does not close the connection, replace the live mapping, or correct events.
        """
        with self._streamControlLock:
            if self._clock is None or self._clock.mapping is None:
                raise RuntimeError("No beginning clock mapping exists.")

            if calibrationEventIds is None:
                selectedIds = list(self._clock.mapping.calibrationEventIds)
            else:
                if not isinstance(calibrationEventIds, (list, tuple)):
                    raise TypeError("calibrationEventIds must be a list or tuple.")
                selectedIds = list(calibrationEventIds)

            if (
                not selectedIds
                or any(not isinstance(value, str) or not value for value in selectedIds)
                or len(set(selectedIds)) != len(selectedIds)
            ):
                raise ValueError("Supply distinct nonempty calibration-event IDs.")

            # Validate references before reading hardware. Do not fit or alter
            # prior records here; the shared fitter validates their observations.
            for eventId in selectedIds:
                matches = [
                    event for event in self._clock.events
                    if event.type == "labJackClockCalibration"
                    and event.eventId == eventId
                ]
                if len(matches) != 1:
                    raise ValueError(f"Calibration ID missing or ambiguous: {eventId}")
                event = matches[0]
                if event.connectionId != self._clock.connectionId:
                    raise ValueError("Cannot combine different device connections.")
                if event.status != "collected":
                    raise ValueError(f"Calibration did not complete successfully: {eventId}")

            # RLock permits the shared collection/fit methods to enforce their
            # own lifecycle rules while this compound operation retains control.
            ending = self.collectClockCalibration(phase="ending")
            return self.fitClockCalibrations(
                selectedIds + [ending.eventId], purpose="postRun"
            )

    def getClockTiming(self, deviceTicks, mappingId=None):
        """Hardware-free conversion and conditional uncertainty query."""
        if self._clock is None:
            raise RuntimeError("No connection clock calibration exists.")
        return self._clock.getTiming(deviceTicks, mappingId=mappingId)

    def collectClockCalibration(self, phase="intermediate"):
        """
        Collect additional clock evidence at an idle acquisition boundary.

        Does not stop a stream, fit a mapping, or change the live mapping.
        """
        if phase not in {"beginning", "intermediate", "ending", "characterization"}:
            raise ValueError("Invalid additional calibration phase.")
        with self._streamControlLock:
            if self.h is None:
                raise RuntimeError("LabJack is not connected.")
            for thread in (self._digitalThread, self.acquisitionThread):
                if thread is not None and thread.is_alive():
                    raise RuntimeError("Additional clock collection currently requires idle acquisition.")
            if self._digitalStreamActive or self.isReading:
                raise RuntimeError("Acquisition still owns the stream.")
            if getattr(self, "_digitalCounterRestore", None) is not None:
                raise RuntimeError("Counter restoration is incomplete.")
            if self._clockStopEvent.is_set():
                raise RuntimeError("Device closure/cancellation was requested.")
            if self._clock is None:
                raise RuntimeError("No connection clock exists.")

            with self._wordPulseLock:
                with self._digitalIOLock:
                    if self.ljm.eReadName(self.h, "STREAM_ENABLE") != 0:
                        raise RuntimeError("Device reports an active stream.")
                    return self._clock.collectCalibration(
                        lambda: self.ljm.eReadName(self.h, "CORE_TIMER"),
                        pglTimestamp.getSecs,
                        self.clockCalibrationDuration,
                        self._clockStopEvent.is_set,
                        phase,
                    )

    def fitClockCalibrations(self, calibrationEventIds, purpose="postRun"):
        """
        Hardware-free fit using explicitly selected calibration-event IDs.

        Does not automatically select unrelated collections or correct events.
        """
        with self._streamControlLock:
            if self._clock is None:
                raise RuntimeError("No retained clock records exist.")
            return self._clock.fitCalibrations(
                calibrationEventIds, pglTimestamp.getSecs, purpose=purpose
            )

    def getClockCalibrationReport(self, mappingId=None):
        """
        Hardware-free inspection, including rejected/incomplete mapping records.

        Unresolved references are reported explicitly. No observations, bounds,
        or successful mapping status are fabricated.
        """
        from copy import deepcopy

        if self._clock is None:
            raise RuntimeError("No retained clock evidence exists.")
        if mappingId is None:
            if self._clock.mapping is None:
                raise RuntimeError("No live mapping exists; specify a retained mapping ID.")
            mappingId = self._clock.mapping.eventId

        events = self.getClockEvents()
        matches = [
            event for event in events
            if event.type == "labJackClockMapping" and event.eventId == mappingId
        ]
        if len(matches) != 1:
            raise ValueError("Mapping ID itself is missing or ambiguous.")
        mapping = matches[0]

        collections, referenceErrors = [], []
        for calibrationId in mapping.calibrationEventIds:
            matches = [
                event for event in events
                if event.type == "labJackClockCalibration"
                and event.eventId == calibrationId
            ]
            if len(matches) != 1:
                referenceErrors.append({
                    "calibrationEventId": calibrationId,
                    "reason": "missing" if not matches else "ambiguous",
                    "matchCount": len(matches),
                })
                continue
            event = matches[0]
            if event.connectionId != mapping.connectionId:
                referenceErrors.append({
                    "calibrationEventId": calibrationId,
                    "reason": "foreignConnection",
                    "matchCount": 1,
                })
                continue
            collections.append(event)

        summaries, validRows = [], []
        for collection in collections:
            rtts = []
            for row in collection.observations:
                before, after = row.get("hostBefore"), row.get("hostAfter")
                if (
                    isinstance(before, (int, float))
                    and isinstance(after, (int, float))
                    and np.isfinite(before) and np.isfinite(after)
                    and after >= before
                ):
                    rtts.append(after - before)
                ticks = row.get("unwrappedTicks")
                if isinstance(ticks, (int, np.integer)) and not isinstance(ticks, (bool, np.bool_)):
                    validRows.append(row)

            summaries.append({
                "calibrationEventId": collection.eventId,
                "phase": collection.phase,
                "status": collection.status,
                "requestedDurationSeconds": collection.requestedDuration,
                "actualDurationSeconds": collection.actualDuration,
                "observationCount": len(collection.observations),
                "selectedObservationCount": sum(
                    row.get("selected") is True for row in collection.observations
                ),
                "rttSeconds": (
                    {
                        "minimum": float(np.min(rtts)),
                        "median": float(np.median(rtts)),
                        "p95": float(np.percentile(rtts, 95)),
                        "maximum": float(np.max(rtts)),
                    } if rtts else None
                ),
            })

        report = {
            "mappingId": mapping.eventId,
            "connectionId": mapping.connectionId,
            "purpose": mapping.purpose,
            "status": mapping.status,
            "method": mapping.method,
            "methodVersion": mapping.methodVersion,
            "calibrationEventIds": list(mapping.calibrationEventIds),
            "collections": summaries,
            "mappingEvent": mapping,
            "calibrationEvents": collections,
            "referenceErrors": referenceErrors,
            "reportStatus": "unavailable",
            "reportFailureReason": None,
            "failureReason": mapping.diagnostics.get("failureReason"),
            "clockScale": None,
            "scaleDeviationPpm": None,
            "feasibleScaleDeviationPpm": None,
            "collectionSeparationSeconds": None,
            "queries": [],
            "assumptions": deepcopy(mapping.assumptions),
            "unknownContributions": [
                "hostTimestampUncertainty",
                "withinRunNonlinearClockVariation",
            ],
            "totalTimingBoundAvailable": False,
        }

        if referenceErrors:
            report["reportStatus"] = "unresolvedReferences"
            return report
        if mapping.status != "validConditional":
            report["reportStatus"] = "mappingRejected"
            return report

        try:
            if len(collections) > 1:
                midpoints = [
                    event.collectionStartedTimestamp
                    + (event.collectionCompletedTimestamp - event.collectionStartedTimestamp) / 2
                    for event in collections
                ]
                report["collectionSeparationSeconds"] = max(midpoints) - min(midpoints)

            parameters = mapping.parameters
            scale = parameters["clockScale"]
            polygon = parameters["affineRegion"]
            if not np.isfinite(scale) or scale <= 0 or not polygon:
                raise ValueError("Invalid mapping parameters.")
            bounds = [
                min(point[1] for point in polygon) * 1e6,
                max(point[1] for point in polygon) * 1e6,
            ]
            if not np.all(np.isfinite(bounds)):
                raise ValueError("Nonfinite scale bounds.")

            queries = []
            if validRows:
                firstTicks = min(row["unwrappedTicks"] for row in validRows)
                lastTicks = max(row["unwrappedTicks"] for row in validRows)
                for label, ticks in (
                    ("beginning", firstTicks),
                    ("midpoint", firstTicks + (lastTicks - firstTicks) / 2),
                    ("ending", lastTicks),
                ):
                    timing = self.getClockTiming(ticks, mapping.eventId)
                    liveTiming = (
                        self.getClockTiming(ticks, self._clock.mapping.eventId)
                        if self._clock.mapping is not None else None
                    )
                    queries.append({
                        "label": label,
                        "deviceTicks": ticks,
                        **timing,
                        "correctionFromLiveSeconds": (
                            timing["timestamp"] - liveTiming["timestamp"]
                            if liveTiming is not None else None
                        ),
                    })

            report.update({
                "clockScale": scale,
                "scaleDeviationPpm": (scale - 1) * 1e6,
                "feasibleScaleDeviationPpm": bounds,
                "queries": queries,
                "reportStatus": "availableConditional",
            })

        except Exception as error:
            report["reportStatus"] = "reportFailed"
            report["reportFailureReason"] = f"{type(error).__name__}: {error}"

        return report

    def getClockCharacterizationHistory(self):
        """Independent serializable characterization snapshots; no hardware reads."""
        from copy import deepcopy
        return deepcopy(getattr(self, "_clockCharacterizationHistory", []))

    def _getClockCheckSummary(self, mappingId):
        """
        Prepare a human-readable two-collection clock check.

        Signs:
            Alignment offset = system time minus nominal device time.
            Negative drift means the LabJack timer gains on the system clock.

        All intervals are conditional on the retained affine clock model.
        """
        import numpy as np

        report = self.getClockCalibrationReport(mappingId)
        if report["reportStatus"] != "availableConditional":
            raise ValueError(
                report.get("failureReason")
                or report.get("reportFailureReason")
                or f"Clock report unavailable: {report['reportStatus']}"
            )

        collections = sorted(
            report["calibrationEvents"],
            key=lambda event: event.collectionStartedTimestamp,
        )
        if len(collections) != 2:
            raise ValueError(
                "This simplified clock-check report requires exactly two "
                "collections. The general multi-collection mapping remains supported."
            )

        mapping = report["mappingEvent"]
        parameters = mapping.parameters
        rate = parameters["nominalTicksPerSecond"]
        tickReference = parameters["tickReference"]
        hostBase = parameters["hostBase"]
        selectedOffset = parameters["hostReferenceOffset"]
        selectedScale = parameters["clockScale"]
        region = np.asarray(parameters["affineRegion"], dtype=float)

        summaries = []
        for index, collection in enumerate(collections):
            rows = collection.observations
            if not rows:
                raise ValueError("Calibration collection has no observations.")

            x = np.asarray([
                (row["unwrappedTicks"] - tickReference) / rate
                for row in rows
            ])
            before = np.asarray([
                row["hostBefore"] - hostBase for row in rows
            ])
            after = np.asarray([
                row["hostAfter"] - hostBase for row in rows
            ])
            midpoint = before + (after - before) / 2

            # Representative device coordinate within this collection.
            referenceX = float(np.median(x))

            # Bring each observation to this collection's reference position
            # using the selected slope. This removes only the small modeled
            # drift DURING the collection, not the drift between collections.
            offsets = (
                midpoint - x
                + (selectedScale - 1) * (referenceX - x)
            )

            # Joint feasible region projected at this collection's reference.
            possibleOffsets = region[:, 0] + region[:, 1] * referenceX
            estimatedOffset = selectedOffset + (selectedScale - 1) * referenceX
            errorLower = float(np.min(possibleOffsets) - estimatedOffset)
            errorUpper = float(np.max(possibleOffsets) - estimatedOffset)

            summaries.append({
                "number": index + 1,
                "calibrationEventId": collection.eventId,
                "observationCount": len(rows),
                "selectedObservationCount": sum(
                    row.get("selected") is True for row in rows
                ),
                "durationSeconds": collection.actualDuration,
                "deviceSeconds": referenceX,
                "estimatedOffsetSeconds": float(estimatedOffset),
                "alignmentErrorSeconds": [errorLower, errorUpper],
                "readAlignmentErrorsSeconds": (
                    offsets - estimatedOffset
                ).tolist(),
                "readHalfBracketSeconds": ((after - before) / 2).tolist(),
                "hostReferenceTime": (
                    hostBase + referenceX + estimatedOffset
                ),
            })

        first, second = summaries
        deviceSpan = second["deviceSeconds"] - first["deviceSeconds"]
        if deviceSpan <= 0:
            raise ValueError("Calibration references have no positive separation.")

        elapsed = second["hostReferenceTime"] - first["hostReferenceTime"]
        drift = (selectedScale - 1) * deviceSpan
        driftBounds = [
            float(np.min(region[:, 1]) * deviceSpan),
            float(np.max(region[:, 1]) * deviceSpan),
        ]

        # Express device gain relative to system time, not its reciprocal.
        # This is the rate of nominal LabJack seconds per system second minus 1.
        scaleLower = 1 + float(np.min(region[:, 1]))
        scaleUpper = 1 + float(np.max(region[:, 1]))
        if scaleLower <= 0:
            raise ValueError("Invalid feasible clock scale.")

        deviceRatePpm = (1 / selectedScale - 1) * 1e6
        deviceRateBoundsPpm = [
            (1 / scaleUpper - 1) * 1e6,
            (1 / scaleLower - 1) * 1e6,
        ]

        # System-minus-device drift after 600 system seconds.
        driftTenMinutes = 600 * (1 - 1 / selectedScale)
        driftTenMinutesBounds = [
            600 * (1 - 1 / scaleLower),
            600 * (1 - 1 / scaleUpper),
        ]

        return {
            "mappingId": mapping.eventId,
            "collections": summaries,
            "elapsedSeconds": float(elapsed),
            "deviceSpanSeconds": float(deviceSpan),
            "driftSeconds": float(drift),
            "driftIntervalSeconds": driftBounds,
            "expectedDriftTenMinutesSeconds": float(driftTenMinutes),
            "expectedDriftTenMinutesIntervalSeconds": driftTenMinutesBounds,
            "deviceRateDeviationPpm": float(deviceRatePpm),
            "deviceRateDeviationIntervalPpm": deviceRateBoundsPpm,
            "affineRegion": region.tolist(),
            "selectedOffsetSeconds": selectedOffset,
            "selectedScale": selectedScale,
            "totalTimingAccuracyEstablished": False,
        }

    def printClockCalibrationReport(self, mappingId=None):
        """
        Print an understandable two-collection clock-check summary.

        Returns the summary. Other mappings still have the general
        getClockCalibrationReport() interface.
        """
        if mappingId is None:
            if self._clock is None or self._clock.mapping is None:
                raise RuntimeError("No clock mapping is available.")
            mappingId = self._clock.mapping.eventId

        pglMessages.printHeader("Clock calibration")

        try:
            summary = self._getClockCheckSummary(mappingId)
        except Exception as error:
            print(f"Clock-check summary unavailable: {error}")
            print("Unavailable uncertainty does not mean zero.")
            return self.getClockCalibrationReport(mappingId)

        for collection in summary["collections"]:
            lower, upper = collection["alignmentErrorSeconds"]
            print(
                f"Calibration {collection['number']}: "
                f"clock-alignment uncertainty "
                f"[{lower * 1e6:+.1f}, {upper * 1e6:+.1f}] µs "
                f"(n={collection['observationCount']}, "
                f"{collection['durationSeconds'] * 1000:.1f} ms collection)"
            )

        lower, upper = summary["driftIntervalSeconds"]
        print(
            f"\nDrift between system clock and LabJack clock over "
            f"{summary['elapsedSeconds']:.3f} s = "
            f"{summary['driftSeconds'] * 1e6:+.1f} µs "
            f"(supported range [{lower * 1e6:+.1f}, {upper * 1e6:+.1f}] µs)"
        )

        lower, upper = summary["expectedDriftTenMinutesIntervalSeconds"]
        print(
            f"Expected drift over 10 minutes: "
            f"{summary['expectedDriftTenMinutesSeconds'] * 1000:+.3f} ms "
            f"(range [{lower * 1000:+.3f}, {upper * 1000:+.3f}] ms)"
        )

        lower, upper = summary["deviceRateDeviationIntervalPpm"]
        print(
            f"Implied T7 clock rate relative to system clock: "
            f"{summary['deviceRateDeviationPpm']:+.3f} ppm "
            f"(range [{lower:+.3f}, {upper:+.3f}] ppm)"
        )
        print(
            "\nNegative drift means the LabJack timer gains on the system clock; "
            "positive T7 ppm means it runs faster."
        )
        print(
            "Collection uncertainty uses all selected brackets in the joint fit. "
            "These are conditional alignment bounds, not independently measured "
            "absolute clock accuracy."
        )
        print(
            "The 10-minute prediction assumes constant relative rate. "
            "Nonlinear drift and host timestamp uncertainty remain unbounded."
        )
        return summary

    def plotClockCalibration(self, mappingId=None):
        """
        Plot the two calibration distributions and their fitted drift.

        Uses ordinary pyplot figures for notebook rendering.
        Returns (distributionFigure, driftFigure); no hardware reads.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if mappingId is None:
            if self._clock is None or self._clock.mapping is None:
                raise RuntimeError("No clock mapping is available.")
            mappingId = self._clock.mapping.eventId

        try:
            summary = self._getClockCheckSummary(mappingId)
        except Exception as error:
            figure, axis = plt.subplots(
                figsize=(10, 4), constrained_layout=True
            )
            axis.set_axis_off()
            axis.text(
                0.02, 0.95,
                f"Clock calibration plot unavailable\n\n{error}\n\n"
                "Unavailable uncertainty does not mean zero.\n"
                "Original calibration records remain retained.",
                va="top", transform=axis.transAxes,
            )
            return figure, None

        distributionFigure, axes = plt.subplots(
            1, 2, figsize=(12, 4.5), constrained_layout=True,
            sharex=True,
        )
        collections = summary["collections"]
        pooled = np.concatenate([
            np.asarray(collection["readAlignmentErrorsSeconds"]) * 1e6
            for collection in collections
        ])
        binEdges = np.histogram_bin_edges(pooled, bins=30)

        for axis, collection in zip(axes, collections):
            values = np.asarray(collection["readAlignmentErrorsSeconds"]) * 1e6
            lower, upper = np.asarray(collection["alignmentErrorSeconds"]) * 1e6

            axis.hist(
                values, bins=binEdges, alpha=0.7,
                label="Individual timer-read estimates",
            )
            axis.axvspan(
                lower, upper, color="tab:green", alpha=0.25,
                label="Supported clock-alignment range",
            )
            axis.axvline(
                0, color="black", linewidth=1,
                label="Selected alignment",
            )
            axis.axvline(lower, color="tab:green", linestyle="--", linewidth=1)
            axis.axvline(upper, color="tab:green", linestyle="--", linewidth=1)
            axis.set(
                title=(
                    f"Calibration {collection['number']} "
                    f"(n={collection['observationCount']})\n"
                    f"Alignment uncertainty [{lower:+.1f}, {upper:+.1f}] µs"
                ),
                xlabel="Timer-read estimate minus selected alignment (µs)",
                ylabel="Observations (n)",
            )
            axis.grid(True, alpha=0.2)
            axis.legend(fontsize=8)

        distributionFigure.suptitle(
            "Clock calibration observations\n"
            "Histogram spread is communication variability; "
            "shaded bounds come from the joint transaction-bracket fit",
            fontsize=11,
        )

        driftFigure, axis = plt.subplots(
            figsize=(11, 5.5), constrained_layout=True
        )
        first, second = collections
        x0 = first["deviceSeconds"]
        span = summary["deviceSpanSeconds"]
        fractions = np.linspace(0, 1, 200)
        deviceCoordinates = x0 + fractions * span
        elapsed = fractions * summary["elapsedSeconds"]

        selectedOffset = summary["selectedOffsetSeconds"]
        selectedScale = summary["selectedScale"]
        region = np.asarray(summary["affineRegion"])
        firstSelectedOffset = first["estimatedOffsetSeconds"]

        line = (
            selectedOffset
            + (selectedScale - 1) * deviceCoordinates
            - firstSelectedOffset
        ) * 1e6
        possible = (
            region[:, 0, None]
            + region[:, 1, None] * deviceCoordinates[None, :]
            - firstSelectedOffset
        ) * 1e6

        axis.fill_between(
            elapsed, possible.min(axis=0), possible.max(axis=0),
            alpha=0.2, color="tab:blue",
            label="Conditional clock-alignment interval",
        )
        axis.plot(
            elapsed, line, color="tab:blue", linewidth=2,
            label="Fitted relative clock drift",
        )

        for index, collection in enumerate(collections):
            pointTime = 0 if index == 0 else summary["elapsedSeconds"]
            pointValue = (
                collection["estimatedOffsetSeconds"] - firstSelectedOffset
            ) * 1e6
            lower, upper = np.asarray(collection["alignmentErrorSeconds"]) * 1e6
            axis.errorbar(
                pointTime, pointValue,
                yerr=np.asarray([[-lower], [upper]]),
                fmt="o", capsize=6, color=f"C{index + 1}",
                label=f"Calibration {index + 1} alignment uncertainty",
            )

        driftText = (
            "Drift between system clock and LabJack clock "
            f"= {summary['driftSeconds'] * 1e6:+.1f} µs"
        )
        # Offset text above the midpoint. Keep horizontal for readability across
        # notebook sizes rather than forcing a potentially illegible rotation.
        axis.annotate(
            driftText,
            xy=(summary["elapsedSeconds"] / 2, summary["driftSeconds"] * 0.5e6),
            xytext=(0, 22), textcoords="offset points",
            ha="center", va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        axis.set(
            title=(
                f"Expected drift over 10 minutes: "
                f"{summary['expectedDriftTenMinutesSeconds'] * 1000:+.3f} ms\n"
                f"T7 rate relative to system clock: "
                f"{summary['deviceRateDeviationPpm']:+.3f} ppm"
            ),
            xlabel="Elapsed system time (s)",
            ylabel="System-minus-LabJack alignment change (µs)",
        )
        axis.axhline(0, color="black", linewidth=0.5)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8, loc="best")
        driftFigure.supxlabel(
            "Conditional constant-rate model. Error bars share calibration "
            "uncertainty; nonlinear drift is not bounded.",
            fontsize=9,
        )

        return distributionFigure, driftFigure

    def checkClockCalibration(self, endpointSeparation):
        """
        Run an idle-only, two-collection clock check and display its outcome.

        endpointSeparation is the start-to-start target in seconds.
        Collection duration comes from the device's calibration settings.

        Leaves the connection open and the live mapping unchanged.
        Existing characterization-history storage is retained for compatibility.
        """
        from copy import deepcopy
        from uuid import uuid4
        import numpy as np
        import matplotlib.pyplot as plt

        if isinstance(endpointSeparation, (bool, np.bool_)):
            raise ValueError("endpointSeparation must be positive and finite.")
        endpointSeparation = float(endpointSeparation)
        if (
            not np.isfinite(endpointSeparation)
            or endpointSeparation <= self.clockCalibrationDuration
        ):
            raise ValueError(
                "endpointSeparation must exceed the configured collection duration."
            )

        run = {
            "characterizationId": str(uuid4()),
            "operation": "checkClockCalibration",
            "endpointSeparationTargetSeconds": endpointSeparation,
            "collectionDurationTargetSeconds": self.clockCalibrationDuration,
            "status": "starting",
            "failureReason": None,
            "calibrationEventIds": [],
            "mappingId": None,
            "clockEvents": [],
            "report": None,
        }
        if not hasattr(self, "_clockCharacterizationHistory"):
            self._clockCharacterizationHistory = []
        self._clockCharacterizationHistory.append(run)

        previousIds = None
        try:
            with self._streamControlLock:
                if self.h is None or self._clock is None:
                    raise RuntimeError(
                        "An open calibrated LabJack connection is required."
                    )

                previousIds = {
                    event.eventId for event in self.getClockEvents()
                }
                print(
                    f"Running LabJack clock check: "
                    f"{self.clockCalibrationDuration * 1000:g} ms collections, "
                    f"{endpointSeparation:g} s separation.",
                    flush=True,
                )
                first = self.collectClockCalibration(phase="characterization")
                run["calibrationEventIds"].append(first.eventId)
                target = first.collectionStartedTimestamp + endpointSeparation

                if pglTimestamp.getSecs() >= target:
                    raise RuntimeError(
                        "The first collection exceeded the requested separation."
                    )

                while True:
                    remaining = target - pglTimestamp.getSecs()
                    if remaining <= 0:
                        break
                    if self._clockStopEvent.wait(min(remaining, 0.1)):
                        raise RuntimeError("Clock check cancelled.")

                second = self.collectClockCalibration(phase="characterization")
                run["calibrationEventIds"].append(second.eventId)

                mapping = self.fitClockCalibrations(
                    run["calibrationEventIds"], purpose="characterization"
                )
                run["mappingId"] = mapping.eventId
                run["clockEvents"] = [
                    event for event in self.getClockEvents()
                    if event.eventId not in previousIds
                ]
                run["report"] = self.getClockCalibrationReport(mapping.eventId)
                run["summary"] = self._getClockCheckSummary(mapping.eventId)
                run["status"] = "fitCompletedConditional"

            try:
                self.printClockCalibrationReport(run["mappingId"])
                self.plotClockCalibration(run["mappingId"])
                plt.show()
            except Exception as displayError:
                run["displayFailure"] = (
                    f"{type(displayError).__name__}: {displayError}"
                )
                print(
                    f"Calibration evidence retained; report/display failed: "
                    f"{displayError}",
                    flush=True,
                )

            return deepcopy(run)

        except BaseException as error:
            run["status"] = "failedOrInterrupted"
            run["failureReason"] = f"{type(error).__name__}: {error}"
            if self._clock is not None:
                run["clockEvents"] = [
                    event for event in self.getClockEvents()
                    if previousIds is None or event.eventId not in previousIds
                ]
            error.labJackCharacterizationReport = deepcopy(run)

            pglMessages.printHeader("Clock calibration")
            print(f"Clock check did not complete: {run['failureReason']}")
            print(
                "Partial observations and failure records are retained. "
                "No accuracy or drift estimate is claimed for this failed check."
            )
            raise    

    def getAnalogReadStatus(self):
        """Independent diagnostics without hardware reads."""
        from copy import deepcopy

        with self.bufferLock:
            status = deepcopy(getattr(self, "_analogStats", {}))
        thread = self.acquisitionThread
        status["running"] = thread is not None and thread.is_alive()
        status["streamActive"] = getattr(self, "_analogStreamActive", False)
        return status


    def getAnalogReadSnapshot(self):
        """
        Copy currently buffered analog data without hardware access or stopping.

        A snapshot is not a shutdown operation. If acquisition remains active,
        later samples are not included and remain in the driver.

        Data with no validated stream origin are retained but cannot be used
        for calibrated timing. Relative sample times are unavailable (NaN)
        when no actual scan rate was established.
        """
        from copy import deepcopy

        with self.bufferLock:
            status = deepcopy(getattr(self, "_analogStats", {}))
            if not status:
                raise RuntimeError("No analog acquisition record is available.")

            channelNames = list(status["channels"])
            channelCount = len(channelNames)
            if channelCount < 1:
                raise RuntimeError("Analog scan order is unavailable.")

            raw = np.asarray(self.analogBuffer, dtype=np.float64).copy()
            segment = deepcopy(getattr(self, "_analogClockSegment", None))
            thread = self.acquisitionThread
            workerRunning = thread is not None and thread.is_alive()
            streamActive = bool(getattr(self, "_analogStreamActive", False))
            snapshotTime = pglTimestamp.getSecs()

        completeCount = raw.size // channelCount
        data = raw[:completeCount * channelCount].reshape(
            completeCount, channelCount
        )
        rate = status.get("scanRate")
        validRate = (
            rate is not None
            and not isinstance(rate, (bool, np.bool_))
            and np.isfinite(rate) and rate > 0
        )
        originValidated = segment is not None and validRate
        validCount = (
            min(completeCount, int(status.get("validSampleCount", 0)))
            if originValidated else 0
        )

        if validRate:
            relativeTime = np.arange(completeCount) / float(rate)
        else:
            relativeTime = np.full(completeCount, np.nan)

        status["running"] = workerRunning
        status["streamActive"] = streamActive
        incomplete = (
            workerRunning or streamActive
            or not originValidated
            or bool(status.get("error"))
            or bool(status.get("cleanupErrors"))
            or not status.get("timelineValid", False)
        )

        timing = {
            "acquisitionId": status.get("acquisitionId"),
            "connectionId": self.connectionId,
            "segment": segment,
            "liveMappingId": segment.get("mappingId") if segment else None,
            "mappingId": segment.get("mappingId") if segment else None,
            "validSampleCount": validCount,
            "timelineValid": bool(
                originValidated and status.get("timelineValid", False)
            ),
            "status": status,
            "clockEvents": deepcopy(status.get("clockEvents", [])),
            "returnedCompleteSampleCount": completeCount,
            "truncatedSampleCount": 0,
            "incompleteTrailingValues": [
                float(value) if np.isfinite(value) else repr(float(value))
                for value in raw[completeCount * channelCount:]
            ],
            "snapshotTimestamp": snapshotTime,
            "snapshotIncomplete": bool(incomplete),
            "snapshotScope": "bufferedValuesAtSnapshotNotAStopConfirmation",
            "additionalSamplesMayArrive": bool(workerRunning or streamActive),
            "relativeSampleTimesAvailable": bool(validRate),
            "channelApertureOffsetsSeconds": None,
            "channelApertureUncertainty": "notEstablished",
        }
        return pglAnalogTraceData(
            time=relativeTime,
            data=data,
            channelNames=channelNames,
            scanRate=float(rate) if validRate else None,
            timing=timing,
        )


    def _validateAnalogStartStamp(
        self, record, preTicks, postTicks, validationTimeout=1.0,
    ):
        """
        Validate a stable start stamp against nearby advancing CORE_TIMER reads.

        Caller holds _digitalIOLock. This performs startup validation only,
        never periodic reads during acquisition or poll().

        A stamp slightly ahead of the current timer may describe an upcoming
        first scan. Do not declare readiness until the timer reaches it and a
        second stamp read agrees. This is conditional consistency evidence,
        not independent proof of first-scan semantics.
        """
        import time

        modulus = 1 << 32
        preRaw = record["coreBeforeStart"]["rawTicks"]
        originalRaw = record["streamStartStamp"]["rawTicks"]

        validation = {
            "method": "stableStampReachedByNearbyCoreTimer",
            "status": "checking",
            "timeoutSeconds": float(validationTimeout),
            "observations": [],
            "failureReason": None,
            "initialStampAheadOfTimer": None,
            "firstScanSemanticsIndependentlyVerified": False,
        }
        record["stampValidation"] = validation
        started = time.monotonic()

        try:
            if validationTimeout <= 0:
                raise ValueError("Startup stamp validation timeout must be positive.")
            if not 0 <= postTicks - preTicks < modulus / 2:
                raise RuntimeError(
                    "Analog startup timer interval is inconsistent or too long "
                    "for this nearby-stamp validation."
                )

            # Resolve near the pre-start timer, allowing a short future stamp.
            # A value behind the pre-start observation is not silently called
            # an almost-complete future wrap.
            delta = (
                (originalRaw - preRaw + modulus // 2) % modulus
                - modulus // 2
            )
            startTicks = int(preTicks + delta)
            if startTicks < preTicks:
                raise RuntimeError("Analog stream-start stamp precedes pre-start timer.")

            validation["initialStampAheadOfTimer"] = bool(startTicks > postTicks)
            validation["initialStampMinusPostTicks"] = int(startTicks - postTicks)
            validation["resolvedStartTicks"] = startTicks
            stableReadCount = 1

            while True:
                if self.stopEvent.is_set() or self._clockStopEvent.is_set():
                    raise RuntimeError("Analog start-stamp validation cancelled.")
                if time.monotonic() - started >= validationTimeout:
                    raise TimeoutError(
                        "Analog timer did not establish a stable reached stream-start "
                        "stamp within the startup validation limit."
                    )

                attempt = {}
                validation["observations"].append(attempt)

                stamp = self._readClockRegister(
                    "STREAM_START_TIME_STAMP", attempt, "streamStartStamp"
                )
                if stamp["rawTicks"] != originalRaw:
                    raise RuntimeError(
                        "Analog stream-start stamp changed during validation."
                    )
                stableReadCount += 1

                timer = self._readClockRegister(
                    "CORE_TIMER", attempt, "coreTimer"
                )
                currentTicks = self._clock.resolveRead(timer)
                if (
                    currentTicks < postTicks
                    or not 0 <= currentTicks - preTicks < modulus / 2
                ):
                    raise RuntimeError(
                        "Analog timer progression is inconsistent during startup."
                    )
                postTicks = currentTicks

                # Do not claim success if the validation operation itself
                # exceeded the bound. Native reads cannot be interrupted here.
                if time.monotonic() - started >= validationTimeout:
                    raise TimeoutError("Analog start-stamp validation exceeded its time limit.")

                if currentTicks >= startTicks:
                    validation.update({
                        "status": "validatedConditional",
                        "stableStampReadCount": stableReadCount,
                        "confirmingTimerTicks": int(currentTicks),
                        "elapsedSeconds": time.monotonic() - started,
                    })
                    record["streamStartStamp"]["unwrappedTicks"] = startTicks
                    return startTicks

                # Backoff, not an assumption that a fixed sleep establishes readiness.
                time.sleep(0.001)

        except BaseException as error:
            validation["status"] = "failed"
            validation["failureReason"] = f"{type(error).__name__}: {error}"
            validation["elapsedSeconds"] = time.monotonic() - started
            raise



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

class pglLabJackClock:
    """
    Connection-owned nominal-rate live mapping.

    Epoch prediction uses all startup transaction brackets under a conditional
    constant affine clock model. It is not proof against every device reset.

    The live mapping is fixed at scale=1.0. A separate feasible scale envelope
    is used for epoch placement; it never silently changes emitted timestamps.
    """

    nominalTicksPerSecond = 40_000_000
    timerModulus = 1 << 32

    def __init__(self, connectionId, deviceIdentity):
        from copy import deepcopy

        self.connectionId = connectionId
        self.deviceIdentity = deepcopy(deviceIdentity)
        self.events = []
        self.observations = []
        self.mapping = None
        self.valid = False
        self.lastResolvedTicks = None

    @staticmethod
    def _rawTicks(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Timer value must not be Boolean.")
        number = float(value)
        if (
            not np.isfinite(number)
            or number != np.floor(number)
            or not 0 <= number < (1 << 32)
        ):
            raise ValueError(f"Invalid uint32 timer value: {value!r}")
        return int(number)

    @staticmethod
    def _clipPolygon(polygon, coefficientA, coefficientB, limit):
        """Clip a convex (offset, scaleDeviation) polygon by a*x+b*y <= limit."""
        if not polygon: return []
        result = []
        previous = polygon[-1]
        previousValue = coefficientA * previous[0] + coefficientB * previous[1] - limit

        for current in polygon:
            currentValue = coefficientA * current[0] + coefficientB * current[1] - limit
            previousInside = previousValue <= 0
            currentInside = currentValue <= 0

            if previousInside != currentInside:
                fraction = previousValue / (previousValue - currentValue)
                result.append((
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                ))
            if currentInside: result.append(current)

            previous, previousValue = current, currentValue

        return result

    @classmethod
    def _fitBeginning(cls, observations):
        """
        Retain nominal offset bounds and the joint feasible affine region.

        Quantization expansion is conditional on near-nominal timer operation.
        No observation is excluded to manufacture feasibility.
        """
        if len(observations) < 3:
            raise ValueError("Calibration needs at least three valid observations.")

        rate = cls.nominalTicksPerSecond
        tickReference = observations[0]["unwrappedTicks"]
        hostBase = observations[0]["hostBefore"]
        tickAllowance = 1 / rate

        coordinates = []
        for row in observations:
            x = (row["unwrappedTicks"] - tickReference) / rate
            lower = (row["hostBefore"] - hostBase) - x - tickAllowance
            upper = (row["hostAfter"] - hostBase) - x + tickAllowance
            coordinates.append((x, lower, upper))

        if coordinates[-1][0] <= coordinates[0][0]:
            raise ValueError("Calibration has no positive device-time span.")

        nominalLower = max(row[1] for row in coordinates)
        nominalUpper = min(row[2] for row in coordinates)
        if nominalLower > nominalUpper:
            raise ValueError("Empty nominal-rate offset intersection; all observations retained.")

        # Eliminate offset to bound scaleDeviation using every pair.
        driftLower, driftUpper = -float("inf"), float("inf")
        for index, (x0, lower0, upper0) in enumerate(coordinates):
            for x1, lower1, upper1 in coordinates[index + 1:]:
                delta = x1 - x0
                if delta < 0:
                    raise ValueError("Device calibration coordinates moved backward.")
                if delta == 0:
                    if max(lower0, lower1) > min(upper0, upper1):
                        raise ValueError("Conflicting brackets at the same device coordinate.")
                    continue
                driftLower = max(driftLower, (lower1 - upper0) / delta)
                driftUpper = min(driftUpper, (upper1 - lower0) / delta)

        if (
            not np.isfinite(driftLower)
            or not np.isfinite(driftUpper)
            or driftLower > driftUpper
            or 1 + driftLower <= 0
        ):
            raise ValueError("Startup affine envelope is inadequately constrained.")

        # x=0 for the first observation, so its bracket directly bounds offset.
        firstLower, firstUpper = coordinates[0][1:]
        polygon = [
            (firstLower, driftLower), (firstUpper, driftLower),
            (firstUpper, driftUpper), (firstLower, driftUpper),
        ]
        for x, lower, upper in coordinates:
            polygon = cls._clipPolygon(polygon, 1.0, x, upper)
            polygon = cls._clipPolygon(polygon, -1.0, -x, -lower)
            if not polygon:
                raise ValueError("Empty joint affine clock region.")

        return {
            "nominalTicksPerSecond": rate,
            "tickReference": int(tickReference),
            "hostBase": float(hostBase),
            "hostReferenceOffset": float((nominalLower + nominalUpper) / 2),
            "hostReference": float(hostBase + (nominalLower + nominalUpper) / 2),
            "clockScale": 1.0,
            "referenceOffsetInterval": [float(nominalLower), float(nominalUpper)],
            "affineRegion": [[float(a), float(b)] for a, b in polygon],
            "selectedCount": len(observations),
            "selectedSpanSeconds": float(coordinates[-1][0]),
            "quantizationAllowanceSeconds": tickAllowance,
        }

    def collectBeginning(self, readTimer, getSecs, duration, cancelled):
        """Collect and retain startup evidence; readTimer performs one fresh read."""
        from copy import deepcopy
        from .pglEvent import (
            pglEventLabJackClockCalibration,
            pglEventLabJackClockMapping,
        )

        if self.mapping is not None:
            raise RuntimeError("The live mapping is already frozen.")
        if isinstance(duration, (bool, np.bool_)) or not np.isfinite(duration) or duration <= 0:
            raise ValueError("Calibration duration must be positive and finite.")

        started = getSecs()
        rows = []
        diagnostics = {
            "epochBootstrapMethod": "conditionalNearestNominalPrediction",
            "epochResolutionCertified": False,
            "failureReason": None,
        }
        status = "collected"
        errorToRaise = None

        try:
            while True:
                if cancelled():
                    raise RuntimeError("Clock calibration cancelled.")

                row = {
                    "hostBefore": getSecs(),
                    "rawTicks": None,
                    "unwrappedTicks": None,
                    "selected": False,
                    "selectionReason": "pendingValidation",
                }
                rows.append(row)
                try:
                    value = readTimer()
                    row["rawValueRepresentation"] = repr(value)
                finally:
                    row["hostAfter"] = getSecs()

                row["rawTicks"] = self._rawTicks(value)
                row["roundTripSeconds"] = row["hostAfter"] - row["hostBefore"]
                if row["roundTripSeconds"] < 0:
                    raise ValueError("Host clock moved backward.")
                row["hostMidpoint"] = row["hostBefore"] + row["roundTripSeconds"] / 2

                if len(rows) == 1:
                    resolved = row["rawTicks"]
                else:
                    first = rows[0]
                    elapsed = row["hostMidpoint"] - first["hostMidpoint"]
                    if not 0 <= elapsed < self.timerModulus / self.nominalTicksPerSecond / 2:
                        raise ValueError("Startup collection exceeds conditional bootstrap scope.")
                    expected = first["unwrappedTicks"] + elapsed * self.nominalTicksPerSecond
                    epoch = round((expected - row["rawTicks"]) / self.timerModulus)
                    resolved = row["rawTicks"] + epoch * self.timerModulus
                    if resolved < rows[-2]["unwrappedTicks"]:
                        raise ValueError("Startup timer moved backward after conditional resolution.")

                row.update({
                    "unwrappedTicks": int(resolved),
                    "selected": True,
                    "selectionReason": "allValidBracketsIncluded",
                })
                if row["hostAfter"] - started >= duration: break

            parameters = self._fitBeginning(rows)

        except BaseException as error:
            status = "cancelled" if cancelled() else "failed"
            diagnostics["failureReason"] = f"{type(error).__name__}: {error}"
            errorToRaise = error

        completed = getSecs()
        calibration = pglEventLabJackClockCalibration(
            connectionId=self.connectionId,
            deviceIdentity=self.deviceIdentity,
            phase="startup",
            collectionStartedTimestamp=started,
            collectionCompletedTimestamp=completed,
            requestedDuration=float(duration),
            observations=rows,
            status=status,
            diagnostics=diagnostics,
        )
        self.events.append(calibration)
        self.observations = deepcopy(rows)

        if errorToRaise is not None:
            self.valid = False
            raise errorToRaise

        mapping = pglEventLabJackClockMapping(
            connectionId=self.connectionId,
            calibrationEventIds=[calibration.eventId],
            createdTimestamp=getSecs(),
            purpose="live",
            status="validConditional",
            method="nominalOffsetWithAffineEpochEnvelope",
            methodVersion="1",
            parameters=parameters,
            uncertainty={
                "boundStatus": "conditional",
                "epochEnvelopeModel": "constantAffineClock",
                "nominalOffsetModel": "constantNominalRate",
                "hostTimestampUncertainty": None,
                "withinRunNonlinearDriftBound": None,
                "totalTimingBoundAvailable": False,
            },
            assumptions=[
                "freshRegisterReadWithinHostBracket",
                "nearNominalTimerForQuantizationAllowance",
                "deviceClockContinuous",
                "constantAffineClockForEpochPrediction",
            ],
        )
        self.events.append(mapping)
        self.mapping = mapping
        self.valid = True
        self.lastResolvedTicks = rows[-1]["unwrappedTicks"]

    def toHost(self, ticks):
        if not self.valid or self.mapping is None:
            raise RuntimeError("LabJack clock mapping is invalid or unavailable.")
        parameters = self.mapping.parameters
        return (
            parameters["hostBase"] + parameters["hostReferenceOffset"]
            + (ticks - parameters["tickReference"]) / self.nominalTicksPerSecond
        )

    def resolveRead(self, observation):
        """
        Resolve a fresh CORE_TIMER bracket against the frozen affine envelope.

        Reject multiple/no supported epochs. Conditional time consistency is
        not proof that no reset occurred.
        """
        if not self.valid or self.mapping is None:
            raise RuntimeError("LabJack clock mapping is invalid or unavailable.")

        parameters = self.mapping.parameters
        raw = observation["rawTicks"]
        before = observation["hostBefore"] - parameters["hostBase"]
        after = observation["hostAfter"] - parameters["hostBase"]
        allowance = parameters["quantizationAllowanceSeconds"]

        lowerCoordinates, upperCoordinates = [], []
        for offset, drift in parameters["affineRegion"]:
            scale = 1 + drift
            lowerCoordinates.append((before - allowance - offset) / scale)
            upperCoordinates.append((after + allowance - offset) / scale)

        lowerTicks = parameters["tickReference"] + min(lowerCoordinates) * self.nominalTicksPerSecond
        upperTicks = parameters["tickReference"] + max(upperCoordinates) * self.nominalTicksPerSecond

        firstEpoch = int(np.ceil((lowerTicks - raw) / self.timerModulus))
        lastEpoch = int(np.floor((upperTicks - raw) / self.timerModulus))
        observation["epochEvidence"] = {
            "candidateTickInterval": [float(lowerTicks), float(upperTicks)],
            "candidateEpochInterval": [firstEpoch, lastEpoch],
            "mappingId": self.mapping.eventId,
            "conditionalOnConstantAffineClock": True,
            "resetExcluded": False,
        }

        if firstEpoch != lastEpoch:
            self.valid = False
            raise RuntimeError("CORE_TIMER epoch is ambiguous or inconsistent with the clock envelope.")

        ticks = raw + firstEpoch * self.timerModulus
        if self.lastResolvedTicks is not None and ticks < self.lastResolvedTicks:
            self.valid = False
            raise RuntimeError("Resolved CORE_TIMER moved backward; clock mapping invalidated.")

        observation["unwrappedTicks"] = int(ticks)
        self.lastResolvedTicks = int(ticks)
        return int(ticks)



    def getTiming(self, deviceTicks, mappingId=None):
        """
        Return a mapped time and joint-region conditional error interval.

        deviceTicks may be fractional for sample-derived coordinates.
        An uncertainty interval is not a total physical-edge timing bound.
        """
        from copy import deepcopy

        if (
            isinstance(deviceTicks, (bool, np.bool_))
            or not isinstance(deviceTicks, (int, float, np.integer, np.floating))
            or not np.isfinite(deviceTicks)
        ):
            raise ValueError("deviceTicks must be a finite numeric coordinate.")

        if mappingId is None:
            if self.mapping is None:
                raise RuntimeError("No live mapping exists.")
            mappingId = self.mapping.eventId

        matches = [
            event for event in self.events
            if event.type == "labJackClockMapping" and event.eventId == mappingId
        ]
        if len(matches) != 1:
            raise ValueError("Mapping ID is missing or ambiguous.")
        mapping = matches[0]
        if mapping.status != "validConditional":
            raise ValueError("Rejected mappings cannot convert timestamps.")

        parameters = mapping.parameters
        rate = parameters["nominalTicksPerSecond"]
        x = (deviceTicks - parameters["tickReference"]) / rate

        # Center both the estimate and projection before adding hostBase.
        scale = parameters["clockScale"]
        estimateRelative = x + parameters["hostReferenceOffset"] + (scale - 1) * x
        projectedRelative = [
            x + offset + drift * x
            for offset, drift in parameters["affineRegion"]
        ]
        lower = min(projectedRelative)
        upper = max(projectedRelative)

        observed = parameters.get("observedTickInterval")
        if observed is None:
            observed = [
                self.observations[0]["unwrappedTicks"],
                self.observations[-1]["unwrappedTicks"],
            ]

        return {
            "timestamp": parameters["hostBase"] + estimateRelative,
            "clockMappingId": mapping.eventId,
            "calibrationEventIds": list(mapping.calibrationEventIds),
            "labJackTimestamp": deviceTicks / rate,
            "clockErrorLowerSeconds": lower - estimateRelative,
            "clockErrorUpperSeconds": upper - estimateRelative,
            "boundStatus": "conditional",
            "intervalMethod": "jointAffinePolygonProjection",
            "outsideObservationSpan": not observed[0] <= deviceTicks <= observed[1],
            "connectionContinuityCurrentlyValid": bool(self.valid),
            "assumptions": deepcopy(mapping.assumptions),
            "unknownContributions": [
                "hostTimestampUncertainty",
                "withinRunNonlinearClockVariation",
            ],
            "totalTimingBoundAvailable": False,
        }

    def collectCalibration(self, readTimer, getSecs, duration, cancelled, phase):
        """
        Collect one additional calibration without fitting or changing live mapping.

        Epoch placement uses the beginning affine envelope and remains conditional.
        All attempted reads and collection failures are retained.
        """
        from copy import deepcopy
        from .pglEvent import pglEventLabJackClockCalibration

        if phase not in {"beginning", "intermediate", "ending", "characterization"}:
            raise ValueError("Invalid additional calibration phase.")
        if isinstance(duration, (bool, np.bool_)):
            raise ValueError("Calibration duration must be positive and finite.")
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Calibration duration must be positive and finite.")
        if self.mapping is None:
            raise RuntimeError("A connection-startup mapping is required.")

        started = getSecs()
        rows = []
        failure = None
        status = "collected"
        caughtError = None

        try:
            if not self.valid:
                raise RuntimeError("Clock continuity is invalid.")
            while True:
                if cancelled():
                    raise RuntimeError("Clock calibration cancelled.")

                row = {
                    "hostBefore": getSecs(),
                    "rawTicks": None,
                    "unwrappedTicks": None,
                    "selected": False,
                    "selectionReason": "pendingValidation",
                    "failure": None,
                }
                rows.append(row)

                try:
                    try:
                        value = readTimer()
                        row["rawValueRepresentation"] = repr(value)
                    finally:
                        row["hostAfter"] = getSecs()

                    row["roundTripSeconds"] = row["hostAfter"] - row["hostBefore"]
                    if row["roundTripSeconds"] < 0:
                        raise ValueError("Host clock moved backward.")
                    row["hostMidpoint"] = (
                        row["hostBefore"] + row["roundTripSeconds"] / 2
                    )
                    row["rawTicks"] = self._rawTicks(value)
                    self.resolveRead(row)
                    row["selected"] = True
                    row["selectionReason"] = "validResolvedObservation"

                except BaseException as error:
                    row["failure"] = f"{type(error).__name__}: {error}"
                    row["selectionReason"] = "observationFailed"
                    self.valid = False
                    raise

                if row["hostAfter"] - started >= duration:
                    break

        except BaseException as error:
            caughtError = error
            failure = f"{type(error).__name__}: {error}"
            status = "cancelled" if cancelled() else "failed"

        event = pglEventLabJackClockCalibration(
            connectionId=self.connectionId,
            deviceIdentity=self.deviceIdentity,
            phase=phase,
            collectionStartedTimestamp=started,
            collectionCompletedTimestamp=getSecs(),
            requestedDuration=duration,
            observations=rows,
            status=status,
            diagnostics={
                "failureReason": failure,
                "epochResolutionCertified": False,
                "epochResolutionMappingId": self.mapping.eventId,
                "epochResolutionMethod": "beginningAffineEnvelope",
                "automaticFitApplied": False,
            },
        )
        self.events.append(event)

        if caughtError is not None:
            caughtError.labJackClockEvents = deepcopy(self.events)
            caughtError.labJackCalibrationEventId = event.eventId
            raise caughtError

        return deepcopy(event)

    @classmethod
    def _fitCalibrationCollections(cls, collections):
        """
        Fit all brackets from an arbitrary set of resolved collections.

        No point estimate is substituted for a collection. No conflicting
        observation is dropped. Uncertainty is conditional on one affine model.
        """
        if len(collections) < 2:
            raise ValueError("At least two calibration collections are required.")

        ordered = sorted(
            collections, key=lambda event: event.collectionStartedTimestamp
        )
        rows = []
        supports = []
        spans = []
        previousEnd = None

        for event in ordered:
            if event.status != "collected":
                raise ValueError(f"Collection {event.eventId} did not complete successfully.")
            if len(event.observations) < 3:
                raise ValueError(f"Collection {event.eventId} has fewer than three observations.")
            if previousEnd is not None and event.collectionStartedTimestamp < previousEnd:
                raise ValueError("Requested calibration collections overlap.")
            previousEnd = event.collectionCompletedTimestamp
            spans.append(event.actualDuration)

            for index, row in enumerate(event.observations):
                # Do not silently omit failed, unresolved, or excluded observations.
                if row.get("selected") is not True:
                    raise ValueError(
                        f"Collection {event.eventId}, observation {index} "
                        "is not eligible for this all-bracket fit."
                    )
                rows.append(row)
                supports.append({
                    "calibrationEventId": event.eventId,
                    "observationIndex": index,
                })

        firstMidpoint = (
            ordered[0].collectionStartedTimestamp
            + ordered[0].collectionCompletedTimestamp
        ) / 2
        lastMidpoint = (
            ordered[-1].collectionStartedTimestamp
            + ordered[-1].collectionCompletedTimestamp
        ) / 2
        separation = lastMidpoint - firstMidpoint

        if separation <= max(spans):
            raise ValueError("Collection separation does not exceed collection duration.")

        rate = cls.nominalTicksPerSecond
        tickReference = rows[0]["unwrappedTicks"]
        if isinstance(tickReference, (bool, np.bool_)) or not isinstance(
            tickReference, (int, np.integer)
        ):
            raise ValueError("Invalid reference ticks.")
        tickReference = int(tickReference)
        hostBase = float(rows[0]["hostBefore"])
        allowance = 1 / rate
        coordinates = []
        previousTicks = None

        for row in rows:
            ticks = row.get("unwrappedTicks")
            if isinstance(ticks, (bool, np.bool_)) or not isinstance(
                ticks, (int, np.integer)
            ):
                raise ValueError("Missing or invalid resolved ticks.")
            ticks = int(ticks)
            before = float(row["hostBefore"])
            after = float(row["hostAfter"])
            if not np.isfinite(before) or not np.isfinite(after) or after < before:
                raise ValueError("Invalid observation bracket.")
            if previousTicks is not None and ticks < previousTicks:
                raise ValueError("Resolved timer coordinates move backward.")
            previousTicks = ticks

            x = (ticks - tickReference) / rate
            coordinates.append((
                x,
                (before - hostBase) - x - allowance,
                (after - hostBase) - x + allowance,
            ))

        values = np.asarray(coordinates, dtype=float)
        xValues, lowerValues, upperValues = values.T
        driftLower, driftUpper = -float("inf"), float("inf")

        # Offset elimination gives finite scale bounds without inventing
        # an oscillator tolerance. Complexity is quadratic in observation count.
        for index in range(len(rows) - 1):
            delta = xValues[index + 1:] - xValues[index]
            same = delta == 0
            if np.any(
                np.maximum(lowerValues[index], lowerValues[index + 1:][same])
                > np.minimum(upperValues[index], upperValues[index + 1:][same])
            ):
                raise ValueError("Conflicting brackets at the same device coordinate.")

            positive = delta > 0
            if np.any(positive):
                driftLower = max(
                    driftLower,
                    float(np.max(
                        (lowerValues[index + 1:][positive] - upperValues[index])
                        / delta[positive]
                    )),
                )
                driftUpper = min(
                    driftUpper,
                    float(np.min(
                        (upperValues[index + 1:][positive] - lowerValues[index])
                        / delta[positive]
                    )),
                )

        if (
            not np.isfinite(driftLower)
            or not np.isfinite(driftUpper)
            or driftLower > driftUpper
            or 1 + driftLower <= 0
        ):
            raise ValueError("No bounded positive affine clock scale is supported.")

        polygon = [
            (float(lowerValues[0]), driftLower),
            (float(upperValues[0]), driftLower),
            (float(upperValues[0]), driftUpper),
            (float(lowerValues[0]), driftUpper),
        ]
        for x, lower, upper in coordinates:
            polygon = cls._clipPolygon(polygon, 1.0, x, upper)
            polygon = cls._clipPolygon(polygon, -1.0, -x, -lower)
            if not polygon:
                raise ValueError("All-collection affine constraints are inconsistent.")

        offset = float(np.mean([point[0] for point in polygon]))
        drift = float(np.mean([point[1] for point in polygon]))
        residuals = []
        for row, x in zip(rows, xValues):
            midpoint = (
                row["hostBefore"] - hostBase
                + (row["hostAfter"] - row["hostBefore"]) / 2
            )
            residuals.append(float(x + offset + drift * x - midpoint))

        return {
            "nominalTicksPerSecond": rate,
            "tickReference": tickReference,
            "hostBase": hostBase,
            "hostReferenceOffset": offset,
            "hostReference": hostBase + offset,
            "clockScale": 1 + drift,
            "secondsPerTick": (1 + drift) / rate,
            "affineRegion": [[float(a), float(b)] for a, b in polygon],
            "feasibleClockScale": [
                1 + min(point[1] for point in polygon),
                1 + max(point[1] for point in polygon),
            ],
            "selectionMethod": "meanOfFeasiblePolygonVertices",
            "collectionCount": len(ordered),
            "observationCount": len(rows),
            "observationSupport": supports,
            "collectionSeparationSeconds": float(separation),
            "observedTickInterval": [
                int(rows[0]["unwrappedTicks"]), int(rows[-1]["unwrappedTicks"]),
            ],
            "quantizationAllowanceSeconds": allowance,
            "midpointResidualSeconds": residuals,
            "precisionAcceptanceThreshold": None,
            "exclusions": [],
        }

    def fitCalibrations(self, calibrationEventIds, getSecs, purpose="postRun"):
        """
        Fit explicitly selected collections only when continuity remains usable.

        Conservative policy:
        - An invalidated connection clock cannot authorize another mapping.
        - Failed collections intersecting the selected span block fitting.
        - Connection identity alone is not proof of uninterrupted device time.

        Previously emitted mappings and observations remain unchanged.
        """
        from copy import deepcopy
        from .pglEvent import pglEventLabJackClockMapping

        if purpose not in {"postRun", "characterization"}:
            raise ValueError("This fit cannot replace the live mapping.")
        if (
            not isinstance(calibrationEventIds, (list, tuple))
            or not calibrationEventIds
            or any(
                not isinstance(value, str) or not value
                for value in calibrationEventIds
            )
            or len(set(calibrationEventIds)) != len(calibrationEventIds)
        ):
            raise ValueError("Supply distinct nonempty calibration-event IDs.")

        parameters = None
        caughtError = None
        failure = None
        continuityEvidence = {
            "policy": "rejectInvalidClockOrKnownFailedCollectionAcrossFitSpan",
            "connectionClockValidAtFit": bool(self.valid),
            "selectedCollectionSpan": None,
            "blockingCalibrationEventIds": [],
            "undetectedResetExcluded": False,
            "epochResolutionCertified": False,
        }

        try:
            if not self.valid:
                raise ValueError(
                    "Connection clock continuity is invalid; a new mapping "
                    "cannot be authorized from this clock object."
                )

            collections = []
            for eventId in calibrationEventIds:
                matches = [
                    event for event in self.events
                    if event.type == "labJackClockCalibration"
                    and event.eventId == eventId
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Calibration ID missing or ambiguous: {eventId}"
                    )

                event = matches[0]
                if event.connectionId != self.connectionId:
                    raise ValueError(
                        "Cannot fit calibrations from different connections."
                    )
                if event.status != "collected":
                    raise ValueError(
                        f"Calibration did not complete successfully: {eventId}"
                    )

                before = event.collectionStartedTimestamp
                after = event.collectionCompletedTimestamp
                if (
                    not np.isfinite(before)
                    or not np.isfinite(after)
                    or after < before
                ):
                    raise ValueError(
                        f"Invalid calibration collection interval: {eventId}"
                    )
                collections.append(event)

            spanStart = min(
                event.collectionStartedTimestamp for event in collections
            )
            spanEnd = max(
                event.collectionCompletedTimestamp for event in collections
            )
            continuityEvidence["selectedCollectionSpan"] = [
                float(spanStart), float(spanEnd),
            ]

            # An unselected failed collection cannot disappear merely because
            # the caller chooses the successful collections on either side.
            for event in self.events:
                if (
                    event.type != "labJackClockCalibration"
                    or event.connectionId != self.connectionId
                    or event.status != "failed"
                ):
                    continue

                before = event.collectionStartedTimestamp
                after = event.collectionCompletedTimestamp
                if (
                    not np.isfinite(before)
                    or not np.isfinite(after)
                    or after < before
                ):
                    raise ValueError(
                        "A failed calibration has invalid temporal provenance; "
                        "its position relative to the selected span is unknown."
                    )

                if before <= spanEnd and after >= spanStart:
                    continuityEvidence["blockingCalibrationEventIds"].append(
                        event.eventId
                    )

            if continuityEvidence["blockingCalibrationEventIds"]:
                raise ValueError(
                    "A failed clock collection intersects the requested fit span. "
                    "Cross-failure fitting is not supported."
                )

            parameters = self._fitCalibrationCollections(collections)

        except Exception as error:
            caughtError = error
            failure = f"{type(error).__name__}: {error}"

        mapping = pglEventLabJackClockMapping(
            connectionId=self.connectionId,
            calibrationEventIds=list(calibrationEventIds),
            createdTimestamp=getSecs(),
            purpose=purpose,
            status="validConditional" if parameters is not None else "rejected",
            method="allCollectionBracketAffine",
            methodVersion="2",
            parameters=parameters,
            uncertainty=(
                {
                    "boundStatus": "conditional",
                    "intervalMethod": "jointAffinePolygonProjection",
                    "hostTimestampUncertainty": None,
                    "withinRunNonlinearDriftBound": None,
                    "totalTimingBoundAvailable": False,
                }
                if parameters is not None else None
            ),
            assumptions=[
                "freshRegisterReadWithinHostBracket",
                "nearNominalTimerForQuantizationAllowance",
                "deviceClockContinuousAcrossSelectedCollections",
                "constantAffineClockAcrossSelectedCollections",
                "conditionalEpochPlacement",
            ],
            diagnostics={
                "failureReason": failure,
                "continuityEvidence": continuityEvidence,
                "liveMappingChanged": False,
                "automaticEventCorrectionApplied": False,
            },
        )
        self.events.append(mapping)

        if caughtError is not None:
            caughtError.labJackClockEvents = deepcopy(self.events)
            caughtError.labJackRejectedMappingId = mapping.eventId
            raise caughtError

        return deepcopy(mapping)

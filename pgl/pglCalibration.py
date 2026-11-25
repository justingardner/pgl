################################################################
#   filename: pglCalibration.py
#    purpose: Handles calibration of display
#         by: JLG
#       date: September 18, 2025
################################################################

#############
# Import modules
#############
import numpy as np
import matplotlib.pyplot as plt
from ._pglComm import pglSerial
from .pglSettings import pglSettings, pglDisplaySettings, pglSettingsManager, pglTraitSettings
from traitlets import Unicode, Int, Instance, Dict, Tuple, Float, List
from datetime import datetime
from .pglExperiment import pglExperiment
from tqdm.notebook import tqdm
from traitlets import HasTraits
from .pglSerialize import pglSerialize
from scipy.interpolate import interp1d
from .pglDevice import pglDigitalIODevice, pglAnalogInputDevice, pglAnalogTraceData
from scipy.io import loadmat
from .pglMessages import pglMessages
from scipy.interpolate import PchipInterpolator
from .pglSettings import pglSettingsManager
from datetime import datetime, date as Date
from .pglStorage import pglStorage

##########################
# Calibration device class
##########################
class pglLuminanceCalibrationDevice():
    '''
    Class representing a calibration device, this 
    should be subclassed for specific devices
    '''
    def __init__(self, description="", verbose=True):
        '''
        Initialize the calibration device
        '''
        self.description = description
        self.verbose = verbose

    def __del__(self):
        '''
        Destructor for the luminance calibration device.
        '''
        pass

    def measure(self):
        '''
        Should return a luminance measurement from the device
        '''
        pass
    
    def units(self):
        '''
        units that the device measures in
        '''
        return 'cd/m2'
        

# for testing
class pglLuminanceCalibrationDeviceDebug(pglLuminanceCalibrationDevice):
    '''
    Class representing a debug calibration device
    '''
    def measure(self):
        '''
        Measure the display characteristics using debug device.
        '''
        self.currentMeasurement = np.random.rand()
        pglMessages.message(f"Measurement: {self.currentMeasurement}")
        return self.currentMeasurement
    
####################################################
# Minolta CS-100A calibration device class
####################################################
class pglLuminanceCalibrationDeviceMinolta(pglLuminanceCalibrationDevice):
    '''
    Class representing a Minolta CS-100A luminance / chrominance meter
    '''
    def __init__(self, description="Minolta CS-100A", verbose=True):
        '''
        Initialize the Minolta calibration.
        '''
        super().__init__(description, verbose)
        
        # tell user to connect device and turn on
        pglMessages.printHeader("Connect Minolta CS-100A")
        pglMessages.print("(pglCalibrationDeviceMinolta) Connect the Minolta CS-100A to the serial port")
        pglMessages.print("(pglCalibrationDeviceMinolta) hold down F key and turn it on.")
        pglMessages.print("(pglCalibrationDeviceMinolta) You should see the letter C on the Minolta display.")
        pglMessages.print("(pglCalibrationDeviceMinolta) Press Enter to continue...")
        input()
        
        # init serial port
        pglMessages.printHeader("Choose Serial Port")
        pglMessages.print("(pglCalibrationDeviceMinolta) Select the serial port that the Minolta CS-100A is connected to.")        
        pglMessages.print("(pglCalibrationDeviceMinolta) This should appear as something like:")
        pglMessages.print("(pglCalibrationDeviceMinolta)   /dev/cu.usbserial-110 - USB-Serial Controller Device")
        self.serial = pglSerial(dataLen=7, parity='e', stopBits=2, baudrate=4800, timeout=5.0)

        if self.serial.isOpen() is False:
            pglMessages.printHeader("Cancelled")
        else:
            pglMessages.printHeader("Minolta CS-100A Connected")
    def __del__(self):
        '''
        Destructor for the Minolta calibration device.
        '''
        self.serial.close()
    
    def __repr__(self):
        return f"<pglCalibrationDeviceMinolta: {self.description}>"

    def measure(self):
        '''
        Measure the display characteristics using Minolta device.
        '''
        # send measurement command
        self.serial.write("MES\r\n")
        
        # read response
        r = self.serial.read()
        
        errorCodes = {
            '00': 'Offending command',
            '01': 'Setting error',
            '11': 'Memory value error',
            '10': 'Measuring range over',
            '19': 'Display range over',
            '20': 'EEPROM error',
            '30': 'Battery exhausted'
        }
    
        try:
            # Convert bytes to string and strip whitespace/newlines
            s = r.decode('ascii').strip()
        
            # Check for error response
            if s.startswith('ER'):
                errorCode = s[2:4]
                errorMessage = errorCodes.get(errorCode, 'Unknown error')
                pglMessages.warning(f"Error from device: {errorMessage} (Code: {errorCode})")
                return None
        
            # Split at first comma to separate 'OK13' from the rest
            firstComma = s.index(',')
            statusAndMode = s[:firstComma]   # e.g., 'OK13'
            rest = s[firstComma+1:]            # e.g., '  10.73, .4795, .4287'
        
            # Extract status ('OK') and mode (numeric)
            status = statusAndMode[:2]
            mode = int(statusAndMode[2:])
        
            # Split the rest of the values and convert to float
            values = [float(v.strip()) for v in rest.split(',')]
            luminance, x, y = values

            if self.verbose:
                pglMessages.print(f"(pglCalibrationDeviceMinolta) Luminance={luminance}, x={x}, y={y} (mode={mode})")
            return luminance
        
        except Exception as e:
            pglMessages.warning(f"Error parsing response ({r}): {e}")
            return None
    

##########################
# Calibration class
##########################
class pglDisplayCalibration():
    '''
    Runs calibration for a display
    '''
    def __init__(self, pgl, luminanceCalibrationDevice : pglLuminanceCalibrationDevice = None, digitalIODevice : pglDigitalIODevice = None, analogInputDevice: pglAnalogInputDevice = None):
        '''
        Initialize the calibration.
        '''
        # set the devices
        if not self.addLuminanceCalibrationDevice(luminanceCalibrationDevice): return
        if not self.addDigitalIODevice(digitalIODevice): return
        if not self.addAnalogInputDevice(analogInputDevice): return

        # keep pgl reference
        self.pgl = pgl

        # make a dict for where to save calibration data
        self.calibrationModeSaveLocation = {
            "minsearch": "init",
            "minmax": "init",
            "full": "calibration",
            "done": "None",
        }
        
        # value for how close to min and max a value can be
        # to just be considered min or max when doing the
        # search for min and max values
        self.epsilon = 0.001
    
    def addLuminanceCalibrationDevice(self, luminanceCalibrationDevice : pglLuminanceCalibrationDevice = None):
        ''''
        add a luminanceCalibrationDevice to run luminanceCalibrations
        
        Args:
            luminanceCalibrationDevice: pglLuminanceCalibrationDevice for measuring luminance
        '''
        self.luminanceCalibrationDevice = None
        if luminanceCalibrationDevice is not None:
            # check luminance calibration device
            if not isinstance(luminanceCalibrationDevice, pglLuminanceCalibrationDevice):
                pglMessages.warning("luminanceCalibrationDevice must be of type pglCalibrationDevice.", level=2)
                return False
            self.luminanceCalibrationDevice = luminanceCalibrationDevice        
            self.luminanceCalibrationDevice.verbose = False
        return True

    def addDigitalIODevice(self, digitalIODevice : pglDigitalIODevice = None):
        ''''
        add a digitalIODevice
        
        Args:
            digitalIODevice: pglDigitalIODevice for digital IO
        '''
        self.digitalIODevice = None
        
        # check digitalIO device
        if digitalIODevice is not None:
            if not isinstance(digitalIODevice, pglDigitalIODevice):
                pglMessages.warning("digitalIODevice must be of type pglDigitalIODevice.", level=2)
                return False
            self.digitalIODevice = digitalIODevice
        return True

    def addAnalogInputDevice(self, analogInputDevice : pglAnalogInputDevice = None):
        ''''
        add a analogInputDevice
        
        Args:
            analogInputDevice: pglAnalogInputDevice for analog input
        '''
        self.analogInputDevice= None
        
        # check analogInputDevice device
        if analogInputDevice is not None:
            if not isinstance(analogInputDevice, pglAnalogInputDevice):
                pglMessages.warning("analogInputDevice must be of type analogInputDevice.", level=2)
                return False
            self.analogInputDevice = analogInputDevice

        return True

    @property
    def calibrationMode(self):
        '''
        Get the current calibration mode.
        '''
        return self._calibrationMode
    @calibrationMode.setter
    def calibrationMode(self, mode):
        '''
        Set the current calibration mode.
        '''
        validModes = ["minmax", "minsearch", "full", "done"]
        if mode not in validModes:
            pglMessages.warning(f"Invalid calibration mode: {mode}. Valid modes are: {validModes}")
            return
        self._calibrationMode = mode
        # get save location for current mode
        self.saveLocation = self.calibrationModeSaveLocation.get(mode, "None")
    def __del__(self):
        '''
        Destructor for the calibration class.
        '''
        pass

    def calibrateTemporal(
        self, settingsName=None, settings=None, displayName=None,
        displaySettings=None, waitToStart=None, *,
        dataChannel=0, minimumAmplitude=0.01, useBatch=True,
        numRepeats=30, stimulusDurationFrames=(2, 3, 8, 0),
    ):
        """
        Record and analyze photodiode onset relative to drawablePresented.

        dataChannel is the photodiode column in the selected analog scan order.
        minimumAmplitude is an explicit detector acceptance threshold in volts.
        Digital output is not required to establish the analog clock origin.

        Display/experiment resources are owned by this operation. Supplied
        analog/digital devices remain caller-owned and are never closed here.
        """
        from copy import deepcopy
        from numbers import Integral, Real
        from pathlib import Path

        if self.analogInputDevice is None:
            raise RuntimeError("An analog input device is required.")
        required = (
            "collectClockCalibration", "fitClockCalibrations",
            "getClockEvents", "getAnalogReadStatus",
        )
        if any(not callable(getattr(self.analogInputDevice, name, None)) for name in required):
            raise RuntimeError(
                "This presentation-relative path requires a clock-calibrated "
                "analog driver. Legacy sync-relative analysis remains separate."
            )
        if self.pgl.isOpen():
            raise RuntimeError(
                "Close the existing display before temporal calibration. "
                "Caller-owned display resources will not be replaced."
            )
        if type(useBatch) is not bool:
            raise TypeError("useBatch must be Boolean.")
        if (
            isinstance(numRepeats, (bool, np.bool_))
            or not isinstance(numRepeats, Integral) or numRepeats < 1
        ):
            raise ValueError("numRepeats must be a positive integer.")
        if (
            isinstance(dataChannel, (bool, np.bool_))
            or not isinstance(dataChannel, Integral) or dataChannel < 0
        ):
            raise ValueError("dataChannel must be a nonnegative column index.")
        if (
            isinstance(minimumAmplitude, (bool, np.bool_))
            or not isinstance(minimumAmplitude, Real)
            or not np.isfinite(minimumAmplitude) or minimumAmplitude <= 0
        ):
            raise ValueError("minimumAmplitude must be positive and finite.")
        durations = list(stimulusDurationFrames)
        if (
            len(durations) != 4
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral) or value < 0
                for value in durations
            )
            or durations[1] < 1 or durations[2] + durations[3] < 1
        ):
            raise ValueError("Frame durations need a white phase and a following black phase.")
        if waitToStart is not None and (
            isinstance(waitToStart, (bool, np.bool_))
            or not isinstance(waitToStart, Real)
            or not np.isfinite(waitToStart) or waitToStart < 0
        ):
            raise ValueError("waitToStart must be nonnegative and finite.")

        # Resolve existing settings without invoking getSettings()'s default
        # creation/default-repair save behavior.
        if settings is not None:
            if not isinstance(settings, pglSettings):
                raise TypeError("settings must be pglSettings.")
            localSettings = deepcopy(settings)
        elif settingsName is not None:
            selected = pglSettingsManager.getSettings(settingsName=settingsName)
            if selected is None:
                raise ValueError("Named experiment settings were not found.")
            localSettings = deepcopy(selected)
        else:
            candidates = [
                pglSettings.load(filename=path, refresh=False)
                for path in sorted(
                    pglSettingsManager.getSettingsDir().glob("*.json")
                )
            ]
            if any(candidate is None for candidate in candidates):
                raise ValueError("A saved settings file could not be restored.")
            defaults = [candidate for candidate in candidates if candidate.isDefault]
            if len(defaults) != 1:
                raise ValueError(
                    "Temporal calibration requires one saved default settings profile "
                    "or explicitly supplied settings. No profile was created or repaired."
                )
            localSettings = deepcopy(defaults[0])
            localSettings.reloadDisplays()
            localSettings.reloadLabJackSettings()

        if displayName is not None or displaySettings is not None:
            localSettings = pglSettingsManager.getSettings(
                settings=localSettings,
                displayName=displayName,
                displaySettings=displaySettings,
            )
        if localSettings is None:
            raise ValueError("Display settings could not be resolved.")

        preset = localSettings.getLabJackPreset()
        analogConfig = preset.getAnalogReadConfig()
        if dataChannel >= len(analogConfig["channels"]):
            raise ValueError("Photodiode dataChannel exceeds the selected analog scan order.")

        # Explicit calibration-only settings copy: do not initialize a second
        # connection, a digital stream, other experiment inputs, or an eye tracker.
        for item in localSettings._digitalIO:
            item.isSelected = False
        for item in localSettings._devices:
            item.isSelected = False
        localSettings.eyetracker = ["None"]
        localSettings.closeScreenOnEnd = True
        localSettings.backgroundColor = [0.0, 0.0, 0.0]

        data = pglDisplayTemporalCalibrationData()
        self.temporalCalibrationData = data
        data.settingsName = localSettings.name
        data.settings = localSettings
        data.creationDateTime = datetime.now()
        data.analogInputDescription = self.analogInputDevice.deviceDescription
        data.digitalIOdeviceDescription = "Not required for calibrated-clock analysis"
        data.numRepeats = int(numRepeats)
        data.stimulusDurationFrames = [int(value) for value in durations]
        data.frameTimestampProvenance = {
            "field": "drawablePresented",
            "method": "batchCommandResults" if useBatch else "flushCommandResults",
            "clockDomain": "assumedPGLHostClock",
            "absoluteUncertaintySeconds": None,
            "successRequired": 1,
        }
        data.acquisitionMetadata = {
            "analogRequested": deepcopy(analogConfig),
            "presetName": preset.name,
            "presetUUID": preset.uuid,
            "useBatch": useBatch,
            "digitalSyncUsedForTimeOrigin": False,
            "measurementPosition": "notRecorded",
            "detectorDescription": "notRecorded",
        }

        print(f"Temporal calibration analog preset: {preset.name!r}")
        print(f"Analog configuration: {analogConfig}")
        print(
            f"Photodiode column: {dataChannel} "
            f"({analogConfig['channels'][dataChannel]}); "
            f"minimum excursion: {minimumAmplitude:g} V; "
            f"presentation mode: {'batch' if useBatch else 'non-batch'}."
        )

        experiment = None
        failure = None
        try:
            experiment = pglExperiment(self.pgl, settings=localSettings)
            if not experiment.isInitialized:
                raise RuntimeError("Calibration display experiment did not initialize.")
            experiment.initScreen(startRecording=False)
            if not self.pgl.isOpen():
                raise RuntimeError("Calibration display did not open.")
            data.displaySettings = experiment.settings.displays[0]

            self.pgl.fullScreen(True)
            self.pgl.waitSecs(0.75)
            self.pgl.clearScreen(0)
            self.pgl.flush()
            if waitToStart:
                self.pgl.waitSecs(waitToStart)

            self._calibrateTemporal(
                analogConfig=analogConfig,
                useBatch=useBatch,
                dataChannel=int(dataChannel),
                minimumAmplitude=float(minimumAmplitude),
            )

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            failure = error
            pglMessages.warning(
                f"Temporal calibration did not complete: {type(error).__name__}: "
                f"{error}. Retained data remain in temporalCalibrationData.",
                level=2,
            )
        finally:
            if experiment is not None:
                cleanupErrors = []
                # These experiment-owned device lists were deliberately disabled.
                for label, action in (
                    ("stop experiment inputs", experiment.endDevices),
                    ("close experiment digital IO", experiment.endDigitalIO),
                    (
                        "close calibration display",
                        lambda: experiment.endScreen(
                            clearScreen=False,
                            resetStencil=failure is None,
                        ),
                    ),
                ):
                    try:
                        action()
                    except (Exception, KeyboardInterrupt, SystemExit) as error:
                        cleanupErrors.append(f"{label}: {type(error).__name__}: {error}")
                data.temporalCleanupErrors = cleanupErrors
                if cleanupErrors:
                    pglMessages.warning(
                        "Temporal calibration cleanup reported errors: "
                        + "; ".join(cleanupErrors),
                        level=2,
                    )
                # Persist cleanup outcome if raw publication already succeeded.
                if data.calibrationDataReference:
                    try:
                        data.save()
                    except Exception as error:
                        data.pendingSaveFailure = f"{type(error).__name__}: {error}"
                        pglMessages.warning(
                            "Temporal cleanup metadata could not be saved. "
                            "The in-memory calibration remains available for retry.",
                            level=2,
                        )

        if failure is not None:
            raise failure
        return data
    
    def _calibrateTemporal(
        self, *, analogConfig, useBatch, dataChannel, minimumAmplitude,
    ):
        """
        One stop-controlled analog stream through the entire frame sequence.

        Retain all attempted frame rows, including missing/failed presentations.
        Save raw recording before analysis; never close the supplied device.
        """
        from copy import deepcopy
        import time

        data = self.temporalCalibrationData
        device = self.analogInputDevice
        framesPerRepeat = sum(data.stimulusDurationFrames)
        cumulativeFrames = np.cumsum(data.stimulusDurationFrames)
        frameRows = []
        errors = []
        started = False
        beginning = None
        mapping = None
        trace = None

        # Metadata uses finite values/None; raw nonfinite frame values go in HDF5.
        data.acquisitionMetadata["errors"] = errors
        data.acquisitionMetadata["status"] = "starting"
        data.analysisStatus = "notAnalyzed"

        def appendError(stage, error):
            errors.append({
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            })

        def scalar(value):
            if value is None:
                return float("nan")
            array = np.asarray(value)
            if array.ndim != 0 or array.dtype.kind not in "iuf":
                raise ValueError("Frame result must be a numeric scalar or None.")
            return float(array)

        def waitForSamples(target):
            # Explicit progress/readiness checks. This sleep is only polling
            # backoff, not an assumption that the hardware has started.
            deadline = time.monotonic() + 10.0
            while True:
                status = device.getAnalogReadStatus()
                if status.get("error") or status.get("cleanupErrors"):
                    raise RuntimeError(f"Analog acquisition failed: {status.get('error')}")
                if status.get("validSampleCount", 0) >= target:
                    return
                if not status.get("running") or not status.get("streamActive"):
                    raise RuntimeError("Analog acquisition stopped before required samples.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Analog acquisition did not provide required sample support.")
                self.pgl.waitSecs(0.005)

        try:
            data.frameRate = float(self.pgl.getFrameRate())
            if not np.isfinite(data.frameRate) or data.frameRate <= 0:
                raise ValueError("Invalid display frame rate.")

            # A fresh acquisition-specific beginning collection follows display
            # initialization; it does not refit the frozen connection mapping.
            beginning = device.collectClockCalibration(phase="beginning")
            data.acquisitionMetadata["beginningCalibrationId"] = beginning.eventId

            started = True
            actualRate = device.startAnalogRead(duration=None, **analogConfig)
            data.analogScanRate = float(actualRate)
            data.acquisitionMetadata["actualScanRate"] = float(actualRate)
            status = device.getAnalogReadStatus()
            if not status.get("ready"):
                raise RuntimeError("Analog startup returned without readiness.")
            waitForSamples(int(np.ceil(0.05 * actualRate)))

            width = self.pgl.screenWidth.deg
            height = self.pgl.screenHeight.deg

            for repeatIndex in range(data.numRepeats):
                status = device.getAnalogReadStatus()
                if status.get("error") or not status.get("running"):
                    raise RuntimeError("Analog acquisition is not active for the next repeat.")

                repeatRows = []
                for frameIndex in range(framesPerRepeat):
                    phase = int(np.searchsorted(cumulativeFrames, frameIndex, side="right"))
                    row = {
                        "repeatIndex": repeatIndex,
                        "frameIndex": frameIndex,
                        "stimulusPhase": phase,
                        "drawablePresented": float("nan"),
                        "processedTime": float("nan"),
                        "success": float("nan"),
                        "drawablePresentedPresent": False,
                    }
                    repeatRows.append(row)

                if useBatch:
                    # Retain intended rows before submission can fail.
                    frameRows.extend(repeatRows)
                    self.pgl.batchStart()
                    for row in repeatRows:
                        color = 1 if row["stimulusPhase"] == 1 else 0
                        self.pgl.rect(0, 0, width, height, color)
                        self.pgl.flush()
                    self.pgl.batchRun()
                    self.pgl.batchEnd()  # Return value may be processedTime; do not use it.
                    results = self.pgl.profileInfo["commandResults"]

                    for field in ("drawablePresented", "processedTime", "success"):
                        values = results.get(field)
                        if values is None:
                            continue
                        values = np.atleast_1d(values)
                        if len(values) != framesPerRepeat:
                            raise ValueError(f"Batch {field} count differs from intended frames.")
                        for row, value in zip(repeatRows, values):
                            row[field] = scalar(value)
                            if field == "drawablePresented":
                                row["drawablePresentedPresent"] = True

                else:
                    for row in repeatRows:
                        frameRows.append(row)
                        color = 1 if row["stimulusPhase"] == 1 else 0
                        self.pgl.rect(0, 0, width, height, color)
                        returnedPresentation = self.pgl.flush()
                        results = self.pgl.commandResults or {}
                        for field in ("drawablePresented", "processedTime", "success"):
                            row[field] = scalar(results.get(field))
                        row["drawablePresentedPresent"] = (
                            results.get("drawablePresented") is not None
                        )
                        # Preserve the public flush return independently.
                        row["flushReturn"] = scalar(returnedPresentation)

            # Retain trailing black samples from the same uninterrupted stream.
            beforeTail = device.getAnalogReadStatus()["samplesReturned"]
            waitForSamples(beforeTail + int(np.ceil(0.05 * actualRate)))
            data.acquisitionMetadata["status"] = "framesCompleted"

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            appendError("acquisition", error)

        finally:
            if started:
                try:
                    trace = device.stopAnalogRead(
                        waitToFinish=False, doNotTruncate=True
                    )
                except (Exception, KeyboardInterrupt, SystemExit) as error:
                    appendError("stopAnalogRead", error)
                    # Do not discard buffered samples because shutdown failed.
                    try:
                        trace = device.getAnalogReadSnapshot()
                        trace.timing["snapshotIncomplete"] = True
                        trace.timing["snapshotReason"] = "stopAnalogReadFailed"
                        data.acquisitionMetadata["partialSnapshotRetained"] = True
                    except (Exception, KeyboardInterrupt, SystemExit) as snapshotError:
                        appendError("getAnalogReadSnapshot", snapshotError)

            if trace is not None:
                data.analogTraceData = trace
                if getattr(trace, "scanRate", None) is not None:
                    data.analogScanRate = float(trace.scanRate)
                rawStatus = trace.timing.get("status", {}) if trace.timing else {}
                if rawStatus.get("error") or rawStatus.get("cleanupErrors"):
                    errors.append({
                        "stage": "analogStatus",
                        "type": "AcquisitionIncomplete",
                        "message": str(rawStatus.get("error") or rawStatus.get("cleanupErrors")),
                    })

            # No endpoint reads while a worker or unresolved stream owns hardware.
            try:
                status = device.getAnalogReadStatus()
                if (
                    beginning is not None and trace is not None
                    and not (getattr(trace, "timing", None) or {}).get("snapshotIncomplete", False)
                ):
                    if status.get("running") or status.get("streamActive"):
                        raise RuntimeError("Cannot calibrate ending while analog stream is active.")
                    ending = device.collectClockCalibration(phase="ending")
                    mapping = device.fitClockCalibrations(
                        [beginning.eventId, ending.eventId], purpose="postRun"
                    )
                    data.acquisitionMetadata["endingCalibrationId"] = ending.eventId
                    data.acquisitionMetadata["postRunMappingId"] = mapping.eventId
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                appendError("endingClockCalibration", error)

            data.clockCalibrationEvents = device.getClockEvents()

        fieldNames = (
            "repeatIndex", "frameIndex", "stimulusPhase",
            "drawablePresented", "processedTime", "success",
            "drawablePresentedPresent",
        )
        data.frameRecords = {}
        for field in fieldNames:
            dtype = (
                np.int64 if field in ("repeatIndex", "frameIndex", "stimulusPhase")
                else np.bool_ if field == "drawablePresentedPresent"
                else np.float64
            )
            data.frameRecords[field] = np.asarray(
                [row[field] for row in frameRows], dtype=dtype
            )
        if not useBatch:
            data.frameRecords["flushReturn"] = np.asarray(
                [row.get("flushReturn", float("nan")) for row in frameRows],
                dtype=np.float64,
            )
        # New recordings have one canonical frame dataset, not a second alias.
        data.flushTimes = None

        if data.analogTraceData is None and started:
            # Without a snapshot, an empty trace would falsely represent
            # a possibly nonempty driver buffer. Keep ownership/data in memory.
            raise RuntimeError(
                "Analog shutdown/snapshot failed. No empty recording was "
                "substituted. Recover buffered data from the supplied device "
                "before closing or starting another acquisition."
            )

        if data.analogTraceData is None:
            # Save a genuine empty/failed acquisition, not fabricated samples.
            empty = pglAnalogTraceData(
                time=np.empty(0),
                data=np.empty((0, len(analogConfig["channels"]))),
                channelNames=analogConfig["channels"],
            )
            empty.scanRate = None
            empty.timing = {
                "validSampleCount": 0,
                "timelineValid": False,
                "status": device.getAnalogReadStatus(),
            }
            data.analogTraceData = empty

        data.acquisitionMetadata["status"] = "partialOrFailed" if errors else "complete"
        data.acquisitionMetadata["frameRowsRetained"] = len(frameRows)
        data.analysisStatus = "notAnalyzed" if mapping is not None else "clockMappingUnavailable"

        # Freeze raw metadata before first publication. Later derived analysis
        # changes only explicit result fields, not trace/acquisition metadata.
        data.save()

        if mapping is not None and data.analogTraceData.nSamples:
            try:
                data.analyzePhotodiodeOnsets(
                    dataChannel=dataChannel,
                    mappingId=mapping.eventId,
                    minimumAmplitude=minimumAmplitude,
                )
            except Exception as error:
                data.analysisStatus = "failed"
                data.analysisFailure = f"{type(error).__name__}: {error}"
                pglMessages.warning(
                    f"Photodiode analysis failed; raw recording retained: {error}",
                    level=2,
                )
            data.save()

        if errors:
            pglMessages.warning(
                "Temporal recording retained partial/failure evidence: "
                + "; ".join(f"{entry['stage']}: {entry['message']}" for entry in errors),
                level=2,
            )
            raise RuntimeError("Temporal acquisition did not complete; retained data are available.")

        return data
            
    def calibrateLuminance(self, settingsName, nRepeats=4, nSteps=256, runValidation=True, nStepsValidation=None, nRepeatsValidation=None, runGammaValidation=True):
        '''
        User faceing function which runs the luminance calibration process. It will get a luminance calibration
        and then validate that you get a linear gamma (gamma=1.0) if runValidation is set and a 2.2 gamma
        for display natural images and movies if runGammaValidation is set.
        
        You need to make sure that you have initialized with a luminnaceCalibrationDevice or run addLuminanceCalibrationDevice
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
            runValidation (bool): If True, run validation after calibration.
            nStepsValidation (int): Number of steps in the validation. If None, will use 1
            nRepeatsValidation (int): Number of times to repeat each measurement in the validation. If None, will use 32
            runGammaValidation (bool): If True, run gamma validation after calibration. This can validate
                for making a gamma of 2.2 for videos
        '''
        if self.luminanceCalibrationDevice is None:
            pglMessages.warning("No luminance calibration device specified. Please specify on initialization of pglDisplayCalibration or use addLuminanceCalibrationDevice")
            return None

        # initialize calibration data
        self.luminanceCalibrationData = None
        self.luminanceValidationData = None
        self.luminanceGammaValidationData = None
        
        # open the display with specified settings
        e = pglExperiment(self.pgl, settingsName)
        if e.isInitialized is False:
            pglMessages.warning(f"(pglDisplayLuminanceCalibrationData:attachSettings) Could not initialize experiment with settings: {settingsName}")
            return None

        e.settings.closeScreenOnEnd = True
        e.initScreen()
        if self.pgl.isOpen() is False:
            pglMessages.warning(f"Display {settingsName} did not open, cannot calibrate.")
            return None
        
        # run calibraiton
        self.luminanceCalibrationData = self._calibrateLuminance(e, settingsName, nRepeats, nSteps, validate=False)
        self.luminanceCalibrationData.display()

        if nStepsValidation is None:
            nStepsValidation = min(8, nSteps)
        if nRepeatsValidation is None:
            nRepeatsValidation = min(1, nRepeats)

        # run validation
        if runValidation:
            self.luminanceValidationData = self.validateLuminance(e, nRepeats=nRepeatsValidation, nSteps=nStepsValidation, gamma=1.0)
            self.luminanceValidationData.display(gamma=1.0)
    
        if runGammaValidation:
            self.luminanceGammaValidationData = self.validateLuminance(e, nRepeats=nRepeatsValidation, nSteps=nStepsValidation, gamma=2.2)
            self.luminanceGammaValidationData.display(gamma=2.2)

        # and save
        self.saveLuminanceCalibration()
        
        # close screen
        e.endScreen()

    def validateLuminance(self, e=None, nRepeats=None, nSteps=None, gamma=1.0):
        '''
        Validate the display calibration by applying inverse gamma and re-measuring. Usually
        run automatically by calibrate() after calibration is complete.
        
        Args:
            e: The experiment variable which contains the display settings (None will start a new one)
            nRepeats (int): Number of times to repeat each measurement (None will use the same as calibration)
            nSteps (int): Number of steps in the validation. (None will use the same as calibration)
        '''
        if self.luminanceCalibrationDevice is None:
            pglMessages.warning("No luminance calibration device specified. Please specify on initialization of pglDisplayCalibration")
            return None

        if not self.checkLuminanceCalibrationData(self.luminanceCalibrationData):
            pglMessages.warning("Calibration data is not valid. Cannot run validation.")
            return None
        pglMessages.message("(pglCalibration) Starting validation with inverse gamma correction...")
        
        closeScreenOnEnd = False
        if e is None:
            # open the display with specified settings
            e = pglExperiment(self.pgl, self.luminanceCalibrationData.settingsName)
            if e.isInitialized is False:
                pglMessages.warning(f"(pglDisplayLuminanceCalibrationData:attachSettings) Could not initialize experiment with settings: {settingsName}")
                return None

            e.initScreen()
            closeScreenOnEnd = True
            if self.pgl.isOpen() is False:
                pglMessages.message(f"Display {self.luminanceCalibrationData.settingsName} did not open, cannot validate.")
                return None
        
        # defaults for nRepeats and nSteps
        if nRepeats is None:
            nRepeats = self.luminanceCalibrationData.nRepeats
        if nSteps is None:
            nSteps = self.luminanceCalibrationData.nSteps
            
        # Calculate inverse gamma table from calibration measurements
        inverseGammaTable = self.luminanceCalibrationData.calculateInverseGamma(gamma=gamma)
        
        # Run calibration with the inverse gamma table
        luminanceValidationData = self._calibrateLuminance(
            e,
            self.luminanceCalibrationData.settingsName,
            nRepeats=nRepeats, 
            nSteps=nSteps,
            validate=True,
            inverseGammaTable=inverseGammaTable
        )
        
        # save the gamma that we were trying to achieve
        luminanceValidationData.gamma = gamma
        
        if closeScreenOnEnd:
            self.luminanceValidationData = luminanceValidationData
            e.endScreen()
            self.save()
            
        return luminanceValidationData
 
    def _calibrateLuminance(self, e, settingsName, nRepeats=4, nSteps=256, validate=False, inverseGammaTable=None):
        '''
        Measure the display characteristics.
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
            validate (bool): If True, use inverseGammaTable instead of linear gamma.
            inverseGammaTable: Tuple of (R, G, B) gamma tables to apply for validation.
        '''
        
        # initialize calibration data
        self.currentLuminanceCalibrationData = pglDisplayLuminanceCalibrationData()
        self.currentLuminanceCalibrationData.deviceDescription = self.luminanceCalibrationDevice.description
        self.currentLuminanceCalibrationData.units = self.luminanceCalibrationDevice.units()
        self.calibrationIndex = 0
        self.fullCalibrationIndex = None
        self.findMinIndex = None
        
        # no progress bar at start
        self.progressBar = None
        
        # start with trying to measure min and max
        self.calibrationMode = "minmax"
        
        # get calibration date and time
        self.currentLuminanceCalibrationData.creationDateTime = datetime.now()

        # save settings name
        self.currentLuminanceCalibrationData.settingsName = settingsName
        self.currentLuminanceCalibrationData.settings = pglSettingsManager.getSettings(settingsName)
        
        # save the current gamma table so we can replace it
        displayNumber = self.currentLuminanceCalibrationData.settings.displays[0].currentDisplayNum
        self.currentGammaTable = self.pgl.getGammaTable(displayNumber)
        
        # Set gamma table - either linear or inverse for validation
        if validate and inverseGammaTable is not None:
            self.pgl.setGammaTable(displayNumber, inverseGammaTable[0], inverseGammaTable[1], inverseGammaTable[2])
        else:
            self.pgl.setGammaTableLinear(displayNumber)
        
        # get the gamma table and save it
        self.currentLuminanceCalibrationData.gammaTableSize = self.pgl.getGammaTableSize(displayNumber)
        gammaTable = self.pgl.getGammaTable(displayNumber)
        self.currentLuminanceCalibrationData.gammaTable = tuple(np.array(table) for table in gammaTable)

        # set number of repeats and steps
        self.currentLuminanceCalibrationData.nRepeats = nRepeats
        self.currentLuminanceCalibrationData.nSteps = nSteps

        # get the display info
        try:
            # get display info if available
            gpu = next(iter(self.pgl.gpuInfo.values()))
            displays = gpu.get('Displays', [])
            self.currentLuminanceCalibrationData.displayInfo = displays[displayNumber]
            
            # get info from pgl
            self.currentLuminanceCalibrationData.metalInfo = self.pgl.info()
            
        except Exception as ex:
            pglMessages.warning(f"Could not get display info: {ex}")
        
        # set the current display
        self.setDisplay()

        # now, tell operator to make sure everything is setup before continuing
        if validate:
            pglMessages.message("Starting validation measurement...")
            pglMessages.printHeader("Validating calibration with inverse gamma")
        else:
            pglMessages.message("Please ensure the calibration device is properly positioned and ready.")
            pglMessages.message("Press Enter to continue...")
            #input("")
            pglMessages.printHeader("Establishing min and max values")
        
        # loop to set display and make measurements
        while (self.setDisplay() != -1):
            # make a measurement
            self.measure()
            # and display
            self.displayProgress()

        # restore the original gamma table
        self.pgl.setGammaTable(displayNumber, self.currentGammaTable[0], self.currentGammaTable[1], self.currentGammaTable[2])
        
        # collect min and max measurements
        _, measurements, _, _ = self.currentLuminanceCalibrationData.getMedianMeasurements()
        self.currentLuminanceCalibrationData.minLuminance = measurements[0]
        self.currentLuminanceCalibrationData.maxLuminance = measurements[-1]
        
        # return the calibration data
        calibrationData = self.currentLuminanceCalibrationData
        self.currentLuminanceCalibrationData = None
        return(calibrationData)
    
    def checkLuminanceCalibrationData(self, luminanceCalibrationData):
        '''
        Check if the calibration data is valid.
        
        Args:
            luminanceCalibrationData: The calibration data to check.
        
        Returns:
            True if valid, False otherwise.
        '''
        
        if luminanceCalibrationData is None:
            pglMessages.warning("No calibration data available. Please run calibrate() first.")
            return False
        
        # Check that calibration data is complete
        if not hasattr(luminanceCalibrationData, 'calibrationValues') or \
        not hasattr(luminanceCalibrationData, 'calibrationMeasurements'):
            pglMessages.warning("Calibration data is incomplete. Please run calibrate() first.")
            return False
        
        # Verify the calibration data matches expected size
        expectedSize = luminanceCalibrationData.nRepeats * luminanceCalibrationData.nSteps
        if len(luminanceCalibrationData.calibrationValues) != expectedSize or \
        len(luminanceCalibrationData.calibrationMeasurements) != expectedSize:
            pglMessages.warning(f"Calibration data size mismatch. Expected {expectedSize}, got {len(self.luminanceCalibrationData.calibrationValues)}")
            return False
        return True 

    def display(self):
        '''
        Display results of calibration and validation
        '''
        if self.checkLuminanceCalibrationData(self.luminanceCalibrationData):
            self.luminanceCalibrationData.display()
        if self.checkLuminanceCalibrationData(self.luminanceValidationData):
            self.luminanceValidationData.display(gamma=1.0)
        if self.checkLuminanceCalibrationData(self.luminanceGammaValidationData):
            self.luminanceGammaValidationData.display(gamma=2.2)

    def displayProgress(self, startProgress=False):
        '''
        Display the progress of the calibration.
        '''
        if startProgress:
            self.progressBar = tqdm(total=self.currentLuminanceCalibrationData.nSteps*self.currentLuminanceCalibrationData.nRepeats, desc="Calibrating", unit="measurements")
        else:
            if self.calibrationMode == "minmax":
                pglMessages.print(f"{self.calibrationValueGet(-1)}: {self.calibrationMeasurementGet(-1)}")
            elif self.calibrationMode == "minsearch":
                pglMessages.print(f"{self.calibrationValueGet(-1)}: {self.calibrationMeasurementGet(-1)}")
            elif self.calibrationMode == "full":
                pglMessages.print(f"{self.calibrationMeasurementGet(-1):<6} ", end="")
                if self.progressBar is not None:
                    self.progressBar.update(1)
        
    def getNextCalibrationValue(self):
        '''
        Get the calibration values. This chooses which value to test
        based on what values have been measured so far. Sometimes
        calibration devices have a problem with low luminance measurements
        so will need to skip values if measurents fail
        '''
        # if we already have a calibration value that hasn't been measured
        # then return that
        if self.currentLuminanceCalibrationData.hasLastMeasurement(self.saveLocation) is False:
            return self.currentLuminanceCalibrationData.getLastValue(self.saveLocation)
        # check for minmax mode to get min and max values
        if self.calibrationMode == "minmax":
            return self.getMinMaxCalibrationValue()
        elif self.calibrationMode == "minsearch":
            # this mode is to try to find a measurable min value
            return self.getFindMinCalibrationValue()
        elif self.calibrationMode == "full":
            # this mode is to do a full calibration
            return self.getFullCalibrationValue()
        else:
            pglMessages.message("Calibration is complete.")
            return None
    
    def getMinMaxCalibrationValue(self):
        '''
        Get the min and max calibration values. This mode is to establish
        the min and max measurable values
        '''
        # first show the max value (this should always be measurable
        if self.calibrationIndex < self.currentLuminanceCalibrationData.nRepeats:
            self.calibrationValueAppend(1.0)
        # next show the min value (this may not be measurable)
        elif self.calibrationIndex < 2*self.currentLuminanceCalibrationData.nRepeats:
            self.calibrationValueAppend(0.0)
        # Check if we have valid min and max values
        else:
            # get the measured min and max
            self.maxCalibrationVal = self.calibrationValueGet(0)
            # see how many measurement failures we had for min
            nMeasurementFailures = sum(x is None for x in self.calibrationMeasurementGet(self.currentLuminanceCalibrationData.nRepeats, 2*self.currentLuminanceCalibrationData.nRepeats))
            if nMeasurementFailures == 0:
                # get the min value
                self.minCalibrationVal = self.calibrationValueGet(-1)
                # and set our mode to full calibration
                self.calibrationMode = "full"
                return self.getFullCalibrationValue()
            else:
                # this calibration mode is to try to find the min value
                self.calibrationMode = "minsearch"
                pglMessages.message("Minimum value was not measurable, trying to find a measureable minimum")
                return self.getFindMinCalibrationValue()
                
        # return the value that was set
        return self.calibrationValueGet(-1)

    def getFindMinCalibrationValue(self):
        '''
        Returns values that do a bisection search to find the minimum
        measurable value
        '''
        
        # if we have no measurements yet, start at 0.5
        if self.findMinIndex is None:
            # remember where we started the min search
            self.findMinIndex = self.calibrationIndex
            # set the interval to search between
            self.findMinUnmeasurableVal = 0.0
            self.findMinMeasurableVal = self.maxCalibrationVal
        else:
            # see if the last measurement was valid
            if self.calibrationMeasurementGet(-1) is not None:
                if self.calibrationValueGet(-1) < self.findMinMeasurableVal:
                    self.findMinMeasurableVal = self.calibrationValueGet(-1)
            else:
                # Unmeasurable, so replace unmeasurable value
                if self.calibrationValueGet(-1) > self.findMinUnmeasurableVal:
                    self.findMinUnmeasurableVal = self.calibrationValueGet(-1)
            # if we are within epsilon of the unmeasurable value, then accept
            if (self.findMinMeasurableVal-self.findMinUnmeasurableVal) < self.epsilon:
                pglMessages.message(f"Minimum measurable value found: {self.findMinMeasurableVal}")
                self.minCalibrationVal = self.findMinMeasurableVal
                self.calibrationMode = "full"
                return self.getFullCalibrationValue()
        # Check the halfway point between the current unmesurable and measurable
        newVal = self.findMinUnmeasurableVal + (self.findMinMeasurableVal - self.findMinUnmeasurableVal) / 2.0
        self.calibrationValueAppend(newVal)
        
        return self.calibrationValueGet(-1)
    
    def getFullCalibrationValue(self):
        '''
        Get the next calibration value for a full calibration
        '''

        # starting fullcalibration
        if self.fullCalibrationIndex is None:
            self.fullCalibrationIndex = self.calibrationIndex
            # display the min and max value
            pglMessages.printHeader(f"Starting Full Calibration between {self.minCalibrationVal} and {self.maxCalibrationVal}")
            self.displayProgress(startProgress=True)

        if ((self.calibrationIndex-self.fullCalibrationIndex) % self.currentLuminanceCalibrationData.nRepeats) == 0:
            # set the next value to measure
            self.currentStep = (self.calibrationIndex-self.fullCalibrationIndex) // self.currentLuminanceCalibrationData.nRepeats
            if self.currentStep == self.currentLuminanceCalibrationData.nSteps:
                pglMessages.print(f"")
                pglMessages.printHeader("Full calibration complete.")
                self.calibrationMode = "done"
                return -1
            value = self.minCalibrationVal + (self.maxCalibrationVal - self.minCalibrationVal) * (self.currentStep / (self.currentLuminanceCalibrationData.nSteps - 1))
            pglMessages.print(f"\n(pglCalibration) Measuring step {self.currentStep+1:>3d}/{self.currentLuminanceCalibrationData.nSteps}: {value:.4f} = ", end="")
            self.calibrationValueAppend(value)
        else:
            # repeat last value
            self.calibrationValueAppend(self.calibrationValueGet(-1))

        return self.calibrationValueGet(-1)

    def calibrationValueAppend(self, value):
        '''
        Append a calibration value to calibrationData
        '''
        self.currentLuminanceCalibrationData.appendValue(value, self.saveLocation)
    
    def calibrationValueGet(self, startIndex, endIndex=None):
        '''
        Get calibration value(s) from calibrationData
        '''
        if endIndex is None:
            return self.currentLuminanceCalibrationData.getValues(startIndex, None, self.saveLocation)
        else:
            return self.currentLuminanceCalibrationData.getValues(startIndex, endIndex, self.saveLocation)
    
    def calibrationMeasurementAppend(self, measurement):
        '''
        Append a calibration measurement to calibrationData
        '''
        self.currentLuminanceCalibrationData.appendMeasurement(measurement, self.saveLocation)
    
    def calibrationMeasurementGet(self, startIndex, endIndex=None):
        '''
        Get calibration measurement(s) from calibrationData
        '''
        if endIndex is None:
            return self.currentLuminanceCalibrationData.getMeasurements(startIndex, None, self.saveLocation)
        else:
            return self.currentLuminanceCalibrationData.getMeasurements(startIndex, endIndex, self.saveLocation)
    
    def setDisplay(self):
        '''
        Set the display to the calibration value
        '''
        # get the next calibration value
        value = self.getNextCalibrationValue()
        if value == -1: return -1
        
        # set the display to that value
        self.pgl.clearScreen(value)
        self.pgl.flush()
        self.pgl.clearScreen(value)
        self.pgl.flush()
        return value
    
    def measure(self): 
        '''
        Measure the display using the calibration device
        '''
        if self.luminanceCalibrationDevice is None:
            pglMessages.message("No calibration device specified.")

        # make measurement
        self.calibrationMeasurementAppend(self.luminanceCalibrationDevice.measure())
        
        # increment index
        self.calibrationIndex += 1
        return self.calibrationMeasurementGet(-1)

    def saveLuminanceCalibration(self):
        '''
        Save calibration data to a file.
        
        Args:
            filename (str, optional): The filename to save the calibration data to.
            If None, a default filename based on the current date will be used.
        '''
        # check calibration data
        if not self.checkLuminanceCalibrationData(self.luminanceCalibrationData):
            pglMessages.warning("Calibration data is not valid. Cannot save.")
            return None
        
        # save the calibration data
        self.luminanceCalibrationData.save()    
        if self.checkLuminanceCalibrationData(self.luminanceValidationData):
            self.luminanceValidationData.save(filename="validation")
        if self.checkLuminanceCalibrationData(self.luminanceGammaValidationData):
            self.luminanceGammaValidationData.save(filename="gammaValidation")
        
    @staticmethod
    def chooseCalibrationFilepath(displayName, calibrationType=None, date=None):
        '''
        List the available calibration directories for a display and let the
        user choose one. Returns the chosen directory Path, or None.

        Args:
            displayName: the display name whose calibrations to list
            date: optional filter (str "YYYYMMDD", datetime, or date)
        '''
        
        filepath = pglDisplayCalibration.getCalibrationFilepath(displayName, makePath=False, calibrationType=calibrationType)
        if filepath is None:
            pglMessages.warning(f"Could not get filepath for calibrations")
            return
        
        # get all directories
        dirList = [d for d in filepath.iterdir() if d.is_dir()]
        
        # filter by date if requested
        if date is not None:
            if isinstance(date, str):
                dateStr = date
            elif isinstance(date, datetime):
                dateStr = date.strftime("%Y%m%d")
            elif isinstance(date, Date):
                dateStr = date.strftime("%Y%m%d")
            else:
                raise TypeError("date must be a string, datetime.date, datetime.datetime, or None")
            dirList = [d for d in dirList if d.name.startswith(dateStr)]

        # sort the directory list
        dirList = sorted(dirList, key=lambda d: d.name)

        # nothing to show
        if len(dirList) == 0:
            pglMessages.warning(f"No calibration directories found in: {filepath}")
            return None

        # print header
        pglMessages.message(f"Calibration directory: {filepath}")

        # print each directory
        for i, d in enumerate(dirList, start=1):
            try:
                # try to parse the typical timestamp name
                dt = datetime.strptime(d.name, "%Y%m%d_%H%M%S")
                printName = dt.strftime("%A %B %-d, %Y %-I:%M%p")

                # try to read a calibration file for extra info
                calibrationFilename = calibrationDir / d.name / "calibration.json"
                if calibrationFilename.exists():
                    try:
                        calibrationData = pglSerialize.load(calibrationFilename)
                        # add some useful summary info
                        printName += f" | nSteps: {calibrationData.nSteps}"
                        printName += f" | nRepeats: {calibrationData.nRepeats}"
                        printName += f" | maxLum: {calibrationData.maxLuminance:.2f}"
                    except:
                        pass

                # add the raw folder name
                printName = f"{printName} ({d.name})"
            except:
                # not a typical name, just show it
                printName = d.name

            pglMessages.print(f"{i}. {printName}", flush=True)

        # ask the user to choose
        pglMessages.print("\nSelect a directory number: ", flush=True)
        choice = int(input())
        if choice < 1 or choice > len(dirList):
            pglMessages.warning(f"Invalid choice: {choice}",level=1)
            return None

        # return the chosen directory
        chosenDir = dirList[choice - 1]
        return chosenDir
    
    @staticmethod
    def getCalibrationFilepath(displayName, makePath=False, calibrationType="luminance"):
        '''
        Get default filepath for displays (usually ~/.pgl/calibrations)
        
        Args:
            displayName: the name of the display to get the filepath for
            makePath: Set to true when saving and it will append the data (and time if necessary)
                and will make the path if it does not exist
        '''
        from pgl import pgl
        
        # make the path be a directory
        settingsPath = pglSettingsManager().getCalibrationsDir()
        settingsPath = settingsPath / pglStorage.makeValidFilename(displayName)
        settingsPath = settingsPath / calibrationType
        
        # Check if directory exists, if so add time
        if makePath:
            # add the data
            dateStem = datetime.now().strftime("%Y%m%d")
            filepath = settingsPath / dateStem
            if filepath.exists():
                dateStem = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = settingsPath / dateStem
        else:
            filepath = settingsPath

        # Create the directory
        if makePath:
            filepath.mkdir(parents=True, exist_ok=True)
        else:
            # check that the path exists
            if not filepath.exists() or not filepath.is_dir():
                pglMessages.warning(f"Could not find calibration directory: {filepath}")
                return None
        
        return(filepath)

# Calibration settings, subclass of pglSettings to inherit load/save functionality
class pglDisplayTemporalCalibrationData(pglTraitSettings):
    
    settingsName = Unicode("Default", help="Settings name used to open display")
    settings = Instance(pglSettings, allow_none=True, help="Settings used during calibration") 
    displaySettings = Instance(pglDisplaySettings, allow_none=True, help="Display settings used during calibration")
    displayInfo = Dict(allow_none=True, default_value=None, help="Display information at time of calibration")
    metalInfo = Dict(allow_none=True, default_value=None, help="PGL info including display info such as UUID, serial number, and other information at time of calibration")   
    creationDateTime = Instance(datetime, default_value=datetime.now(), help="Date and time of calibration creation")
    digitalIOdeviceDescription = Unicode("Unknown", help="Desciption of digitalIO measurement device")
    analogInputDescription = Unicode("Unknown", help="Desciption of analog input measurement device")
    analogTraceData = Instance(pglAnalogTraceData, allow_none=True, default_value=None, help="analong measurement from photodiode")
    analogScanRate = Float(1000.0, help="Actual device-returned analog scan rate (Hz).")
    syncChannel = Int(1, help="Channel with sync pulse on it")     
    syncChannelThreshold = Float(0.2, help="Threshold for considering sync to be active")
    stimulusDurationFrames = List(Int, help="Frames used for each part of stimulus: first number is pre-stimulus black, second number is the actual stimulus duration, third number is post-stimulus black, fourth number is inter-trial interval ")    
    numRepeats = Int(help="Number of repeats of stimulus")
    frameRate = Float(help="Frame rate of display")
    flushTimes = Instance(np.ndarray, allow_none=True, default_value=None, help="Matrix of repeats x frames which reports the flush times")
    onsetDelay = Float(allow_none=True, default_value=None, help="delay to frame onset, computed from calibration data")
    
    def _displayLegacyTemporal(self, fig=None):
        '''
        display the timing calibration data
        
        Args:
            fig:     matplotlib fig to plot into, None to create new plot
        '''
        if self.analogTraceData is None:
            pglMessages.message(f"No data to display)")
            return
        
        # display the analog traces        
        self.analogTraceData.display(fig=fig, digitalSyncChannel=self.syncChannel, digitalSyncThreshold=self.syncChannelThreshold, ignoreInitial=0)
        self.displayOnset()
        
    def save(self, filename=None, filepath=None):
        """
        Save raw HDF5 once, then publish calibration.json with derived results.

        Previous JSON revisions and staging files are retained.
        Publication is per-file, not an atomic two-file transaction.
        Failures propagate; the in-memory data remain available for retry.
        """
        import os
        from pathlib import Path
        from uuid import uuid4
        from .pglSerialize import pglSerialize

        if filename not in (None, "calibration", "calibration.json"):
            raise ValueError("Temporal recordings use the filename calibration.json.")

        storage = self._getTemporalStorageClass()

        if filepath is None:
            retainedDirectory = getattr(self, "_temporalSaveDirectory", None)
            if retainedDirectory is not None:
                directory = Path(retainedDirectory)
            else:
                # Reserve a unique directory atomically; never reuse an earlier run.
                parent = pglSettingsManager.getDisplayTemporalCalibrationDir(
                    displaySettings=self.displaySettings, newCalibration=False
                )
                parent.mkdir(parents=True, exist_ok=True)
                directory = parent / (
                    datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    + "_" + uuid4().hex[:8]
                )
                directory.mkdir(exist_ok=False)
        else:
            directory = Path(filepath).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)

        self._temporalSaveDirectory = str(directory)
        frames, metadata = self._prepareTemporalPayload()

        # Preserve legacy arrays in the HDF5 payload without assuming they are
        # drawablePresented. The layout/provenance is explicitly recorded.
        legacyFlush = self.flushTimes
        if legacyFlush is not None:
            if frames:
                count = len(next(iter(frames.values())))
                if legacyFlush.size != count:
                    raise ValueError(
                        "flushTimes and frameRecords differ in size. "
                        "Provide matching frame records; values were not reshaped silently."
                    )
            frames["legacyFlushTimes"] = legacyFlush.reshape(-1)

        reference = dict(self.calibrationDataReference)
        if reference:
            if "calibrationUUID" in reference and reference["calibrationUUID"] != self.uuid:
                raise ValueError("Raw reference belongs to another calibration.")
            restored = storage.loadRaw(directory, reference)
            self._assertSameTemporalPayload(
                self.analogTraceData, frames, metadata, restored
            )
        else:
            if (directory / "calibration.json").exists():
                raise FileExistsError(
                    "Existing calibration.json has no bound raw reference on this "
                    "object. Use a new directory for migration or a new recording."
                )
            reference = storage.saveRaw(
                directory, self.analogTraceData, frames, metadata
            )
            reference["calibrationUUID"] = self.uuid
            # Retain publication immediately: a later JSON failure must not cause
            # retry to overwrite or republish the immutable raw recording.
            self.calibrationDataReference = reference

        text = self.toJSON(strict=True)
        destination = directory / "calibration.json"
        attemptId = str(uuid4())
        staging = directory / f".calibration.{attemptId}.staging.json"

        with staging.open("x", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        if destination.exists():
            # Preserve the previous bytes before replacing the public JSON name.
            previous = destination.read_bytes()
            oldObject = pglSerialize.fromJSON(
                previous.decode("utf-8"), strict=True
            )
            if not isinstance(oldObject, type(self)) or oldObject.uuid != self.uuid:
                raise ValueError("Existing JSON belongs to another calibration.")
            with (directory / f"calibration.previous-{attemptId}.json").open("xb") as file:
                file.write(previous)

            # Atomic replacement of the JSON name, retaining staging evidence.
            publish = directory / f".calibration.{attemptId}.publish.json"
            os.link(staging, publish)
            os.replace(publish, destination)
        else:
            # Fail rather than overwrite a destination created concurrently.
            os.link(staging, destination)

        self._temporalLastSaveFailure = None
        return destination

    @classmethod
    def load(cls, displayName=None, filename=None, filepath=None, date=None):
        """
        Restore HDF5-backed temporal data or an existing JSON-only recording.

        Does not recalibrate, analyze, or rewrite historical recordings.
        Missing legacy timing evidence remains explicitly unavailable.
        """
        from pathlib import Path
        from .pglSerialize import pglSerialize

        if filename not in (None, "calibration", "calibration.json"):
            raise ValueError("Temporal recordings use the filename calibration.json.")

        if filepath is None:
            filepath = pglDisplayCalibration.chooseCalibrationFilepath(
                displayName, date=date, calibrationType="temporal"
            )
            if filepath is None:
                return None

        directory = Path(filepath).expanduser().resolve()
        filename = directory / "calibration.json"

        # Ordinary restoration preserves legacy trait defaults. Do not reinterpret
        # legacy sync-relative onsetDelay as presentation-relative latency.
        obj = pglSerialize.fromJSON(
            filename.read_text(encoding="utf-8"), filename=str(filename)
        )
        if not isinstance(obj, cls):
            raise TypeError("Calibration JSON restored an unexpected class.")

        reference = obj.calibrationDataReference
        if reference:
            storage = cls._getTemporalStorageClass()
            restored = storage.loadRaw(directory, reference)
            if restored["metadata"].get("calibrationUUID") != obj.uuid:
                raise ValueError("JSON and raw calibration identities disagree.")

            if reference.get("calibrationUUID", obj.uuid) != obj.uuid:
                raise ValueError("JSON raw reference belongs to another calibration.")

            obj.analogTraceData = restored["trace"]
            frames = dict(restored["frameArrays"])
            legacyFlush = frames.pop("legacyFlushTimes", None)
            shape = restored["metadata"].get("legacyFlushTimesShape")

            if restored["metadata"].get("legacyFlushTimesStored"):
                if legacyFlush is None or shape is None:
                    raise ValueError("Missing retained flush-time array/shape.")
                obj.flushTimes = legacyFlush.reshape(tuple(shape))
            else:
                if legacyFlush is not None:
                    raise ValueError("Unexpected legacy flush-time dataset.")
                obj.flushTimes = None

            obj.frameRecords = frames

        # A loaded historical directory is not an automatic save destination.
        # Reanalysis requires an explicit save destination.
        obj._temporalLoadedDirectory = str(directory)
        return obj
    
    def getDisplayName(self):
        '''
        Get the display name associated with this data
        '''
        return self.displaySettings.name
    
    def getUUID(self):
        '''
        Get the UUID associated with this data
        '''
        return self.displaySettings.uuid
    
    def displayOnset(self):
        # draw vertical lines for onset and offset of video frames
        if self.onsetDelay is not None:
            times = [self.onsetDelay + 1000*i/self.frameRate for i in range(self.stimulusDurationFrames[1]+1)]
            pglMessages.print(f"times: {times}")
            [plt.axvline(x=t, color='red', linestyle='--') for t in times]

    def computeFrameOnsetDelay(self):
        '''
        Use saved analogTraceData to estimate frame onset time
        
        '''
        if self.analogTraceData is None:
            pglMessages.message("No analog data recorded to computeFrameOnsetDelay from. Run temporal calibration first")
            return
        
        # chanel which has data on it
        dataChannel = 0 if self.syncChannel == 1 else 1
        # get the cycleData
        cycleData = self.analogTraceData.getCycles(digitalSyncChannel=self.syncChannel, digitalSyncThreshold=self.syncChannelThreshold, ignoreInitial=0)
        
        # get mask for last 250 ms of data
        maxTime = np.max(self.analogTraceData.time)
        mask = self.analogTraceData.time >= (maxTime-0.25)
        #print(self.analogTraceData.data[dataChannel].size)
        #print(mask.size)
        baseline = np.mean(self.analogTraceData.data[mask][dataChannel])
        baselineSTD = np.std(self.analogTraceData.data[mask][dataChannel])
        
        # calculate standard deviation around 0 for beginning of cycleData
        #baseline = np.mean(cycleData['median'][dataChannel][mask])
        #baselineSTD = np.std(cycleData['median'][dataChannel][mask])
        
        # now see when the median trace goes 2 std above the baseline
        onset = np.argmax(cycleData['median'][dataChannel]>(baseline+baselineSTD*6))
        #print(f"baseline: {baseline} baselineSTD: {baselineSTD}")
        #print(f"onset: {onset}")
        
        # let's refine the search for the onset, by fitting a linear
        # finction and finding the intersection with baseline

        # let's refine the search for the onset, by fitting a linear
        # finction and finding the intersection with baseline
        cycle = cycleData['median'][dataChannel]
        minCycle = np.min(cycle)
        maxCycle = np.max(cycle)

        # set the thresholds for where we will do the linear fit.
        leftThreshold = baseline + baselineSTD * 1.5
        rightThreshold = baseline + baselineSTD * 20

        # Search backwards from onset
        leftIndex = onset
        while leftIndex > 0 and cycle[leftIndex] > leftThreshold:
            leftIndex -= 1

        # Find right point
        rightIndex = onset
        while rightIndex < len(cycle) - 1 and cycle[rightIndex] < rightThreshold:
            rightIndex += 1

        leftIndex = max(0, onset - 2)
        rightIndex = min(len(cycle) - 1, onset + 2)

        # Make sure we have enough points to fit
        linearFit = False
        if rightIndex - leftIndex < 3:
            pglMessages.warning("Warning: fitting window too small, using fallback",level=1)
        else:
            # Fit linear function over this adaptive window
            xFit = np.arange(leftIndex, rightIndex + 1)
            yFit = cycle[leftIndex:rightIndex + 1]
            pglMessages.print(f"xFit: {xFit}")
            pglMessages.print(f"yFit: {yFit}")
            coeffs = np.polyfit(xFit, yFit, 1)
            m, c = coeffs
            
            # Find intercept with 20% to max from baseline
            target = baseline + 0.20 * (maxCycle-minCycle)
            if m != 0:
                onset = (target - c) / m
                linearFit = True

        plt.ion()
        plt.figure(figsize=(14,7))
        plt.plot(cycle, 'k-', label='Median trace', linewidth=1.5)
            #plt.axvline(x=58, color='r', linestyle='--', linewidth=2)
        plt.axvline(x=onset, color='g', linestyle='--', linewidth=2)
        if linearFit:
            xLineExtended = np.array([onset, rightIndex])
            yLineExtended = m * xLineExtended + c
            plt.plot(xLineExtended, yLineExtended, 'b-', linewidth=2.5, label='Fitted line', alpha=0.8)
            plt.xlim(onset-10,onset+10)
            
        # return onset in ms (not the scanRate of the analog device)
        self.onsetDelay = float(onset*1000/self.analogScanRate)


    calibrationDataReference = Dict(
        default_value={}, help="Reference to immutable calibrationData.h5."
    )
    frameRecords = Dict(
        default_value={}, help="Numeric frame-record arrays stored in HDF5."
    )
    acquisitionMetadata = Dict(
        default_value={}, help="Acquisition configuration, validity and provenance."
    )
    clockCalibrationEvents = List(
        default_value=[], help="Retained clock calibration and mapping events."
    )
    frameTimestampProvenance = Dict(
        default_value={}, help="Presentation timestamp field and clock assumptions."
    )
    onsetResults = List(
        default_value=[], help="Per-repeat onset results and exclusions."
    )
    detectionConfiguration = Dict(
        default_value={}, help="Explicit photodiode onset-detection configuration."
    )
    timingUncertainty = Dict(
        default_value={}, help="Clock and measurement uncertainty with assumptions."
    )
    analysisMethod = Unicode(
        "legacySyncRelative",
        help="Analysis method; legacy values are not presentation-relative latency."
    )
    analysisVersion = Unicode("legacy", help="Temporal analysis version.")
    analysisStatus = Unicode("notAnalyzed", help="Analysis outcome.")
    timingEvidenceStatus = Unicode(
        "unavailable", help="Availability of calibrated timing evidence."
    )
    temporalPersistenceVersion = Int(2, help="Temporal data storage schema.")

    @staticmethod
    def _getTemporalStorageClass():
        from .pglTemporalCalibrationFile import pglTemporalCalibrationFile
        return pglTemporalCalibrationFile

    def toJSONdict(self, type="all"):
        """
        HDF5-backed recordings serialize references, not duplicate raw arrays.

        Objects without an HDF5 reference retain legacy inline serialization.
        save() always publishes HDF5 before writing a new JSON reference.
        """
        result = super().toJSONdict(type)
        if self.calibrationDataReference:
            result["analogTraceData"] = None
            result["flushTimes"] = None
            result["frameRecords"] = {}
        return result

    def _prepareTemporalPayload(self):
        """Prepare raw arrays and small metadata without changing their values."""
        from copy import deepcopy

        if self.analogTraceData is None:
            raise ValueError(
                "No analog trace is available. Retain partial acquisition "
                "diagnostics in memory; do not publish a fabricated recording."
            )

        frames = dict(self.frameRecords)
        reservedName = "legacyFlushTimes"
        if reservedName in frames:
            raise ValueError(f"{reservedName} is reserved by temporal persistence.")

        flushShape = None
        if self.flushTimes is not None:
            if (
                not isinstance(self.flushTimes, np.ndarray)
                or self.flushTimes.dtype.kind not in "iuf"
            ):
                raise TypeError("flushTimes must be a numeric array or None.")
            flushShape = list(self.flushTimes.shape)

        # The storage writer requires equally sized numeric frame columns.
        # Preserve legacy flushTimes independently if it has another layout.
        metadata = {
            "calibrationUUID": self.uuid,
            "acquisitionMetadata": deepcopy(self.acquisitionMetadata),
            "frameTimestampProvenance": deepcopy(self.frameTimestampProvenance),
            "legacyFlushTimesShape": flushShape,
            "legacyFlushTimesStored": self.flushTimes is not None,
        }
        return frames, metadata

    @staticmethod
    def _assertSameTemporalPayload(trace, frames, metadata, restored):
        """Do not silently reuse an immutable raw artifact for changed raw data."""
        from .pglSerialize import pglSerialize

        previous = restored["trace"]
        if (
            trace.data.dtype != previous.data.dtype
            or trace.time.dtype != previous.time.dtype
            or trace.data.shape != previous.data.shape
            or trace.time.shape != previous.time.shape
            or not np.array_equal(trace.data, previous.data, equal_nan=True)
            or not np.array_equal(trace.time, previous.time, equal_nan=True)
        ):
            raise ValueError(
                "Raw analog arrays differ from the published artifact. "
                "Save changed raw data in a new calibration directory."
            )

        def traceMetadata(value):
            return {
                key: item for key, item in vars(value).items()
                if not key.startswith("_") and key not in ("time", "data")
            }

        if pglSerialize.toJSON(traceMetadata(trace), strict=True) != (
            pglSerialize.toJSON(traceMetadata(previous), strict=True)
        ):
            raise ValueError(
                "Raw trace metadata changed after publication. Keep derived "
                "clock/analysis results in the temporal data fields instead."
            )

        previousFrames = restored["frameArrays"]
        if set(frames) != set(previousFrames):
            raise ValueError("Frame record fields changed after publication.")
        for key, array in frames.items():
            other = previousFrames[key]
            if (
                array.dtype != other.dtype
                or array.shape != other.shape
                or not np.array_equal(array, other, equal_nan=True)
            ):
                raise ValueError(f"Raw frame field changed after publication: {key}")

        if pglSerialize.toJSON(metadata, strict=True) != (
            pglSerialize.toJSON(restored["metadata"], strict=True)
        ):
            raise ValueError(
                "Acquisition metadata changed after raw publication. "
                "Use separate analysis/result fields for derived information."
            )

    onsetAnalysisHistory = List(default_value=[], help="Retained prior temporal analysis results.")

    def analyzePhotodiodeOnsets(
        self, *, dataChannel, mappingId, minimumAmplitude,
        thresholdFraction=0.20, baselineDuration=0.020,
        searchBefore=0.005, searchAfter=0.050,
    ):
        """
        Compare photodiode threshold crossings with drawablePresented.

        All times are seconds. Channels are explicit array-column indices.
        Baseline is the median over the pre-search baseline window.
        Threshold is baseline + thresholdFraction * (search maximum - baseline).

        Interpolate only between an actual below/above sample pair. Preserve
        that pair as the sample-localization interval; interpolation is not
        independent evidence of sub-sample accuracy.

        Existing onsetDelay remains legacy sync-relative and is not relabeled.
        """
        from copy import copy, deepcopy
        from numbers import Integral, Real
        from .pglSerialize import pglSerialize

        def finite(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Real)
                and np.isfinite(value)
            )

        if (
            isinstance(dataChannel, (bool, np.bool_))
            or not isinstance(dataChannel, Integral)
            or dataChannel < 0
        ):
            raise ValueError("dataChannel must be an explicit nonnegative column index.")
        for name, value in (
            ("minimumAmplitude", minimumAmplitude),
            ("baselineDuration", baselineDuration),
            ("searchAfter", searchAfter),
        ):
            if not finite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not finite(searchBefore) or searchBefore < 0:
            raise ValueError("searchBefore must be finite and nonnegative.")
        if not finite(thresholdFraction) or not 0 < thresholdFraction < 1:
            raise ValueError("thresholdFraction must lie strictly between zero and one.")
        if not isinstance(mappingId, str) or not mappingId:
            raise ValueError("Specify the mapping ID used for analysis.")

        trace = self.analogTraceData
        if trace is None or not callable(getattr(trace, "getSampleTiming", None)):
            raise ValueError("A trace supporting calibrated sample timing is required.")
        if not getattr(trace, "timing", None):
            raise ValueError("Legacy trace has no calibrated timing evidence.")

        samples = np.asarray(trace.data)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if (
            samples.ndim != 2 or samples.shape[0] != len(trace.time)
            or not 0 <= dataChannel < samples.shape[1]
        ):
            raise ValueError("Invalid samples × channels shape or channel index.")

        # Independent timing metadata, shared read-only sample arrays.
        # Never attach endpoint records by mutating the immutable raw trace.
        view = copy(trace)
        view.timing = deepcopy(trace.timing)
        records = {}
        for event in (
            list(view.timing.get("clockEvents", []))
            + list(self.clockCalibrationEvents)
        ):
            eventId = getattr(event, "eventId", None)
            if not isinstance(eventId, str) or not eventId:
                raise ValueError("Clock record lacks an identity.")
            previous = records.get(eventId)
            if previous is not None:
                if previous.toJSON(strict=True) != event.toJSON(strict=True):
                    raise ValueError("Conflicting clock records share an identity.")
            else:
                records[eventId] = deepcopy(event)
        view.timing["clockEvents"] = list(records.values())

        liveMappingId = view.timing["liveMappingId"]
        if mappingId not in records or liveMappingId not in records:
            raise ValueError("Requested or live mapping record is unavailable.")

        segment = view.timing["segment"]
        actualRate = segment["scanRate"]
        if not finite(actualRate) or actualRate <= 0:
            raise ValueError("Invalid actual scan rate.")
        relativeTime = np.asarray(trace.time)
        if (
            relativeTime.ndim != 1
            or not np.all(np.isfinite(relativeTime))
            or not np.allclose(
                relativeTime, np.arange(len(samples)) / actualRate,
                rtol=0, atol=1e-10,
            )
        ):
            raise ValueError("Relative sample times conflict with the recorded scan rate.")

        # This also checks retained mapping/calibration reference identities.
        origin = view.getSampleTiming(0, mappingId)
        hostSpacing = origin["sampleSpacingSeconds"]
        if not finite(hostSpacing) or hostSpacing <= 0:
            raise ValueError("Invalid mapped sample spacing.")
        hostTimes = origin["timestamp"] + np.arange(len(samples)) * hostSpacing

        frames = self.frameRecords
        required = ("repeatIndex", "frameIndex", "drawablePresented", "success")
        if any(name not in frames for name in required):
            raise ValueError("Required per-frame presentation records are missing.")
        arrays = {name: np.asarray(frames[name]) for name in required}
        if (
            any(array.ndim != 1 for array in arrays.values())
            or len({len(array) for array in arrays.values()}) != 1
        ):
            raise ValueError("Frame columns must be equal-length vectors.")
        for name in ("repeatIndex", "frameIndex"):
            array = arrays[name]
            if array.dtype.kind not in "iu" or np.any(array < 0):
                raise ValueError(f"{name} must contain nonnegative integer indices.")

        durations = list(self.stimulusDurationFrames)
        if (
            len(durations) != 4
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral) or value < 0
                for value in durations
            )
            or durations[1] < 1
            or durations[2] + durations[3] < 1
        ):
            raise ValueError("Stimulus frame durations must include white and following black frames.")
        firstWhite = int(durations[0])
        firstBlack = int(durations[0] + durations[1])
        framesPerRepeat = sum(durations)

        validCount = int(view.timing["validSampleCount"])
        if not 0 <= validCount <= len(samples):
            raise ValueError("Invalid valid-sample prefix length.")

        config = {
            "dataChannel": int(dataChannel),
            "channelName": trace.channelNames[dataChannel],
            "minimumAmplitudeVolts": float(minimumAmplitude),
            "thresholdFraction": float(thresholdFraction),
            "baselineDurationSeconds": float(baselineDuration),
            "searchBeforeSeconds": float(searchBefore),
            "searchAfterSeconds": float(searchAfter),
            "baselineMethod": "medianBeforeSearchWindow",
            "amplitudeMethod": "maximumInSearchWindowMinusBaseline",
            "onsetMethod": "firstRisingThresholdCrossingLinearInterpolation",
            "interpolationAccuracyEstablished": False,
            "mappingId": mappingId,
            "liveMappingId": liveMappingId,
        }
        results = []

        for repeatIndex in range(self.numRepeats):
            result = {
                "repeatIndex": repeatIndex,
                "status": "excluded",
                "reason": None,
                "mappingId": mappingId,
                "liveMappingId": liveMappingId,
            }
            results.append(result)

            def exclude(reason):
                result["reason"] = reason

            rows = np.flatnonzero(arrays["repeatIndex"] == repeatIndex)
            # Preserve received frame order. Do not sort away duplicates/gaps.
            if (
                len(rows) != framesPerRepeat
                or not np.array_equal(
                    arrays["frameIndex"][rows], np.arange(framesPerRepeat)
                )
            ):
                exclude("missingDuplicateOrOutOfOrderFrameRecords")
                continue

            presented = arrays["drawablePresented"][rows]
            success = arrays["success"][rows]
            if (
                presented.dtype.kind not in "iuf"
                or success.dtype.kind not in "iuf"
                or not np.all(np.isfinite(presented))
                or np.any(presented <= 0)
                or not np.all(success == 1)
                or np.any(np.diff(presented) <= 0)
            ):
                exclude("unusablePresentationRecords")
                continue

            frameOnset = float(presented[firstWhite])
            frameOffset = float(presented[firstBlack])
            result["framePresentationHostTime"] = frameOnset
            result["frameOffsetPresentationHostTime"] = frameOffset

            searchStart = frameOnset - searchBefore
            searchStop = min(frameOnset + searchAfter, frameOffset)
            baselineStart = searchStart - baselineDuration

            # Require recorded valid support through the requested windows.
            if (
                baselineStart < hostTimes[0]
                or searchStop > hostTimes[-1]
                or searchStop <= searchStart
            ):
                exclude("analysisWindowOutsideRecording")
                continue

            baselineIndices = np.flatnonzero(
                (hostTimes >= baselineStart) & (hostTimes < searchStart)
            )
            searchIndices = np.flatnonzero(
                (hostTimes >= searchStart) & (hostTimes < searchStop)
            )
            if len(baselineIndices) < 3 or len(searchIndices) < 3:
                exclude("insufficientSampleSupport")
                continue

            lastRequired = max(int(baselineIndices[-1]), int(searchIndices[-1]))
            if lastRequired >= validCount:
                exclude("analysisWindowTouchesInvalidTimeline")
                continue

            baselineValues = samples[baselineIndices, dataChannel]
            searchValues = samples[searchIndices, dataChannel]
            if (
                not np.all(np.isfinite(baselineValues))
                or not np.all(np.isfinite(searchValues))
                or np.any(baselineValues == -9999)
                or np.any(searchValues == -9999)
            ):
                exclude("nonfiniteOrMissingAnalogSamples")
                continue

            baseline = float(np.median(baselineValues))
            peak = float(np.max(searchValues))
            amplitude = peak - baseline
            result.update({
                "baselineVolts": baseline,
                "baselineStdVolts": float(np.std(baselineValues)),
                "peakVolts": peak,
                "amplitudeVolts": amplitude,
            })
            if amplitude < minimumAmplitude:
                exclude("insufficientSignalAmplitude")
                continue

            threshold = baseline + thresholdFraction * amplitude
            result["thresholdVolts"] = threshold
            if searchValues[0] >= threshold:
                exclude("signalAlreadyAboveThresholdAtSearchStart")
                continue

            candidates = np.flatnonzero(
                (searchValues[:-1] < threshold)
                & (searchValues[1:] >= threshold)
            )
            if not len(candidates):
                exclude("noSupportedRisingThresholdCrossing")
                continue

            pair = int(candidates[0])
            left = int(searchIndices[pair])
            right = int(searchIndices[pair + 1])
            if right != left + 1:
                exclude("nonAdjacentCrossingSamples")
                continue

            difference = float(samples[right, dataChannel] - samples[left, dataChannel])
            if difference <= 0:
                exclude("nonPositiveCrossingSlope")
                continue
            fraction = float((threshold - samples[left, dataChannel]) / difference)
            if not 0 <= fraction <= 1:
                exclude("unsupportedCrossingInterpolation")
                continue

            samplePosition = left + fraction
            selected = view.getSampleTiming(samplePosition, mappingId)
            live = view.getSampleTiming(samplePosition, liveMappingId)
            leftTime = view.getSampleTiming(left, mappingId)
            rightTime = view.getSampleTiming(right, mappingId)

            if any(
                item["outsideObservationSpan"]
                for item in (selected, leftTime, rightTime)
            ):
                exclude("crossingOutsideSelectedCalibrationSpan")
                continue

            # Interval propagation, not RSS. This bounds the sampled threshold
            # location under the clock model—not detector/display physics.
            partialLower = (
                leftTime["timestamp"] + leftTime["clockErrorLowerSeconds"] - frameOnset
            )
            partialUpper = (
                rightTime["timestamp"] + rightTime["clockErrorUpperSeconds"] - frameOnset
            )
            result.update({
                "status": "valid",
                "reason": None,
                "samplePosition": float(samplePosition),
                "crossingSampleIndices": [left, right],
                "photodiodeOnsetHostTime": selected["timestamp"],
                "livePhotodiodeOnsetHostTime": live["timestamp"],
                "onsetDelaySeconds": selected["timestamp"] - frameOnset,
                "liveOnsetDelaySeconds": live["timestamp"] - frameOnset,
                "clockErrorIntervalSeconds": [
                    selected["clockErrorLowerSeconds"],
                    selected["clockErrorUpperSeconds"],
                ],
                "sampleSpacingSeconds": selected["sampleSpacingSeconds"],
                "sampleBracketDelaySeconds": [
                    leftTime["timestamp"] - frameOnset,
                    rightTime["timestamp"] - frameOnset,
                ],
                "partialClockAndSampleDelayIntervalSeconds": [
                    partialLower, partialUpper,
                ],
                "totalTimingBoundAvailable": False,
            })

        valid = [result for result in results if result["status"] == "valid"]
        delays = [result["onsetDelaySeconds"] for result in valid]
        uncertainty = {
            "validRepeatCount": len(valid),
            "excludedRepeatCount": len(results) - len(valid),
            "meanDelaySeconds": float(np.mean(delays)) if delays else None,
            "acrossRepeatStdSeconds": (
                float(np.std(delays, ddof=1)) if len(delays) > 1 else None
            ),
            "sharedClockUncertaintyReducedByRepeatCount": False,
            "clockErrorEnvelopeSeconds": (
                [
                    min(result["clockErrorIntervalSeconds"][0] for result in valid),
                    max(result["clockErrorIntervalSeconds"][1] for result in valid),
                ] if valid else None
            ),
            "unknownContributions": [
                "withinRunNonlinearClockVariation",
                "hostTimestampUncertainty",
                "presentationTimestampUncertainty",
                "channelApertureUncertainty",
                "onsetDetectionNoiseAndModelError",
                "photodiodeAndAmplifierResponse",
            ],
            "presentationClockAssumption": "drawablePresentedInPGLHostClock",
            "totalTimingBoundAvailable": False,
        }

        # Commit derived results only; raw HDF5-backed fields remain unchanged.
        if self.onsetResults or self.analysisStatus != "notAnalyzed":
            self.onsetAnalysisHistory.append({
                "method": self.analysisMethod,
                "version": self.analysisVersion,
                "status": self.analysisStatus,
                "configuration": deepcopy(self.detectionConfiguration),
                "results": deepcopy(self.onsetResults),
                "uncertainty": deepcopy(self.timingUncertainty),
            })
        self.onsetResults = results
        self.detectionConfiguration = config
        self.timingUncertainty = uncertainty
        self.analysisMethod = "photodiodeThresholdRelativeToDrawablePresented"
        self.analysisVersion = "1"
        self.analysisStatus = "complete" if valid else "noValidRepeats"
        self.timingEvidenceStatus = "conditionalCalibratedClock"
        return deepcopy(results)


    analysisFailure = Unicode("", help="Retained derived-analysis failure.")
    pendingSaveFailure = Unicode("", help="Retained final save failure.")
    temporalCleanupErrors = List(default_value=[], help="Owned-resource cleanup errors.")


    def printTemporalTimingReport(self):
        """Print retained analysis results without refitting or hardware access."""
        from collections import Counter
        from copy import deepcopy
        import numpy as np

        pglMessages.printHeader("Display temporal calibration")
        if self.analysisMethod != "photodiodeThresholdRelativeToDrawablePresented":
            print("Legacy sync-relative recording; presentation-relative latency unavailable.")
            print("Clock-alignment uncertainty is unavailable, not zero.")
            return {
                "method": self.analysisMethod,
                "presentationRelativeLatencyAvailable": False,
            }

        valid = [row for row in self.onsetResults if row.get("status") == "valid"]
        exclusions = Counter(
            row.get("reason", "unspecified")
            for row in self.onsetResults if row.get("status") != "valid"
        )
        summary = deepcopy(self.timingUncertainty)

        print(f"Analysis: {self.analysisStatus} | method version: {self.analysisVersion}")
        print(
            f"Repeats: {len(valid)} valid, "
            f"{len(self.onsetResults) - len(valid)} excluded"
        )

        if valid:
            delays = np.asarray([row["onsetDelaySeconds"] for row in valid])
            print(
                f"Photodiode onset relative to recorded presentation: "
                f"{np.mean(delays) * 1000:+.3f} ms (mean)"
            )
            if len(delays) > 1:
                print(
                    f"Across-repeat standard deviation: "
                    f"{np.std(delays, ddof=1) * 1000:.3f} ms"
                )
            else:
                print("Across-repeat standard deviation: unavailable (one valid repeat)")

            bounds = [
                min(row["clockErrorIntervalSeconds"][0] for row in valid),
                max(row["clockErrorIntervalSeconds"][1] for row in valid),
            ]
            print(
                "Envelope of onset-specific conditional clock errors: "
                f"[{bounds[0] * 1e6:+.1f}, {bounds[1] * 1e6:+.1f}] µs"
            )
            spacing = [row["sampleSpacingSeconds"] * 1e6 for row in valid]
            print(
                f"Analog sample spacing: {min(spacing):.2f}–"
                f"{max(spacing):.2f} µs"
            )
        else:
            print("Photodiode-to-presentation delay: unavailable.")

        criterion = self.detectionConfiguration
        fraction = criterion.get("thresholdFraction")
        if fraction is not None:
            print(
                f"Onset criterion: {fraction * 100:g}% of the measured "
                "baseline-to-peak excursion; adjacent-sample interpolation."
            )
        print(
            "Interpolation does not establish sub-sample accuracy. "
            "Shared clock uncertainty is not reduced by the repeat count."
        )
        for reason, count in sorted(exclusions.items()):
            print(f"Excluded: {reason} (n={count})")
        unknown = summary.get("unknownContributions", [])
        if unknown:
            print("Additional unknown contributions: " + ", ".join(unknown))
        print(
            "Presentation timestamps are assumed to share the PGL clock. "
            "No total worst-case latency accuracy is claimed."
        )
        return {
            "method": self.analysisMethod,
            "presentationRelativeLatencyAvailable": bool(valid),
            "summary": summary,
            "exclusions": dict(exclusions),
        }

    def plotPresentationLatency(self, fig=None):
        """
        Plot retained onset results and original analog samples.

        Trace alignment uses recorded presentation times and the selected
        clock mapping. No nominal-period alignment or new onset fit is applied.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.analysisMethod != "photodiodeThresholdRelativeToDrawablePresented":
            raise ValueError("This recording has no presentation-relative analysis.")

        if fig is None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
        else:
            fig.clear()
            axes = fig.subplots(2, 1)

        valid = [row for row in self.onsetResults if row.get("status") == "valid"]
        trace = self.analogTraceData
        if not valid or trace is None:
            axes[0].text(
                0.5, 0.5, "No valid presentation-relative onset results",
                transform=axes[0].transAxes, ha="center", va="center",
            )
            axes[1].text(
                0.5, 0.5, "Review the retained per-repeat exclusion reasons",
                transform=axes[1].transAxes, ha="center", va="center",
            )
            return fig

        samples = np.asarray(trace.data)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        channel = self.detectionConfiguration["dataChannel"]
        if not 0 <= channel < samples.shape[1]:
            raise ValueError("Recorded photodiode channel is outside the trace.")

        validCount = trace.timing["validSampleCount"]
        indices = np.arange(min(len(samples), validCount))
        searchBefore = self.detectionConfiguration["searchBeforeSeconds"]
        searchAfter = self.detectionConfiguration["searchAfterSeconds"]

        for index, row in enumerate(valid):
            # The retained affine mapping makes spacing constant. Center around
            # the saved crossing rather than materializing another host-time array.
            times = (
                row["photodiodeOnsetHostTime"] - row["framePresentationHostTime"]
                + (indices - row["samplePosition"]) * row["sampleSpacingSeconds"]
            )
            offset = (
                row["frameOffsetPresentationHostTime"]
                - row["framePresentationHostTime"]
            )
            windowEnd = max(searchAfter, offset + 0.01)
            mask = (times >= -searchBefore - 0.01) & (times <= windowEnd)
            values = samples[indices[mask], channel].astype(float, copy=True)
            values[(values == -9999) | ~np.isfinite(values)] = np.nan

            axes[0].plot(
                times[mask] * 1000, values,
                alpha=0.45, linewidth=0.9,
                label=f"Repeat {row['repeatIndex'] + 1}" if len(valid) <= 8 else None,
            )
            axes[0].plot(
                row["onsetDelaySeconds"] * 1000,
                row["thresholdVolts"], "o", color="tab:green", markersize=3,
            )
            axes[0].axvline(
                offset * 1000, color="tab:red", alpha=0.15, linewidth=0.8,
                label="Recorded black-frame presentation" if index == 0 else None,
            )

        axes[0].axvline(
            0, color="black", linestyle="--",
            label="Recorded white-frame presentation",
        )
        axes[0].plot(
            [], [], "o", color="tab:green", markersize=4,
            label="Detected threshold crossing",
        )
        axes[0].set(
            title="Raw photodiode traces relative to recorded presentation",
            xlabel="Time relative to white-frame presentation (ms)",
            ylabel="Photodiode signal (V)",
        )

        repeatNumbers = np.asarray([row["repeatIndex"] + 1 for row in valid])
        delays = np.asarray([row["onsetDelaySeconds"] for row in valid])
        liveDelays = np.asarray([row["liveOnsetDelaySeconds"] for row in valid])
        clockLower = np.asarray([
            row["clockErrorIntervalSeconds"][0] for row in valid
        ])
        clockUpper = np.asarray([
            row["clockErrorIntervalSeconds"][1] for row in valid
        ])

        axes[1].errorbar(
            repeatNumbers, delays * 1000,
            yerr=np.vstack([-clockLower, clockUpper]) * 1000,
            fmt="o", capsize=3, color="tab:blue",
            label="Selected mapping ± conditional clock interval",
        )
        axes[1].plot(
            repeatNumbers, liveDelays * 1000, "x",
            color="tab:orange", label="Original live-mapping delay",
        )
        for index, row in enumerate(valid):
            lower, upper = row["sampleBracketDelaySeconds"]
            axes[1].vlines(
                repeatNumbers[index], lower * 1000, upper * 1000,
                color="gray", linewidth=4, alpha=0.35,
                label="Threshold-crossing sample bracket" if index == 0 else None,
            )

        stdText = (
            f"{np.std(delays, ddof=1) * 1000:.3f} ms"
            if len(delays) > 1 else "unavailable"
        )
        axes[1].set(
            title=(
                f"Mean onset delay: {np.mean(delays) * 1000:+.3f} ms | "
                f"Across-repeat SD: {stdText} | n={len(valid)}"
            ),
            xlabel="Repeat (number)",
            ylabel="Photodiode onset minus recorded presentation (ms)",
        )
        for axis in axes:
            axis.grid(True, alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle(
            "Presentation-relative photodiode timing\n"
            "Clock intervals are shared-model uncertainty, not total latency error bars",
            fontsize=11,
        )
        return fig

    def display(self, fig=None):
        """Select the display appropriate to the recording's analysis provenance."""
        if self.analysisMethod == "photodiodeThresholdRelativeToDrawablePresented":
            self.printTemporalTimingReport()
            return self.plotPresentationLatency(fig=fig)
        return self._displayLegacyTemporal(fig=fig)




        
# Calibration settings, subclass of pglSettings to inherit load/save functionality
class pglDisplayLuminanceCalibrationData(pglTraitSettings):
    
    settingsName = Unicode("Default", help="Settings name used to open display")
    displayInfo = Dict(help="Display information at time of calibration")
    settings = Instance(pglSettings, allow_none=True, help="Settings used during calibration") 
    metalInfo = Dict(help="PGL info including display info such as UUID, serial number, and other information at time of calibration")   
    gammaTableSize = Int(-1, help="Size of the gamma table at time of calibration")
    gammaTable = Tuple(Instance(np.ndarray), Instance(np.ndarray), Instance(np.ndarray), allow_none=True, help="Gamma table at time of calibration")
    creationDateTime = Instance(datetime, default_value=datetime.now(), help="Date and time of calibration creation")
    nRepeats = Int(4, help="Number of repeats per calibration value")
    nSteps = Int(256, help="Number of steps in the calibration")
    initValues = Instance(np.ndarray, allow_none=True, help="Initial display values used in calibration, if any, used for finding min and max values")
    initMeasurements = Instance(np.ndarray, allow_none=True, help="Measured luminance values corresponding to initValues calibration")
    calibrationValues = Instance(np.ndarray, allow_none=True, help="Display values used in calibration")
    calibrationMeasurements = Instance(np.ndarray, allow_none=True, help="Measured luminance values from calibration")
    minLuminance = Float(-1.0, help="Minimum measured luminance from calibration")
    maxLuminance = Float(-1.0, help="Maximum measured luminance from calibration")
    gamma = Float(0.0, help="If not 0 then this is a validation run trying to achieve this gamma - i.e. 1.0 = linear")
    units = Unicode("cd/m2", help="Units that were used for measurement")
    deviceDescription = Unicode("Unknown", help="Desciption of measurement device")
    trimThreshold = Float(allow_none=True, default_value=None, help="Threshold to trim flat portions of curves when computing inverse, value of 1 means trim at either end of curve if the values are not increasing by more than 1/nSteps. Defaults to None, try values of less than one and see them with display")
    def getDisplayName(self):
        '''
        Get the display name associated with this data
        '''
        return self.displayInfo.get("DisplayName",self.settingsName)
    
    def getUUID(self):
        '''
        Get the UUID associated with this data
        '''
        return self.metalInfo.get("display.uuid","Unknown")
        
        
    def save(self, filename=None, filepath=None):
        '''
        save
        
        Args:
            filename: If none defaults to calibration.json
            filepath: If none defaults to calibrationDir / display name / data
        '''
        # get a filepath
        if filepath is None:
            filepath = pglSettingsManager.getDisplayLuminanceCalibrationDir(displaySettings=self.settings.displays[0], newCalibration=True)
        
        # get the filename
        if filename is None:
            filename = "calibration"

        pglMessages.print(filepath / filename)
        # call parent to save
        super().save(filepath / filename)
    
    @classmethod
    def load(cls, displayName, filename=None, filepath=None, date=None):
        '''
        load
        
        Args:
            displayName: displayName to load
            filename: If none defaults to calibration.json
            filepath: If none defaults to calibrationDir / display name / data
        '''
        # get a filepath
        if filepath is None:
            filepath = pglDisplayCalibration.chooseCalibrationFilepath(displayName, date=date, calibrationType="luminance")
            if filepath is None: return
        # get the filename
        if filename is None:
            filename = "calibration.json"
        
        pglMessages.print(f"Loading {filepath / filename}")
        
        # call super load to instantiate the object and load it                
        return super(pglDisplayLuminanceCalibrationData, cls).load(filepath / filename)
    
    @classmethod
    def loadMatlab(cls, filename):
        '''
        load a calibration from mgl Matlab file
        
        Args:
            filename: The filename of the .mat calibration file
        '''
        try:
            matData = loadmat(filename, squeeze_me=True, struct_as_record=False)
        except Exception as loadError:
            pglMessages.warning(f"Failed to load {filename}: {loadError}")
            return None

        # check for calib strcuture
        calib = matData.get('calib', None)
        if calib is None:
            pglMessages.warning(f"Could not find calib structure in: {filename}")
            return None
        
        # instantiate the class
        c = cls()
        
        # get creation time
        try:
            c.creationDateTime = datetime.strptime(calib.date, '%d-%b-%Y %H:%M:%S')
        except (AttributeError, ValueError) as dateError:
            pglMessages.warning(f"Could not parse date: {dateError}")
            c.creationDateTime = None
            
        # set num repeats, because we only have the stored median value, always set to 1
        c.nRepeats = 1
        
        try:
            c.calibrationValues = np.array(calib.uncorrected.outputValues).T
            c.calibrationMeasurements = np.array(calib.uncorrected.luminance).T
            c.nSteps = len(c.calibrationValues)
        except (AttributeError, ValueError) as e:
            pglMessages.warning(f"Could not parse calibration: {e}")
            return None
        
        # get min and max luminance
        c.minLuminance = np.min(c.calibrationMeasurements)
        c.maxLuminance = np.max(c.calibrationMeasurements)

        # get the monitor ID and set as DisplayName  
        try:
            c.displayInfo["DisplayName"] = calib.monitor.ID
        except (AttributeError, ValueError) as e:
            pglMessages.warning(f"Could not parse monitor ID: {e}")
            
        c.deviceDescription = "Import from matlab"
        return(c,calib) 
    
    def attachSettings(self, settingsName):
        '''
        attach settings to the data (useful for import from matlab)
        
        Args:
            pgl: pgl instance
            settingsName: name of settings to attach to these data
        '''
           
        # set information
        self.settingsName = settingsName
        self.settings = pglSettingsManager.getSettings(settingsName)
        
         
    def print(self, verbose=False):
        '''
        print the calibration data in a readable format.
        
        Args:
            verbose (bool): If True, print detailed information.
        '''   
        pglMessages.printHeader()
        pglMessages.print (f"Calibration Data for settings: {self.settingsName}")
        if self.gammaTableSize > 0:
            pglMessages.print (f"Gamma Table Size: {self.gammaTableSize}")
        pglMessages.print (f"Creation Date and Time: {self.creationDateTime}")
        pglMessages.print (f"Number of Repeats: {self.nRepeats} number of Steps: {self.nSteps}")
        if verbose:
            values, measurements, minMeasurements, maxMeasurements = self.getMedianMeasurements()
            for iStep in range(self.nSteps):
                pglMessages.print(f"Step {iStep:3d}/{self.nSteps}: {values[iStep]:<5.3f} = median: {measurements[iStep]:<7.3f} (min: {minMeasurements[iStep]:<7.3f}, max: {maxMeasurements[iStep]:<7.3f}, percent difference: {((maxMeasurements[iStep]-minMeasurements[iStep])/measurements[iStep]*100):<5.3f}%)")
            # compute max difference between max and min in percentage
            maxDiff = np.max(np.abs(maxMeasurements - minMeasurements)/measurements * 100)
            pglMessages.print(f"Maximum difference between max and min measurements: {maxDiff:.3f}%")
        if verbose > 1:
            super().print()
            
    def displayInverse(self, gamma=1.0, fig=None):
        '''
        display a graph of the inverse table
        
        Args:
            gamma (float): Gamma to achieve (default 1.0)
            fig (matplotlib.fig, optional): If provided, plots into this figure
        '''
        # use the passed-in axis, or create a new figure/axis if none given
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = fig.add_subplot(111)

        inverseTable = self.calculateInverseGamma(gamma=gamma, gammaTableSize=1024, trimThreshold=self.trimThreshold)
        values = np.linspace(0, 1, 1024)
        ax.plot(values, inverseTable[0], 'o--', label='Median', color='green', markeredgecolor='green', markersize=2)

        ax.legend()
        ax.set_xlabel("Display Value (normalized 0-1)")
        ax.set_ylabel(f"Table value (normalized 0-1)")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        ax.set_title(f"{self.getDisplayName()} UUID: {self.getUUID()}\n{self.deviceDescription}: {self.creationDateTime}\nnRepeats: {self.nRepeats} nSteps: {self.nSteps} min = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nInverse table for gamma: {gamma if gamma is not None else 'N/A'}")

        ax.grid(True)

        plt.show()        
        
    def display(self, gamma=None, fig=None,):
        '''
        display graph of calibration data
        Args:
            gamma (float, optional): If provided, display the ideal gamma curve (1.0 or 2.2 or whatever) for comparison.
            fig (matplotlib.fig, optional): If provided, plot into this figure instead of creating a new figure.
            inverseGamma (float): If not None, will plot the inverse table to achieve the input value
        '''
        if self.calibrationValues is None or self.calibrationMeasurements is None:
            pglMessages.message("No calibration data to display.")
            return

        # set the ideal gamma if we have one saved
        if gamma is None and self.gamma != 0:
            gamma = self.gamma

        # use the passed-in axis, or create a new figure/axis if none given
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = fig.add_subplot(111)
        # plot raw data points
        ax.plot(self.calibrationValues, self.calibrationMeasurements, '.')

        # plot median
        values, measurements, minMeasurements, maxMeasurements = self.getMedianMeasurements()
        if gamma is not None:
            # plot points without line
            ax.plot(values, measurements, 'o', label='Median', color='black', markeredgecolor='white', markersize=8)
            # plot ideal gamma curve
            idealMeasurements = np.power(values, gamma)
            idealMeasurements = idealMeasurements * (self.maxLuminance - self.minLuminance) + self.minLuminance
            ax.plot(values, idealMeasurements, 'r--', label=f'Ideal Gamma {gamma}')
        else:
            ax.plot(values, measurements, 'o-', label='Median', color='black', markeredgecolor='white', markersize=8)
            
        if self.trimThreshold:
            (trimX, trimY) = self.trimFlatRegions(x=values,y=measurements,trimThreshold=self.trimThreshold)
            # plot vertical lines at trimThreshold values
            ax.axvline(x=trimX[0], color='red', linestyle='--', label='trim border for inverese')
            ax.axvline(x=trimX[-1], color='red', linestyle='--', label='trim border for inverese')
            
            
        ax.legend()
        ax.set_xlabel("Display Value (normalized 0-1)")
        ax.set_ylabel(f"Measured Luminance ({self.units})")

        if gamma is None:
            ax.set_title(f"{self.getDisplayName()} UUID: {self.getUUID()}\n{self.deviceDescription}: {self.creationDateTime}\nnRepeats: {self.nRepeats} nSteps: {self.nSteps} min = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nCalibration data")
        else:
            ax.set_title(f"{self.getDisplayName()} UUID: {self.getUUID()}\n{self.deviceDescription}: {self.creationDateTime}\nnRepeats: {self.nRepeats} nSteps: {self.nSteps} min = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nValidation for gamma: {gamma if gamma is not None else 'N/A'}")

        ax.grid(True)

        plt.show()        
        
    def appendValue(self, value, mode="calibration"):
        '''
        Append a calibration value.
        
        Args:
            value (float): The display value to append.
            mode (str): "init" to append to initial values, "calibration" to append to calibration values.
        '''
        if mode == "init":
            if self.initValues is None:
                self.initValues = np.array([value])
            else:
                self.initValues = np.append(self.initValues, value)
        else:
            if self.calibrationValues is None:
                self.calibrationValues = np.array([value])
            else:
                self.calibrationValues = np.append(self.calibrationValues, value)

    def getValues(self, startIndex, endIndex=None, mode="calibration"):
        '''
        Get calibration value(s).
        
        Args:
            index (int or slice): Index or slice of values to get.
            mode (str): "init" to get from initial values, "calibration" to get from calibration values.
            
        Returns:
            float or np.ndarray: The requested calibration value(s).
        '''
        if mode == "init":
            if self.initValues is None:
                return None
            if endIndex is None:
                return self.initValues[startIndex]
            else:                
                return self.initValues[startIndex:endIndex]
        else:
            if self.calibrationValues is None:
                return None
            if endIndex is None:
                return self.calibrationValues[startIndex]
            else:
                return self.calibrationValues[startIndex:endIndex]
            
    def getMedianMeasurements(self):
        '''
        Get the median of the calibration measurements for each step.
        
        Returns:
            tuple: A tuple containing four numpy arrays: (medianValues, medianMeasurements, min_measurements, max_measurements)
        '''
        
        if self.calibrationMeasurements is None:
            return None
        
        values = np.zeros(self.nSteps)
        measurements = np.zeros(self.nSteps)
        minMeasurement = np.zeros(self.nSteps)
        maxMeasurement = np.zeros(self.nSteps)
        for iStep in range(self.nSteps):
            # get value and measurements
            value = self.calibrationValues[iStep*self.nRepeats:(iStep+1)*self.nRepeats] if self.calibrationValues is not None else None
            measurement = self.calibrationMeasurements[iStep*self.nRepeats:(iStep+1)*self.nRepeats] if self.calibrationMeasurements is not None else None
            # compute median if both value and measurement are available
            if value is not None and measurement is not None:
                values[iStep] = np.median(value)
                measurements[iStep] = np.median(measurement)
                # compute min and max
                minMeasurement[iStep] = np.min(measurement)
                maxMeasurement[iStep] = np.max(measurement)
        
        return (values, measurements, minMeasurement, maxMeasurement)

    def getMeasurements(self, startIndex, endIndex=None,mode="calibration"):
        '''
        Get calibration measurement(s).
        
        Args:
            index (int or slice): Index or slice of measurements to get.
            mode (str): "init" to get from initial measurements, "calibration" to get from calibration measurements.
            
        Returns:
            float or np.ndarray: The requested calibration measurement(s).
        '''
        if mode == "init":
            if self.initMeasurements is None:
                return None
            if endIndex is None:
                return self.initMeasurements[startIndex]
            else:                
                return self.initMeasurements[startIndex:endIndex]
        else:
            if self.calibrationMeasurements is None:
                return None
            if endIndex is None:
                return self.calibrationMeasurements[startIndex]
            else:
                return self.calibrationMeasurements[startIndex:endIndex]
 
    def appendMeasurement(self, measurement, mode="calibration"):
        '''
        Append a calibration measurement.
        
        Args:
            measurement (float): The measured luminance value to append.
            mode (str): "init" to append to initial measurements, "calibration" to append to calibration measurements.
        '''
        if mode == "init":
            if self.initMeasurements is None:
                self.initMeasurements = np.array([measurement])
            else:
                self.initMeasurements = np.append(self.initMeasurements, measurement)
        else:
            if self.calibrationMeasurements is None:
                self.calibrationMeasurements = np.array([measurement])
            else:
                self.calibrationMeasurements = np.append(self.calibrationMeasurements, measurement)
                
    def hasLastMeasurement(self, mode="calibration"):
        '''
        Check if there is a last measurement.
        
        Args:
            mode (str): "init" to check initial measurements, "calibration" to check calibration measurements.
            
        Returns:
            bool: True if there is a measurement for each value, False otherwise.
        '''
        if mode == "init":
            # If no values exist yet, return True (nothing missing)
            if self.initValues is None:
                return True
            # If values exist but no measurements, return False
            if self.initMeasurements is None:
                return False
            # Check if counts match
            return len(self.initMeasurements) == len(self.initValues)
        else:
            if self.calibrationValues is None:
                return True
            if self.calibrationMeasurements is None:
                return False
            return len(self.calibrationMeasurements) == len(self.calibrationValues)       

    def getLastValue(self, mode="calibration"):
        '''
        Get the last calibration value.
        
        Args:
            mode (str): "init" to get from initial values, "calibration" to get from calibration values.
            
        Returns:
            float: The last calibration value.
        '''
        if mode == "init":
            if self.initValues is None or len(self.initValues) == 0:
                return None
            return self.initValues[-1]
        else:
            if self.calibrationValues is None or len(self.calibrationValues) == 0:
                return None
            return self.calibrationValues[-1]
        
    def calculateInverseGamma(self, gamma = 1.0, gammaTableSize=None, trimThreshold=None):
        '''
        Calculate inverse gamma table from calibration measurements.
        
        Args:
            gamma: The target gamma value for inverse (default is 1.0 = linear table).
            trimThreshold: If not None, then will trim flat ends of curve, set to 1 to
              use a threshold in which each step has to increase the luminance by 1/nSteps
              
        
        Returns:
            Tuple of three numpy arrays (R, G, B) for the inverse gamma table.
        '''
        # Average the repeated measurements for each step
        nSteps = self.nSteps
        nRepeats = self.nRepeats
        
        # Reshape and average
        calValues = np.array(self.calibrationValues).reshape(nSteps, nRepeats)
        calMeasurements = np.array(self.calibrationMeasurements).reshape(nSteps, nRepeats)
        
        medianValues = np.median(calValues, axis=1)
        medianMeasurements = np.median(calMeasurements, axis=1)

        if trimThreshold:
            (medianValues, medianMeasurements) = self.trimFlatRegions(medianValues, medianMeasurements, trimThreshold)
        
        # Normalize measurements to 0-1 range
        minLum = np.min(medianMeasurements)
        maxLum = np.max(medianMeasurements)
        normalizedMeasurements = (medianMeasurements - minLum) / (maxLum - minLum)
        
        # Create interpolation function: maps desired linear output to required input
        # We want: given a desired output level, what input do we need?
        #interpFunc = interp1d(normalizedMeasurements, medianValues, 
        #                    kind='cubic', 
        #                    bounds_error=False, 
        #                    fill_value='extrapolate')
        
        
        # ensure strictly increasing x values before interpolation
        sortIdx = np.argsort(normalizedMeasurements)
        normalizedMeasurements = normalizedMeasurements[sortIdx]
        medianValues = medianValues[sortIdx]

        # drop points that aren't strictly increasing (duplicates / flattened top)
        mask = np.concatenate(([True], np.diff(normalizedMeasurements) > 1e-6))
        normalizedMeasurements = normalizedMeasurements[mask]
        medianValues = medianValues[mask]
        
        interpFunc = PchipInterpolator(normalizedMeasurements, medianValues, extrapolate=True)
        
        # Create inverse gamma table
        if gammaTableSize is None: gammaTableSize = self.gammaTableSize
        linearOutput = np.linspace(0, 1, gammaTableSize)
        gammaOutput = np.power(linearOutput, gamma)
        inverseGamma = interpFunc(gammaOutput)
        
        # Clip to valid range [0, 1] and convert to float32
        inverseGamma = np.clip(inverseGamma, 0, 1).astype(np.float32)
        
        # For RGB, use the same correction for all channels (can be modified for per-channel)
        return (inverseGamma, inverseGamma.copy(), inverseGamma.copy())
    
    def trimFlatRegions(self, x, y, trimThreshold=0.5):
        """
        Trim flat regions from the start and end of a monotonic curve.
        
        Args:
            x: input values
            y: measured luminance values
            trimThreshold: minimum fraction of the x-range that must be covered
                            per step to not be considered "flat"
        
        Returns:
            trimmed x, y arrays
        """
        yRange = y.max() - y.min()
        xSteps = len(x)
        minStep = trimThreshold * (yRange / xSteps)
        #print(f"yRange: {yRange} xSteps: {xSteps} minStep: {minStep}")

        # find where the curve starts rising meaningfully from the bottom
        dy = np.diff(y)
        startIdx = 0
        for i in range(len(dy)):
            if dy[i] > minStep:
                startIdx = i
                break

        # find where the curve stops rising meaningfully at the top
        endIdx = len(y) - 1
        for i in range(len(dy) - 1, -1, -1):
            #print(f"i={i} dy[i]={dy[i]}")
            if dy[i] > minStep:
                endIdx = i + 1
                break
        
        #print(f"{x[startIdx]}:{x[endIdx]}")
        return x[startIdx:endIdx+1], y[startIdx:endIdx+1]
    def setDisplayToGamma(self, pgl, display, gamma = 1.0):
        '''
        Uses the saved gamma calibration data to set the display to the requested gamma
        
        Args:
            display (pglDisplaySettings): The display that the gamma is being set on
            gamma (Float): The gamma to achieve: 1.0 = linear (default), 2.2 = industry standard for natural iamges, 0 = do nothing
        '''
        if gamma == 0.0: return
        
        # set the gamma table size
        self.gammaTableSize = display.gammaTableSize

        inverseGammaTable = self.calculateInverseGamma(gamma, self.gammaTableSize)
        if inverseGammaTable == None:
            pglMessages.warning(f"Not able to compute gamma table for display {display.displayName}")
            return
        
        # check display Num
        if display.currentDisplayNum == -1:
            pglMessages.warning(f"Display {display.displayName} is not currently connected, cannot set gamma table")
            return

        # set the gamma table
        pgl.setGammaTable(display.currentDisplayNum, rgbGammaTable=inverseGammaTable)
        
        

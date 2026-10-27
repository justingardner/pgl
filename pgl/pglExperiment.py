################################################################
#   filename: pglExperiment.py
#    purpose: Experiment class which handles timing, parameter randomization
#             subject response, synchronizing with measurement hardware, as well
#             as saving experimental data
#         by: JLG
#       date: September 2, 2025
################################################################

#############
# Import modules
#############
from datetime import date as Date, datetime
from IPython.display import clear_output
import numpy as np
import random
import math
from dataclasses import dataclass, field
from pathlib import Path
from IPython.display import display, HTML
import ipywidgets as widgets
from traitlets import Float, TraitError, TraitError, observe, Instance, Int, Unicode, Dict, validate, Bool, Tuple
from .pglParameter import pglParameter, pglParameterBlock
from .pglEvent import pglEvent
from .pglSerialize import pglSerialize
from traitlets import List
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
from enum import Enum
from .pglTimestamp import pglTimestamp
from .pglSettings import pglDisplaySettings, pglTraitSettings, pglStateDataSettings
from .pglMessages import pglMessages
import fsspec
import posixpath
from .pglStorage import pglStorage
from .pglDialog import pglDialogs
from types import SimpleNamespace
from .pglDevice import pglDigitalIODevice

#######################
# for returning stats
#######################
@dataclass
class Stats:
    mean: float
    median: float
    std: float
    min: float
    max: float

##############################################s
# Experiment base class
##############################################
class pglExperimentBase(pglStateDataSettings):
    '''
    Base class for pglExperiment which runs experiments
    and pglExperimentAnalysis which is used for loading and
    analyzing experimental data. This class handles loading
    and saving settings, state and data.
    '''
    def __init__(self):
        # initialize variables
        self.settings = None
        self.state = None
        self.data = None
        self.experimentSettings = None
        self.pgl = None
        self.eyeTracker = None
        self.nTasks = 0
    
    @classmethod
    def isValidExperimentPath(cls, verbose=False, fullDataPath=None, settings=None,
                            settingsName=None, experimentName=None, subjectID=None,
                            sessionName=None, runName=None, filesystem=None,
                            filesystemPrefix=None, dataPath=None):
        """Check required experiment/task files, allowing an optional recording."""
        from .pglChoose import pglChoose

        filesystem, fullDataPath, _ = pglChoose.getExperimentPath(
            fullDataPath=fullDataPath,
            settings=settings,
            settingsName=settingsName,
            experimentName=experimentName,
            subjectID=subjectID,
            sessionName=sessionName,
            runName=runName,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
            dataPath=dataPath,
        )
        if filesystem is None: return False

        sep = filesystem.sep
        requiredExperimentFiles = ["data.json", "experimentSettings.json", "settings.json", "pgl.json", "state.json"]
        requiredFiles = ["data.json", "settings.json", "state.json"]

        # Check top-level required JSON files.
        for fileName in requiredExperimentFiles:
            filePath = f"{fullDataPath}{sep}{fileName}"
            if not filesystem.isfile(filePath):
                if verbose: pglMessages.message(f"Missing file: {filePath}")
                return False

        # Permit the explicitly named recording artifact alongside task dirs.
        for item in filesystem.ls(fullDataPath, detail=False):
            fileName = item.rsplit(sep, 1)[-1]
            if fileName.endswith(".json"): continue
            if fileName == "recording.h5" and filesystem.isfile(item): continue

            if not filesystem.isdir(item):
                if verbose: pglMessages.message(f"Unexpected non-directory artifact: {item}")
                return False

            # Check task files only if any task component exists.
            taskFilesFound = [name for name in requiredFiles if filesystem.isfile(f"{item}{sep}{name}")]
            if not taskFilesFound: continue

            for name in requiredFiles:
                filePath = f"{item}{sep}{name}"
                if not filesystem.isfile(filePath):
                    if verbose: pglMessages.message(f"Missing file in task directory: {filePath}")
                    return False

            # Parameters directory may be empty.
            parametersPath = f"{item}{sep}parameters"
            if not filesystem.isdir(parametersPath):
                if verbose: pglMessages.message(f"No parameters directory in {item}")
                return False

            for subItem in filesystem.ls(parametersPath, detail=False):
                if not filesystem.isdir(subItem): continue
                for name in requiredFiles:
                    filePath = f"{subItem}{sep}{name}"
                    if not filesystem.isfile(filePath):
                        if verbose: pglMessages.message(f"Missing parameter file: {filePath}")
                        return False

        return True

    @classmethod
    def load(cls, settings=None, dataPath=None, experimentName=None, subjectID=None, sessionName=None, runName=None, fullDataPath=None, filesystem=None, filesystemPrefix=None):
        '''
        Load the experiment settings, state and data.         
        '''
        # get the experiment directory
        from .pglChoose import pglChoose
        filesystem, experimentPath, filesystemPrefix = pglChoose.getExperimentPath(settings=settings, dataPath=dataPath, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, fullDataPath=fullDataPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix)
        if filesystem is None:
            if dataPath is not None:
                pglMessages.message(f"Could not locate experiment directory {dataPath}")
            return
 
        # call parent class to load the experiment data, settings, and state
        pglMessages.message(f"(pglExperiment:load) Loading experimentdata from: {experimentPath}")
        obj = super().load(dataPath=experimentPath, filesystem=filesystem,filesystemPrefix=filesystemPrefix, loadAsClass=cls)
        if obj is None:
            if experimentPath is not None:
                pglMessages.warning(f"Unable to load {experimentPath}")
            return None

        # keep where we were loaded from
        obj.filesystem = filesystem
        obj.fullDataPath = experimentPath
        obj.filesystemPrefix = filesystemPrefix
        
        # if we have
        # load experiment settings
        experimentSettingsPath = f"{experimentPath}{filesystem.sep}experimentSettings.json"
        if filesystem.exists(experimentSettingsPath):
            obj.experimentSettings = pglSerialize.load(experimentSettingsPath, filesystem=filesystem)
            if obj.experimentSettings is None:
                pglMessages.warning(f"Could not load experiment settings for {experimentSettingsPath}")
        else:
            pglMessages.warning(f"No experimentSettings.json found for: {experimentSettingsPath}",level=1)
        
        # load pgl state
        pglStatePath = f"{experimentPath}{filesystem.sep}pgl.json"
        if filesystem.exists(pglStatePath):
            obj.pglState = pglSerialize.load(pglStatePath, filesystem=filesystem)
            if obj.pglState is None:
                pglMessages.warning(f"Could not load pgl state for {pglStatePath}")
                return None
        else:
            pglMessages.warning(f"No pgl.json found for: {pglStatePath}", level=1)            
            
        # load all the tasks
        if obj.experimentSettings:
            # initialize tasks, this is only relevant for pglRun
            # becuase it lazy-loads tasks, and this will prevent
            # it from trying to do so
            if hasattr(obj,'_tasks'): obj._tasks = []
            
            for iTask, taskName in enumerate(obj.experimentSettings.tasks):
                # get the task directory
                taskDirName = pglTask.getTaskDirectoryName(iTask,taskName)
                taskPath = f"{experimentPath}{filesystem.sep}{taskDirName}"

                # load the task data
                if filesystem.isdir(taskPath):
                    # load the task
                    task = pglTask.load(dataPath=taskPath, filesystem=filesystem)
                
                    if task is None:
                        pglMessages.warning(f"Could not load task {taskName}", level=1)
                    else:
                        # add the task to the experiment
                        obj.addTask(task, addToTaskList=False)
                        pass
                else:
                    pglMessages.warning(f"Could not find task {taskName}: {taskPath}")
        
        # return the created object
        return obj
       
    def addTask(self, task, addToTaskList=True, addPhase=False):
        '''
        Add a task to the experiment.
        
        Args:
            addPhase (bool): If True, the task will be added as a new phase. If False, it will be added to the current phase.
        '''
        # give it a reference to pgl and experiment
        task.pgl = self.pgl
        task.e = self
        self.nTasks += 1
        task.settings.taskID = self.nTasks

        # if we want to add the task as a new phase
        if addPhase and self.tasks is not None:            
            maxPhaseNum = max(
                (task.settings.phaseNum for task in self.tasks
                if task.settings.phaseNum is not None),
                default=-1
            )
            task.settings.phaseNum = maxPhaseNum + 1

        # set whether to save eye tracker info
        if self.eyeTracker is not None:
            task.settings.saveEyeTracker = True

        # add the task
        self.tasks.append(task)
        
        # save in experimentSettings
        if addToTaskList and task.settings is not None:
            self.experimentSettings.tasks.append(task.settings.taskSaveName)

    def display(self):
        '''
        Display a timeline of experiment events.
        '''
        # display experiment data
        self.data.display(self)
        
        # display task data
        if hasattr(self, "tasks"):
            for task in self.tasks:
                task.display()   

    def print(self, verbose=True):
        '''
        Print a summary of the experiment events.
        Warn and skip summaries whose required contents could not be loaded.
        '''
        from .pglTimestamp import pglTimestamp
        timestamp = pglTimestamp()
        runPath = getattr(self, "fullDataPath", "(unknown run path)")

        # check required experiment contents before accessing their fields
        experimentSettings = self.experimentSettings
        if experimentSettings is None:
            pglMessages.warning(f"Skipping experiment summary for {runPath!r}: experiment settings could not be loaded.")
            return

        # print experiment identifiers even if the data is missing
        pglMessages.print(f"Experiment: {experimentSettings.experimentName} | Subject ID: {experimentSettings.subjectID}")
        pglMessages.print(f"Session: {experimentSettings.sessionName} | Run: {experimentSettings.runName}")

        data = self.data
        if data is None:
            pglMessages.warning(f"Skipping remaining experiment summary for {runPath!r}: experiment data could not be loaded.")
            return

        if data.startDateTime:
            pglMessages.print(f"{data.startDateTime}")
        pglMessages.print(f"Duration: {timestamp.formatDuration(self.experimentDuration())}")

        numVols = data.getNumEvents(type="volumeTrigger")
        if numVols != 0:
            pglMessages.print(f"Number of volume triggers: {numVols}")
        if numVols > 1:
            triggerStats = data.getTriggerStats()
            pglMessages.print(f"Median time between triggers: {triggerStats.median:.3f}s")
            pglMessages.print(f"Mean ± std time between triggers: {triggerStats.mean:.3f} ± {triggerStats.std:.6f}s")

        # accessing the tasks property may trigger loading
        try:
            tasks = getattr(self, "tasks", [])
        except RuntimeError as error:
            pglMessages.warning(f"Skipping task summary for {runPath!r}: {error}")
            return

        if tasks is None:
            pglMessages.warning(f"Skipping task summary for {runPath!r}: tasks are unavailable.")
            return

        # print task data
        for task in tasks:
            pglMessages.print(f"taskName: {task.settings.taskName} (nTrials={task.data.nTrials})")
            if verbose:
                pglMessages.printHeader()
                task.print()
                
    def experimentDuration(self,data=None):
        '''
        Return the total time of the experiment in seconds.
        '''
        # work on passed in data or self data
        if data is None:
            data = self.data
        # check for no data
        if data is None or data.startTime is None or data.endTime is None:
            return 0
        # check to see if this has volumes recorded
        if data.getNumEvents(type="volumeTrigger") > 1:        
            # get the timestamps of the first and last volume triggers
            volumeTimestamps = [e.timestamp for e in data.events if e.type == "volumeTrigger"]
            volumeTR = np.median(np.diff(volumeTimestamps))
            # return the difference between the first and last timestamp
            # because the experiment type as recorded by endTime and startTIme
            # will typically record longer until the experimenter pressed the ESC key to end
            # add one TR to account for the last volume trigger
            return volumeTimestamps[-1] - volumeTimestamps[0] + volumeTR
        else:
            # return timestamps for end compared to start
            return data.endTime - data.startTime

    def getNearestVolumeTrigger(self, event=None, direction='nearest'):
        '''
        Find the nearest volume trigger to a given event.
        
        Args:
            event: The event to find the nearest volume trigger for
            direction: 'nearest' (default), 'before', or 'after'
        
        Returns:
            int: volume_number (starting at 1) or None if not found
        '''
        if event is None:
            return None
        
        # Get only the volume triggers and number them sequentially
        volumeTriggers = [(i + 1, e.timestamp) for i, e in 
                        enumerate([e for e in self.data.events if e.type == "volumeTrigger"])]
        
        if not volumeTriggers:
            return None
        
        if direction == 'before':
            # Find closest timestamp before the event
            beforeTriggers = [vt for vt in volumeTriggers if vt[1] <= event.timestamp]
            if not beforeTriggers:
                return None
            volumeNumber, nearestTimestamp = max(beforeTriggers, key=lambda x: x[1])
        elif direction == 'after':
            # Find closest timestamp after the event
            afterTriggers = [vt for vt in volumeTriggers if vt[1] >= event.timestamp]
            if not afterTriggers:
                return None
            volumeNumber, nearestTimestamp = min(afterTriggers, key=lambda x: x[1])
        else:  # direction == 'nearest' (default)
            # Find closest timestamp in either direction
            volumeNumber, nearestTimestamp = min(volumeTriggers, key=lambda x: abs(x[1] - event.timestamp))
        
        return volumeNumber
    
##############################################s
# Experiment class
##############################################
class pglExperiment(pglExperimentBase):
    '''
    Experiment class which handles timing, parameter randomization,
    subject response, synchronizing with measurement hardware etc
    '''
    def __init__(self, pgl=None, settingsName=None, settings=None, displayName=None, displaySettings=None, subjectID="s0000", experimentName=None, sessionName=None, runName=None):
        '''
        Initialize the pglExperiment class.
        
        Args:
            pgl (pgl): An instance of the pgl class.
            settingsName (str): The name of the settings to use. If not set (and settings not set), will use default settings
            settings (pglSettings): An instance of the pglSettings class. If set, will supersede settingsName.
            displayName (str): The name of the display to use. If set, will be incorporated into settings (and supersede any
                conflicting settings). If there is no settings/settingsName will use default settings
            displaySettings (pglDisplaySettings): The settings of the dispaly to use, will supersed the displayName if set and
                behave in a similar fashion
            subjectID (str): The identifier for the subject participating in the experiment.
        '''
        try:
            # init super
            super().__init__()
            self.tasks = []

            # clear the text screen
            #clear_output(wait=True)
        
            self.isInitialized=False

            if not pgl:
                pglMessages.warning("Need to pass in valid pgl")
                return
            
            # save pgl
            self.pgl = pgl

            # load settings
            self.settings = pgl.getSettings(settingsName=settingsName, settings=settings, displaySettings=displaySettings, displayName=displayName)
            if self.settings is None: return

            # initialize experiment state and data
            self.state = pglExperimentState()
            self.data = pglExperimentData()
            
            # get experiment settings
            self.experimentSettings = pglExperimentSettings()
            if experimentName:
                self.experimentSettings.experimentName = experimentName
            if sessionName:
                self.experimentSettings.sessionName = sessionName
            else:
                # default session name is session_YYYY-MM-DD
                self.experimentSettings.sessionName = f"session_{datetime.now().strftime("%Y-%m-%d")}"
            if runName:
                self.experimentSettings.runName = runName
            else:
                # default run name is run_HH-MM-DD
                self.experimentSettings.runName = f"run_{datetime.now().strftime("%H-%M-%S")}"
                
            self.experimentSettings.subjectID = subjectID
            self.isInitialized=True
            
            # set to flush screen
            self.flush = True

        except Exception as e:
            pglMessages.warning(f"Could not initialize experiment. Error {type(e).__name__}: {e}")    
            return

        
    def __repr__(self):
        return f"<pglExperiment: {len(self.task)} phases>"
    
    def initScreen(self, backgroundColor=-1, recordFlush=True, recordCommands=False, estimatedDuration=600):
        '''
        Initialize the screen, start recording, and configure tasks.

        Args:
            backgroundColor: RGB background color; -1 uses settings.
            recordFlush: Retain flush presentation timing.
            recordCommands: Begin command buffering immediately.
            estimatedDuration: Initial recording capacity estimate in seconds.

        Movie timing is retained independently of the two recording flags.
        Existing recordings must be explicitly saved/exported and cleared.
        '''
        from .pglKeyboardMouse import pglKeyboardMouse

        if self.settings is None: raise RuntimeError("No settings found to open screen.")
        if not isinstance(recordFlush, bool) or not isinstance(recordCommands, bool):
            raise TypeError("recordFlush and recordCommands must be Boolean.")
        if isinstance(estimatedDuration, (bool, np.bool_)):
            raise ValueError("estimatedDuration must be a positive number of seconds.")
        estimatedDuration = float(estimatedDuration)
        if not math.isfinite(estimatedDuration) or estimatedDuration <= 0:
            raise ValueError("estimatedDuration must be finite and positive.")

        # Check before screen cleanup: never discard or disturb an old session.
        if getattr(self, "_recordingId", None) is not None:
            raise RuntimeError("This experiment already owns a recording. Create a new experiment for another run.")
        if getattr(self.pgl, "_recordingState", None) in ("active", "stopped"):
            raise RuntimeError("Save/export the previous recording, then call pgl.recordingClear().")
        if self.pgl.commandRecording or self.pgl.flushRecording:
            raise RuntimeError("Stop and export existing capture before initializing this experiment.")
        if self.pgl.commandRecordingMetadata is not None or self.pgl._presentationLog is not None:
            raise RuntimeError("Export existing recording data, then call pgl.recordingClear().")
        if getattr(self.pgl, "_moviePlaybacks", []) or getattr(self.pgl, "_movieDrawSamples", []):
            raise RuntimeError("Export existing movie data, then call pgl.recordingClear().")

        # Initialize recovery bookkeeping before task configuration can fail.
        self._runSavePath = None
        self._runStarted = False
        self._recordingOutcome = "error"
        self.data.runErrors = []
        self.data.saveStatus = "notAttempted"
        self.data.cleanupStatus = "notAttempted"
        self.data.eyeTrackerSaveStatus = "notAttempted"
        self.state.openScreen = False

        # Get background color. Only the scalar sentinel means "use settings".
        if isinstance(backgroundColor, (int, float)) and backgroundColor == -1:
            backgroundColor = self.settings.backgroundColor

        try:
            # Get screen parameters.
            if len(self.settings.displays) < 1:
                raise RuntimeError(f"Settings {self.settings.name} is not associated with a display.")
            self.state.display = self.settings.displays[0]

            # Make PGL quieter.
            self.setVerbose()

            # Preserve the existing screen-initialization cleanup behavior.
            self.pgl.cleanUp()

            # Save original screen resolution before changing it.
            self.state.originalScreenResolution = self.pgl.getResolution(self.state.display.currentDisplayNum)

            if self.state.display.uuid == "windowed":
                # Open the screen.
                self.pgl.open(0, screenWidth=self.state.display.windowSize[0], screenHeight=self.state.display.windowSize[1], backgroundColor=backgroundColor)
                if not self.pgl.isOpen(): raise RuntimeError("Failed to open windowed screen.")
                self.pgl.setWindowFrameInDisplay(
                    0,
                    screenX=self.state.display.windowPosition[0],
                    screenY=self.state.display.windowPosition[1],
                    screenWidth=self.state.display.windowSize[0],
                    screenHeight=self.state.display.windowSize[1],
                )
            else:
                if self.state.display.currentDisplayNum == -1:
                    raise RuntimeError(f"Display {self.state.display.name} is not connected.")

                # Compare the current display mode with the desired mode.
                if self.state.display.displayModes:
                    displayMode = self.state.display.displayModes[0]
                    if displayMode == self.state.originalScreenResolution:
                        pglMessages.message(f"Display mode settings match current mode: {self.state.originalScreenResolution}")
                    else:
                        self.pgl.setResolutionUsingDisplayModeSettings(self.state.display.currentDisplayNum, displayMode)
                        self.state.screenResolution = self.pgl.getResolution()
                        pglMessages.message(
                            f"Changing screen resolution to: {self.state.screenResolution[0]} x "
                            f"{self.state.screenResolution[1]} {self.state.screenResolution[2]}Hz "
                            f"{self.state.screenResolution[3]}bits from: "
                            f"{self.state.originalScreenResolution[0]} x "
                            f"{self.state.originalScreenResolution[1]} "
                            f"{self.state.originalScreenResolution[2]}Hz "
                            f"{self.state.originalScreenResolution[3]}bits"
                        )

                # Open the screen.
                self.pgl.open(whichScreen=self.state.display.currentDisplayNum, backgroundColor=backgroundColor)

            if not self.pgl.isOpen(): raise RuntimeError("Failed to open screen.")

            # Set visual angle coordinates.
            self.pgl.visualAngle(self.state.display.displayDistance, self.state.display.displaySize[0], self.state.display.displaySize[1])

            # Flip left-right and/or up-down if specified in settings.
            if self.state.display.flipLeftRight: self.pgl.flipLeftRight()
            if self.state.display.flipUpDown: self.pgl.flipUpDown()

            # Set the gamma table.
            self.setGamma(self.settings, self.state.display)

            # Add a keyboard device if not already loaded.
            keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
            if not keyboardDevices:
                keyboardMouse = pglKeyboardMouse(eatKeys=None)
                self.pgl.devicesAdd(keyboardMouse)

                # Check whether the keyboard listener is running.
                if not keyboardMouse.isRunning():
                    pglMessages.warning(
                        "Accessibility permission not granted for keyboard/mouse access.\n"
                        "On macOS, enable your terminal or application in Privacy & Security > Accessibility."
                    )
                    raise RuntimeError("Keyboard/mouse listener could not start.")
            else:
                # If already loaded, use the existing keyboard device.
                keyboardMouse = keyboardDevices[0]
                if not keyboardMouse.isRunning(): keyboardMouse.start()

            # Clear pending keyboard/mouse events and retain the device.
            keyboardMouse.clear()
            self.keyboardMouse = keyboardMouse

            # A comma-separated response list supports keys such as space/F1.
            # Without commas, each character specifies a response key.
            if ',' in self.settings.responseKeys:
                self.responseKeysList = [k.strip() for k in self.settings.responseKeys.split(',')]
            else:
                self.responseKeysList = list(self.settings.responseKeys)

            # Empty or None control keys are disabled.
            def getControlKeyCode(key, settingName):
                if not key: return None
                keyCode = keyboardMouse.charToKeyCode(key)
                if keyCode is None: raise ValueError(f"Unknown {settingName}: {key!r}")
                return int(keyCode)

            self.state.responseKeyCodesList = [keyboardMouse.charToKeyCode(k) for k in self.responseKeysList]
            self.state.startKeyCode = getControlKeyCode(self.settings.startKey, "startKey")
            self.state.endKeyCode = getControlKeyCode(self.settings.endKey, "endKey")
            self.state.volumeTriggerKeyCode = getControlKeyCode(self.settings.volumeTriggerKey, "volumeTriggerKey")

            # Waiting for keyboard synchronization requires configured keys.
            if self.settings.startOnVolumeTrigger and self.state.volumeTriggerKeyCode is None:
                raise ValueError("startOnVolumeTrigger requires a volumeTriggerKey.")
            if self.settings.manualPreStart and self.state.startKeyCode is None:
                raise ValueError("manualPreStart requires a startKey.")

            # If eatKeys is set, include only configured keys.
            if self.settings.eatKeys:
                eatKeyCodes = self.state.responseKeyCodesList.copy()
                for keyCode in (self.state.startKeyCode, self.state.endKeyCode, self.state.volumeTriggerKeyCode):
                    if keyCode is not None: eatKeyCodes.append(keyCode)
                keyboardMouse.setEatKeys(eatKeyCodes)

            # Wait half a second for the Metal app to initialize.
            self.pgl.waitSecs(0.5)

            # Flush the screen to get rid of transients.
            self.pgl.flush()
            self.pgl.flush()

            # Initialize eye tracker.
            try:
                self.initEyeTracker()
            except Exception as error:
                pglMessages.warning(f"Could not initialize eye tracker: {error}")

            # Display device status.
            self.pgl.deviceStatus()

            # Set up digital IO and other devices.
            self.initDigitalIO()
            self.initDevices()

            # Start before configure(), which may create stimulus resources.
            self._startExperimentRecording(estimatedDuration, recordFlush, recordCommands)

            # A failed configuration is not a successfully initialized task.
            # Propagate it through the same partial-save recovery path.
            for task in self.tasks: task.configure(self)

            self.state.openScreen = True

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            self.state.runFinishedWithError = True
            self._recordingOutcome = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
            self._recordRunError("screen initialization", error)

            errors = []
            self._attemptRunAction("record initialization end time", lambda: setattr(self.data, "endTime", self.pgl.getSecs()), errors)

            try:
                if getattr(self, "_recordingId", None) is not None:
                    # Recording has started: save available task/recording data
                    # before renderer or device cleanup.
                    self._finalizeRun()
                else:
                    # No recording was started. Attempt independent cleanup.
                    self._attemptRunAction("close experiment screen", lambda: self.endScreen(clearScreen=False), errors)
                    self._attemptRunAction("close digital IO", self.endDigitalIO, errors)
                    self._attemptRunAction("close experiment devices", self.endDevices, errors)
            except (Exception, KeyboardInterrupt, SystemExit) as finalizationError:
                self._recordRunError("initialization finalization", finalizationError)
            finally:
                self.state.openScreen = False

            raise
        
    def setGamma(self, settings, display):
        '''
        set the gamma table based on settings
        
        Args:
            settings (pglSettings): Used for determining what gamma to use
            display (pglDisplaySettings): Used to determine what displayNum to set the gamma on
        '''
        # no gamma correction asked for
        if settings.calibrateForGamma == 0.0:
            return
        
        # No calibration
        if display.luminanceCalibration[0] == "None":
            if settings.calibrateForGamma is not None and settings.calibrateForGamma[0] != 0.0:
                pglMessages.warning(f"Gamma is set to {settings.calibrateForGamma[0]} but no calibration found for display {display.name}")
                return
        
        # save the original gamma table
        self.state.originalGammaTable = self.pgl.getGammaTable(display.currentDisplayNum)

        # set the gamma to whatever is at the top of the calibrateForGamma list
        gamma = settings.calibrateForGamma[0]
        display.setGamma(self.pgl, gamma=gamma)
        
        # save the gamma table
        self.state.gammaTable = self.pgl.getGammaTable(display.currentDisplayNum)
        
    ################################################################
    # Restore and close the screen with independent cleanup attempts
    ################################################################
    def endScreen(self, clearScreen=True):
        """Restore display settings and close this experiment's screen.

        clearScreen=False avoids drawing commands after an interrupted exchange.
        Failures propagate after the other cleanup operations are attempted.
        """
        from .pglKeyboardMouse import pglKeyboardMouse

        errors = []

        # Stop each keyboard listener independently.
        try:
            keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse) or []
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            keyboardDevices = []
            errors.append(error)
            self._recordRunError("find keyboard devices", error)

        for keyboardDevice in keyboardDevices:
            self._attemptRunAction("stop keyboard/mouse device", keyboardDevice.stop, errors)

        if self.settings.closeScreenOnEnd:
            # Normal callers can request a final blank. Recovery callers skip
            # this because the renderer's command stream may be interrupted.
            if clearScreen:
                def clearDisplay():
                    self.pgl.clearScreen(self.settings.backgroundColor)
                    self.pgl.flush()

                self._attemptRunAction("clear screen", clearDisplay, errors)

            # Restore each display setting even if another operation fails.
            if self.state.originalGammaTable is not None:
                self._attemptRunAction(
                    "restore gamma table",
                    lambda: self.pgl.setGammaTable(
                        self.state.display.currentDisplayNum,
                        rgbGammaTable=self.state.originalGammaTable,
                    ),
                    errors,
                )

            if self.state.screenResolution is not None and self.state.originalScreenResolution is not None:
                if self.state.screenResolution != self.state.originalScreenResolution:
                    self._attemptRunAction(
                        "restore screen resolution",
                        lambda: self.pgl.setResolution(
                            self.state.display.currentDisplayNum,
                            screenResolution=self.state.originalScreenResolution,
                        ),
                        errors,
                    )

            def closeDisplay():
                if self.pgl.close() is False: raise RuntimeError("PGL could not close its renderer.")
                self.state.openScreen = False

            self._attemptRunAction("close renderer connection", closeDisplay, errors)

        self._attemptRunAction("restore verbosity", self.endVerbose, errors)

        if errors: raise errors[0]
        return True
                

    def setEatAllKeys(self, eatAllKeys=False):
        '''
        Sets whether to eat all keys. If False, any keys specified by setEatKeys will still be eaten
        Args: 
            eatAllKeys: bool (whether to eat all keys or not)
        '''
        return self.pgl.setEatAllKeys(eatAllKeys)

    def setEatKeys(self, eatKeys=""):
        '''
        Args: 
            eatKeys (list): list of characters to eat. e.g. ['return','esc','1']
        '''
        return self.pgl.setEatKeys(eatKeys)
    
    ################################################################
    # Run with protected saving and finalization
    ################################################################
    def run(self):
        """Run acquisition, save partial data on failure, and preserve its error."""
        if getattr(self, "_recordingReleased", False):
            raise RuntimeError("This experiment's recording was already saved and released. Create a new experiment.")

        if not self.state.openScreen:
            self.state.runFinishedWithError = True
            pglMessages.warning("Screen is not open. Call initScreen() before running the experiment.")
            return

        if getattr(self, "_runStarted", False):
            raise RuntimeError("This experiment has already run. Create a new experiment for another run.")

        # Verify ownership before running or cleaning up anything.
        metadata = self._getExperimentRecording()
        if metadata is not None and self.pgl._recordingState != "active":
            raise RuntimeError("The experiment recording is already stopped.")
        if metadata is None and getattr(self.pgl, "_recordingState", None) == "active":
            raise RuntimeError("PGL has an active recording not owned by this experiment.")

        self._runStarted = True
        if not hasattr(self, "_runSavePath"): self._runSavePath = None
        if not hasattr(self.data, "runErrors"): self.data.runErrors = []

        self.data.saveStatus = "notAttempted"
        self.data.cleanupStatus = "notAttempted"
        self.data.eyeTrackerSaveStatus = "notAttempted" if self.eyeTracker is not None else "notApplicable"
        self.state.runFinishedWithError = True
        self._recordingOutcome = "error"

        try:
            self._run()

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            # Preserve unfinished tasks; do not invoke task-ending hooks or
            # renderer commands before emergency saving.
            self.state.runFinishedWithError = True
            self._recordingOutcome = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
            self._recordRunError("acquisition", error)

            errors = []
            self._attemptRunAction("record experiment end time", lambda: setattr(self.data, "endTime", self.pgl.getSecs()), errors)

            try:
                self._finalizeRun()
            except (Exception, KeyboardInterrupt, SystemExit) as finalizationError:
                try:
                    pglMessages.warning(f"Additional finalization failure: {type(finalizationError).__name__}: {finalizationError}")
                except Exception:
                    pass

            # Preserve the original acquisition exception and traceback.
            raise

        else:
            # "completed" means acquisition returned normally. Existing aborted-
            # run heuristics remain responsible for trash routing; an end-key
            # exit is not classified as a Python error.
            self.state.runFinishedWithError = False
            self._recordingOutcome = "completed"
            self._finalizeRun()

    ################################################################
    # Store errors as plain serializable metadata
    ################################################################
    def _recordRunError(self, operation, error):
        """Record an error without retaining its live traceback object."""
        import traceback

        # Error reporting itself must not replace the original failure.
        try:
            errorInfo = {
                "operation": operation,
                "module": type(error).__module__,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
            if not hasattr(self.data, "runErrors"): self.data.runErrors = []
            self.data.runErrors.append(errorInfo)
        except Exception:
            pass

        try:
            pglMessages.warning(f"{operation}: {type(error).__name__}: {error}")
        except Exception:
            pass

    ################################################################
    # Attempt an independent save or cleanup operation
    ################################################################
    def _attemptRunAction(self, operation, action, errors):
        """Collect failures so other independent operations can still run."""
        try:
            result = action()
            if result is False: raise RuntimeError(f"{operation} returned False.")
            return True

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            # Interrupts during finalization are deferred, not treated as
            # success. The caller raises an error after the remaining attempts.
            errors.append(error)
            self._recordRunError(operation, error)
            return False

    ################################################################
    # Finalize successful or interrupted acquisition
    ################################################################
    def _finalizeRun(self):
        """Stop owned capture, attempt local saving, then clean up devices."""
        errors = []

        def stopRecording():
            metadata = self._getExperimentRecording()
            if metadata is not None:
                self.pgl.recordingStop(outcome=getattr(self, "_recordingOutcome", "error"))

        # Stop before any save or cleanup command can enter the recording.
        self._attemptRunAction("stop experiment recording", stopRecording, errors)

        # Attempt local saving before potentially blocking device operations.
        if not self._attemptRunAction("save experiment", lambda: self.save(includeEyeTracker=False), errors):
            self.data.saveStatus = "failed"

        # Normal acquisition already cleared/flushed the display. Recovery
        # must not send drawing commands on a potentially interrupted socket.
        cleanupErrorCount = len(errors)
        self._attemptRunAction("close experiment screen", lambda: self.endScreen(clearScreen=False), errors)
        self._attemptRunAction("close digital IO", self.endDigitalIO, errors)
        self._attemptRunAction("close experiment devices", self.endDevices, errors)
        self.data.cleanupStatus = "failed" if len(errors) > cleanupErrorCount else "complete"

        # Essential local files have been attempted before EyeLink transfer.
        if self.eyeTracker is not None:
            eyeTrackerErrorCount = len(errors)
            dataPath = getattr(self, "_runSavePath", None)

            if dataPath is not None:
                def saveEyeTracker():
                    filename = dataPath / self.settings.eyetracker[0].lower()
                    return self.eyeTracker.save(filename)

                self._attemptRunAction("save eye tracker", saveEyeTracker, errors)
            else:
                self._attemptRunAction("stop eye tracker", self.eyeTracker.stop, errors)

            self.data.eyeTrackerSaveStatus = (
                "failed" if len(errors) > eyeTrackerErrorCount
                else "complete" if dataPath is not None
                else "notSaved"
            )

        # Persist final outcome metadata without repeating task or EDF saves.
        dataPath = getattr(self, "_runSavePath", None)
        if dataPath is not None:
            if not self._attemptRunAction("update final experiment state", lambda: self.state.saveStrict(dataPath / "state.json"), errors):
                self.data.saveStatus = "failed"
            if not self._attemptRunAction("update final experiment data", lambda: self.data.saveStrict(dataPath / "data.json"), errors):
                self.data.saveStatus = "failed"

        # run() preserves an earlier acquisition error. Otherwise this exposes
        # the first finalization failure after all independent attempts.
        if errors: raise errors[0]
    
    def _run(self):
        '''
        Execute the experiment.

        The public run() method handles saving, exception recovery, and cleanup.
        '''
        # Default to error; the public wrapper marks successful acquisition.
        self.state.runFinishedWithError = True

        if not self.state.openScreen:
            pglMessages.warning("Screen is not open. Call initScreen() before running the experiment.")
            return

        # Start eye tracker recording if we have an eye tracker.
        if self.eyeTracker is not None: self.eyeTracker.start()

        # Calculate the distinct phases that we will need to cover.
        self.state.phaseNums = sorted({task.settings.phaseNum for task in self.tasks if task.settings.phaseNum is not None}) or None
        self.state.currentPhaseIndex = 0

        # Initialize variables.
        self.state.experimentDone = False
        self.state.experimentStarted = False
        self.state.volumeNumber = 0

        # Manual pre-start waits for the start key before scanner synchronization.
        manualPreStart = bool(self.settings.manualPreStart)
        manualPreStartVolumes = 0
        ignoreInitialVolumes = self.settings.ignoreInitialVolumes

        # See whether we need to run eye calibration.
        if self.settings.eyetracker is not None: self.calibrateEyeTracker()

        # Disabled keys must not accidentally match an event with keyCode=None.
        def matchesKey(event, keyCode, keydownOnly=False):
            return keyCode is not None and event.type == "keyboard" and event.keyCode == keyCode and (not keydownOnly or event.eventType == "keydown")

        # Wait for a start key or scanner trigger only when configured.
        if self.state.startKeyCode is not None or self.settings.startOnVolumeTrigger:
            while not self.state.experimentStarted:
                if manualPreStart:
                    self.pgl.text(f"Press {self.settings.startKey} key to make experiment start", y=0)
                elif self.settings.startOnVolumeTrigger:
                    self.pgl.text("Waiting for volume trigger to start experiment", y=0)
                else:
                    self.pgl.text(f"Press {self.settings.startKey} key to start experiment", y=0)

                # Flush to display text, then poll for events.
                self.pgl.flush()
                events = self.pgl.poll()
                self.data.events.extend(events)

                # Match the configured start key.
                if any(matchesKey(event, self.state.startKeyCode, keydownOnly=True) for event in events):
                    if manualPreStart:
                        manualPreStart = False
                        pglMessages.message(f"Manual pre-start ended after {manualPreStartVolumes} volumes.", messageType='experiment')
                    else:
                        self.state.experimentStarted = True

                # Handle scanner triggers while waiting to start.
                if self.settings.startOnVolumeTrigger:
                    for event in events:
                        if matchesKey(event, self.state.volumeTriggerKeyCode, keydownOnly=True):
                            if manualPreStart:
                                manualPreStartVolumes += 1
                            elif ignoreInitialVolumes > 0:
                                ignoreInitialVolumes -= 1
                            else:
                                pglMessages.message(f"Ignored {self.settings.ignoreInitialVolumes} initial volumes.", messageType='experiment')
                                self.state.experimentStarted = True
                                self.state.volumeNumber += 1
                                self.data.events.append(pglEventVolumeTrigger(timestamp=event.timestamp))

                # Allow aborting before any tasks have started.
                if any(matchesKey(event, self.state.endKeyCode) for event in events):
                    self.state.experimentStarted = True
                    self.state.experimentDone = True

        else:
            # No start gate: begin immediately.
            self.state.experimentStarted = True

        # An abort at the waiting screen must not start tasks.
        if self.state.experimentDone:
            self.pgl.clearScreen(self.settings.backgroundColor)
            self.pgl.flush()
            self.data.endTime = self.pgl.getSecs()
            pglMessages.message("Experiment ended before tasks started.", messageType='experiment')
            return

        # Start the experiment before invoking task start hooks, so a failure
        # in a start hook still has an experiment start timestamp.
        pglMessages.message("Experiment started.", messageType='experiment')
        self.data.startTime = self.pgl.getSecs()
        self.data.startDateTime = datetime.now().astimezone().isoformat()

        if not self.state.phaseNums:
            raise ValueError("Experiment needs at least one task with a phaseNum.")

        self.startPhase(phaseNum=self.state.phaseNums[0])

        while not self.state.experimentDone:
            # Poll for events.
            events = self.pgl.poll()
            self.data.events.extend(events)

            # End all running tasks when the configured end key is received.
            if any(matchesKey(event, self.state.endKeyCode) for event in events):
                self.state.experimentDone = True
                for task in self.currentTasks: task.end()
                continue

            # Check for volume trigger key.
            for iEvent, event in enumerate(events):
                if matchesKey(event, self.state.volumeTriggerKeyCode, keydownOnly=True):
                    # Remove it from the response-processing list. The original
                    # keyboard event is already retained in experiment data.
                    events.pop(iEvent)
                    self.state.volumeNumber += 1
                    self.data.events.append(pglEventVolumeTrigger(timestamp=event.timestamp))
                    break

            # Match response keys and preserve their original event timestamps.
            subjectResponses = [
                (self.state.responseKeyCodesList.index(event.keyCode), event.timestamp)
                for event in events
                if (
                    event.type == "keyboard"
                    and event.eventType == "keydown"
                    and event.keyCode in self.state.responseKeyCodesList
                )
            ]

            # Update tasks in the current phase.
            phaseDone = False
            updateTime = self.pgl.getSecs()
            for task in self.currentTasks:
                task.update(
                    updateTime=updateTime,
                    subjectResponses=subjectResponses,
                    phaseNum=self.state.phaseNum,
                    tasks=self.currentTasks,
                    events=events,
                )
                if task.done(): phaseDone = True

            # Update the screen.
            if self.flush: self.pgl.flush()

            # Go to the next phase or end the experiment.
            if phaseDone:
                for task in self.currentTasks: task.end()

                if self.state.currentPhaseIndex >= len(self.state.phaseNums) - 1:
                    self.state.experimentDone = True
                else:
                    self.state.currentPhaseIndex += 1
                    self.startPhase(phaseNum=self.state.phaseNums[self.state.currentPhaseIndex])

        # Clear the screen after normal acquisition.
        self.pgl.clearScreen(self.settings.backgroundColor)
        self.pgl.flush()

        # Mark acquisition end before saving or device cleanup.
        self.data.endTime = self.pgl.getSecs()
        pglMessages.message("Experiment done.", messageType='experiment')
        pglMessages.printHeader(messageType='experiment')
    
    def initDigitalIO(self):
        '''
        initialize digital IO device if settings calls for them to be initialized
        '''
        
        # end (just in case digitalIO was already started)
        self.endDigitalIO()
        
        # clear any existing configuration
        self.state.digitalIODevices = []
        self.state.triggerWords = {}
        self.state.triggers = {}
        self.settings.digitalIOWords = []
        self.settings.digitalIOChannels = []

        for digitalIODevice in self.settings.digitalIO:
            if digitalIODevice.lower() == "labjack":
                
                # try to initialize labjack device
                from .pglLabJack import pglLabJack
                labJack = pglLabJack()
                
                # if it is working add it
                if labJack.isActive:
                    self.state.digitalIODevices.append(labJack)
                    pglMessages.message("LabJack initialized")
                else:
                    pglMessages.message("LabJack did not initialize")
                                        
            elif digitalIODevice.lower() == "datapixx":
                
                # try to initialize datapixx
                from .pglVPixx import pglDataPixx
                dataPixx = pglDataPixx()
                
                if dataPixx.isActive:
                    self.state.digitalIODevices.append(dataPixx)
                    pglMessages.message("DataPixx initialized")
                else:
                    pglMessages.message("DataPixx did not initialize")
            else:
                pglMessages.warning(f"Unknown digitalIO device: {digitalIODevice}")       
        

    def getDigitalIODevice(self, digitalIODevice = None):
        '''
        Returns the digitalIODevice that has been initialized or None if there is no digialIODevice
        Without any arguments returns the 1st in the list if there is more than one (digialIODevices are
        selected in pgl.settings()
        
        Args:
            digitalIODevice (pglDigitalIODevice): Type of digitalIODevice to return (e.g. pglLabJack). If there
                is no match, then None will be returned. 
        '''
        
        # no digitalIODevice specified, return the first in list
        if digitalIODevice is None:
            return next(iter(self.state.digitalIODevices), None)

        # look for device and return it if specified
        return next((device for device in self.state.digitalIODevices if isinstance(device, digitalIODevice)),None)
        
    def configureTriggers(self, channels, groupName=None, digitalIODevice=None, pulseLen=5):
        """
        Configure named triggers on the digital IO device.

        Args:
            channels (dict, int, str, or list): A dictionary mapping trigger keys to
                channel numbers or physical pin names, or unnamed channel values.
                Integer values specify logical channel numbers. Physical pin names
                receive automatically allocated logical IDs. Unnamed entries receive
                keys such as "trigger00", based on their logical channel number.
            groupName (str or None): Hardware bank used for numeric channel values.
            digitalIODevice (type or None): Device class to select; None uses the first device.
            pulseLen (int): Pulse duration in milliseconds.
        """
        if isinstance(channels, dict):
            entries = list(channels.items())
            explicitKeys = True
        elif isinstance(channels, (int, str)):
            entries = [(None, channels)]
            explicitKeys = False
        elif isinstance(channels, (list, tuple, range)):
            entries = [(None, value) for value in channels]
            explicitKeys = False
        else:
            raise TypeError("channels must be a dictionary, integer, string, or sequence of channel values")

        if not entries:
            return

        # Validate input before allocating IDs or changing hardware.
        for channelKey, value in entries:
            if explicitKeys and (not isinstance(channelKey, str) or not channelKey):
                pglMessages.warning("Each channelKey must be a nonempty string")
                return
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                pglMessages.warning("Each channel must be an integer or a physical pin name")
                return
            if isinstance(value, int) and value < 0:
                pglMessages.warning("Channel numbers must be nonnegative")
                return
            if isinstance(value, str) and not value:
                pglMessages.warning("Physical pin names must be nonempty")
                return

        device = self.getDigitalIODevice(digitalIODevice=digitalIODevice)
        if device is None:
            pglMessages.warning("No matching digitalIO device has been found. Make sure that devices have been configured in pgl.settings() and are properly connected")
            return

        # Reserve existing IDs and explicitly requested IDs before allocating new ones.
        usedChannels = {config.channel for config in self.settings.digitalIOChannels}
        usedChannels.update(device.digitalChannels)
        usedChannels.update(value for _, value in entries if isinstance(value, int))
        nextChannel = max(usedChannels, default=-1) + 1

        existingKeys = {config.channelKey for config in self.settings.digitalIOChannels}
        existingKeys.update(self.state.triggers)

        # Build and validate all metadata before configuring hardware.
        channelSettings = []
        for channelKey, value in entries:
            if isinstance(value, str):
                channel = nextChannel
                channelName = value
                nextChannel += 1
            else:
                channel = value
                channelName = None

            if channelKey is None:
                channelKey = f"trigger{channel:02d}"

            if channelKey in existingKeys:
                pglMessages.warning(f"Trigger key already configured: {channelKey}")
                return

            existingKeys.add(channelKey)
            config = pglDigitalIOChannel(channelKey=channelKey, channel=channel, channelName=channelName, groupName=groupName, pulseLen=pulseLen, deviceType=device.deviceType)
            channelSettings.append(config)

        existingChannels = set(device.digitalChannels)
        newChannels = [config.channel for config in channelSettings]

        if len(newChannels) != len(set(newChannels)) or existingChannels.intersection(newChannels):
            pglMessages.warning("Logical channel numbers must be unique on each device")
            return

        for config in channelSettings:
            if not device.setupDigitalOutput(channel=config.channel, channelName=config.channelName, groupName=config.groupName, pulseLen=config.pulseLen):
                pglMessages.warning(f"Failed to configure trigger {config.channelKey!r}")
                return

            self.settings.digitalIOChannels.append(config)
            self.state.triggers[config.channelKey] = (config, device)
    
    def configureTriggerWords(self, wordDict: dict, digitalIODevice=None, channels=None, pulseLen=5):
        """
        Configure named trigger words.

        Args:
            wordDict (dict[str, int]): Word names mapped to unsigned integer values.
            digitalIODevice (type or None): Device class to select; None uses the first device.
            channels (list[int | str] or None): Configured logical channel numbers or
                physical channel names, ordered from lowest to highest bit.
                None uses all configured channels in logical channel number order.
            pulseLen (int): Pulse duration in milliseconds, stored with each word.
        """
        device = self.getDigitalIODevice(digitalIODevice=digitalIODevice)
        if device is None:
            pglMessages.warning("No matching digitalIO device has been found. Make sure that devices have been configured in pgl.settings() and are properly connected")
            return

        if not isinstance(wordDict, dict):
            pglMessages.warning("wordDict must be a dictionary mapping word names to integer values")
            return

        configuredChannels = [config for config, configuredDevice in self.state.triggers.values() if configuredDevice is device]

        if channels is None:
            selectedChannels = sorted(configuredChannels, key=lambda config: config.channel)
        else:
            channels = [channels] if isinstance(channels, (int, str)) else list(channels)
            selectedChannels = []

            for channel in channels:
                if isinstance(channel, bool) or not isinstance(channel, (int, str)):
                    pglMessages.warning("Each channel must be an integer channel number or a string channel name")
                    return

                if isinstance(channel, int):
                    matches = [config for config in configuredChannels if config.channel == channel]
                else:
                    matches = [config for config in configuredChannels if config.channelName == channel]

                if not matches:
                    pglMessages.warning(f"Channel {channel!r} is not configured for {device.deviceType}")
                    return
                if len(matches) > 1:
                    pglMessages.warning(f"Channel {channel!r} matches multiple configured entries")
                    return

                selectedChannels.append(matches[0])

        channelNumbers = [config.channel for config in selectedChannels]

        if not channelNumbers:
            pglMessages.warning("At least one configured channel is required for trigger words")
            return
        if len(channelNumbers) != len(set(channelNumbers)):
            pglMessages.warning("Each logical channel can appear only once in a trigger word")
            return

        # Check the device's actual configuration before its setup method is called.
        missingChannels = set(channelNumbers) - device.digitalChannels.keys()
        if missingChannels:
            pglMessages.warning(f"Channels are not initialized on the selected device: {sorted(missingChannels)}")
            return

        maxValue = (1 << len(channelNumbers)) - 1
        existingKeys = {word.key for word in self.settings.digitalIOWords}
        existingKeys.update(self.state.triggerWords)

        # Validate all metadata before changing the device's word configuration.
        wordSettings = []
        for key, value in wordDict.items():
            if not isinstance(key, str) or not key:
                pglMessages.warning("Each word key must be a nonempty string")
                return
            if key in existingKeys:
                pglMessages.warning(f"Trigger word already configured for {device.deviceType}: {key}")
                return
            if isinstance(value, bool) or not isinstance(value, int):
                pglMessages.warning(f"Value for word '{key}' must be an integer")
                return
            if not 0 <= value <= maxValue:
                pglMessages.warning(f"Value for word '{key}' must be between 0 and {maxValue}")
                return

            word = pglDigitalIOWord(key=key, value=value, pulseLen=pulseLen, deviceType=device.deviceType)
            wordSettings.append(word)

        # check for existing configuration - 
        hasConfiguredWords = any(wordDevice is device for _, wordDevice in self.state.triggerWords.values())
        if hasConfiguredWords and channelNumbers != device.wordDigitalChanels:
            pglMessages.warning("Cannot change word channels while trigger words are configured on this device")
            return
        
        # configure the ouptut channels for writing words
        device.setupDigitalOutputWord(channels=channelNumbers)

        # and keep the settings and state 
        for word in wordSettings:
            self.settings.digitalIOWords.append(word)
            self.state.triggerWords[word.key] = (word, device)
    
    def sendTriggerWord(self, key, task=None):
        """Send a configured trigger word by its unique key."""
        target = self.state.triggerWords.get(key)
        
        if target is None:
            pglMessages.message(f"Trigger word {key!r} has not been configured")
            # setup trigger event
            e = pglEventOutputTrigger(triggerName=key, triggerWord=None, timestamp=pglTimestamp.getSecs(), triggerSent=False)
        else:
            # get the word and device to send
            word, device = target
            # send it
            timestamp = device.digitalOutputWord(word.value)
            if timestamp is None: 
                pglMessages.warning(f"Device did not return a timestamp",level=1)
                # setup trigger event
                e = pglEventOutputTrigger(triggerName=key, triggerWord=word.value, timestamp=pglTimestamp.getSecs(), triggerSent=False)
            else:
                # setup trigger event
                e = pglEventOutputTrigger(triggerName=key, triggerWord=word.value, timestamp=timestamp, triggerSent=True)

        # add event to events
        self.data.events.append(e)
        if task is not None: task.data.events.append(e)

    def sendTrigger(self, key, task=None):
        """Send a configured trigger; return its timestamp or None."""
        target = self.state.triggers.get(key)
        if target is None:
            pglMessages.message(f"Trigger {key!r} has not been configured")
            e = pglEventOutputTrigger(triggerName=key, triggerValue=None,
                                    timestamp=pglTimestamp.getSecs(), triggerSent=False)
            timestamp = None
        else:
            config, device = target
            timestamp = device.digitalOutputPulse(config.channel)
            if timestamp is None:
                pglMessages.warning("Device did not return a timestamp", level=1)
                e = pglEventOutputTrigger(triggerName=key, triggerValue=config.channel,
                                        timestamp=pglTimestamp.getSecs(), triggerSent=False)
            else:
                e = pglEventOutputTrigger(triggerName=key, triggerValue=config.channel,
                                        timestamp=timestamp, triggerSent=True)

        self.data.events.append(e)
        if task is not None:
            task.data.events.append(e)

        return timestamp    
    
    def endDigitalIO(self):
        """Close each digital IO device independently, retaining failed devices."""
        errors = []
        failedDevices = []

        # One device failure must not prevent closing the others.
        for digitalIODevice in self.state.digitalIODevices:
            if not self._attemptRunAction("close digital IO device", digitalIODevice.close, errors):
                failedDevices.append(digitalIODevice)

        # Retain failed devices so cleanup can be retried.
        self.state.digitalIODevices = failedDevices
        if not failedDevices:
            self.state.triggers = {}
            self.state.triggerWords = {}

        if errors: raise errors[0]
        return True
            
    def initDevices(self):
        '''
        initialize devices
        '''
        if self.settings.devices:
            pglMessages.warning('Devices not implemented yet')

            

    def endDevices(self):
        '''
        end digitalIO devices
        '''
        pass
        
        
    def initEyeTracker(self):
        '''Initialize eye tracker if we have an eye tracker.'''
        # load eye tracker settings
        self.eyeTrackerSettings = self.pgl.getEyeTrackerSettings(settings=self.settings)
        
        if self.settings.eyetracker[0] == "Eyelink":
            from .pglEyelink import pglEyelink
            # set edf filename to current date (note it has to be 8.3 characters
            # since SR Research has progressed from the days of DOS)
            self.settings.edfFilename = f"{datetime.now().strftime('%Y%m%d')}"

            # init the eyeink
            pglMessages.message(f"(pglExperiment) Initialize Eyelink with filename: {self.settings.edfFilename}",messageType='experiment')
            self.eyeTracker = pglEyelink(pgl=self.pgl, edfFilename=self.settings.edfFilename)  

            # check if running
            if self.eyeTracker.eyelink is None:
                self.eyeTracker = None      

            # get from settings
            nEyelinkCalibrationPoints = self.eyeTrackerSettings.nEyelinkCalibrationPoints
            eyelinkCalibrationProportion = self.eyeTrackerSettings.eyelinkCalibrationProportion
 
            if self.eyeTracker is not None:
                self.eyeTracker.setCustomCalibrationPoints(margin=eyelinkCalibrationProportion, numPoints=nEyelinkCalibrationPoints)

        elif self.settings.eyetracker[0] == "None":
            self.eyeTracker = None
        else:
            pglMessages.warning(f"Unknown eye tracker type {self.settings.eyetracker[0]}")
            self.eyeTracker = None


    def calibrateEyeTracker(self):
        '''
        Run eye tracker calibration if we have an eye tracker and it is not calibrated yet.
        '''
        if self.eyeTracker is not None:
            # wait for key press to calibrate
            self.state.waitingForCalibration = True
            self.state.runCalibration = False
            # FIX: These could be exposed 
            self.settings.calibrateKey = 'space'
            self.settings.calibrateKeyCode = self.keyboardMouse.charToKeyCode(self.settings.calibrateKey)
            self.settings.skipCalibrationKey = 'return'
            self.settings.skipCalibrationKeyCode = self.keyboardMouse.charToKeyCode(self.settings.skipCalibrationKey)
            # instructions
            displayText = f"Press {self.settings.calibrateKey} to calibrate eye tracker. {self.settings.skipCalibrationKey} to skip."
            pglMessages.message(displayText, messageType='experiment')

            k = self.pgl.devicesGetKeyboard()            
            eatKeys = k.eatKeyCodes

            # eat relevant keys
            self.pgl.setEatKeys(keyChars=[self.settings.calibrateKey, self.settings.skipCalibrationKey])

            # wait till we get a response
            while self.state.waitingForCalibration:
                self.pgl.text(displayText, y=0)
                # flush to display text
                self.pgl.flush()
                # poll for events
                events = self.pgl.poll()
                self.data.events.extend(events)

                # see if we have a match to startKey
                if [e for e in events if e.type == "keyboard" and e.eventType == "keydown"and e.keyCode == self.settings.calibrateKeyCode]:
                    self.state.waitingForCalibration = False
                    self.state.runCalibration = True
                elif [e for e in events if e.type == "keyboard" and e.eventType == "keydown" and e.keyCode == self.settings.skipCalibrationKeyCode]:
                    self.state.waitingForCalibration = False
                    self.state.runCalibration = False
                    pglMessages.message("Skipping eye tracker calibration.",messageType='experiment')

            # reset eat keys
            self.pgl.setEatKeys(eatKeys)    

            # if we should run calibration, then do it
            if self.state.runCalibration:
                self.eyeTracker.calibrate()
                # and restart saving data, as calibrate seems to stop it
                self.eyeTracker.start()


    def saveEyeTrackerEvent(self, eventType="segment", taskID=None, trialNum=None, segmentNum=None, timestamp=None, phaseNum=None):
        '''Save an eye tracker event for synchronization. This is called by tasks during updates if settings.saveEyeTracker is True.'''
        self.eyeTracker.sendMessage(f"pgl: {eventType} taskID={taskID} trialNum={trialNum} segmentNum={segmentNum} timestamp={timestamp} phaseNum={phaseNum}")

    def startPhase(self, phaseNum=0):
        '''
        Start the current phase of the experiment.
        '''
        self.state.phaseNum = phaseNum
        
        # starting phase
        pglMessages.message(f"Starting phase: {self.state.phaseNum+1}/{len(self.state.phaseNums)}",messageType='experiment')
        
        # get the current tasks based on the current phase number
        self.currentTasks = [task for task in self.tasks if task.settings.phaseNum is None or task.settings.phaseNum == self.state.phaseNum]

        # set start time
        startTime = self.pgl.getSecs()
        for task in self.currentTasks:
            if task.start(startTime) == False:
                # if task start returns Fales, then end it and remove it form the current tasks list
                pglMessages.warning(f"(pglExperiment:startPhase) Task {task.settings.taskName} failed to start. Ending task.")
                task.end()
                self.currentTasks.remove(task)
            
        # if there are no currentTasks, it means that all the tasks for this phase have already been completed, so we should move to the next phase    
        if len(self.currentTasks) == 0:
            # check if we have ended all phases 
            if self.state.phaseNum >= len(self.state.phaseNums)-1:
                self.state.experimentDone = True
            else:
                # update phase
                self.state.currentPhaseIndex += 1
                self.startPhase(phaseNum=self.state.phaseNums[self.state.currentPhaseIndex])

    def isAbortedRun(self):
        '''
        heuristics to reutnr true/false on aborted runs
        '''
        aborted = False
        for task in self.tasks:
            if task.settings.nTrials != np.inf:
                # if less trials than nTrials were run, then it is aborted
                if task.state.currentTrial+1 < task.settings.nTrials:
                    pglMessages.message(f"Detected aborted task with {task.state.currentTrial+1}/{task.settings.nTrials} trials run: {task.settings.taskSaveName}", messageType='experiment')
                    aborted=True
            else:
                # if less than 10 trials were run it is aborted
                if task.state.currentTrial < 10:
                    pglMessages.message(f"Detected aborted task with less than 10 trials: {task.settings.taskSaveName}",messageType='experiment')
                    aborted=True
        return aborted
    
    ################################################################
    # Save experiment components independently
    ################################################################
    def save(self, includeEyeTracker=True):
        """Save available run contents, continuing after component failures.

        Repeated calls reuse this run's directory. Experiment recording buffers
        are automatically released after HDF5 and its JSON reference are saved.

        The bundle is not atomic, although recording.h5 is published through
        its own temporary-file replacement.
        """
        errors = []
        dataPath = getattr(self, "_runSavePath", None)

        if dataPath is None:
            # Failed aborted-run classification must not prevent saving.
            aborted = False
            if self.settings.saveAbortedRunsToTrash:
                try:
                    aborted = self.isAbortedRun()
                except (Exception, KeyboardInterrupt, SystemExit) as error:
                    errors.append(error)
                    self._recordRunError("classify aborted run", error)

            runName = Path(self.experimentSettings.runName)
            if aborted: runName = Path("trash") / runName

            basePath = (
                Path(self.settings.dataPath).expanduser()
                / self.experimentSettings.experimentSaveName
                / self.experimentSettings.subjectID
                / self.experimentSettings.sessionName
                / runName
            )

            # Never overwrite a previous run. Retry directory-name collisions.
            dataPath = basePath
            suffix = datetime.now().strftime("%H-%M-%S-%f")
            attempt = 0
            while True:
                try:
                    dataPath.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    attempt += 1
                    dataPath = basePath.parent / f"{basePath.name}_{suffix}_{attempt}"

            # Retain the destination immediately for discovery and retries.
            self._runSavePath = dataPath
            self.fullDataPath = str(dataPath)

        pglMessages.message(f"Saving experiment data to: {dataPath}", messageType='experiment')
        self.data.saveStatus = "inProgress"

        # Save essential local data before HDF5 export or device operations.
        components = (
            ("experiment data", self.data, "data.json"),
            ("experiment state", self.state, "state.json"),
            ("experiment settings", self.experimentSettings, "experimentSettings.json"),
            ("settings", self.settings, "settings.json"),
        )
        for label, component, filename in components:
            self._attemptRunAction(
                f"save {label}",
                lambda component=component, filename=filename: component.saveStrict(dataPath / filename),
                errors,
            )

        # Preserve the parent's optional traits.json convention.
        if set(self.traits()) != set(pglTraitSettings().traits()):
            self._attemptRunAction(
                "save experiment traits",
                lambda: pglSerialize.saveStrict(self, filename=dataPath / "traits.json"),
                errors,
            )

        # Save task components independently.
        for taskIndex, task in enumerate(self.tasks):
            try:
                taskPath = dataPath / task._getTaskDirectoryName()
                taskPath.mkdir(parents=True, exist_ok=True)
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                errors.append(error)
                self._recordRunError(f"create task {taskIndex + 1} directory", error)
                continue

            for componentName in ("data", "state", "settings"):
                self._attemptRunAction(
                    f"save task {taskIndex + 1} {componentName}",
                    lambda task=task, componentName=componentName, taskPath=taskPath:
                        getattr(task, componentName).saveStrict(taskPath / f"{componentName}.json"),
                    errors,
                )

            parameterPath = taskPath / "parameters"
            if self._attemptRunAction(
                f"create task {taskIndex + 1} parameters directory",
                lambda parameterPath=parameterPath: parameterPath.mkdir(parents=True, exist_ok=True),
                errors,
            ):
                for parameterIndex, parameter in enumerate(task.parameters):
                    self._attemptRunAction(
                        f"save task {taskIndex + 1} parameter {parameterIndex + 1}",
                        lambda parameter=parameter, parameterPath=parameterPath:
                            parameter.saveBundleStrict(dataPath=parameterPath),
                        errors,
                    )

        # Cached PGL state does not require a renderer query.
        self._attemptRunAction("save PGL state", lambda: self.pgl.saveStrict(dataPath / "pgl.json"), errors)

        # HDF5 failure must not prevent the other experiment files above.
        # Repeated saves after release reuse the existing recording artifact.
        if getattr(self, "_recordingId", None) is not None:
            saved = self._attemptRunAction(
                "save experiment recording",
                lambda: self._saveExperimentRecording(dataPath),
                errors,
            )
            if not saved:
                reference = dict(getattr(self.data, "recording", {}))
                reference["recordingId"] = self._recordingId
                reference["saveStatus"] = "failed"
                self.data.recording = reference

        # Direct save() callers retain EDF retrieval. run() postpones it until
        # local saving and screen/device cleanup have been attempted.
        if includeEyeTracker and self.eyeTracker is not None:
            def saveEyeTracker():
                filename = dataPath / self.settings.eyetracker[0].lower()
                return self.eyeTracker.save(filename)

            saved = self._attemptRunAction("save eye tracker", saveEyeTracker, errors)
            self.data.eyeTrackerSaveStatus = "complete" if saved else "failed"

        # Persist the recording reference before releasing its only RAM copy.
        self.data.saveStatus = "failed" if errors else "complete"
        referenceSaved = self._attemptRunAction(
            "save final save status",
            lambda: self.data.saveStrict(dataPath / "data.json"),
            errors,
        )
        if not referenceSaved:
            self.data.saveStatus = "failed"

        reference = getattr(self.data, "recording", {})
        if (referenceSaved and getattr(self, "_recordingId", None) is not None
                and reference.get("saveStatus") == "complete"):
            if not self._attemptRunAction(
                "release saved experiment recording",
                self._releaseExperimentRecording,
                errors,
            ):
                # Buffer-release failure is separate from artifact success:
                # do not relabel the successfully written HDF5 as failed.
                self.data.saveStatus = "failed"
                self._attemptRunAction(
                    "save recording release failure",
                    lambda: self.data.saveStrict(dataPath / "data.json"),
                    errors,
                )

        if errors: raise errors[0]
        return dataPath
    
    def getLastRun(self, task=None, allRuns=False):
        """Load saved runs or their matching tasks.

        Args:
            task (pglTask, optional): A task from self.tasks. When provided,
                return the task at the same index in each selected run.
            allRuns (bool): Return all matches, newest first, instead of
                only the latest run's match.

        Returns:
            When allRuns=False:
                pglRun, pglTask, or None if no match exists.
            When allRuns=True:
                A list of pglRun or pglTask objects, or an empty list.
                Tasks missing from a saved run are omitted.
        """
        try:
            from .pglSession import pglRun

            emptyResult = [] if allRuns else None

            # Resolve the requested task's position in the current experiment.
            taskIndex = None
            if task is not None:
                taskIndex = next(
                    (index for index, currentTask in enumerate(self.tasks)
                    if currentTask is task),
                    None,
                )

                if taskIndex is None:
                    pglMessages.warning(
                        f"Could not find matching task for {task.settings.taskName}"
                    )
                    return emptyResult

            # Directory containing this session's saved runs.
            dataPath = (
                Path(self.settings.dataPath).expanduser()
                / self.experimentSettings.experimentSaveName
                / self.experimentSettings.subjectID
                / self.experimentSettings.sessionName
            )

            if not dataPath.is_dir():
                return emptyResult

            requiredFiles = (
                "experimentSettings.json",
                "settings.json",
                "state.json",
                "data.json",
            )

            def isRunDirectory(directory):
                return directory.is_dir() and all(
                    (directory / fileName).is_file()
                    for fileName in requiredFiles
                )

            # Look for runDirs
            runDirs = (
                directory
                for directory in dataPath.iterdir()
                if isRunDirectory(directory)
            )

            runs = [
                pglRun(fullDataPath=runDir)
                for runDir in runDirs
            ]

            if not runs:
                return emptyResult

            runs.sort(key=lambda run: run.data.startTime, reverse=True)

            # Only inspect the latest run unless all runs were requested.
            selectedRuns = runs if allRuns else runs[:1]

            if task is None:
                return selectedRuns if allRuns else selectedRuns[0]

            matchingTasks = []
            for run in selectedRuns:
                savedTask = run.getTaskAt(taskIndex)
                if savedTask is not None:
                    matchingTasks.append(savedTask)

            if allRuns:
                return matchingTasks

            return matchingTasks[0] if matchingTasks else None
            
        except Exception as e:
            pglMessages.warning(f"Unable to load last run: {e}")

    def getLastRunParameters(self, task=None):
        '''
        Get the parameters from the last run
        '''
        lastRun = self.getLastRun(task)
        if lastRun: 
            return lastRun.parameters
        else:
            None
            
    def printLastRuns(self, task):
        '''
        displays last runs in session
        eex
        Args:
            task (pglTask): If set will display how many trials were run in specified task
            
        '''
        try:
            self.setVerbose(verbose=False)
            tasks = self.getLastRun(task=task, allRuns=True)

            self.setVerbose()
            if tasks: self.pgl.printHeader("previous runs from this session", messageType='experiment')
                        
            for iRun, task in enumerate(tasks):
                self.setVerbose()
                pglMessages.print(f"{iRun:2d}:{task.settings.taskSaveName} nTrials={task.state.currentTrial:3d} ran {pglTimestamp.formatDuration(self.pgl.getSecs()-task.data.startTime)} ago", messageType='experiment')
                self.setVerbose(verbose=False)
            
            self.setVerbose()
            if tasks: self.pgl.printHeader(messageType='experiment')

            
        except Exception as e:
            pglMessages.warning(f"Unable to load last run: {e}")

    def setVerbose(self, verbose=True):
        if not self.settings.verbose:
            # only let experiment and parameter message through (warnings will also print)
            if verbose:
                pglMessages.setEnabledTypes(['experiment','parameter'])
            else:
                pglMessages.setEnabledTypes(set())
            self.pgl.verbose=False

    def endVerbose(self):
        if not self.settings.verbose:
            # Restore verbose settings
            self.pgl.verbose=True
            pglMessages.setEnabledTypes(None)
            
    ################################################################
    # Start this experiment's recording
    ################################################################
    def _startExperimentRecording(self, estimatedDuration, recordFlush, recordCommands):
        """Start capture before task configuration and bind it to this experiment."""
        if getattr(self, "_recordingId", None) is not None:
            raise RuntimeError("This experiment already owns a recording. Create a new experiment for another run.")

        self.pgl.recordingStart(
            estimatedDuration=estimatedDuration,
            recordFlush=recordFlush,
            recordCommands=recordCommands,
        )

        metadata = self.pgl._recordingMetadata
        self._recordingId = metadata["recordingId"]

        # Store identifiers, not a reference to the live experiment.
        metadata["experiment"] = {
            "experimentUUID": self.uuid,
            "experimentName": self.experimentSettings.experimentName,
            "experimentSaveName": self.experimentSettings.experimentSaveName,
            "subjectID": self.experimentSettings.subjectID,
            "sessionName": self.experimentSettings.sessionName,
            "runName": self.experimentSettings.runName,
        }

        # Filename is published only after a successful HDF5 write.
        self.data.recording = {
            "recordingId": self._recordingId,
            "filename": None,
            "formatVersion": None,
            "saveStatus": "notAttempted",
        }

    ################################################################
    # Verify recording ownership without exporting the buffers
    ################################################################
    def _getExperimentRecording(self):
        """Return owned live metadata, or None after automatic release.

        A released experiment must never access a later experiment's buffers.
        Its saved artifact is handled separately by _saveExperimentRecording().
        """
        recordingId = getattr(self, "_recordingId", None)
        if recordingId is None: return None
        if getattr(self, "_recordingReleased", False): return None

        metadata = getattr(self.pgl, "_recordingMetadata", None)
        if not isinstance(metadata, dict) or metadata.get("recordingId") != recordingId:
            raise RuntimeError("This experiment's recording was cleared or replaced; refusing to use another recording.")

        if metadata.get("experiment", {}).get("experimentUUID") != self.uuid:
            raise RuntimeError("Recording ownership does not match this experiment.")

        if getattr(self.pgl, "_recordingState", None) not in ("active", "stopped"):
            raise RuntimeError("This experiment's recording session is unavailable.")

        return metadata
    
    ################################################################
    # Save the owned recording into the run directory
    ################################################################
    def _saveExperimentRecording(self, dataPath):
        """Save owned capture or reuse its already-published artifact.

        Buffers are released separately, only after data.json successfully
        persists the recording reference.
        """
        import h5py
        from .pglRecordingFile import pglRecordingFile

        filename = (Path(dataPath) / "recording.h5").resolve()

        if getattr(self, "_recordingReleased", False):
            # Do not inspect, save, or clear the current PGL recording here:
            # it may now belong to another experiment.
            savedPath = getattr(self, "_recordingSavedPath", None)
            reference = getattr(self, "_recordingSavedReference", None)

            if savedPath is None or filename != Path(savedPath):
                raise RuntimeError("Released recording cannot be saved to a different run directory.")
            if not isinstance(reference, dict) or reference.get("recordingId") != self._recordingId:
                raise RuntimeError("Released recording has no valid saved reference.")
            if not filename.is_file():
                raise FileNotFoundError(f"Saved recording is missing: {filename}")

            # Read only the small root metadata, not the recording arrays.
            with h5py.File(filename, "r") as file:
                if file.attrs.get("formatName") != pglRecordingFile.formatName:
                    raise RuntimeError("Saved recording artifact has an unexpected format.")
                if file.attrs.get("formatVersion") != reference["formatVersion"]:
                    raise RuntimeError("Saved recording artifact has an unexpected version.")
                metadata = pglRecordingFile._readMetadata(file)

            if (metadata.get("recordingId") != self._recordingId
                    or metadata.get("experiment", {}).get("experimentUUID") != self.uuid):
                raise RuntimeError("Saved recording artifact does not belong to this experiment.")

            self.data.recording = dict(reference)
            return filename

        metadata = self._getExperimentRecording()
        if metadata is None: return None

        if self.pgl._recordingState != "stopped":
            raise RuntimeError("Stop acquisition before saving the experiment recording.")
        if self.pgl.commandRecording or self.pgl.flushRecording:
            raise RuntimeError("Low-level capture was restarted after session finalization.")

        # Add task-run boundaries without changing the raw recording clock.
        metadata["experimentStartTime"] = self.data.startTime
        metadata["experimentEndTime"] = self.data.endTime

        self.pgl.recordingSave(filename)

        # Small reference only: recording buffers never enter experiment JSON.
        reference = {
            "recordingId": metadata["recordingId"],
            "filename": "recording.h5",
            "formatVersion": pglRecordingFile.formatVersion,
            "saveStatus": "complete",
        }
        self.data.recording = dict(reference)

        # Retain only publication bookkeeping, not a copy of captured data.
        self._recordingSavedPath = filename
        self._recordingSavedReference = dict(reference)
        return filename
        
    ################################################################
    # release experiment recording
    ################################################################
    def _releaseExperimentRecording(self):
        """Release this experiment's saved buffers after reference publication.

        Called by save() only after data.json was successfully written.
        Never release an active, replaced, or foreign recording.
        """
        if getattr(self, "_recordingId", None) is None: return
        if getattr(self, "_recordingReleased", False): return

        reference = getattr(self.data, "recording", {})
        if (reference.get("recordingId") != self._recordingId
                or reference.get("saveStatus") != "complete"):
            raise RuntimeError("Cannot release a recording without a successful saved reference.")

        metadata = self._getExperimentRecording()
        if metadata is None or self.pgl._recordingState != "stopped":
            raise RuntimeError("Cannot release an unavailable or active experiment recording.")
        if self.pgl.commandRecording or self.pgl.flushRecording:
            raise RuntimeError("Cannot release recording while low-level capture is active.")

        self.pgl.recordingClear()
        self._recordingReleased = True

##############################################
# Settings for pglTask
##############################################
class pglTaskSettings(pglTraitSettings):
    taskName = Unicode("Default task", help="Name of the task")
    taskSaveName = Unicode("defaultTask", help="Name to use when saving task data (defaults to camelCase version of taskName)")    
    phaseNum = Int(default_value=None, allow_none=True, help="Phase number for the task. Set to None if this should run in all phases")
    seglen = List(Float(), help="List of segment lengths in seconds.")
    segmin = List(Float(), help="Minimum length of a segment.")
    segmax = List(Float(), help="Maximum length of a segment.")
    waitUntilVolumeTrigger = List(Bool(), help="List of nSegments where if set to true will run through the segment length and then wait for a volume trigger to continue.")
    nSegments = Int(help="Number of segments in the task.")
    nTrials = Float(np.inf, help="Number of trials to run for.")
    # old way of doing this - config replaces
    fixedParameters = Dict(default_value={}, help="Dictionary of fixed parameters for the task.")
    config = Instance(SimpleNamespace,args=(),kw={},help="Place for configuration variables")
    saveEyeTracker = Bool(False, help="Whether to save eye tracker events this task (if we have an eye tracker).")    
    taskID = Int(0, help="Numeric identifier for the task, used for pglExperiment to keep track of tasks.")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    # observe changes to taskName and if taskSaveName is not set
    # set taskSaveName to a camelCase version of taskName
    @observe("taskName")
    def toCamelCase(self, change) -> None:
        if self.taskSaveName == "" or self.taskSaveName == "defaultTask":
            # split taskName into words
            words = change['new'].strip().split()
            if not words:
                return
            # convert to camelCase and save as taskSaveName
            firstWord = words[0][0].lower() + words[0][1:] if words[0] else ""
            restWords = "".join(word[0].upper() + word[1:] if word else "" for word in words[1:])
            self.taskSaveName = firstWord + restWords
        
    # observe changes in seglen, segmin, segmax to keep them in sync
    @observe("seglen", "segmin", "segmax")
    def _updateSegments(self, change):

        # hold off on trait notifications while we update
        with self.hold_trait_notifications():

            # if seglen change, then just make seming and segmax the same as seglen
            if change["name"] == "seglen":
                self.segmin = list(self.seglen)
                self.segmax = list(self.seglen)

            elif change["name"] == "segmin":
                # if segmax is longer than semin, truncate it
                if len(self.segmax) > len(self.segmin):
                    self.segmax = self.segmax[:len(self.segmin)]
                
                # if segmax is shorter than segmin, extend it
                if len(self.segmax) < len(self.segmin):
                    self.segmax += self.segmin[len(self.segmax):]
                
                # ensure segmax is not less than segmin
                for i, (minVal, maxVal) in enumerate(zip(change['new'], self.segmax)):
                    self.segmax[i] = max(minVal, maxVal)
                    
                # set seglen to average of segmin/segmax
                self.seglen = [(minVal + maxVal) / 2.0 for minVal, maxVal in zip(self.segmin, self.segmax)]

            elif change["name"] == "segmax":
                # if segmin is longer than semax, truncate it
                if len(self.segmin) > len(self.segmax):
                    self.segmin = self.segmin[:len(self.segmax)]
                
                # if segmin is shorter than segmax, extend it
                if len(self.segmin) < len(self.segmax):
                    self.segmin += self.segmax[len(self.segmin):]
                
                # ensure segmax is not less than segmin
                for i, (minVal, maxVal) in enumerate(zip(change['new'], self.segmin)):
                    self.segmin[i] = min(minVal, maxVal)

                # set seglen to average of segmin/segmax
                self.seglen = [(minVal + maxVal) / 2.0 for minVal, maxVal in zip(self.segmin, self.segmax)]
        
        self.nSegments = len(self.seglen)
        
        # make length of waitUntilVolumeTrigger same as nSegments
        self.waitUntilVolumeTrigger = (self.waitUntilVolumeTrigger + [False] * self.nSegments)[:self.nSegments]
    
    @observe("waitUntilVolumeTrigger")
    def _updateWaitUntilVolumeTrigger(self, change):
        # make same length as seglen
        self.waitUntilVolumeTrigger = (self.waitUntilVolumeTrigger + [False] * self.nSegments)[:self.nSegments]
    '''
    Settings for pglTask
    '''
    def __init__(self):
        super().__init__()
        
    def updateTraitsFromDict(self, data, filename="<dict>", typeConverter=None):
        """
        Override to convert parameter dicts to pglParameter instances.
        Only converts items in the 'parameters' list, not other dict values.
        """
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # ONLY convert the 'parameters' key specifically
        if 'parameters' in data and isinstance(data['parameters'], list):
            converted_params = []
            for item in data['parameters']:
                if isinstance(item, dict):
                    # Extract the two required positional arguments
                    name = item.get('name', 'unnamed')
                    validValues = item.get('validValues', [])
                    
                    # Create pglParameter with those two args
                    param = pglParameter(name, validValues)
                    
                    # Update any other attributes that might be stored
                    # (like blockNum, currentTrial, etc. from your serialized data)
                    for key, value in item.items():
                        if key not in ['name', 'validValues'] and hasattr(param, key):
                            setattr(param, key, value)
                    
                    converted_params.append(param)
                elif isinstance(item, pglParameter):
                    # Already correct type
                    converted_params.append(item)
                else:
                    pglMessages.warning(f"Warning: Unexpected type in parameters: {type(item)}",level=1)
                    converted_params.append(item)
            data['parameters'] = converted_params
        
        # Call parent implementation to handle ALL other traits normally
        pglTraitSettings.updateTraitsFromDict(self, data, filename, typeConverter)
##############################################
# State for pglTask
##############################################
class pglTaskState(pglTraitSettings):
    phaseNum = Int(default_value=None, allow_none=True, help="Current experiment phase number.")
    currentTrial = Int(default_value=0, help="Current trial number.")
    currentSegment = Int(default_value=0, help="Current segment number.")
    subjectResponses = List(Int(), help="List of subject response key codes.")
    gotResponse = Bool(default_value=False, help="whetehr the subject has responded or not")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

##############################################
# State for pglTask
##############################################
class pglTaskData(pglTraitSettings):
    startTime = Float(default_value=None, allow_none=True, help="Task start time.")
    endTime = Float(default_value=None, allow_none=True, help="Task end time.")
    events = List(Instance(pglEvent), help="List of task events.")
    params = List(Dict(), help="List of task parameter dictionaries.")
    trialVariables=List(Dict(),help="List of task variable dictionaries. One for each trial, with any variables computed or discovered by task")
    responseMapping = Dict(default_value={True: ("Correct", "green"), False: ("Incorrect", "red")}, help="response mapping for handleSubjectResponses")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    @property
    def nTrials(self):
        '''
        Compute number of trials that have been run
        '''
        nTrials = 0
        for event in self.events:
            # if we find a new trial event, reset the beginning time
            if isinstance(event, pglEventTrial):
                if event.eventType == "start":
                    nTrials += 1
        return nTrials

    def display(self, taskName="task", responseMapping=None, nTotalTrials=None, ax=None):
        '''
        Display the experiment data.
        '''
        # use responseMapping for displaying subject responses
        if responseMapping is None:
            responseMapping = self.responseMapping

        # get trial timestamps
        trialTimestamps = np.array([e.timestamp for e in self.events if isinstance(e, pglEventTrial)])
        if len(trialTimestamps) <= 2:
            pglMessages.message("Insufficient trial events found to display.")
            return
        
        # get the max trial length
        maxTrialLength = np.diff(trialTimestamps[:-1]).max()
        
        # init timeline
        timeline = timelinePlot(ax=ax, startTime=0, endTime=maxTrialLength)
        
        # init a dict for counting the number of different responseTypes found in the events
        responseCounts = {respType: 0 for respType in responseMapping}
        
        # for each event, add to timeline
        trialStart = None
        gotResponse = False
        gotOutputTrigger = False
        gotUnsentTrigger = False
        nTrials = 0
        
        for event in self.events:
            # if we find a new trial event, reset the beginning time
            if isinstance(event, pglEventTrial):
                trialStart = event.timestamp
                if event.eventType == "start":
                    nTrials += 1
            elif trialStart is not None:
                # display segment events
                if isinstance(event, pglEventSegment) and event.eventType == pglEventSegment.boundaryType.START.value:
                    timeline.addTriangleMarker(time=event.timestamp - trialStart, color='blue', label=f'{event.segmentNum}', direction='up')
                # display subject response events
                elif isinstance(event, pglEventSubjectResponse):
                    gotResponse = True
                    label, color = responseMapping.get(event.responseType, ('?', 'gray'))
                    timeline.addTriangleMarker(time=event.timestamp - trialStart, color=color, label=label[0], direction='down')   
                    # update response counts
                    if event.responseType in responseCounts:
                        responseCounts[event.responseType] += 1
                # display output trigger events
                elif isinstance(event, pglEventOutputTrigger):
                    if event.timestamp is not None:                        
                        value = getattr(event, "value", None)
                        label = f"{value}" if value is not None else 'x'
                        sent = getattr(event, "triggerSent", True)
                        if sent:
                            gotOutputTrigger = True
                        else:
                            gotUnsentTrigger = True
                        # display triangle
                        timeline.addTriangleMarker(time=event.timestamp - trialStart, color='purple' if sent else 'orange', label=label, direction='down')                        
                        
        # compute duration
        if self.endTime is not None and self.startTime is not None:
            duration = pglTimestamp.formatDuration(self.endTime-self.startTime)
        else:
            duration = ""
        
        if nTotalTrials is not None and not np.isinf(nTotalTrials):
            trials = f"{nTrials}/{int(nTotalTrials)}"
        else:
            trials = f"{nTrials}"
            
        timeline.setTitle(f"{taskName}: {trials} trials, {duration}")
        
        # Display legend.
        legend = [{'label': 'Segment', 'color': 'blue'}]

        if gotResponse:
            totalResponses = sum(responseCounts.values())
            for respType, (label, color) in responseMapping.items():
                count = responseCounts.get(respType, 0)
                if count != 0:
                    percent = count / totalResponses * 100 if totalResponses > 0 else 0
                    legend.append({'label': f'{label} (n={count}: {percent:.1f}%)', 'color': color})
        
        if gotOutputTrigger:
            legend.append({'label': 'Output trigger', 'color': 'purple'})
        
        if gotUnsentTrigger:
            legend.append({'label': 'Output trigger', 'color': 'orange'})

        timeline.addLegend(legend)
        if ax is None:
            timeline.show()
##############################################
# Task base 
##############################################
class pglTaskBase(pglTraitSettings):
    settings = Instance(pglTaskSettings, allow_none=True, help="Settings for the task")
    state = Instance(pglTaskState, allow_none=True, help="State of task")
    data = Instance(pglTaskData, allow_none=True, help="Data of task")
    parameters = List(Instance(pglParameter), default_value=[], help="parameters of task")
  
    def save(self, dataPath, filesystem=None, filesystemPrefix=None):
        """Save current task contents, including parameters, through fsspec.

        dataPath is the destination run directory.
        Does not read or copy original acquisition files.
        """
        filesystem, dataPath, _ = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            create=True,
        )

        if filesystem is None:
            raise ValueError("Could not resolve the task destination filesystem.")

        taskPath = posixpath.join(dataPath, self._getTaskDirectoryName())
        filesystem.makedirs(taskPath, exist_ok=True)

        self.settings.save(
            filename=posixpath.join(taskPath, "settings.json"),
            filesystem=filesystem,
        )
        self.state.save(
            filename=posixpath.join(taskPath, "state.json"),
            filesystem=filesystem,
        )
        self.data.save(
            filename=posixpath.join(taskPath, "data.json"),
            filesystem=filesystem,
        )

        parameterPath = posixpath.join(taskPath, "parameters")
        filesystem.makedirs(parameterPath, exist_ok=True)

        for parameter in self.parameters:
            parameter.save(dataPath=parameterPath, filesystem=filesystem)

        pglMessages.message(
            f"Saved task {self.settings.taskName} to {taskPath}"
        )

    def saveBundleStrict(self, dataPath, filesystem=None, filesystemPrefix=None):
        """Strictly save current task components and parameter bundles.

        dataPath is the destination run directory.
        Errors propagate; previously written output may remain.
        """
        components = {}
        for name in ("settings", "state", "data"):
            component = getattr(self, name, None)
            if component is None: raise ValueError(f"Cannot save task: missing {name}")
            if not callable(getattr(component, "saveStrict", None)):
                raise TypeError(f"Cannot save task {name}: no saveStrict() method")
            components[name] = component

        parameters = list(self.parameters)
        for parameter in parameters:
            if not callable(getattr(parameter, "saveBundleStrict", None)):
                raise TypeError(f"Cannot save parameter {type(parameter).__name__}: no saveBundleStrict() method")

        taskDirectory = self._getTaskDirectoryName()
        filesystem, dataPath, _ = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            create=True,
        )
        if filesystem is None: raise ValueError("Could not resolve task destination filesystem")

        taskPath = posixpath.join(dataPath, taskDirectory)
        filesystem.makedirs(taskPath, exist_ok=True)

        for name, component in components.items():
            component.saveStrict(filename=posixpath.join(taskPath, f"{name}.json"), filesystem=filesystem)

        parameterPath = posixpath.join(taskPath, "parameters")
        filesystem.makedirs(parameterPath, exist_ok=True)

        for parameter in parameters:
            parameter.saveBundleStrict(dataPath=parameterPath, filesystem=filesystem)

        return taskPath

    def _getTaskDirectoryName(self):
        """Return the directory name used to save this task."""

        #taskDirectoryName = f"task{self.settings.taskID:02d}_{self.settings.taskSaveName}"
        taskDirectoryName = self.getTaskDirectoryName(taskID=self.settings.taskID-1,taskName=self.settings.taskSaveName)

        return taskDirectoryName
    
    @classmethod
    def getTaskDirectoryName(cls, taskID, taskName):
        
        taskDirectoryName = f"task{taskID+1:02d}_{taskName}"
        return taskDirectoryName
    
    @classmethod
    def load(cls, dataPath, filesystem=None):
        '''
        Load the task data.
        '''
        pglMessages.message(f"Loading task data from: {dataPath}", messageType='detailed')

        # validate filesystem
        filesystem, taskPath, _ = pglStorage.validateFilesystem(filesystem, dataPath)

        # load settings, state and data
        try:
            settings = pglSerialize.load(filename=f"{taskPath}{filesystem.sep}settings.json", filesystem=filesystem)
            state = pglSerialize.load(filename=f"{taskPath}{filesystem.sep}state.json", filesystem=filesystem)
            data = pglSerialize.load(filename=f"{taskPath}{filesystem.sep}data.json", filesystem=filesystem)
        except Exception as e:
            pglMessages.warning(f"Could not load task data from {taskPath}: {e}")
            return None

        # load parameters
        parameters = []
        parametersDir = f"{taskPath}{filesystem.sep}parameters"
        try:
            for paramDir in filesystem.ls(parametersDir, detail=False):
                param = pglParameter.from_file(paramDir, filesystem=filesystem)
                if param is None:
                    pglMessages.warning(f"Skipping parameter that failed to load: {paramDir}")
                    continue
                parameters.append(param)
        except Exception as e:
            pglMessages.warning(f"Could not load task parameters from {parametersDir}: {e}")
            return None

        # instantiate class
        obj = cls()
        obj.settings = settings
        obj.state = state
        obj.data = data
        obj.parameters = parameters
        return obj

    def display(self, ax=None):
        '''
        Display the task data
        '''
        self.data.display(taskName=self.settings.taskName, nTotalTrials=self.settings.nTrials, ax=ax)
    
    def print(self):
        '''
        Print a summary of the task data
        '''
        from .pglTimestamp import pglTimestamp
        timestamp = pglTimestamp()
        
        if self.data.startTime is None:
            pglMessages.message(f"Task {self.settings.taskName} was never run")
            return
        
        if self.data.endTime is None:
            pglMessages.message(f"Task {self.settings.taskName} has no endTime")
            return
        
        # print task name and number of trials
        pglMessages.print(f"Task: {self.settings.taskName} | Trials: {self.state.currentTrial+1}")
        pglMessages.print(f"Duration={timestamp.formatDuration(self.data.endTime - self.data.startTime)} | startTime={self.data.startTime} | endTime={self.data.endTime}")

        # print seglen and waitFor
        pglMessages.print(f"seglen={self.settings.seglen}")
        pglMessages.print(f"waitUntilVolumeTrigger={self.settings.waitUntilVolumeTrigger}")
                
        # print fixedParameters
        pglMessages.print('\n'.join(f"{key}={value}" for key, value in self.settings.fixedParameters.items()))
        pglMessages.print('-' * 40)
        

        # print parameters
        for p in self.parameters:
            pglMessages.print(f"{p.settings.name}")
        
        # print trial by trial information
        for iTrial, params in enumerate(self.data.params):
            # find matching trial event
            trialEvent = next((event for event in self.data.events if isinstance(event, pglEventTrial) and event.trialNum == iTrial), None)
            trialStart = trialEvent.timestamp-self.data.startTime if trialEvent else "No trial event found"
            if hasattr(self,'e'):
                trialVolume = self.e.getNearestVolumeTrigger(trialEvent)
            else:
                trialVolume = None
            if trialVolume is None:
                pglMessages.print(f"Trial {iTrial+1} at {trialStart:.2f}s: " + ', '.join(f"{key}={value}" for key, value in params.items()))
            else:
                pglMessages.print(f"Trial {iTrial+1} at {trialStart:.2f}s (vol={trialVolume}): " + ', '.join(f"{key}={value}" for key, value in params.items()))
                          
##############################################
# Task class
##############################################
class pglTask(pglTaskBase):
    # this are set every trial, which allows us to
    # randomize the length of each segment (based on segmin/segmax)
    # or jump segment, by dynamically changing from Inf to current time
    _thisTrialSeglen = []
    
    # reference to pgl, set by pglExperiment when added
    pgl = None
    
    '''
    Class representing a task in the experiment. For example, a fixation task. Or
    a stimulus task which controls when and what stimuli are presented
    '''
    def __init__(self, pgl=None, phaseNum=0):
        self.pgl = pgl
        self.settings = pglTaskSettings()
        self.state = pglTaskState()
        self.data = pglTaskData()
        self.parameters: List[pglParameter] = []
        
        # default segment length
        self.settings.seglen = [1.0]
        
        # set phaseNum
        self.settings.phaseNum = phaseNum
        
        # these get set by update
        self.tasks = None
        self.e = None
        self.waitUntilVolumeTrigger = False

    def configure(self, e):
        '''
        Configure gets called on tasks, just after initScreen runs so that tasks can use
        any information needed from the pglExperiment to initialize, 
        '''
        pass

    def start(self, startTime):
        '''
        Start the task.
        '''
        # if task is already started, then do nothing
        if self.data.startTime is not None:
            return

        # set task start time
        self.data.startTime = startTime
        
        # run overrideable startTask methos
        if self.startTask(startTime) == False:
            return False
        
        # start trial
        self.state.currentTrial = -1
        self.startTrial(startTime)
        
        return True

    def startTask(self, startTime):
        '''
        Can be overridden in subclasses to add custom behavior when the task starts.
        Must return True to run the task as intended, or False to abort running the task
        '''
        return True
    
    def startSegment(self, updateTime):
        '''
        Called exactly once, each time a new segment genuinely starts.
        Override in subclasses to add custom per-segment behavior
        (e.g. loading a stimulus). Base implementation does nothing.
        '''    
        pass

    def _startSegment(self, updateTime):
        '''
        Internal control logic for advancing to the next segment or ending the trial.
        Not meant to be overridden — override startSegment() instead for custom
        per-segment behavior.
        '''
        if self.state.currentSegment >= 0 and self._thisTrialSeglen[self.state.currentSegment] == 0:
            self._thisTrialSeglen[self.state.currentSegment] = updateTime - self.state.segmentStartTime

        if (self.state.currentSegment + 1) >= self.settings.nSegments:
            self.endTrial(updateTime)
            if self.done(self.state.currentTrial + 1):
                self.state.currentTrial += 1
            else:
                self.startTrial(updateTime)   # recurses into _startSegment internally, not startSegment
        else:
            self.state.currentSegment += 1
            self.state.segmentStartTime = updateTime
            self.data.events.append(pglEventSegment(self.state.currentSegment, updateTime))

            if self.settings.saveEyeTracker:
                self.e.saveEyeTrackerEvent(eventType="segment", taskID=self.settings.taskID,
                    trialNum=self.state.currentTrial, segmentNum=self.state.currentSegment,
                    timestamp=updateTime, phaseNum=self.settings.phaseNum)

            self.waitUntilVolumeTrigger = False
            self.startSegment(updateTime)
            
    def startTrial(self, startTime):
        '''
        Start a trial.
        '''
        self.state.currentTrial += 1
        self.state.currentSegment = -1
        self.data.events.append(pglEventTrial(self.state.currentTrial, startTime))
        self.state.trialStartTime = startTime

        if self.settings.saveEyeTracker:
            self.e.saveEyeTrackerEvent(eventType="trial", taskID=self.settings.taskID,
                trialNum=self.state.currentTrial, segmentNum=self.state.currentSegment,
                timestamp=startTime, phaseNum=self.settings.phaseNum)

        # parameters and trial variables
        self.data.params.append({})
        self.currentParams = self.data.params[-1]
        for parameter in self.parameters:
            self.data.params[-1].update(parameter.get())
        self.data.trialVariables.append({})

        # draw segment lengths before any segment starts
        self._thisTrialSeglen = [
            float('inf') if math.isinf(min_val) or math.isinf(max_val)
            else random.uniform(min_val, max_val)
            for min_val, max_val in zip(self.settings.segmin, self.settings.segmax)
        ]

        # print trial header before the segment hook can print anything
        pglMessages.print(f"({self.settings.taskName}) Trial {self.state.currentTrial+1}: ", end='', messageType='experiment')
        for name, value in self.data.params[-1].items():
            pglMessages.print(f'{name}={value}', end=' ', messageType='experiment')
        pglMessages.print("", messageType='experiment')

        # now start the first segment (currentSegment -1 -> 0)
        self._startSegment(startTime)

    def endTrial(self, endTime):
        '''
        End a trial.
        '''

    def addParameter(self, param):
        '''
        Add a parameter to the task.
        '''
        self.parameters.append(param)

    def update(self, updateTime, subjectResponses, phaseNum, tasks, events):
        '''
        Update the task.
        '''
        if self.data.endTime is not None: return
        # store references
        self.state.subjectResponses = []
        self.state.phaseNum = phaseNum
        
        # custom handling of events
        self.handleEvents(events)
        
        # check for end of segment
        if self.waitUntilVolumeTrigger:
            if self.e.state.volumeNumber > self.lastVolumeNumber:
                # volume trigger received, end segment
                self._startSegment(updateTime)
        if  updateTime - self.state.segmentStartTime >= self._thisTrialSeglen[self.state.currentSegment]:
            # check if we need to wait until volume trigger
            if self.settings.waitUntilVolumeTrigger[self.state.currentSegment]:
                self.waitUntilVolumeTrigger = True
                self.lastVolumeNumber = self.e.state.volumeNumber
            else:
                # call startSegment to begin next segment
                self._startSegment(updateTime)
        
        # if there are responses, call response callback
        if subjectResponses != []:
            # Pass each subjectResponse in sequence to handleSubjectResponse
            for subjectResponse, timestamp in subjectResponses:
                # adding subject response to state
                self.state.subjectResponses.append(subjectResponse)
                # call the subject response handler
                responseType = self.handleSubjectResponse(subjectResponse, timestamp)
                # save as an event if responseType is not None
                # responseType can be used to specify different types of responsees
                # and is defined by the subclass
                if responseType is not None:
                    self.data.events.append(pglEventSubjectResponse(response=subjectResponse, timestamp=timestamp, responseType=responseType))
                
        # update the screen
        self.updateScreen()


    def handleSubjectResponse(self, response, updateTime) -> None:
        '''
        Handle subject responses. To handle subject responses, override this method
        If you provide a return value (e.g. 1 or 0, or 'correct'/'incorrect'), then
        that value will be stored in the pglEventSubjectResponse event.
        '''
        pass
    
    def handleEvents(self, events) -> None:
        '''
        Handle keyboard/mouse events. For subclasses that need to handle keyboard or mouse
        events (for example to handle typing text), subclass this method.
        '''
        pass
    
    def updateScreen(self):
        '''
        Update the screen.
        '''
        pass
    
    def done(self, trialNum=None):
        '''
        Check if the task is done.
        '''
        # check current trial number by default
        if trialNum is None: trialNum = self.state.currentTrial
        # check if we are done
        taskDone = trialNum >= self.settings.nTrials
        if taskDone: self.end()
        return taskDone

    def end(self):
        '''
        end of task
        '''
        # Guard against calling end() twice
        if self.data.endTime is not None: return

        # call end task
        self.endTask()
        
        # record end time
        pglMessages.message(f"Ending task {self.settings.taskName}",messageType='experiment')
        endTime = self.pgl.getSecs()
        self.data.endTime = endTime
        # put in time stamps for end of last segment and trial
        self.data.events.append(pglEventSegment(self.state.currentSegment, endTime, eventType=pglEventSegment.boundaryType.END))
        self.data.events.append(pglEventTrial(self.state.currentTrial, endTime, eventType=pglEventTrial.boundaryType.END))

    def endTask(self):
        '''
        subclass overrideable function that gets called for any end of task processing
        '''
        pass
    
    def jumpSegment(self):
        '''
        Jump to the next segment.
        '''
        # set current segment length to 0 to force jump
        self._thisTrialSeglen[self.state.currentSegment] = 0


##############################################
# small class to carry information about digital IO words
##############################################
class pglDigitalIOWord(pglTraitSettings):
    # channel
    key = Unicode("", help='String that can be used by sendTriggerWord to set word value')
    value = Int(help="Word that will be sent")
    pulseLen = Int(help="length of pulse")
    deviceType = Unicode("", help="type of device that the channel was configured on")
    
##############################################
# small class to carry information about digitalIOChannels
##############################################
class pglDigitalIOChannel(pglTraitSettings):

    channelKey = Unicode(help="Unique trigger name associated with the logical channel")
    channel = Int(help="Channel number that has been configured")
    channelName = Unicode(allow_none=True, default_value=None, help="Name of channel if available")
    groupName = Unicode(allow_none=True, help='Name of channel group')
    pulseLen = Int(help="length of pulse that has been configured")
    deviceType = Unicode("", help="type of device that the channel was configured on")
    
##############################################
# Settings for pglExperiment
##############################################
class pglExperimentSettings(pglTraitSettings):
    experimentName = Unicode("Default experiment", help="Name of the experiment")
    sessionName = Unicode("", help="Session name of experiment")
    runName = Unicode("", help="Name of run of experiment")
    experimenterName = Unicode("", help="Name of experimenter who ran in experiment")
    experimentSaveName = Unicode("defaultExperiment", help="Name to use when saving experiment data (defaults to camelCase version of experimentName)")
    subjectID = Unicode("s0000", help="Identifier for the subject participating in the experiment.")
    tasks = List(Unicode(), default_value=[], help="Task names")
    digitalIOChannels = List(Instance(pglDigitalIOChannel), help='List of digitalIO channesl that have been configured')
    digitalIOWords = List(Instance(pglDigitalIOWord), default_value=[], help='List of words, keys and associated values that have been configured')

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    # observe changes to experimentName and if experimentSaveName is not set
    # set experimentSaveName to a camelCase version of experimentName
    @observe("experimentName")
    def toCamelCase(self, change) -> None:
        if self.experimentSaveName == "" or self.experimentSaveName == "defaultExperiment":
            # split experimentName into words
            words = change['new'].strip().split()
            if not words:
                return
            # convert to camelCase and save as experimentSaveName
            self.experimentSaveName = words[0].lower() + "".join(word.capitalize() for word in words[1:])
    
    @validate("subjectID")
    def _validateSubjectID(self, proposal):
        value = proposal["value"]

        if (
            not isinstance(value, str)
            or len(value) < 2
            or value[0] != "s"
            or not value[1:].isdigit()
        ):
            raise TraitError("(experimentSettings) ❌ Error: subjectID must be in format 'sXXXX' where X is a digit.")
        return value           

##############################################
# Data for pglExperiment
##############################################
class pglExperimentData(pglTraitSettings):
    startTime = Float(0.0, help="Time in secs of start of experiment")
    startDateTime = Unicode("", help="Experiment start date and time in ISO format")
    endTime = Float(0.0, help="Time in secs of end of experiment")
    events = List(Instance(pglEvent), default_value=[], help="List of events from experiment")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    def __repr__(self):
        return f"pglExperimentData(startTime={self.startTime}, endTime={self.endTime}, {len(self.events)} events)"
    
    def getNumEvents(self, type=None, eventType=None, keyChar=None):
        # filter for type
        if type is None:
            return len(self.events)
        filteredEvents = [event for event in self.events if event.type == type]
        # filter for events
        if eventType is not None:
            filteredEvents = [event for event in filteredEvents if event.eventType == eventType]
        # filter for keyChar
        if keyChar is not None:
            filteredEvents = [event for event in filteredEvents if getattr(event, "keyChar", None) == keyChar]
        return len(filteredEvents)
    
    def display(self, e=None, ax=None):
        '''
        Display the experiment data.
        '''
        if len(self.events) == 0:
            pglMessages.print("(pglExperimentData) No events to display.")
            return
        
        # get info from experiment if provided
        if e is not None:
            self.volumeTriggerKey = e.settings.volumeTriggerKey
        else:
            #settings = pglSettingsManager.getSettings()
            #self.volumeTriggerKey = settings.volumeTriggerKey 
            self.volumeTriggerKey = "`"
        
        # Get the time at which start the timeline, if there is a keyboard
        # event that happens before the start of the experiment (like when the experimenter
        # hits space to start the experiment), then adjust the start time to show that as a negative time)
        startTime = 0
        firstKeydownEvent = next((event for event in self.events if event.type == "keyboard" and event.eventType == "keydown"), None)
        if firstKeydownEvent is not None:
            if firstKeydownEvent.timestamp < self.startTime:
                startTime = firstKeydownEvent.timestamp - self.startTime
            
        # track number of volumes
        nVols = 0
        nKeys = 0
        nOutputTriggers = 0
        nUnsentTriggers = 0
        
        # init timeline
        timeline = timelinePlot(ax=ax, startTime=startTime, endTime=max(self.endTime-self.startTime,10))
        # for each event, add to timeline
        for event in self.events:
            if event.type == "keyboard":
                if event.eventType == "keydown":
                    if event.keyChar != self.volumeTriggerKey:
                        timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='green', label=f'{event.keyChar}', direction='down')
                        nKeys += 1
                elif (event.keyChar == "escape"):
                    timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='red', label=f'{event.keyChar}', direction='down')
                    nKeys += 1
            elif event.type == "volumeTrigger":
                timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='blue', direction='up')
                nVols += 1
            elif event.type == "outputTrigger":
                if event.timestamp is not None:
                    value = getattr(event, "value", None)
                    label = f"{value}" if value is not None else 'x'
                    sent = getattr(event, "triggerSent", True)
                    if sent:
                        nOutputTriggers += 1
                    else:
                        nUnsentTriggers += 1
                    # set marker
                    timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='purple' if sent else 'orange', label=label, direction='up') 
        
        if self.startDateTime != "":
            dateTime = datetime.fromisoformat(self.startDateTime)
            readableTime = dateTime.strftime("%I:%M %p").lstrip("0")
            timeline.setTitle(f"Experiment Events: {dateTime:%b. %d, %Y} at {readableTime}")
        else:
            timeline.setTitle("Experiment Events")
            
        legend = [{'label': f'Keypress (n={nKeys})', 'color': 'green'}]
        
        if nVols > 0:
            legend.append({'label': f'Volumes (n={nVols})', 'color': 'blue'})
        
        if nOutputTriggers > 0:
            legend.append({'label': f'Output triggers (n={nOutputTriggers})', 'color': 'purple'})

        if nUnsentTriggers > 0:
            legend.append({'label': f'Unsent triggers (n={nUnsentTriggers})', 'color': 'orange'})

        timeline.addLegend(legend)
        
        if not ax: timeline.show()
    def getTriggerStats(self):
        '''
        Get the median time between volume triggers.
        '''
        # get all volume trigger events
        volumeTriggerEvents = [event for event in self.events if event.type == "volumeTrigger"]
        # get the timestamps of the volume trigger events
        timestamps = [event.timestamp for event in volumeTriggerEvents]
        # get the differences between the timestamps
        diffs = np.diff(timestamps)
        # return the median of the differences
        return Stats(
            mean=np.mean(diffs) if len(diffs) > 0 else None,
            median=np.median(diffs) if len(diffs) > 0 else None,
            std=np.std(diffs) if len(diffs) > 0 else None,
            min=np.min(diffs) if len(diffs) > 0 else None,
            max=np.max(diffs) if len(diffs) > 0 else None
        )
    
##############################################
# State for pglExperiment
##############################################
##############################################
# State for pglExperiment
##############################################
class pglExperimentState(pglTraitSettings):
    phaseNum = Int(default_value=None, allow_none=True, help="Current experiment phase number.")
    phaseNums = List(trait=Int(), default_value=None, allow_none=True, help="List of experiment phase numbers.")
    currentPhaseIndex = Int(default_value=0, help="Index into phaseNums for the current phase.")

    openScreen = Bool(default_value=False, help="Whether the experiment screen is currently open.")
    runFinishedWithError = Bool(default_value=False, help="Whether experiment acquisition ended with an error.")
    volumeNumber = Int(default_value=0, help="Current scanner-volume number.")
    experimentStarted = Bool(default_value=False, help="Whether the experiment has started.")
    experimentDone = Bool(default_value=False, help="Whether the experiment has completed.")

    # None means the corresponding control key is disabled.
    responseKeyCodesList = List(Int(), help="List of subject response key codes.")
    startKeyCode = Int(default_value=None, allow_none=True, help="Key code used to begin the experiment; None disables the key.")
    endKeyCode = Int(default_value=None, allow_none=True, help="Key code used to end the experiment; None disables the key.")
    volumeTriggerKeyCode = Int(default_value=None, allow_none=True, help="Scanner-volume trigger key code; None disables the key.")

    originalGammaTable = Tuple(List, List, List, default_value=None, allow_none=True, help="Original display gamma table: (red, green, blue).")
    gammaTable = Tuple(List, List, List, default_value=None, allow_none=True, help="Current display gamma table: (red, green, blue).")
    display = Instance(pglDisplaySettings, default_value=None, allow_none=True, help="Current display settings.")
    originalScreenResolution = Tuple(Int(), Int(), Int(), Int(), default_value=None, allow_none=True, help="Original screen resolution.")
    screenResolution = Tuple(Int(), Int(), Int(), Int(), default_value=None, allow_none=True, help="Current screen resolution.")

    digitalIODevices = List(Instance(pglDigitalIODevice), help="List of digital IO devices the experiment is using.")
    triggerWords = Dict(help="Runtime lookup of trigger word keys to configurations and devices.")
    triggers = Dict(help="Runtime lookup of trigger keys to configurations and devices.")

    # Make sure any variables added by experimenter code get saved.
    _serializeUnregisteredFields = True

##############################################
# timelinePlot
##############################################
class timelinePlot:
    """
    A timeline visualization with triangular event markers and vertical line markers.
    
    Usage:
        timeline = TimelinePlot(startTime=0, endTime=100)
        timeline.addTriangleMarker(time=10, color='red', label='Start')
        timeline.addVerticalMarker(time=50, color='blue', label='Checkpoint', labelSide='right')
        timeline.show()
    """
    
    def __init__(self, startTime=0, endTime=100, figsize=(12, 4), ax=None):
        """
        Initialize the timeline plot.
        
        Args:
            startTime (float): Start time for the timeline
            endTime (float): End time for the timeline
            figsize (tuple): Figure size (width, height)
        """
        self.startTime = startTime
        self.endTime = endTime
        
        # Create figure and axis
        if ax:
            self.fig = ax.figure
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Setup the timeline axis
        self.ax.set_xlim(startTime, endTime)
        self.ax.set_ylim(-1, 2)  # Room for markers above and below
        
        # Draw the main timeline (horizontal line)
        self.ax.axhline(y=0, color='black', linewidth=2)
        
        # Axis labels
        self.ax.set_xlabel('Time (sec)', fontsize=12)
        self.ax.set_yticks([])  # Hide y-axis ticks
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        
        # Storage for markers (for legend if needed)
        self.markers = []
 
    def addTriangleMarker(self, time, color='red', label='', labelOffset=0.3, 
                          markerSize=10, fontsize=10, direction='down'):
        """
        Add a triangle marker at a specific time with tip touching the timeline.
        
        Args:
            time (float): Time position for the marker
            color (str): Color of the triangle
            label (str): Text label for the triangle
            labelOffset (float): Vertical offset for the label from the triangle edge
            markerSize (float): Size of the triangle (height in data units)
            fontsize (int): Font size for the label
            direction (str): 'down' for downward-pointing (▼) or 'up' for upward-pointing (▲)
        """
        # Convert markerSize to data coordinates (approximate)
        height = markerSize * 0.02  # Scale factor for visual size
        width = height * 0.8  # Make it slightly narrower
        
        # Create triangle vertices based on direction
        if direction == 'down':
            # Downward triangle: tip at timeline, base above
            vertices = [
                [time, 0],                    # Tip at timeline
                [time - width/2, height],     # Top left
                [time + width/2, height],     # Top right
            ]
            labelY = height + labelOffset
            labelVa = 'bottom'
        else:  # direction == 'up'
            # Upward triangle: tip at timeline, base below
            vertices = [
                [time, 0],                    # Tip at timeline
                [time - width/2, -height],    # Bottom left
                [time + width/2, -height],    # Bottom right
            ]
            labelY = -height - labelOffset
            labelVa = 'top'
        
        # Draw triangle as polygon
        triangle = patches.Polygon(vertices, closed=True, 
                                  facecolor=color, edgecolor='black', 
                                  linewidth=0.5, clip_on=False)
        self.ax.add_patch(triangle)
        
        # Add label if provided
        if label:
            self.ax.text(time, labelY, label, 
                        ha='center', va=labelVa, fontsize=fontsize,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.8))
        
        self.markers.append({'type': 'triangle', 'time': time, 'label': label, 
                            'color': color, 'direction': direction})
    
    def addVerticalMarker(self, time, color='blue', label='', labelSide='right',
                          lineHeight=1.5, linewidth=2, fontsize=9, rotation=90):
        """
        Add a vertical line marker at a specific time.
        
        Args:
            time (float): Time position for the marker
            color (str): Color of the vertical line
            label (str): Text label for the marker
            labelSide (str): 'left' or 'right' - side for the label
            lineHeight (float): Height of the vertical line
            linewidth (float): Width of the vertical line
            fontsize (int): Font size for the label
            rotation (int): Text rotation (90 for vertical, 0 for horizontal)
        """
        # Draw vertical line
        yBottom = -lineHeight / 2
        yTop = lineHeight / 2
        self.ax.plot([time, time], [yBottom, yTop], 
                    color=color, linewidth=linewidth)
        
        # Add label if provided
        if label:
            # Position label on left or right
            if labelSide == 'right':
                xOffset = 0.02 * (self.endTime - self.startTime)  # 2% of range
                ha = 'left'
            else:  # left
                xOffset = -0.02 * (self.endTime - self.startTime)
                ha = 'right'
            
            self.ax.text(time + xOffset, 0, label,
                        ha=ha, va='center', fontsize=fontsize,
                        rotation=rotation, color=color)
        
        self.markers.append({'type': 'vertical', 'time': time, 'label': label, 'color': color})
    
    def addTimeRange(self, start, end, color='lightgray', alpha=0.3, label=''):
        """
        Add a shaded time range (useful for highlighting periods).
        
        Args:
            start (float): Start time of the range
            end (float): End time of the range
            color (str): Color of the shaded region
            alpha (float): Transparency (0-1)
            label (str): Optional label for the range
        """
        self.ax.axvspan(start, end, color=color, alpha=alpha, label=label)
    
    def setTitle(self, title, fontsize=14):
        """Set the plot title."""
        self.ax.set_title(title, fontsize=fontsize, fontweight='bold')
    
    def addLegend(self, items, location='upper right', fontsize=10):
        """
        Add a legend with colored text labels (no symbols).
        
        Args:
            items (list): List of dicts with 'label' and 'color' keys
                         Example: [{'label': 'Keypress', 'color': 'red'}, ...]
            location (str): Legend location ('upper right', 'upper left', 'lower right', 'lower left', etc.)
            fontsize (int): Font size for legend text
        """
        from matplotlib.lines import Line2D
        
        # Create dummy line objects with the colors
        handles = []
        labels = []
        
        for item in items:
            # Create invisible line with the desired color
            handle = Line2D([0], [0], marker='', linestyle='', 
                          markersize=0, color=item['color'])
            handles.append(handle)
            labels.append(item['label'])
        
        # Create legend
        legend = self.ax.legend(handles, labels, loc=location, 
                               fontsize=fontsize, framealpha=0.9,
                               handlelength=0, handletextpad=0.5,
                               labelcolor='linecolor')  # KEY: Use line color for labels
        
        # Make text bold (optional)
        for text in legend.get_texts():
            text.set_weight('bold') 
    
    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()
    
    def save(self, filename, dpi=300):
        """
        Save the plot to a file.
        
        Args:
            filename (str): Output filename
            dpi (int): Resolution in dots per inch
        """
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        pglMessages.message(f"Timeline saved to {filename}")
    
    def getMarkers(self):
        """Return list of all markers added."""
        return self.markers

#################################################################
# Events that specify trial timing
#################################################################
class pglEventTrial(pglEvent):

    class boundaryType(Enum):
        START = 'start'
        END = 'end'

    def __init__(self, trialNum=None, timestamp=None, eventType=None):
        super().__init__(type="trial")

        # handle default
        if eventType is None:
            eventType = self.boundaryType.START
            
        # set attributes
        self.trialNum = trialNum
        self.timestamp = timestamp
        self.eventType = eventType.value

    def print(self):
        pglMessages.message(f"Trial {self.eventType} at: {self.timestamp}")
        
#################################################################
# Events that specify segment timing
#################################################################
class pglEventSegment(pglEvent):

    class boundaryType(Enum):
        START = 'start'
        END = 'end'

    def __init__(self, segmentNum = None, timestamp=None, eventType=None):
        super().__init__(type="segment")

        # handle default
        if eventType is None:
            eventType = self.boundaryType.START
        
        # set attributes
        self.segmentNum = segmentNum
        self.eventType = eventType.value
        self.timestamp = timestamp

    def print(self):
        pglMessages.message(f"Segment {self.segmentNum} {self.eventType} at: {self.timestamp}")
        

#################################################################
# Events that specify subject response
#################################################################
class pglEventSubjectResponse(pglEvent):
    
    def __init__(self, response=None, timestamp=None, responseType=None):
        super().__init__(type="subjectResponse")
        
        # set attributes
        self.response = response
        self.timestamp = timestamp
        self.responseType = responseType

#################################################################
# Events that specifys mri volume trigger
#################################################################
class pglEventVolumeTrigger(pglEvent):
    
    def __init__(self, timestamp=None):
        super().__init__(type="volumeTrigger")
        
        # set attributes
        self.timestamp = timestamp

#################################################################
# Events that specifys a trigger was sent
#################################################################
class pglEventOutputTrigger(pglEvent):
    
    def __init__(self, triggerName=None, triggerValue=None, triggerWord=None, timestamp=None, triggerSent=True):
        super().__init__(type="outputTrigger")
        
        # set attributes
        self.timestamp = timestamp
        self.triggerName = triggerName
        self.triggerSent = triggerSent

        # set value
        if triggerValue is not None:
            self.value = triggerValue
            self.triggerWord = False
        elif triggerWord is not None:
            self.value = triggerWord
            self.triggerWord = True
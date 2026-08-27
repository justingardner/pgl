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
from .pglKeyboardMouse import pglKeyboardMouse
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
from . import pglTimestamp
from .pglEyeTracker import pglEyeTracker
from .pglEyelink import pglEyelink, pglEyelinkData
from .pglSettings import pglDisplaySettings, pglTraitSettings, pglStateDataSettings
from .pglMessages import pglMessages
import fsspec
import posixpath
from .pglBase import pglBase
from .pglDialog import pglDialogs
from types import SimpleNamespace

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
        '''
        Check whether the experiment dir has all the correct files
        '''
        from .pglPipeline import pglChoose
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
            dataPath=dataPath)

        sep = filesystem.sep

        # Top-level required JSON files
        requiredExperimentFiles = [
            "data.json", "experimentSettings.json", "settings.json",
            "pgl.json", "state.json"
        ]
        requiredFiles = ["data.json", "settings.json", "state.json"]

        # Check top-level JSON files
        for fileName in requiredExperimentFiles:
            filePath = f"{fullDataPath}{sep}{fileName}"
            if not filesystem.isfile(filePath):
                if verbose:
                    pglMessages.message(f"Missing file: {filePath}")
                return False

        # Check non-json entries
        for item in filesystem.ls(fullDataPath, detail=False):
            fileName = item.rsplit(sep, 1)[-1]

            if not fileName.endswith(".json"):

                # Everything non-json should be a directory
                if not filesystem.isdir(item):
                    if verbose:
                        pglMessages.message(f"Expected directory, found: {item}")
                    return False

                # Check task files only if any task file exists
                taskFilesFound = [
                    fileName for fileName in requiredFiles
                    if filesystem.isfile(f"{item}{sep}{fileName}")
                ]

                if len(taskFilesFound) > 0:

                    # If one exists, all must exist
                    for fileName in requiredFiles:
                        filePath = f"{item}{sep}{fileName}"
                        if not filesystem.isfile(filePath):
                            if verbose:
                                pglMessages.message(
                                    f"Missing file in task directory {item}: {fileName}")
                            return False

                    # parameters must be a directory (may be empty)
                    parametersPath = f"{item}{sep}parameters"
                    if not filesystem.isdir(parametersPath):
                        if verbose:
                            pglMessages.message(
                                f"No parameters directory in {item}")
                        return False

                    # Any directories inside parameters must contain required task files
                    for subItem in filesystem.ls(parametersPath, detail=False):
                        if not filesystem.isdir(subItem):
                            continue

                        for fileName in requiredFiles:
                            filePath = f"{subItem}{sep}{fileName}"
                            if not filesystem.isfile(filePath):
                                if verbose:
                                    pglMessages.message(
                                        f"Missing file in parameter directory {subItem}: {fileName}")
                                return False

        return True  

    @classmethod
    def load(cls, settings=None, dataPath=None, experimentName=None, subjectID=None, sessionName=None, runName=None, fullDataPath=None, filesystem=None, filesystemPrefix=None):
        '''
        Load the experiment settings, state and data.         
        '''
        # get the experiment directory
        from .pglPipeline import pglChoose
        filesystem, experimentPath, filesystemPrefix = pglChoose.getExperimentPath(settings=settings, dataPath=dataPath, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, fullDataPath=fullDataPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix)
        if filesystem is None:
            if dataPath is not None:
                pglMessages.message(f"Could not locate experiment directory {dataPath}")
            return
 
        # call parent class to load the experiment data, settings, and state
        print(f"(pglExperiment:load) Loading experimentdata from: {experimentPath}")
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
                taskPath = f"{experimentPath}{filesystem.sep}{taskName}"

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
                    pglMessages.warning(f"Could not find task {taskName}: taskPath")
        
        # return the created object
        return obj
       
    def addTask(self, task, addToTaskList=True):
        '''
        Add a task to the experiment.
        '''
        # give it a reference to pgl and experiment
        task.pgl = self.pgl
        task.e = self
        self.nTasks += 1
        task.taskID = self.nTasks

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

    def print(self):
        '''
        Print a summary of the experiment events.
        '''
        from pgl import pglTimestamp
        timestamp = pglTimestamp()
        # print separator
        print("=" * 80)
        
        # print experiment name, subject ID, and duration
        print(f"Experiment: {self.experimentSettings.experimentName} | Subject ID: {self.experimentSettings.subjectID}")
        print(f"Duration: {timestamp.formatDuration(self.experimentDuration())}")
        
        # FIX, FIX, FIX - old way
        #displayInfo = f"Display: {self.settings.displayName[0] if self.settings.displayName and len(self.settings.displayName) > 0 else 'Unknown'} "
        #displayInfo += f"{self.pglState.screenWidthPixels}x{self.pglState.screenHeightPixels} @ {self.pglState.frameRate}Hz "
        #displayInfo += f"{self.pglState.screenWidthDegrees:.2f}x{self.pglState.screenHeightDegrees:.2f} deg "
        #displayInfo += f"{self.settings.displayWidth:.2f}x{self.settings.displayHeight:.2f} cm at {self.settings.displayDistance:.2f} cm "
        #print(displayInfo)
        
        numVols = self.data.getNumEvents(type="volumeTrigger")
        print(f"Number of volume triggers: {numVols}")
        if numVols > 1:
            triggerStats = self.data.getTriggerStats()
            print(f"Median time between triggers: {triggerStats.median:.3f}s")
            print(f"Mean ± std time between triggers: {triggerStats.mean:.3f} ± {triggerStats.std:.6f}s")

        # print task names
        for taskName in self.experimentSettings.tasks:
            print(f"taskName: {taskName}")

        # print task data
        if hasattr(self, "tasks"):
            for task in self.tasks:
                # print separtor
                print("=" * 80)
                # print task
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
    
    def initScreen(self, backgroundColor=-1):
        '''
        Initialize the screen for the experiment. This will call pgl.open() and
        set parameters according to what is set in setParameters
        
        Args:
            backgroundColor: The background color as a list of RGB values, each between 0 and 1. If omitted, will use the color from settings.
        '''    
        if self.settings is None:
            print("(pglExperiment:initScreen) No settings found to open screen.")
            return
        # get background color
        if backgroundColor == -1:
            backgroundColor = self.settings.backgroundColor
        
        try:
            # get screen parameters
            if len(self.settings.displays) < 1:
                pglMessages.warning(f"Settings {self.settings.name} is not associated with a display")
                return
            self.state.display = self.settings.displays[0]
        
            # close all other screens
            self.pgl.cleanUp()
            
            # set screen resolution if necessary
            self.state.originalScreenResolution = self.pgl.getResolution(self.state.display.currentDisplayNum)

            if self.state.display.uuid == "windowed":
                # open the screen
                self.pgl.open(0, screenWidth=self.state.display.windowSize[0], screenHeight=self.state.display.windowSize[1], backgroundColor=backgroundColor)                        
                self.pgl.setWindowFrameInDisplay(0, screenX=self.state.display.windowPosition[0], screenY=self.state.display.windowPosition[1], screenWidth=self.state.display.windowSize[0], screenHeight=self.state.display.windowSize[1])
            else:
                if self.state.display.currentDisplayNum == -1:
                    pglMessages.warning(f"Could not open display {self.state.display.name} because it is not connected")
                    return
                # compare to what is desired
                displayMode = None
                if self.state.display.displayModes:
                    displayMode = self.state.display.displayModes[0]
                    if displayMode == self.state.originalScreenResolution:
                        pglMessages.message("Match")
                    else:
                        self.pgl.setResolutionUsingDisplayModeSettings(self.state.display.currentDisplayNum, displayMode)      
                        self.state.screenResolution = self.pgl.getResolution()
                        pglMessages.message(f"Changing screen resolution to: {self.state.screenResolution[0]} x {self.state.screenResolution[1]} {self.state.screenResolution[2]}Hz {self.state.screenResolution[3]}bits " +
                                            f"from: {self.state.originalScreenResolution[0]} x {self.state.originalScreenResolution[1]} {self.state.originalScreenResolution[2]}Hz {self.state.originalScreenResolution[3]}bits")
                
                # open the screen
                self.pgl.open(whichScreen=self.state.display.currentDisplayNum, backgroundColor=backgroundColor)        
            
            # check whether it was opened
            if not self.pgl.isOpen():   
                pglMessages.warning("Failed to open screen.")
                return
            
            # set visual angle coordinates
            self.pgl.visualAngle(self.state.display.displayDistance, self.state.display.displaySize[0], self.state.display.displaySize[1])

            # flip left-right and/or up-down if specified in settings
            if self.state.display.flipLeftRight: self.pgl.flipLeftRight()
            if self.state.display.flipUpDown: self.pgl.flipUpDown()
            
            # set the gamma table
            self.setGamma(self.settings, self.state.display)
            
            # add keyboard device if not already loaded
            keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
            if not keyboardDevices:
                # nothing loaded, so create it
                keyboardMouse = pglKeyboardMouse(eatKeys=None)
                self.pgl.devicesAdd(keyboardMouse)
                # check if listener is running
                if not keyboardMouse.isRunning():
                    warningMessage = "Accessibility permission not granted for keyboard/mouse access.\n" + \
                        "On macOS, go to System Preferences -> Security & Privacy -> Privacy -> Accessibility\n" + \
                        "and add your terminal application (e.g. Terminal, iTerm, etc) to the list of apps allowed to control your computer.\n" + \
                        "If you are running VS Code and it already has permissions granted, try running directly from a terminal with:\n" + \
                        "    /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Code"
                    pglMessages.warning(warningMessage)
                    startTime = pglTimestamp.getSecs()     
                    displayMessageTime = 15  
                    while((pglTimestamp.getSecs()-startTime) < displayMessageTime):      
                        self.pgl.text("Accessibility permission error")
                        self.pgl.text("Need to grant for keyboard and mouse")
                        self.pgl.text("See instructions printed to console")
                        self.pgl.text(f"Screen will close in {int(displayMessageTime-(pglTimestamp.getSecs()-startTime))} s")
                        self.pgl.flush()
                    
                    self.endScreen()
                    return
            else:
                # if already loaded, just grab it
                keyboardMouse = keyboardDevices[0]
                # and if it is not running, start it
                if not keyboardMouse.isRunning():
                    keyboardMouse.start()
            
            # clear the mouse and keyboard queues of any pending events
            keyboardMouse.clear()

            # keep a pointer to keyboardMouse
            self.keyboardMouse = keyboardMouse
            
            # If response keys is a comma-separated list, split it into a list (this is so you can do "1,space,F1,2"
            if ',' in self.settings.responseKeys:
                self.responseKeysList = [k.strip() for k in self.settings.responseKeys.split(',')]
            else:
                # if no commas, then just make a list of characters
                self.responseKeysList = list(self.settings.responseKeys)
                
            # get keyCodes
            self.state.responseKeyCodesList = [keyboardMouse.charToKeyCode(k) for k in self.responseKeysList]
            self.state.startKeyCode = keyboardMouse.charToKeyCode(self.settings.startKey)
            self.state.endKeyCode = keyboardMouse.charToKeyCode(self.settings.endKey)
            self.state.volumeTriggerKeyCode = keyboardMouse.charToKeyCode(self.settings.volumeTriggerKey)
            
            # if eatKeys is set, then compose a list of all keys as keyCodes
            if self.settings.eatKeys:
                # Collect all individual keys
                eatKeyCodes = self.state.responseKeyCodesList.copy()  # Start with response keys list
        
                # Add single keys if they exist
                if self.settings.startKey:
                    eatKeyCodes.append(self.state.startKeyCode)
                if self.settings.endKey:
                    eatKeyCodes.append(self.state.endKeyCode)
                if self.settings.volumeTriggerKey:
                    eatKeyCodes.append(self.state.volumeTriggerKeyCode)
        
                # Register these as keys to be eaten
                keyboardMouse.setEatKeys(eatKeyCodes)
                
            # wait half a second for metal app to initialize
            self.pgl.waitSecs(0.5)
            
            # flush screen to get rid of any transients
            self.pgl.flush()
            self.pgl.flush()
            
            # initialize eye tracker
            self.initEyeTracker()    

            # display device status
            self.pgl.deviceStatus()

            # mark that we have opened the screen
            self.state.openScreen = True

        except Exception as e:
            pglMessages.warning(f"Could not open screen. Error {type(e).__name__}: {e}")    
            self.state.openScreen = False
            return

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
        
    def endScreen(self):
        '''
        Close the screen
        '''
        try:
            # stop the keyboard listener
            keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
            if keyboardDevices is not []:
                for keyboardDevice in keyboardDevices:
                    print("(pglExperiment:endScreen) Stopping keyboard/mouse device.")
                    print(keyboardDevice)
                    keyboardDevice.stop()

            if self.settings.closeScreenOnEnd:
                # clear screen
                self.pgl.clearScreen(self.settings.backgroundColor)
                self.pgl.flush()
                
                # reset gamma
                if self.state.originalGammaTable:
                    self.pgl.message("Restoring gamma table")
                    self.pgl.setGammaTable(self.state.display.currentDisplayNum, rgbGammaTable=self.state.originalGammaTable)
                    pglMessages.message("Restoring original gamma table")
                    
                # reset screen dimensions
                if self.state.screenResolution:
                    if self.state.screenResolution != self.state.originalScreenResolution:
                        self.pgl.setResolution(self.state.display.currentDisplayNum, screenResolution=self.state.originalScreenResolution)
                        pglMessages.message(f"Restoring resolution back to: {self.state.originalScreenResolution[0]} x {self.state.originalScreenResolution[1]} {self.state.originalScreenResolution[2]}Hz {self.state.originalScreenResolution[3]}bits")
                
                # close screen
                self.pgl.close()
                self.state.openScreen = False
        except Exception as e:
            pglMessages.warning(f"Could not close screen. Error {type(e).__name__}: {e}")    
            return
                

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
    
    def run(self):
        '''
        Run the experiment.
        '''
        # default to error, as normal operation will set this to False
        self.state.runFinishedWithError = True

        if self.state.openScreen == False:
            pglMessages.warning("Screen is not open. Call initScreen() before running the experiment.")
            return
        

        # start eye tracker recording if we have an eye tracker
        if self.eyeTracker is not None:
            self.eyeTracker.start()

        # calculate the phases that we will need to cover
        self.state.phaseNums = sorted(task.settings.phaseNum for task in self.tasks if task.settings.phaseNum is not None) or None
        
        # intialize variables
        self.state.experimentDone = False
        self.state.volumeNumber = 0

        # set manual pre-start (this will put up a screen and wait for start key
        # before waiting for volume trigger
        manualPreStart = True if self.settings.manualPreStart else False
        manualPreStartVolumes = 0
        ignoreInitialVolumes = self.settings.ignoreInitialVolumes
        
        # see if we need to run eye calibration
        if self.settings.eyetracker is not None:
            self.calibrateEyeTracker()
            
        # wait for key press to start experiment
        if self.settings.startKey is not [] or self.settings.startOnVolumeTrigger:
            self.state.experimentStarted = False
            while not self.state.experimentStarted:
                if manualPreStart:
                    self.pgl.text(f"Press {self.settings.startKey} key to make experiment start",y=0)
                elif self.settings.startOnVolumeTrigger:
                    self.pgl.text("Waiting for volume trigger to start experiment",y=0)
                else:
                    self.pgl.text(f"Press {self.settings.startKey} key to start experiment",y=0)
                # flush to display text
                self.pgl.flush()
                # poll for events
                events = self.pgl.poll()
                self.data.events.extend(events)

                # see if we have a match to startKey
                if [e for e in events if e.type == "keyboard" and e.eventType == "keydown" and e.keyCode == self.state.startKeyCode]:
                    # end manual pre-start
                    if manualPreStart:
                        manualPreStart = False
                        print(f"(pglExperiment:run) Manual pre-start ended after {manualPreStartVolumes} volumes.")
                    # or start experiment
                    else:
                        self.state.experimentStarted = True
                
                # if waiting to startOnVolumeTrigger, check for that key                
                if self.settings.startOnVolumeTrigger:
                    for e in events:
                        if e.type == "keyboard" and e.eventType == "keydown" and e.keyCode == self.state.volumeTriggerKeyCode:
                            if manualPreStart:
                                # keep count of volumes
                                manualPreStartVolumes += 1
                            elif ignoreInitialVolumes>0:
                                # ignore initial volumes
                                ignoreInitialVolumes -= 1
                            else:
                                print(f"(pglExperiment:run) Ignored {self.settings.ignoreInitialVolumes} initial volumes.")
                                self.state.experimentStarted = True
                                self.state.volumeNumber += 1
                                # and add a volume event
                                self.data.events.append(pglEventVolumeTrigger(timestamp=e.timestamp))
                
                # Check for end key to allow aborting before starting    
                if [e for e in events if e.type == "keyboard" and e.keyCode == self.state.endKeyCode]:
                    self.state.experimentStarted = True
                    self.state.experimentDone = True
        
        # start the experiment
        self.startPhase(phaseNum=0)
        print(f"(pglExperiment:run) Experiment started.")
        self.data.startTime = self.pgl.getSecs()

        while not self.state.experimentDone:
            
            # poll for events
            events = self.pgl.poll()
            self.data.events.extend(events)

            # see if we have a match to endKey
            if [e for e in events if e.type == "keyboard" and e.keyCode == self.state.endKeyCode]:
                self.state.experimentDone = True
                # end all running tasks
                for task in self.currentTasks: task.end()
                continue

            # Check for volume trigger key
            for i, e in enumerate(events):
                if e.type == "keyboard" and e.keyCode == self.state.volumeTriggerKeyCode and e.eventType == "keydown":
                    # remove it from the events list 
                    events.pop(i)
                    # and update volumeNumber
                    self.state.volumeNumber += 1
                    # and add a volume trigger event
                    self.data.events.append(pglEventVolumeTrigger(timestamp=e.timestamp))
                    break

            # grab any events that match the keyList and return their index within that list and timestamp
            subjectResponses = [
                (self.state.responseKeyCodesList.index(e.keyCode), e.timestamp)
                for e in events
                if (
                    e.type == "keyboard"
                    and e.eventType == "keydown"
                    and e.keyCode in self.state.responseKeyCodesList
                )
            ]

            # update tasks in current phase
            phaseDone = False
            updateTime = self.pgl.getSecs()
            for task in self.currentTasks:
                # update task
                task.update(updateTime=updateTime, subjectResponses=subjectResponses, phaseNum=self.state.phaseNum, tasks=self.currentTasks, events=events)
                # check if task is done
                if task.done(): phaseDone = True
            
            # update the screen
            if self.flush: self.pgl.flush()
            #else: print("No flush")

            # go to next phase or end experiment
            if phaseDone:
                # end all tasks in current phase
                for task in self.currentTasks: task.end()
                # check if we have ended all phases
                if self.state.phaseNum >= len(self.state.phaseNums)-1:
                    self.state.experimentDone = True
                else:
                    # update phase
                    self.state.currentPhaseIndex += 1
                    self.startPhase(phaseNum=self.state.phaseNums[self.state.currentPhaseIndex])

        # clear screen
        self.pgl.clearScreen(self.settings.backgroundColor)
        self.pgl.flush()

        # stop eye tracker recording if we have an eye tracker
        if self.eyeTracker is not None:
            self.eyeTracker.stop()

        # mark end time
        self.data.endTime = self.pgl.getSecs()
        print("(pglExperiment:run) Experiment done.")
        
        # save data
        self.save()
        
        # close screen
        self.endScreen()

        # if we got here then we finished the run without error
        self.state.runFinishedWithError = False
    
    def initEyeTracker(self):
        '''Initialize eye tracker if we have an eye tracker.'''
        if self.settings.eyetracker[0] == "Eyelink":
            # set edf filename to current date (note it has to be 8.3 characters
            # since SR Research has progressed from the days of DOS)
            self.settings.edfFilename = f"{datetime.now().strftime('%Y%m%d')}"

            # init the eyeink
            print(f"(pglExperiment) Initialize Eyelink with filename: {self.settings.edfFilename}")
            self.eyeTracker = pglEyelink(pgl=self.pgl, edfFilename=self.settings.edfFilename)  

            # check if running
            if self.eyeTracker.eyelink is None:
                self.eyeTracker = None      

            # FIX: these should come from some settings
            if self.eyeTracker is not None:
                self.eyeTracker.setCustomCalibrationPoints(margin=0.7, numPoints=9)

        elif self.settings.eyetracker[0] == "None":
            self.eyeTracker = None
        else:
            print("(pglExperiment) ❌ Unknown eye tracker type {self.settings.eyetracker[0]}")
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
            pglMessages.message(displayText)

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
                    print("(pglExperiment:calibrateEyeTracker) Skipping eye tracker calibration.")

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
        
        # get the current tasks based on the current phase number
        self.currentTasks = [task for task in self.tasks if task.settings.phaseNum is None or task.settings.phaseNum == self.state.phaseNum]

        # set start time
        startTime = self.pgl.getSecs()
        for task in self.currentTasks:
            task.start(startTime)

        print(f"(pglExperiment:startPhase) Starting phase: {self.state.phaseNum}/{len(self.state.phaseNums)}")
        
    
    def save(self):
        '''
        Save the experiment settings, state and data.         
        '''
        # Create the directory to save data into (dataDir/experimentSaveName/subjectID/YYYYMMDD_HHMMSS)
        try:
            dataPath = Path(self.settings.dataPath).expanduser() / self.experimentSettings.experimentSaveName / self.experimentSettings.subjectID / self.experimentSettings.sessionName / self.experimentSettings.runName
            dataPath.mkdir(parents=True, exist_ok=True)    
        except Exception as e:
            print(f"(pglExperiment:save) ❌ Could not create data directory {dataPath}: {e}")
            return
        
        # give user feedback where things are being saved
        print(f"(pglExperiment:save) Saving experiment data to: {dataPath}")
        
        # save eye tracker data if we have an eye tracker
        if self.eyeTracker is not None:
            eyeTrackerFilename = dataPath / f"{self.settings.eyetracker[0].lower()}"
            self.eyeTracker.save(eyeTrackerFilename)

        # save experiment settings
        self.experimentSettings.save(dataPath / "experimentSettings.json")
        
        # save pgl state
        self.pgl.save(dataPath / "pgl.json")
        
        # save each task
        for task in self.tasks: task.save(dataPath)   
        
        # and call parent to save rest
        super().save(dataPath=dataPath)
    
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
                    print(f"Warning: Unexpected type in parameters: {type(item)}")
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

    def display(self, taskName="task", responseMapping=None, ax=None):
        '''
        Display the experiment data.
        '''
        # use responseMapping for displaying subject responses
        if responseMapping is None:
            responseMapping = self.responseMapping

        # get trial timestamps
        trialTimestamps = np.array([e.timestamp for e in self.events if isinstance(e, pglEventTrial)])
        if len(trialTimestamps) <= 2:
            print("(pglTaskData:display) Insufficient trial events found to display.")
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
                        
        timeline.setTitle(f"{taskName}: {nTrials} trials")
        
        # display legend
        legend = [{'label': 'Segment', 'color': 'blue'}]
        # add the response values
        if gotResponse:
            for respType, (label, color) in responseMapping.items():
                # get statistics for this response type
                count = responseCounts.get(respType, 0)
                percent = (count / sum(responseCounts.values()) * 100) if nTrials > 0 else 0
                legend.append({'label': f'{label} (n={count}: {percent:.1f}%)', 'color': color})
        timeline.addLegend(legend)
        if not ax: timeline.show()

##############################################
# Task base 
##############################################
class pglTaskBase(pglTraitSettings):
    settings = Instance(pglTaskSettings, allow_none=True, help="Settings for the task")
    state = Instance(pglTaskState, allow_none=True, help="State of task")
    data = Instance(pglTaskData, allow_none=True, help="Data of task")
    parameters = List(Instance(pglParameter), default_value=[], help="parameters of task")
  
    def save(self, dataPath):
        '''
        Save the task settings, state and data.
        '''
        try:
            dataPath = Path(dataPath) / self.getTaskDirectoryName()
            dataPath.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"(pglTask:save) ❌ Could not create task data directory {dataPath}: {e}")
            return
        
        # save settings, state and data
        self.settings.save(dataPath / "settings.json")
        self.state.save(dataPath / "state.json")
        self.data.save(dataPath / "data.json")
        
        # save parameters
        try:
            parameterPath = dataPath / "parameters"
            parameterPath.mkdir(parents=True, exist_ok=True)
            for parameter in self.parameters:
                parameter.save(parameterPath)
        except Exception as e:
            pglMessages.warning(f"Could not save task parameters to {dataPath}: {e}")
        pglMessages.message(f"Saved task {self.settings.taskName} to {dataPath}")

    def getTaskDirectoryName(self):
        """Return the directory name used to save this task."""
        taskDirectoryName = self.settings.taskSaveName
        phaseNum = self.settings.phaseNum

        if phaseNum is not None and phaseNum != 0:
            taskDirectoryName += f"Phase{phaseNum:02d}"

        return taskDirectoryName
    @classmethod
    def load(cls, dataPath, filesystem=None):
        '''
        Load the task data.
        '''
        print(f"(pglTask:load) Loading task data from: {dataPath}")

        # validate filesystem
        filesystem, taskPath, _ = pglBase.validateFilesystem(filesystem, dataPath)

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
        self.data.display(taskName=self.settings.taskName, ax=ax)
    
    def print(self):
        '''
        Print a summary of the task data
        '''
        from pgl import pglTimestamp
        timestamp = pglTimestamp()
        
        # print task name and number of trials
        print(f"Task: {self.settings.taskName} | Trials: {self.state.currentTrial+1}")
        print(f"Duration={timestamp.formatDuration(self.data.endTime - self.data.startTime)} | startTime={self.data.startTime} | endTime={self.data.endTime}")

        # print seglen and waitFor
        print(f"seglen={self.settings.seglen}")
        print(f"waitUntilVolumeTrigger={self.settings.waitUntilVolumeTrigger}")
                
        # print fixedParameters
        print('\n'.join(f"{key}={value}" for key, value in self.settings.fixedParameters.items()))
        print('-' * 40)
        

        # print parameters
        for p in self.parameters:
            print(f"{p.settings.name}")
        
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
                print(f"Trial {iTrial+1} at {trialStart:.2f}s: " + ', '.join(f"{key}={value}" for key, value in params.items()))
            else:
                print(f"Trial {iTrial+1} at {trialStart:.2f}s (vol={trialVolume}): " + ', '.join(f"{key}={value}" for key, value in params.items()))
                          
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


    def start(self, startTime):
        '''
        Start the task.
        '''
        # if task is already started, then do nothing
        if self.data.startTime is not None:
            return

        # set task start time
        self.data.startTime = startTime
        
        # start trial
        self.state.currentTrial = -1
        self.startTrial(startTime)

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
            if self.settings.saveEyeTracker:
                self.e.saveEyeTrackerEvent(eventType="segment", taskID=self.settings.taskID,
                    trialNum=self.state.currentTrial, segmentNum=self.state.currentSegment,
                    timestamp=updateTime, phaseNum=self.settings.phaseNum)

            self.state.currentSegment += 1
            self.state.segmentStartTime = updateTime
            self.data.events.append(pglEventSegment(self.state.currentSegment, updateTime))
            self.waitUntilVolumeTrigger = False

            # a real segment actually started — call the overridable hook
            self.startSegment(updateTime)
    def startTrial(self, startTime):
        '''
        Start a trial.
        '''
        # update values
        self.state.currentTrial += 1
        self.data.events.append(pglEventTrial(self.state.currentTrial, startTime))
        self.state.trialStartTime = startTime

        # save eye tracker event for synchronization        
        if self.settings.saveEyeTracker:
            self.e.saveEyeTrackerEvent(eventType="trial", taskID=self.settings.taskID, trialNum=self.state.currentTrial, segmentNum=self.state.currentSegment, timestamp=startTime, phaseNum=self.settings.phaseNum)

        # get current parameters
        self.data.params.append({})
        self.currentParams = self.data.params[-1]
        for parameter in self.parameters: 
            self.data.params[-1].update(parameter.get())

        # initialize the trialVariables for this trial
        # trialVariables are set by the task to store any computed, incidental, discovered trial-by-trial variables
        self.data.trialVariables.append({})
        
        # start segment (startSegment will update currentSegment to 0)
        self.state.currentSegment = -1
        self._startSegment(startTime)
        
        # get a random length for each segment. If segmin==segmax, then fixed length
        self._thisTrialSeglen = [
            # if either segmin or segmax is infinite, set to infinite
            float('inf') if math.isinf(min_val) or math.isinf(max_val) 
            # otherwise choose a random length between min and max
            else random.uniform(min_val, max_val)
            for min_val, max_val in zip(self.settings.segmin, self.settings.segmax)
        ]

        # print trial
        print(f"({self.settings.taskName}) Trial {self.state.currentTrial+1}: ", end='')
        
        # and variable settings
        for name,value in self.data.params[-1].items():
            print(f'{name}={value}', end=' ')
        print(f"")

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

        # record end time
        print(f"Ending task {self.settings.taskName}")
        endTime = self.pgl.getSecs()
        self.data.endTime = endTime
        
        # put in time stamps for end of last segment and trial
        self.data.events.append(pglEventSegment(self.state.currentSegment, endTime, eventType=pglEventSegment.boundaryType.END))
        self.data.events.append(pglEventTrial(self.state.currentTrial, endTime, eventType=pglEventTrial.boundaryType.END))

    def jumpSegment(self):
        '''
        Jump to the next segment.
        '''
        # set current segment length to 0 to force jump
        self._thisTrialSeglen[self.state.currentSegment] = 0

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
            print("(pglExperimentData) No events to display.")
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
        firstKeydownEvent = next((event for event in self.events if event.type == "keyboard" and event.eventType == "keydown"), None)
        if firstKeydownEvent is not None:
            if firstKeydownEvent.timestamp < self.startTime:
                startTime = firstKeydownEvent.timestamp - self.startTime
        else:
            startTime = 0
            
        # track number of volumes
        nVols = 0
        nKeys = 0
        
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
                
        timeline.setTitle("Experiment Events")
        timeline.addLegend([{'label': f'Keypress (n={nKeys})', 'color': 'green'},{'label': f'Volumes (n={nVols})', 'color': 'blue'}])
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
class pglExperimentState(pglTraitSettings):
    phaseNum = Int(default_value=None, allow_none=True, help="Current experiment phase number.")
    phaseNums = List(trait=Int(), default_value=None, allow_none=True, help="List of experiment phase numbers.")
    currentPhaseIndex = Int(default_value=0, help="Index into phaseNums for the current phase.")
    openScreen = Bool(default_value=False, help="Whether the experiment screen is currently open.")
    runFinishedWithError = Bool(default_value=False, help="Whether the experiment ended with an error.")
    volumeNumber = Int(default_value=0, help="Current scanner-volume number.")
    experimentStarted = Bool(default_value=False, help="Whether the experiment has started.")
    experimentDone = Bool(default_value=False, help="Whether the experiment has completed.")
    responseKeyCodesList = List(Int(), help="List of response key codes received during the experiment.")
    startKeyCode = Int(default_value=0, help="Key code used to begin the experiment.")
    endKeyCode = Int(default_value=0, help="Key code used to end the experiment.")
    volumeTriggerKeyCode = Int(default_value=0, help="Key code received for each scanner-volume trigger.")
    originalGammaTable = Tuple(List, List, List, default_value=None, allow_none=True, help="Original display gamma table: (red, green, blue).")
    gammaTable = Tuple(List, List, List, default_value=None, allow_none=True, help="Current display gamma table: (red, green, blue).")
    display = Instance(pglDisplaySettings, default_value=None, allow_none=True, help="Current display settings.")
    originalScreenResolution = Tuple(Int(),Int(),Int(),Int(), default_value=None, allow_none=True, help="Original screen resolution: (left, top, width, height).")
    screenResolution = Tuple(Int(),Int(),Int(),Int(), default_value=None, allow_none=True, help="Current screen resolution: (left, top, width, height).")

    # make sure any variabels added by experimenter code gets saved
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
        print(f"Timeline saved to {filename}")
    
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
        print(f"(pglEventTrial) Trial {self.eventType} at: {self.timestamp}")
        
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
        print(f"(pglEventSegment) Segment {self.segmentNum} {self.eventType} at: {self.timestamp}")
        

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

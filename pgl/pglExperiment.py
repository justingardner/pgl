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
import json
import numpy as np
import itertools
import random
import math
from dataclasses import dataclass, field
from .pglKeyboardMouse import pglKeyboardMouse
from pathlib import Path
from .pglSettings import pglSettingsManager, pglSettings, pglSettingsEditable
from IPython.display import display, HTML
import ipywidgets as widgets
from .pglBase import pglDisplayMessage
from traitlets import Float, TraitError, TraitError, observe, Instance, Int, Unicode, Dict, validate, Bool
from .pglParameter import pglParameter, pglParameterBlock
from .pglEvent import pglEvent
from .pglSerialize import pglSerialize
from typing import List as ListType, Optional
from traitlets import List
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from enum import Enum
from . import pglTimestamp
from .pglEyeTracker import pglEyeTracker
from .pglEyelink import pglEyelink, pglEyelinkData

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
class pglExperimentBase(pglSettingsManager):
    '''
    Base class for pglExperiment which runs experiments
    and pglExperimentAnalysis which is used for loading and
    analyzing experimental data. This class handles loading
    and saving settings, state and data.
    '''
    def __init__(self):
        # init super
        super().__init__()
        
        # initialize variables
        self.settings = None
        self.state = None
        self.data = None
        self.experimentSettings = None
        self.pgl = None
        self.eyeTracker = None
        self.tasks = []
                
    def loadSettings(self, settingsName=None, settings=None):
        
        if (settings is None) and (settingsName is None):
            # if no settings provided, then just use default settings
            self.settings = pglSettings()
            print("(pglExperiment) No settings provided, using default settings.")
            return

        # get settings
        self.settings = settings
        if self.settings is None:
            self.settings = self.getSettings(settingsName)
        
        # if there was some error, then display it
        if self.settings is None:
            # display error in HTML
            pglDisplayMessage("<b>(pglExperiment:initScreen)</b> ❌ Could not find settings, using default settings.")
            # create default settings
            self.settings = pglSettings()

    def load(self, experimentName="", subjectID="", date = None):
        '''
        Load the experiment settings, state and data.         
        '''
        # default values
        self.pgl = None
        self.task = []

        self.pglTimestamp = pglTimestamp()
        dataDir = Path(self.settings.dataPath).expanduser() / experimentName / subjectID 
        # check that data path exists
        if not dataDir.exists() or not dataDir.is_dir():
            print(f"(pglExperiment:load) ❌ Could not find data directory: {dataDir}")
            return
        
        # Get all directories in the dataPath
        dirList = [d for d in dataDir.iterdir() if d.is_dir()]

        # check date setting, so that we can filter out experiments by date        
        if date is not None:
            # Convert to YYYYMMDD string
            if isinstance(date, str):
                dateStr = date
            elif isinstance(date, datetime):
                dateStr = date.strftime("%Y%m%d")
            elif isinstance(date, Date):
                dateStr = date.strftime("%Y%m%d")
            else:
                raise TypeError("date must be a string, datetime.date, datetime.datetime, or None")

            # Keep only directories whose names start with the date
            dirList = [d for d in dirList if d.name.startswith(dateStr)]

        # sort directory list
        dirList = sorted(dirList, key=lambda d: d.name)
        
        # print the experiment name and subjetID
        print(f"(pglExperiment:load) Data directory: {dataDir}")
        print(f"Experiment name: {experimentName} | Subject ID: {subjectID}")
        
        # Print all the directories
        for i, d in enumerate(dirList, start=1):
            try:
                # for typical data files, parse into a datatime
                dt = datetime.strptime(d.name, "%Y%m%d_%H%M%S")
                # and print a more user-friendly name
                dataPrintname = dt.strftime("%A %B %-d, %Y %-I:%M%p")
                # try to read the data.json file
                dataFilename = dataDir / d.name / "data.json"
                if dataFilename.exists():
                    try:
                        # load the data so that we can print information about the experiment
                        experimentData = pglSerialize.load(dataDir / d.name / "data.json")
                        experimentSettings = pglSerialize.load(dataDir / d.name / "experimentSettings.json")
                        # print number of volumes
                        numVols = experimentData.getNumEvents(type="volumeTrigger")
                        if numVols==0:
                            numVols = experimentData.getNumEvents(type="keyboard", eventType="keydown", keyChar="5")
                        dataPrintname += f" | nVols: {numVols}"
                        # print duration
                        dataPrintname += f" | {self.pglTimestamp.formatDuration(self.experimentDuration(experimentData))}"
                        # print experiment name
                        dataPrintname += f" | {experimentSettings.experimentName}"
                    except:
                        pass
                # Add the filename
                dataPrintname = f"{dataPrintname} ({d.name})"
            except:
                # not a typical name, just display
                dataPrintname = d.name
            print(f"{i}. {dataPrintname}", flush=True)

        # Ask the user to choose
        print("\nSelect a directory number: ", flush=True)
        choice = int(input())
        if choice < 1 or choice > len(dirList):
            print(f"(pglExperiment:load) ❌ Invalid choice: {choice}")
            return

        # Get the selected directory
        selectedDir = dataDir / dirList[choice - 1]

        # load the experiment data, settings, and state
        print(f"(pglExperiment:load) Loading experimentdata from: {selectedDir}")
        self.data = pglSerialize.load(selectedDir / "data.json")
        self.settings = pglSerialize.load(selectedDir / "settings.json")
        self.experimentSettings = pglSerialize.load(selectedDir / "experimentSettings.json")
        self.state = pglSerialize.load(selectedDir / "state.json")
        
        # load pgl state
        self.pglState = pglSerialize.load(selectedDir / "pgl.json")
        
        # load all the tasks
        dirList = [d for d in selectedDir.iterdir() if d.is_dir()]
        dirList = sorted(dirList, key=lambda d: d.name)
        for i, d in enumerate(dirList, start=1):
            # get the task directory
            taskDir = selectedDir / d.name
            # load the task data
            task = pglTask()
            task.load(taskDir)
            
            # add the task to the experiment
            self.addTask(task)
        
        # load the eye tracker data
        if self.settings.eyetracker[0] == "Eyelink":
            eyeTrackerFilename = selectedDir / f"{self.settings.eyetracker[0].lower()}.asc"
            self.eyeTrackerData = pglEyelinkData(str(eyeTrackerFilename))
        elif self.settings.eyetracker[0] == "None":
            self.eyeTracker = None
        else:
            print("(pglExperiment) ❌ Unknown eye tracker type {self.settings.eyetracker[0]}")
            self.eyeTracker = None
   
    def addTask(self, task):
        '''
        Add a task to the experiment.
        '''
        # give it a reference to pgl and experiment
        task.pgl = self.pgl
        task.e = self
        task.taskID = len(self.tasks)

        # set whether to save eye tracker info
        if self.eyeTracker is not None:
            task.settings.saveEyeTracker = True

        # add the task
        self.tasks.append(task)
        
        # save in experimentSettings
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
        
        displayInfo = f"Display: {self.settings.displayName[0] if self.settings.displayName and len(self.settings.displayName) > 0 else 'Unknown'} "
        displayInfo += f"{self.pglState.screenWidthPixels}x{self.pglState.screenHeightPixels} @ {self.pglState.frameRate}Hz "
        displayInfo += f"{self.pglState.screenWidthDegrees:.2f}x{self.pglState.screenHeightDegrees:.2f} deg "
        displayInfo += f"{self.settings.displayWidth:.2f}x{self.settings.displayHeight:.2f} cm at {self.settings.displayDistance:.2f} cm "
        print(displayInfo)
        
        numVols = self.data.getNumEvents(type="volumeTrigger")
        print(f"Number of volume triggers: {numVols}")
        if numVols > 1:
            triggerStats = self.data.getTriggerStats()
            print(f"Median time between triggers: {triggerStats.median:.3f}s")
            print(f"Mean ± std time between triggers: {triggerStats.mean:.3f} ± {triggerStats.std:.6f}s")
        
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
    def __init__(self, pgl=None, settingsName=None, settings=None, subjectID="s0000", experimentName=""):
        '''
        Initialize the pglExperiment class.
        
        Args:
            pgl (pgl): An instance of the pgl class.
            settingsName (str): The name of the settings to use. If not set (and settings not set), will use default settings
            settings (pglSettings): An instance of the pglSettings class. If set, will supersede settingsName.
            subjectID (str): The identifier for the subject participating in the experiment.
        '''
        # init super
        super().__init__()

        # load settings
        self.loadSettings(settingsName=settingsName, settings=settings)

        if pgl is None:
            # If pgl is none, then this is a load
            if experimentName == "":
                print(f"(pglExperiment) No exerimentName provided for loading experiment data")
                return
            # load the experimentName
            self.load(experimentName=experimentName, subjectID=subjectID)
            return
        else:
            # save pgl
            self.pgl = pgl

        # initialize experiment state and data
        self.state = pglExperimentState()
        self.data = pglExperimentData()
        
        # get experiment settings
        self.experimentSettings = pglExperimentSettings()
        if experimentName != "":
            self.experimentSettings.experimentName = experimentName
        self.experimentSettings.subjectID = subjectID
        
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
            
        # open screen
        if self.settings.displayNumber == 0:
            self.pgl.open(whichScreen=0, screenWidth=self.settings.windowWidth, screenHeight=self.settings.windowHeight, backgroundColor=backgroundColor)        
        else:
            self.pgl.open(whichScreen=self.settings.displayNumber-1, backgroundColor=backgroundColor)        
        if not self.pgl.isOpen():   
            pglDisplayMessage("<b>(pglExperiment:initScreen)</b> ❌ Failed to open screen.", useHTML=True, duration=5)
            return
        
        # set visual angle coordinates
        self.pgl.visualAngle(self.settings.displayDistance,self.settings.displayWidth,self.settings.displayHeight)
        
        # flip left-right and/or up-down if specified in settings
        if self.settings.flipLeftRight: self.pgl.flipLeftRight()
        if self.settings.flipUpDown: self.pgl.flipUpDown()
        
        # set the gamma table
        self.setGammaTable()
        
        # add keyboard device if not already loaded
        keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
        if not keyboardDevices:
            # nothing loaded, so create it
            keyboardMouse = pglKeyboardMouse(eatKeys=None)
            self.pgl.devicesAdd(keyboardMouse)
            # check if listener is running
            if not keyboardMouse.isRunning():
                pglDisplayMessage("<b>(pglExperiment:initScreen)</b> ❌ Accessibility permission not granted for keyboard/mouse access.", useHTML=True)
                pglDisplayMessage("On macOS, go to System Preferences -> Security & Privacy -> Privacy -> Accessibility, and add your terminal application (e.g. Terminal, iTerm, etc) to the list of apps allowed to control your computer.", useHTML=True)
                pglDisplayMessage("If you are running VS Code and it already has permissions granted, try running directly from a terminal with:", useHTML=True)
                pglDisplayMessage("              /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Electron", useHTML=True)
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
        
        # mark that we have opened the screen
        self.state.openScreen = True

        # initialize eye tracker
        self.initEyeTracker()        

    def setGammaTable(self):
        '''
        set the gamma table based on settings
        '''
        # no gamma correction asked for
        if self.settings.calibrateForGamma == 0.0:
            return
        
        # No calibration
        if self.settings.calibration[0] == "None":
            return
        
    def endScreen(self):
        '''
        Close the screen
        '''
        # stop the keyboard listener
        keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
        if keyboardDevices is not []:
            for keyboardDevice in keyboardDevices:
                print("(pglExperiment:endScreen) Stopping keyboard/mouse device.")
                print(keyboardDevice)
                keyboardDevice.stop()

        if self.settings.closeScreenOnEnd:
            # close screen
            self.pgl.close()
            self.state.openScreen = False

    def eatAllKeys(self, eat=False):
        '''
        Args: 
            eat: bool (whether to eat all keys or not)
        '''
        return self.pgl.eatAllKeys(eat)

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
        if self.state.openScreen == False:
            pglDisplayMessage("(pglExperiment:run) ❌ Screen is not open. Call initScreen() before running the experiment.",useHTML=True, duration=5)
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

            # grab any events that match the keyList and return their index within that list
            subjectResponses = [keyIndex for e in events if e.type == "keyboard" and e.eventType == "keydown" and e.keyCode in self.state.responseKeyCodesList for keyIndex in [self.state.responseKeyCodesList.index(e.keyCode)]]
                
            # update tasks in current phase
            phaseDone = False
            updateTime = self.pgl.getSecs()
            for task in self.currentTasks:
                # update task
                task.update(updateTime=updateTime, subjectResponses=subjectResponses, phaseNum=self.state.phaseNum, tasks=self.currentTasks, events=events)
                # check if task is done
                if task.done(): phaseDone = True
            
            # update the screen
            self.pgl.flush()

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
    
    def initEyeTracker(self):
        '''Initialize eye tracker if we have an eye tracker.'''
        if self.settings.eyetracker[0] == "Eyelink":
            # set edf filename to current date (note it has to be 8.3 characters
            # since SR Research has progressed from the days of DOS)
            self.settings.edfFilename = f"P{datetime.now().strftime('%Y%m%d')}"

            # init the eyeink
            print(f"(pglExperiment) Initialize Eyelink with filename: {self.settings.edfFilename}")
            self.eyeTracker = pglEyelink(pgl=self.pgl, edfFilename=self.settings.edfFilename)        

            # FIX: these should come from some settings
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
            self.settings.skipCalibrationKey = 'enter'
            self.settings.skipCalibrationKeyCode = self.keyboardMouse.charToKeyCode(self.settings.skipCalibrationKey)
            # instructions
            displayText = f"Press {self.settings.calibrateKey} to calibrate eye tracker. {self.settings.skipCalibrationKey} to skip."
            print("(pglExperiment:calibrateEyeTracker) " + displayText)
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
                
            # if we should run calibration, then do it
            if self.state.runCalibration:
                self.eyeTracker.calibrate()

    def saveEyeTrackerEvent(self, eventType="segment", taskID=None, trialNum=None, segmentNum=None, timestamp=None):
        '''Save an eye tracker event for synchronization. This is called by tasks during updates if settings.saveEyeTracker is True.'''
        self.eyeTracker.sendMessage(f"pgl: {eventType} taskID={taskID} trialNum={trialNum} segmentNum={segmentNum} timestamp={timestamp}")

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
            dataDir = Path(self.settings.dataPath).expanduser() / self.experimentSettings.experimentSaveName / self.experimentSettings.subjectID / datetime.now().strftime("%Y%m%d_%H%M%S")
            dataDir.mkdir(parents=True, exist_ok=True)    
        except Exception as e:
            print(f"(pglExperiment:save) ❌ Could not create data directory {dataDir}: {e}")
            return
        
        # give user feedback where things are being saved
        print(f"(pglExperiment:save) Saving experiment data to: {dataDir}")
        
        # save eye tracker data if we have an eye tracker
        if self.eyeTracker is not None:
            eyeTrackerFilename = dataDir / f"{self.settings.eyetracker[0].lower()}"
            self.eyeTracker.save(eyeTrackerFilename)

        # save pgl state
        self.pgl.save(dataDir / "pgl.json")
        
        # save settings
        self.settings.save(dataDir / "settings.json")
        self.experimentSettings.save(dataDir / "experimentSettings.json")

        # save state
        self.state.save(dataDir / "state.json")
        
        # save data
        self.data.save(dataDir / "data.json")

        # save each task
        for task in self.tasks: task.save(dataDir)   

##############################################s
# experiment analysis class
##############################################
class pglExperimentAnalysis(pglExperimentBase):
    '''
    Experiment analysis class loads, data, settings and state
    of experiment, provides functions for displaying the data
    and members for extracting the data from the experiment
    in various ways
    '''
    def __init__(self, experimentName, subjectID="s0000", date=None, settingsName=None, settings=None):
        '''
        Initialize the pglExperimentAnalysis class.
        
        Args:
            subjectID (str): The identifier for the subject participating in the experiment.
            ExperimentName (str): The name of the experiment to load.
        '''
        # init super
        super().__init__()

        # load settings - this is primarily to get the data dir
        self.loadSettings(settingsName=settingsName, settings=settings)

        # load the experimentName
        self.load(experimentName=experimentName, subjectID=subjectID, date=date)
    
    def getTrialsByParameter(self, parameterName: str, taskName: str = None):
        '''
        Extracts trial data grouped by parameterName
        
        Args:
            parameterName (str): Name of parameter to group data by
        
        Returns:
            dictionary with fields
                parameterName (str): Name of parameter that the trials are sorted by
                nParameterValues (int): Number of different parameter values
                parameterValues (list): List of all parameter values
                trialNum: 
                volumeNum:
                tiralTime:
        '''
        # figure out what task we are working on
        if taskName is None:
            task = self.tasks[0]
        else:
            # search for the taskName (case insensitive)
            task = next((t for t in self.tasks if t.settings.taskName.lower() == taskName.lower()), None)
            # if not found, check if they meant the taskSaveName
            if task is None:
                task = next((t for t in self.tasks if t.settings.taskSaveName.lower() == taskName.lower()), None)
        
        if task is None:
            print(f"(pglExperimentAnalysis:getTrialsByParameter) ❌ Could not find {taskName} in experiemnt.\nValid tasks are: {' '.join(t.settings.taskName for t in self.tasks)}")
            return None
                
        # gather all the different parameter names
        parameters = task.parameters
        # get all the parameters recursively
        # so that we get all parameters in blocks 
        def collectParameters(parameterList):
            parameters = []
            for p in parameterList:
                if isinstance(p, pglParameterBlock):
                    parameters.extend(collectParameters(p.settings.parameters))
                else:
                    parameters.append(p)
            return parameters
        parameters = collectParameters(parameters)
        
        # get the matching parameter
        parameter = next((p for p in parameters if p.settings.name == parameterName), None)
        if parameter is None:
            print(f"(pglExperimentAnalysis:getTrialsByParameter) ❌ Could not find '{parameterName}' in parameters {[p.settings.name for p in parameters]}")
            return
        
        # initialize the list of lists for volumes by conditions        
        validValues = parameter.settings.validValues
        volumes = [[] for _ in range(len(validValues))]
        startTimes = [[] for _ in range(len(validValues))]
        trialNums = [[] for _ in range(len(validValues))]
        nTrials = [0 for _ in range(len(validValues))]
        nTrialsTotal = 0
        
        # loop over trials, collecting the params dictionary for each trial
        for iTrial, params in enumerate(task.data.params):
            # find matching trial event
            trialEvent = next((event for event in task.data.events if isinstance(event, pglEventTrial) and event.trialNum == iTrial), None)
            
            # get the trials tart time and volume
            trialStart = trialEvent.timestamp - task.data.startTime if trialEvent else "No trial event found"
            trialVolume = self.getNearestVolumeTrigger(trialEvent)

            # if we found a volume trigger
            if trialVolume is not None:
                # get the value that was set for this trial
                trialValue = params.get(parameter.settings.name,None)
                # if it matches the valid values
                if trialValue in validValues:
                    # get the index
                    conditionIndex = validValues.index(trialValue)
                    
                    # and populate arrays with data
                    volumes[conditionIndex].append(trialVolume)
                    startTimes[conditionIndex].append(trialStart)
                    trialNums[conditionIndex].append(iTrial+1)
                    nTrials[conditionIndex] += 1
                    nTrialsTotal += 1
        
        # pack everything up
        return {
            'parameterName': parameter.settings.name,
            'parameterValues': validValues,
            'parameter': parameter,
            'nTrialsTotal': nTrialsTotal,
            'volumes': volumes,
            'startTimes': startTimes,
            'trialNums': trialNums,
            'nTrials': nTrials
        }
                            
##############################################
# Task class
##############################################
class pglTask:
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
        Start a segment.
        '''
        # if the segment len is set to 0, it is a jump segment command
        # so reset the segment length to how long actually elapsed
        # so there is a record of how long we were in that segment
        if self.state.currentSegment >=0 and self._thisTrialSeglen[self.state.currentSegment] == 0:
            self._thisTrialSeglen[self.state.currentSegment] = updateTime - self.state.segmentStartTime

        # check for end of trial
        if (self.state.currentSegment+1) >= self.settings.nSegments: 
            # end the trial
            self.endTrial(updateTime)
            # if we are done with all trials
            if self.done(self.state.currentTrial+1):
                # just update currentTrial, but do not start a new trial
                self.state.currentTrial+=1
            else:
                # start a new trial
                self.startTrial(updateTime)
        else:    
            # save eye tracker event for synchronization        
            if self.settings.saveEyeTracker:
                self.e.saveEyeTrackerEvent(eventType="segment", taskID=self.settings.taskID, trialNum=self.state.currentTrial, segmentNum=self.state.currentSegment, timestamp=updateTime)
            
            # update to next segment
            self.state.currentSegment += 1
            self.state.segmentStartTime = updateTime
            self.data.events.append(pglEventSegment(self.state.currentSegment, updateTime))
            
            # default to false, this will get reset
            # at end of segment clock if set for this segment
            self.waitUntilVolumeTrigger = False

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
            self.e.saveEyeTrackerEvent(eventType="trial", taskID=self.settings.taskID, trialNum=self.state.currentTrial, segmentNum=self.state.currentSegment, timestamp=startTime)

        # get current parameters
        self.data.params.append({})
        self.currentParams = self.data.params[-1]
        for parameter in self.parameters: 
            self.data.params[-1].update(parameter.get())

        # start segment (startSegment will update currentSegment to 0)
        self.state.currentSegment = -1
        self.startSegment(startTime)
        
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
        # store references
        self.state.subjectResponses = subjectResponses
        self.state.phaseNum = phaseNum
        self.state.tasks = tasks
        
        # custom handling of events
        self.handleEvents(events)
        
        # check for end of segment
        if self.waitUntilVolumeTrigger:
            if self.e.state.volumeNumber > self.lastVolumeNumber:
                # volume trigger received, end segment
                self.startSegment(updateTime)
        if  updateTime - self.state.segmentStartTime >= self._thisTrialSeglen[self.state.currentSegment]:
            # check if we need to wait until volume trigger
            if self.settings.waitUntilVolumeTrigger[self.state.currentSegment]:
                self.waitUntilVolumeTrigger = True
                self.lastVolumeNumber = self.e.state.volumeNumber
            else:
                # call startSegment to begin next segment
                self.startSegment(updateTime)
        
        # if there are responses, call response callback
        if subjectResponses != []:
            # Pass each subjectResponse in sequence to handleSubjectResponse
            for subjectResponse in subjectResponses:
                # call the subject response handler
                responseType = self.handleSubjectResponse(subjectResponse, updateTime)
                # save as an event if responseType is not None
                # responseType can be used to specify different types of responsees
                # and is defined by the subclass
                if responseType is not None:
                    self.data.events.append(pglEventSubjectResponse(response=subjectResponse, timestamp=updateTime, responseType=responseType))
                
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
    
    def save(self, dataDir):
        '''
        Save the task settings, state and data.
        '''
        try:
            dataDir = dataDir / self.settings.taskSaveName
            dataDir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"(pglTask:save) ❌ Could not create task data directory {dataDir}: {e}")
            return
        
        # save settings, state and data
        self.settings.save(dataDir / "settings.json")
        self.state.save(dataDir / "state.json")
        self.data.save(dataDir / "data.json")
        
        # save parameters
        try:
            paramDir = dataDir / "parameters"
            paramDir.mkdir(parents=True, exist_ok=True)
            for param in self.parameters:
                param.save(paramDir)
        except Exception as e:
            print(f"(pglTask:save) ❌ Could not save task parameters to {dataDir}: {e}")

    def load(self, taskDir):
        '''
        Load the task data.
        '''
        print(f"(pglTask:load) Loading task data from: {taskDir}")
        
        # load settings, state and data
        try:
            self.settings = pglSerialize.load(taskDir / "settings.json")
            self.state = pglSerialize.load(taskDir / "state.json")
            self.data = pglSerialize.load(taskDir / "data.json")
        except Exception as e:
            print(f"(pglTask:load) ❌ Could not load task data from {taskDir}: {e}")    
        
        try:
            # load parameters
            self.parameters = []
            for paramDir in (taskDir / "parameters").iterdir():
                param = pglParameter.from_file(paramDir)
                self.parameters.append(param)
        except Exception as e:
            print(f"(pglTask:load) ❌ Could not load task parameters from {taskDir / 'parameters'}: {e}")

    def display(self):
        '''
        Display the task data
        '''
        self.data.display(self.settings.taskName)
    
    def print(self):
        '''
        Print a summary of the task data
        '''
        from pgl import pglTimestamp
        timestamp = pglTimestamp()
        
        # print task name and number of trials
        print(f"Task: {self.settings.taskName} | Trials: {self.state.currentTrial+1}")
        print(f"Duration={timestamp.formatDuration(self.data.endTime - self.data.startTime)} | startTime={self.data.startTime} | endTime={self.data.endTime}")
        
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
            trialVolume = self.e.getNearestVolumeTrigger(trialEvent)
            if trialVolume is None:
                print(f"Trial {iTrial+1} at {trialStart:.2f}s: " + ', '.join(f"{key}={value}" for key, value in params.items()))
            else:
                print(f"Trial {iTrial+1} at {trialStart:.2f}s (vol={trialVolume}): " + ', '.join(f"{key}={value}" for key, value in params.items()))

##############################################
# test task for testing settings
##############################################
class pglTestTask(pglTask):
    responseText = ""
    def updateScreen(self):
        # put upt the bulls eye
        self.pgl.bullseye()
        # display how to end
        self.pgl.text("Press 'ESC' to quit",xAlign=1)
        # and text for what trial we are on 
        # This will just update every trial
        self.pgl.text(f"Trial {self.state.currentTrial+1}",xAlign=1)
        if self.e is not None:
            self.pgl.text(f"Volume {self.e.state.volumeNumber}",xAlign=1)
            elapsed = self.pgl.getSecs() - self.e.data.startTime
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.pgl.text(f"{minutes:02d}:{seconds:02d}",xAlign=1)
            if self.responseText != "":
                self.pgl.text(self.responseText,xAlign=1)
    
    def handleSubjectResponse(self, response, updateTime):
        self.responseText = f"Subject response received: {response} at {updateTime - self.e.data.startTime:.2f} seconds"


##############################################
# Settings for pglExperiment
##############################################
class pglExperimentSettings(pglSettingsEditable):
    experimentName = Unicode("Default experiment", help="Name of the experiment")
    experimentSaveName = Unicode("defaultExperiment", help="Name to use when saving experiment data (defaults to camelCase version of experimentName)")
    subjectID = Unicode("s0000", help="Identifier for the subject participating in the experiment.")
    tasks = List(trait=Unicode(), default_value=[], help="Task names")
    
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
@dataclass
class pglExperimentData(pglSerialize):
    startTime: float = 0.0
    endTime: float = 0.0
    events: ListType[pglEvent] = field(default_factory=list) 
    
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
    
    def display(self, e=None):
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
            self.volumeTriggerKey = ""
        
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
        timeline = timelinePlot(startTime=startTime, endTime=max(self.endTime-self.startTime,10))
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
        timeline.show()
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
@dataclass
class pglExperimentState(pglSerialize):
    phaseNum: Optional[int] = None
    phaseNums: Optional[ListType[int]] = None    
    currentPhaseIndex: int = 0
    #currentTasks: ListType[pglTask] = field(default_factory=list)
    openScreen: bool = False
    volumeNumber: int = 0
    experimentStarted: bool = False
    experimentDone: bool = False
    responseKeyCodesList: ListType[int] = field(default_factory=list)
    startKeyCode: int = 0
    endKeyCode: int = 0
    volumeTriggerKeyCode: int = 0

##############################################
# Settings for pglTask
##############################################
class pglTaskSettings(pglSettingsEditable):
    taskName = Unicode("Default task", help="Name of the task")
    taskSaveName = Unicode("defaultTask", help="Name to use when saving task data (defaults to camelCase version of taskName)")    
    phaseNum = Int(default_value=None, allow_none=True, help="Phase number for the task. Set to None if this should run in all phases")
    seglen = List(Float(), help="List of segment lengths in seconds.")
    segmin = List(Float(), help="Minimum length of a segment.")
    segmax = List(Float(), help="Maximum length of a segment.")
    waitUntilVolumeTrigger = List(Bool(), help="List of nSegments where if set to true will run through the segment length and then wait for a volume trigger to continue.")
    nSegments = Int(help="Number of segments in the task.")
    nTrials = Float(np.inf, help="Number of trials to run for.")
    fixedParameters = Dict(default_value={}, help="Dictionary of fixed parameters for the task.")
    saveEyeTracker = Bool(False, help="Whether to save eye tracker events this task (if we have an eye tracker).")    
    taskID = Int(0, help="Numeric identifier for the task, used for pglExperiment to keep track of tasks.")

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
        pglSettingsEditable.updateTraitsFromDict(self, data, filename, typeConverter)
##############################################
# State for pglTask
##############################################
@dataclass
class pglTaskState(pglSerialize):
    phaseNum: Optional[int] = None
    currentTrial: int = 0
    currentSegment: int = 0
    subjectResponses: ListType[int] = field(default_factory=list)

##############################################
# State for pglTask
##############################################
@dataclass
class pglTaskData(pglSerialize):
    startTime: Optional[float] = None
    endTime: Optional[float] = None
    events: ListType[pglEvent] = field(default_factory=list) 
    params: ListType[dict] = field(default_factory=list)
    
    def display(self, taskName="task", responseMapping={True:('Correct','green'), False:('Incorrect','red')}):
        '''
        Display the experiment data.
        '''
        # get trial timestamps
        trialTimestamps = np.array([e.timestamp for e in self.events if isinstance(e, pglEventTrial)])
        if len(trialTimestamps) < 2:
            print("(pglTaskData:display) Insufficient trial events found to display.")
            return
        
        # get the max trial length
        maxTrialLength = np.diff(trialTimestamps[:-1]).max()
        
        # init timeline
        timeline = timelinePlot(startTime=0, endTime=maxTrialLength)
        
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
        timeline.show()


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
    
    def __init__(self, startTime=0, endTime=100, figsize=(12, 4)):
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
        print(f"(pglEventSegment) Segment {self.eventType} at: {self.timestamp}")
        

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

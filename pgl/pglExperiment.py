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
from datetime import datetime
import json
import numpy as np
import itertools
import random
import math
from datetime import datetime
from dataclasses import dataclass, field
from .pglKeyboardMouse import pglKeyboardMouse
from pathlib import Path
from .pglSettings import pglSettingsManager, pglSettings, pglSettingsEditable
from IPython.display import display, HTML
from .pglBase import pglDisplayMessage
from traitlets import Float, TraitError, TraitError, observe, Instance, Int, Unicode, Dict, validate, Bool
from .pglParameter import pglParameter, pglParameterBlock
from .pglEvent import pglEvent
from .pglSerialize import pglSerialize
from typing import List as ListType
from traitlets import List
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from enum import Enum
from . import pglTimestamp

##############################################s
# Experiment class
##############################################
class pglExperiment(pglSettingsManager):
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
        
        # initialize tasks
        self.task = [[]]
        
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
        keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
        if keyboardDevices is not []:
            for keyboardDevice in keyboardDevices:
                keyboardDevice.listener.eatAllKeys = eat

    def addTask(self, task, phaseNum = None):
        '''
        Add a task to the experiment.
        '''
        # get which phase to add the task to
        if phaseNum == None: phaseNum = len(self.task)-1

        # check if phaseNum is valid
        if phaseNum < 0 or phaseNum > len(self.task):
            print(f"(pglExperiment:addTask) Experiment only has phases from 0 to {len(self.task)-1}.")
            print(f"                         You can add a new phase by using addTask and setting phaseNum to {len(self.task)}")
            return
        
        # add a phase
        if phaseNum == len(self.task):
            self.task.append([])

        # give it a reference to pgl and experiment
        task.pgl = self.pgl
        task.e = self

        # add the task
        self.task[phaseNum].append(task)

    def run(self):
        '''
        Run the experiment.
        '''
        if self.state.openScreen == False:
            pglDisplayMessage("(pglExperiment:run) ❌ Screen is not open. Call initScreen() before running the experiment.",useHTML=True, duration=5)
            return
        
        self.state.experimentDone = False
        self.state.volumeNumber = 0

        # wait for key press to start experiment
        if self.settings.startKey is not [] or self.settings.startOnVolumeTrigger:
            self.state.experimentStarted = False
            if self.settings.startOnVolumeTrigger:
                self.pgl.text("Waiting for volume trigger to start experiment",y=0)
            else:
                self.pgl.text(f"Press {self.settings.startKey} key to start experiment",y=0)
            # flush to display text
            self.pgl.flush()
            while not self.state.experimentStarted:
                # poll for events
                events = self.pgl.poll()
                self.data.events.extend(events)

                # see if we have a match to startKey
                if [e for e in events if e.type == "keyboard" and e.keyCode == self.state.startKeyCode]:
                    self.state.experimentStarted = True
                
                # if waiting to startOnVolumeTrigger, check for that key                
                if self.settings.startOnVolumeTrigger:
                    if [e for e in events if e.type == "keyboard" and e.keyCode == self.state.volumeTriggerKeyCode]:
                        self.state.experimentStarted = True
                        self.state.volumeNumber += 1
                        # and add a volume event
                        self.data.events.append(pglEventVolumeTrigger(timestamp=e.timestamp))
                
                # Check for end key to allow aborting before starting    
                if [e for e in events if e.type == "keyboard" and e.keyCode == self.state.endKeyCode]:
                    self.state.experimentStarted = True
                    self.state.experimentDone = True

        self.startPhase()
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
                for task in self.task[self.state.currentPhase]: task.end()
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
            subjectResponse = [keyIndex for e in events if e.type == "keyboard" and e.eventType == "keydown" and e.keyCode in self.state.responseKeyCodesList for keyIndex in [self.state.responseKeyCodesList.index(e.keyCode)]]

            # update tasks in current phase
            phaseDone = False
            updateTime = self.pgl.getSecs()
            for task in self.task[self.state.currentPhase]:
                # update task
                task.update(updateTime=updateTime, subjectResponse=subjectResponse, phaseNum=self.state.currentPhase, tasks=self.task[self.state.currentPhase], events=events)
                # check if task is done
                if task.done(): phaseDone = True
            
            # update the screen
            self.pgl.flush()

            # go to next phase or end experiment
            if phaseDone:
                # end all tasks in current phase
                for task in self.task[self.state.currentPhase]: task.end()
                # update phase
                self.state.currentPhase += 1
                # check if we have ended all phases
                if self.state.currentPhase >= len(self.task):
                    self.state.experimentDone = True
                else:
                    self.startPhase()
        
        # mark end time
        self.data.endTime = self.pgl.getSecs()
        print("(pglExperiment:run) Experiment done.")
        
        # save data
        self.save()
        
        # close screen
        self.endScreen()

    def startPhase(self):
        '''
        Start the current phase of the experiment.
        '''
        print(f"(pglExperiment:startPhase) Starting phase: {self.state.currentPhase+1}/{len(self.task)}")
        startTime = self.pgl.getSecs()
        for task in self.task[self.state.currentPhase]:
            task.start(startTime)
    
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
        
        # save settings
        self.settings.save(dataDir / "settings.json")
        self.experimentSettings.save(dataDir / "experimentSettings.json")

        # save state
        self.state.save(dataDir / "state.json")
        
        # save data
        self.data.save(dataDir / "data.json")

        # save each task
        for phaseTasks in self.task:
            for task in phaseTasks:
                task.save(dataDir)   
                     
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

    def load(self, experimentName="", subjectID=""):
        '''
        Load the experiment settings, state and data.         
        '''
        self.pglTimestamp = pglTimestamp()
        dataDir = Path(self.settings.dataPath).expanduser() / experimentName / subjectID 
        # check that data path exists
        if not dataDir.exists() or not dataDir.is_dir():
            print(f"(pglExperiment:load) ❌ Could not find data directory: {dataDir}")
            return
        
        # Get all directories in the dataPath
        dirList = [d for d in dataDir.iterdir() if d.is_dir()]
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
                        experimentData = pglSerialize.load(dataDir / d.name / "data.json")
                        # print number of volumes
                        numVols = experimentData.getNumEvents(type="pglEventVolumeTrigger")
                        if numVols==0:
                            numVols = experimentData.getNumEvents(type="keyboard", eventType="keydown", keyChar="5")
                        dataPrintname += f" | nVols: {numVols}"
                        # print duration
                        dataPrintname += f" | Duration: {self.pglTimestamp.formatDuration(experimentData.endTime-experimentData.startTime)}"
                    except:
                        pass
                # Add the filename
                dataPrintname = f"{dataPrintname} ({d.name})"
            except:
                # not a typical name, just display
                dataPrintname = d.name
            print(f"{i}. {dataPrintname}")
                
        # Ask the user to choose
        choice = int(input("\nSelect a directory number: "))

        # Get the selected directory
        selectedDir = dirList[choice - 1]

        print("\nYou selected:", selectedDir)


    def display(self):
        '''
        Display a timeline of experiment events.
        '''
        # display experiment data
        self.data.display(self)
        
        # display task data
        for phaseTasks in self.task:
            for task in phaseTasks:
                task.display()   

    
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
    def __init__(self, pgl=None):
        self.pgl = pgl
        self.settings = pglTaskSettings()
        self.state = pglTaskState()
        self.data = pglTaskData()
        
        # default seglen
        self.settings.seglen = [1.0]
        
        # these get set by update
        self.phaseNum = None
        self.tasks = None
        self.e = None

    def start(self, startTime):
        '''
        Start the task.
        '''
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
            self._thisTrialSeglen[self.state.currentSegment] = updateTime - self.segmentStartTime

        # update to next segment
        self.state.currentSegment += 1
        self.segmentStartTime = updateTime
        self.data.events.append(pglEventSegment(self.state.currentSegment, updateTime))
        
        # default to false, this will get reset
        # at end of segment clock if set for this segment
        self.waitUntilVolumeTrigger = False
        
        # check for end of trial
        if self.state.currentSegment >= self.settings.nSegments: 
            # new trial
            self.startTrial(updateTime)


    def startTrial(self, startTime):
        '''
        Start a trial.
        '''
        # update values
        self.state.currentTrial += 1
        self.data.events.append(pglEventTrial(self.state.currentTrial, startTime))
        
        # get current parameters
        self.data.params.append({})
        self.currentParams = self.data.params[-1]
        for parameter in self.settings.parameters: 
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
        #for name,value in self.data.params[-1].items():
        #    print(f'{name}={value}', end=' ')

    def addParameter(self, param):
        '''
        Add a parameter to the task.
        '''
        self.settings.parameters.append(param)

    def update(self, updateTime, subjectResponse, phaseNum, tasks, events):
        '''
        Update the task.
        '''
        # store references
        self.phaseNum = phaseNum
        self.tasks = tasks
        
        # custom handling of events
        self.handleEvents(events)
        
        # update the screen
        self.updateScreen()

        # check for end of segment
        if self.waitUntilVolumeTrigger:
            if self.e.state.volumeNumber > self.lastVolumeNumber:
                # volume trigger received, end segment
                self.startSegment(updateTime)
        if  updateTime - self.segmentStartTime >= self._thisTrialSeglen[self.state.currentSegment]:
            # check if we need to wait until volume trigger
            if self.settings.waitUntilVolumeTrigger[self.state.currentSegment]:
                self.waitUntilVolumeTrigger = True
                self.lastVolumeNumber = self.e.state.volumeNumber
            else:
                # call startSegment to begin next segment
                self.startSegment(updateTime)
        
        # if there are responses, call response callback
        if subjectResponse is not []:
            # call the subject response handler
            responseType = self.handleSubjectResponse(subjectResponse, updateTime)
            # save as an event if responseType is not None
            # responseType can be used to specify different types of responsees
            # and is defined by the subclass
            if responseType is not None:
                self.data.events.append(pglEventSubjectResponse(response=subjectResponse, timestamp=updateTime, responseType=responseType))
                

    def handleSubjectResponse(self, responses, updateTime) -> None:
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
    
    def done(self):
        '''
        Check if the task is done.
        '''
        # check if we are done
        taskDone = self.state.currentTrial >= self.settings.nTrials
        if taskDone: self.end()
        return taskDone

    def end(self):
        '''
        end of task
        '''
        # record end time
        endTime = self.pgl.getSecs()
        self.data.endTime = endTime
        
        # put in time stamps for end of last segment and trial
        self.data.events.append(pglEventSegment(self.state.currentSegment, endTime, boundary=pglEventSegment.boundaryType.END))
        self.data.events.append(pglEventTrial(self.state.currentTrial, endTime, boundary=pglEventTrial.boundaryType.END))

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
        
        self.settings.save(dataDir / "settings.json")
        self.state.save(dataDir / "state.json")
        self.data.save(dataDir / "data.json")

    def load(self):
        '''
        Load the task data.
        '''
        pass

    def display(self):
        '''
        Display the task data
        '''
        self.data.display(self.settings.taskName)

##############################################
# test task for testing settings
##############################################
class pglTestTask(pglTask):
    responseText = ""
    def updateScreen(self):
        # put upt the bulls eye
        self.pgl.bullseye()
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
    
    def handleSubjectResponse(self, responses, updateTime):
        for response in responses:
            self.responseText = f"Subject response received: {response} at {updateTime - self.e.data.startTime:.2f} seconds"


##############################################
# Settings for pglExperiment
##############################################
class pglExperimentSettings(pglSettingsEditable):
    experimentName = Unicode("Default experiment", help="Name of the experiment")
    experimentSaveName = Unicode("defaultExperiment", help="Name to use when saving experiment data (defaults to camelCase version of experimentName)")
    subjectID = Unicode("s0000", help="Identifier for the subject participating in the experiment.")
    
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
        # get infor from experiment if provided
        if e is not None:
            self.volumeTriggerKey = e.settings.volumeTriggerKey
        else:
            self.volumeTriggerKey = ""
        
        # init timeline
        timeline = timelinePlot(startTime=0, endTime=self.endTime-self.startTime)
        
        # for each event, add to timeline
        for event in self.events:
            if event.type == "keyboard":
                if event.eventType == "keydown":
                    if event.keyChar == self.volumeTriggerKey:
                        timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='blue', direction='up')
                    else:
                        timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='green', label=f'{event.keyChar}', direction='down')
                elif (event.keyChar == "escape"):
                    timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='red', label=f'{event.keyChar}', direction='down')
        timeline.setTitle("Experiment Events")
        timeline.addLegend([{'label': 'Keypress', 'color': 'green'},{'label': 'Volumes', 'color': 'blue'}])
        timeline.show()

##############################################
# State for pglExperiment
##############################################
@dataclass
class pglExperimentState(pglSerialize):
    currentPhase: int = 0
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
    seglen = List(Float(), help="List of segment lengths in seconds.")
    segmin = List(Float(), help="Minimum length of a segment.")
    segmax = List(Float(), help="Maximum length of a segment.")
    waitUntilVolumeTrigger = List(Bool(), help="List of nSegments where if set to true will run through the segment length and then wait for a volume trigger to continue.")
    nSegments = Int(help="Number of segments in the task.")
    nTrials = Float(np.inf, help="Number of trials to run for.")
    parameters = List(Instance(pglParameter), default_value=[], help="List of parameters (type pglParameter) for the task.")
    fixedParameters = Dict(default_value={}, help="Dictionary of fixed parameters for the task.")
    
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
            self.taskSaveName = words[0].lower() + "".join(word.capitalize() for word in words[1:])
        
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

##############################################
# State for pglTask
##############################################
@dataclass
class pglTaskState(pglSerialize):
    currentTrial: int = 0
    currentSegment: int = 0

##############################################
# State for pglTask
##############################################
@dataclass
class pglTaskData(pglSerialize):
    startTime: float = 0.0
    endTime: float = 0.0
    events: ListType[pglEvent] = field(default_factory=list) 
    params: ListType[dict] = field(default_factory=list)
    
    def display(self, taskName="task", responseMapping={True:('Correct','green'), False:('Incorrect','red')}):
        '''
        Display the experiment data.
        '''
        # get trial timestamps
        trialTimestamps = np.array([e.timestamp for e in self.events if isinstance(e, pglEventTrial)])
        if len(trialTimestamps) == 0:
            print("(pglTaskData:display) No trial events found to display.")
            return
        
        # get the max trial length
        maxTrialLength = np.diff(trialTimestamps).max()
        
        # init timeline
        timeline = timelinePlot(startTime=0, endTime=maxTrialLength)
        
        # for each event, add to timeline
        trialStart = None
        gotResponse = False
        for event in self.events:
            # if we find a new trial event, reset the beginning time
            if isinstance(event, pglEventTrial):
                trialStart = event.timestamp
            elif trialStart is not None:
                # display segment events
                if isinstance(event, pglEventSegment) and event.boundary == pglEventSegment.boundaryType.START.value:
                    timeline.addTriangleMarker(time=event.timestamp - trialStart, color='blue', label=f'{event.segmentNum}', direction='up')
                # display subject response events
                elif isinstance(event, pglEventSubjectResponse):
                    gotResponse = True
                    label, color = responseMapping.get(event.responseType, ('?', 'gray'))
                    timeline.addTriangleMarker(time=event.timestamp - trialStart, color=color, label=label[0], direction='down')   
        timeline.setTitle(f"{taskName}: Trial Events")
        # display legend
        legend = [{'label': 'Segment', 'color': 'blue'}]
        # add the response values
        if gotResponse:
            for respType, (label, color) in responseMapping.items():
                legend.append({'label': label, 'color': color})
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

    def __init__(self, trialNum=None, timestamp=None, boundary=None):
        super().__init__(type="pglEventTrial")

        # handle default
        if boundary is None:
            boundary = self.boundaryType.START
            
        # set attributes
        self.trialNum = trialNum
        self.timestamp = timestamp
        self.boundary = boundary.value

    def print(self):
        print(f"(pglEventTrial) Trial {self.boundary} at: {self.timestamp}")
        
#################################################################
# Events that specify segment timing
#################################################################
class pglEventSegment(pglEvent):

    class boundaryType(Enum):
        START = 'start'
        END = 'end'

    def __init__(self, segmentNum = None, timestamp=None, boundary=None):
        super().__init__(type="pglEventSegment")

        # handle default
        if boundary is None:
            boundary = self.boundaryType.START
        
        # set attributes
        self.segmentNum = segmentNum
        self.boundary = boundary.value
        self.timestamp = timestamp

    def print(self):
        print(f"(pglEventSegment) Segment {self.boundary} at: {self.timestamp}")
        

#################################################################
# Events that specify subject response
#################################################################
class pglEventSubjectResponse(pglEvent):
    
    def __init__(self, response=None, timestamp=None, responseType=None):
        super().__init__(type="pglEventSubjectResponse")
        
        # set attributes
        self.response = response
        self.timestamp = timestamp
        self.responseType = responseType

#################################################################
# Events that specifys mri volume trigger
#################################################################
class pglEventVolumeTrigger(pglEvent):
    
    def __init__(self, timestamp=None):
        super().__init__(type="pglEventVolumeTrigger")
        
        # set attributes
        self.timestamp = timestamp

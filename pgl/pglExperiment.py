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
import numpy as np
import itertools
import random
import math
from . import pglKeyboardMouse
from pathlib import Path
from .pglSettings import pglSettingsManager, pglSettings
from IPython.display import display, HTML
from .pglBase import pglDisplayMessage

#############
# Experiment class
#############
class pglExperiment(pglSettingsManager):
    '''
    Experiment class which handles timing, parameter randomization,
    subject response, synchronizing with measurement hardware etc
    '''
    def __init__(self, pgl, settingsName=None, settings=None):
        '''
        Initialize the pglExperiment class.
        
        Args:
            pgl (pgl): An instance of the pgl class.
            settingsName (str): The name of the settings to use. If not set (and settings not set), will use default settings
            settings (pglSettings): An instance of the pglSettings class. If set, will supersede settingsName.
        '''
        # save pgl
        self.pgl = pgl
        
        # current phase of experiment
        self.currentPhase = 0
        self.openScreen = False
        
        # initialize tasks
        self.task = [[]]

        # get  settings
        self.settings = settings
        if self.settings is None:
            self.settings = self.getSettings(settingsName)
        
        # if there was some error, then display it
        if self.settings is None:
            # display error in HTML
            pglDisplayMessage("<b>(pglExperiment:initScreen)</b> ❌ Could not find settings, using default settings.")
            # create default settings
            self.settings = pglSettings()
            
        
    def __repr__(self):
        return f"<pglExperiment: {len(self.task)} phases>"
    
    def initScreen(self):
        '''
        Initialize the screen for the experiment. This will call pgl.open() and
        set parameters according to what is set in setParameters
        '''        
        if self.settings is None:
            print("(pglExperiment:initScreen) No settings found to open screen.")
            return
        
        # open screen
        if self.settings.displayNumber == 0:
            self.pgl.open(0, self.settings.windowWidth, self.settings.windowHeight)        
        else:
            self.pgl.open(self.settings.displayNumber-1)        
            
        # set visual angle coordinates
        self.pgl.visualAngle(self.settings.displayDistance,self.settings.displayWidth,self.settings.displayHeight)
        
        # add keyboard device if not already loaded
        keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
        if not keyboardDevices:
            keyboardMouse = pglKeyboardMouse(eatKeys=None)
            self.pgl.devicesAdd(keyboardMouse)
            # check if listener is running
            if not keyboardMouse.isRunning():
                pglDisplayMessage("<b>(pglExperiment:initScreen)</b> ❌ Accessibility permission not granted for keyboard/mouse access.", useHTML=True)
                pglDisplayMessage("On macOS, go to System Preferences -> Security & Privacy -> Privacy -> Accessibility, and add your terminal application (e.g. Terminal, iTerm, etc) to the list of apps allowed to control your computer.", useHTML=True)
                pglDisplayMessage("If you are running VS Code and it already has permissions granted, try running directly from a terminal with:", useHTML=True)
                pglDisplayMessage("              /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Electron", useHTML=True)
        else:
            # if already loaded, just grab it
            keyboardMouse = keyboardDevices[0]
        
        # If response keys is a comma-separated list, split it into a list (this is so you can do "1,space,F1,2"
        if ',' in self.settings.responseKeys:
            self.responseKeysList = [k.strip() for k in self.settings.responseKeys.split(',')]
        else:
            # if no commas, then just make a list of characters
            self.responseKeysList = list(self.settings.responseKeys)

        # if eatKeys is set, then compose a list of all keys as keyCodes
        if self.settings.eatKeys:
            eatKeyCodes = []
    
            # Collect all individual keys
            allKeys = self.responseKeysList.copy()  # Start with response keys list
    
            # Add single keys if they exist
            if self.settings.startKey:
                allKeys.append(self.settings.startKey)
            if self.settings.endKey:
                allKeys.append(self.settings.endKey)
            if self.settings.volumeTriggerKey:
                allKeys.append(self.settings.volumeTriggerKey)
    
            # Convert all to keycodes
            for keyChar in allKeys:
                keyCode = keyboardMouse.charToKeyCode(keyChar)
                if keyCode is not None:
                    eatKeyCodes.append(keyCode)
    
            keyboardMouse.setEatKeys(eatKeyCodes)
            
        # wait half a second for metal app to initialize
        self.pgl.waitSecs(0.5)
        
        # flush screen to get rid of any transients
        self.pgl.flush()
        self.pgl.flush()
        
        # mark that we have opened the screen
        self.openScreen = True

    def endScreen(self):
        '''
        Close the screen
        '''
        # stop the keyboard listener
        keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
        if keyboardDevices is not []:
            for keyboardDevice in keyboardDevices:
                keyboardDevice.stop()

        if self.settings.closeScreenOnEnd:
            # close screen
            self.pgl.close()

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

        # give it a reference to pgl
        task.pgl = self.pgl

        # add the task
        self.task[phaseNum].append(task)

    def run(self):
        '''
        Run the experiment.
        '''
        experimentDone = False
        self.volumeNumber = 0

        # wait for key press to start experiment
        if self.settings.startKey is not [] or self.settings.startOnVolumeTrigger:
            experimentStarted = False
            if self.settings.startOnVolumeTrigger:
                self.pgl.text("Waiting for volume trigger to start experiment",y=0)
            else:
                self.pgl.text(f"Press {self.settings.startKey} key to start experiment",y=0)
            # flush to display text
            self.pgl.flush()
            while not experimentStarted:
                # poll for events
                events = self.pgl.poll()

                # see if we have a match to startKey
                if [e for e in events if e.type == "keyboard" and e.keyChar in self.settings.startKey]:
                    experimentStarted = True
                
                # if waiting to startOnVolumeTrigger, check for that key                
                if self.settings.startOnVolumeTrigger:
                    if [e for e in events if e.type == "keyboard" and e.keyChar in self.settings.volumeTriggerKey]:
                        experimentStarted = True
                        self.volumeNumber += 1
                
                # Check for end key to allow aborting before starting    
                if [e for e in events if e.type == "keyboard" and e.keyChar in self.settings.endKey]:
                    experimentStarted = True
                    experimentDone = True

        self.startPhase()
        print(f"(pglExperiment:run) Experiment started.")
        self.startTime = self.pgl.getSecs()

        while not experimentDone:
            
            # poll for events
            events = self.pgl.poll()

            # see if we have a match to endKey
            if [e for e in events if e.type == "keyboard" and e.keyChar in self.settings.endKey]:
                experimentDone = True
                continue

            # Check for volume trigger key
            if [e for e in events if e.type == "keyboard" and e.keyChar in self.settings.volumeTriggerKey and e.eventType == "keydown"]:    
                self.volumeNumber += 1

            # grab any events that match the keyList and return their index within that list
            subjectResponse = [keyIndex for e in events if e.type == "keyboard" and e.eventType == "keydown" and e.keyChar in self.responseKeysList for keyIndex in [self.responseKeysList.index(e.keyChar)]]

            # update tasks in current phase
            phaseDone = False
            updateTime = self.pgl.getSecs()
            for task in self.task[self.currentPhase]:
                # update task
                task.update(updateTime=updateTime, subjectResponse=subjectResponse, phaseNum=self.currentPhase, tasks=self.task[self.currentPhase], experiment=self)
                # check if task is done
                if task.done(): phaseDone = True
            
            # update the screen
            self.pgl.flush()

            # go to next phase or end experiment
            if phaseDone:
                # update phase
                self.currentPhase += 1
                # check if we have ended all phases
                if self.currentPhase >= len(self.task):
                    experimentDone = True
                else:
                    self.startPhase()
        
        self.endTime = self.pgl.getSecs()
        print("(pglExperiment:run) Experiment done.")
        
        # close screen
        self.endScreen()

    def startPhase(self):
        '''
        Start the current phase of the experiment.
        '''
        print(f"(pglExperiment:startPhase) Starting phase: {self.currentPhase+1}/{len(self.task)}")
        startTime = self.pgl.getSecs()
        for task in self.task[self.currentPhase]:
            task.start(startTime)

#############
# Task class
#############
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
        self._seglen = []
        self.nTrials = np.inf
        self.parameters=[]
        self.name="Task"
        self.seglen = [1.0]  # default segment length of 1 second
        
        self.phaseNum = None
        self.tasks = None
        self.e = None
    ################################################################
    # seglen property
    ################################################################
    @property
    def seglen(self):
        # Get the current segment length.
        return self._seglen
    @seglen.setter
    def seglen(self, seglen_):
        # seglen setting set both segmin/segmax
        self._segmin = seglen_
        self._segmax = seglen_
        self._seglen = seglen_
        self.nSegments = len(seglen_)

    def start(self, startTime):
        '''
        Start the task.
        '''
        # set task start time
        self.taskStartTime = startTime
        
        # start trial
        self.currentTrial = -1
        self.startTrial(startTime)
        
    def startSegment(self, startTime):
        '''
        Start a segment.
        '''
        self.currentSegment += 1
        self.segmentStartTime = startTime

    def startTrial(self, startTime):
        '''
        Start a trial.
        '''
        # update values
        self.currentTrial += 1
        self.trialStartTime = startTime
        
        # start segment (startSegment will update currentSegment to 0)
        self.currentSegment = -1
        self.startSegment(startTime)
        
        # get a random length for each segment. If segmin==segmax, then fixed length
        self._thisTrialSeglen = [
            # if either segmin or segmax is infinite, set to infinite
            float('inf') if math.isinf(min_val) or math.isinf(max_val) 
            # otherwise choose a random length between min and max
            else random.uniform(min_val, max_val)
            for min_val, max_val in zip(self._segmin, self._segmax)
        ]
        print("seglen: ",[f"{x:.2g}" for x in self._thisTrialSeglen])

        # get current parameters
        self.currentParams = {}
        for parameter in self.parameters: 
            self.currentParams.update(parameter.get())

        # print trial
        print(f"({self.name}) Trial {self.currentTrial+1}: ", end='')
        
        
        # and variable settings
        for name,value in self.currentParams.items():
            print(f'{name}={value}', end=' ')
        print()

    def addParameter(self, param):
        '''
        Add a parameter to the task.
        '''
        self.parameters.append(param)

    def update(self, updateTime, subjectResponse, phaseNum, tasks, experiment):
        '''
        Update the task.
        '''
        # store references
        self.phaseNum = phaseNum
        self.tasks = tasks
        self.e = experiment
        
        # update the screen
        self.updateScreen()

        # check for end of segment
        if updateTime - self.segmentStartTime >= self._thisTrialSeglen[self.currentSegment]:
            # if the segment len is set to 0, it is a jump segment command
            # so reset the segment length to how long actually elapsed
            # so there is a record of how long we were in that segment
            if self._thisTrialSeglen[self.currentSegment] == 0:
                self._thisTrialSeglen[self.currentSegment] = updateTime - self.segmentStartTime
            # call startSegment to begin next segment
            self.startSegment(updateTime)
            # check for end of trial
            if self.currentSegment >= self.nSegments: 
                # new trial
                self.startTrial(updateTime)
        
        # if there are responses, call response callback
        if subjectResponse is not []:
            self.handleSubjectResponse(subjectResponse, updateTime)

    def handleSubjectResponse(self, responses, updateTime):
        '''
        Handle subject responses.
        '''
        for response in responses:
            print(f"(pglExperiment) Subject response received: {response} at {updateTime}")

    def updateScreen(self):
        '''
        Update the screen.
        '''
        pass
    
    def done(self):
        '''
        Check if the task is done.
        '''
        return self.currentTrial >= self.nTrials

    def jumpSegment(self):
        '''
        Jump to the next segment.
        '''
        # set current segment length to 0 to force jump
        self._thisTrialSeglen[self.currentSegment] = 0

# test task for testing settings
class pglTestTask(pglTask):
    responseText = ""
    def updateScreen(self):
        # put upt the bulls eye
        self.pgl.bullseye()
        # and text for what trial we are on 
        # This will just update every trial
        self.pgl.text(f"Trial {self.currentTrial+1}",xAlign=1)
        if self.e is not None:
            self.pgl.text(f"Volume {self.e.volumeNumber}",xAlign=1)
            elapsed = self.pgl.getSecs() - self.e.startTime
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.pgl.text(f"{minutes:02d}:{seconds:02d}",xAlign=1)
            if self.responseText != "":
                self.pgl.text(self.responseText,xAlign=1)
    
    def handleSubjectResponse(self, responses, updateTime):
        for response in responses:
            self.responseText = f"Subject response received: {response} at {updateTime - self.e.startTime:.2f} seconds"

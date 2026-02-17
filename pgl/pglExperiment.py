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
from .pglSettings import pglSettingsManager
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
    def __init__(self, pgl, settingsName=None, suppressInitScreen=False, suppressEndScreen=False):
        # save pgl
        self.pgl = pgl
        
        # initialize screen
        if not suppressInitScreen: self.initScreen(settingsName)

        # load parameters
        self.loadParameters()
        self.suppressEndScreen = suppressEndScreen
        
        # current phase of experiment
        self.currentPhase = 0
        
        # initialize tasks
        self.task = [[]]
        
    def __repr__(self):
        return f"<pglExperiment: {len(self.task)} phases>"
    
    def initScreen(self, settingsName=None, settings=None):
        '''
        Initialize the screen for the experiment. This will call pgl.open() and
        set parameters according to what is set in setParameters
        '''
        # get  settings
        if settings is None:
            settings = self.getSettings(settingsName)
        
        if settings is None:
            # display error in HTML
            display(HTML("<b>(pglExperiment:initScreen)</b> ❌ Could not find settings to open screen."))
            return
        
        # open screen
        if settings.displayNumber == 0:
            self.pgl.open(0, settings.windowWidth, settings.windowHeight)        
        else:
            self.pgl.open(settings.displayNumber-1)        
            
        self.pgl.visualAngle(settings.displayDistance,settings.displayWidth,settings.displayHeight)
        
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


        # wait half a second for metal app to initialize
        self.pgl.waitSecs(0.5)
        
        self.pgl.flush()
        self.pgl.flush()

    def endScreen(self):
        '''
        Close the screen
        '''

        # close screen
        self.pgl.close()
    def loadParameters(self):
        '''
        Load experiment parameters from configuration file.
        '''
        # this should be settable in a parameter dialog
        self.startKeypress = ["space"]
        self.endKeypress = ["escape"]
        self.keyList = ["1", "2", "3"]

        pass

    def saveParameters(self):
        '''
        Save experiment parameters to configuration file.
        '''
        pass

    def setParameters(self):
        '''
        Brings up a dialog to set experiment parameters.
        '''
        pass
    
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

        # wait for key press to start experiment
        if self.startKeypress is not []:
            experimentStarted = False
            print(f"(pglExperiment:run) Waiting for key press ({[key for key in self.startKeypress]}) to start experiment...")
            while not experimentStarted:
                # poll for events
                events = self.pgl.poll()

                # see if we have a match to startKeypress
                if [e for e in events if e.type == "keyboard" and e.keyChar in self.startKeypress]:
                    experimentStarted = True
                    
                if [e for e in events if e.type == "keyboard" and e.keyChar in self.endKeypress]:
                    experimentStarted = True
                    experimentDone = True

        self.startPhase()
        print(f"(pglExperiment:run) Experiment started.")

        while not experimentDone:
            
            # poll for events
            events = self.pgl.poll()

            # see if we have a match to endKeypress
            if [e for e in events if e.type == "keyboard" and e.keyChar in self.endKeypress]:
                experimentDone = True
                continue

            # grab any events that match the keyList and return their index within that list
            subjectResponse = [keyIndex for e in events if e.type == "keyboard" and e.keyChar in self.keyList for keyIndex in [self.keyList.index(e.keyChar)]]

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
        
        print("(pglExperiment:run) Experiment done.")
        
        # stop the keyboard listener
        keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
        if keyboardDevices is not []:
            for keyboardDevice in keyboardDevices:
                keyboardDevice.stop()

        # close screen
        if not self.suppressEndScreen: self.endScreen()

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
    def updateScreen(self):
        # put upt the bulls eye
        self.pgl.bullseye()
        # and text for what trial we are on 
        # This will just update every trial
        self.pgl.text(f"Trial {self.currentTrial+1}")
    
    def handleSubjectResponse(self, responses, updateTime):
        for response in responses:
            self.pgl.text(f"Subject response received: {response} at {updateTime}")

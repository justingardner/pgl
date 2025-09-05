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
from . import pglKeyboard
    
#############
# Experiment class
#############
class pglExperiment:
    '''
    Experiment class which handles timing, parameter randomization,
    subject response, synchronizing with measurement hardware etc
    '''
    def __init__(self, pgl, suppressInitScreen=False, suppressEndScreen=False):
        # save pgl
        self.pgl = pgl
        
        # initialize screen
        if not suppressInitScreen: self.initScreen()

        # load parameters
        self.loadParameters()
        self.suppressEndScreen = suppressEndScreen
        
        # current phase of experiment
        self.currentPhase = 0
        
        # initialize tasks
        self.task = [[]]
        
    def __repr__(self):
        return f"<pglExperiment: {len(self.task)} phases>"

    def initScreen(self):
        '''
        Initialize the screen for the experiment. This will call pgl.open() and
        set parameters according to what is set in setParameters
        '''
        
        # open screen
        #self.pgl.open(1,800,600)
        self.pgl.open()
        self.pgl.visualAngle(57,40,30)
        
        # add keyboard device
        keyboardDevices = self.pgl.devicesGet(pglKeyboard)
        if not keyboardDevices:
            self.pgl.devicesAdd(pglKeyboard(eatKeys=True))
        else:
            for keyboardDevice in keyboardDevices:
                keyboardDevice.start(eatKeys=True)

        # wait half a second for metal app to initialize
        self.pgl.waitSecs(0.5)

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
        self.startKeypress = ["Key.space"]
        self.endKeypress = ["Key.esc"]
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

        # add the task
        self.task[phaseNum].append(task)

    def run(self):
        '''
        Run the experiment.
        '''
        self.startPhase()

        experimentDone = False
        d = self.pgl.devicesGet(pglKeyboard)
        print(d)

        # wait for key press to start experiment
        if self.startKeypress is not []:
            experimentStarted = False
            print(f"(pglExperiment:run) Waiting for key press ({[key for key in self.startKeypress]}) to start experiment...")
            while not experimentStarted:
                # poll for events
                events = self.pgl.poll()

                # see if we have a match to startKeypress
                if [e for e in events if e.type == "keyboard" and e.keyStr in self.startKeypress]:
                    experimentStarted = True
                    
                if [e for e in events if e.type == "keyboard" and e.keyStr in self.endKeypress]:
                    experimentStarted = True
                    experimentDone = True


        print(f"(pglExperiment:run) Experiment started.")

        while not experimentDone:
            
            # poll for events
            events = self.pgl.poll()

            # see if we have a match to endKeypress
            if [e for e in events if e.type == "keyboard" and e.keyStr in self.endKeypress]:
                experimentDone = True
                continue

            # grab any events that match the keyList and return their index within that list
            subjectResponse = [keyIndex for e in events if e.type == "keyboard" and e.keyStr in self.keyList for keyIndex in [self.keyList.index(e.keyStr)]]

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
        keyboardDevices = self.pgl.devicesGet(pglKeyboard)
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
    '''
    Class representing a task in the experiment. For example, a fixation task. Or
    a stimulus task which controls when and what stimuli are presented
    '''
    def __init__(self):
        self._seglen = []
        self.nTrials = 10
        self.parameters=[]
        self.name="Task"
    ################################################################
    # seglen property
    ################################################################
    @property
    def seglen(self):
        # Get the current segment length.
        return self._seglen
    @seglen.setter
    def seglen(self, seglen_):
        # seglen setting set sboth segmin/segmax
        self._segmin = seglen_
        self._segmax = seglen_
        self._seglen = seglen_
        self.nSegments = len(seglen_)

    def start(self, startTime):
        '''
        Start the task.
        '''
        # set clocks
        self.taskStartTime = startTime
        self.trialStartTime = startTime
        self.segmentStartTime = startTime
        
        # reset blockNum, 
        self.blockNum = -1
        # set blockLen just to trigger startBlock in startTrial
        self.blockLen = 1

        # start trial
        self.currentTrial = -1
        self.startTrial(startTime)

    def startBlock(self, startTime):
        '''
        Start a block.
        '''
        # get randomization of parameters
        (self.parameterNames, self.parameterBlock) = self.getParameterBlock()
        
        # set variables
        self.blockNum += 1
        self.blockStartTime = startTime
        self.blockLen = len(self.parameterBlock)
        
        # display block information
        print(f"({self.name}) Block {self.blockNum+1}: {self.blockLen} trials randomized over: {self.parameterNames}")


    def startTrial(self, startTime):
        '''
        Start a trial.
        '''
        # check if we are at the end of a block
        if (self.currentTrial%self.blockLen) == self.blockLen-1:
            self.startBlock(startTime)

        # update values
        self.currentTrial += 1
        self.trialStartTime = startTime
        self.currentSegment = 0

        # make a dictionary of the current parameters. 
        self.currentParams = dict(zip(self.parameterNames, self.parameterBlock[self.currentTrial%self.blockLen]))

        # print trial
        print(f"({self.name}) Trial {self.currentTrial+1}: ", end='')
        # and variable settings
        for name,value in self.currentParams.items():
            print(f'{name}={value}', end=' ')
        print()

    def getParameterBlock(self):
        '''
        Get a set of parameters to run over, for example if you have
        direction and coherence parameters, this would calculate all
        combination of those parameters and return them for the task
        to run as a block of trials.
        '''
        # get variable names
        paramNames = [p.name for p in self.parameters if p.randomizationBlock == 1]
        # get all valid values from all parameters where randomizationBlock == 1
        allValidValues = [p.validValues for p in self.parameters if p.randomizationBlock == 1]
        # get cartesian combination
        parameterBlock = itertools.product(*allValidValues)
        # and randomly shuffle the order
        parameterBlock = list(parameterBlock)
        random.shuffle(parameterBlock)
        # return the block
        return (paramNames, parameterBlock)

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
        if updateTime - self.segmentStartTime >= self._segmax[self.currentSegment]:
            # reset segment clock
            self.segmentStartTime = updateTime
            # update current segment
            self.currentSegment += 1
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

#############
# Parameter class
#############
class pglParameter:
    '''
    Class representing a parameter in the experiment.
    '''
    def __init__(self, name: str, validValues: list|tuple|np.ndarray, randomizationBlock: int=1, helpStr: str=""):
        '''
        Initialize the parameter.
        
        Args:
            name (str): The name of the parameter.
            validValues (list, optional): A list of valid values for the parameter. If None, any value is valid.
            randomizationBlock (int, optional): Sets the block of parameters which get randomized together, so
              if this is 1, then all other parameters with the same randomizationBlock will be randomized
              together, meaning that all combinations of these parameters will be generated a block randomization
              will randomize over all combinations. Typically this is 1 for variables that you want to randomize
              over and 0 for values that you do not want to randomize over
            helpStr (str, optional): Help string describing the parameter.
        '''
        # validate name
        if not isinstance(name, str):
            raise TypeError("(pglParameter) ❌ Error: Parameter name must be a string.")
        self.name = name
        
        # check if it is a list of values
        if not isinstance(validValues, (list, tuple, np.ndarray)):
            raise TypeError("(pglParameter) ❌ Error: validValues must be a list, tuple, or ndarray.")
        self.validValues = validValues

        # validate randomizationBlock
        if not isinstance(randomizationBlock, int):
            raise TypeError("(pglParameter) ❌ Error: randomizationBlock must be an int.")
            return
        self.randomizationBlock = randomizationBlock

        # validate helpStr
        if not isinstance(helpStr, str):
            raise TypeError("(pglParameter) ❌ Error: helpStr must be a string.")
        self.helpStr = helpStr

    def __repr__(self):
        return f"pglParameter(name={self.name}, validValues={self.validValues}, randomizationBlock={self.randomizationBlock}, helpStr={self.helpStr})"

    def __str__(self):
        # display help string
        if self.helpStr == "":
            helpStr = ""
        else:
            helpStr = f"# {self.helpStr} #\n"
        
        # display full string
        return f"{helpStr}{self.name}: {self.validValues} (randomizationBlock={self.randomizationBlock})"

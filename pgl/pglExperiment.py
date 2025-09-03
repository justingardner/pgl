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
    
#############
# Experiment class
#############
class pglExperiment:
    '''
    Experiment class which handles timing, parameter randomization,
    subject response, synchronizing with measurement hardware etc
    '''
    def __init__(self, pgl, suppressInitScreen=False):
        # save pgl
        self.pgl = pgl
        
        # initialize screen
        if not suppressInitScreen: self.initScreen()

        # load parameters
        self.loadParameters()
        
        # current phase of experiment
        self.currentPhase = 0
        
        # tasks
        self.task = [[]]
        
    def __repr__(self):
        return f"<pglExperiment: {len(self.task)} phases>"

    def initScreen(self):
        '''
        Initialize the screen for the experiment. This will call pgl.open() and
        set parameters according to what is set in setParameters
        '''
        
        # open screen
        self.pgl.open(1,800,600)
        self.pgl.visualAngle(57,40,30)

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
        while not experimentDone:
            
            # poll for events
            events = []#self.pgl.devicesPoll()

            # update tasks in current phase
            phaseDone = False
            updateTime = self.pgl.getSecs()
            for task in self.task[self.currentPhase]:
                # update task
                task.update(updateTime=updateTime, events=events, phaseNum=self.currentPhase, tasks=self.task[self.currentPhase], experiment=self)
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
        self.currentSegment = 0
        self.currentTrial = 0  
        self.nTrials = 10
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
        self.currentSegment = 0
        self.currentTrial = 0
        # set clocks
        self.taskStartTime = startTime
        self.trialStartTime = startTime
        self.segmentStartTime = startTime
        print("Start")

    def update(self, updateTime, events, phaseNum, tasks, experiment):
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
                # start at first segment
                self.currentSegment = 0
                self.currentTrial += 1
                self.trialStartTime = updateTime
                print(f"(pglTask:update) Trial {self.currentTrial} started.")

    def updateScreen(self):
        '''
        Update the screen.
        '''
        print(".")
        pass
    
    def done(self):
        '''
        Check if the task is done.
        '''
        return self.currentTrial >= self.nTrials

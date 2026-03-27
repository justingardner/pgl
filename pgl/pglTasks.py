################################################################
#   filename: pglTasks.py
#    purpose: Some pre-defined tasks for use in PGL experiments
#         by: JLG
#       date: March 26, 2026
################################################################

#############
# Import modules
#############
from .pglExperiment import pglTask
from .pglStaircase import pglStaircaseUpDown
import numpy as np
from .pglParameter import pglParameter

#############
# Fuxaton task: 2AFC on which arm of the fixation cross dims
#############
class pglFixationTaskLeftRight(pglTask):    
    ########################
    def __init__(self, pgl, demo=False):
        super().__init__()
        
        # for demo make fixSie large, and slow dowin timing
        if demo:
            fixSize = 10.0
            slowDownFactor = 5
        else:
            fixSize = 1.0
            slowDownFactor = 1
        
        # task name
        self.settings.taskName = "Fixation Task Left Right"
        
        # keep fixed parameters in settings (so they get saved)
        self.settings.fixedParameters = {
            'fixSize':fixSize,
            'slowDownFactor':slowDownFactor
        }
        self.fixSize = self.settings.fixedParameters['fixSize']
        
        # segments. Fixation, Stimulus, Response
        self.settings.seglen = (slowDownFactor * np.array([0.5, 0.2, 3.0])).tolist()
        
        # add a parameter for which side of the fixation cross dims
        self.addParameter(pglParameter('side',(-1,1)))
        
        # initialize stairase
        self.staircase = pglStaircaseUpDown()
        self.staircase.startStaircase()
    
    ########################
    def startSegment(self, startTime):
        '''
        Start a segment.
        '''
        super().startSegment(startTime)
        # display stimulus only during first segment
        if self.state.currentSegment==0:
            # reset response
            self.gotResponse = False
            # fixation cross starts out white
            self.horizontalColor = self.verticalColor = 1.0
            # get the current decrement value from the staircase
            self.decrement = self.staircase.get()
        elif self.state.currentSegment==1:
            # during the stimulus phase, the left and right sides are different colored
            self.leftColor = self.rightColor = 1.0
            # dim left or right side of fixation cross
            if self.currentParams['side']==-1:
                self.leftColor = 1-self.decrement
            else:
                self.rightColor = 1-self.decrement
        elif self.state.currentSegment==2:
            # during response phase, make vertical change color
            self.verticalColor = [0.0, 1.0, 1.0]

    ########################
    def updateScreen(self):
        
        # draw disc blocking stimulus
        self.pgl.arc(0, 0, 0, self.fixSize/2, 0, 2*np.pi, color=self.pgl.clearScreenColor)
        if self.state.currentSegment==1:
            # draw the left side
            self.pgl.line(-self.fixSize/2, 0, 0, 0, self.leftColor)
            # draw the right side
            self.pgl.line(0, 0, self.fixSize/2, 0, self.rightColor)
        else:
            # just draw the horizotnal line
            self.pgl.line(-self.fixSize/2, 0, self.fixSize/2, 0, self.horizontalColor)

        # draw the vertical line
        self.pgl.line(0, -self.fixSize/2, 0, self.fixSize/2, self.verticalColor)    
        
    ########################
    def handleSubjectResponse(self, response, updateTime):
        # already received a response
        if self.gotResponse: return None
        # mark that we got a response
        self.gotResponse = True
        # default to incorrect
        self.leftColor = self.rightColor = self.horizontalColor = self.verticalColor = [1.0, 0.0, 0.0]
        correct = False
        # check if response is correct
        if ((response==0 and self.currentParams['side']==-1) or
            (response==1 and self.currentParams['side']==1)):
            correct = True 
            self.leftColor = self.rightColor = self.horizontalColor = self.verticalColor = [0.0, 1.0, 0.0]
        # update staircase
        self.staircase.update(self.decrement, correct)
        print(f"(fixationTaskLeftRight) Decrement {self.decrement}: {'correct' if correct else 'incorrect'}")
        # return response type
        return correct

# todo: If subject does not respond within time limit, treat as incorrect response
# todo: Only allow one response per trial
# todo: Restart at previous threshold


# Set up bar task
class pglBarTask(pglTask):
    
    ########################
    def __init__(self, pgl, volumePeriod=1.0, barSweepPeriod=12.0,sweepWidth=None,sweepHeight=None):
        super().__init__()
        
        # set task parameters, these will automatically be saved in the settings file
        self.settings.taskName = "Bar Mapping Task"
        
        # make barSweepPeriod a multiple of volumePeriod
        nVolumesPerSweep = round(barSweepPeriod/volumePeriod)
        barSweepPeriod = nVolumesPerSweep * volumePeriod

        # set seglens
        self.settings.seglen = [volumePeriod/2] * (nVolumesPerSweep)
        # ensure we wait for volume trigger at end of last segment
        self.settings.waitUntilVolumeTrigger[:] = [True] * (nVolumesPerSweep)
        # set number of directions
        directions = np.arange(0,360,45)
        nDirections = len(directions)
        
        # display how long this is expected to take
        totalTime = nDirections * barSweepPeriod
        totalVolumes = nDirections * nVolumesPerSweep
        print(f"(pglBarTask) Total expected task time: {totalTime//60:.0f} minutes {totalTime%60:.1f} seconds, {totalVolumes} volumes  ({nDirections} directions, {barSweepPeriod:.1f} seconds/sweep, {nVolumesPerSweep} volumes/sweep)")
        
        # fixed parameters, these will automatically be saved in the settings file
        self.settings.fixedParameters = {
            'barWidth':2,
            'directions':np.arange(0,360,45),
            'volumePeriod':volumePeriod,
            'nVolumesPerSweep':nVolumesPerSweep,
            'barSweepPeriod':barSweepPeriod,
            'sweepWidth':sweepWidth if sweepWidth is not None else pgl.screenWidth.deg,
            'sweepHeight':sweepHeight if sweepHeight is not None else pgl.screenHeight.deg
        }        
        p = self.settings.fixedParameters

        # direction of bars
        self.addParameter(pglParameter('directions',p['directions']))
        
        # initalize stimulus
        self.bars = pgl.bar(width=p['barWidth'], nVolumesPerSweep=p['nVolumesPerSweep'], sweepWidth=p['sweepWidth'], sweepHeight=p['sweepHeight'])
    
    ########################
    def updateScreen(self):
        self.bars.display(dir=self.currentParams['directions'], volumeNumber=self.e.state.volumeNumber)
        
    ########################
    def getStimulusFrames(self, pgl, events, settings):
        p = self.settings.fixedParameters
        self.bars = pgl.bar(width=p['barWidth'], nVolumesPerSweep=p['nVolumesPerSweep'], sweepWidth=p['sweepWidth'], sweepHeight=p['sweepHeight'])
        screenWidth = 800
        screenHeight = 600
        
        # initialize volume and trial number
        volumeNumber = 0
        trialNumber = 0
        dir = 0
        
        # combine experiment and task events
        events = sorted(events + self.data.events, key=lambda event: event.timestamp)

        # pre-allocate frames array
        nVols = len([e for e in events if e.type == 'keyboard' and e.eventType == 'keydown' and e.keyChar == "5"])
        frames = np.zeros((nVols, screenHeight, screenWidth, 4))
        print(f"(pglBarTask:getStimulusFrames) Capturing {nVols} frames")

        # open screen for off screen rendering
        pgl.open(0,settings.windowWidth,settings.windowHeight)
        pgl.visualAngle(settings.displayDistance, settings.displayWidth, settings.displayHeight)
        pgl.clearScreen(0.5)
        pgl.frameGrabInit()
        pgl.flush()

        # cycle over events
        for e in events:
            # find volume trigger events (Note, this is the old type, need to fix later
            # for when we have actual pglVolumeTrigger events)
            if e.type == 'keyboard' and e.eventType == 'keydown' and e.keyChar == "5":
                volumeNumber += 1
            # This will need to be fixed to be trial, and e.eventType == 'start'
            if e.type == 'pglEventTrial' and e.boundary== 'start':
                # get the current direction for this trial
                dir = self.data.params[trialNumber].get('directions',0)
                # update trial number
                trialNumber += 1
            # once we are at the first volume, then make an image
            if trialNumber >= 1 and volumeNumber >= 1:
                # draw the bar stimulus
                self.bars.display(dir=dir, volumeNumber=volumeNumber)
                pgl.flush()
                # capture the frame
                frames[volumeNumber-1] = pgl.frameGrab()
                # print what frame we got
                print(f"(pglBarTask:getStimulusFrames) Captured frame for dir={dir} volumeNumber={volumeNumber}/{nVols}")
            

        # close screen        
        pgl.frameGrabEnd()
        pgl.close()
        
        # return frames
        return frames
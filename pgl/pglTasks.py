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
from .pglParameter import pglParameter
from .pglMessages import pglMessages
from .pglEyeTracker import pglEyePositionSample
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, pgl, volumePeriod=1.0, barSweepPeriod=24.0,sweepWidth=None,sweepHeight=None,randomSeed=None):
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
        dirParam = pglParameter('directions',p['directions'], randomSeed=randomSeed)
        self.addParameter(dirParam)
        self.settings.randomSeed = dirParam.settings.randomSeed
        
        # initalize stimulus
        self.bars = pgl.bar(width=p['barWidth'], nVolumesPerSweep=p['nVolumesPerSweep'], sweepWidth=p['sweepWidth'], sweepHeight=p['sweepHeight'])
    
    ########################
    def updateScreen(self):
        self.bars.display(dir=self.currentParams['directions'], volumeNumber=self.e.state.volumeNumber)
        
    ########################
    def getStimulusFrames(self, pgl, events, settings, screenWidth=800, screenHeight=600):
        p = self.settings.fixedParameters
        
        print(f"(pglBarTask:getStimulusFrames) Initializing bar stimulus with width={p['barWidth']}, nVolumesPerSweep={p['nVolumesPerSweep']}, sweepWidth={p['sweepWidth']}, sweepHeight={p['sweepHeight']}")
        self.bars = pgl.bar(width=p['barWidth'], nVolumesPerSweep=p['nVolumesPerSweep'], sweepWidth=p['sweepWidth'], sweepHeight=p['sweepHeight'])
        
        # initialize volume and trial number
        volumeNumber = 0
        trialNumber = 0
        dir = 0
        
        # combine experiment and task events
        events = sorted(events + self.data.events, key=lambda event: event.timestamp)

        # pre-allocate frames array
        nVols = len([e for e in events if e.type == 'volumeTrigger'])
        frames = np.zeros((nVols, screenHeight, screenWidth, 4))
        print(f"(pglBarTask:getStimulusFrames) Capturing {nVols} frames")

        # compute x and y coordinates
        y, x = np.indices((screenHeight, screenWidth))
        xDeg = np.degrees(2*np.arctan(((x - screenWidth/2)*(settings.displayWidth/screenWidth))/(2*settings.displayDistance)))
        yDeg = np.degrees(2*np.arctan(((y - screenHeight/2)*(settings.displayHeight/screenHeight))/(2*settings.displayDistance)))        
        
        # compute time in seconds
        timeStamps = [e.timestamp for e in events if e.type == 'volumeTrigger']
        timeStamps = [t - timeStamps[0] for t in timeStamps]
        
        # open screen for off screen rendering
        pgl.open(0,screenWidth,screenHeight)
        pgl.visualAngle(settings.displayDistance, settings.displayWidth, settings.displayHeight)
        pgl.clearScreen(0.5)
        pgl.frameGrabInit()
        pgl.flush()

        # cycle over events
        for e in events:
            # find volume trigger events
            if e.type == 'volumeTrigger':
                volumeNumber += 1
            # Find trial start events
            if e.type == 'trial' and e.eventType == 'start':
                #if (trialNumber == 0): startVolumeNumber = volumeNumber
                # get the current direction for this trial
                dir = self.data.params[trialNumber].get('directions',0)
                # update trial number
                trialNumber += 1
            # for each segment start, draw the bar stimulus
            if e.type == 'segment' and e.eventType == 'start':
                # draw the bar stimulus
                self.bars.display(dir=dir, volumeNumber=volumeNumber)
                pgl.flush()
                # capture the frame
                frames[volumeNumber-1] = pgl.frameGrab()
                # print what frame we got
                print(f"(pglBarTask:getStimulusFrames) Captured frame for dir={dir} segmentNum = {e.segmentNum} volumeNumber={volumeNumber}/{nVols} phase: {volumeNumber-self.bars.volumeNumber})")
            
        # close screen        
        pgl.frameGrabEnd()
        pgl.close()
        
        # return frames
        return frames, xDeg, yDeg, timeStamps
    
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
            # add some more info about experiemnt
            if self.e.state.display.luminanceCalibration[0] == "None":
                self.pgl.text(f"No luminance calibration", line=-1, xAlign=-1)
            elif self.e.settings.calibrateForGamma[0] == 0:
                self.pgl.text(f"Using default gamma table", line=-1, xAlign=-1)
            else:
                self.pgl.text(f"gamma: {self.e.settings.calibrateForGamma[0]} using calibration: {self.e.state.display.luminanceCalibration[0]}", line=-1, xAlign=-1)
    
    def handleSubjectResponse(self, response, updateTime):
        self.responseText = f"Subject response received: {response} at {updateTime - self.e.data.startTime:.2f} seconds"


##############################################
# Eye tracking calibration task
##############################################
class pglEyeTrackingCalibrationTask(pglTask):
    
    ########################
    def __init__(self, pgl, nCalibrationPoints=17, calibrationWidth=10, calibrationHeight=10, nRepeats=1):
        super().__init__(pgl)

        # task name
        self.settings.taskName = "Eye Tracking Calibration"
        
        # calibration settings
        self.settings.config.calibrationPoints = self.makeCalibrationPoints(nCalibrationPoints=nCalibrationPoints, calibrationWidth=calibrationWidth, calibrationHeight=calibrationHeight)
        if self.settings.config.calibrationPoints is None: return
        self.settings.config.nCalibrationPoints = nCalibrationPoints
        self.settings.config.calibrationWidth = calibrationWidth
        self.settings.config.calibrationHeight = calibrationHeight

        # set parameters for stable fixation check. Duration is the time (in seconds) that
        # stable fixation must be held for, Fixation tolerance is how far one sample
        # can be (max) from the next sample to be considered stable fixation. If greater
        # than this amount, then the duration interval is reset to 0. Samples are taken
        # once every screen refresh
        self.settings.config.stableDuration= 0.5
        self.settings.config.stableFixationTolerance = 0.5
        
        # number of repeats and trials
        self.settings.nRepeats = nRepeats
        self.settings.nTrials = nRepeats * nCalibrationPoints
        
        # First segment is to give subject time to acquire targer
        # Second segment is over when stable fixation is detected
        self.settings.seglen = [0.2, np.inf]

        # add parameters for calibration points
        calibrationPoints = pglParameter('calibrationPoint',self.settings.config.calibrationPoints)        
        self.addParameter(calibrationPoints)
        
    ########################
    # startSegment
    ########################
    def startSegment(self, startTime):
        '''
        Start a segment.
        '''        
        if self.state.currentSegment == 1:
            # initialize stable fixation window
            self.state.stableFixationStart = startTime
            self.state.stableFixationSamples = []
            

    ########################
    # updateScreen
    ########################
    def updateScreen(self):
        '''
        draw calibration points
        '''
        calibrationPoint = self.currentParams['calibrationPoint']
        self.pgl.fixationABC(x=calibrationPoint[0], y=calibrationPoint[1])
        
        # if we are in the segment for testing stable fixation
        if self.state.currentSegment == 1:
            # get next sample
            if self.e.eyeTracker is None:
                # no eye tracker initialized, just wait 1 s
                if self.pgl.getSecs() - self.startSegment > (1 - self.seglen[0]):
                    self.jumpSegment()
                    pglMessages.message("eye tracker is None")
                    return
            else:
                # get eye position
                eyePosition = self.e.eyeTracker.getEyePosition().get('either',None)
                print(f'{eyePosition},',end='')
                if eyePosition is None:
                    # must be failed eye tracker, reset stableFixation
                    self.state.stableFixationSamples = []
                else:
                    # check if this is the first sample
                    if len(self.state.stableFixationSamples) == 0:
                        # keep the time
                        self.state.stableFixationStart = self.pgl.getSecs()
                    # check distance of last fixation, note that - is overloaded for euclidean distance
                    elif eyePosition - self.state.stableFixationSamples[-1] > self.settings.config.stableFixationTolerance:
                        # sample is too far away from last fixation, so must have broken fixation, restart
                        self.state.stableFixationSamples = []
                        self.state.stableFixationStart = self.pgl.getSecs()
                    
                    # save sample
                    self.state.stableFixationSamples.append(eyePosition)

                    # passed fixation tolerance test, so check if we now have passed duration 
                    if self.pgl.getSecs() - self.state.stableFixationStart > self.settings.config.stableDuration:
                        # passed stable fixation test, so we now are done,
                        self.data.trialVariables[-1]['eyePosition'] = pglEyePositionSample.median(self.state.stableFixationSamples)
                        pglMessages.message(f"stable fixation: {self.data.trialVariables[-1]['eyePosition']}")
                        #  jump to next segment will move to next fixation point
                        self.jumpSegment()
                        return
                                        
    ########################
    # handle events
    ########################
    def handleEvents(self, events):
        for event in events:
            if event.eventType == 'keydown' and event.keyChar == 'space':
                # jump out of segment (can be used for aborting stable fixation check)
                pglMessages.message(f"stable fixation: {pglEyePositionSample.median(self.data.trialVariables[-1]['eyePosition'])}")
                self.jumpSegment()

    ########################
    # display
    ########################
    def display(self, ax=None):
        '''
        Display calibration targets and measured eye positions.

        Calibration targets are shown as solid circles. Eye positions measured
        for trials are shown as '+' markers in the color corresponding to the
        calibration target presented on that trial.
        '''
        # Create an axes if the caller did not provide one.
        if ax is None:
            figure, ax = plt.subplots()

        calibrationPoints = self.settings.config.calibrationPoints

        if not calibrationPoints:
            pglMessages.warning("No calibration points available.")
            return ax

        # Get colormap for each of the calibration points, and make a dict to
        # retrieve the right color for each calibration point
        colorMap = plt.get_cmap('tab20')
        pointColors = {
            tuple(calibrationPoint): colorMap(pointIndex)
            for pointIndex, calibrationPoint in enumerate(calibrationPoints)
        }

        # Draw every calibration target once as a filled circle.
        for pointIndex, calibrationPoint in enumerate(calibrationPoints):
            x, y = calibrationPoint
            color = pointColors[tuple(calibrationPoint)]

            ax.scatter(x,y,s=100,marker='o',color=color,edgecolors='white',linewidths=0.75,zorder=2,label=f'Calibration point {pointIndex + 1}')

        # self.data.params and self.data.trialVariables have one entry
        # per trial, in matching order.
        for trialIndex, (trialParams, trialVariables) in enumerate(
            zip(self.data.params, self.data.trialVariables)
        ):
            calibrationPoint = trialParams.get('calibrationPoint', None)

            if calibrationPoint is None: continue

            eyePosition = trialVariables.get('eyePosition', None)

            # A trial can have no eyePosition, e.g. no tracker or aborted trial.
            if eyePosition is None: continue

            calibrationPoint = tuple(calibrationPoint)

            # Plot measured position as a plus sign matching its target's color.
            ax.scatter(eyePosition.x,eyePosition.y,s=150,marker='+',color=pointColors[calibrationPoint],linewidths=2.5,zorder=3)

        ax.axhline(0, color='0.8', linewidth=0.75, zorder=0)
        ax.axvline(0, color='0.8', linewidth=0.75, zorder=0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Horizontal position (deg)')
        ax.set_ylabel('Vertical position (deg)')
        ax.set_title(self.settings.taskName)
        ax.grid(True, alpha=0.25)

        return ax
    
    ########################
    # make calibration points
    ########################
    def makeCalibrationPoints(self, nCalibrationPoints, calibrationWidth, calibrationHeight):
        """
        Validate a requested calibration-point count and return calibration
        target positions centered at (0, 0).

        Parameters
        ----------
        nCalibrationPoints : int
            Must be one of: 5, 9, 13, or 17.

        calibrationWidth : float
            Total horizontal width covered by calibration targets.
            Targets range from -calibrationWidth / 2 to +calibrationWidth / 2.

        calibrationHeight : float
            Total vertical height covered by calibration targets.
            Targets range from -calibrationHeight / 2 to +calibrationHeight / 2.

        Returns
        -------
        list[tuple[float, float]]
            List of (x, y) calibration-target coordinates.
        """

        validCalibrationPointCounts = (5, 9, 13, 17)

        if nCalibrationPoints not in validCalibrationPointCounts:
            pglMessages.warning(f"nCalibrationPoints must be one of {validCalibrationPointCounts}; got {nCalibrationPoints}")
            return None

        if not isinstance(calibrationWidth, (int, float)):
            pglMessages.warning(f"calibrationWidth must be a finite numeric value; got {calibrationWidth!r}")
            return None

        if not isinstance(calibrationHeight, (int, float)):
            pglMessages.warning(f"calibrationHeight must be a finite numeric value; got {calibrationHeight!r}")
            return None

        if calibrationWidth <= 0:
            pglMessages.warning(f"calibrationWidth must be greater than zero; got {calibrationWidth}")
            return None

        if calibrationHeight <= 0:
            pglMessages.warning(f"calibrationHeight must be greater than zero; got {calibrationHeight}")
            return None

        halfWidth = calibrationWidth / 2.0
        halfHeight = calibrationHeight / 2.0

        innerWidth = calibrationWidth / 4.0
        innerHeight = calibrationHeight / 4.0

        centerPoint = (0.0, 0.0)

        # Four outer corners.
        outerCornerPoints = [
            (-halfWidth, -halfHeight),
            ( halfWidth, -halfHeight),
            ( halfWidth,  halfHeight),
            (-halfWidth,  halfHeight),
        ]

        # Four midpoint targets on the outer rectangle.
        outerEdgePoints = [
            (0.0,       -halfHeight),  # top
            (halfWidth,  0.0),         # right
            (0.0,        halfHeight),  # bottom
            (-halfWidth, 0.0),         # left
        ]

        # Four corners of an inner rectangle.
        innerCornerPoints = [
            (-innerWidth, -innerHeight),
            ( innerWidth, -innerHeight),
            ( innerWidth,  innerHeight),
            (-innerWidth,  innerHeight),
        ]

        # Four midpoint targets on the inner rectangle.
        innerEdgePoints = [
            (0.0,        -innerHeight),
            (innerWidth,  0.0),
            (0.0,         innerHeight),
            (-innerWidth, 0.0),
        ]

        if nCalibrationPoints == 5:
            # Center + four corners.
            calibrationPoints = [
                centerPoint,
                *outerCornerPoints,
            ]

        elif nCalibrationPoints == 9:
            # Standard 3 x 3 layout:
            # center + four corners + top/right/bottom/left edge midpoints.
            calibrationPoints = [
                centerPoint,
                *outerCornerPoints,
                *outerEdgePoints,
            ]

        elif nCalibrationPoints == 13:
            # 9-point outer layout plus four inner-diagonal points.
            calibrationPoints = [
                centerPoint,
                *outerCornerPoints,
                *outerEdgePoints,
                *innerCornerPoints,
            ]

        else:  # nCalibrationPoints == 17
            # Outer 8-point ring + inner 8-point ring + center.
            calibrationPoints = [
                centerPoint,
                *outerCornerPoints,
                *outerEdgePoints,
                *innerCornerPoints,
                *innerEdgePoints,
            ]

        assert len(calibrationPoints) == nCalibrationPoints

        return calibrationPoints

##############################################
# display a message and prompt for subject to hit key
##############################################
class pglMessageAckTask(pglTask):
    
    ########################
    def __init__(self, pgl, message="Hit any button to continue", ackKey='space'):
        super().__init__(pgl)

        # task name
        self.settings.taskName = "Acknowledge Message"

        # set seglens
        self.settings.seglen = [np.inf]

        # fixed parameters, these will automatically be saved in the settings file
        self.settings.fixedParameters = {
            'message': message,
            'ackKey': ackKey,
        }        
        self.settings.nTrials=1

        #self.state.ackKeyCode = self.pgl.devicesGetKeyboard().charToKeyCode(self.settings.calibrateKey)

        # eat only asked for key
        #k = self.pgl.devicesGetKeyboard()            
        #self.state.eatKeys = k.eatKeyCodes


    ########################
    # updateScren
    ########################
    def updateScreen(self):
        '''
        draw text
        '''
        self.pgl.text(self.settings.fixedParameters['message'],y=0)

    ########################
    # handleSubjectResponse
    ########################    
    def handleSubjectResponse(self, response, updateTime):
        self.jumpSegment()


    ########################
    # handleEvents
    ########################
    def handleEvents(self, events):
        '''
        '''
        #if [e for e in events if e.type == "keyboard" and e.eventType == "keydown"and e.keyCode == self.state.ackKeyCode]: 
        #    self.jumpSegment()
        pass


        

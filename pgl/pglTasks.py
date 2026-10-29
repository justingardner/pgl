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
# Fixation task: 2AFC on which arm of the fixation cross dims
#############
class pglFixationTaskLeftRight(pglTask):    
    ########################
    def __init__(self, pgl, demo=False):
        super().__init__(pgl)
        
        # for demo make fixSie large, and slow dowin timing
        if demo:
            fixSize = 10.0
            slowDownFactor = 5
        else:
            fixSize = 10.0
            slowDownFactor = 1
        
        # task name
        self.settings.taskName = "Fixation Task Left Right"
        
        # keep fixed parameters in settings (so they get saved)
        self.settings.config.fixSize = fixSize
        self.settings.config.slowDownFactor = slowDownFactor
        
        # segments. Fixation, Stimulus, Response
        self.settings.seglen = (slowDownFactor * np.array([0.5, 0.2, 3.0])).tolist()
        
        # do not print messages while running
        self.settings.verbose = False
        
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
        self.pgl.arc(0, 0, 0, self.settings.config.fixSize / 2, 0, 2 * np.pi, color=self.e.settings.backgroundColor)
        self.pgl.arc(0, 0, 0, self.settings.config.fixSize / 2, 0, 2 * np.pi, color=0.5)
        #if self.state.currentSegment==1:
            # draw the left side
            #self.pgl.line(-self.settings.config.fixSize/2, 0, 0, 0, self.leftColor)
            # draw the right side
            #self.pgl.line(0, 0, self.settings.config.fixSize/2, 0, self.rightColor)
        #else:
            # just draw the horizotnal line
            #self.pgl.line(-self.settings.config.fixSize/2, 0, self.settings.config.fixSize/2, 0, self.horizontalColor)

        # draw the vertical line
        #self.pgl.line(0, -self.settings.config.fixSize/2, 0, self.settings.config.fixSize/2, self.verticalColor)    
        
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
    def __init__(self, pgl, volumePeriod=1.0, barSweepPeriod=24.0, sweepWidth=None, sweepHeight=None, randomSeed=None):
        super().__init__(pgl)

        self.settings.taskName = "Bar Mapping Task"

        # Validate timing independently of the screen, so runDuration is
        # available before initScreen() for recording-capacity estimates.
        volumePeriod = float(volumePeriod)
        barSweepPeriod = float(barSweepPeriod)
        if not np.isfinite(volumePeriod) or volumePeriod <= 0: raise ValueError("volumePeriod must be finite and greater than zero.")
        if not np.isfinite(barSweepPeriod) or barSweepPeriod <= 0: raise ValueError("barSweepPeriod must be finite and greater than zero.")

        # Require at least two volumes so a half-length catch is nonempty.
        nVolumesPerSweep = round(barSweepPeriod / volumePeriod)
        if nVolumesPerSweep < 2: raise ValueError("barSweepPeriod must round to at least two scanner volumes.")
        barSweepPeriod = nVolumesPerSweep * volumePeriod

        # set as main task for detecting aborted runs
        self.settings.isMainTask = True
        
        # Every segment runs for half a nominal volume period, then waits
        # for the next scanner trigger.
        self.settings.seglen = [volumePeriod / 2] * nVolumesPerSweep
        self.settings.waitUntilVolumeTrigger = [True] * nVolumesPerSweep

        # Store stimulus settings, but do not read screen dimensions yet.
        # None means configure() will use the initialized screen dimensions.
        config = self.settings.config
        config.barWidth = 2
        config.directions = np.arange(0, 360, 45)
        config.volumePeriod = volumePeriod
        config.nVolumesPerSweep = nVolumesPerSweep
        config.barSweepPeriod = barSweepPeriod
        config.sweepWidth = sweepWidth
        config.sweepHeight = sweepHeight

        # Catch duration rounds down for odd sweep lengths.
        config.nVolumesForBlankCatchTrial = nVolumesPerSweep // 2

        # One planned run contains every direction once and one blank catch.
        # These estimates do not change the task's trial limit.
        config.nTrialsPerRun = len(config.directions) + 1
        config.nVolumesPerRun = len(config.directions) * config.nVolumesPerSweep + config.nVolumesForBlankCatchTrial
        config.runDuration = config.nVolumesPerRun * config.volumePeriod

        # Report nominal duration, excluding calibration and pre-start waiting.
        catchDuration = config.nVolumesForBlankCatchTrial * config.volumePeriod
        print(f"(pglBarTask) Planned run: {config.nTrialsPerRun} trials ({len(config.directions)} stimulus sweeps + 1 blank catch), {config.runDuration // 60:.0f} minutes {config.runDuration % 60:.1f} seconds, {config.nVolumesPerRun} volumes.")
        print(f"  Stimulus sweep: {config.barSweepPeriod:.1f} seconds, {config.nVolumesPerSweep} volumes.")
        print(f"  Blank catch: {catchDuration:.1f} seconds, {config.nVolumesForBlankCatchTrial} volumes.")

        # Insert a randomly positioned catch without consuming a direction.
        dirParam = pglParameter('directions', config.directions, randomSeed=randomSeed, catchTrialEvery=config.nTrialsPerRun)
        self.addParameter(dirParam)
        self.settings.randomSeed = dirParam.settings.randomSeed

        # set the number of trials to run for
        self.settings.nTrials = self.settings.config.nTrialsPerRun
        
        # Stimulus creation is deferred until the screen has been initialized.
        self.bars = None
        
    ########################
    def configure(self, e):
        """Resolve screen-dependent dimensions and create the bar stimulus.

        Called by initScreen() after the screen and visual-angle coordinates
        are initialized. Resolved dimensions remain in config for saving and
        subsequent stimulus reconstruction.
        """
        super().configure(e)

        if not self.pgl.isOpen(): raise RuntimeError("The screen must be open before configuring the bar task.")
        if self.pgl.coordinateFrame != "visualAngle": raise RuntimeError("The bar task requires visual-angle coordinates.")

        config = self.settings.config

        # Resolve defaults only now, never from stale or uninitialized screen
        # dimensions during task construction.
        sweepWidth = self.pgl.screenWidth.deg if config.sweepWidth is None else config.sweepWidth
        sweepHeight = self.pgl.screenHeight.deg if config.sweepHeight is None else config.sweepHeight
        sweepWidth = float(sweepWidth)
        sweepHeight = float(sweepHeight)

        # Reject invalid geometry here instead of crashing during acquisition
        # when the checkerboard attempts to index an empty coordinate array.
        if not np.isfinite(sweepWidth) or sweepWidth <= 0: raise ValueError(f"Bar sweep width must be finite and positive; got {sweepWidth}.")
        if not np.isfinite(sweepHeight) or sweepHeight <= 0: raise ValueError(f"Bar sweep height must be finite and positive; got {sweepHeight}.")

        config.sweepWidth = sweepWidth
        config.sweepHeight = sweepHeight

        # Construct the checkerboard only after its dimensions are valid.
        self.bars = self.pgl.bar(width=config.barWidth, nVolumesPerSweep=config.nVolumesPerSweep, sweepWidth=config.sweepWidth, sweepHeight=config.sweepHeight)
        if self.bars is None: raise RuntimeError("Could not create the bar stimulus.")
        
    ########################
    def modifyTrialSegments(self, seglen):
        """Shorten blank catches to the configured number of scanner volumes."""
        if self.currentParams['directions'] is None:
            return seglen[:self.settings.config.nVolumesForBlankCatchTrial]
        return seglen
                
    ########################
    def startTrial(self, startTime):
        ''' print header to make it easier to read trial conditions'''
        self.pgl.printHeader()
        super().startTrial(startTime)
        
    ########################
    def updateScreen(self):
        
        # draw the bars (but if the dir is None then this is a catch trial)
        dir = self.currentParams['directions']
        if dir is not None: self.bars.display(dir=dir, volumeNumber=self.e.state.volumeNumber)
        
    ########################
    def getStimulusFrames(self, pgl, events, settings, screenWidth=800, screenHeight=600):
        """Reconstruct stimulus and blank frames from recorded segment starts.

        Catch duration comes from the recorded events, so shortened catches
        need no special timing calculation here.

        Requires scanner volume triggers. Volumes with no segment start
        remain zero-filled, matching the existing reconstruction behavior.
        """
        config = self.settings.config

        print(
            f"(pglBarTask:getStimulusFrames) Initializing bar stimulus with "
            f"width={config.barWidth}, nVolumesPerSweep={config.nVolumesPerSweep}, "
            f"sweepWidth={config.sweepWidth}, sweepHeight={config.sweepHeight}"
        )

        self.bars = pgl.bar(
            width=config.barWidth,
            nVolumesPerSweep=config.nVolumesPerSweep,
            sweepWidth=config.sweepWidth,
            sweepHeight=config.sweepHeight,
        )

        # Merge scanner events and task boundaries without modifying inputs.
        combinedEvents = sorted(list(events) + list(self.data.events), key=lambda event: event.timestamp)
        volumeTimestamps = [event.timestamp for event in combinedEvents if event.type == 'volumeTrigger']
        nVols = len(volumeTimestamps)
        if nVols == 0: raise ValueError("Cannot reconstruct volume-indexed frames without volume triggers.")

        frames = np.zeros((nVols, screenHeight, screenWidth, 4))
        timeStamps = [timestamp - volumeTimestamps[0] for timestamp in volumeTimestamps]
        print(f"(pglBarTask:getStimulusFrames) Capturing {nVols} frames")

        # Preserve the existing pixel-to-visual-angle coordinate calculation.
        y, x = np.indices((screenHeight, screenWidth))
        xDeg = np.degrees(2 * np.arctan(((x - screenWidth / 2) * (settings.displayWidth / screenWidth)) / (2 * settings.displayDistance)))
        yDeg = np.degrees(2 * np.arctan(((y - screenHeight / 2) * (settings.displayHeight / screenHeight)) / (2 * settings.displayDistance)))

        volumeNumber = 0
        trialNumber = 0
        direction = None
        screenOpened = False
        captureStarted = False

        try:
            # Initialize the standalone reconstruction screen.
            pgl.open(0, screenWidth, screenHeight)
            screenOpened = pgl.isOpen()
            if not screenOpened: raise RuntimeError("Could not open the reconstruction screen.")

            pgl.visualAngle(settings.displayDistance, settings.displayWidth, settings.displayHeight)
            pgl.clearScreen(0.5)
            pgl.frameGrabInit()
            captureStarted = True
            pgl.flush()

            for event in combinedEvents:
                if event.type == 'volumeTrigger': volumeNumber += 1

                if event.type == 'trial' and event.eventType == 'start':
                    if trialNumber >= len(self.data.params):
                        raise ValueError("Recorded trial starts exceed saved parameter entries.")

                    # None explicitly denotes a catch; a missing key is an error.
                    direction = self.data.params[trialNumber]['directions']
                    trialNumber += 1

                if event.type == 'segment' and event.eventType == 'start':
                    if trialNumber == 0: raise ValueError("Segment start occurred before any trial start.")
                    if not 1 <= volumeNumber <= nVols:
                        raise ValueError("Segment start has no valid scanner-volume index.")

                    # Clear explicitly so a catch cannot retain the prior bars.
                    # This is standalone reconstruction, not concurrent tasks.
                    pgl.clearScreen(0.5)
                    if direction is not None: self.bars.display(dir=direction, volumeNumber=volumeNumber)

                    pgl.flush()
                    frames[volumeNumber - 1] = pgl.frameGrab()

                    # Do not access bar sweep state during a blank catch.
                    print(
                        f"(pglBarTask:getStimulusFrames) Captured frame for "
                        f"dir={direction} segmentNum={event.segmentNum} "
                        f"volumeNumber={volumeNumber}/{nVols}"
                    )

        finally:
            # Attempt screen closure even if stopping frame capture fails.
            try:
                if captureStarted: pgl.frameGrabEnd()
            finally:
                if screenOpened: pgl.close()

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
    def __init__(self, pgl):
        super().__init__(pgl)

        # task name
        self.settings.taskName = "Eye Tracking Calibration"

    ########################
    def configure(self, e):
        '''
        Configure the eye tracker settings, this will be called by initScreen once the experiment is setup
        '''
        if e.eyeTrackerSettings is None:
            pglMessages.message("No eye tracker settings found, cannot run eye tracking calibration task")
            self.settings.config.hasEyeTracker = False
            return
        
        # check if eye tracker is initialized
        if e.eyeTracker is None:
            pglMessages.message("No eye tracker initialized, cannot run eye tracking calibration task")
            self.settings.config.hasEyeTracker = False
            return
        
        # has eye tracker
        self.settings.config.hasEyeTracker = True
        
        # compute calibration Width and Height as percentage of screen
        calibrationWidth = e.eyeTrackerSettings.calibrationWidth
        calibrationHeight = e.eyeTrackerSettings.calibrationHeight

        # calibration settings
        self.settings.config.calibrationPoints = self.makeCalibrationPoints(nCalibrationPoints=e.eyeTrackerSettings.nCalibrationPoints, calibrationWidth=calibrationWidth, calibrationHeight=calibrationHeight)
        if self.settings.config.calibrationPoints is None: return
        self.settings.config.nCalibrationPoints = e.eyeTrackerSettings.nCalibrationPoints
        self.settings.config.calibrationWidth = e.eyeTrackerSettings.calibrationWidth
        self.settings.config.calibrationHeight = e.eyeTrackerSettings.calibrationHeight

        # set parameters for stable fixation check. Duration is the time (in seconds) that
        # stable fixation must be held for, Fixation tolerance is how far one sample
        # can be (max) from the next sample to be considered stable fixation. If greater
        # than this amount, then the duration interval is reset to 0. Samples are taken
        # once every screen refresh
        self.settings.config.stableDuration= float(e.eyeTrackerSettings.stableDuration)/1000.0
        self.settings.config.stableFixationTolerance = e.eyeTrackerSettings.stableFixationTolerance
        
        # number of repeats and trials
        self.settings.nRepeats = e.eyeTrackerSettings.nRepeats
        self.settings.nTrials = e.eyeTrackerSettings.nRepeats * e.eyeTrackerSettings.nCalibrationPoints
        
        # First segment is to give subject time to acquire targer
        # Second segment is over when stable fixation is detected
        self.settings.seglen = [0.2, np.inf]

        # add parameters for calibration points
        calibrationPoints = pglParameter('calibrationPoint',self.settings.config.calibrationPoints)        
        self.addParameter(calibrationPoints)
       
    ########################
    # startSegment
    ########################
    def startTask(self, startTime):
        '''
        If no eye tracker is initialized, then return False to abort running this task
        '''
        if self.settings.config.hasEyeTracker == False: 
            return False
        else:
            return True

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
                    pglMessages.message("No eye tracker")
                    self.jumpSegment()
                    return
            else:
                # get eye position
                eyePosition = self.e.eyeTracker.getEyePosition().get('either',None)

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
                        pglMessages.message(f"stable fixation for {calibrationPoint}: {self.data.trialVariables[-1]['eyePosition']}")
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
        if self.settings.config.hasEyeTracker == False: 
            pglMessages.message("No eye tracker initialized, cannot display eye tracking calibration task")
            return
        
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
        self.settings.config.message = message
        self.settings.config.ackKey = ackKey
        self.settings.nTrials=1

    ########################
    def configure(self, e):

        self.state.ackKeyCode = self.pgl.devicesGetKeyboard().charToKeyCode(self.settings.config.ackKey)

    ########################
    # updateScren
    ########################
    def updateScreen(self):
        '''
        draw text
        '''
        self.pgl.text(self.settings.config.message,y=0)

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
        if [e for e in events if e.type == "keyboard" and e.eventType == "keydown"and e.keyCode == self.state.ackKeyCode]: 
            self.jumpSegment()


        

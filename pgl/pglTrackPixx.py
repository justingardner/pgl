################################################################
#   filename: pglTrackPixx.py
#    purpose: Code for working with TrackPixx3 eye tracker
#         by: JLG
#       date: August 1, 2025
################################################################

#############å
# Import modules
#############
from pgl import pglEyeTracker
from pgl import pglDevice
import numpy as np

###################################
# TrackPixx3 device
###################################
class pglTrackPixx3(pglEyeTracker):
    """
    Class for interfacing with the TrackPixx3 eye tracker.
    Inherits from pglEyeTracker.
    """
    def __init__(self, pgl=None, deviceType="TrackPixx3"):
        """
        Initialize the TrackPixx3 device.
        """
        # call superclass constructor
        super().__init__(pgl, deviceType)

        # get library
        try:
            import pypixxlib._libdpx as dp
        except ImportError: 
            print("(pglTrackPixx3) pypixxlib is not installed. Please install it to use TrackPixx3.")
            return
        self.dp = dp

        #-->    # It is mandatory to call 'DPxOpen()' prior to using any VPixx device
        self.dp.DPxOpen()
        if not self.dp.DPxDetectDevice("TRACKPIXX"):
            print("(pglTrackPixx3) TrackPixx3 device not detected.")
            return
        
        # wake the device up
        print("(pglTrackPixx3) Wake up TrackPixx3")

        print(f"(pglTrackPixx3) TPxIsAwakePictureRequest: {self.dp.TPxIsAwakePictureRequest()}")
        self.dp.DPxSetTPxAwake()
        self.dp.DPxWriteRegCache()
        self.dp.TPxSetAwakePictureRequest()
        self.dp.DPxWriteRegCache()
        print(f"(pglTrackPixx3) TPxIsAwakePictureRequest: {self.dp.TPxIsAwakePictureRequest()}")
        

        # Adapted from TPxCalibrationTesting.py example code

        # dp.TPxShowOverlay() will activate the console tracker window to display the camera
        # image and pupil center tracking position.  The window will
        # appear at the top left corner of the monitor connected to the out2
        # display port of the DATAPixx3.
        # This overlay can be hidden with dp.TPxHideOverlay()

        # 'ClearCalibration' will permanently destroy the current calibration.
        #-->    # It is mandatory to clear the previous calibration before starting a
        # new one.
        self.dp.TPxClearDeviceCalibration()

        #-->    # PpSizeCalClear disable pupil size calibration.  it is mandatory to
        # clear pupil size calibration before to start a new one.
        self.dp.TPxPpSizeCalibrationClear()

        # This will awaken the TPx and turn on the illuminator.
        #-->	# It is mandatory to awaken the TPx before using it.
        self.dp.DPxSetTPxAwake()

        #-->    # dp.DPxUpdateRegCache() will apply previous Datapixx calls that were 
        # not applied yet. Do not forget to call this whenever you must apply a
        # setting or send a command to hardware. Note that each call imposes
        # a USB transaction latency, so it may be prudent to batch these settings
        # updates together as done here with self.dp.TPxClearDeviceCalibration(),
        # self.dp.TPxPpSizeCalibrationClear(), and self.dp.DPxSetTPxAwake()
        self.dp.DPxUpdateRegCache()

        # Set the infrared LED array intensity, which must be an
        # integer value between 0 and 8.  At 0, the illuminator is off while
        # at 8 the illuminator is at maximum intensity.
        # The absolute value of the luminous intensity depends on the present hardware.
        # Skipping this command means the previous value will be used.  If the value
        # was never set, the default value will be used.
        # The default value is 8 (maximum intensity)
        #-->    # Using too much or not enough light will prevent good tracking results.
        # Experimenting with different intensity values and evaluating tracking quality
        # is the best way to optimize this parameter.
        self.ledIntensity = self.dp.TPxGetLEDIntensity()
        self.lens = self.dp.TPxGetLens()
        print(f"(pglTrackPixx3) TrackPixx3 initialized with LED intensity {self.ledIntensity} and lens {self.lens*25+25} mm.")
    
    #################################################
    # calibrateEyeImage
    ################################################
    def calibrateEyeImage(self):
        """
        Calibrate the eye image.
        """
        if self.isPGLOpen() is False:
            print("(pglTrackPixx3:calibrateEyeImage) PGL screen is not open. Cannot calibrate eye image.")
            return False

        # Adapted from TPxCalibrationTesting.py example code Step 2
        #self.dp.DPxOpen()
        # Get initial search limits for TPx. Search limits for either eye are
        # specified as 1 x 4 vector with the (x,y) coordinates of top left corner of
        # the search window and (x,y) coordinates of the bottom right corner. If
        # search limits have not been set, the vector is set to [0,0,0,0]
        self.dp.TPxClearSearchLimits()
        self.dp.DPxUpdateRegCache()
        (leftSearchLimit, rightSearchLimit) = self.dp.TPxGetSearchLimits()

        print(f"(pglTrackPixx3) TrackPixx3 calibrating eye image with LED intensity {self.ledIntensity} and lens {self.lens*25+25} mm.")

        # get device time
        lastTime = self.dp.DPxGetTime()
        thisTime = self.dp.DPxGetTime()
        print(f"(pglTrackPixx3:calibrateEyeImage) Initial time: {lastTime} {thisTime}")

        # stay in a loop, drawing camera images and allowing experimenter/subject to adjust parameters
        loopCalibration = True
        while loopCalibration:

            # wait for a duration of 1/60 second.
            if (thisTime - lastTime) > 1/60: 
                # clear screen
                self.pgl.clearScreen((0,0,0))

                # get camera image
                cameraImage = self.getCameraImage()
                if cameraImage is None:
                    print("(pglTrackPixx3:calibrateEyeImage) No camera image available. Cannot display.")
                    loopCalibration = False
                cameraImage.display()

                # Get eye data in camera space
                expectedIrisSize = self.dp.TPxGetIrisExpectedSize()
                (ppLeftMajor, _, ppRightMajor, _) = self.dp.TPxGetPupilSize() 
                (ppLeftX, ppLeftY, ppRightX, ppRightY) = self.dp.TPxGetPupilCoordinatesInPixels() 

                # covert to degrees for display
                ppLeftXDeg = ppLeftX * self.pgl.xPix2Deg + cameraImage.displayLeft
                ppLeftYDeg = -ppLeftY * self.pgl.yPix2Deg + cameraImage.displayTop
                ppRightXDeg = ppRightX * self.pgl.xPix2Deg + cameraImage.displayLeft
                ppRightYDeg = -ppRightY * self.pgl.yPix2Deg + cameraImage.displayTop

                # get center of left and right pupils in degrees
                eyeLeft = (ppLeftXDeg, ppLeftYDeg)
                eyeRight = (ppRightXDeg, ppRightYDeg)

                # draw cross at the pupil center
                if ppLeftMajor > 0:
                    self.pgl.line(eyeLeft[0], eyeLeft[1]+self.pgl.yPix2Deg * ppLeftMajor/2, eyeLeft[0], eyeLeft[1]-self.pgl.yPix2Deg * ppLeftMajor/2, color=[0,1,0])
                    self.pgl.line(eyeLeft[0]-self.pgl.xPix2Deg * ppLeftMajor/2, eyeLeft[1], eyeLeft[0]+self.pgl.xPix2Deg * ppLeftMajor/2, eyeLeft[1], color=[0,1,0])
                else:
                    self.pgl.fixationCross(1,cameraImage.displayLeft, cameraImage.displayTop, color=[1,0,0])
                if ppRightMajor > 0:
                    self.pgl.line(eyeRight[0], eyeRight[1]+self.pgl.yPix2Deg * ppRightMajor/2, eyeRight[0], eyeRight[1]-self.pgl.yPix2Deg * ppRightMajor/2, color=[0,1,1])
                    self.pgl.line(eyeRight[0]-self.pgl.xPix2Deg * ppRightMajor/2, eyeRight[1], eyeRight[0]+self.pgl.xPix2Deg * ppRightMajor/2, eyeRight[1], color=[0,1,1])
                else:
                    self.pgl.fixationCross(1,cameraImage.displayRight, cameraImage.displayTop, color=[1,0,0])

                # flush screen
                self.pgl.flush()

                # restart frame time counter
                lastTime = thisTime
            # If not time for a full refresh, just update time
            else:
                # update timer (TPx)
                self.dp.DPxUpdateRegCache()
                thisTime = self.dp.DPxGetTime()

                # poll for button press events
                events = self.pgl.devicesPoll()
                if events is None: continue
                #print(events)
                for event in events:
                    # handle the events
                    if event.id == "white left":
                        # exit the calibration loop
                        print("(pglTrackPixx3:calibrateEyeImage) Exiting calibrate eye image loop")
                        self.pgl.clearScreen((0,0,0))
                        self.pgl.flush()
                        self.pgl.clearScreen((0,0,0))
                        self.pgl.flush()
                        loopCalibration = False
                    elif event.id == "yellow left":
                        # decrease LED intensity
                        self.ledIntensity = max(0, self.ledIntensity - 1)
                        self.dp.TPxSetLEDIntensity(self.ledIntensity)
                        self.dp.DPxUpdateRegCache()

                        print(f"(pglTrackPixx3:calibrateEyeImage) Decreased LED intensity to {self.ledIntensity} {self.dp.TPxGetLEDIntensity()}.")

                    elif event.id == "red left":
                        # increase LED intensity
                        self.ledIntensity = min(8, self.ledIntensity + 1)
                        self.dp.TPxSetLEDIntensity(self.ledIntensity)
                        self.dp.DPxUpdateRegCache()
                        print(f"(pglTrackPixx3:calibrateEyeImage) Increased LED intensity to {self.ledIntensity}.")
                    elif event.id == "green left":
                        # increase lens
                        self.lens = min(2, self.lens + 1)
                        self.dp.TPxSetLens(self.lens)
                        self.dp.DPxUpdateRegCache()
                        print(f"(pglTrackPixx3:calibrateEyeImage) Increased lens focal length to {self.lens*25+25} mm.")
                    elif event.id == "blue left":
                        # decrease lens
                        self.lens = max(0, self.lens - 1)
                        self.dp.DPxUpdateRegCache()
                        self.dp.TPxSetLens(self.lens)
                        print(f"(pglTrackPixx3:calibrateEyeImage) Decreased lens focal length to {self.lens*25+25} mm.")
                    else:
                        print(f"(pglTrackPixx3:calibrateEyeImage) Unknown event: {event}")
    #################################################
    # calibrateEyePosition
    ################################################
    def calibrateEyePosition(self):
        """
        Calibrate the eye position. Based on TPxCalibrationTesting.py example code Step 4.
        """
         ###################################################
        ## Step 4, Calibrations and calibration results. ##
        ###################################################
        # This step uses PsychToolbox to display targets on screen and gathers
        # raw eye data in order to calibrate the tracker. It then calculates and
        # displays the results of the evaluation. It also saves those results
        # for further analysis.

        #=========================== Chinrest =====================================
        # Chinrest calibration uses 13 points of calibration specified by the
        # experiment. The disposition and dispersion of the points will determine
        # the coverage of the screen. WARNING: The disposition of the targets will
        # influence the quality of the calibration. Certain dispositions will prevent a successful
        # calibration. We recommend the disposition presented below which has proven
        # to provide good calibration results.  Chinrest calibrations provide
        # very good precision and reliability. However the subject's head must be stabilized
        # by a chinrest or else the slight head movement can produce a tracking error.
        # With a chinrest calibration, the relationship between raw eye
        # data and screen gaze position is determined by 4 different polynomials
        # representing each axis of each eye. The same calibration process will be
        # performed for each independant axis and eye giving four indenpendant
        # calibration processes and results: right eye x axis, right eye y axis,
        # left eye x axis and left eye y axis.
        #==========================================================================

        # It is NOT MANDATORY to convert between coordinate systems. However, you
        # must pay particular attention to different systems if switching between applications
        # or toolboxes using different coordinates systems.

        # 'eyeToCalibrate' indicates which eyes should be calibrated and
        # whether to retry to capturing data for that specific eye if the data was
        # evaluated as invalid.
        # 0 - ignore both eyes
        # 1 - calibrate left eye only
        # 2 - calibrate right eye only
        # 3 - calibrate both eyes
        eyeToCalibrate = 3

        #====================== Display ======================================

        # Specify the targets to calibrate using the Psychtoolbox coordinate system

        #     calibration target screen positions
        #  _______________________________________
        # |                                       |
        # |       7           2           6       |
        # |                                       |
        # |             11          10            |
        # |                                       |
        # |       5           1           3       |
        # |                                       |
        # |             12          13            |
        # |                                       |
        # |       9           4           8       |
        # |_______________________________________|

        # calibration width and height in degrees
        calibrationWidth = 20
        calibrationHeight = 20

        # calibration extents in degrees
        calibrationTop = calibrationHeight / 2
        calibrationBottom = -calibrationHeight / 2
        calibrationLeft = -calibrationWidth / 2
        calibrationRight = calibrationWidth / 2

        # calibration points in degrees
        calibrationPoints = np.array([
            [0, 0], # point 1: center
            [0, calibrationTop], # point 2: top
            [calibrationRight, 0], # point 3: right
            [0, calibrationBottom], # point 4: bottom
            [calibrationLeft, 0], # point 5: left
            [calibrationRight, calibrationTop], # point 6: top right corner
            [calibrationLeft, calibrationTop], # point 7: top left corner
            [calibrationRight, calibrationBottom], # point 8: bottom right corner
            [calibrationLeft, calibrationBottom], # point 9: bottom left corner
            [calibrationRight/2, calibrationTop/2], # point 10: mid top right corner
            [calibrationLeft/2, calibrationTop/2], # point 11: mid top left corner
            [-calibrationLeft/2, calibrationBottom/2], # point 12: mid bottom left corner
            [calibrationRight/2, calibrationBottom/2], # point 13: mid bottom right corner
         ])
        nCalibrationPoints = calibrationPoints.shape[0]

        # minimum display time between calibration points
        calibrationMinimumDisplayTime = 1.5

        # time in seconds after target onset to grab calibration point
        calibrationTime = 1

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #                       CALIBRATION STATE MACHINE
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # This state machine will :
        # 1 - Display a target -> enter state showing_dot
        # 2 - Shrink the target, swap colour and refresh every 0.07 seconds (make
        #     the target dynamic for better subject attention)
        # 3 - When 0.95 seconds has elapsed since target display, stop updating the
        #     target and save the eye information from the tracker
        # 4 - Repeat the first 3 steps for all calibration points
        #     (13 for chinrest, 16 for remote)
        # 5 - Finalize the calibration (see more details in the corresponding section)
        # 6 - Exit if accepted, try again if rejected


        # 'iCalibrationPoint' is the calibration point index that indicates what calibration point to
        # display and calibrate.
        iCalibrationPoint = 0

        # 't', 't2' and 't3' are time markers to determine how much time has elapsed since
        # a specific event.  They are used to control the display and the amount of time a
        # target is displayed before it is calibrated
 
        # 't' is actual time
        # 'tTargetPoint' is the time at which point a target is displayed
        # 'tDisplayUpdate' is the time of the last target display update (targets are dynamic)
        t = 0
        tCalibrationPointDisplay = t
        tDisplayUpdate = t

        # 'showingCalibrationPoint' is a state entered when a new target is displayed up until the eye
        # data for this target is gathered
        showingCalibrationPoint = False
    
        # 'finishCalibration' is the state of the entire calibration process
        finishCalibration = 0

        # 'currentCalibrationX' and 'currentCalibrationY' are the coordinates of the target to calibrate in the coordinate
        # system selected for this calibration
        currentCalibrationX = 0
        currentCalibrationY = 0

        # 'rawEyeData' contains the raw eye information from the tracker. The first
        # dimension represents each of the calibration points. The second
        # dimension represents each eye and axis.
        # column    description
        # 1         right eye horzontal axis (x)
        # 2         right eye vertical axis (y)
        # 3         left eye horizontal axis (x)
        # 4         left eye vertical axis (y)
        rawEyePosition = np.empty((nCalibrationPoints,4))

        # Wait for red button press to start calibration
        print("(pglTrackPixx3:calibrateEyePosition) Press red button to start calibration or white button to exit")
        event = self.pgl.eventsWaitFor(["red left", "white left"])
        if event.id == "white left":
            print("(pglTrackPixx3:calibrateEyePosition) Calibration cancelled.")
            return False
        print("(pglTrackPixx3:calibrateEyePosition) Calibration started.")
        self.pgl.clearScreen((0,0,0))
        self.pgl.flush()

        # loop until break or return
        while True:
            # Ensure calibrationMinimumDisplayTime has elapsed since the previous target display
            if ((t - tCalibrationPointDisplay) > calibrationMinimumDisplayTime):

                # get x and y coordinates of the current calibration point
                currentCalibrationX, currentCalibrationY = calibrationPoints[iCalibrationPoint,:]
                
                # print target positin
                if self.pgl.verbose > 0:
                    print(f"(pglTrackPixx3:calibrateEyePosition) Calibration point {iCalibrationPoint+1}/{nCalibrationPoints} at ({currentCalibrationX}, {currentCalibrationY})")
                
                # update calibration point 
                iCalibrationPoint += 1
                if iCalibrationPoint >= nCalibrationPoints:
                    # all calibration points have been displayed, exit loop
                    print("(pglTrackPixx3:calibrateEyePosition) All calibration points displayed.")
                    self.pgl.clearScreen((0,0,0))
                    self.pgl.flush()
                    finishCalibration = 1
                    break

                # Show new target specified by index i in xy
        #-->    # To have a valid calibration, it is necessary to ensure that the
                # subject is looking at the correct spot when getting the eye
                # information for that target, or else the gaze calibration will be
                # corrupted and invalid. Note that the system does not oblige you to
    		    # dispplay the target on screen, but not displaying the target will render the calibration almost impossible.
    		    # Any kind of stimulis could be used as long as its coordinates correspond
    		    # to the coordinates of the 'GetEyeDuringCalibrationRaw' in an orthongonal two
                # dimentional coordinate system for that target.
                self.pgl.dots(currentCalibrationX, currentCalibrationY)
                self.pgl.flush()

                # Enter state showing_dot
                showingCalibrationPoint = True
                # reset new target timer
                tCalibrationPointDisplay = t
            elif ((t - tCalibrationPointDisplay) > calibrationTime):
                #====================== Chinrest ==========================
                # GetEyeDuringCalibrationRaw acquires eye data from tracker.
                # It also saves that data in memory and is used (once all
                # targets are run) to calculate the formula used to convert
                # raw eye data to calibrate screen position.
    #-->        # It is mandatory to call this function to gather eye
                # information for all targets included in the calibration
                rawEyePosition[iCalibrationPoint,:] = self.dp.TPxGetEyePositionDuringCalib_returnsRaw(currentCalibrationX, currentCalibrationY, eyeToCalibrate)
                # print
                if self.pgl.verbose > 0:
                    print(f"(pglTrackPixx3:calibrateEyePosition) Eye data for calibration point {iCalibrationPoint} acquired: {rawEyePosition[iCalibrationPoint,0:4]}")
                # clear screen
                self.pgl.clearScreen((0,0,0))
                self.pgl.flush()
            
            # update timer
            self.dp.DPxUpdateRegCache()
            t = self.dp.DPxGetTime()
    
        #========================== Chinrest ==============================
        # 'TPxBestPolyFinishCalibration()' uses the data captured in the preceeding steps and
        # runs an optimization process to determine the parameters that
        # will best convert raw eye data to a calibrated gaze position on screen.
    #--># It is MANDATORY to call TPxBestPolyFinishCalibration() in order to calibrate
        # the tracker for a chinrest calibration
        #==================================================================
        self.dp.TPxBestPolyFinishCalibration()
        self.dp.DPxUpdateRegCache()
        
        self.dp.TPxClearDeviceCalibration()
        self.dp.DPxUpdateRegCache()


    #################################################
    # getCameraImage
    #################################################
    def getCameraImage(self):
        """
        Get the current camera image from the TrackPixx3 device.
        """
        if self.isPGLOpen() is False:
            print("(pglTrackPixx3:getCameraImage) PGL screen is not open. Cannot get camera image.")
            return None
        
        # check status
        if self.currentStatus < 0:
            print("(pglTrackPixx3:getCameraImage) TrackPixx3 did not initialize properly.")
            return None
        
        # Get the camera image
        self.dp.DPxUpdateRegCache()
        image = self.dp.TPxGetEyeImage()
        if image is None:
            print("(pglTrackPixx3) Failed to get camera image.")
            return None
        # convert to pglImage
        image = self.pgl.imageCreate(image)
        return image
    
    ################################################
    # isPGLOpen
    #################################################
    def isPGLOpen(self):
        """
        Check if the pgl is open.
        
        Returns:
            bool: True if pgl is open, False otherwise.
        """
        if self.pgl is None:
            return False
        else:
            return self.pgl.isOpen()
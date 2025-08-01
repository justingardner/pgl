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

        print(f"(pglTrackPixx3) TrackPixx3 initialized with LED intensity {self.ledIntensity} and lens {self.lens*25+25} mm.")

        # get device time
        lastTime = self.dp.DPxGetTime()
        thisTime = self.dp.DPxGetTime()

        # stay in a loop, drawing camera images and allowing experimenter/subject to adjust parameters
        loopCalibration = True
        while loopCalibration:

            # wait for a duration of 1/60 second.
            if (thisTime - lastTime) > 1/60: # Just refresh at 60 Hz.
                # clear screen
                self.pgl.clearScreen((0,0,0))

                # get camera image
                cameraImage = self.getCameraImage()
                cameraImage.display()

                # flush screen
                self.pgl.flush()


                # restart frame time counter
                lastTime = thisTime
            # If not time for a full refresh, just update time
            else:
                # poll for button press events
                self.pgl.devicesPoll()
                event = self.pgl.eventsGet()                
                if event is not None:
                    # handle the events
                    if event.id == "white left":
                        # exit the calibration loop
                        print("(pglTrackPixx3:calibrateEyeImage) Exiting calibrate eye image loop")
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
                        print(f"(pglTrackPixx3:calibrateEyeImage) Unknown event: {event.id}")

                # update timer (TPx)
                self.dp.DPxUpdateRegCache()
                thisTime = self.dp.DPxGetTime()



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
################################################################
#   filename: pglEyeTracker.py
#    purpose: Eye tracker class for PGL
#         by: JLG
#       date: August 1, 2025
################################################################

###########
# Import
##########
from pgl import pglDevice
#################################################################
# Parent class for devices
#################################################################
class pglEyeTracker(pglDevice):
    """
    Parent class for eye trackers in PGL.
    This class is intended to be subclassed by specific eye tracker implementations.
    """
    
    def __init__(self, pgl=None, deviceType=None):
        """
        Initialize the eye tracker device.
        """
        # default device type if not provided
        if deviceType is None: deviceType = "pglEyeTracker"
        # call superclass constructor
        super().__init__(deviceType)
        # state of the calibration and tracking
        self.isCalibrated = False
        self.calibrationTime = None
        self.isTracking = False 
        self.pgl = pgl

    def __del__(self):
        """Destructor to clean up resources."""
        # stop tracking if it is active
        if self.isTracking: 
            print("(pglEyeTracker) Eye tracker is still tracking, stopping before cleanup.")
            self.stop()
        print(f"(pglEyeTracker) Eye tracker {self.deviceType} shutdown.")
        self.status = -1
        self.isCalibrated = False


    def calibrate(self):
        """Calibrate the eye tracker."""
        # perform calibration logic here
        self.isCalibrated = True
        self.calibrationTime = self.pglTimestamp.getDateAndTime()

    def calibrate(self):
        """Calibrate the eye tracker."""
        # perform calibration logic here
        self.isCalibrated = True
        self.calibrationTime = self.pglTimestamp.getDateAndTime()

    def calibrateEyeImage(self):
        """
        Calibrate the eye image.

        This method should be implemented by subclasses to display the eye image and
        Allow experimenter/subject to adjust the illuminator intensity, focus, position
        and other paramteres to optimize view of the eye
        """
        print("(pglEyeTracker:calibrateEyeImage) This method should be implemented by the specific eye tracker subclass.")
        return None
  
    def getCameraImage(self):
        """
        Get the current camera image from the eye tracker.
        To be implemented by subclasses.

        Returns:
            pglImage: The current camera image from the eye tracker.
        """
        print("(pglEyeTracker:getCameraImage) This method should be implemented by the specific eye tracker subclass.")
        return None

    def start(self):
        """Start eye tracking."""
        if not self.isCalibrated:
            print("(pglEyeTracker:start) Eye tracker must be calibrated before starting tracking.")
            return
        # start tracking
        self.isTracking = True
        print("(pglEyeTracker) Eye tracking started.")


    def stop(self):
        """Stop eye tracking."""
        self.isTracking = False
        print("Eye tracking stopped.")
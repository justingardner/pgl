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
from .pglData import pglTimeSeries, pglEventsData

#################################################################
# Parent class for eye tracker devices
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

    def start(self):
        """Start eye tracking."""
        if not self.isCalibrated:
            print("(pglEyeTracker:start) ❌ Eye tracker must be calibrated before starting tracking.")
            return
        # start tracking
        self.isTracking = True
        print("(pglEyeTracker) Eye tracking started.")


    def stop(self):
        """Stop eye tracking."""
        self.isTracking = False
        print("(pglEyeTracker) Eye tracking stopped.")

    def save(self, filename):
        """Stop recording and retrieve data file.
        
        Args:
            filename (str): Name of the file to save locally
        """
        # This method should be implemented by subclasses to save the eye tracking data
        raise NotImplementedError("saveData method must be implemented by subclasses of pglEyeTracker.")

#################################################################
# saccade events
#################################################################
class pglEventSaccade(pglEvent):
    timeStart: float
    timeEnd: float    
    xStart: float
    yStart: float
    xEnd: float
    yEnd: float
    maxVelocity: float
    duration: float
    amplitude: float
    direction: float
    
    def __init__(self, timeStart, timeEnd, xStart, yStart, xEnd, yEnd, maxVelocity=None):
        '''
        init with field names from annotation above
        
        Args:
            timeStart: float timestamp for start of saccade
            timeEnd: float timestamp for end of saccade   
            xStart: float x position of start of saccade
            yStart: float y position of start of saccade
            xEnd: float x position of end of saccade
            yEnd: float y position of end of saccade
            maxVelocity: float max velocity of saccade
        '''
        # compute fields
        duration = timeEnd-timeStart
        amplitude = np.sqrt((xEnd-xStart)**2 + (yEnd-yStart)**2)
        if amplitude > 0:
            direction = np.degrees(np.arctan2(yEnd - yStart, xEnd - xStart))
        else:
            direction = np.nan
        
        # and use super init to set them (as super set all annotation fields)
        super().__init__("saccadeEvent", timeStart, timeEnd, xStart, yStart, xEnd, yEnd, maxVelocity, duration, amplitude, direction)
    
    
    
#################################################################
# Parent class for eye tracker data
#################################################################
class pglEyeTrackerData():
    def __init__(self):
        """
        initializes the eye tracker data. 
        """
        # initializes time series and events
        self.timeSeries = None
        self.trialEvents = None

    def addTimeseries(self, timeSeries, channelNames, units, sampleRate):
        '''
        Add time series data
        
        Args:
            timeSeries (Array): Array of timeSeries data, rows are time, columns are different variables
            channelNames (List): List of string names of each column of data
            units (List): List of strings which identify units of each column of data
            samplingRate (Int): Sampling rate of data
        '''
        # set the timeseries
        self.timeSeries = pglTimeSeries.fromArray(data=timeSeries, channelNames=channelNames, units=units, sampleRate=sampleRate)
    
    def addTrialEvents(self, data, fieldNames, units):
        '''
        Add trial events which specify when the trials start and stop
        
        Args:
            taskID (int): The taskID which specifies which task the trials belong to
            trialStart (List): List of timestamps of trials
            segmentStart (List): List of segment start times
        '''
        #self.trialEvents = pglEventsData.fromArray(pglEventTrialEyeTracker, data=data, channelNames=fieldNames, units=units)
        
        pass
    
    def addSaccadeEvents(self):
        '''
        Add saccade events as a data matrix
        
        FIX, FIX, FIX - not tested yet
        '''
        pass
        self.trialEvents = pglEventsData.fromArray(pglEventSaccade, data=data, channelNames=fieldNames, units=units)
        
    def addSaccadeEvents(self):
    
    
    def addBlinkEvents(self):
        pass
        

    def print(self):
        """
        print information about the eye tracker data
        """
        # if there is timeSeries data then print information about that
        if self.timeSeries is not None:
            self.timeSeries.print()
            
        # print trial events
        if self.trialEvents is not None:
            self.trialEvents.print()
    
    def display(self):
        """
        display plots of data
        """
        pass

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
from .pglEvent import pglEvent
from dataclasses import dataclass, field
import numpy as np

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
@dataclass
class pglEventSaccade(pglEvent):
    eye: float = field(metadata={"units": "eye: -1 = left, 1 = right"})
    timeStart: float = field(metadata={"units": "s"})
    timeEnd: float = field(metadata={"units": "s"})
    xStart: float = field(metadata={"units": "deg"})
    yStart: float = field(metadata={"units": "deg"})
    xEnd: float = field(metadata={"units": "deg"})
    yEnd: float = field(metadata={"units": "deg"})
    maxVelocity: float = field(metadata={"units": "deg/s"})
    duration: float = field(metadata={"units": "s"})
    amplitude: float = field(metadata={"units": "deg"})
    direction: float = field(metadata={"units": "deg"})
    
    def __init__(self, eye, timeStart, timeEnd, xStart, yStart, xEnd, yEnd, maxVelocity=None, duration=None, amplitude=None, direction=None):
        '''
        init with field names from annotation above
        
        Args:
            eye: float indicating which eye ( -1 = left, 1 = right)
            timeStart: float timestamp for start of saccade
            timeEnd: float timestamp for end of saccade   
            xStart: float x position of start of saccade
            yStart: float y position of start of saccade
            xEnd: float x position of end of saccade
            yEnd: float y position of end of saccade
            maxVelocity: float max velocity of saccade
            duration: float duration of saccade
            amplitude: float amplitude of saccade
            direction: float direction of saccade
        '''
        # compute fields
        if duration is None: duration = timeEnd - timeStart
        if amplitude is None:
            amplitude = np.sqrt((xEnd-xStart)**2 + (yEnd-yStart)**2)
        if direction is None:
            if amplitude > 0:
                direction = np.degrees(np.arctan2(yEnd - yStart, xEnd - xStart))
            else:
                direction = np.nan
        
        # and use super init to set them (as super set all annotation fields)
        super().__init__(
            type="saccadeEvent",
            eye=eye,
            timeStart=timeStart,
            timeEnd=timeEnd,
            xStart=xStart,
            yStart=yStart,
            xEnd=xEnd,
            yEnd=yEnd,
            maxVelocity=maxVelocity,
            duration=duration,
            amplitude=amplitude,
            direction=direction
        )
    
#################################################################
# saccade events
#################################################################
@dataclass
class pglEventBlink(pglEvent):
    timeStart: float = field(metadata={"units": "s"})
    timeEnd: float = field(metadata={"units": "s"})
    duration: float = field(metadata={"units": "s"})
    
    def __init__(self, timeStart, timeEnd, duration=None):
        '''
        init with field names from annotation above
        
        Args:
            timeStart: float timestamp for start of blink
            timeEnd: float timestamp for end of blink
            duration: float duration of blink
        '''
        # compute fields
        if duration is None: duration = timeEnd - timeStart
        
        # and use super init to set them (as super set all annotation fields)
        super().__init__(
            type="blinkEvent",
            timeStart=timeStart,
            timeEnd=timeEnd,
            duration=duration,
        )
#################################################################
# trial events
#################################################################
@dataclass
class pglEventEyeTrackerTrial(pglEvent):
    taskID: float = field(metadata={"units": "n"})
    trialNum: float = field(metadata={"units": "n"})
    segmentNum: float = field(metadata={"units": "n"})
    timestamp: float = field(metadata={"units": "s"})
 
    
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
        self.trialEvents = pglEventsData(pglEventEyeTrackerTrial)
        self.saccadeEvents = pglEventsData(pglEventSaccade)
        self.blinkEvents = pglEventsData(pglEventBlink)

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

    def alignTimeSeriesToTrialEvents(self, trialEvents):
        """
        Align the time series data by trial events.
        
        Args:
            trialEvents (list of pglEventEyeTrackerTrial): The list of trial events to align to.
        """
        # check if trialEvents is a list of pglEventEyeTrackerTrial
        if not isinstance(trialEvents, list) or not all(isinstance(e, pglEventEyeTrackerTrial) for e in trialEvents):
            raise ValueError("(pglEyeTrackerData:alignTimeSeriesToTrialEvents) ❌ trialEvents must be a list of pglEventEyeTrackerTrial instances.")


        # check if timeSeries data exists
        if self.timeSeries is None:
            print("(pglEyeTrackerData:alignTimeSeriesToTrialEvents) ❌ No time series data to align.")
            return
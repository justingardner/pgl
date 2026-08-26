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
from traitlets import Float

################################
# # class for an eye sample
################################
class pglEyePositionSample(pglTraitSettings):
    '''
    sample of an eye position
    '''
    x = Float(help='x position of gaze sample')
    y = Float(help='y position of gaze sample')
    pupilSize = Float(allow_none=True, help='pupilSize')
    whichEye = Enum(values=['left', 'right', 'both', 'unknown'],default_value='unknown',help='Eye of gaze sample')
    
    def __init__(self, x, y, pupilSize=None, whichEye='unknown'):
        '''
        initialize with x, y and pupilSize
        
        Args:
            x (float): x position of eye
            y (float): y position of eye
            pupilSize (float): pupil size
            whichEye (str): EIther 'left', 'right', 'both' means the average of left/right, defaults to 'unknown'
        '''
        super().__init__()
        self.x = x
        self.y = y
        self.pupilSize = pupilSize
        self.whichEye = whichEye
        
    def __sub__(self, other):
        '''
        Subtraction operator computes euclidian distance between two samples
        '''
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    @classmethod
    def average(cls, samples):
        '''
        Returns the average of a collection of eye position samples.
        '''

        x = np.mean([sample.x for sample in samples])
        y = np.mean([sample.y for sample in samples])

        pupilSizes = [
            sample.pupilSize
            for sample in samples
            if sample.pupilSize is not None
        ]

        pupilSize = np.mean(pupilSizes) if pupilSizes else None

        return cls(x, y, pupilSize, 'both')
    
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
    
    def getEyePosition(self):
        '''
        Gets a sample of the current eye position.

        Returns:
            A dictionary with pglEyePositionSample for:
                'left': left eye
                'right': right eye
                'either': whichever eye has valid data; if both have valid
                        data, will be the average.
            If there is no data for an eye, its value will be None.
            
        '''
        # This method should be implemented by subclasses
        # note that the below_addEitherEyePosition can be used by subclass to compute
        # the either entry from left and right entries 
        raise NotImplementedError("getEyePosition method must be implemented by subclasses of pglEyeTracker.")
        
    def _addEitherEyePosition(self, eyePositions):
        '''
        Helper function, which can be called by subclassed getEyePosition.

        Adds an 'either' entry to an eye position dictionary.
        '''

        left = eyePositions['left']
        right = eyePositions['right']

        if left is not None and right is not None:

            eyePositions['either'] = pglEyePositionSample.average([left, right])
            eyePositions['either'].whichEye = 'both'

        elif left is not None:
            
            eyePositions['either'] = left
            eyePositions['either'].whichEye = 'left'

        elif right is not None:
            eyePositions['either'] = right
            eyePositions['either'].whichEye = 'right'

        else:
            eyePositions['either'] = None

        return eyePositions


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
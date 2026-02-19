################################################################
#   filename: pglStaircase.py
#    purpose: Class for managing staircase procedures
#         by: JLG
#       date: September 17, 2025
################################################################

#############
# Import modules
#############
import numpy as np
import itertools
import random
from . import pglParameter
from .pglTimestamp import pglTimestamp
from .pglSettings import pglSerializable
from dataclasses import dataclass
from traitlets import Float, Int, List
from .pglData import pglData
from .pglSettings import _pglSettings
    
##########################
# Staircase class
##########################
class pglStaircase():
    '''
    Class representing a staircase parameter in the experiment.
    This can be added to an experiment as a parameter
    '''
    def __init__(self, description="staircase"):
        '''
        Initialize the staircase, by setting up structures to save data. this
        should be called by the subclass after the parameters are set.
        '''
        # save the description
        self.description = description
        # history of staircases, each staircase is an entry in this list
        self.history = []
        # initialize the first staircase
        self.startStaircase()
        
    def get(self):
        '''
        Get the test value for the current trial. This shouuld
        be implemented by the subclass
        '''
        pass

    def update(self, value, correct):
        '''
        Subclass should handle what to do with the update, and
        then call this method to store the value and response
        in the history.
        '''
        self.history[-1]["values"].append(value)
        self.history[-1]["responses"].append(correct)
        self.history[-1]["nTrials"] += 1

    def getThresholdEstimate(self):
        '''
        Get the current threshold estimate of the staircase parameter.
        '''
        pass
    
    def plot(self):
        '''
        Plot the history of the staircase parameter.
        '''
        pass
    
    def startStaircase(self, startVal=None):
        '''
        start a staircase, initializes the history. For subclasses,
        this should be called after you initialize the staircase parameters
        so they get stored in the history correctly.
        '''
        # Each entry of the history represents a staircase run
        # we first save all the values of the staircase to the history
        self.history.append(self.__dict__.copy())
        
        # add the start time
        self.history[-1]["startTime"] = self.pglTimestamp.getSecs()
        
        # initialize values and responses
        self.history[-1]["values"] = []
        self.history[-1]["responses"] = []
        self.history[-1]["nTrials"] = 0

    def test(self, observerModel=None, nTrials=100, minVal=0.0, maxVal=100.0, threshold=7.5, slope=3.0):
        '''
        Test the staircase by running a simple simulation.
        '''
        
        # get an observer model
        if observerModel is None:
            observerModel = pglObsserverModel(threshold=threshold, slope=slope)

        # start a new staircase
        self.startStaircase()
        
        # run the staircase for nTrials
        for trial in range(nTrials):
            # get the current stimulus value from the staircase
            stimulus = self.get()
            # 2AFC response comparing stimulus to 0
            response = observerModel.get2AFCResponse(stimulus, 0)
            # update the staircase
            self.update(stimulus, response)

        return self.getThresholdEstimate()

##########################
# Observer model class
##########################
class pglObserverModel():
    '''
    Class representing an observer model for simulating responses.
    '''
    def __init__(self, threshold=7.5, slope=3.0):
        '''
        Initialize the observer model.
        '''
        self.threshold = threshold
        self.slope = slope
    
    def get2AFCResponse(self, stimulusValue1, stimulustValue2):
        '''
        Get the response of the observer model to a given stimulus value.
        Response is coded as 1 for correct, 0 for incorrect
        '''
        # simulate response
        return random.random() < self.psychometricFunction(np.abs(stimulusValue1-stimulusValue2))

    def psychometricFunction(self, stimulusValues):
        '''
        Get the psychometric function for a range of stimulus values.
        '''
        # weibull function
        return 1 - 0.5 * np.exp(-(stimulusValues / self.threshold) ** self.slope)

##########################
# UpDown Staircase class
##########################
class pglStaircaseUpDown(pglStaircase):
    '''
    Class representing a simple up-down staircase in the experiment.
    This is a subclass of pglStaircase so that you can add it
    to an experiment like other parameters. It does 
    '''
    def __init__(self, nUp=2, nDown=1, stepSize=1,
                 startVal=50.0, minVal=0.0, maxVal=100.0,
                 stepChangeSizes=[0.5, 0.25, 0.125], stepChangeReversals=[3, 6, 9],
                 description=""):
        '''
        Initialize the up-down staircase 
        '''

        # staircase parameters
        self.nUp = nUp
        self.nDown = nDown
        self.stepSize = stepSize
        self.startVal = startVal
        self.minVal = minVal
        self.maxVal = maxVal
        self.stepChangeSizes = stepChangeSizes
        self.stepChangeReversals = stepChangeReversals

        # run init to start the staircase history (this needs to come
        # after the parameters are set)
        super().__init__(description)
      
    def startStaircase(self, startVal=None):
        '''
        start the staircase
        '''

        # initialize current value
        self.currentVal = self.startVal

        # initialize other variables
        self.correctStreak = 0
        self.incorrectStreak = 0
        self.reversals = 0
        self.lastDirection = None  

        # call superclass method to set up history
        super().startStaircase(startVal)
        
        # add reversals to the dictionary
        self.history[-1]["reversals"] = []
        
    def get(self):
        '''
        Get the test value for the current trial.
        '''
        return self.currentVal
    
    def update(self,value,response):
        '''
        Update the staircase with the latest trial results.
        '''
        # update the history by calling superclass method
        super().update(value, response)
        
        # update the streaks
        if response == 1:
            self.correctStreak += 1
            self.incorrectStreak = 0
        else:
            self.incorrectStreak += 1
            self.correctStreak = 0

        # check for streaks that trigger a change
        if self.correctStreak == self.nUp:
            # go up
            self.currentVal += self.stepSize
            # clamp to max
            if self.currentVal > self.maxVal: self.currentVal = self.maxVal
            # reset streak
            self.correctStreak = 0
            # check for reversal
            if self.lastDirection == "down":
                self.reversals += 1
                self.history[-1]["reversals"].append(1)
            else:
                self.history[-1]["reversals"].append(0)
            # remember the last direction
            self.lastDirection = "up"
        elif self.incorrectStreak == self.nDown:
            # go down
            self.currentVal -= self.stepSize
            # clamp to min
            if self.currentVal < self.minVal: self.currentVal = self.minVal
            # reset streak
            self.incorrectStreak = 0
            # check for reversal
            if self.lastDirection == "up":
                self.reversals += 1
                self.history[-1]["reversals"].append(1)
            else:
                self.history[-1]["reversals"].append(0)
            # remember the last direction
            self.lastDirection = "down"
            
@dataclass
pglStaircaseData(pglData):
    '''
    Class representing the data from a staircase.
    This is used to save/load staircase data.
    '''
    startTime: float = 0.0
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    responses: np.ndarray = field(default_factory=lambda: np.array([]))
    nTrials: int = 0
    reversals: list = field(default_factory=list)
    

    
    def __init__(self):
        '''
        Initialize the staircase data.
        '''
        self.history = []


pglStaircaseData(_pglSettings):
    nDown = Int(2, min=0, step=1, help="Number of trials in a row before increasing difficulty")
    nUp = Int(1, min=0, step=1, help="Number of trials in a row before increasing difficulty")
    startVal = Float(default_value=None, allow_none=True, help="Starting value for the staircase")
    minVal = Float(default_value=None, allow_none=True, help="Starting value for the staircase")
    maxVal = Float(default_value=None, allow_none=True, help="Starting value for the staircase")
    stepSize = Float(1.0, min=0.0, help="Step size for the staircase")
    stepChangeSizes = List(Float(), default_value=[0.5, 0.25, 0.125], help="Step sizes to change to at each reversal count")
    stepChangeReversals = List(Int(), default_value=[2, 4, 6], help="Reversal counts at which to change step size")

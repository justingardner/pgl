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
import matplotlib.pyplot as plt
from .pglTimestamp import pglTimestamp
from dataclasses import dataclass, field
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
        # init list of all data
        self.allData = []
        # save description
        self.description = description
        # timestamp object for getting times
        self.pglTimestamp = pglTimestamp()
        
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
        self.data.values.append(value)
        self.data.responses.append(correct)
        self.data.nTrials += 1
    
    def finished(self):
        '''
        returns whether the staircase should finish. Subclasses
        should implement this if needed.
        '''
        return False
        
    def startStaircase(self, startVal=None):
        '''
        start a staircase, initializes the history. For subclasses,
        this should be called after you initialize the staircase parameters
        so they get stored in the history correctly.
        '''
        # initialize data
        self.data = pglStaircaseData()
        # save in our list of all data
        self.allData.append(self.data)
        # save the description
        self.data.description = self.description

        # add the start time
        self.data.startTime = self.pglTimestamp.getSecs()

    def endStaircase(self):
        '''
        call to end the staircase
        '''
        # add end time
        self.data.endTime = self.pglTimestamp.getSecs()
        
    def getThresholdEstimate(self):
        '''
        Get the current threshold estimate of the staircase parameter.
        '''
        pass
    
    def plot(self):
        '''
        Plot the history of the staircase parameter.
        '''
        ax = self.data.plot()
        return ax
            
    def test(self, observerModel=None, nTrials=100, minVal=0.0, maxVal=1.0, threshold=0.075, slope=3.0):
        '''
        Test the staircase by running a simple simulation.
        '''
        
        # get an observer model
        if observerModel is None:
            observerModel = pglObserverModel(threshold=threshold, slope=slope)

        # start a new staircase
        self.startStaircase()
        
        # run the staircase for nTrials
        while not self.finished() and (self.data.nTrials < nTrials):
            # get the current stimulus value from the staircase
            stimulus = self.get()
            # 2AFC response comparing stimulus to 0
            response = observerModel.get2AFCResponse(stimulus, 0)
            # update the staircase
            self.update(stimulus, response)

        # end the staircase
        self.endStaircase()
        
        # plot the results
        ax = self.plot()
        
        # plot a horizontal line at the threshold
        trueThreshold = threshold*(-np.log(1-0.707))**(1/slope)
        ax.axhline(y=threshold, color='g', linestyle='--', label='True Threshold')
        ax.legend()
        
        return

##########################
# Observer model class
##########################
class pglObserverModel():
    '''
    Class representing an observer model for simulating responses.
    '''
    def __init__(self, threshold=0.075, slope=3.0):
        '''
        Initialize the observer model.
        '''
        self.threshold = threshold
        self.slope = slope
    
    def get2AFCResponse(self, stimulusValue1, stimulusValue2):
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
    def __init__(self, nUp=2, nDown=1, stepSize=0.05,
                 startVal=0.3, minVal=0.0, maxVal=1.0,
                 stepChangeSizes=[0.05, 0.025, 0.0125], stepChangeReversals=[2, 4, 6],
                 numReversalsToFinish=12,
                 description="upDownStaircase"):
        '''
        Initialize the up-down staircase 
        '''

        # staircase parameters
        self.settings = pglStaircaseSettings()
        self.settings.nUp = nUp
        self.settings.nDown = nDown
        self.settings.stepSize = stepSize
        self.settings.startVal = startVal
        self.settings.minVal = minVal
        self.settings.maxVal = maxVal
        self.settings.stepChangeSizes = stepChangeSizes
        self.settings.stepChangeReversals = stepChangeReversals
        self.settings.numReversalsToFinish = numReversalsToFinish
        
        # run init to start the staircase history (this needs to come
        # after the parameters are set)
        super().__init__(description)
      
    def startStaircase(self, startVal=None):
        '''
        start the staircase
        '''

        # initialize current value
        self.currentVal = self.settings.startVal

        # initialize other variables
        self.correctStreak = 0
        self.incorrectStreak = 0
        self.lastDirection = None  

        # call superclass method to initialize data
        super().startStaircase(startVal)
        
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
        if self.correctStreak == self.settings.nUp:
            # go down
            self.currentVal -= self.settings.stepSize
            # clamp to max
            if self.currentVal > self.settings.maxVal: self.currentVal = self.settings.maxVal
            # reset streak
            self.correctStreak = 0
            # check for reversal
            if self.lastDirection == "up":
                self.data.reversals.append(self.data.nTrials)
                self.updateStepSize()
            # remember the last direction
            self.lastDirection = "down"
        elif self.incorrectStreak == self.settings.nDown:
            # go up
            self.currentVal += self.settings.stepSize
            # clamp to min
            if self.currentVal < self.settings.minVal: self.currentVal = self.settings.minVal
            # reset streak
            self.incorrectStreak = 0
            # check for reversal
            if self.lastDirection == "down":
                self.data.reversals.append(self.data.nTrials)
                self.updateStepSize()
            # remember the last direction
            self.lastDirection = "up"
        # clamp current value
        if self.currentVal < self.settings.minVal:
            self.currentVal = self.settings.minVal
        if self.currentVal > self.settings.maxVal:
            self.currentVal = self.settings.maxVal
    def updateStepSize(self):
        # get number of reversals
        numReversals = len(self.data.reversals)

        # loog in stepChangeReversals for a match
        for revThreshold, stepSize in zip(
            self.settings.stepChangeReversals,
            self.settings.stepChangeSizes
        ):
            # if there is a match then set stepsize
            if numReversals >= revThreshold:
                self.settings.stepSize = stepSize

    def finished(self):
        '''
        returns whether the staircase should finish. 
        '''
        if len(self.data.reversals) >= self.settings.numReversalsToFinish:
            return True

            
@dataclass
class pglStaircaseData(pglData):
    '''
    Class representing the data from a staircase.
    This is used to save/load staircase data.
    '''
    description: str = ""
    startTime: float = 0.0
    values: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    nTrials: int = 0
    reversals: list = field(default_factory=list)  
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        '''
        Plot the staircase data.
        '''
        ax.plot(self.values, marker='o')
        ax.set_title(f'Staircase Data: {self.description}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Staircase Value')
        for rev in self.reversals:
            ax.axvline(x=rev, color='r', linestyle='--', label='Reversal' if rev == self.reversals[0] else "")
        if self.reversals:
            ax.legend()
        return ax

class pglStaircaseSettings(_pglSettings):
    nDown = Int(2, min=0, step=1, help="Number of trials in a row before increasing difficulty")
    nUp = Int(1, min=0, step=1, help="Number of trials in a row before increasing difficulty")
    startVal = Float(default_value=None, allow_none=True, help="Starting value for the staircase")
    minVal = Float(default_value=None, allow_none=True, help="Starting value for the staircase")
    maxVal = Float(default_value=None, allow_none=True, help="Starting value for the staircase")
    stepSize = Float(1.0, min=0.0, help="Step size for the staircase")
    stepChangeSizes = List(Float(), default_value=[0.5, 0.25, 0.125], help="Step sizes to change to at each reversal count")
    stepChangeReversals = List(Int(), default_value=[2, 4, 6], help="Reversal counts at which to change step size")
    numReversalsToFinish = Int(8, min=1, help="Number of reversals at which the staircase is considered to have converged and should therefore finish")

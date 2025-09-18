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
    
##########################
# Staircase class
##########################
class pglStaircase():
    '''
    Class representing a staircase parameter in the experiment.
    This can be added to an experiment as a parameter
    '''
    def __init__(self, description=""):
        '''
        Initialize the staircase parameter.
        '''
    def get(self):
        '''
        Get the test value for the current trial.
        '''
        pass

    def update(self, value, correct):
        '''
        Update the staircase based on the response.
        '''
        pass
    
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
    
    def restart(self, startVal=None):
        '''
        Restart the staircase 
        '''
        pass
    
##########################
# UpDown Staircase class
##########################
class pglStaircaseUpDown(pglStaircase):
    '''
    Class representing a simple up-down staircase parameter in the experiment.
    This is a subclass of pglStaircase so that you can add it
    to an experiment like other parameters. It does 
    '''
    def __init__(self, nUp=2, nDown=1, stepSize=1,
                 startVal=50.0, minVal=0.0, maxVal=100.0,
                 stepChangeSizes=[0.5, 0.25, 0.125], stepChangeReversals=[3, 6, 9],
                 description=""):
        '''
        Initialize the up-down staircase parameter.
        '''

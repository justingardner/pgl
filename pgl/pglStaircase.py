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
class pglStaircase(pglParameter):
    '''
    Class representing a staircase parameter in the experiment.
    This is a subclass of pglParameter so that you can add it
    to an experiment like other parameters. It does 
    '''
    def __init__(self, parameters: list, helpStr: str=""):
        '''
        Initialize the staircase parameter.
        '''

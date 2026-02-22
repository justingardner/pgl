################################################################
#   filename: pglParameter.py
#    purpose: Parameter class which handles parameter generation
#             and randomization for experiments.
#         by: JLG
#       date: September 17, 2025
################################################################

#############
# Import modules
#############
import numpy as np
import itertools
import random
    
#############
# Parameter class
#############
class pglParameter:
    '''
    Class representing a parameter in the experiment.
    '''
    def __init__(self, name: str, validValues: list|tuple|np.ndarray, description: str=""):
        '''
        Initialize the parameter.
        
        Args:
            name (str): The name of the parameter.
            validValues (list, optional): A list of valid values for the parameter. 
            description (str, optional): Description string describing the parameter.
        '''
        # validate name
        if not isinstance(name, str):
            raise TypeError("(pglParameter) ❌ Error: Parameter name must be a string.")
        self.name = name
        
        # check if it is a list of values
        if not isinstance(validValues, (list, tuple, np.ndarray)):
            raise TypeError("(pglParameter) ❌ Error: validValues must be a list, tuple, or ndarray.")
        self.validValues = list(validValues)
        
        # validate description
        if not isinstance(description, str):
            raise TypeError("(pglParameter) ❌ Error: description must be a string.")
        self.description = description
        
        # set to trigger a new block for first trial
        self.currentTrial = -1
        self.blockNum = -1
        self.blockLen = 1

    def __repr__(self):
        return f"pglParameter(name={self.name}, validValues={self.validValues}, description={self.description})"

    def __str__(self):
        # display description
        if self.description == "":
            description = ""
        else:
            description = f"# {self.description} #\n"

        # display full string
        return f"{description}{self.name}: {self.validValues} (randomizationBlock={self.randomizationBlock})"

    def get(self):
        '''
        Get the current value of the parameter.
        '''
        # check if we are at the end of a block
        if (self.currentTrial%self.blockLen) == self.blockLen-1:
            self.startBlock()

        # update trial number
        self.currentTrial += 1

        return dict(zip(self.parameterNames, self.parameterBlock[self.currentTrial%self.blockLen]))
    
    def getParameterBlock(self):
        '''
        Get a set of parameters to run over, will
        produce a random ordering of that parameter
        '''
        paramNames = [self.name]
        # turn the list into a list of tuples to be 
        # compatible with tuples of parameters
        # from above and then shuffle
        parameterBlock = [(v,) for v in self.validValues]
        random.shuffle(parameterBlock)
        
        # return the block
        return (paramNames, parameterBlock)

    def startBlock(self):
        '''
        Start a block.
        '''
        # get randomization of parameters
        (self.parameterNames, self.parameterBlock) = self.getParameterBlock()
        
        # set variables
        self.blockNum += 1
        self.blockLen = len(self.parameterBlock)
        
        # display block information        
        print(f"Block {self.blockNum+1}: {self.blockLen} randomized over: {self.parameterNames}")

##########################
# Parameter block class
##########################
class pglParameterBlock(pglParameter):
    '''
    Class representing a block of parameters in the experiment.
    This is a subclass of pglParameter which allows you to group
    multiple parameters together into a single block.
    '''
    def __init__(self, parameters: list, description: str=""):
        '''
        Initialize the parameter block.
        
        Args:
            name (str): The name of the parameter block.
            parameters (list): A list of pglParameter instances to include in the block.
            description (str, optional): Description string describing the parameter block.
        '''
        # validate parameters
        if not isinstance(parameters, list) or not all(isinstance(p, pglParameter) for p in parameters):
            raise TypeError("(pglParameterBlock) ❌ Error: parameters must be a list of pglParameter instances.")
        self.parameters = parameters

        # validate description
        if not isinstance(description, str):
            raise TypeError("(pglParameterBlock) ❌ Error: description must be a string.")
        self.description = description

        # set to trigger a new block for first trial
        self.currentTrial = -1
        self.blockNum = -1
        self.blockLen = 1

        # parameter names and valid values from each parameter in the list
        self.paramNames = [p.name for p in self.parameters]
        allParameterValues = [p.validValues for p in self.parameters]
        # get cartesian combination
        self.allParameterValues = list(itertools.product(*allParameterValues))
        self.name = ""

    def __repr__(self):
        return f"pglParameterBlock(parameters={self.parameters}, description={self.description})"

    def __str__(self):
        # display description
        if self.description == "":
            description = ""
        else:
            description = f"# {self.description} #\n"

        # display full string
        paramStrs = [str(p) for p in self.parameters]
        paramStr = "\n".join(paramStrs)
        return f"{description}:\n{paramStr}"

    def getParameterBlock(self):
        '''
        This  will create a block over all of the parameters in the
        list, for example if you have direction and coherence parameters, this would
        calculate all combination of those parameters and return them for the task
        to run as a block of trials.
        '''
        random.shuffle(self.allParameterValues)
        return (self.paramNames, self.allParameterValues)


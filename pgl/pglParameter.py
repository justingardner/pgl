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
from dataclasses import dataclass
from pathlib import Path
from .pglSerialize import pglSerialize
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

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
    def __init__(self, name: str, validValues: list|tuple|np.ndarray, description: str="", randomSeed=None):
        '''
        Initialize the parameter.
        
        Args:
            name (str): The name of the parameter.
            validValues (list, optional): A list of valid values for the parameter. 
            description (str, optional): Description string describing the parameter.
        '''
        # initialize state, data, and settings
        self.settings = pglParameterSettings()
        self.state = pglParameterState()
        self.data = pglParameterData()
        
        # validate name
        if not isinstance(name, str):
            raise TypeError("(pglParameter) ❌ Error: Parameter name must be a string.")
        self.settings.name = name
        
        # check if it is a list of values
        if not isinstance(validValues, (list, tuple, np.ndarray)):
            raise TypeError("(pglParameter) ❌ Error: validValues must be a list, tuple, or ndarray.")
        self.settings.validValues = list(validValues)
        
        # validate description
        if not isinstance(description, str):
            raise TypeError("(pglParameter) ❌ Error: description must be a string.")
        self.settings.description = description
        
        # set to trigger a new block for first trial
        self.state.currentTrial = -1
        self.state.blockNum = -1
        self.state.blockLen = 1
        
        # initialize random number generation
        self.setRandomSeed(randomSeed)

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

    def setRandomSeed(self, randomSeed=None):
        '''
        Set the random seed for the parameter.
        '''
        # initialize random number generation
        if randomSeed is None:
            randomSeed = np.random.default_rng().integers(0, 2**32)
        
        # save randomSeed and start random number generator
        self.settings.randomSeed = randomSeed
        self._rng = np.random.default_rng(randomSeed)

    def get(self):
        '''
        Get the current value of the parameter.
        '''
        # increment trial number
        self.state.currentTrial += 1
        self.state.currentTrialInBlock += 1
        
        # check if we need to start a new block
        if self.state.blockNum == -1 or self.state.currentTrialInBlock >= self.data.blockLengths[self.state.blockNum]:
            self.startBlock()

        # get parameter values for this trial
        paramNames = self.data.parameterNames[self.state.blockNum]
        paramValues = self.data.parameterBlocks[self.state.blockNum][self.state.currentTrialInBlock]
        
        return dict(zip(paramNames, paramValues))    
    
    def getParameterBlock(self):
        '''
        Get a set of parameters to run over, will
        produce a random ordering of that parameter
        '''
        paramNames = [self.settings.name]
        # turn the list into a list of tuples to be 
        # compatible with tuples of parameters
        # from above and then shuffle
        parameterBlock = [(v,) for v in self.settings.validValues]
        self._rng.shuffle(parameterBlock)
        
        # return the block
        return (paramNames, parameterBlock)

    def startBlock(self):
        '''
        Start a block.
        '''
        # get randomization of parameters
        (paramNames, parameterBlock) = self.getParameterBlock()
        
        # append to the data lists
        self.data.parameterNames.append(paramNames)
        self.data.parameterBlocks.append(parameterBlock)
        self.data.blockLengths.append(len(parameterBlock))
        
        # increment block number and reset trial in block
        self.state.blockNum += 1
        self.state.currentTrialInBlock = 0
        
        # display block information        
        print(f"Block {self.state.blockNum+1}: {len(parameterBlock)} trials randomized over: {paramNames}")

    def print(self):
        """
        Print the details of the pglParameter instance.
        """
        print(f"(pglParameter) Name: {self.settings.name}")
        print(f"Valid Values: {self.settings.validValues}")
        print(f"Description: {self.settings.description}")
        print(f"Random Seed: {self.settings.randomSeed}")
        
        #print values for each block
        for blockNum, (paramNames, parameterBlock) in enumerate(zip(self.data.parameterNames, self.data.parameterBlocks)):
            print(f"Block {blockNum+1}: {len(parameterBlock)} trials randomized over: {paramNames}")
     
    def save(self, parameterDir='.'):
        '''
        Save the parameter settings, state and data.         
        '''
        import traceback
        # Create the directory to save data into
        
        try:
            dataDir = Path(parameterDir) / f"{self.settings.name}"
            dataDir.mkdir(parents=True, exist_ok=True)    
        except Exception as e:
            traceback.print_exc()  # Show full traceback
            print(f"(pglParameter:save) ❌ Could not create data directory {parameterDir}: {e}")
            return

        # give user feedback where things are being saved
        print(f"(pglParameter:save) Saving parameter {self.settings.name} to: {dataDir}")
        
        # save random number generator state
        self.state.randomNumberGeneratorState = self._rng.bit_generator.state
        
        # save settings, state and data
        self.settings.save(dataDir / "settings.json")
        self.state.save(dataDir / "state.json")
        self.data.save(dataDir / "data.json")
    
    def load(self, parameterDir='.'):
        '''
        Load the parameter settings, state and data.         
        '''
        # Create the directory to load data from
        try:
            dataDir = Path(parameterDir) / f"{self.settings.name}"
            if not dataDir.exists():
                raise FileNotFoundError(f"Data directory {dataDir} does not exist.")
        except Exception as e:
            print(f"(pglParameter:load) ❌ Could not access data directory {dataDir}: {e}")
            return
        
        # give user feedback where things are being loaded from
        print(f"(pglParameter:load) Loading parameter {self.settings.name} from: {dataDir}")
        
        # load settings, state and data
        self.settings.load(dataDir / "settings.json")
        self.state.load(dataDir / "state.json")
        self.data.load(dataDir / "data.json")
        
        # update random number generator state
        self._rng.__setstate__(self.state.randomSeed)

##########################
# Parameter block class
##########################
class pglParameterBlock(pglParameter):
    '''
    Class representing a block of parameters in the experiment.
    This is a subclass of pglParameter which allows you to group
    multiple parameters together into a single block.
    '''
    def __init__(self, parameters: list, description: str="", randomSeed=None):
        '''
        Initialize the parameter block.
        
        Args:
            name (str): The name of the parameter block.
            parameters (list): A list of pglParameter instances to include in the block.
            description (str, optional): Description string describing the parameter block.
            randomSeed (int, optional): Seed for random number generation. If None, a random seed is used.

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
        
        # set random seed for the block
        self.setRandomSeed(randomSeed)

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
        self.rng.shuffle(self.allParameterValues)
        return (self.paramNames, self.allParameterValues)


##############################################
# Settings for pglParameter
##############################################
@dataclass
class pglParameterSettings(pglSerialize):
    name: str = ""
    validValues: List = field(default_factory=list)
    description: str = ""
    randomSeed: int = 0
    
##############################################
# Data for pglParameter
##############################################
@dataclass
class pglParameterData(pglSerialize):
    # Lists to store all blocks
    parameterNames: List[List[str]] = field(default_factory=list)
    parameterBlocks: List[List[Tuple]] = field(default_factory=list)
    blockLengths: List[int] = field(default_factory=list)
    
    def __repr__(self):
        return f"pglParameterData({len(self.parameterBlocks)} blocks, {sum(self.blockLengths)} total trials)"        
    
##############################################
# State for pglParameter
##############################################
@dataclass
class pglParameterState(pglSerialize):
    currentTrial: int = -1
    currentTrialInBlock: int = -1
    blockNum: int = -1
    randomNumberGeneratorState: int = 0
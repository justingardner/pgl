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
import json
from pathlib import Path
from .pglSerialize import pglSerialize, pglGetAllSubclasses
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
        
        # save the className
        self.settings.className = self.__class__.__name__
        
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
        return f"pglParameter(name={self.settings.name}, validValues={self.settings.validValues}, description={self.settings.description})"

    def __str__(self):
        # display description
        if self.settings.description == "":
            description = ""
        else:
            description = f"# {self.settings.description} #\n"

        # display full string
        return f"{description}{self.settings.name}: {self.settings.validValues}"

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
    
    @classmethod
    def from_file(cls, parameterDir):
        '''
        Instatiates a pglParameter class by loading from parameterDir
        used for loading saved parameters.
        
        Note that this will correctly read the class name from settings
        Which allows it to create subclasses. Though to work, the classMap
        has to be updated whenever a new subclass is created that needs
        to be serialized
        '''
        
        # check parameterDir
        try:
            if not parameterDir.exists():
                raise FileNotFoundError(f"Data directory {parameterDir} does not exist.")
        except Exception as e:
            print(f"(pglParameter:from_file) ❌ Could not access data directory {parameterDir}: {e}")
            return

        try:
            # peek at settings to see what class to instantiate
            settingsPath = Path(parameterDir) / "settings.json"
            data = json.loads(settingsPath.read_text())
        except Exception as e:
            print(f"(pglParameter) Could not load settings.json from {parameterDir}: {e}")
            return
        
        # get the className
        className = data.get('__class__', 'pglParameter')

        # create hte classMap of all subtypes, this just creates
        # a dictionary of strings and matching class types i.e. {'pglParameter',pglParameter}  
        classMap = {pglParameter.__name__: pglParameter, **pglGetAllSubclasses(pglParameter)}      

        # dispatch to right class
        targetClass = classMap.get(className, pglParameter)
        
        # now instantiate and load the right type
        obj = targetClass.__new__(targetClass)
        
        # call load to load from the directory
        obj.load(Path(parameterDir))
        
        # return the created object
        return obj
        
    def load(self, parameterDir):
        '''
        Load the parameter settings, state and data.         
        '''
        # Create the directory to load data from
        try:
            if not parameterDir.exists():
                raise FileNotFoundError(f"Data directory {parameterDir} does not exist.")
        except Exception as e:
            print(f"(pglParameter:load) ❌ Could not access data directory {parameterDir}: {e}")
            return

        # load settings, state and data
        self.settings = pglParameterSettings.load(parameterDir / "settings.json")
        self.state = pglParameterState.load(parameterDir / "state.json")
        self.data = pglParameterData.load(parameterDir / "data.json")        
        
        # update random number generator state
        self._rng = np.random.default_rng()
        self._rng.__setstate__(self.state.randomNumberGeneratorState)
        
        # give user feedback on load
        print(f"(pglParameter:load) Loaded parameter {self.settings.name} from: {parameterDir}")        

##########################
# Parameter block class
##########################
class pglParameterBlock(pglParameter):
    '''
    Class representing a block of parameters in the experiment.
    This is a subclass of pglParameter which allows you to group
    multiple parameters together into a single block.
    '''
    def __init__(self, parameters: list, name: str="", description: str="", randomSeed=None):
        '''
        Initialize the parameter block.
        
        Args:
            parameters (list): A list of pglParameter instances to include in the block.
            name (str, optional): Name to override default of paramname1_paramname2...
            description (str, optional): Description string describing the parameter block.
            randomSeed (int, optional): Seed for random number generation. If None, a random seed is used.
        '''

        # validate parameters first, since we need them to build validValues
        if not isinstance(parameters, list) or not all(isinstance(p, pglParameter) for p in parameters):
            raise TypeError("(pglParameterBlock) ❌ Error: parameters must be a list of pglParameter instances.")
        
        # build the cartesian product of all parameter values
        self.parameters = parameters
        paramNames = [p.settings.name for p in self.parameters]
        allParameterValues = [p.settings.validValues for p in self.parameters]
        allParameterValues = list(itertools.product(*allParameterValues))

        # join names of parameters, e.g. "direction_coherence"
        if name == "":
            name = "_".join(paramNames)
            
        # call super().__init__ 
        super().__init__(name=name, validValues=allParameterValues,description=description,randomSeed=randomSeed)
        
        # subclass-specific state
        self.paramNames = paramNames

    def getParameterBlock(self):
        '''
        This  will create a block over all of the parameters in the
        list, for example if you have direction and coherence parameters, this would
        calculate all combination of those parameters and return them for the task
        to run as a block of trials.
        '''
        # create a copy of allParameterValues
        block = self.settings.validValues
        # randomly shuffle
        self._rng.shuffle(block)
        # and return
        return (self.paramNames, block)


##############################################
# Settings for pglParameter
##############################################
@dataclass
class pglParameterSettings(pglSerialize):
    name: str = ""
    className: str = "pglParameter" 
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
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
import posixpath
from .pglMessages import pglMessages
from .pglBase import pglBase

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
    def __init__(self, name: str, validValues: list|tuple|np.ndarray, description: str="", blankEvery=None, randomSeed=None):
        '''
        Initialize the parameter.
        
        Args:
            name (str): The name of the parameter.
            validValues (list, optional): A list of valid values for the parameter. 
            blankEvery (int): Insert a blank every n trials. For example, if set to 5 then 
                in every 5 trials (randomly) you will have one blank (value set to None)
                defaults to None which inserts on blanks
            description (str, optional): Description string describing the parameter.
        '''
        # initialize state, data, and settings
        if not hasattr(self, 'settings') or self.settings is None:
            self.settings = pglParameterSettings()
        if not hasattr(self, 'state') or self.state is None:
            self.state = pglParameterState()
        if not hasattr(self, 'data') or self.data is None:
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

        # set blankEvery
        if blankEvery:
            self.settings.blankEvery = blankEvery
            self.state.currentTrialInBlankBlock = -1
            self.setCurrentBlankTrial()
        
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

    def setCurrentBlankTrial(self):
        self.state.currentBlankTrial = self._rng.integers(0, self.settings.blankEvery)

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
        
        # deal with blanks, which can happen every (blankEvery) trials
        if self.settings.blankEvery:
            # update what trial in the blank cycle we are in
            self.state.currentTrialInBlankBlock += 1
            
            # if we are past the end of the block, then reset the block
            if self.state.currentTrialInBlankBlock >= self.settings.blankEvery:
                self.state.currentTrialInBlankBlock = 0
                # and reset which trial is the blank one
                self.setCurrentBlankTrial()
            
            # see if this is the blank trial
            if self.state.currentBlankTrial == self.state.currentTrialInBlankBlock:
                # save trial number in data
                self.data.blankTrials.append(self.state.currentTrial)
                # reset currentTrialInBlock, since we are not doing a blank and not advancing in block
                self.state.currentTrialInBlock -= 1
                # if so, set the parameters to None and return
                paramNames = self.data.parameterNames[self.state.blockNum]
                return dict.fromkeys(paramNames)     

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
            
        if self.settings.blankEvery:
            print(f"BlankTrials: {self.data.blankTrials}")
     
    def save(self, parameterDir='.', filesystem=None):
        '''
        Save the parameter settings, state and data.         
        '''
        import traceback
        # validate filesystem
        filesystem, parameterDir, _ = pglBase.validateFilesystem(filesystem, parameterDir)

        # Create the directory to save data into
        try:
            dataDir = posixpath.join(str(parameterDir), self.settings.name)
            filesystem.makedirs(dataDir, exist_ok=True)
        except Exception as e:
            traceback.print_exc()  # Show full traceback
            print(f"(pglParameter:save) ❌ Could not create data directory {parameterDir}: {e}")
            return        

        # give user feedback where things are being saved
        print(f"(pglParameter:save) Saving parameter {self.settings.name} to: {dataDir}")
        
        # save random number generator state
        self.state.randomNumberGeneratorState = self._rng.bit_generator.state
        
        # save settings, state and data
        self.settings.save(dataDir / "settings.json", filesystem=filesystem)
        self.state.save(dataDir / "state.json", filesystem=filesystem)
        self.data.save(dataDir / "data.json", filesystem=filesystem)
    
    @classmethod
    def from_file(cls, parameterDir, filesystem=None):
        '''
        Instatiates a pglParameter class by loading from parameterDir
        used for loading saved parameters.
        
        Note that this will correctly read the class name from settings
        Which allows it to create subclasses. Though to work, the classMap
        has to be updated whenever a new subclass is created that needs
        to be serialized
        '''

        # validate filesystem
        filesystem, parameterDir, _ = pglBase.validateFilesystem(filesystem, parameterDir)

        import posixpath

        # check parameterDir
        try:
            if not filesystem.exists(str(parameterDir)):
                raise FileNotFoundError(f"Data directory {parameterDir} does not exist.")
        except Exception as e:
            print(f"(pglParameter:from_file) ❌ Could not access data directory {parameterDir}: {e}")
            return

        try:
            # peek at settings to see what class to instantiate
            settingsPath = posixpath.join(str(parameterDir), "settings.json")
            with filesystem.open(settingsPath, "r") as f:
                data = json.loads(f.read())
        except Exception as e:
            print(f"(pglParameter) Could not load settings.json from {parameterDir}: {e}")
            return        
        
        # get the className
        className = data.get('className', 'pglParameter')
        
        # create the instance
        obj = cls.createClassInstance(className)
        
        # call load to load from the directory
        obj.load(parameterDir, filesystem=filesystem)
        
        # return the created object
        return obj

    @staticmethod
    def createClassInstance(className):
        '''
            create an instance of pglParameter with the correct class give in className
        '''
        # create tte classMap of all subtypes, this just creates
        # a dictionary of strings and matching class types i.e. {'pglParameter',pglParameter}  
        classMap = {pglParameter.__name__: pglParameter, **pglGetAllSubclasses(pglParameter)}      

        # dispatch to right class
        targetClass = classMap.get(className, pglParameter)

        # now instantiate and load the right type
        obj = targetClass.__new__(targetClass)
        return obj

    def load(self, parameterDir, filesystem=None):
        '''
        Load the parameter settings, state and data.         
        '''
        
        # validate filesystem
        filesystem, parameterDir, _ = pglBase.validateFilesystem(filesystem, parameterDir)

        # Create the directory to load data from
        try:
            if not filesystem.exists(str(parameterDir)):
                raise FileNotFoundError(f"Data directory {parameterDir} does not exist.")
        except Exception as e:
            pglMessages.warning(f"Could not access data directory {parameterDir}: {e}")
            return
        
        self.settings = pglParameterSettings.load(posixpath.join(str(parameterDir), "settings.json"), filesystem=filesystem)
        self.state = pglParameterState.load(posixpath.join(str(parameterDir), "state.json"), filesystem=filesystem)
        self.data = pglParameterData.load(posixpath.join(str(parameterDir), "data.json"), filesystem=filesystem)
        
        # update random number generator state
        self._rng = np.random.default_rng()
        self._rng.__setstate__(self.state.randomNumberGeneratorState)
        
        # give user feedback on load
        print(f"(pglParameter:load) Loaded parameter {self.settings.name} from: {parameterDir}")        
    
    @classmethod
    def from_settings_state_data(cls, settings, state, data):
        '''
        Instatiates a pglParameter class given settings state and data.
        '''
        obj = cls.createClassInstance(settings.className)
        
        # set data classes
        obj.settings = settings
        obj.state = state
        obj.data = data
        
        # update random number generator state
        obj._rng = np.random.default_rng()
        obj._rng.__setstate__(obj.state.randomNumberGeneratorState)

        return(obj)
##########################
# Parameter batch class
##########################
class pglParameterBatch(pglParameter):
    '''
    This works like a regular parameter, but for each trial it gives a batch of values
    So, for example, say you have validValues = [0,1,2,3,4,5,6,7,8]
    and you have batchSize = 3
    then on each trial you will get 3 random values until all values are visited. e.g.:
    trial 1: [3,7,2]
    trial 2: [4,1,0]
    trial 3: [8,5,6]
    ...
    NOTE: This is designed to work with pglParameterNestedBlock, but not for pglParameterBlock
    '''
    def __init__(self, name: str, validValues: list|tuple|np.ndarray, batchSize: int, description: str="", blankEvery=None, randomSeed=None):
        '''
        Initialize the parameter.
        
        Args:
            name (str): The name of the parameter.
            batchSize (int): Size of the batch of parameters that will be provided on each trial
              The number of validValues must be evenly divisible by batchSize
            validValues (list, optional): A list of valid values for the parameter. 
            description (str, optional): Description string describing the parameter.
        '''
        # specialized setting for this
        self.settings = pglParameterSettingsBatch()
        
        # call super init
        super().__init__(name=name, validValues=validValues, description=description, blankEvery=blankEvery, randomSeed=randomSeed)

        # validate the batchSize
        self.settings.batchSize = batchSize
        if len(validValues) % batchSize != 0:
            pglMessages.warning(f"batchSize ({batchSize}) must evenly divide the number of validValues ({len(validValues)})")
            raise ValueError()

    def getParameterBlock(self):
        '''
        Get a set of parameters to run over, will
        produce a random ordering of that parameter
        in batches of batchSize values per trial.
        '''
        paramNames = [self.settings.name]
        
        # shuffle the valid values
        parameterBlock = list(self.settings.validValues)
        self._rng.shuffle(parameterBlock)
        
        # split into batches of batchSize
        batches = [tuple(parameterBlock[i:i+self.settings.batchSize]) 
                for i in range(0, len(parameterBlock), self.settings.batchSize)]
        
        # wrap in list of tuples to be compatible with other parameter blocks
        batches = [(b,) for b in batches]
        
        return (paramNames, batches)        
        
##########################
# Parameter block class
##########################
class pglParameterBlock(pglParameter):
    '''
    Class representing a block of parameters in the experiment.
    This is a subclass of pglParameter which allows you to group
    multiple parameters together into a single block.
    '''
    def __init__(self, parameters: list, name: str="", description: str="", blankEvery=None, randomSeed=None):
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

        # keep the list of pglParameters
        self.settings = pglParameterSettingsBlock()        
        self.settings.parameters = parameters
        self.settings.parameterNames = [p.settings.name for p in self.settings.parameters]

        # build the cartesian product of all parameter values
        allParameterValues = [p.settings.validValues for p in self.settings.parameters]
        allParameterValues = list(itertools.product(*allParameterValues))

        # join names of parameters, e.g. "direction_coherence"
        if name == "":
            name = "_".join(self.settings.parameterNames)
            
        # call super init
        super().__init__(name=name, validValues=allParameterValues,description=description,blankEvery=blankEvery, randomSeed=randomSeed)
        
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
        return (self.settings.parameterNames, block)
    
    def load(self, parameterDir,filesystem=None):
        '''
        Load function handles recreation of self.settings.parameters list
        As these will just get saved out as a dicts because they are not
        derived from pglSerialize.
        '''
        # call super
        super().load(parameterDir, filesystem=filesystem)

        # now for each parameter recreate the structure
        parameters = []
        for i, p in enumerate(self.settings.parameters):
            # get the saved structures
            settings = p['settings']
            state = p['state']
            data = p['data']

            # and recreate the parameter
            parameters.append(pglParameter.from_settings_state_data(settings,state,data))
        
        # now reset the parameters
        self.settings.parameters = parameters
##########################
# Parameter nested block class
##########################
class pglParameterNestedBlock(pglParameterBlock):
    '''
    Class representing a nested block of parameters in the experiment.
    This is a subclass of pglParameter which allows you to do nested blocking
    of parameters. 
    
    e.g. Given param1 = [1, 2, 3] and param2 = [A, B] an unrandomized sequence looks like:

    Trial  1: param1=1, param2=A  -|
    Trial  2: param1=1, param2=A   | param1 block 1 (param2 block 1 for param1=1)
    Trial  3: param1=1, param2=A  -|
    Trial  4: param1=2, param2=B  -|
    Trial  5: param1=2, param2=B   | param1 block 2 (param2 block 1 for param1=2)
    Trial  6: param1=2, param2=B  -|
    Trial  7: param1=3, param2=A  -|
    Trial  8: param1=3, param2=A   | param1 block 3 (param2 block 1 for param1=3)
    Trial  9: param1=3, param2=A  -|
    Trial 10: param1=1, param2=B  -|
    Trial 11: param1=1, param2=B   | param1 block 4 (param2 block 2 for param1=1)
    Trial 12: param1=1, param2=B  -|

    Note that param1=1 has seen param2=A then param2=B across its two blocks,
    completing one full param2 cycle. param1=2 and param1=3 each track their
    own independent param2 cycles. With randomization, the order of blocks and
    the assignment of param2 values within each param1's cycle are both shuffled.
    
    Here is a randomized example:

    Trial  1: param1 = 1, param2 = B  -|
    Trial  2: param1 = 3, param2 = A   | Param1 block 1 (param2 block 1)
    Trial  3: param1 = 2, param2 = B  -|
    Trial  4: param1 = 2, param2 = A  -|
    Trial  5: param1 = 3, param2 = B   | Param1 block 2 (param2 block 2)
    Trial  6: param1 = 1, param2 = A  -|
    Trial  7: param1 = 3, param2 = B  -|
    Trial  8: param1 = 1, param2 = B   | Param1 block 3 (param2 block 3)
    Trial  9: param1 = 2, param2 = A  -|
    Trial 10: param1 = 2, param2 = B  -|
    Trial 11: param1 = 3, param2 = A   | Param 1 block 4 (param2 block 4)
    Trial 12: param1 = 1, param2 = A  -|
    ...

    Note how every 3 trials param1 will go through all of its values. Every 6 trials
    each value of param1 will see each value of param2    
    '''
    def __init__(self, parameters: list, name: str="", description: str="", blankEvery=None, randomSeed=None):
        # validate parameters, in particular the parameters can only include pglParameterBatch
        # if it is in the last place of the list 
        for i, p in enumerate(parameters):
            if isinstance(p, pglParameterBatch) and i != len(parameters) - 1:
                raise TypeError(f"(pglParameterNestedBlock) ❌ Error: pglParameterBatch can only be the last parameter in the list, but found one at position {i}.")
        
        # init using super
        super().__init__(parameters=parameters, name=name, description=description, blankEvery=blankEvery, randomSeed=randomSeed)
        
    def getParameterBlock(self):
        # recursive function used to build the nested blocks
        def buildNestedBlock(parameters):
            # Base case: only one parameter left, just return its block
            if len(parameters) == 1:
                _, block = parameters[0].getParameterBlock()
                return block
            
            # Recursively build rest blocks for each value of the top parameter
            restBlocks = {}
            for val in parameters[0].settings.validValues:
                restBlocks[val] = buildNestedBlock(parameters[1:])
            
            # Generate len(restBlocks[val]) blocks of the top parameter
            nBlocks = len(restBlocks[parameters[0].settings.validValues[0]])
            topBlock = []
            for _ in range(nBlocks):
                _, block = parameters[0].getParameterBlock()
                topBlock.extend(block)
            
            # Combine topBlock with restBlocks using a pointer per val
            pointers = {val: 0 for val in parameters[0].settings.validValues}
            result = []
            for (val,) in topBlock:
                restTuple = restBlocks[val][pointers[val]]
                pointers[val] += 1
                result.append((val,) + restTuple)
            
            return result
        
        # generate the block, using the recursive function
        block = buildNestedBlock(self.settings.parameters)
        return (self.settings.parameterNames, block)

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
    blankEvery: int | None = None

@dataclass
class pglParameterSettingsBatch(pglParameterSettings):
    batchSize: int = 1

@dataclass
class pglParameterSettingsBlock(pglParameterSettings):
    parameters: List[pglParameter] = field(default_factory=list)
    parameterNames: List[str] = field(default_factory=list)
    
##############################################
# Data for pglParameter
##############################################
@dataclass
class pglParameterData(pglSerialize):
    # Lists to store all blocks
    parameterNames: List[List[str]] = field(default_factory=list)
    parameterBlocks: List[List[Tuple]] = field(default_factory=list)
    blockLengths: List[int] = field(default_factory=list)
    blankTrials: List[int] = field(default_factory=list)
    
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
    currentTrialInBlankEvery = -1
    currentBlankTrial = -1
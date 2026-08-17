################################################################
#   filename: pglPipeline.py
#    purpose: Pipeline 
#         by: JLG
#       date: Aug 1, 2026
################################################################

#############
# Import
#############
from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from .pglData import pglDataMatrix
from datetime import datetime
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from enum import Enum, auto
import fsspec
from fsspec import AbstractFileSystem
import posixpath
from pathlib import Path
import re
from .pglExperiment import pglExperimentData, pglExperimentBase, pglExperimentSettings, pglTaskBase, pglEventTrial, pglEventSegment
from .pglBase import pglBase
from .pglSettings import pglSettings
from .pglDialog import pglDialogs
from .pglParameter import pglParameter, pglParameterBlock
from typing import Annotated
import numpy as np
import matplotlib.pyplot as plt
from .pglSettings import pglSettingsManager


########################
# action status
########################
class pglActionStatus(Enum):
    INITIALIZED = auto()
    VALIDATED = auto()
    CONFIGURED = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    
########################
# class pglAction
########################
class pglAction(pglTraitSettings):
    name = Unicode("", help="Name of action")
    status = Instance(pglActionStatus, help="action status")
    error = Instance(Exception, allow_none=True, default_value=None, help="error")
    
    # settings for the action, required to be a pglTraitSettings. Subclass sould override this
    settings = Instance(pglTraitSettings, allow_none=True, default_value=None, help='Settings for this action')
    
    # init action
    #-----------------
    def __init__(self):
        '''
        initialize the action
        '''
        self.name = self.__class__.__name__
        self.status = pglActionStatus.INITIALIZED
        self.error: Exception | None = None

        self.version = "0.0"
        
        self.inputData = {}
        self.outputData = {}

    def configure(self) -> None:
        # set status
        self.status.pglActionStatus.CONFIGURED
        
    def run(self):
        pass
    
    def print(self):
        '''
        print the action
        '''
        print(f"Action: {self.name} status: {self.status.name}")
        
##########################
# pglPipeline
##########################
class pglPipeline(pglAction):
    # init
    #--------------------------------
    def __init__(self):
        super().__init__()
        pass
    
##################################
# pglRun
##################################
class pglRun(pglExperimentBase):
    
    # filesystem, name and prefix for where the session is loaded from
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    fullDataPath = Unicode(allow_none=True, default_value="", help="Full path to data", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    
    # These will be lazy-loaded as needed
    _experimentSettings = Instance(pglExperimentSettings, allow_none=True, default_value=None, help="settings of the experiemnt")
    _settings = Instance(pglSettings, allow_none=True, default_value=None, help="settings that this experiment was run with")
    _data = Instance(pglExperimentData, allow_none=True, default_value=None, help="data from experiemnt")
    _tasks = List(Instance(pglTaskBase), allow_none=True, default_value=None, help="tasks from experiment")
    
    ##########################
    # Lazy-loaded properties
    ##########################
    @property
    def experimentSettings(self):
        '''Experiment settings, loaded from disk on first access.'''
        if self._experimentSettings is None:
            pglMessages.message(f"Loading experiment settings for: {self.filesystemPrefix}/{self.fullDataPath}")
            filesystem, fullDataPath, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            self._experimentSettings = pglExperimentSettings.load(filename=Path(fullDataPath) / "experimentSettings", filesystem=filesystem)
        return self._experimentSettings

    @experimentSettings.setter
    def experimentSettings(self, value):
        self._experimentSettings = value

    @property
    def settings(self):
        '''Settings the experiment was run with, loaded on first access.'''
        if self._settings is None:
            pglMessages.message(f"Loading settings for: {self.filesystemPrefix}/{self.fullDataPath}")
            filesystem, fullDataPath, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            self._settings = pglSettings.load(filename=Path(fullDataPath) / "settings", filesystem=filesystem)
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    @property
    def data(self):
        '''Experiment data, loaded from disk on first access.'''
        if self._data is None:
            pglMessages.message(f"Loading data for: {self.filesystemPrefix}/{self.fullDataPath}")
            filesystem, fullDataPath, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            self._data = pglExperimentData.load(filename=Path(fullDataPath) / "data", filesystem=filesystem)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def tasks(self):
        '''Experiment tasks'''
        if self._tasks is None:
            pglMessages.message(f"Loading tasks for: {self.filesystemPrefix}/{self.fullDataPath}")
            filesystem, fullDataPath, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            taskNames = self.experimentSettings.tasks
            self._tasks = []
            for taskName in taskNames:
                print(f"taskName: {taskName}")
                self._tasks.append(pglTaskBase.load(taskDir=posixpath.join(str(fullDataPath), taskName), filesystem=filesystem))
        return self._tasks

    @tasks.setter
    def tasks(self, value):
        self._tasks = value
        
    def getTask(self, taskName):
        '''
        get a named task
        '''
        for task in self.tasks:
            if task.settings.taskSaveName == taskName:
                return task
        return None

    def __init__(self, fullDataPath=None, filesystem=None, filesystemPrefix=None):
        '''
        Initialize the pglRun class
        
        Args:
            dataPath: The directory where the run is saved
        '''
        # init super
        super().__init__()

        # keep the path and filesystem
        if filesystem is not None and fullDataPath is not None:
            self.filesystem, self.fullDataPath, self.filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem,dataPath=fullDataPath,filesystemPrefix=filesystemPrefix)
        
    def getTaskNames(self):
        '''
        Extracts task names from experimentSettings
        '''
        return(", ".join(self.experimentSettings.tasks))
    
    def display(self, ax=None):
        '''
        display plot of the run
        '''
        # display
        try:
            # compute how many axes we need
            nTasks = len(self.tasks)
            fig, _ = plt.subplots(nTasks+1,1,figsize=(12,4*(nTasks+1)), constrained_layout=True)
            
            # display experiment
            self.data.display(ax=fig.axes[0])
            
            # display tasks
            for iTask, task in enumerate(self.tasks):
                task.display(ax=fig.axes[iTask+1])
            
            plt.show()
            
        except Exception as e:
            print(f"error: {e}")
    
    def getTrialsByParameter(self, parameterName: str, taskName: str = None):
        '''
        Extracts trial data grouped by parameterName
        
        Args:
            parameterName (str): Name of parameter to group data by
        
        Returns:
            dictionary with fields
                parameterName (str): Name of parameter that the trials are sorted by
                nParameterValues (int): Number of different parameter values
                parameterValues (list): List of all parameter values
                trialNum: 
                volumeNum:
                trialTime:
        '''
        # figure out what task we are working on
        if taskName is None:
            task = self.tasks[0]
        else:
            # search for the taskName (case insensitive)
            task = next((t for t in self.tasks if t.settings.taskName.lower() == taskName.lower()), None)
            # if not found, check if they meant the taskSaveName
            if task is None:
                task = next((t for t in self.tasks if t.settings.taskSaveName.lower() == taskName.lower()), None)
        
        if task is None:
            print(f"(pglExperimentAnalysis:getTrialsByParameter) ❌ Could not find {taskName} in experiemnt.\nValid tasks are: {' '.join(t.settings.taskName for t in self.tasks)}")
            return None
                
        # gather all the different parameter names
        parameters = task.parameters
        # get all the parameters recursively
        # so that we get all parameters in blocks 
        def collectParameters(parameterList):
            parameters = []
            for p in parameterList:
                if isinstance(p, pglParameterBlock):
                    parameters.extend(collectParameters(p.settings.parameters))
                else:
                    parameters.append(p)
            return parameters
        parameters = collectParameters(parameters)
        
        # get the matching parameter
        parameter = next((p for p in parameters if p.settings.name == parameterName), None)
        if parameter is None:
            print(f"(pglExperimentAnalysis:getTrialsByParameter) ❌ Could not find '{parameterName}' in parameters {[p.settings.name for p in parameters]}")
            return
        
        # initialize the list of lists for volumes by conditions        
        validValues = parameter.settings.validValues
        volumes = [[] for _ in range(len(validValues))]
        startTimes = [[] for _ in range(len(validValues))]
        trialNums = [[] for _ in range(len(validValues))]
        nTrials = [0 for _ in range(len(validValues))]
        nTrialsTotal = 0
        
        # loop over trials, collecting the params dictionary for each trial
        for iTrial, params in enumerate(task.data.params):
            # find matching trial event
            trialEvent = next((event for event in task.data.events if isinstance(event, pglEventTrial) and event.trialNum == iTrial), None)
            
            # get the trials tart time and volume
            trialStart = trialEvent.timestamp - task.data.startTime if trialEvent else "No trial event found"
            trialVolume = self.getNearestVolumeTrigger(trialEvent)

            # if we found a volume trigger
            if trialVolume is not None:
                # get the value that was set for this trial
                trialValue = params.get(parameter.settings.name,None)
                # if it matches the valid values
                if trialValue in validValues:
                    # get the index
                    conditionIndex = validValues.index(trialValue)
                    
                    # and populate arrays with data
                    volumes[conditionIndex].append(trialVolume)
                    startTimes[conditionIndex].append(trialStart)
                    trialNums[conditionIndex].append(iTrial+1)
                    nTrials[conditionIndex] += 1
                    nTrialsTotal += 1
        
        # pack everything up
        return pglTrialsByParameter(
            parameterName=parameter.settings.name,
            parameterValues=validValues,
            parameter=parameter,
            nTrialsTotal=nTrialsTotal,
            volumes=volumes,
            startTimes=startTimes,
            trialNums=trialNums,
            nTrials=nTrials
        )

##################################
# pglTrialsByParameter
##################################
class pglTrialsByParameter(pglTraitSettings):
    parameterName = Unicode(help="Name of parameter that was used to sort trials by")
    parameterValues = List(help="List of all values that the parameter can take")
    parameter = Instance(pglParameter, help="The pglParameter instance of the parameter")
    nTrialsTotal = Int(help="total number of trials")
    volumes = List(List(Int()),help="A list of lists of volumes, one list for each value of the parameter")
    startTimes = List(List(Float()),help="A list of lists of times, one list for each value of the parameter")
    trialNums = List(List(Int()),help="A list of lists of trial volumes, one list for each value of the parameter")
    nTrials = List(Int(),help="A list of number of trials, one list for each value of the parameter")
    
           
##################################
# pglSession
##################################
class pglSession(pglTraitSettings):

    # filesystem, name and prefix for where the session is loaded from
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    
    # List of all runs
    runs = List(Instance(pglRun), allow_none=True, help="List of all runs")
    def __init__(self, filesystem=None, filesystemPrefix='', runList=[]):
        '''
        Init
        
        Args:
            fullDataPath (str): path to data for session
            filesystem: filesystem where path exists (None for local)
            filesystemPrefix: Prefix like ssh://
            runList: List of paths to runs
        '''
        
        self.filesystem = filesystem                
        self.filesystemPrefix = filesystemPrefix
        
        for runPath in runList:
            self.runs.append(pglRun(fullDataPath=runPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix))
    
##################################################################
# pglChooseSession. Base class for walking directory structures.
# implements reading of child directories and putting them in a list
# creating a class around those directories, see pglChoose classes below
##################################################################
class pglChooseLevel(pglTraitSettings):
    '''
    Base class for one level of the dataPath hierarchy (experiment,
    subject, run, ...). Subclasses just declare which class their
    children are; discovery logic itself lives here
    '''
    name = Unicode("", help="Name of this level (experiment name, subjectID, etc.)", visible=False)
    childList = List(Instance(pglTraitSettings), settingsListKey="name", help="List of child levels found under this one")
    # subclasses override this with the class to instantiate for each
    # child directory found; None means this is a leaf level (no
    # further recursion into subdirectories)
    childClass = None
    
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization",visible=False)
    fullDataPath = Unicode(allow_none=True, default_value="", help="Full path to data", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)


    def __init__(self, name="", dataPath="", filesystem=None, filesystemPrefix=None, entries=None):
        super().__init__()

        self.name = name
        self.filesystem, self.dataPath, self.filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix)
        self.childList = self._getChildren(entries) if self.childClass is not None else []

    @classmethod
    def create(cls, name="", dataPath="", filesystem=None, filesystemPrefix=None):
        '''
        Factory method: validates that dataPath qualifies as this
        level (via _isValid), then builds the instance and, for
        non-leaf levels, checks that it actually ended up with at
        least one valid child. A level with no valid children isn't
        considered valid itself (e.g. a subject directory with no
        valid runs isn't really a subject). Returns None if either
        check fails, otherwise returns the fully-built instance.
        '''
        filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix)

        # load all the entries in the directory
        try:
            entries = filesystem.ls(dataPath, detail=True)
        except (FileNotFoundError, OSError):
            return None
        
        # check if the disrecotry is valid (this is an overwriteable function for specific
        # checks like if the directory contains all the files necessary for a run)
        if not cls._isValid(name=name, dataPath=dataPath, filesystem=filesystem, entries=entries):
            return None

        # there are some children create the instance, note that we use the original filesystem
        # and dataPath so that the dataPath can be stored with its filesystem prefix if it has one
        instance = cls(name=name, dataPath=dataPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix, entries=entries)

        # There should be a list of children now (this is what selection is over).
        # So, drop out here if the list is empty. Alternatively, if this is a leaf
        # (i.e. has no childClass) then no check necessary
        if cls.childClass is not None and len(instance.childList) == 0:
            return None

        # return the initialized instance
        return instance

    @classmethod
    def _isValid(cls, name=None, dataPath=None, filesystem=None, entries=None):
        '''
        Subclass-overrideable check for whether dataPath qualifies as
        this level based on its own properties (name pattern, presence
        of a specific file, etc). Default: always valid.
        '''
        return True

    def _getChildren(self, entries):
        '''
        Find all directories directly under dataPath and instantiate
        one childClass instance per directory that passes validation
        (including the "has valid children" check, if applicable).
        '''
    
        if entries is None:
            entries = self.filesystem.ls(self.dataPath, detail=True)

        children = []
        for entry in entries:
            if entry["type"] != "directory":
                continue
            childName = entry["name"].rstrip("/").split("/")[-1]
            child = self.childClass.create(name=childName, dataPath=entry["name"], filesystem=self.filesystem, filesystemPrefix=self.filesystemPrefix)
            if child is not None:
                children.append(child)
        return children        
             
################################################################################
# Each one of these classes sits at one level of the file structure hierarchy
# So they can be used to walk the experiment directory and load runs
################################################################################        
class pglChooseRun(pglChooseLevel):
    # this is the root, so no more recursion beyond this point
    childClass = None

    dataPath = Unicode(allow_none=True, default_value=None, help="Where the data for this run lives", enabled=False)

    _tasks = Unicode(allow_none=True, default_value="", help="Stimulus type used for this run", enabled=False)
    _run = Instance(pglRun, allow_none=True, default_value=None, serialize=False, help="Class representing run data", visible=False)
    
    @property
    def tasks(self):
        '''String representing tasks, lazy-loaded.'''
        if not self._tasks:
            self._tasks = self.run.getTaskNames()
        return self._tasks

    @property
    def run(self):
        '''String representing tasks, lazy-loaded.'''
        if not self._run:
            self.run = pglRun(fullDataPath=self.dataPath, filesystem=self.filesystem, filesystemPrefix=self.filesystemPrefix)   
        return self._run
    
    @run.setter
    def run(self, value):
        self._run = value

    # display
    def display(self, fig=None):
        '''
        display the run
        '''
        self.run.display(fig=fig)

class pglChooseSession(pglChooseLevel):
    childList = List(Instance(pglTraitSettings), settingsListKey="name", traitDisplayName="Select run(s)", multiSelect=True, maxRowsVisible=6, hasPlotButton=True, buttonFunction="display", help="Runs in session dir")
    childClass = pglChooseRun
                
class pglChooseSubject(pglChooseLevel):
    # re-declare childList, so we can give it a proper name
    childList = List(Instance(pglTraitSettings), settingsListKey="name", traitDisplayName="Choose session", help="Sessions in subject dir")
    childClass = pglChooseSession
    
    @classmethod
    def _isValid(cls, name=None, dataPath=None, filesystem=None, entries=None):
        
        # check whether it is a directory of form sXXXXX
        lastDir = Path(dataPath).name
        return bool(re.match(r"^s\d+$", lastDir))
    
class pglChooseExperiment(pglChooseLevel):
    # re-declare childList, so we can give it a proper name
    childList = List(Instance(pglTraitSettings), settingsListKey="name", traitDisplayName="Choose subject", help="Subjects in experiment dir")
    childClass = pglChooseSubject
    
class pglChooseData(pglChooseLevel):
    # re-declare childList, so we can give it a proper name
    childList = List(Instance(pglTraitSettings), settingsListKey="name", traitDisplayName="Choose experiment", help="Experiments in data path")
    childClass = pglChooseExperiment
    
class pglChoose():
    '''
    Class which provides ways to choose experiment directories
    '''
    @classmethod
    def getExperimentPath(cls, fullDataPath=None, settings=None, settingsName=None, experimentName=None, subjectID=None, sessionName=None, runName=None, filesystem=None, filesystemPrefix=None, dataPath=None):
        '''
        get the directory of the experiment. Many ways to call this to make it easy to get the correct experiemnt dir
        
        If you want to browse the full experiments:
        
            # use default settings to find dataDir
            pglChoose.getExperimentPath()
            
            # use settings name to find dataDir:
            pglChoose.getExperimentPath(settingsName='windowed)

            # or, call directly with the setting:
            s = pglSettingsManager.getSettings(settingsName='windowed')
            pglChoose.getExperimentPath(settings=s)
            
            # or, pass in an explicit path
            pglChoose.getExperimentPath(dataPath='/path/to/experiments')

        If you know the exact path:
            pglChoose.getExperimentPath('/data/experimentDir/subjectDir/sessionDir/runDir')
            
        If you want to browse runs for a particular experiment:
            pglChoose.getExperimentPath(experimentName='experimentName')
            
        
        Returns:
            A tuple consisting of:
                (filesystem, fullDataPath, filesystemPrefix)
            where:
                filesystem: fsspec filesystem for the path
                fullDataPath: path within in filesystem
                filesystemPrefix: Any filesystem prefix (e.g. ssh://gru.stanford.edu/) this is NOT needed
                    to access the path, it is returned in case the calling function wants to save it
                    so that the same path can be accessed again
        
        '''
        from .pglBase import pglBase
        if fullDataPath:
            # validate and return
            filesystem, fullDataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=fullDataPath, filesystemPrefix=filesystemPrefix)
            return (filesystem, fullDataPath, filesystemPrefix)
            
        # if not fullDatadir passed in, construct it from arguments
        else:
            if not dataPath: 
                if not settings and not settingsName:
                    # get the default settings
                    settings = pglSettingsManager.getSettings(settingsName=settingsName)
                if settings:
                    # set dataPath to where settings tells us it is
                    dataPath= settings.dataPath

            # expand user
            fullDataPath = Path(dataPath).expanduser()
            
            # now that we have the start of a path, validate the filesystem
            filesystem, fullDataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=fullDataPath, filesystemPrefix=filesystemPrefix)
            if filesystem is None:
                pglMessages.warning("Could not find dataPath: {fullDataPath}")
                return
            
            # add on experiment name
            if experimentName:
                fullDataPath = Path(fullDataPath) / experimentName
                # check that experimentName exists
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Experiment directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on subject experiment namers
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='experimentNames', filesystem=filesystem)
                if filesystem is None: 
                    return None 
                else: 
                    return (filesystem, fullDataPath, filesystemPrefix)
           
            # add a subjectID
            if subjectID:
                fullDataPath = fullDataPath / subjectID
                # check the subjectID 
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Subject directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on subject IDs
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='subjectIDs', filesystem=filesystem)
                if filesystem is None: 
                    return None 
                else: 
                    return (filesystem, fullDataPath, filesystemPrefix)
                
            # add a sessionName
            if sessionName:
                fullDataPath = fullDataPath / sessionName
                # check the sessionName 
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Session directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on session names
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='sessionNames', filesystem=filesystem)
                if filesystem is None: 
                    return None 
                else: 
                    return (filesystem, fullDataPath, filesystemPrefix)
                
            # add a runName
            if runName:
                fullDataPath = fullDataPath / runName
                # check the runName
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Run directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
                else:
                    # choose based on run names
                    (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='runNames', filesystem=filesystem)
                    if filesystem is None: 
                        return None 
                    else: 
                        return (filesystem, fullDataPath, filesystemPrefix)
              
        return (filesystem, fullDataPath, filesystemPrefix)
    
    @ classmethod
    def _chooseDialog(cls, fullDataPath, filesystem=None, chooseLevel=None):
        '''
        Function that will put up a dialog to choose an experiment for loading
        '''
        # put up dialog
        if chooseLevel == 'experimentNames':
            s = pglChooseData(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        elif chooseLevel == 'subjectIDs':
            s = pglChooseExperiment(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        elif chooseLevel == 'sessionNames':
            s = pglChooseSubject(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        elif chooseLevel == 'runNames':
            s = pglChooseSession(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        else:
            pglMessages.warning(f"Unkown choose level: {chooseLevel}")
            return (None, None)
        
        # walk structure to get runs that are selected
        runNames = cls.walkInstances(s)
        if not runNames:
            pglMessages.message("No runs selected")
            return (None, None)
        elif len(runNames)>1:
            pglMessages.message(f"Multiple runs selecting, using {runNames[0]}")

        return (filesystem, runNames[0])

    # walk the structure to get to the leaves (which have runs)        
    @classmethod
    def walkInstances(cls, node, depth=0):
        selectedPaths = []
        childClass = getattr(type(node), "childClass", None)
        if childClass is None:
            # Leaf instance — get the dataPath if it was selected
            if node.isSelected: 
                selectedPaths.append(node.dataPath)
            return selectedPaths

        # childList holds the child instances
        for child in node.childList:
            selectedPaths.extend(cls.walkInstances(child, depth + 1))

        return(selectedPaths)
    
#################################
# class pglActionLoadSession
#################################
class pglActionLoadSessionSettings(pglTraitSettings):
    selectedPaths = List(Unicode(), help="Paths of runs selected for loading")
    filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// which can be set if the files are not local")

class pglActionLoadSession(pglAction):
    
    # settings
    settings = Instance(pglActionLoadSessionSettings, help="settings")
    
    def configure(self, dataPath: str) -> None:
        # have the user choose the session info
        # Fix, fix, fix, How do we get dataPath in here?
        # What do we do if the pass is not valid?
        # how do we make a default plan
        
        # put up traits dialog to have user select the experiments
        s = pglChooseExperiment(dataPath=dataPath)
        s = pglDialogs.traitsDialog(s)

        # walk the structure to get to the leaves (which have runs)        
        def walkInstances(node, depth=0):
            selectedPaths = []
            childClass = getattr(type(node), "childClass", None)
            if childClass is None:
                # Leaf instance — get the dataPath if it was selected
                if node.isSelected: selectedPaths.append(node.dataPath)
                return selectedPaths

            # childList holds the child instances
            for child in node.childList:
                selectedPaths.extend(walkInstances(child, depth + 1))
                
            return(selectedPaths)

        self.settings = pglActionLoadSessionSettings()
        self.settings.selectedPaths = walkInstances(s)
        self.settings.filesystemPrefix = s.filesystemPrefix
    
    def run(self) -> Annotated[pglSession, "session"]:
        # just create the session variable
        session = pglSession(
            filesystemPrefix=self.settings.filesystemPrefix,
            runList = self.settings.selectedPaths
        )
        
        # and return
        return session
        
##################################################################
# class pglActionRecreateExperimentDataFromTasks
##################################################################
from .pglExperiment import pglEventSegment, pglEventVolumeTrigger
class pglActionRecreateExperimentDataFromTasksChooseTaskName(pglTraitSettings):
    taskName = List(Unicode(), default_value=[], help="Tasks in run", visible=False)

class pglActionRecreateExperimentDataFromTasksChooseRun(pglTraitSettings):
    runName = Unicode(help="Name of run", visible=False)
    taskNames = List(Instance(pglActionRecreateExperimentDataFromTasksChooseTaskName), default_value=[], settingsListKey="taskName", traitDisplayName="Select run(s)", multiSelect=True, maxRowsVisible=2, help="Tasks in run")
    
class pglActionRecreateExperimentDataFromTasksSettings(pglTraitSettings):
    TR = Float(1.0, help="The TR that was used for frame acuqistiion")
    nVols = Int(0, help="Number of volumes in acquisition, if set to 0, will create out till end of task")
    runList = List(Instance(pglActionRecreateExperimentDataFromTasksChooseRun), default_value=[], settingsListKey="runName", traitDisplayName="Run", help="run list")
    taskNameList = List(Unicode(), help="List of task names fore each run",visible=False)

class pglActionRecreateExperimentDataFromTasks(pglAction):
    '''
    Fixer for sessions that were run when pglExperimentData was not being saved correctly
    This will recreate startTime, endTime and volume events by examining
    the task data
    '''
    
    settings = Instance(pglActionRecreateExperimentDataFromTasksSettings, allow_none=True, help="settings")
    
    #----------------------------------------
    #########################################
    def configure(self, session: pglSession) -> None:
        '''
        Configure the action, by having the user select the TR and taskName
        
        Args:
            session (pglSession): The session to run on
        '''
        # keep the session as we will need it in run
        self.session = session
        
        # put up settings
        self.settings = pglActionRecreateExperimentDataFromTasksSettings()
        for run in session.runs:
            # append to the list of runs
            chooseRun = pglActionRecreateExperimentDataFromTasksChooseRun()
            chooseRun.runName = Path(run.fullDataPath).name
            self.settings.runList.append(chooseRun)
            # get all the taskNames
            taskNames = run.experimentSettings.tasks
            for iTaskName, taskName in enumerate(taskNames):
                chooseTaskNames = pglActionRecreateExperimentDataFromTasksChooseTaskName()
                chooseTaskNames.taskName = taskName
                if iTaskName == 0:
                    chooseTaskNames.isSelected = True
                # add add to the run list
                self.settings.runList[-1].taskNames.append(chooseTaskNames)
            
        self.settings = pglDialogs.traitsDialog(self.settings)
        if self.settings:
            for iRun, run in enumerate(self.session.runs):
                for taskNames in self.settings.runList[iRun].taskNames:
                    if taskNames.isSelected:
                        self.settings.taskNameList.append(taskNames.taskName[0])
    
    #----------------------------------------
    #########################################
    def run(self) -> Annotated[pglSession, "sessionWithFixedExperimentalData"]:
        '''
        run the fix
        '''
        # for each run
        for iRun, run in enumerate(self.session.runs):
            # get selected task name
            taskName = self.settings.taskNameList[iRun]
            
            # get the selected task
            task = run.getTask(taskName)
            
            if task:
                # check to make sure the experiment started on volume trigger
                if not run.settings.startOnVolumeTrigger:
                    pglMessagaes.warning("Run did not start on volume trigger - alignment of volumes to task is not guaranteed")
                
                # get the start and end time and use that for the experiment settings
                run.data.startTime = task.data.startTime
                run.data.endTime = task.data.endTime
                
                # start making volume trigger events
                volumeTriggerEvents = []
                startTime = task.data.startTime

                # find the next segment that is marked as waitUntilVolumeTrigger
                waitUntilVolumeTriggerSegments = [i for i, value in enumerate(task.settings.waitUntilVolumeTrigger) if value]

                # iterate over segments to find next one which marks a volume trigger
                eventsIterator = iter(task.data.events)
                
                def makeVolumeEvents(startTime, stopTime, TR, currentVolumeNum):
                    # make equaly spaced triggers from triggerStartTime to this time
                    duration = stopTime - startTime

                    nTRs = round(duration / TR)
                    actualTR = duration / nTRs

                    slop = actualTR - TR
                    if abs(slop) > 0.1 * TR:
                        pglMessages.warning(f"Warning for {nTRs} volumes beginning at {currentVolumeNum}: spacing requires {slop:.3f}s of slop ({100 * abs(slop) / TR:.1f}% of TR)", level=1)


                    times = [
                        startTime + i * actualTR
                        for i in range(nTRs + 1)
                    ]

                    # Explicitly pin the endpoints
                    times[0] = startTime
                    times[-1] = stopTime
                    
                    return times
                
                # get the next segment that has waitUntilVolumeTrigger set
                segment = next((e for e in eventsIterator if isinstance(e, pglEventSegment) and (e.segmentNum in waitUntilVolumeTriggerSegments)), None)
                
                # while we find such segments
                volumeTriggers = []
                while segment:
                    # make volume triggers between them
                    volumeTriggers += makeVolumeEvents(startTime, segment.timestamp, self.settings.TR, len(set(volumeTriggers)))
                    # start a new cycle by using this segments timestamp as the next start time                    
                    startTime = segment.timestamp
                    segment = next((e for e in eventsIterator if isinstance(e, pglEventSegment) and (e.segmentNum in waitUntilVolumeTriggerSegments)), None)
                
                if self.settings.nVols > 0:
                    endTime = run.data.startTime + self.settings.nVols * self.settings.TR
                else:
                    endTime = run.data.endTime
                # make the remaining volume triggers to end of experiment
                if endTime - startTime > self.settings.TR:
                    # round to nearest TR
                    endTime = startTime + round((endTime - startTime) / self.settings.TR) * self.settings.TR
                    # create volume triggers
                    volumeTriggers += makeVolumeEvents(startTime, endTime, self.settings.TR, len(set(volumeTriggers)))
                
                # sort and remove duplicates
                volumeTriggers = sorted(set(volumeTriggers))
                
                # clip to desired length
                if self.settings.nVols > 0:
                    volumeTriggers = volumeTriggers[:self.settings.nVols]
                
                # compute some statistics and display
                diff = np.diff(volumeTriggers)
                pglMessages.message(f"nTriggers: {len(volumeTriggers)} Mean: {np.mean(diff):.3f}, SD: {np.std(diff, ddof=1):.3f}")
                
                # clear old events
                run.data.events = [e for e in run.data.events if not isinstance(e, pglEventVolumeTrigger)]
                
                # generate events
                for triggerTime in volumeTriggers:
                    # create the volume trigger event
                    t = pglEventVolumeTrigger()
                    t.timestamp = triggerTime
                    
                    # add it to the event list
                    run.data.events.append(t)    
                
                # sort events
                run.data.events.sort(key=lambda e: e.timestamp)            
            else:
                pglMessages.warning(f"Could not find task {taskName}")
            
            #run.data.print()
        # return session
        return self.session
        
    
##################################
# saves data locally
##################################
class pglActionSave(pglAction):
    '''
    '''
    #----------------------------------------
    #########################################
    def configure(self, session: pglSession | None = None) -> None:
        self.session = session

    #----------------------------------------
    #########################################
    def run(self) -> None:
        if self.session:
            self.session.save()
    


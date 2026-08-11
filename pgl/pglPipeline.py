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
from pathlib import Path
import re
from .pglExperiment import pglExperimentData, pglExperimentBase, pglExperimentSettings
from .pglBase import pglBase
from .pglSettings import pglSettings


##########################
# pglDataPort
##########################
class pglDataPort():

    def __init__(self, name, dataType, optional=False):
        if not (isinstance(dataType, type) and issubclass(dataType, pglDataMatrix)):
            raise TypeError(f"dataType must be a subclass of pglDataMtrix, got {dataType!r}")
        
        self.name = name
        self.dataType = dataType
        self.optional = optional
        
##########################
# pglPortList
##########################
class pglPortList:
    '''
    Ordered collection of pglDataPort, addressable both positionally
    (like a list) and by name (like a dict), where the name comes
    from the port itself rather than an external key.
    '''
    def __init__(self, ports=None):
        self._ports: list[pglDataPort] = []
        if ports:
            for p in ports:
                self.append(p)

    def append(self, port: 'pglDataPort'):
        if not port.name:
            raise ValueError("pglDataPort must have a name before being added")
        if port.name in self:
            raise ValueError(f"Duplicate port name: {port.name!r}")
        self._ports.append(port)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ports[key]
        for p in self._ports:
            if p.name == key:
                return p
        raise KeyError(key)

    def __contains__(self, key):
        if isinstance(key, str):
            return any(p.name == key for p in self._ports)
        return key in self._ports

    def __iter__(self):
        return iter(self._ports)

    def __len__(self):
        return len(self._ports)

    def keys(self):
        return [p.name for p in self._ports]

    def items(self):
        return [(p.name, p) for p in self._ports]
 
########################
# action status
########################
class pglActionStatus(Enum):
    INITIALIZED = auto()
    CONFIGURED = auto()
    VALIDATED = auto()
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
    
    # Dict of inputs to the action, where each entry is a string name key and a pglDataPort value
    # Subclasses should override with specific pglData subclases for specificity.
    inputPorts: pglPortList = None

    # Dict of inputs to the action, where each entry is a string name key and a pglDataPort value
    # Subclasses should override with specific pglData subclases for specificity.
    outputPorts: pglPortList = None
    
    # settings for the action, required to be a pglTraitSettings. Subclass sould override this
    settings = Instance(pglTraitSettings, help='Settings for this action')
    
    # init action
    #-----------------
    def __init__(self):
        '''
        initialize the action
        '''
        self.name = self.__class__.__name__
        self.status = pglActionStatus.INITIALIZED
        self.error: Exception | None = None

        self.version = 0.0
        
        self.inputData = {}
        self.outputData = {}

    def configure(self):
        pass
        
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
    
#################################
# runs
#################################
class pglRunInfo(pglTraitSettings):
    name = Unicode("", help="Name in the form of: experimentName subjectID startTime")
    experimentName = Unicode("", help="Name of experiment")
    subjectID = Unicode("", "SubjectID should be of form S0000")
    date = Instance(datetime, help="Date and time of run")
    taskNames = List(Unicode, help="Names of the tasks run")    
class pglRunData(pglDataMatrix):
    pass

#################################
# Sessions
#################################
class pglSessionInfo(pglTraitSettings):
    name = Unicode("", help="Name in the form of: experimentName subjectID startTime")
    experimentName = Unicode("", help="Name of experiment")
    subjectID = Unicode("", "SubjectID should be of form S0000")
    date = Instance(datetime, help="Date of experiment")
    runs = List(Instance(pglRunInfo), help="Runs in session")

class pglSessionData(pglDataMatrix):
    pass

#################################
# Datasets
#################################
class pglDatasetInfo(pglTraitSettings):
    name = Unicode("", help="Name in the form of: experimentName subjectID startTime")
    experimentName = Unicode("", help="Name of experiment")
    subjectIDs = List(Unicode, help="List of all subjectIDs")
    sessions = List(Instance(pglSessionInfo), help="List of sessions in the dataset")
class pglDatasetData(pglDataMatrix):
    pass

#################################
# pglChooseSession
#################################
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

##################################
# pglRun
##################################
class pglRun(pglExperimentBase):
    
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    fullDataPath = Unicode(allow_none=True, default_value="", help="Full path to data", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    
    # These will be lazy-loaded as needed
    _experimentSettings = Instance(pglExperimentSettings, allow_none=True, default_value=None, help="settings of the experiemnt")
    _settings = Instance(pglSettings, allow_none=True, default_value=None, help="settings that this experiment was run with")
    _data = Instance(pglExperimentData, allow_none=True, default_value=None, help="data from experiemnt")
    
    ##########################
    # Lazy-loaded properties
    ##########################
    @property
    def experimentSettings(self):
        '''Experiment settings, loaded from disk on first access.'''
        if self._experimentSettings is None:
            filesystem, fullDataPath = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            self._experimentSettings = pglExperimentSettings.load(filename=Path(fullDataPath) / "experimentSettings", filesystem=filesystem)
        return self._experimentSettings

    @experimentSettings.setter
    def experimentSettings(self, value):
        self._experimentSettings = value

    @property
    def settings(self):
        '''Settings the experiment was run with, loaded on first access.'''
        if self._settings is None:
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
            pglMessages.message(f"Loading {self.filesystemPrefix}/{self.fullDataPath}")
            filesystem, fullDataPath, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            self._data = pglExperimentData.load(filename=Path(fullDataPath) / "data", filesystem=filesystem)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    
    def __init__(self, fullDataPath, filesystem, filesystemPrefix):
        '''
        Initialize the pglRun class
        
        Args:
            dataPath: The directory where the run is saved
        '''
        # init super
        super().__init__()

        # keep the path and filesystem
        self.filesystem, self.fullDataPath, self.filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem,dataPath=fullDataPath,filesystemPrefix=filesystemPrefix)
        
    def getTaskNames(self):
        '''
        Extracts task names from experimentSettings
        '''
        
        taskString = ", ".join(self.experimentSettings.tasks)
        
        # return taskString
        return taskString
    
    def display(self, fig=None):
        '''
        display plot of the run
        '''
        # display
        try:
            self.data.display(fig=fig)
            fig.suptitle(f"{self.fullDataPath}")
        except Exception as e:
            print(f"error: {e}")
             
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

################################################################################
# # Each one of these classes sits at one level of the file structure hierarchy
################################################################################        
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
    
class pglChoose(pglChooseLevel):
    # re-declare childList, so we can give it a proper name
    childList = List(Instance(pglTraitSettings), settingsListKey="name", traitDisplayName="Choose experiment", help="Experiments in data path")
    childClass = pglChooseExperiment
    
#################################
# class pglActionLoadSession
#################################
class pglActionLoadSession(pglAction):
    
    # data input / output contract
    inputPorts = pglPortList()
    outputPorts = pglPortList([
        pglDataPort("session", pglSessionData, optional=False)
    ])
    
    def __init__(self):
        super().__init__()
    
    def configure(self):
        # have the user choose the session info
        pass
    
    def run(self):
        self.outputData = {'session': pglSessionData()}
        
    

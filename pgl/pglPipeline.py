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
    Base class for one level of the dataDir hierarchy (experiment,
    subject, run, ...). Subclasses just declare which class their
    children are; discovery logic itself lives here
    '''
    name = Unicode("", help="Name of this level (experiment name, subjectID, etc.)")
    childList = List(Instance(pglTraitSettings), settingsListKey="name", help="List of child levels found under this one")
    
    # subclasses override this with the class to instantiate for each
    # child directory found; None means this is a leaf level (no
    # further recursion into subdirectories)
    childClass = None

    def __init__(self, name="", dataDir="", filesystem=None):
        super().__init__()

        self.name = name
        self.dataDir = dataDir
        self.filesystem = filesystem if filesystem is not None else fsspec.filesystem("file")

        self.childList = self._getChildren() if self.childClass is not None else []

    def _getChildren(self):
        '''
        Find all directories directly under dataDir and instantiate
        one childClass instance per directory found.
        '''
        children = []
        for entry in self.filesystem.ls(self.dataDir, detail=True):
            if entry["type"] != "directory":
                continue
            childName = entry["name"].rstrip("/").split("/")[-1]
            
            children.append(
                self.childClass(name=childName, dataDir=entry["name"], filesystem=self.filesystem)
            )
        return children

class pglChooseRun(pglChooseLevel):
    childClass = None
class pglChooseSubject(pglChooseLevel):
    childClass = pglChooseRun
class pglChooseExperiment(pglChooseLevel):
    childClass = pglChooseSubject
class pglChooseSession(pglChooseLevel):
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
        
    

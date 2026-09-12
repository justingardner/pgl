################################################################
#   filename: pglChoose.py
#    purpose: Dialogs for choosing experiments, runs etc 
#         by: JLG
#       date: Sept 12, 2026
################################################################

#############
# Import
#############
from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings, pglSettingsManager
from .pglDialog import pglDialogs
from .pglBase import pglBase
from pathlib import Path
from .pglSession import pglRun
from fsspec import AbstractFileSystem
from traitlets import Unicode, List, Instance
import re

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
    Class which provides ways to choose runs and experiment directories
    '''

    @classmethod
    def getSessionRuns(cls, fullDataPath=None, settings=None, settingsName=None, experimentName=None, subjectID=None, sessionName=None, runName=None, filesystem=None, filesystemPrefix=None, dataPath=None):
        # choose runs in a session, return a list of runs
        return cls.getExperimentPath(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath, allowMultipleRuns=True)

    @classmethod
    def getExperimentPath(cls, fullDataPath=None, settings=None, settingsName=None, experimentName=None, subjectID=None, sessionName=None, runName=None, filesystem=None, filesystemPrefix=None, dataPath=None, allowMultipleRuns=False):
        '''
        get the directory of the experiment. Many ways to call this to make it easy to get the correct experiemnt dir
        
        If you want to browse the full experiments:
        
            # use default settings to find dataDir
            pglChoose.getExperimentPath()
            
            # use settings name to find dataDir:
            pglChoose.getExperimentPath(settingsName='windowed')

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
                if not settings:
                    # get the default settings
                    settings = pglSettingsManager.getSettings(settingsName=settingsName)
                    if settings is None:
                        pglMessages.warning(f"Could not find settings {settingsName}")
                        return (None, None, None)
                if settings:
                    # set dataPath to where settings tells us it is
                    dataPath= settings.dataPath

            # expand user
            fullDataPath = Path(dataPath).expanduser()
            
            # now that we have the start of a path, validate the filesystem
            filesystem, fullDataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=fullDataPath, filesystemPrefix=filesystemPrefix)
            if filesystem is None:
                pglMessages.warning("Could not find dataPath: {fullDataPath}")
                return (None, None, None)
            
            # add on experiment name
            if experimentName:
                fullDataPath = Path(fullDataPath) / experimentName
                # check that experimentName exists
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Experiment directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on subject experiment names
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='experimentNames', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                if filesystem is None: 
                    return (None, None, None)
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
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='subjectIDs', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                if filesystem is None: 
                    return (None, None, None)
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
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='sessionNames', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                if filesystem is None: 
                    return (None, None, None)
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
                    (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='runNames', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                    if filesystem is None: 
                        return (None, None, None)
                    else: 
                        return (filesystem, fullDataPath, filesystemPrefix)
              
        return (filesystem, fullDataPath, filesystemPrefix)
    
    @ classmethod
    def _chooseDialog(cls, fullDataPath, filesystem=None, chooseLevel=None, allowMultipleRuns=False):
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
            if not allowMultipleRuns:
                pglMessages.message(f"Multiple runs selecting, using {runNames[0]}")
            else:
                return (filesystem, runNames)

        if allowMultipleRuns is False:
            return (filesystem, runNames[0])
        else:
            return (filesystem, runNames)

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
 

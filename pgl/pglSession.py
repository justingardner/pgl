################################################################
#   filename: pglSession.py
#    purpose: Classes which abstracts runs and sessions,
#             providing methods for accessing, loading and saving
#             behavioral, MRI, MEG, eyetracking and other data
#         by: JLG
#       date: Sept 12, 2026
################################################################

##############
# Imports
##############
import numpy as np
from .pglExperiment import pglExperimentData, pglExperimentBase, pglExperimentSettings, pglTaskBase, pglEventTrial
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from fsspec import AbstractFileSystem
from .pglSettings import pglSettings, pglSettingsManager, pglTraitSettings

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
            print("GOT NONE HERE")
            pglMessages.message(f"Loading tasks for: {self.filesystemPrefix}/{self.fullDataPath}")
            filesystem, fullDataPath, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.fullDataPath, filesystemPrefix=self.filesystemPrefix)
            taskNames = self.experimentSettings.tasks
            self._tasks = []
            for taskName in taskNames:
                print(f"taskName: {taskName}")
                self._tasks.append(pglTaskBase.load(dataPath=f"{fullDataPath}{filesystem.sep}{taskName}", filesystem=filesystem))
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
 

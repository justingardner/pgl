################################################################
#   filename: pglActions.py
#    purpose: actions
#         by: JLG
#       date: Sept 12, 2026
################################################################

from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from pathlib import Path
from .pglExperiment import pglEventSegment
from .pglSettings import pglSettings
from .pglDialog import pglDialogs
from typing import Annotated
import numpy as np
import matplotlib.pyplot as plt
from .pglChoose import pglChoose
from fsspec import AbstractFileSystem
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from .pglPipeline import pglAction
from .pglSession import pglSession

#################################
# class pglActionLoadSession
#################################
class pglActionLoadSessionSettings(pglTraitSettings):
    selectedPaths = List(Unicode(), help="Paths of runs selected for loading")
    filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// which can be set if the files are not local")

class pglActionLoadSession(pglAction):
    
    # settings
    settings = Instance(pglActionLoadSessionSettings, help="Settings for loading session")
    
    def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, experimentName: str=None, subjectID: str=None, sessionName: str=None, runName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None) -> None:

        # Choose the runs to load
        filesystem, runList, filesystemPrefix = pglChoose.getSessionRuns(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath)
        if filesystem is None:
            return

        # and put into settings
        self.settings = pglActionLoadSessionSettings()
        self.settings.selectedPaths = runList
        self.settings.filesystemPrefix = filesystemPrefix
    
        # we are now configured, so call super to set status
        super().configure()
        
    def _run(self):
        '''
        Run the action to load the session
        
        Returns:
            pglSession: The loaded session (or None if )
        '''
        # import session
        from .pglSession import pglSession
        
        # just create the session variable
        session = pglSession(filesystemPrefix=self.settings.filesystemPrefix, runList = self.settings.selectedPaths)
        
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
    



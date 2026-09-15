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
from .pglPipeline import pglAction
from .pglSession import pglSession, pglMNE


#################################
# Collection of predefined actions
#################################
class pglActions():
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # loadSession
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class loadSession(pglAction):
        # settings
        selectedPaths = List(Unicode(), help="Paths of runs selected for loading")
        filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// which can be set if the files are not local")
        
        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, experimentName: str=None, subjectID: str=None, sessionName: str=None, runName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None) -> None:

            # Choose the runs to load
            filesystem, runList, filesystemPrefix = pglChoose.getSessionRuns(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath)
            if filesystem is None:
                return

            # and put into settings
            self.selectedPaths = runList
            self.filesystemPrefix = filesystemPrefix
        
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self):
            '''
            Run the action to load the session
            
            Returns:
                pglSession: The loaded session (or None if )
            '''
            # import session
            from .pglSession import pglSession
            
            # just create the session variable
            session = pglSession(filesystemPrefix=self.filesystemPrefix, runList = self.selectedPaths)
            
            # and return
            return session
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # load FieldLine
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class loadFieldline(pglAction):
    
        # settings
        selectedPaths = List(Unicode(), help="Paths of fif files selected for loading")
        filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// which can be set if the files are not local")
    
        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None) -> None:
            """
            Choose Fieldline FIF files and store their paths in settings.
            """

            # Choose the FIF files to load.
            filesystem, fifList, filesystemPrefix = pglChoose.getFieldline(
                fullDataPath=fullDataPath,
                settings=settings,
                settingsName=settingsName,
                filesystem=filesystem,
                filesystemPrefix=filesystemPrefix,
                dataPath=dataPath,
            )

            # User cancelled or the data path could not be accessed.
            if filesystem is None:
                return

            # Put selected FIF paths into settings.
            self.selectedPaths = fifList
            self.filesystemPrefix = filesystemPrefix or ""
        
            # We are now configured, so call super to set status.
            super().configure()    
    
        ################################
        # run
        ################################
        def _run(self, session: pglSession | None = None, verbose: bool = True):
            """
            Load selected Fieldline FIF files and concatenate them into one
            MNE Raw object when more than one file was selected.
            """

            if not self.selectedPaths:
                self.setError("No Fieldline FIF files selected")
                return None

            # load libraries
            from pgl import pglBase
            try:            
                import mne
            except Exception as e:
                self.setError("mne library not available")
                return None

            # Validate the filesystem once, using the first selected FIF path plus the saved filesystem prefix.
            filesystem, _, _ = pglBase.validateFilesystem(dataPath=self.selectedPaths[0],filesystemPrefix=self.filesystemPrefix)

            if filesystem is None:
                self.setError(f"Could not access Fieldline FIF file: {self.selectedPaths[0]}")
                return None

            # initialize class which holds mne data
            mneData = pglMNE()
            
            # load the fif files
            for fifPath in self.selectedPaths:
                try:
                    pglMessages.message(f"Loading Fieldline FIF file: {fifPath}")

                    # MNE does not directly use an fsspec ssh:// URL. Open the
                    # path through the established fsspec filesystem and provide
                    # the resulting binary file object to MNE.
                    #
                    # preload=True ensures the data are loaded before fifFile is
                    # closed when leaving the context manager.
                    with filesystem.open(fifPath, "rb") as fifFile:
                        raw = mne.io.read_raw_fif(fifFile,preload=True,verbose=False)

                    mneData.add(raw, filename=fifPath, filesystemPrefix=self.filesystemPrefix)

                except Exception as e:
                    self.setError(f"Could not load FIF file {fifPath}: {e}")

            # if session is None, then create one
            if session is None: session = pglSession()
            
            # add the mne data to the session
            session.add(mneData)
            
            # return the session
            return session

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # concatenate
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneConcatenate(pglAction):
        
        selectedRawFilenames = List(Unicode(), help="Paths of runs selected for loading")
        
        ################################
        # configure
        ################################
        def configure(self, session: pglSession, all=False) -> None:

            # Choose the runs to concatenate
            if session.mne:
                # get all filenames (removing full path, just getting name)
                allRawFilenames = [Path(filename).name for filename in session.mne.rawFilenames]

                if all:
                    self.selectedRawFilenames = allRawFilenames
                else:
                    self.selectedRawFilenames = pglChoose.chooseList(allRawFilenames)

            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession):
            '''
            Run the action to load the session
            
            Returns:
                pglSession: The loaded session (or None if )
            '''
            selectedRaws = [
                session.mne.raws[iRaw]
                for iRaw, rawFilename in enumerate(session.mne.rawFilenames)
                if Path(rawFilename).name in self.selectedRawFilenames
            ]

            import mne
            from collections import Counter

            # Find the most common set of bad channels
            badSets = [frozenset(raw.info["bads"]) for raw in selectedRaws]
            mostCommonBads, nMostCommon = Counter(badSets).most_common(1)[0]

            # Report runs whose bad-channel set differs from the most common set
            bads = set()
            for iRaw, raw in enumerate(selectedRaws):
                rawBads = frozenset(raw.info["bads"])

                if rawBads != mostCommonBads:
                    pglMessages.message(f"{Path(self.selectedRawFilenames[iRaw]).name} has bads: {sorted(rawBads)}, which do not match the most common set: {sorted(mostCommonBads)}")
  
                bads |= set(raw.info["bads"])

            # Set all the bads to be the union of all bad channels
            for raw in selectedRaws:
                raw.info["bads"] = list(bads)                
                
            # concatenate
            session.mne.raw = mne.concatenate_raws(selectedRaws,preload=True,verbose=False)                            
            pglMessages.message(f"Set bads for concatenation to: {bads}")
            
            # and return
            return session
 
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # configure events
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneConfigureEvents(pglAction):
        
        triggerChannel = Unicode("", help="Paths of runs selected for loading")
        triggerShortestEvent = Int(1, help="shortest event length for a trigger")
        triggerLabels = List(Dict(allow_none=True, default_value={}), help='dict of all trigger names where key is trigger number and value is name')
        triggerLabelNames = List(Unicode(), help='Names of the trigger label dicts')
        
        ################################
        # configure
        ################################
        def configure(self, triggerChannel="di2", triggerShortestEvent = 1, triggerLabels=None, triggerLabelNames=None) -> None:

            # Set the trigger channel
            self.triggerChannel = triggerChannel
            self.triggerShortestEvent = triggerShortestEvent

            # get triggerLabels            
            if triggerLabels is None:
                self.triggerLabels = []
            elif isinstance(triggerLabels,dict):
                self.triggerLabels = [triggerLabels]
            else:
                self.triggerLabels = triggerLabels

            # get triggerLabelNames
            if triggerLabelNames is None:
                self.triggerLabelNames = []
            elif isinstance(triggerLabelNames,str):
                self.triggerLabelNames = [triggerLabelNames]
            else:
                self.triggerLabelNames = triggerLabelNames
                
            # check for the same length
            if len(self.triggerLabels) != len(self.triggerLabelNames):
                pglMessages.warning(f"Mismatched number of labels ({len(self.triggerLabels)}) and labelNames ({len(self.triggerLabelNames)})",level=1)                
            
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession):
            '''
            Run the action to find events, this will put up a dialog for confirmation from the experimenter
            
            Returns:
                pglSession
            '''
            import mne
            import pandas as pd
            
            # get the events
            session.mne.events = mne.find_events(session.mne.raw, stim_channel=self.triggerChannel, shortest_event=self.triggerShortestEvent)
            
            # Create event labels
            session.mne.eventLabels = pd.DataFrame()
            session.mne.eventLabels['code'] = session.mne.events[:, 2]
            
            # read in the events
            for iTriggerLabels, triggerLabels in enumerate(self.triggerLabels):
                # default to nan for labels
                session.mne.eventLabels[self.triggerLabelNames[iTriggerLabels]] = pd.NA
                unusedLabels = []
                
                for label, codes in triggerLabels.items():
                    # Permit a single integer for a code (or it can be a range)
                    if isinstance(codes, int): codes = [codes]

                    # label all the events
                    mask = session.mne.eventLabels["code"].isin(codes)
                    session.mne.eventLabels.loc[mask, self.triggerLabelNames[iTriggerLabels]] = label

                    # check if the labels were not used
                    if not mask.any(): unusedLabels.append(label)
                
                if unusedLabels:
                    pglMessages.warning(
                        f"Trigger label column '{self.triggerLabelNames[iTriggerLabels]}': the following labels "
                        f"were configured but not used by any event: {unusedLabels}",
                    level=1,
                    )
                # Check whether every event received a label in this column.
                unlabeledMask = session.mne.eventLabels[self.triggerLabelNames[iTriggerLabels]].isna()
                if unlabeledMask.any():
                    unlabeledCodes = session.mne.eventLabels.loc[unlabeledMask,"code"].unique()

                    pglMessages.warning(f"Trigger label column '{self.triggerLabelNames[iTriggerLabels]}' did not label "
                        f"{unlabeledMask.sum()} of {len(session.mne.eventLabels)} events. "
                        f"Unlabeled codes: {unlabeledCodes.tolist()}",
                        level=1,
                    )
            
            # make labels for display using the first one
            if session.mne.eventLabels.shape[1] >= 3:
                labelColumn = session.mne.eventLabels.columns[1]
                eventID = (
                    session.mne.eventLabels[["code", labelColumn]]
                    .dropna()
                    .drop_duplicates()
                    .set_index(labelColumn)["code"]
                    .to_dict()
                )
            else:
                eventID = None
                

            # and display
            mne.viz.plot_events(session.mne.events, event_id=eventID, sfreq=session.mne.raw.info["sfreq"], first_samp=session.mne.raw.first_samp)
            
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
    



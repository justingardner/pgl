################################################################
#   filename: pglActions.py
#    purpose: actions
#         by: JLG
#       date: Sept 12, 2026
################################################################

from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from pathlib import Path
from .pglExperiment import pglEventSegment
from .pglSettings import pglSettings
from .pglDialog import pglDialogs
from typing import Annotated
from .pglChoose import pglChoose
from .pglPipeline import pglAction
from .pglSession import pglSession, pglMNE
import numpy as np
from numbers import Integral
import matplotlib.pyplot as plt
from fsspec import AbstractFileSystem
from traitlets import HasTraits, Enum, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
import pandas as pd

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
        
        triggerChannel = Unicode("di2", help="Paths of runs selected for loading")
        triggerShortestEvent = Int(1, help="shortest event length for a trigger")
        triggerLabelSets = Dict(default_value={},help=("Mapping of scheme name to label definitions. "
            "Example: {'stimulusType': {'thingsStim': range(1, 201), "
            "'blank': [1022], 'catch': [1023]}}"
            ),
        )
        tmin = Float(0.0, help="Time to start triggered epoch in seconds")
        tmax = Float(1.0, help="Time to end triggered epoch in seconds")
        
        ################################
        # configure
        ################################
        def configure(self, triggerChannel: str = None, triggerShortestEvent: int = 1, triggerLabelSets: Dict = None, tmin: float = None, tmax: float = None) -> None:

            # Set the trigger channel
            if triggerChannel: self.triggerChannel = triggerChannel
            if triggerShortestEvent: self.triggerShortestEvent = triggerShortestEvent

            # get triggerLabels            
            if triggerLabelSets: self.triggerLabelSets = triggerLabelSets
            
            # tmin and tmax are min and max in seconds of epochs
            if tmin: self.tmin = tmin
            if tmax: self.tmax = tmax
                
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession):
            """
            Find raw trigger events, create an event-label DataFrame, create one Epochs
            object with metadata, create one grand-average Evoked object, and display
            events using the first configured labeling scheme.

            Stores:
                session.mne.events:
                    Canonical MNE events array, shape (nEvents, 3):
                    [sample, previousValue, rawTriggerCode].

                session.mne.eventsID:
                    Pandas DataFrame with one row per event. Includes the raw event
                    code plus one column for each configured labeling scheme.

                session.mne.epochs:
                    One MNE Epochs object with session.mne.eventsID attached as
                    metadata.

                session.mne.evoked:
                    Grand-average Evoked object across all epochs.
            """
            import mne

            # -------------------------------------------------------------------------
            # Find the canonical raw events matrix.
            # MNE format: [sample, previousValue, eventCode]
            # -------------------------------------------------------------------------
            events = mne.find_events(
                session.mne.raw,
                stim_channel=self.triggerChannel,
                shortest_event=self.triggerShortestEvent,
            )

            if len(events) == 0:
                raise RuntimeError(
                    f"No events found in trigger channel '{self.triggerChannel}'."
                )

            rawCodes = events[:, 2].astype(int)

            # -------------------------------------------------------------------------
            # Make the event-label DataFrame.
            #
            # This is the canonical location for all alternate event grouping schemes.
            # One event may have a label in multiple scheme columns.
            # -------------------------------------------------------------------------
            eventsDf = pd.DataFrame(
                {
                    "sample": events[:, 0].astype(int),
                    "previousValue": events[:, 1].astype(int),
                    "code": rawCodes,
                }
            )

            # A readable representation of the original raw trigger code.
            eventsDf["rawLabel"] = eventsDf["code"].astype(str)

            # -------------------------------------------------------------------------
            # Add one DataFrame column per label scheme.
            #
            # triggerLabelSets format:
            #
            # {
            #     "stimulusType": {
            #         "thingsStim": range(1, 201),
            #         "blank": [1022],
            #         "catch": [1023],
            #     },
            #     "responseType": {
            #         "correct": [2001],
            #         "incorrect": [2002],
            #     },
            # }
            # -------------------------------------------------------------------------
            for schemeName, labelDefinitions in self.triggerLabelSets.items():

                # Start with no event assigned in this scheme.
                schemeLabels = pd.Series(
                    np.nan,
                    index=eventsDf.index,
                    dtype=object,
                )

                for configuredLabel, configuredCodes in labelDefinitions.items():

                    labelName = str(configuredLabel)

                    # Permit a single integer, list, tuple, set, NumPy array, or range.
                    if isinstance(configuredCodes, Integral):
                        codes = [int(configuredCodes)]

                    elif isinstance(configuredCodes, str):
                        raise TypeError(
                            f"Label '{labelName}' in scheme '{schemeName}' has a "
                            f"string trigger-code definition ({configuredCodes!r}). "
                            f"Use an integer or iterable of integers instead."
                        )

                    else:
                        try:
                            codes = [int(code) for code in configuredCodes]
                        except TypeError as error:
                            raise TypeError(
                                f"Label '{labelName}' in scheme '{schemeName}' must "
                                f"map to an integer or iterable of integers."
                            ) from error

                    mask = eventsDf["code"].isin(codes).to_numpy()

                    # Warn if configured event codes are absent from the recording.
                    if not mask.any():
                        pglMessages.warning(
                            f"Label '{labelName}' in event scheme '{schemeName}' "
                            f"did not match any raw event codes.",
                            level=1,
                        )
                        continue

                    # An event can only get one label within a single scheme.
                    overlapMask = mask & schemeLabels.notna().to_numpy()

                    if overlapMask.any():
                        overlappingCodes = np.unique(
                            eventsDf.loc[overlapMask, "code"].to_numpy()
                        ).tolist()

                        raise ValueError(
                            f"Event scheme '{schemeName}' has overlapping label "
                            f"definitions. Label '{labelName}' overlaps a prior "
                            f"label for raw trigger codes: {overlappingCodes}"
                        )

                    schemeLabels.loc[mask] = labelName

                eventsDf[schemeName] = schemeLabels

                # Report trigger codes that did not receive a label in this scheme.
                unlabeledMask = eventsDf[schemeName].isna().to_numpy()

                if unlabeledMask.any():
                    unlabeledCodes = np.unique(
                        eventsDf.loc[unlabeledMask, "code"].to_numpy()
                    ).tolist()

                    pglMessages.warning(
                        f"Event scheme '{schemeName}' did not assign labels to "
                        f"{unlabeledMask.sum()} of {len(eventsDf)} events. "
                        f"Unlabeled raw codes: {unlabeledCodes}",
                        level=1,
                    )

            # -------------------------------------------------------------------------
            # Store the single canonical event representation.
            # -------------------------------------------------------------------------
            session.mne.events = events
            session.mne.eventsID = eventsDf

            # -------------------------------------------------------------------------
            # Create one Epochs object.
            #
            # MNE still needs an event_id mapping to create Epochs. This mapping is
            # local because the DataFrame is now the authoritative label store.
            #
            # Every raw trigger code is included, even if it has no label in one or
            # more configured grouping schemes.
            # -------------------------------------------------------------------------
            rawEventId = {
                f"raw/{int(code)}": int(code)
                for code in np.unique(rawCodes)
            }

            epochs = mne.Epochs(
                session.mne.raw,
                events=session.mne.events,
                event_id=rawEventId,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=None,
                metadata=session.mne.eventsID,
                preload=True,
                verbose=False,
            )

            session.mne.epochs = epochs

            # One grand-average Evoked over all retained epochs.
            session.mne.evoked = epochs.average()

            # -------------------------------------------------------------------------
            # Make a temporary events array for displaying the FIRST label scheme.
            #
            # This does not get stored. The canonical event data remains:
            #     session.mne.events
            #     session.mne.eventsID
            #
            # We cannot permanently replace raw event codes with one scheme's grouped
            # codes because that would discard information needed by other schemes.
            # -------------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(24, 8))

            if self.triggerLabelSets:
                firstSchemeName = next(iter(self.triggerLabelSets))

                plotMask = eventsDf[firstSchemeName].notna().to_numpy()

                if plotMask.any():
                    plotEvents = events[plotMask].copy()
                    plotLabels = eventsDf.loc[plotMask, firstSchemeName]

                    # Keep labels in their configured dictionary order where possible.
                    configuredLabels = [
                        str(labelName)
                        for labelName in self.triggerLabelSets[firstSchemeName]
                        if str(labelName) in set(plotLabels)
                    ]

                    plotEventId = {
                        labelName: eventCode
                        for eventCode, labelName in enumerate(configuredLabels, start=1)
                    }

                    # Temporarily map label strings back to MNE integer event codes.
                    plotEvents[:, 2] = (
                        plotLabels.map(plotEventId)
                        .to_numpy(dtype=int)
                    )

                    mne.viz.plot_events(
                        plotEvents,
                        event_id=plotEventId,
                        sfreq=session.mne.raw.info["sfreq"],
                        first_samp=session.mne.raw.first_samp,
                        axes=ax,
                        show=False,
                    )

                    fig.suptitle(f"Events grouped by: {firstSchemeName}")

                else:
                    pglMessages.warning(
                        f"The first event scheme '{firstSchemeName}' did not label "
                        f"any events. Displaying raw events instead.",
                        level=1,
                    )

                    rawPlotEventId = {
                        str(int(code)): int(code)
                        for code in np.unique(rawCodes)
                    }

                    mne.viz.plot_events(
                        events,
                        event_id=rawPlotEventId,
                        sfreq=session.mne.raw.info["sfreq"],
                        first_samp=session.mne.raw.first_samp,
                        axes=ax,
                        show=False,
                    )

                    fig.suptitle("Raw events")

            else:
                rawPlotEventId = {
                    str(int(code)): int(code)
                    for code in np.unique(rawCodes)
                }

                mne.viz.plot_events(
                    events,
                    event_id=rawPlotEventId,
                    sfreq=session.mne.raw.info["sfreq"],
                    first_samp=session.mne.raw.first_samp,
                    axes=ax,
                    show=False,
                )

                fig.suptitle("Raw events")

            fig.tight_layout(rect=(0, 0, 1, 0.96))

            return session

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # filter
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneFilter(pglAction):
        
        lowCutoff = Float(0.0, help="Low pass cutoff for filtering")
        highCutoff = Float(0.0, help="High pass cutoff for filtering")
        notch = Bool(True, help="apply notch filter")
        notchFrequency = Float(60.0, help="Frequency at which to notch")
        
        ################################
        # configure
        ################################
        def configure(
            self,
            lowCutoff: float = 1.0,
            highCutoff: float = 80.0,
            notch: bool = True,
            notchFrequency: float = 60.0        
        ) -> None:

            # set cutoffs
            self.lowCutoff = lowCutoff
            self.highCutoff = highCutoff
            self.notch = notch
            self.notchFrequency = notchFrequency
            
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
           Run the filtering
            
            Returns:
                pglSession: fitered session
            '''
            # import mne
            import mne
            
            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None
                
            # apply low and high pass filter
            session.mne.raw.load_data().filter(l_freq=self.lowCutoff,h_freq=None)
            session.mne.raw.load_data().filter(l_freq=None,h_freq=self.highCutoff)
            
            # apply notch filter
            if self.notch:                
                meg_picks = mne.pick_types(session.mne.raw.info, meg=True)
                session.mne.raw.notch_filter(freqs=self.notchFrequency, picks=meg_picks)
            
            # display spectrum    
            session.mne.raw.compute_psd(fmax=100).plot(average=False, picks="data", exclude="bads",amplitude=False)            
            
            # and return
            return session
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # bads handling
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneBads(pglAction):
        
        # parameters
        method = Enum(["interpolate", "drop"], default_value="interpolate", help="Method for handling bads")
        excludeWhenPositionIsNaN = Bool(True, help="If a sensor was 'excluded' during data collection, usually because for some reason it didn't turn on, it's position may show up as NaN but it will not be marked as 'bad'. Lets double check for any cases like this to make sure we mark and drop these sensors.")
        extraBads = List(Unicode(), help="List of extra sensors to exclude as bads")
        interpolationOrigin = Enum(["auto", "zero"],default_value="zero",help=(
                "Set origin for interpolation. 'auto' fits a sphere from head "
                "digitization points and requires raw.info['dig'] to contain "
                "sufficient extra (headshape) or EEG points -- cardinal/HPI points "
                "alone are not enough and will raise an error. 'zero' sets the "
                "origin to [0,0,0] in the head coordinate frame, i.e. the "
                "fiducial-based head center (midpoint of LPA/RPA) -- accurate only "
                "if fiducials were placed by convention rather than measured "
                "asymmetrically."
            ),
        )
        
        ################################
        # configure
        ################################
        def configure(
            self,
            method: str = None,
            excludeWhenPositionIsNaN: bool = None,
            extraBads: list = None,
            interpolationOrigin: str = None,
        ) -> None:

            # set parameters
            if method: self.method = method
            if excludeWhenPositionIsNaN: self.excludeWhenPostionIsNan = self.excludeWhenPositionIsNaN
            if extraBads: self.extraBads = extraBads
            if interpolationOrigin: self.interpolationOrigin = interpolationOrigin
             
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Run the bads processing
            
            Returns:
                pglSession: session
            '''
            # import mne
            import mne
            
            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            # set extra bads if they are configured                
            if self.extraBads:
                # validate the extrabads list by comparing against actual channel names
                validChNames = set(session.mne.raw.info["ch_names"])
                invalidBads = [ch for ch in self.extraBads if ch not in validChNames]
                
                # if we have invalids then abort, otherwise extend the bads list
                if invalidBads:
                    self.setError(f"extraBads contains channels not found in raw data: {invalidBads}")
                    return None
                else:
                    session.mne.raw.info["bads"].extend(self.extraBads)
            
            # exclude any sensors that have NaN in their location
            if self.excludeWhenPositionIsNaN:
                bads_NaNs=[]
                # look for channels that have NaN in their locations,
                # as this indicates that they were excluded during data collection
                for i in range(0,session.mne.raw.info["nchan"]):
                    ch_pos = session.mne.raw.info["chs"][i]["loc"][:3]
                    if np.isnan(ch_pos).any():
                        bads_NaNs.append(session.mne.raw.info["chs"][i]["ch_name"])
                
                # we found some bad channels
                if bads_NaNs:
                    # check if they are in the existing channels
                    existingBads = set(session.mne.raw.info["bads"])
                    newBads = [ch for ch in bads_NaNs if ch not in existingBads]
                    pglMessages.message(f"Found {len(bads_NaNs)} channels with NaNs in the location, indicating they were excluded: {bads_NaNs}. ")
                    # message and add any new Bads
                    if newBads:
                        pglMessages.message(f"{len(newBads)} were not already marked bad and have been added.")
                        session.mne.raw.info["bads"].extend(newBads)

            # what to do with the bads
            bads = session.mne.raw.info["bads"]
            if bads:
                if self.method == "interpolate":
                    pglMessages.message(f"Interpolating {len(bads)} bad sesnsors: {bads}", emphasize=True)
                    #-- Interpolate bads - set origin to zero in "HEAD" frame
                    if self.interpolationOrigin == "zero":
                        session.mne.raw.interpolate_bads(origin=[0,0,0],reset_bads=True)
                    else:
                        session.mne.raw.interpolate_bads(origin="auto",reset_bads=True)
                else:
                    pglMessages.message(f"Dropping {len(bads)} bad sesnsors: {bads}", emphasize=True)
                    #-- drop bads
                    session.mne.raw.drop_channels(session.mne.raw.info["bads"])
            else:
                pglMessages.message(f"No bad sensors to {self.method}")

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
    



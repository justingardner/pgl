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
from .pglExperiment import pglEventTrial, pglEventSegment

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

    ################################################################
    # load NetStation
    ################################################################
    class loadNetStation(pglAction):

        # settings
        selectedPaths = List(Unicode(), help="Paths of MFF recording directories selected for loading")
        filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// for non-local recordings")

        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None) -> None:
            """
            Choose NetStation MFF recordings and store their paths.
            """

            # Choose the MFF recordings to load.
            filesystem, mffList, filesystemPrefix = pglChoose.getNetStation(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath)

            # User cancelled or the data path could not be accessed.
            if filesystem is None:
                return

            # Store selected MFF paths.
            self.selectedPaths = mffList or []
            self.filesystemPrefix = filesystemPrefix or ""

            # We are now configured, so call super to set status.
            super().configure()

        ################################
        # run
        ################################
        def _run(self, session: pglSession | None = None, verbose: bool = True):
            """
            Load selected NetStation MFF recordings into pglMNE.

            Remote MFF packages are copied to temporary local storage.
            Data are preloaded before the temporary copies are removed.
            """

            if not self.selectedPaths:
                self.setError("No NetStation MFF recordings selected")
                return None

            # Load libraries.
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from fsspec.implementations.local import LocalFileSystem
            from pgl import pglBase

            try:
                import mne
            except ImportError as e:
                self.setError(f"mne library not available: {e}")
                return None

            # Recreate the filesystem using the saved path and prefix.
            filesystem, _, _ = pglBase.validateFilesystem(dataPath=self.selectedPaths[0], filesystemPrefix=self.filesystemPrefix)

            if filesystem is None:
                self.setError(f"Could not access NetStation MFF recording: {self.selectedPaths[0]}")
                return None

            # Initialize the class which holds MNE data.
            mneData = pglMNE()
            loadedCount = 0

            # Load the MFF recordings.
            for mffPath in self.selectedPaths:
                try:
                    if verbose:
                        pglMessages.message(f"Loading NetStation MFF recording: {mffPath}")

                    if not filesystem.isdir(mffPath):
                        raise ValueError("Expected an MFF recording directory")

                    if isinstance(filesystem, LocalFileSystem):
                        # MNE can read local MFF packages directly.
                        localMffPath = filesystem._strip_protocol(mffPath)
                        raw = mne.io.read_raw_egi(localMffPath, preload=True, verbose=False)

                    else:
                        # MNE needs the complete MFF package on local storage.
                        with TemporaryDirectory(prefix="pglNetStation_") as tempDir:
                            recordingName = str(mffPath).rstrip("/").rsplit("/", 1)[-1]
                            localMffPath = Path(tempDir) / recordingName

                            if verbose:
                                pglMessages.message("Copying remote MFF package to temporary local storage")

                            filesystem.get(str(mffPath).rstrip("/"), str(localMffPath), recursive=True)
                            raw = mne.io.read_raw_egi(str(localMffPath), preload=True, verbose=False)

                        # preload=True keeps signal data available after cleanup.

                    # Save the original source path rather than the temporary path.
                    mneData.add(raw, filename=mffPath, filesystemPrefix=self.filesystemPrefix)
                    loadedCount += 1

                except Exception as e:
                    self.setError(f"Could not load MFF recording {mffPath}: {e}")

            # Do not add an empty container if every recording failed.
            if loadedCount == 0:
                return None

            # If session is None, create one.
            if session is None: session = pglSession()

            # Add the MNE data to the session.
            session.add(mneData)

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
        
        ################################
        # configure
        ################################
        def configure(self, triggerChannel: str = None, triggerShortestEvent: int = 1, triggerLabelSets: Dict = None) -> None:

            # Set the trigger channel
            if triggerChannel: self.triggerChannel = triggerChannel
            if triggerShortestEvent: self.triggerShortestEvent = triggerShortestEvent

            # get triggerLabels            
            if triggerLabelSets: self.triggerLabelSets = triggerLabelSets
            
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession):
            """
            Find raw trigger events, create an event-label DataFrame, create one Epochs
            object with metadata, and display events using the first configured labeling scheme.

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
            
            # Select EEG and MEG channels, excluding channels marked bad.
            dataPicks = mne.pick_types(session.mne.raw.info, meg=True, eeg=True, exclude="bads")
                
            # apply low and high pass filter
            session.mne.raw.load_data().filter(l_freq=self.lowCutoff,h_freq=None,picks=dataPicks)
            session.mne.raw.load_data().filter(l_freq=None,h_freq=self.highCutoff,picks=dataPicks)
            
            # apply notch filter
            if self.notch:                
                meg_picks = mne.pick_types(session.mne.raw.info, meg=True)
                session.mne.raw.notch_filter(freqs=self.notchFrequency, picks=dataPicks)
            
            # display spectrum    
            maxFrequency = min(100.0, session.mne.raw.info["sfreq"] / 2)
            spectrum = session.mne.raw.compute_psd(picks=dataPicks, fmax=maxFrequency)
            power = spectrum.get_data()

            # Identify zero-power or invalid channels.
            zeroChannels = [name for name, values in zip(spectrum.ch_names, power) if np.all(values == 0)]
            invalidChannels = [name for name, values in zip(spectrum.ch_names, power) if not np.all(np.isfinite(values))]
            plotChannels = [name for name in spectrum.ch_names if name not in zeroChannels + invalidChannels]

            print(f"Zero-power channels: {zeroChannels}")
            print(f"Nonfinite-power channels: {invalidChannels}")

            if plotChannels:
                spectrum.plot(picks=plotChannels, average=False, amplitude=False)
            else:
                pglMessages.warning("No channels with valid nonzero power to plot")
            
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
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # make epochs
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneCreateEpochs(pglAction):

        # parameters
        tmin = Float(0.0, help="Time to start triggered epoch in seconds")
        tmax = Float(1.0, help="Time to end triggered epoch in seconds")
                
        ################################
        # configure
        ################################
        def configure(self, tmin: float = None, tmax: float = None) -> None:

            # tmin and tmax are min and max in seconds of epochs
            if tmin: self.tmin = tmin
            if tmax: self.tmax = tmax
                
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Display the grand average
            
            Returns:
                pglSession: fitered session
            '''
            # import mne
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            # -------------------------------------------------------------------------
            # Create one Epochs object.
            #
            # MNE still needs an event_id mapping to create Epochs. This mapping is
            # local because the DataFrame is now the authoritative label store.
            #
            # Every raw trigger code is included, even if it has no label in one or
            # more configured grouping schemes.
            # -------------------------------------------------------------------------
            rawCodes = session.mne.events[:, 2].astype(int)
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

            # and return
            return session
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # plot evoked
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mnePlotEvoked(pglAction):

        # parameters
        set = Unicode(allow_none=True, default_value=True, help="Name of label set passed to configure events from which to compute evoked (defaults to first set)")
        label = Unicode(allow_none=True, default_value=True, help="Name of label passed in the set to configure events for which to compute the evoked (defaults to average across all event types)")
        picks = Unicode("mag",help="For topomap plotting type of topomap e.g. eeg, mag, grad - defined by mne - also can be name (or initial part of name of a sensor)")
        minFreq = Float(0.0, help="minimum frequency for display of frequency plot of evoked")
        maxFreq = Float(np.inf, help="minimum frequency for display of frequency plot of evoked")
        
        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            # set traitlets
            self.configureTraits(**kwargs)   
            
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Display the evoked
            
            Returns:
                pglSession: session
            '''
            # import mne
            import mne

            # validate picks
            self.picks = session.mne.validatePicks(self.picks)
            
            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw mne loaded")
                return None

            if session.mne.epochs is None:
                self.setError("Session does not have epochs created")
                return None

            # create the grand average and plot
            if self.set is None:
                self.set = session.mne.eventsID.columns[3]
            elif self.set not in session.mne.eventsID.columns:
                pglMessages.warning(f"Could not find label set: {self.set}")
                return
            
            if self.label is None:
                # just get average across all conditions
                session.mne.evoked = session.mne.epochs.average()
            else:
                labels = session.mne.eventsID[self.set].dropna().unique()
                if self.label not in labels:
                    pglMessages.warning(f"Unknown label {self.label!r} in column {self.set!r}. Available labels: {list(labels)}")
                    return
                session.mne.evoked = session.mne.epochs[f'{self.set} == "{self.label}"'].average()

            if session.mne.isSensor(self.picks):
                fig = session.mne.evoked.plot(picks=[self.picks],show=False)
                
                evoked = session.mne.evoked.copy().pick([self.picks])
                timeSeries = evoked.data[0]
                sampleRate = evoked.info["sfreq"]

                fftValues = np.fft.rfft(timeSeries)
                freqs = np.fft.rfftfreq(len(timeSeries), d=1 / sampleRate)
                magnitude = np.abs(fftValues)

                figFft, ax = plt.subplots(figsize=(12, 6))
                ax.stem(freqs, magnitude, linefmt="C0-", markerfmt="C0.", basefmt=" ")
                ax.set(xlabel="Frequency (Hz)", ylabel="FFT magnitude", title=f"Evoked FFT magnitude: {self.picks}", xlim=(self.minFreq, self.maxFreq))
                figFft.tight_layout()
                
                
            else:
                fig = session.mne.evoked.plot_joint(times="peaks",picks=self.picks,show=False)
            fig.set_size_inches(20, 8)
            
            # Spectrum of the evoked response
            if session.mne.isSensor(self.picks):
                evoked = session.mne.evoked.copy()
                nTimes = evoked.data.shape[1]
                spectrum = evoked.compute_psd(method="welch", picks=[self.picks], fmin=self.minFreq, fmax=self.maxFreq, n_fft=nTimes, n_per_seg=nTimes, n_overlap=0, window="boxcar", remove_dc=True)
                psds, freqs = spectrum.get_data(return_freqs=True)
                # get the first sepctrum and convert to fT^2/Hz
                psd = psds[0] * 1e30
                # get rid of dc
                psd = psd[1:]
                freqs = freqs[1:]
            else:
                spectrum = session.mne.evoked.compute_psd(method="welch", picks=self.picks, fmin=self.minFreq, fmax=self.maxFreq)
                psds, freqs = spectrum.get_data(return_freqs=True)
                psd = psds.mean(axis=0)
                # convert to fT^2/Hz    
                psd = psd * 1e30

            figPsd, ax = plt.subplots(figsize=(12, 6))
            ax.stem(freqs, psd, linefmt="C0-", markerfmt="C0.", basefmt=" ")
            ax.set(xlabel="Frequency (Hz)", ylabel="Power spectral density (fT²/Hz)", title=f"Evoked response spectrum: {self.picks}")
            figPsd.tight_layout()
            
            # and return
            return session
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # downsample evoked
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneDownsampleEvoked(pglAction):

        # parameters
        downsampleFrequency = Float(200.0, help="Downsample frequency")
                
        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            # set traitlets
            self.configureTraits(**kwargs)     
            
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Downsample the evoked. Works from raw to avoid boundary effects
            
            Returns:
                pglSession: session
            '''
            # import mne
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw mne loaded")
                return None

            if session.mne.epochs is None:
                self.setError("Session does not have epochs created")
                return None


            if session.mne.raw.info["sfreq"] < self.downsampleFrequency:
                self.setError(f"Raw sampling rate is {session.mne.raw.info['sfreq']} Hz, below requested target rate of {self.downsampleFrequency} Hz.")
                return None

            # Copy so session.mne.raw remains unchanged.
            # resample() needs loaded data.
            raw = session.mne.raw.copy().load_data()

            # MNE returns (rawResampled, eventsResampled) when `events=` is supplied.
            raw, events = raw.resample(
                sfreq=self.downsampleFrequency,
                npad="auto",
                events=session.mne.events,
                verbose=False,
            )
            
            # MNE requires an event_id mapping from label -> integer event code.
            # Here we epoch every raw trigger code.
            eventId = {
                str(int(code)): int(code)
                for code in np.unique(events[:, 2])
            }

            epochs = mne.Epochs(
                raw=raw,
                events=events,
                event_id=eventId,
                tmin=session.mne.epochs.tmin,
                tmax=session.mne.epochs.tmax,
                baseline=(session.mne.epochs.tmin, 0),
                preload=True,
                metadata=session.mne.eventsID.copy(),
                reject_by_annotation=True,
                verbose=False,
            )

            session.mne.epochs = epochs
            
            # and return
            return session

    class mneComputeNCSNR(pglAction):
        """
        Compute single-sensor noise-corrected SNR (NCSNR).

        Example:

            action.configure(
                set="grouped",
                label="things",
                conditionIDColumn="code",
                chType="mag",
            )

        This selects all epochs where:

            epochs.metadata["grouped"] == "things"

        It then treats each unique value in `epochs.metadata["code"]` as a
        separate signal condition. Multiple trials with the same code are treated
        as repetitions of that same condition.
        """

        channelName = Unicode(allow_none=True, default_value=None, help="Full or partial channel name. If None, use the largest absolute Evoked response.")
        chType = Unicode("mag", help="MNE channel type for automatic peak-channel selection, e.g. eeg, mag, or grad.")
        set = Unicode(allow_none=True, default_value=None, help="Metadata column used to select trials, e.g. grouped.")
        label = Unicode(allow_none=True, default_value=None, help="Metadata label used to select trials, e.g. things.")
        conditionIDColumn = Unicode("code", help="Metadata column identifying individual signal conditions.")

        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            self.configureTraits(**kwargs)
            super().configure()

        ################################
        # run
        ################################
        def _run(self, session: pglSession):
            # ---------------------------------------------------------------------
            # Validate session data.
            # ---------------------------------------------------------------------
            if not hasattr(session.mne, "epochs"):
                self.setError("session.mne.epochs does not exist.")
                return None

            if not hasattr(session.mne, "eventsID"):
                self.setError("session.mne.eventsID does not exist.")
                return None

            epochs = session.mne.epochs
            eventsID = session.mne.eventsID

            if epochs.metadata is None:
                self.setError("session.mne.epochs.metadata does not exist. Attach session.mne.eventsID when creating Epochs.")
                return None

            # ---------------------------------------------------------------------
            # Validate configured selection fields.
            # ---------------------------------------------------------------------
            if self.set is None:
                self.setError("No set was configured. For example: set='grouped'.")
                return None

            if self.label is None:
                self.setError("No label was configured. For example: label='things'.")
                return None

            if self.set not in eventsID.columns:
                self.setError(f"Column {self.set!r} does not exist in session.mne.eventsID. Available columns: {list(eventsID.columns)}")
                return None

            if self.set not in epochs.metadata.columns:
                self.setError(f"Column {self.set!r} does not exist in epochs.metadata. Available columns: {list(epochs.metadata.columns)}")
                return None

            if self.conditionIDColumn not in eventsID.columns:
                self.setError(f"Condition-ID column {self.conditionIDColumn!r} does not exist in session.mne.eventsID. Available columns: {list(eventsID.columns)}")
                return None

            if self.conditionIDColumn not in epochs.metadata.columns:
                self.setError(f"Condition-ID column {self.conditionIDColumn!r} does not exist in epochs.metadata. Available columns: {list(epochs.metadata.columns)}")
                return None

            availableLabels = eventsID[self.set].dropna().unique()

            if self.label not in availableLabels:
                self.setError(f"Label {self.label!r} does not occur in eventsID[{self.set!r}]. Available labels: {list(availableLabels)}")
                return None

            # ---------------------------------------------------------------------
            # Resolve analysis channel.
            #
            # If no channel name is configured, use the channel that has the
            # largest absolute response anywhere in session.mne.evoked.
            # ---------------------------------------------------------------------
            evokedPeakTime = None
            evokedPeakAmplitude = None

            if self.channelName is None:
                if not hasattr(session.mne, "evoked"):
                    self.setError("channelName was not provided and session.mne.evoked does not exist. Provide channelName or create an Evoked first.")
                    return None

                try:
                    channelName, evokedPeakTime, evokedPeakAmplitude = session.mne.evoked.get_peak(ch_type=self.chType, mode="abs", return_amplitude=True)
                except Exception as error:
                    self.setError(f"Could not find an Evoked peak channel for chType={self.chType!r}: {error}")
                    return None

                if channelName not in epochs.ch_names:
                    self.setError(f"Automatically selected Evoked peak channel {channelName!r} does not occur in session.mne.epochs.")
                    return None

                pglMessages.message(f"No channelName provided; using Evoked peak channel {channelName!r} at {evokedPeakTime * 1000:.1f} ms.")

            else:
                matchingChannels = [name for name in epochs.ch_names if self.channelName.lower() in name.lower()]

                if not matchingChannels:
                    self.setError(f"No channel matched {self.channelName!r}. Available channels: {epochs.ch_names}")
                    return None

                if len(matchingChannels) > 1:
                    self.setError(f"Channel query {self.channelName!r} matched multiple channels: {matchingChannels}. Use a more specific name.")
                    return None

                channelName = matchingChannels[0]

            # ---------------------------------------------------------------------
            # Identify the individual conditions from the complete events table.
            #
            # For example, with:
            #
            #     set="grouped"
            #     label="things"
            #     conditionIDColumn="code"
            #
            # this may yield:
            #
            #     [2, 3, 4, ..., 200]
            # ---------------------------------------------------------------------
            eventSelectionMask = eventsID[self.set].eq(self.label)

            conditionIDs = eventsID.loc[eventSelectionMask, self.conditionIDColumn].dropna().unique().tolist()

            try:
                conditionIDs = sorted(conditionIDs)
            except TypeError:
                conditionIDs = sorted(conditionIDs, key=str)

            if len(conditionIDs) < 2:
                self.setError(f"Selection {self.set!r} == {self.label!r} has only {len(conditionIDs)} unique IDs in {self.conditionIDColumn!r}. NCSNR requires at least two conditions.")
                return None

            # ---------------------------------------------------------------------
            # Extract trial data for every condition.
            #
            # Crucially, extraction uses epochs.metadata rather than eventsID,
            # because some original events may have been dropped by MNE.
            # ---------------------------------------------------------------------
            retainedLabelMask = epochs.metadata[self.set].eq(self.label)

            if not retainedLabelMask.any():
                self.setError(f"No retained epochs have {self.set!r} == {self.label!r}.")
                return None

            conditionData = {}
            conditionMeans = {}
            nRepsPerCondition = {}
            missingConditions = []
            insufficientRepetitions = {}

            for conditionID in conditionIDs:
                conditionMask = (retainedLabelMask & epochs.metadata[self.conditionIDColumn].eq(conditionID)).to_numpy()

                if not conditionMask.any():
                    missingConditions.append(conditionID)
                    continue

                # data shape: (nRepetitions, nTimes)
                data = epochs[conditionMask].get_data(picks=[channelName])[:, 0, :]
                nReps = data.shape[0]

                if nReps < 2:
                    insufficientRepetitions[conditionID] = nReps
                    continue

                conditionData[conditionID] = data
                conditionMeans[conditionID] = data.mean(axis=0)
                nRepsPerCondition[conditionID] = nReps

            if missingConditions:
                self.setError(f"No retained epochs were found for condition IDs within {self.set!r} == {self.label!r}: {missingConditions}")
                return None

            if insufficientRepetitions:
                details = ", ".join(f"{conditionID}: {nReps}" for conditionID, nReps in insufficientRepetitions.items())
                self.setError(f"NCSNR requires at least 2 repetitions per condition. Insufficient conditions (ID: nReps): {details}")
                return None

            # This preserves the order in which usable condition data were stored.
            conditionIDs = list(conditionData)

            # ---------------------------------------------------------------------
            # 1. Estimate typical single-trial noise.
            #
            # Within each condition, calculate trial-to-trial variance at every
            # time point. Average those variance estimates across conditions.
            # ---------------------------------------------------------------------
            noiseVarByCondition = np.asarray([np.var(conditionData[conditionID], axis=0, ddof=1) for conditionID in conditionIDs])

            noiseVar = noiseVarByCondition.mean(axis=0)
            noiseSTD = np.sqrt(noiseVar)

            # ---------------------------------------------------------------------
            # 2. Estimate noise-corrected signal variance across condition means.
            # ---------------------------------------------------------------------
            conditionMeansArray = np.asarray([conditionMeans[conditionID] for conditionID in conditionIDs])
            repsArray = np.asarray([nRepsPerCondition[conditionID] for conditionID in conditionIDs])

            observedVar = np.var(conditionMeansArray, axis=0, ddof=1)

            residualNoiseVar = np.mean(noiseVarByCondition / repsArray[:, np.newaxis], axis=0)

            # The estimate can be slightly negative due to finite sampling.
            signalVar = np.maximum(observedVar - residualNoiseVar, 0)
            signalSTD = np.sqrt(signalVar)

            # ---------------------------------------------------------------------
            # 3. NCSNR = signal SD / typical single-trial noise SD.
            # ---------------------------------------------------------------------
            ncsnr = np.divide(signalSTD, noiseSTD, out=np.full_like(signalSTD, np.nan, dtype=float), where=noiseSTD > 0)

            # Each condition receives equal weight, even if repetition counts vary.
            grandMean = conditionMeansArray.mean(axis=0)
            times = epochs.times.copy()

            if np.all(np.isnan(ncsnr)):
                peakIndex = None
                peakTime = np.nan
                peakNcsnr = np.nan
            else:
                peakIndex = np.nanargmax(ncsnr)
                peakTime = times[peakIndex]
                peakNcsnr = ncsnr[peakIndex]

            # ---------------------------------------------------------------------
            # Store results.
            # ---------------------------------------------------------------------
            session.mne.ncsnr = {
                "channelName": channelName,
                "chType": self.chType,
                "set": self.set,
                "label": self.label,
                "conditionIDColumn": self.conditionIDColumn,
                "conditionIDs": conditionIDs,
                "nConditions": len(conditionIDs),
                "nRepsPerCondition": nRepsPerCondition,
                "times": times,
                "conditionMeans": conditionMeansArray,
                "grandMean": grandMean,
                "noiseVarByCondition": noiseVarByCondition,
                "noiseVar": noiseVar,
                "noiseSTD": noiseSTD,
                "observedVar": observedVar,
                "residualNoiseVar": residualNoiseVar,
                "signalVar": signalVar,
                "signalSTD": signalSTD,
                "ncsnr": ncsnr,
                "peakIndex": peakIndex,
                "peakTime": peakTime,
                "peakNcsnr": peakNcsnr,
                "evokedPeakTime": evokedPeakTime,
                "evokedPeakAmplitude": evokedPeakAmplitude,
            }

            # ---------------------------------------------------------------------
            # Plot.
            # ---------------------------------------------------------------------
            with plt.rc_context({"figure.constrained_layout.use": False, "figure.autolayout": False}):
                fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True, layout=None)

                axes[0].plot(times, grandMean, color="black", linewidth=1.25)
                axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
                axes[0].set_ylabel("Grand mean\n(native units)")
                axes[0].set_title(f"Grand mean across {len(conditionIDs)} conditions — {channelName}")
                axes[0].grid(alpha=0.25)

                axes[1].plot(times, ncsnr, color="tab:blue", linewidth=1.25)
                axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)

                if peakIndex is not None:
                    axes[1].axvline(peakTime, color="tab:red", linestyle="--", linewidth=1)
                    axes[1].plot(peakTime, peakNcsnr, marker="o", color="tab:red")
                    axes[1].annotate(f"Peak: {peakNcsnr:.3f}\n{peakTime * 1000:.1f} ms", xy=(peakTime, peakNcsnr), xytext=(8, 8), textcoords="offset points", color="tab:red")

                axes[1].set_xlabel("Time (s)")
                axes[1].set_ylabel("Noise-corrected SNR")
                axes[1].set_title("NCSNR across individual conditions")
                axes[1].grid(alpha=0.25)

                fig.suptitle(f"NCSNR: {self.set} = {self.label} | condition IDs from {self.conditionIDColumn}", fontsize=14)
                fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, hspace=0.35)

                plt.show()

            return session
 
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # eyeblink detecion
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneEyeblinkDetection(pglAction):

        # parameters
        leftEyeChannel = Unicode('L401', help="Left channel used for detecting eye blink")
        rightEyeChannel = Unicode('R401', help="Left channel used for detecting eye blink")
                
        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            # set parameters
            self.configureTraits(**kwargs)
                
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''            
            Returns:
                pglSession:  session
            '''
            # import mne
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            # look up left and right eye channels
            leftEyeChannel = session.mne.lookupSensor(self.leftEyeChannel)
            if leftEyeChannel is None:
                pglMessages.warning(f"Could not find left eye channel: {self.leftEyeChannel}")
            rightEyeChannel = session.mne.lookupSensor(self.rightEyeChannel)
            if rightEyeChannel is None:
                pglMessages.warning(f"Could not find left eye channel: {self.rightEyeChannel}")
            
            # extract data from two frontal OPM sensors
            ch1_data = raw.copy().pick_channels([self.rightEyeChannel]).get_data()[0] 
            ch2_data = raw.copy().pick_channels(['L401_bz-s92']).get_data()[0] 
            # differential signal (Left - Right)
            occ_signal = ch1_data - ch2_data
            # create new channel called EOG
            sfreq = raw.info['sfreq']
            new_ch_name = 'EOG' 
            info_new = mne.create_info([new_ch_name], sfreq=sfreq, ch_types=['eog'])            

            # and return
            return session

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # align
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class alignSessionToMNE(pglAction):

        # parameters
        runNum = Int(0, min=0, help='run in session for alignment')
        taskName = Unicode('', help='task used for alignment')
        taskID = Int(allow_none=True, default_value=None, help='ID of task, used to disambiguate when there are multiple instance of a task in the experiment')
        trialNum = Int(0, min=0, help='trial in run for alignment')
        segmentNum = Int(0, min=0, help='segment in trial for alignment')
        
        channelName = Unicode("", help='name of channel in mne for alignment')
        pulseNum = Int(0, min=0, help='Number of pulse in channel to use for alignment')
        highCutoff = Float(0.5, help='Cutoff value of channel to consider as an event')
                
        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            # set parameters
            self.configureTraits(**kwargs)
                
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Aligns triggers as recorded in eeg/meg session to trial events as recorded in session
            
            Returns:
                pglSession:  session
            '''
            # validate runNum
            if self.runNum > len(session.runs):
                self.setError(f"Specififed alignment run ({self.runNum}) is not in range of number of runs: 0-{len(session.runs)}")
                return None
            
            # get the run
            run = session.runs[self.runNum]
            
            # get the task, if taskName is not set get the task that has the largest number of trials run, preferring
            # tasks in which settings.nTrials has been set (i.e. that have been set to run nTrials and have completed those trials)
            if self.taskName == '':
                hasNTrialsSet = False
                task = None

                # for each task 
                for t in run.tasks:
                    # for the first task, select it
                    if task is None:
                        task = t
                        hasNTrialsSet = t.settings.nTrials != np.inf
                        continue
                    # if all the previous tasks have not had infinite trails set
                    if not hasNTrialsSet:
                        # then either the task has more trials than the last one, or it has a finite number of trials set
                        if t.settings.nTrials != np.inf or t.data.nTrials > task.data.nTrials:
                            task = t
                            hasNTrialsSet = t.settings.nTrials != np.inf
                    # only tasks that have nTrials set and have larger number of trials
                    elif t.settings.nTrials != np.inf and t.data.nTrials > task.data.nTrials:
                        task = t
            else:
                # named task asked for, so try to find it 
                matchingTasks = [(iTask, t) for iTask, t in enumerate(run.tasks) if t.taskName == self.taskName]

                # no match
                if not matchingTasks:
                    self.setError(f"No task matches taskName '{self.taskName}'")
                    return None

                #single match just return it
                if len(matchingTasks) == 1:
                    task = matchingTasks[0][1]
                    
                # muultiple matches, see if taskID is se
                elif self.taskID is None:
                    self.setError(f"Multiple tasks match taskName '{self.taskName}'; specify taskID")
                    return None
                else:
                    task = next((t for iTask, t in matchingTasks if iTask == self.taskID), None)
                    if task is None:
                        self.setError(f"No task matches taskName '{self.taskName}' and taskID {self.taskID}")
                        return None
            
            # validate trialNum
            if self.trialNum >= task.data.nTrials:
                self.setError(f"trialNum {self.trialNum} out of range: 0-{task.data.nTrials}")
                return None
                            
            # validate segNum
            if self.segmentNum >= len(task.settings.seglen):
                self.setError(f"segmentNum {self.segmentNum} out of range: 0-{len(task.settings.seglen)}")
                return None
                    
            # get session reference time
            iTrial = -1
            sessionReferenceTime = None
            for e in task.data.events:
                if isinstance(e, pglEventTrial):
                    iTrial+=1
                elif isinstance(e, pglEventSegment):
                    # if we found the matching trial and segment
                    if self.trialNum==iTrial and e.segmentNum == self.segmentNum:
                        sessionReferenceTime = e.timestamp
            
            # no session reference time
            if sessionReferenceTime is None:
                self.setError(f"Could not find matching trial and segment number for reference time")
                return None
                   
            pglMessages.message(f"Found session reference time: {sessionReferenceTime}")     
            
            # set all runs to have this sessionReferenceTime
            for run in session.runs:
                run.data.referenceTime = sessionReferenceTime            
            
            # import mne
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            if not isinstance(self.channelName, str) or self.channelName not in session.mne.raw.ch_names:
                self.setError(f"Unknown channelName: {self.channelName!r}")
                return None

            channelData = session.mne.raw.get_data(picks=[self.channelName])[0]
            isHigh = channelData > self.highCutoff

            # Find low-to-high transitions; count an initially high channel as a pulse.
            pulseSamples = np.flatnonzero(np.diff(isHigh.astype(int), prepend=0) == 1)

            if self.pulseNum > len(pulseSamples):
                self.setError(f"Requested pulse {self.pulseNum}, but found only {len(pulseSamples)} pulses on {self.channelName!r}")
                return None

            session.mne.referenceTime = float(session.mne.raw.times[pulseSamples[self.pulseNum]])
            
            pglMessages.message(f"Found mne reference time: {session.mne.referenceTime}")

            # and return
            return session

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # action stub
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class stub(pglAction):

        # parameters
                
        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            # set parameters
            self.configureTraits(**kwargs)
                
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            
            
            Returns:
                pglSession:  session
            '''
            # import mne
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            # check for mne epochs
            if session.mne.epochs is None:
                self.setError("session does not have epochs created: run mneCreateEpochs")
                return None

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
    



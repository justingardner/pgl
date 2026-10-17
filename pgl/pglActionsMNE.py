################################################################
#   filename: pglActionsMNE.py
#    purpose: actions for processing EEG/MEG data using mne-python
#         by: JLG
#       date: Oct 12, 2026
################################################################
from .pglMessages import pglMessages
from pathlib import Path
from .pglExperiment import pglEventSegment
from .pglSettings import pglSettings
from .pglChoose import pglChoose
from .pglPipeline import pglAction, pglActionCollection
from .pglSession import pglSession, pglMNE
import numpy as np
from numbers import Integral
import matplotlib.pyplot as plt
from fsspec import AbstractFileSystem
from traitlets import HasTraits, Enum, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance, Union
import pandas as pd
from .pglExperiment import pglEventTrial, pglEventSegment
from tqdm.auto import tqdm
from typing import Callable

class pglActionsMNE(pglActionCollection):
    
    # for printing 
    _actionHeader_loading = "Load recordings"
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # load FieldLine
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class loadFieldline(pglAction):
        '''
        Load fif files created by Fieldline
        '''
            
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
                
                    # and add to the report
                    mneData.report.add_raw(raw, title=f"Raw: {fifPath}", psd=True)

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
        '''
        Load mff files created by Net Station
        '''

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

    _actionHeader_alignment = "Alignment of session to MNE data"
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
            
            # get the task
            task = run.getTask(taskName=self.taskName, taskID=self.taskID)
            if task is None:
                self.setError("Could not get task")
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
    class alignSegmentsToTriggers(pglAction):

        # Parameters
        segments = List(Int(), allow_none=True, default_value=None, help='List of segments to align to')
        taskName = Unicode('', help='Task used for alignment')
        taskID = Int(
            allow_none=True,
            default_value=None,
            help='ID of task, used to disambiguate multiple instances of a task',
        )

        channelName = Unicode('', help='Name of channel in MNE for alignment')
        highCutoff = Float(0.5, help='Cutoff value of channel to consider as an event')

        outputChannelName = Unicode('trigger',help='Name of output trigger channel; empty string disables output')
        useSegmentTimeAsTruth = Bool(True, help='Use corrected segment times rather than recorded digital pulse times for output triggers')

        ################################
        # configure
        ################################
        def configure(self, segments: int | list | None = None, **kwargs) -> None:

            if segments is None:
                segments = 0

            if isinstance(segments, int):
                self.segments = [segments]
            else:
                self.segments = segments

            self.configureTraits(**kwargs)
            super().configure()

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:

            import mne

            def histogramEdges(values):
                if values.size == 0:
                    return np.linspace(0, 1, 81)

                lower, upper = float(np.min(values)), float(np.max(values))
                padding = max((upper - lower) * 0.025, abs(lower) * 1e-9, 1e-6)
                return np.linspace(lower - padding, upper + padding, 81)

            def plotHistogram(ax, values, edges, color, title, xlabel):
                ax.hist(values, bins=edges, edgecolor="black", linewidth=0.4, color=color, alpha=0.8)
                ax.set_xlim(edges[0], edges[-1])
                ax.set_title(f"{title} (n={values.size})")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
                ax.set_axisbelow(True)
                ax.grid(axis="y", alpha=0.3)

                if values.size:
                    medianMs = float(np.median(values))
                    ax.axvline(medianMs, color="red", linestyle=":", label=f"Median: {medianMs:.6g} ms")
                    ax.legend()

            def plotIntervalWindow(ax, values, edges, color, title, rangeName):
                visibleMask = (values >= edges[0]) & (values <= edges[-1])
                visibleValues = values[visibleMask]
                nMissing = int(values.size - visibleValues.size)
                windowTitle = f"{title} — {rangeName} range; {nMissing}/{values.size} outside window"
                plotHistogram(ax, visibleValues, edges, color, windowTitle, "Inter-pulse interval (ms)")

            # Validate MNE data and the common reference.
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw MNE loaded")
                return None

            raw = session.mne.raw
            sfreq = float(raw.info["sfreq"])

            if not isinstance(self.channelName, str) or self.channelName not in raw.ch_names:
                self.setError(f"Unknown channelName: {self.channelName!r}")
                return None

            mneReferenceTime = getattr(session.mne, "referenceTime", None)
            if mneReferenceTime is None or not np.isfinite(mneReferenceTime):
                self.setError("MNE referenceTime is missing or invalid; run alignment first")
                return None

            if self.outputChannelName:
                if not self.outputChannelName.strip():
                    self.setError("outputChannelName must not contain only whitespace")
                    return None

                if self.outputChannelName in raw.ch_names:
                    pglMessages.message(f"Output channel {self.outputChannelName!r} already exists. Overwriting")

            # Detect onsets, counting an initially high channel as a trigger.
            channelData = raw.get_data(picks=[self.channelName])[0]
            isHigh = channelData > self.highCutoff
            triggerIndexes = np.flatnonzero(np.diff(isHigh.astype(int), prepend=0) == 1)
            triggerTimes = raw.times[triggerIndexes]
            triggerAlignedTimes = triggerTimes - mneReferenceTime
            nTriggers = triggerAlignedTimes.size

            if nTriggers == 0:
                self.setError("No triggers detected")
                return None

            usedTriggerMask = np.zeros(nTriggers, dtype=bool)
            runResults = []

            # Match and correct each run independently; keep plotting data local.
            for runNum, run in enumerate(session.runs):
                task = run.getTask(taskName=self.taskName, taskID=self.taskID)

                if task is None:
                    self.setError(f"Could not get task for run {runNum}")
                    return None

                oldReferenceTime = getattr(run.data, "referenceTime", None)
                if oldReferenceTime is None or not np.isfinite(oldReferenceTime):
                    self.setError(f"Run {runNum} referenceTime is missing or invalid; run alignment first")
                    return None

                segmentTimestamps = np.asarray([
                    e.timestamp for e in task.data.events
                    if isinstance(e, pglEventSegment) and e.segmentNum in self.segments
                ], dtype=float)

                if segmentTimestamps.size == 0:
                    self.setError(f"No segment events matched self.segments in run {runNum}")
                    return None

                if not np.all(np.isfinite(segmentTimestamps)):
                    self.setError(f"Run {runNum} has non-finite segment timestamps")
                    return None

                if segmentTimestamps.size > nTriggers:
                    self.setError(f"Run {runNum} has more selected segments than available triggers")
                    return None

                segmentAlignedTimes = segmentTimestamps - oldReferenceTime

                # Find the nearest trigger on either side; prefer the earlier one in a tie.
                insertionIndexes = np.searchsorted(triggerAlignedTimes, segmentAlignedTimes)
                leftIndexes = np.clip(insertionIndexes - 1, 0, nTriggers - 1)
                rightIndexes = np.clip(insertionIndexes, 0, nTriggers - 1)
                leftDistances = np.abs(triggerAlignedTimes[leftIndexes] - segmentAlignedTimes)
                rightDistances = np.abs(triggerAlignedTimes[rightIndexes] - segmentAlignedTimes)
                matchedIndexes = np.where(leftDistances <= rightDistances, leftIndexes, rightIndexes)

                # Preserve one-to-one matching within and across runs.
                if np.unique(matchedIndexes).size != matchedIndexes.size:
                    self.setError(f"Run {runNum}: multiple segments selected the same nearest trigger")
                    return None

                if np.any(usedTriggerMask[matchedIndexes]):
                    self.setError(f"Run {runNum}: a matched trigger was already assigned to another run")
                    return None

                usedTriggerMask[matchedIndexes] = True
                matchedTriggerAlignedTimes = triggerAlignedTimes[matchedIndexes]

                # Decrease the reference by the median trigger-minus-segment offset.
                originalDifferences = matchedTriggerAlignedTimes - segmentAlignedTimes
                medianOffsetSeconds = float(np.median(originalDifferences))
                newReferenceTime = float(oldReferenceTime - medianOffsetSeconds)

                # Keep the established pairs and compute corrected residuals.
                correctedSegmentTimes = segmentTimestamps - newReferenceTime
                timeDifferencesMs = (matchedTriggerAlignedTimes - correctedSegmentTimes) * 1000

                # Compute intervals within this run only, using its matched triggers.
                triggerPlotTimes = np.sort(matchedTriggerAlignedTimes)
                segmentPlotTimes = np.sort(correctedSegmentTimes)
                triggerIntervalsMs = np.diff(triggerPlotTimes) * 1000
                segmentIntervalsMs = np.diff(segmentPlotTimes) * 1000

                runResults.append({
                    "run": run,
                    "runNum": runNum,
                    "referenceTime": newReferenceTime,
                    "referenceShiftMs": -medianOffsetSeconds * 1000,
                    "correctedSegmentTimes": correctedSegmentTimes.copy(),
                    "matchedTriggerAlignedTimes": matchedTriggerAlignedTimes.copy(),
                    "triggerPlotTimes": triggerPlotTimes,
                    "segmentPlotTimes": segmentPlotTimes,
                    "triggerIntervalsMs": triggerIntervalsMs,
                    "segmentIntervalsMs": segmentIntervalsMs,
                    "timeDifferencesMs": timeDifferencesMs,
                })

            if not runResults:
                self.setError("Session has no runs")
                return None

            # Validate and prepare the output channel before modifying raw or runs.
            stimRaw = None

            if self.outputChannelName:
                sampleArrays = []
                valueArrays = []

                for result in runResults:
                    run = result["run"]
                    runNum = result["runNum"]

                    # Preserve segment-event order so trigger codes stay paired
                    # with the corresponding events.
                    eventAlignedTimes = (
                        result["correctedSegmentTimes"]
                        if self.useSegmentTimeAsTruth
                        else result["matchedTriggerAlignedTimes"]
                    )

                    triggerValues = getattr(run, "triggers", None)
                    if triggerValues is None:
                        self.setError(f"Run {runNum}: run.triggers is missing")
                        return None

                    try:
                        triggerValues = np.asarray(triggerValues)
                    except (TypeError, ValueError) as exc:
                        self.setError(f"Run {runNum}: could not read run.triggers: {exc}")
                        return None

                    if triggerValues.ndim != 1:
                        self.setError(f"Run {runNum}: run.triggers must be a one-dimensional array")
                        return None

                    if triggerValues.size != eventAlignedTimes.size:
                        self.setError(
                            f"Run {runNum}: run.triggers has {triggerValues.size} values, "
                            f"but there are {eventAlignedTimes.size} selected segment events"
                        )
                        return None

                    # Reject strings, objects, complex values, and booleans.
                    if triggerValues.dtype.kind not in "iuf":
                        self.setError(f"Run {runNum}: run.triggers must contain numeric integer codes")
                        return None

                    # Zero is reserved for baseline. Codes must be exactly
                    # representable in MNE's floating-point raw data.
                    if (
                        not np.all(np.isfinite(triggerValues))
                        or np.any(triggerValues <= 0)
                        or np.any(triggerValues > 2**53 - 1)
                        or np.any(triggerValues != np.floor(triggerValues))
                    ):
                        self.setError(
                            f"Run {runNum}: trigger codes must be finite positive integers "
                            f"no larger than {2**53 - 1}; zero is reserved for baseline"
                        )
                        return None

                    # Return from the common aligned timebase to raw-relative seconds.
                    eventRawTimes = eventAlignedTimes + mneReferenceTime

                    if not np.all(np.isfinite(eventRawTimes)):
                        self.setError(f"Run {runNum}: output trigger times are non-finite")
                        return None

                    # Index into raw's data. Do not add raw.first_samp here.
                    eventSamples = raw.time_as_index(eventRawTimes, use_rounding=True)

                    outside = (eventSamples < 0) | (eventSamples >= raw.n_times)
                    if np.any(outside):
                        self.setError(
                            f"Run {runNum}: {np.count_nonzero(outside)} output triggers "
                            "fall outside the raw recording"
                        )
                        return None

                    sampleArrays.append(eventSamples)
                    valueArrays.append(triggerValues.astype(np.float64))

                    # Retain each run's generated sample positions for plotting.
                    result["outputSamples"] = np.sort(eventSamples)

                allSamples = np.concatenate(sampleArrays)
                allValues = np.concatenate(valueArrays)

                # Sort samples and codes together, including across run boundaries.
                order = np.argsort(allSamples)
                allSamples = allSamples[order]
                allValues = allValues[order]

                sampleGaps = np.diff(allSamples)

                if np.any(sampleGaps == 0):
                    self.setError("Multiple output triggers map to the same raw sample")
                    return None

                # Require a zero sample between pulses so repeated codes remain distinct.
                if np.any(sampleGaps == 1):
                    self.setError(
                        "Output triggers occupy adjacent samples; one-sample pulses "
                        "need at least one zero sample between them"
                    )
                    return None

                # Build one-sample pulses directly on the original raw sample grid.
                triggerTrace = np.zeros(raw.n_times, dtype=np.float64)
                triggerTrace[allSamples] = allValues

                stimInfo = mne.create_info(
                    ch_names=[self.outputChannelName],
                    sfreq=sfreq,
                    ch_types=["stim"],
                )
                stimRaw = mne.io.RawArray(
                    triggerTrace[np.newaxis, :],
                    stimInfo,
                    first_samp=raw.first_samp,
                    verbose=False,
                )
                stimRaw.set_meas_date(raw.info["meas_date"])

            # Pool within-run intervals, never concatenated pulse times.
            triggerIntervalsMs = np.concatenate([r["triggerIntervalsMs"] for r in runResults])
            segmentIntervalsMs = np.concatenate([r["segmentIntervalsMs"] for r in runResults])
            timeDifferencesMs = np.concatenate([r["timeDifferencesMs"] for r in runResults])

            triggerEdges = histogramEdges(triggerIntervalsMs)
            segmentEdges = histogramEdges(segmentIntervalsMs)
            differenceEdges = histogramEdges(timeDifferencesMs)

            # Add a generated-trigger subplot directly below each run overlay.
            hasOutput = stimRaw is not None
            rowsPerRun = 2 if hasOutput else 1
            nRows = 3 + rowsPerRun * len(runResults)

            fig = plt.figure(figsize=(15, 2.8 * nRows), constrained_layout=True)
            grid = fig.add_gridspec(nRows, 2)

            triggerLeftAx = fig.add_subplot(grid[0, 0])
            triggerRightAx = fig.add_subplot(grid[0, 1])
            segmentLeftAx = fig.add_subplot(grid[1, 0], sharex=triggerLeftAx)
            segmentRightAx = fig.add_subplot(grid[1, 1], sharex=triggerRightAx)
            differenceAx = fig.add_subplot(grid[2, :])

            # Left column: trigger range. Right column: segment range.
            plotIntervalWindow(
                triggerLeftAx, triggerIntervalsMs, triggerEdges,
                "tab:blue", "Trigger intervals", "trigger",
            )
            plotIntervalWindow(
                segmentLeftAx, segmentIntervalsMs, triggerEdges,
                "tab:orange", "Segment intervals", "trigger",
            )
            plotIntervalWindow(
                triggerRightAx, triggerIntervalsMs, segmentEdges,
                "tab:blue", "Trigger intervals", "segment",
            )
            plotIntervalWindow(
                segmentRightAx, segmentIntervalsMs, segmentEdges,
                "tab:orange", "Segment intervals", "segment",
            )

            plotHistogram(
                differenceAx, timeDifferencesMs, differenceEdges, "tab:purple",
                "Matched timing differences after per-run correction",
                "Trigger time − segment time (ms)",
            )
            differenceAx.axvline(0, color="black", linestyle="--", alpha=0.5)

            for runIndex, result in enumerate(runResults):
                rowNum = 3 + rowsPerRun * runIndex
                ax = fig.add_subplot(grid[rowNum, :])

                triggerPlotTimes = result["triggerPlotTimes"]
                segmentPlotTimes = result["segmentPlotTimes"]
                plotOrigin = segmentPlotTimes[0]

                ax.eventplot(
                    triggerPlotTimes - plotOrigin,
                    lineoffsets=0,
                    linelengths=0.8,
                    colors="tab:blue",
                    linewidths=1.5,
                    label="Matched triggers",
                    zorder=2,
                )
                ax.eventplot(
                    segmentPlotTimes - plotOrigin,
                    lineoffsets=0,
                    linelengths=0.4,
                    colors="tab:orange",
                    linewidths=2,
                    label="Session pulses",
                    zorder=3,
                )

                shiftMs = result["referenceShiftMs"]
                ax.set_title(
                    f"Run {result['runNum']} — {segmentPlotTimes.size} pairs; "
                    f"reference shift={shiftMs:.6f} ms"
                )
                ax.set_ylabel("Events")
                ax.set_yticks([])
                ax.set_ylim(-0.6, 0.6)
                ax.legend(loc="upper right")
                ax.grid(axis="x", alpha=0.3)

                xlabel = "Time relative to first corrected selected segment in this run (s)"

                if hasOutput:
                    outputAx = fig.add_subplot(grid[rowNum + 1, :], sharex=ax)
                    outputSamples = result["outputSamples"]

                    # Read actual generated codes from the trace, at their rounded
                    # raw sample positions, then convert to the overlay's timebase.
                    outputTimes = outputSamples / sfreq - mneReferenceTime - plotOrigin
                    outputValues = triggerTrace[outputSamples]

                    # Draw each one-sample pulse with its true width and amplitude.
                    # NaN separators avoid joining separate pulses.
                    pulseTimes = np.column_stack((
                        outputTimes,
                        outputTimes,
                        outputTimes + 1 / sfreq,
                        outputTimes + 1 / sfreq,
                        np.full(outputTimes.size, np.nan),
                    )).ravel()
                    pulseValues = np.column_stack((
                        np.zeros(outputValues.size),
                        outputValues,
                        outputValues,
                        np.zeros(outputValues.size),
                        np.full(outputValues.size, np.nan),
                    )).ravel()

                    outputAx.axhline(0, color="black", linewidth=0.6, alpha=0.5)
                    outputAx.plot(
                        pulseTimes, pulseValues,
                        color="tab:green", linewidth=1,
                        label=self.outputChannelName,
                    )
                    outputAx.scatter(
                        outputTimes, outputValues,
                        color="tab:green", s=12, zorder=3,
                    )

                    timingSource = (
                        "corrected segment times"
                        if self.useSegmentTimeAsTruth
                        else "matched recorded pulse times"
                    )
                    outputAx.set_title(
                        f"Run {result['runNum']} — generated triggers "
                        f"(n={outputSamples.size}; {timingSource})"
                    )
                    outputAx.set_xlabel(xlabel)
                    outputAx.set_ylabel("Trigger code")
                    outputAx.ticklabel_format(axis="y", style="plain", useOffset=False)
                    outputAx.set_ylim(-0.05 * outputValues.max(), 1.1 * outputValues.max())
                    outputAx.legend(loc="upper right")
                    outputAx.grid(alpha=0.3)
                    ax.tick_params(axis="x", labelbottom=False)

                    # Both plots use the same limits and time origin.
                    left = min(
                        triggerPlotTimes[0] - plotOrigin,
                        segmentPlotTimes[0] - plotOrigin,
                        outputTimes[0],
                    )
                    right = max(
                        triggerPlotTimes[-1] - plotOrigin,
                        segmentPlotTimes[-1] - plotOrigin,
                        outputTimes[-1] + 1 / sfreq,
                    )
                    padding = max((right - left) * 0.025, 2 / sfreq)
                    ax.set_xlim(left - padding, right + padding)
                else:
                    ax.set_xlabel(xlabel)

            plt.show()

            session.mne.report.add_figure(fig=plt.gcf(), title="Align segments to triggers")

            # All runs and output triggers have validated.
            # Add the channel, then commit corrected run reference times.
            # Add or overwrite the output channel.
            if stimRaw is not None:
                raw.load_data()

                if self.outputChannelName in raw.ch_names:
                    raw.apply_function(
                        lambda data: triggerTrace.copy(),
                        picks=[self.outputChannelName],
                        channel_wise=True,
                    )
                    raw.set_channel_types({self.outputChannelName: "stim"})
                else:
                    raw.add_channels([stimRaw], force_update_info=True)

            # Commit corrected run reference times.
            for result in runResults:
                result["run"].data.referenceTime = result["referenceTime"]

            return session
        
    # for printing 
    _actionHeader_preprocessing = "Basic preprocessing steps"

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # concatenate
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class concatenate(pglAction):
        '''
        Concatenate mne raw runs into a single raw
        '''
        
        
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
    # filter
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class filter(pglAction):
        '''
        Filter raw data
        '''
        
        lowCutoff = Float(1.0, help="Low pass cutoff for filtering")
        highCutoff = Float(80.0, help="High pass cutoff for filtering")
        notch = Bool(True, help="apply notch filter")
        notchFrequency = Float(60.0, help="Frequency at which to notch")
            
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
                fig = spectrum.plot(picks=plotChannels, average=False, amplitude=False)
                session.mne.report.add_figure(fig=fig, title="Power spectrum")
            else:
                pglMessages.warning("No channels with valid nonzero power to plot")
            
            # and return
            return session
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # bads handling
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class bads(pglAction):
        '''
        Handle bads, selecting either to interpolate or drop
        '''
        
        # parameters
        method = Enum(["interpolate", "drop"], default_value="drop", help="Method for handling bads")
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
    # mark inter runs as bad
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class markInterRunsAsBad(pglAction):
        '''
        mark time inbetween runs (as defined by epochs) as bad
        '''
    
        tmin = Float(0.0, help="Start of each epoch window relative to its trigger, in seconds")
        tmax = Float(1.0, help="End of each epoch window relative to its trigger, in seconds")

        def _run(self, session: pglSession) -> pglSession:
            """
            Mark samples outside the union of complete epoch windows as bad,
            and plot all events with the excluded intervals shaded red.

            Existing annotations are preserved, except annotations previously
            created by this action, which are replaced.
            """
            import mne
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            # ---------------------------------------------------------------------
            # Validate inputs.
            # ---------------------------------------------------------------------
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            if session.mne.events is None or len(session.mne.events) == 0:
                self.setError("session does not have events configured")
                return None

            if not np.isfinite(self.tmin) or not np.isfinite(self.tmax) or self.tmin > self.tmax:
                self.setError("tmin and tmax must be finite, with tmin <= tmax")
                return None

            raw = session.mne.raw
            events = np.asarray(session.mne.events)
            sfreq = float(raw.info["sfreq"])
            nSamples = raw.n_times
            recordingEnd = nSamples / sfreq
            badDescription = "BAD_inter_run"

            # ---------------------------------------------------------------------
            # Compute epoch windows in sample coordinates relative to raw start.
            #
            # MNE includes both tmin and tmax. Represent each window as [start, stop),
            # so stop is one sample beyond the final included sample.
            # ---------------------------------------------------------------------
            startOffset = int(round(self.tmin * sfreq))
            endOffset = int(round(self.tmax * sfreq))

            eventSamples = events[:, 0].astype(np.int64) - raw.first_samp
            starts = eventSamples + startOffset
            stops = eventSamples + endOffset + 1

            # MNE drops epochs extending outside the recording. Do not reserve
            # samples for these incomplete epochs.
            completeMask = (starts >= 0) & (stops <= nSamples)
            windows = sorted(zip(starts[completeMask].tolist(), stops[completeMask].tolist()))

            # ---------------------------------------------------------------------
            # Merge overlapping or touching epoch windows.
            # ---------------------------------------------------------------------
            keptIntervals = []

            for start, stop in windows:
                if keptIntervals and start <= keptIntervals[-1][1]:
                    keptIntervals[-1][1] = max(keptIntervals[-1][1], stop)
                else:
                    keptIntervals.append([start, stop])

            # ---------------------------------------------------------------------
            # Find the complement: samples not covered by any complete epoch.
            # ---------------------------------------------------------------------
            badIntervals = []
            cursor = 0

            for start, stop in keptIntervals:
                if cursor < start:
                    badIntervals.append((cursor, start))
                cursor = stop

            if cursor < nSamples:
                badIntervals.append((cursor, nSamples))

            badOnsets = np.asarray([start / sfreq for start, stop in badIntervals], dtype=float)
            badDurations = np.asarray([(stop - start) / sfreq for start, stop in badIntervals], dtype=float)

            # ---------------------------------------------------------------------
            # Preserve unrelated annotations and replace our previous annotations.
            #
            # With orig_time=None, set_annotations() expects onsets relative to the
            # current raw start and adds raw.first_time internally.
            #
            # With an absolute orig_time, stored onsets already include first_time.
            # ---------------------------------------------------------------------
            annotations = raw.annotations.copy()
            annotations.delete(np.flatnonzero(annotations.description == badDescription))

            if annotations.orig_time is None:
                annotations.onset -= raw.first_time
                newOnsets = badOnsets
            else:
                newOnsets = badOnsets + raw.first_time

            newAnnotations = mne.Annotations(onset=newOnsets, duration=badDurations, description=[badDescription] * len(badIntervals), orig_time=annotations.orig_time)
            raw.set_annotations(annotations + newAnnotations)

            # ---------------------------------------------------------------------
            # Plot every event using its canonical raw trigger code.
            # Both event times and shaded intervals are relative to raw start.
            # ---------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(24, 8))

            rawCodes = events[:, 2].astype(int)
            plotEventId = {f"raw/{int(code)}": int(code) for code in np.unique(rawCodes)}

            mne.viz.plot_events(events, event_id=plotEventId, sfreq=sfreq, first_samp=raw.first_samp, axes=ax, show=False)

            for start, stop in badIntervals:
                ax.axvspan(start / sfreq, stop / sfreq, facecolor="red", alpha=0.2, edgecolor="none", zorder=0)

            ax.set_xlim(0, recordingEnd)
            ax.set_xlabel("Time from recording start (s)")
            ax.set_title(f"Events and excluded intervals — epoch window [{self.tmin:g}, {self.tmax:g}] s")

            # Keep MNE's event legend and add a separate shading legend.
            eventLegend = ax.get_legend()
            if eventLegend is not None:
                ax.add_artist(eventLegend)

            excludedPatch = Patch(facecolor="red", alpha=0.2, label="Outside complete epoch windows")
            ax.legend(handles=[excludedPatch], loc="upper left")

            excludedSeconds = float(badDurations.sum())
            fig.suptitle(f"Excluded: {excludedSeconds:.2f} s / {recordingEnd:.2f} s ({100 * excludedSeconds / recordingEnd:.1f}%)")
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            if getattr(session.mne, "report", None) is not None:
                session.mne.report.add_figure(fig=fig, title="Inter-epoch excluded intervals")
            else:
                plt.show()

            return session        
        
    _actionHeader_event = "Events and epoching"
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # configure events
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class configureEvents(pglAction):
        '''
        Configure events by providing trigger channel and labels
        '''
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
            
            session.mne.report.add_figure(fig=fig, title="Configure Events")

            return session
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # make epochs
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class createEpochs(pglAction):
        '''
        Create epochs based on what createEvents has designated
        '''

        # parameters
        tmin = Float(0.0, help="Time to start triggered epoch in seconds")
        tmax = Float(1.0, help="Time to end triggered epoch in seconds")
        baseline = Tuple(Float, Float, allow_none=True, default_value=None, help="Baseline for epochs, mne uses this tuple to find the beginning and end of the interval to get the average of and subtract that from the interval, set to None for no baselining")
                
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Makes epochs
            
            Returns:
                pglSession: session with mne raw data that has gone through event creation
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
                baseline=self.baseline,
                metadata=session.mne.eventsID,
                preload=True,
                verbose=False,
            )

            session.mne.epochs = epochs

            # and return
            return session
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # make epochs respecting Bads annotations from blinks
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class createEpochsWithBads(pglAction):
        '''
        Create epochs respecting where bads have been set (e.g., from eyeblink annotations).
        '''
        

        tmin = Float(0.0, help="Time to start triggered epoch in seconds")
        tmax = Float(1.0, help="Time to end triggered epoch in seconds")
        baseline = Tuple(Float, Float, allow_none=True, default_value=None, help="Baseline for epochs, mne uses this tuple to find the beginning and end of the interval to get the average of and subtract that from the interval, set to None for no baselining")
        badSegments = Enum(["nan", "drop"], default_value="nan", help="Samples inside BAD annotations (e.g. blinks). nan: keep the epoch, set those samples to NaN (use nan-aware statistics). drop: drop every epoch overlapping a BAD annotation (MNE default)")
        maxBadFraction = Float(1.0, min=0.0, max=1.0, help="nan mode: also drop epochs with more than this fraction of bad samples. 1.0 keeps all")
        resampleFrequency = Float(0.0, help="If > 0, epoch a copy of raw resampled to this rate (replaces mneDownsampleEvoked); 0 for no resampling")

        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:
            self.configureTraits(**kwargs)
            super().configure()

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Create the epochs. In nan mode samples inside BAD annotations are set to NaN (all channels),
            the baseline is the nan-mean of the baseline window (so bad samples never enter it), and
            epochs are dropped if they straddle a file-concatenation boundary (as MNE does), have no
            usable baseline sample, or exceed maxBadFraction.
            '''
            import mne

            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            raw, events = session.mne.raw, session.mne.events
            if self.resampleFrequency > 0:
                raw, events = raw.copy().load_data().resample(sfreq=self.resampleFrequency, npad="auto", events=events, verbose=False)
            sfreq = raw.info["sfreq"]

            # Every raw trigger code gets an event_id; the DataFrame is the authoritative label store.
            rawEventId = {f"raw/{int(code)}": int(code) for code in np.unique(events[:, 2].astype(int))}
            nanMode = self.badSegments == "nan"

            # In nan mode MNE must neither baseline (NaN would spread) nor reject; we do both ourselves.
            epochs = mne.Epochs(raw, events=events, event_id=rawEventId, tmin=self.tmin, tmax=self.tmax,
                                baseline=None if nanMode else self.baseline, metadata=session.mne.eventsID,
                                preload=True, reject_by_annotation=not nanMode, verbose=False)

            if nanMode:
                # bad[i, j]: is sample j of epoch i inside a BAD annotation (MNE's own definition)?
                index = (epochs.events[:, 0] - raw.first_samp + int(round(epochs.tmin * sfreq)))[:, None] + np.arange(len(epochs.times))
                bad = np.isnan(raw.get_data(picks=[0], reject_by_annotation="NaN")[0])[index]

                # concatenation boundaries (zero-duration annotations) as sample indices
                isBoundary = np.array([d.upper().endswith("BOUNDARY") for d in raw.annotations.description], dtype=bool)
                offset = raw.first_time if raw.info["meas_date"] is not None else 0.0
                boundary = np.round((raw.annotations.onset[isBoundary] - offset) * sfreq).astype(int)
                straddles = ((boundary[None, :] >= index[:, :1]) & (boundary[None, :] <= index[:, -1:])).any(axis=1)

                if self.baseline is not None:
                    inBase = (epochs.times >= self.baseline[0] - 1e-9) & (epochs.times <= self.baseline[1] + 1e-9)
                    noBaseline = bad[:, inBase].all(axis=1)
                else:
                    noBaseline = np.zeros(len(epochs), bool)

                reasons = np.where(straddles, "BAD_boundary", np.where(bad.mean(axis=1) > self.maxBadFraction, "BAD_fraction",
                                   np.where(noBaseline, "BAD_baseline", "")))
                drop = np.flatnonzero(reasons != "")
                nBefore = len(epochs)
                if len(drop):
                    epochs.drop(drop, reason=[str(r) for r in reasons[drop]], verbose=False)
                bad = bad[reasons == ""]

                data = epochs.get_data(copy=False)                      # view on the preloaded array
                for iEpoch in range(len(epochs)):
                    data[iEpoch][:, bad[iEpoch]] = np.nan
                if self.baseline is not None:
                    data -= np.nanmean(data[:, :, inBase], axis=2, keepdims=True)

                pglMessages.message(f"Epochs: kept {len(epochs)}/{nBefore}; {100 * bad.mean():.1f}% of samples set to NaN "
                                    f"({int(bad.any(axis=1).sum())} epochs contain NaN)"
                                    + (f"; dropped {dict(zip(*np.unique(reasons[drop], return_counts=True)))}" if len(drop) else ""))

            session.mne.epochs = epochs
            return session
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # downsample epochs
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class downsampleEpochs(pglAction):
        '''
        Downsample the epochs to the specified frequency.
        '''
        

        # parameters
        downsampleFrequency = Float(200.0, help="Downsample frequency")
                
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
                baseline=session.mne.epochs.baseline,
                preload=True,
                metadata=session.mne.eventsID.copy(),
                reject_by_annotation=True,
                verbose=False,
            )

            session.mne.epochs = epochs
            
            # and return
            return session
    
    _actionHeader_plots = "Make plots"
    
    #+ங+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # plot evoked
    #+ங+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    class plotEvoked(pglAction):

        # parameters
        set = Unicode(allow_none=True, default_value=None, help="Name of label set passed to configure events from which to compute evoked (defaults to first set)")
        label = Unicode(allow_none=True, default_value=None, help="Name of label passed in the set to configure events for which to compute the evoked (defaults to average across all event types)")
        picks = Unicode("mag", help="For topomap plotting type of topomap e.g. eeg, mag, grad - defined by mne - also can be name (or initial part of name of a sensor)")
        minFreq = Float(0.0, help="minimum frequency for display of frequency plot of evoked")
        maxFreq = Float(np.inf, help="maximum frequency for display of frequency plot of evoked")
        yMin = Float(default_value=None, allow_none=True, help="Shared minimum for evoked traces and topomap color ranges, in MNE display units. Set together with yMax; None uses automatic scaling.")
        yMax = Float(default_value=None, allow_none=True, help="Shared maximum for evoked traces and topomap color ranges, in MNE display units. Set together with yMin; None uses automatic scaling.")

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            """
            Display the evoked response.

            yMin and yMax control both the evoked trace limits and
            the color ranges of all topomaps. They do not affect
            FFT or PSD plots.

            Returns:
                pglSession: session
            """
            # Check for MNE session before accessing its methods.
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw mne loaded")
                return None

            if session.mne.epochs is None:
                self.setError("Session does not have epochs created")
                return None

            # Validate picks.
            self.picks = session.mne.validatePicks(self.picks)
            isSensor = session.mne.isSensor(self.picks)

            # Require both limits for a shared fixed scale.
            if (self.yMin is None) != (self.yMax is None):
                self.setError("Set both yMin and yMax, or leave both as None.")
                return None

            if self.yMin is not None:
                if not np.isfinite(self.yMin) or not np.isfinite(self.yMax):
                    self.setError("yMin and yMax must be finite values.")
                    return None

                if self.yMin >= self.yMax:
                    self.setError("yMin must be less than yMax.")
                    return None

            # Create the grand average.
            if self.set is None:
                self.set = session.mne.eventsID.columns[3]
            elif self.set not in session.mne.eventsID.columns:
                pglMessages.warning(f"Could not find label set: {self.set}")
                return None

            if self.label is None:
                # Average across all conditions.
                session.mne.evoked = session.mne.epochs.average()
            else:
                labels = session.mne.eventsID[self.set].dropna().unique()
                if self.label not in labels:
                    pglMessages.warning(f"Unknown label {self.label!r} in column {self.set!r}. Available labels: {list(labels)}")
                    return None

                session.mne.evoked = session.mne.epochs[f'{self.set} == "{self.label}"'].average()

            # Apply the same fixed limits to traces and topomaps.
            # With no explicit limits, MNE scales each plot automatically.
            traceKwargs = {}
            topoKwargs = {}

            if self.yMin is not None:
                sharedLimits = (self.yMin, self.yMax)
                channelTypes = set(session.mne.evoked.get_channel_types())
                traceKwargs["ylim"] = {chType: sharedLimits for chType in channelTypes}
                topoKwargs["vlim"] = sharedLimits
                topoKwargs["contours"] = np.linspace(self.yMin, self.yMax, 7)
                
            # Plot the evoked response.
            if isSensor:
                fig = session.mne.evoked.plot(picks=[self.picks], show=False, **traceKwargs)

                evoked = session.mne.evoked.copy().pick([self.picks])
                timeSeries = evoked.data[0]
                sampleRate = evoked.info["sfreq"]

                fftValues = np.fft.rfft(timeSeries)
                freqs = np.fft.rfftfreq(len(timeSeries), d=1 / sampleRate)
                magnitude = np.abs(fftValues)

                # Matplotlib requires a finite upper x limit.
                fftMaxFreq = self.maxFreq if np.isfinite(self.maxFreq) else freqs[-1]

                figFft, ax = plt.subplots(figsize=(12, 6))
                ax.stem(freqs, magnitude, linefmt="C0-", markerfmt="C0.", basefmt=" ")
                ax.set(xlabel="Frequency (Hz)", ylabel="FFT magnitude", title=f"Evoked FFT magnitude: {self.picks}", xlim=(self.minFreq, fftMaxFreq))
                figFft.tight_layout()

            else:
                fig = session.mne.evoked.plot_joint(times="peaks", picks=self.picks, show=False, ts_args=traceKwargs, topomap_args=topoKwargs)

            fig.set_size_inches(20, 8)
            session.mne.report.add_figure(fig=fig, title="evoked", section="Plot Evoked")

            # Spectrum of the evoked response.
            if isSensor:
                evoked = session.mne.evoked.copy()
                nTimes = evoked.data.shape[1]
                spectrum = evoked.compute_psd(method="welch", picks=[self.picks], fmin=self.minFreq, fmax=self.maxFreq, n_fft=nTimes, n_per_seg=nTimes, n_overlap=0, window="boxcar", remove_dc=True)
                psds, freqs = spectrum.get_data(return_freqs=True)

                # Get the first spectrum and convert to fT²/Hz.
                psd = psds[0] * 1e30

                # Remove DC only if it is present.
                nonDc = freqs > 0
                psd = psd[nonDc]
                freqs = freqs[nonDc]

            else:
                spectrum = session.mne.evoked.compute_psd(method="welch", picks=self.picks, fmin=self.minFreq, fmax=self.maxFreq)
                psds, freqs = spectrum.get_data(return_freqs=True)
                psd = psds.mean(axis=0)

                # Convert to fT²/Hz.
                psd = psd * 1e30

            figPsd, ax = plt.subplots(figsize=(12, 6))
            ax.stem(freqs, psd, linefmt="C0-", markerfmt="C0.", basefmt=" ")
            ax.set(xlabel="Frequency (Hz)", ylabel="Power spectral density (fT²/Hz)", title=f"Evoked response spectrum: {self.picks}")
            figPsd.tight_layout()

            session.mne.report.add_figure(fig=figPsd, title="spectrum", section="Plot Evoked")

            # Third figure: single peak response, with sensor labels.
            if not isSensor:
                evokedTopo = session.mne.evoked.copy().pick(self.picks)
                peakChannel, peakTime = evokedTopo.get_peak(ch_type=self.picks, mode="abs")

                figPeak = evokedTopo.plot_topomap(times=[peakTime], ch_type=self.picks, show_names=True, sensors=True, time_unit="ms", size=8, cmap=("RdBu_r", False), show=False, **topoKwargs)
                figPeak.set_size_inches(12, 10)
                figPeak.suptitle(f"Peak response: {peakTime * 1000:.1f} ms ({peakChannel})", fontsize=18)
                figPeak.subplots_adjust(top=0.88)

                # Enlarge sensor labels and the timepoint title.
                for ax in figPeak.axes:
                    for text in ax.texts:
                        text.set_fontsize(16)
                    ax.title.set_fontsize(20)

                session.mne.report.add_figure(fig=figPeak, title="Peak topo with labels", section="Plot Evoked")

            return session
        
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # plot evoked NaN
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class plotEvokedNan(pglAction):
        '''
        Plot the evoked response with NaN values handled appropriately (i.e. using nanmean)
        '''
        

        set = Unicode(allow_none=True, default_value=None, help="Label set; defaults to the first set")
        label = Unicode(allow_none=True, default_value=None, help="Label to average; defaults to all conditions")
        picks = Unicode("mag", help="Channel type or sensor name")
        minFreq = Float(0.0, help="Minimum displayed frequency")
        maxFreq = Float(float("inf"), help="Maximum displayed frequency")

        @staticmethod
        def _nanmean(data):
            """Average across epochs, leaving NaN where all epochs are missing."""
            import numpy as np

            counts = np.sum(~np.isnan(data), axis=0)
            totals = np.nansum(data, axis=0)
            return np.divide(totals, counts, out=np.full(totals.shape, np.nan), where=counts > 0)

        @staticmethod
        def _valid_segment(data):
            """Longest contiguous finite segment; never join samples across gaps."""
            import numpy as np

            edges = np.diff(np.r_[False, np.isfinite(data), False].astype(int))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)

            if not starts.size:
                return data[:0]

            index = np.argmax(stops - starts)
            return data[starts[index]:stops[index]]

        def _run(self, session: pglSession) -> pglSession:
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal import welch

            # Validate inputs.
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw mne loaded")
                return None

            if session.mne.epochs is None:
                self.setError("Session does not have epochs created")
                return None

            self.picks = session.mne.validatePicks(self.picks)
            is_sensor = session.mne.isSensor(self.picks)
            epochs = session.mne.epochs

            # Select epochs only when a particular label is requested.
            if self.label is not None:
                columns = session.mne.eventsID.columns

                if self.set is None and len(columns) < 4:
                    pglMessages.warning("No default label set is available")
                    return session

                label_set = columns[3] if self.set is None else self.set

                if label_set not in columns:
                    pglMessages.warning(f"Could not find label set: {label_set}")
                    return session

                labels = session.mne.eventsID[label_set].dropna().unique()

                if self.label not in labels:
                    pglMessages.warning(f"Unknown label {self.label!r}. Available labels: {list(labels)}")
                    return session

                epochs = epochs[f"{label_set} == {self.label!r}"]

            if epochs.get_data().shape[0] == 0:
                pglMessages.warning("No epochs remain for this selection")
                session.mne.evoked = None
                return session

            # MNE accepts a callable for averaging across epochs.
            session.mne.evoked = epochs.average(method=self._nanmean)
            evoked = session.mne.evoked.copy().pick([self.picks] if is_sensor else self.picks)
            data = evoked.data
            sfreq = evoked.info["sfreq"]

            if not np.isfinite(data).any():
                pglMessages.warning("Selected evoked data contain no finite values; skipping plots")
                return session

            # Display units.
            channel_type = evoked.get_channel_types()[0]
            scale, unit = {"mag": (1e15, "fT"), "grad": (1e13, "fT/cm"), "eeg": (1e6, "µV")}.get(channel_type, (1.0, "SI units"))

            # Finite amplitude limits for the trace plot.
            finite_values = data[np.isfinite(data)] * scale
            lower, upper = finite_values.min(), finite_values.max()
            padding = 0.05 * (upper - lower) if upper > lower else max(abs(lower) * 0.05, 1.0)

            trace_args = dict(spatial_colors=True, scalings={channel_type: scale}, units={channel_type: unit}, ylim={channel_type: (lower - padding, upper + padding)}, gfp=False)

            # Small topomaps require complete spatial snapshots.
            complete_times = np.flatnonzero(np.isfinite(data).all(axis=0))

            if not is_sensor and complete_times.size:
                if np.isfinite(data).all():
                    topo_times = "peaks"
                else:
                    # Up to three evenly spaced complete snapshots; no imputation.
                    indices = np.linspace(0, complete_times.size - 1, min(3, complete_times.size), dtype=int)
                    topo_times = evoked.times[complete_times[indices]]

                fig = evoked.plot_joint(times=topo_times, picks=self.picks, ts_args=trace_args, show=False)
            else:
                fig = evoked.plot(show=False, **trace_args)

                if not is_sensor:
                    pglMessages.warning("No timepoint has valid data at every selected sensor; omitting the small topomaps")

            fig.set_size_inches(20, 8)
            fig.suptitle(f"NaN-aware evoked: {self.picks}")
            
            session.mne.report.add_figure(fig=fig, title="evoked", section="Plot Evoked (nan-aware)")

            # FFT / PSD: use contiguous data without changing the sampling interval.
            fmin = max(0.0, self.minFreq)
            fmax = min(self.maxFreq, sfreq / 2)

            if fmin < fmax:
                psds = []
                n_fft = data.shape[1] if is_sensor else min(2048, data.shape[1])

                for name, series in zip(evoked.ch_names, data):
                    segment = self._valid_segment(series)

                    if segment.size < 2:
                        pglMessages.warning(f"{name}: insufficient contiguous data for FFT/PSD")
                        continue

                    if segment.size < series.size:
                        pglMessages.warning(f"{name}: FFT/PSD use the longest valid segment ({segment.size}/{series.size} samples)")

                    if is_sensor:
                        fft_freqs = np.fft.rfftfreq(segment.size, d=1 / sfreq)
                        magnitude = np.abs(np.fft.rfft(segment * scale))
                        keep = (fft_freqs >= fmin) & (fft_freqs <= fmax)

                        if keep.any():
                            fig_fft, ax = plt.subplots(figsize=(12, 6))
                            ax.stem(fft_freqs[keep], magnitude[keep], linefmt="C0-", markerfmt="C0.", basefmt=" ")
                            ax.set(xlabel="Frequency (Hz)", ylabel=f"FFT magnitude ({unit}, unnormalized)", title=f"Evoked FFT: {self.picks}", xlim=(fmin, fmax))
                            fig_fft.tight_layout()

                    freqs, psd = welch(segment, fs=sfreq, window="boxcar" if is_sensor else "hamming", nperseg=min(n_fft, segment.size), noverlap=0, nfft=n_fft, detrend="constant", scaling="density")
                    psds.append(psd)

                if psds:
                    mean_psd = self._nanmean(np.asarray(psds)) * scale**2
                    keep = (freqs >= fmin) & (freqs <= fmax)

                    if is_sensor:
                        keep &= freqs > 0

                    if keep.any():
                        fig_psd, ax = plt.subplots(figsize=(12, 6))
                        ax.stem(freqs[keep], mean_psd[keep], linefmt="C0-", markerfmt="C0.", basefmt=" ")
                        ax.set(xlabel="Frequency (Hz)", ylabel=f"PSD ({unit}²/Hz)", title=f"Evoked spectrum: {self.picks}", xlim=(fmin, fmax))
                        fig_psd.tight_layout()
                        
                        session.mne.report.add_figure(fig=fig_psd, title="spectrum", section="Plot Evoked (nan-aware)")
                    else:
                        pglMessages.warning("No PSD bins fall within the requested frequency range")
            else:
                pglMessages.warning("Invalid frequency range; skipping FFT/PSD")

            # Peak topomap: require finite values at every selected sensor.
            if not is_sensor:
                complete_times = np.flatnonzero(np.isfinite(data).all(axis=0))

                if not complete_times.size:
                    pglMessages.warning("No complete sensor snapshot is available; skipping topomap")
                else:
                    channel_index, time_index = np.unravel_index(np.argmax(np.abs(data[:, complete_times])), (data.shape[0], complete_times.size))
                    peak_time = evoked.times[complete_times[time_index]]
                    peak_channel = evoked.ch_names[channel_index]

                    # Fix symmetric color limits in the displayed units.
                    peak_index = np.argmin(np.abs(evoked.times - peak_time))
                    vmax = float(np.max(np.abs(data[:, peak_index])) * scale)
                    vmax = vmax if vmax > 0 else 1.0

                    # The False in cmap disables interactive colorbar rescaling.
                    fig_peak = evoked.plot_topomap(times=[peak_time], ch_type=channel_type, scalings={channel_type: scale}, units={channel_type: unit}, show_names=True, sensors=True, contours=6, time_unit="ms", size=8, vlim=(-vmax, vmax), cmap=("RdBu_r", False), show=False)
                    fig_peak.set_size_inches(12, 10)
                    fig_peak.suptitle(f"Peak response: {peak_time * 1000:.1f} ms ({peak_channel})", fontsize=20)
                    fig_peak.subplots_adjust(top=0.88)

                    # Enlarge electrode labels, timepoint title, and colorbar text.
                    for ax in fig_peak.axes:
                        for text in ax.texts:
                            text.set_fontsize(16)
                        ax.title.set_fontsize(20)
                        ax.tick_params(labelsize=14)
                        ax.xaxis.label.set_fontsize(16)
                        ax.yaxis.label.set_fontsize(16)
                        
                    session.mne.report.add_figure(fig=fig_peak, title="Peak topo", section="Plot Evoked (nan-aware)")

            return session
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # plot evoked Each
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class plotEvokedEach(pglAction):
        '''
        Plot the evoked response for each label set
        '''
        

        # Parameters
        set = Unicode(default_value=None, allow_none=True, help="Label set used to group epochs (defaults to the first set)")
        picks = Unicode("mag", help="Channel type to plot: mag, grad, or eeg")
        nCols = Int(4, min=1, help="Number of labels across each row")

        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:
            self.configureTraits(**kwargs)
            super().configure()

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            """
            Compute label-specific evoked responses with a progress bar.
            Display each completed row as a static image, independent of
            the notebook's interactive Matplotlib backend.
            """

            import mne
            import numpy as np
            from io import BytesIO
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from IPython.display import display, Image
            from tqdm.auto import tqdm

            # Check session
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw mne loaded")
                return None

            if session.mne.epochs is None:
                self.setError("Session does not have epochs created")
                return None

            # Validate channel type
            self.picks = session.mne.validatePicks(self.picks)

            if self.picks not in ("mag", "grad", "eeg"):
                self.setError("picks must be a channel type: mag, grad, or eeg")
                return None

            epochs = session.mne.epochs
            eventsID = session.mne.eventsID

            # Resolve label set
            labelSet = self.set

            if labelSet is None:
                if len(eventsID.columns) <= 3:
                    self.setError("No label sets found in eventsID")
                    return None

                labelSet = eventsID.columns[3]

            if labelSet not in eventsID.columns:
                self.setError(f"Could not find label set: {labelSet}")
                return None

            if epochs.metadata is None or labelSet not in epochs.metadata.columns:
                self.setError(f"Epoch metadata does not contain label set: {labelSet}")
                return None

            # Select channels once
            megType = self.picks if self.picks in ("mag", "grad") else False
            channelIndices = mne.pick_types(epochs.info, meg=megType, eeg=self.picks == "eeg", exclude="bads")

            if len(channelIndices) == 0:
                self.setError(f"No usable {self.picks!r} channels found")
                return None

            # Find epochs for each label
            labels = eventsID[labelSet].dropna().unique()
            selections = []

            for label in labels:
                mask = epochs.metadata[labelSet].eq(label).fillna(False).to_numpy(dtype=bool)
                epochIndices = np.flatnonzero(mask)

                if len(epochIndices) == 0:
                    pglMessages.warning(f"No epochs remaining for label {label!r}; skipping")
                    continue

                selections.append((label, epochIndices))

            if not selections:
                self.setError(f"No epochs found for labels in {labelSet!r}")
                return None

            nLabels = len(selections)
            nCols = min(self.nCols, nLabels)
            nRows = (nLabels + nCols - 1) // nCols

            with tqdm(total=nLabels, desc="Evoked plots", unit="stimulus") as progress:

                for row in range(nRows):
                    rowSelections = selections[row * nCols:(row + 1) * nCols]

                    # Create an independent static figure, not a pyplot/widget figure
                    fig = Figure(figsize=(5 * nCols, 6), dpi=100, constrained_layout=False)
                    FigureCanvasAgg(fig)

                    try:
                        outerGrid = fig.add_gridspec(1, nCols, left=0.055, right=0.97, bottom=0.12, top=0.80, wspace=0.45)
                        fig.suptitle(f"Evoked responses by {labelSet} — {self.picks} — row {row + 1}/{nRows}", fontsize=16)

                        for col, (label, epochIndices) in enumerate(rowSelections):
                            progress.set_postfix_str(f"Row {row + 1}/{nRows}: {label}", refresh=True)

                            # Compute this stimulus immediately before plotting
                            evoked = epochs[epochIndices].average(picks=channelIndices)
                            _, peakTime = evoked.get_peak(ch_type=self.picks, mode="abs")

                            # Topomap and colorbar above, evoked below
                            innerGrid = outerGrid[0, col].subgridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1, 0.05], hspace=0.40, wspace=0.08)

                            axTopo = fig.add_subplot(innerGrid[0, 0])
                            axColorbar = fig.add_subplot(innerGrid[0, 1])
                            axEvoked = fig.add_subplot(innerGrid[1, :])

                            # Fixed color limits and no interactive colorbar
                            evoked.plot_topomap(times=[peakTime], ch_type=self.picks, axes=[axTopo, axColorbar], colorbar=True, sensors=True, contours=6, time_unit="ms", cmap=("RdBu_r", False), vlim=(-30, 30), show=False)
                            axTopo.set_title(f"{label}\nn = {len(epochIndices)} | peak = {peakTime * 1000:.1f} ms", fontsize=11)
                            axColorbar.tick_params(labelsize=8)

                            # Butterfly plot explicitly attached to this row's axes
                            evoked.plot(axes=axEvoked, spatial_colors=True, gfp=False, time_unit="ms", show=False)
                            axEvoked.axvline(peakTime * 1000, color="black", linestyle="--", linewidth=1)
                            axEvoked.set_title("")
                            axEvoked.tick_params(labelsize=8)

                            progress.update(1)
                            
                        # Add this completed row to the report
                        session.mne.report.add_figure(fig=fig, title=f"Row {row + 1}/{nRows}", section=f"Evoked set: {labelSet}", caption="Labels: " + ", ".join(str(label) for label, _ in rowSelections), image_format="png")

                        # Render and display only this row; no live widget or callbacks
                        with BytesIO() as buffer:
                            fig.savefig(buffer, format="png", dpi=100, facecolor="white")
                            display(Image(data=buffer.getvalue()))

                    finally:
                        fig.clear()

            return session

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # plot evoked each NaN
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class plotEvokedEachNan(pglAction):
        '''
        Plot the evoked response for each label set, handling NaN values appropriately (ie. using nanmean).
        '''

        # Parameters
        set = Unicode(default_value=None, allow_none=True, help="Label set used to group epochs (defaults to the first set)")
        picks = Unicode("mag", help="Channel type to plot: mag, grad, or eeg")
        nCols = Int(4, min=1, help="Number of labels across each row")
        topoTime = Float(default_value=None, allow_none=True, help="Topomap time in ms for every label; None selects each label's peak")
        topoMin = Float(-30.0, help="Topomap minimum in displayed units")
        topoMax = Float(30.0, help="Topomap maximum in displayed units")
        timeseriesMin = Float(default_value=None, allow_none=True, help="Time series y-axis minimum in displayed units; None uses automatic limits")
        timeseriesMax = Float(default_value=None, allow_none=True, help="Time series y-axis maximum in displayed units; None uses automatic limits")

        ################################
        # NaN-aware average
        ################################
        @staticmethod
        def _nanMean(data):
            """Average across epochs, leaving NaN where all epochs are missing."""
            import numpy as np

            counts = np.sum(~np.isnan(data), axis=0)
            totals = np.nansum(data, axis=0)
            return np.divide(totals, counts, out=np.full(totals.shape, np.nan), where=counts > 0)

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            """
            Compute NaN-aware label-specific evoked responses.

            Display each completed row as a static image, independent of
            the notebook's interactive Matplotlib backend.
            """
            import mne
            import numpy as np
            from io import BytesIO
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from IPython.display import display, Image
            from tqdm.auto import tqdm

            # Check session.
            if session.mne is None or session.mne.raw is None:
                self.setError("Session does not have raw mne loaded")
                return None

            if session.mne.epochs is None:
                self.setError("Session does not have epochs created")
                return None

            # Validate channel type.
            self.picks = session.mne.validatePicks(self.picks)

            if self.picks not in ("mag", "grad", "eeg"):
                self.setError("picks must be a channel type: mag, grad, or eeg")
                return None

            # Validate configured times and limits.
            for parameterName in ("topoTime", "topoMin", "topoMax", "timeseriesMin", "timeseriesMax"):
                parameterValue = getattr(self, parameterName)

                if parameterValue is not None and not np.isfinite(parameterValue):
                    self.setError(f"{parameterName} must be finite")
                    return None

            if self.topoMin >= self.topoMax:
                self.setError("topoMin must be less than topoMax")
                return None

            if self.timeseriesMin is not None and self.timeseriesMax is not None:
                if self.timeseriesMin >= self.timeseriesMax:
                    self.setError("timeseriesMin must be less than timeseriesMax")
                    return None

            epochs = session.mne.epochs
            eventsID = session.mne.eventsID

            # Resolve label set.
            labelSet = self.set

            if labelSet is None:
                if len(eventsID.columns) <= 3:
                    self.setError("No label sets found in eventsID")
                    return None

                labelSet = eventsID.columns[3]

            if labelSet not in eventsID.columns:
                self.setError(f"Could not find label set: {labelSet}")
                return None

            if epochs.metadata is None or labelSet not in epochs.metadata.columns:
                self.setError(f"Epoch metadata does not contain label set: {labelSet}")
                return None

            # Resolve a shared topomap sample when a fixed time is requested.
            fixedTimeIndex = None

            if self.topoTime is not None:
                firstTime = epochs.times[0] * 1000
                lastTime = epochs.times[-1] * 1000

                if self.topoTime < firstTime or self.topoTime > lastTime:
                    self.setError(f"topoTime must be within the epoch range: {firstTime:.3f} to {lastTime:.3f} ms")
                    return None

                fixedTimeIndex = int(np.argmin(np.abs(epochs.times * 1000 - self.topoTime)))

            # Select channels once, excluding channels marked bad.
            megType = self.picks if self.picks in ("mag", "grad") else False
            channelIndices = mne.pick_types(epochs.info, meg=megType, eeg=self.picks == "eeg", exclude="bads")

            if len(channelIndices) == 0:
                self.setError(f"No usable {self.picks!r} channels found")
                return None

            # Display units.
            scale, unit = {"mag": (1e15, "fT"), "grad": (1e13, "fT/cm"), "eeg": (1e6, "µV")}[self.picks]
            scalings = {self.picks: scale}
            units = {self.picks: unit}

            # Find epochs for each label.
            labels = eventsID[labelSet].dropna().unique()
            selections = []

            for label in labels:
                mask = epochs.metadata[labelSet].eq(label).fillna(False).to_numpy(dtype=bool)
                epochIndices = np.flatnonzero(mask)

                if len(epochIndices) == 0:
                    pglMessages.warning(f"No epochs remaining for label {label!r}; skipping")
                    continue

                selections.append((label, epochIndices))

            if not selections:
                self.setError(f"No epochs found for labels in {labelSet!r}")
                return None

            nLabels = len(selections)
            nCols = min(self.nCols, nLabels)
            nRows = (nLabels + nCols - 1) // nCols

            with tqdm(total=nLabels, desc="NaN-aware evoked plots", unit="stimulus") as progress:

                for row in range(nRows):
                    rowSelections = selections[row * nCols:(row + 1) * nCols]

                    # Independent static figure, not a pyplot/widget figure.
                    fig = Figure(figsize=(5 * nCols, 6), dpi=100, constrained_layout=False)
                    FigureCanvasAgg(fig)

                    try:
                        outerGrid = fig.add_gridspec(1, nCols, left=0.055, right=0.97, bottom=0.12, top=0.80, wspace=0.45)
                        fig.suptitle(f"NaN-aware evoked responses by {labelSet} — {self.picks} — row {row + 1}/{nRows}", fontsize=16)

                        for col, (label, epochIndices) in enumerate(rowSelections):
                            progress.set_postfix_str(f"Row {row + 1}/{nRows}: {label}", refresh=True)

                            # Average available values independently at each sensor/time.
                            evoked = epochs[epochIndices].average(picks=channelIndices, method=self._nanMean)
                            data = evoked.data
                            finiteMask = np.isfinite(data)

                            # Topomap and colorbar above, butterfly plot below.
                            innerGrid = outerGrid[0, col].subgridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1, 0.05], hspace=0.40, wspace=0.08)

                            axTopo = fig.add_subplot(innerGrid[0, 0])
                            axColorbar = fig.add_subplot(innerGrid[0, 1])
                            axEvoked = fig.add_subplot(innerGrid[1, :])

                            labelTitle = f"{label}\nn = {len(epochIndices)}"

                            # Keep the label visible even when no plot is possible.
                            if not finiteMask.any():
                                pglMessages.warning(f"{label!r}: evoked data contain no finite values; skipping plots")

                                axTopo.set_title(labelTitle, fontsize=11)
                                axTopo.text(0.5, 0.5, "No finite evoked data", ha="center", va="center", transform=axTopo.transAxes)
                                axTopo.set_axis_off()
                                axColorbar.set_axis_off()

                                axEvoked.text(0.5, 0.5, "No finite evoked data", ha="center", va="center", transform=axEvoked.transAxes)
                                axEvoked.set_axis_off()

                                progress.update(1)
                                continue

                            # Automatic butterfly limits use only finite values.
                            finiteValues = data[finiteMask] * scale
                            lower, upper = finiteValues.min(), finiteValues.max()
                            padding = 0.05 * (upper - lower) if upper > lower else max(abs(lower) * 0.05, 1.0)

                            traceMin = lower - padding if self.timeseriesMin is None else self.timeseriesMin
                            traceMax = upper + padding if self.timeseriesMax is None else self.timeseriesMax

                            # Keep a valid range when only one configured bound is
                            # supplied and it lies beyond the automatic opposite bound.
                            if traceMin >= traceMax:
                                if self.timeseriesMax is None:
                                    traceMax = traceMin + max(2 * padding, abs(traceMin) * 0.05, 1.0)
                                else:
                                    traceMin = traceMax - max(2 * padding, abs(traceMax) * 0.05, 1.0)

                            traceLimits = {self.picks: (traceMin, traceMax)}

                            # Select either the shared time or this label's peak.
                            topoIndex = None
                            markerTime = None
                            missingMessage = ""

                            if fixedTimeIndex is not None:
                                markerTime = evoked.times[fixedTimeIndex]
                                labelTitle = f"{labelTitle} | time = {markerTime * 1000:.1f} ms"

                                if finiteMask[:, fixedTimeIndex].all():
                                    topoIndex = fixedTimeIndex
                                else:
                                    missingMessage = "Incomplete sensor snapshot\nTopomap omitted"
                                    pglMessages.warning(f"{label!r}: not every selected sensor has finite data at {markerTime * 1000:.3f} ms; omitting topomap")

                            else:
                                completeTimes = np.flatnonzero(finiteMask.all(axis=0))

                                if completeTimes.size:
                                    # Largest absolute response among complete snapshots.
                                    peakFlatIndex = np.argmax(np.abs(data[:, completeTimes]))
                                    _, peakTimeIndex = np.unravel_index(peakFlatIndex, (data.shape[0], completeTimes.size))
                                    topoIndex = int(completeTimes[peakTimeIndex])
                                    markerTime = evoked.times[topoIndex]
                                    labelTitle = f"{labelTitle} | peak = {markerTime * 1000:.1f} ms"

                                else:
                                    missingMessage = "No complete sensor snapshot\nTopomap omitted"
                                    pglMessages.warning(f"{label!r}: no timepoint has finite data at every selected sensor; omitting topomap")

                            # Plot only complete spatial snapshots; never impute data.
                            if topoIndex is not None:
                                topoSampleTime = evoked.times[topoIndex]

                                evoked.plot_topomap(times=[topoSampleTime], ch_type=self.picks, axes=[axTopo, axColorbar], scalings=scalings, units=units, colorbar=True, sensors=True, contours=6, time_unit="ms", cmap=("RdBu_r", False), vlim=(self.topoMin, self.topoMax), show=False)
                                axTopo.set_title(labelTitle, fontsize=11)
                                axColorbar.tick_params(labelsize=8)

                            else:
                                axTopo.set_title(labelTitle, fontsize=11)
                                axTopo.text(0.5, 0.5, missingMessage, ha="center", va="center", transform=axTopo.transAxes)
                                axTopo.set_axis_off()
                                axColorbar.set_axis_off()

                            # Missing values remain gaps in the butterfly plot.
                            # Convert any infinities to NaN on a plotting copy only.
                            traceEvoked = evoked.copy()
                            traceEvoked.data[~finiteMask] = np.nan

                            traceEvoked.plot(axes=axEvoked, spatial_colors=True, scalings=scalings, units=units, ylim=traceLimits, gfp=False, time_unit="ms", show=False)

                            # For fixed times, retain the marker even if the topomap
                            # was omitted because some sensors were missing.
                            if markerTime is not None:
                                axEvoked.axvline(markerTime * 1000, color="black", linestyle="--", linewidth=1)

                            axEvoked.set_title("")
                            axEvoked.tick_params(labelsize=8)

                            progress.update(1)

                        # Add this completed row to the report
                        session.mne.report.add_figure(fig=fig, title=f"Row {row + 1}/{nRows}", section=f"Evoked set (nan-aware): {labelSet}", caption="Labels: " + ", ".join(str(label) for label, _ in rowSelections), image_format="png")

                        # Render and display only this row.
                        with BytesIO() as buffer:
                            fig.savefig(buffer, format="png", dpi=100, facecolor="white")
                            display(Image(data=buffer.getvalue()))

                    finally:
                        fig.clear()

            return session

    _actionHeader_ncsnr = "NCSNR computation"

    class computeNCSNR(pglAction):
        """
        Compute single-sensor NCSNR and a repeat-adjusted noise ceiling.

        Example:
            action.configure(set="grouped", label="things", conditionIDColumn="code", chType="mag", nTrials=3)

        Conditions are identified by conditionIDColumn within the selected metadata
        group. Repeated trials of each condition estimate single-trial noise.

        nTrials specifies the number of repeats averaged per condition in the
        responses being evaluated, not the total number of epochs.

        Noise ceiling (%) = 100 * NCSNR^2 / (NCSNR^2 + 1 / nTrials).

        This assumes independent noise across repeats. The ceiling is expressed
        as percent explainable variance, not correlation.
        """

        channelName = Unicode(allow_none=True, default_value=None, help="Full or partial channel name. If None, use the largest absolute Evoked response.")
        chType = Unicode("mag", help="MNE channel type for automatic peak-channel selection, e.g. eeg, mag, or grad.")
        set = Unicode(allow_none=True, default_value=None, help="Metadata column used to select trials, e.g. grouped.")
        label = Unicode(allow_none=True, default_value=None, help="Metadata label used to select trials, e.g. things.")
        conditionIDColumn = Unicode("code", help="Metadata column identifying individual signal conditions.")
        nTrials = Int(default_value=3, min=1, help="Number of repeats averaged per condition for the noise ceiling. Use 1 for single-trial responses.")

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
            # Identify conditions from the complete events table.
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
            # Extract retained trials using epochs.metadata.
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

                # Shape: (nRepetitions, nTimes).
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

            conditionIDs = list(conditionData)

            # ---------------------------------------------------------------------
            # 1. Estimate typical single-trial noise variance.
            # ---------------------------------------------------------------------
            noiseVarByCondition = np.asarray([np.var(conditionData[conditionID], axis=0, ddof=1) for conditionID in conditionIDs])
            noiseVar = noiseVarByCondition.mean(axis=0)
            noiseSTD = np.sqrt(noiseVar)

            # ---------------------------------------------------------------------
            # 2. Estimate noise-corrected signal variance across condition means.
            #
            # Actual repetition counts correct the noise remaining in each mean.
            # Each condition receives equal weight.
            # ---------------------------------------------------------------------
            conditionMeansArray = np.asarray([conditionMeans[conditionID] for conditionID in conditionIDs])
            repsArray = np.asarray([nRepsPerCondition[conditionID] for conditionID in conditionIDs])

            observedVar = np.var(conditionMeansArray, axis=0, ddof=1)
            residualNoiseVar = np.mean(noiseVarByCondition / repsArray[:, np.newaxis], axis=0)

            signalVar = np.maximum(observedVar - residualNoiseVar, 0)
            signalSTD = np.sqrt(signalVar)

            # ---------------------------------------------------------------------
            # 3. NCSNR = signal SD / typical single-trial noise SD.
            #
            # Preserve NaN where single-trial noise SD is zero.
            # ---------------------------------------------------------------------
            ncsnr = np.divide(signalSTD, noiseSTD, out=np.full_like(signalSTD, np.nan, dtype=float), where=noiseSTD > 0)

            # ---------------------------------------------------------------------
            # 4. Convert NCSNR to a repeat-adjusted noise ceiling (%).
            #
            # nTrials specifies the target number of averaged repeats. It does not
            # change the single-trial NCSNR estimate.
            # ---------------------------------------------------------------------
            ncsnrSquared = np.square(ncsnr)
            noiseCeiling = 100.0 * ncsnrSquared / (ncsnrSquared + 1.0 / self.nTrials)

            grandMean = conditionMeansArray.mean(axis=0)
            times = epochs.times.copy()

            if np.all(np.isnan(ncsnr)):
                peakIndex = None
                peakTime = np.nan
                peakNcsnr = np.nan
                peakNoiseCeiling = np.nan
            else:
                peakIndex = np.nanargmax(ncsnr)
                peakTime = times[peakIndex]
                peakNcsnr = ncsnr[peakIndex]
                peakNoiseCeiling = noiseCeiling[peakIndex]

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
                "nTrials": self.nTrials,
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
                "noiseCeiling": noiseCeiling,
                "noiseCeilingUnits": "percent",
                "peakIndex": peakIndex,
                "peakTime": peakTime,
                "peakNcsnr": peakNcsnr,
                "peakNoiseCeiling": peakNoiseCeiling,
                "evokedPeakTime": evokedPeakTime,
                "evokedPeakAmplitude": evokedPeakAmplitude,
            }

            # ---------------------------------------------------------------------
            # Plot.
            # ---------------------------------------------------------------------
            with plt.rc_context({"figure.constrained_layout.use": False, "figure.autolayout": False}):
                fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True, layout=None)

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

                axes[1].set_ylabel("Noise-corrected SNR")
                axes[1].set_title("Single-trial NCSNR across individual conditions")
                axes[1].grid(alpha=0.25)

                axes[2].plot(times, noiseCeiling, color="tab:green", linewidth=1.25)
                axes[2].axvline(0, color="gray", linestyle="--", linewidth=1)

                if peakIndex is not None:
                    axes[2].axvline(peakTime, color="tab:red", linestyle="--", linewidth=1)
                    axes[2].plot(peakTime, peakNoiseCeiling, marker="o", color="tab:red")
                    axes[2].annotate(f"Peak: {peakNoiseCeiling:.1f}%\n{peakTime * 1000:.1f} ms", xy=(peakTime, peakNoiseCeiling), xytext=(8, 8), textcoords="offset points", color="tab:red")

                axes[2].set_xlabel("Time (s)")
                axes[2].set_ylabel("Noise ceiling (%)")
                axes[2].set_ylim(0, 110)
                axes[2].set_yticks(np.arange(0, 101, 20))
                axes[2].set_title(f"Noise ceiling for condition responses averaged over {self.nTrials} repeat(s)")
                axes[2].grid(alpha=0.25)

                figureTitle = f"NCSNR and noise ceiling: {self.set} = {self.label} | condition IDs from {self.conditionIDColumn} | nTrials={self.nTrials}"
                fig.suptitle(figureTitle, fontsize=14)
                fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.92, hspace=0.40)

                plt.show()

                session.mne.report.add_figure(fig=fig, title=figureTitle)

            return session
 
    _actionHeader_blinks = "Blink processing"
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # eyeblink detecion
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mneEyeblinkDetection(pglAction):
        '''
        Left-right channel difference used for eyeblink (probably superceded by using ICA components)
        '''
        

        # parameters
        leftEyeChannel = Unicode('L401', help="Left channel used for detecting eye blink")
        rightEyeChannel = Unicode('R401', help="Left channel used for detecting eye blink")
                
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
    # ICA: fit and display components
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class ica(pglAction):
        '''
        Compute and display ICA components
        '''
        

        chType = Unicode("mag", help="Channel type to fit ICA on: mag, grad or eeg (one type: the component topography is later used as sensor weights)")
        nComponents = Int(20, min=2, help="Number of ICA components")
        fitHighPass = Float(1.0, help="ICA is fit on data high-passed at this frequency (skipped if raw is already high-passed at or above it)")
        decim = Int(5, min=1, help="Maximum decimation factor for ICA fitting; reduced automatically for lower sampling rates")
        minFitSfreq = Float(200.0, min=1.0, help="Minimum samples/second retained for ICA fitting; data below this rate are not decimated")
        randomState = Int(42, help="Random seed for FastICA")

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Fit ICA and display the leading topographies (labeled ICA000, ICA001, ...).
            Choose the blink component by eye and pass its number to mneMarkBlinks.

            Stores: session.mne.ica
            '''
            import mne

            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None
            if self.chType not in ("mag", "grad", "eeg"):
                self.setError("chType must be mag, grad or eeg")
                return None

            raw = session.mne.raw
            picks = mne.pick_types(raw.info, meg=self.chType if self.chType in ("mag", "grad") else False,
                                   eeg=self.chType == "eeg", exclude="bads")

            # Fit on a copy only if something must change: the high-pass, or old blink marks.
            # Old marks must be hidden, otherwise the blinks are excluded from the fit and
            # the blink component disappears.
            oldMarks = np.flatnonzero(raw.annotations.description == "BAD_blink")
            needsHighPass = raw.info["highpass"] < self.fitHighPass - 1e-6
            fitRaw = raw
            if len(oldMarks) or needsHighPass:
                fitRaw = raw.copy().load_data()
                annotations = fitRaw.annotations.copy()
                annotations.delete(oldMarks)
                fitRaw.set_annotations(annotations)
                if needsHighPass:
                    fitRaw.filter(l_freq=self.fitHighPass, h_freq=None, picks=picks, verbose=False)

            # set decimation (not to exceed. minFitSfreq)
            sfreq = fitRaw.info["sfreq"]
            fitDecim = min(self.decim, max(1, int(sfreq / self.minFitSfreq)))

            pglMessages.message(f"Initializing ICA with {self.nComponents} components on {len(picks)} {self.chType} channels")
            ica = mne.preprocessing.ICA(n_components=min(self.nComponents, len(picks) - 1),
                                        method="fastica", random_state=self.randomState, max_iter="auto")
            
            pglMessages.message(f"ICA fitting: raw {sfreq:g} Hz, decim={fitDecim}, approximately {sfreq / fitDecim:g} samples/s")
            ica.fit(fitRaw, picks=picks, decim=self.decim, reject_by_annotation=True, verbose=False)

            session.mne.ica = ica
            ica.plot_components(picks=list(range(ica.n_components_)), show=True)
            
            session.mne.report.add_ica(ica=ica, title="ICA", inst=session.mne.raw)
            
            return session

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # mark blinks as bad segments
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class markBlinks(pglAction):
        '''
        Mark blinks based on ICA components. Must run ICA first and then set the component parameter to the desired ICA component.
        '''

        component = Int(0, min=0, help="ICA component (from mne.ica) whose topography weights the sensors")
        lowCutoff = Float(1.0, help="Low edge (Hz) of the band-pass on the weighted channel; 0 for none")
        highCutoff = Float(10.0, help="High edge (Hz) of the band-pass on the weighted channel; 0 for none")
        threshold = Float(5.0, help="Event criterion in robust SDs (median/MAD) of the filtered channel")
        minSeparation = Float(0.5, min=0.0, help="Bounce guard: peaks closer than this (s) collapse to the largest one")
        windowBefore = Float(0.15, min=0.0, help="Seconds before the event peak to mark as bad")
        windowAfter = Float(0.35, min=0.0, help="Seconds after the event peak to mark as bad")

        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            Weight the sensors by the chosen topography, find blink peaks, and mark a window around
            each as BAD_blink in session.mne.raw.annotations (replacing earlier BAD_blink marks).
            Epochs made earlier are not updated: re-run mneCreateEpochs.
            '''
            import mne
            from scipy.signal import find_peaks

            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None
            ica = getattr(session.mne, "ica", None)
            if ica is None:
                self.setError("session does not have an ICA: run mneBlinkICA first")
                return None
            if self.component >= ica.n_components_:
                self.setError(f"component {self.component} out of range: 0-{ica.n_components_ - 1}")
                return None

            raw = session.mne.raw
            sfreq = raw.info["sfreq"]

            # -- 1. one channel: sensors weighted by the topography (unit gain for that topography)
            topo = ica.get_components()[:, self.component]
            blink = (topo / np.sum(topo ** 2)) @ raw.get_data(picks=ica.ch_names)
            if self.lowCutoff > 0 or self.highCutoff > 0:
                blink = mne.filter.filter_data(blink, sfreq, self.lowCutoff or None, self.highCutoff or None, verbose=False)
            center = np.median(blink)
            z = (blink - center) / (1.4826 * np.median(np.abs(blink - center)))      # robust z-score
            if np.percentile(z, 99.9) < -np.percentile(z, 0.1):                      # ICA sign is arbitrary: blinks up
                z = -z

            # -- 2. events = peaks above threshold; peaks within minSeparation of a larger one are dropped
            peaks, _ = find_peaks(z, height=self.threshold, distance=max(1, int(round(self.minSeparation * sfreq))))
            times = peaks / sfreq                                                    # s from first sample
            intervals = np.diff(times)
            interval_text = f"Between blinks: mean {intervals.mean():.2f} s, min {intervals.min():.2f} s, max {intervals.max():.2f} s" if intervals.size else "Between blinks: N/A (fewer than 2 blinks)"
            starts = np.clip(times - self.windowBefore, 0, None)
            stops = np.clip(times + self.windowAfter, None, raw.n_times / sfreq)

            # -- 3. mark as bad, replacing earlier blink marks
            offset = raw.first_time if raw.info["meas_date"] is not None else 0.0
            annotations = raw.annotations.copy()
            annotations.delete(np.flatnonzero(annotations.description == "BAD_blink"))
            raw.set_annotations(annotations + mne.Annotations(
                onset=starts + offset, duration=stops - starts,
                description=["BAD_blink"] * len(peaks), orig_time=raw.info["meas_date"]))

            # -- 4. plots: all events overlaid, then the whole recording with window bars
            bad = np.zeros(len(z), bool)
            for start, stop in zip(starts, stops):
                bad[int(round(start * sfreq)): int(round(stop * sfreq)) + 1] = True

            fig, ax = plt.subplots(figsize=(7, 4.5))
            rel = np.arange(-int(round((self.windowBefore + 0.25) * sfreq)), int(round((self.windowAfter + 0.25) * sfreq)) + 1)
            ax.axvspan(-self.windowBefore, self.windowAfter, color="tab:red", alpha=0.12, lw=0)
            if len(peaks):
                index = peaks[:, None] + rel
                stack = np.where((index >= 0) & (index < len(z)), z[np.clip(index, 0, len(z) - 1)], np.nan)
                ax.plot(rel / sfreq, stack.T, color="0.45", lw=0.6, alpha=0.5)
                ax.plot(rel / sfreq, np.nanmean(stack, axis=0), color="k", lw=2)
            ax.axhline(self.threshold, color="tab:red", ls="--", lw=1)
            ax.set(xlabel="Time from event peak (s)", ylabel="Blink channel (robust SD)", title=f"{len(peaks)} events — component {self.component} (shaded = marked window)\n{interval_text}")
            ax.set_ylim(-3 * self.threshold, 4 * self.threshold)
            fig.tight_layout()
            session.mne.report.add_figure(fig=fig, title="All events", section="Blinks")

            fig, ax = plt.subplots(figsize=(14, 4))
            t = np.arange(len(z)) / sfreq
            ax.fill_between(t, 0, 1, where=bad, transform=ax.get_xaxis_transform(), color="tab:red", alpha=0.25, lw=0)
            ax.plot(t, z, color="k", lw=0.5)
            ax.plot(times, z[peaks], "v", color="tab:red", ms=4)
            ax.axhline(self.threshold, color="tab:red", ls="--", lw=1)
            ax.set(xlabel="Time (s)", ylabel="Blink channel (robust SD)", xlim=(0, t[-1]),
                   title=f"Component {self.component}: {len(peaks)} events, {100 * bad.mean():.1f}% of recording marked")
            ax.set_ylim(-3 * self.threshold, 4 * self.threshold)
            fig.tight_layout()

            session.mne.report.add_figure(fig=fig, title="Timeseries", section="Blinks")
            
            if len(peaks) == 0:
                pglMessages.warning(f"No blinks above {self.threshold} robust SD; any earlier BAD_blink marks were removed")
            else:
                pglMessages.message(f"Marked {len(peaks)} blinks as BAD_blink ({100 * bad.mean():.1f}% of recording). Re-run mneCreateEpochs to apply.")
            return session
   
   
    _actionHeader_spatial_filtering = "Spatial filtering"
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # multiorigin SSS using Foster's inverse
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class mSSSWithFosters(pglAction):
        '''
        Spatial filtering using multi-origin SSS with Foster's inverse method (Xan McPherson method)
        '''
        center1 = Instance(np.ndarray, default_value=np.zeros(3), help="Center of first expansion sphere returned by FIT_SPHERES_TO_MRI (defaults to zeros / i.e. origin of sensor space)")
        center2 = Instance(np.ndarray, default_value=np.zeros(3), help="Center of second expansion sphere returned by FIT_SPHERES_TO_MRI (defaults to zeros / i.e. origin of sensor space)")
        doFosters = Bool(True, help="Whether to apply Foster's inverse method")
        doMultiOrigin = Bool(False, help="Whether to use multi-origin SSS")
        Lin = Int(8, help="Vector spherical harmonic inner truncation order (default is 8) - signals inside that will be kept by the method")
        Lout = Int(3, help="Vector spherical harmonic outer truncation order (default is 3) - signals outside that will be removed by the method")
        runTmin = Union([Float(), List(Float(allow_none=True))], default_value=0.0, allow_none=True, help="Run start time(s) in seconds relative to recording start. None means recording start.")
        runTmax = Union([Float(), List(Float(allow_none=True))], default_value=None, allow_none=True, help="Run end time(s) in seconds relative to recording start. None means recording end.")
        noiseTmin = Union([Float(), List(Float(allow_none=True))], default_value=0.0, allow_none=True, help="Noise covariance start time(s) in seconds relative to recording start. None means recording start.")
        noiseTmax = Union([Float(), List(Float(allow_none=True))], default_value=1.0, allow_none=True, help="Noise covariance end time(s) in seconds relative to recording start. None means recording end.")

        def configure(self, runTmin=None, runTmax=None, noiseTmin=None, noiseTmax=None, **kwargs):
            """Keep all-None inputs for session discovery; otherwise normalize intervals to lists."""
            import math
            from numbers import Real

            parameters = {"runTmin": runTmin, "runTmax": runTmax, "noiseTmin": noiseTmin, "noiseTmax": noiseTmax}

            # All None means interval discovery will happen in _run.
            if all(value is None for value in parameters.values()):
                normalizedParameters = parameters.copy()
            else:
                # Determine interval count from explicit lists.
                listLengths = {}

                for parameterName, value in parameters.items():
                    if isinstance(value, list):
                        if not value:
                            self.setError(f"{parameterName} must not be an empty list.")
                            return
                        listLengths[parameterName] = len(value)

                if len(set(listLengths.values())) > 1:
                    lengthSummary = ", ".join(f"{name}={length}" for name, length in listLengths.items())
                    self.setError(f"All interval parameter lists must have the same length. Received: {lengthSummary}.")
                    return

                nIntervals = next(iter(listLengths.values()), 1)

                # Broadcast scalars and None; validate and copy explicit lists.
                normalizedParameters = {}

                for parameterName, value in parameters.items():
                    values = list(value) if isinstance(value, list) else [value] * nIntervals
                    normalizedValues = []

                    for intervalIndex, item in enumerate(values, start=1):
                        if item is not None:
                            if isinstance(item, bool) or not isinstance(item, Real):
                                self.setError(f"{parameterName}, interval {intervalIndex}: expected a number or None, received {item!r}.")
                                return

                            item = float(item)

                            if not math.isfinite(item) or item < 0:
                                self.setError(f"{parameterName}, interval {intervalIndex}: expected a finite, non-negative time, received {item!r}.")
                                return

                        normalizedValues.append(item)

                    normalizedParameters[parameterName] = normalizedValues

                # In list mode, None starts mean recording start.
                # None ends are resolved against the recording during execution.
                for intervalName, startName, endName in (("run", "runTmin", "runTmax"), ("noise", "noiseTmin", "noiseTmax")):
                    for intervalIndex, (start, end) in enumerate(zip(normalizedParameters[startName], normalizedParameters[endName]), start=1):
                        effectiveStart = 0.0 if start is None else start

                        if end is not None and end < effectiveStart:
                            self.setError(f"Invalid {intervalName} interval {intervalIndex}: {endName} ({end}) must be >= {startName} ({effectiveStart}).")
                            return

            # Store only after interval validation succeeds.
            for parameterName, values in normalizedParameters.items(): setattr(self, parameterName, values)

            # Configure any remaining action parameters.
            self.configureTraits(**kwargs)

            super().configure()
        
        def _run(self, session: pglSession, mSSSFosters: Callable) -> pglSession:
            import numpy as np
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            # function to get channel types from raw needed in multi-origin SSS with Foster's inverse method
            def getChannelTypes(raw):
                """Return 1 for magnetometers, 0 for gradiometers, in the order of
                apply_preprocessing's meg_picks. Returns None (after warning) if the raw
                can't be used."""
                megPicks = mne.pick_types(raw.info, meg=True)
                if len(megPicks) == 0:
                    pglMessages.warning("No MEG channels found in raw (EEG-only or non-MEG data?)")
                    return None

                types = np.array(raw.get_channel_types(picks=megPicks))
                unsupported = set(types) - {"mag", "grad"}
                if unsupported:
                    pglMessages.warning(f"Unsupported MEG channel types: {sorted(unsupported)}")
                    return None

                return (types == "mag").astype(int)
            
            # get the channel types from the data
            channelTypes = getChannelTypes(session.mne.raw)
            if channelTypes is None:
                self.setError("Failed to get channel types from raw data")
                return None
            
            
            # get the run and noise boundaries
            # interval parameters are None.
            if all(value is None for value in (self.runTmin, self.runTmax, self.noiseTmin, self.noiseTmax)):
                derivedParameters = session.mne.derivedParameters

                if derivedParameters is None or "runAndNoiseIntervals" not in derivedParameters:
                    self.setError("Run and noise intervals not found in derived parameters. Run getRunAndNoiseBoundaries first.")
                    return None

                intervals = derivedParameters["runAndNoiseIntervals"]
                runs = intervals.get("runs")
                selectedNoiseSegments = intervals.get("selectedNoiseSegments")

                if runs is None or len(runs) == 0:
                    self.setError("No run intervals found in derived parameters.")
                    return None

                if selectedNoiseSegments is None or len(selectedNoiseSegments) != len(runs):
                    self.setError("Derived parameters must contain one selected noise segment per run.")
                    return None

                missingNoise = [index for index, segment in enumerate(selectedNoiseSegments, start=1) if segment is None]

                if missingNoise:
                    self.setError(f"No selected noise segment available for run(s): {missingNoise}.")
                    return None

                # Build aligned lists: entry i describes run i and its selected noise.
                self.runTmin = [start for start, end in runs]
                self.runTmax = [end for start, end in runs]
                self.noiseTmin = [start for start, end in selectedNoiseSegments]
                self.noiseTmax = [end for start, end in selectedNoiseSegments]
                
                # Run boundaries may default to the beginning/end of the current Raw.
                lastTime = float(session.mne.raw.times[-1])
                self.runTmin = [0.0 if value is None else value for value in self.runTmin]
                self.runTmax = [lastTime if value is None else value for value in self.runTmax]

                # Noise boundaries must both be explicitly available.
                for intervalIndex, (start, end) in enumerate(zip(self.noiseTmin, self.noiseTmax), start=1):
                    if start is None or end is None:
                        self.setError(f"Noise interval {intervalIndex}: noiseTmin and noiseTmax must both be specified; received ({start}, {end}).")
                        return None
                    
            try:
                # make a copy of the source raw data to avoid modifying the original
                sourceRaw = session.mne.raw
                sourceRaw.load_data()
                outputRaw = sourceRaw.copy()
                megPicks = mne.pick_types(sourceRaw.info, meg=True)

                for iRun, (runStart, runEnd, noiseStart, noiseEnd) in enumerate(zip(self.runTmin, self.runTmax, self.noiseTmin, self.noiseTmax), start=1):
                    pglMessages.printHeader(f"Fostering run {iRun}/{len(self.runTmin)}")
                    pglMessages.print(f"Run {iRun}: {runStart} - {runEnd} (noise {noiseStart} - {noiseEnd})")
                    diagnostics = {}
                    processedRaw = mSSSFosters(
                        center1=self.center1,
                        center2=self.center2,
                        raw=sourceRaw,
                        do_fos=self.doFosters,
                        Ntmin=noiseStart,
                        Ntmax=noiseEnd,
                        do_msss=self.doMultiOrigin,
                        ch_types=channelTypes,
                        Lin=self.Lin,
                        Lout=self.Lout,
                        run_tmin=runStart,
                        run_tmax=runEnd,
                        diagnostics=diagnostics,
                    )
                    fig, summary = self.plotMsssDiagnostics(sourceRaw, diagnostics, noiseTmin=noiseStart, noiseTmax=noiseEnd, dataLabel="Before preprocessing")
                    session.mne.report.add_figure(fig=fig, title="Power spectrum", section="Xan's mSSS with Fosters")
                    
                    start = int(sourceRaw.time_as_index(runStart, use_rounding=True)[0])
                    stop = int(sourceRaw.time_as_index(runEnd, use_rounding=True)[0]) + 1
                    outputRaw._data[megPicks, start:stop] = processedRaw._data[megPicks, start:stop]

                session.mne.raw = outputRaw
            except Exception as e:
                self.setError(f"Error running mSSS Foster's inverse method: {str(e)}")
                return None

            return session
        
        def plotMsssDiagnostics(
                self,
                raw,
                diagnostics,
                *,
                noiseCov=None,
                noiseTmin=None,
                noiseTmax=None,
                dataLabel="Original recording",
                nCols=6,
                chunkSeconds=20.0,
                spectrumScale="log",
                basisMagScale=None,
                whitenRcond=1e-10,
                pinvRcond=1e-10,
                topoLogFloor=None,
                show=True,
            ):
                """
                Plot noise-whitened joint internal/external RMS diagnostics.

                Display conventions
                -------------------
                Spectrum:
                    Coefficient RMS divided by the largest coefficient RMS across
                    BOTH internal and external modes.

                Topomaps:
                    Each spatial pattern is normalized to peak absolute magnitude 1,
                    then multiplied by its relative RMS coefficient.

                    Therefore:
                        max(abs(displayedMap[:, j])) == relativeRmsWeights[j]

                    Topomap colors are dimensionless, not fT.

                    Positive and negative magnitudes use a symmetric-log scale.
                    Above topoLogFloor, successive decades are equally spaced on
                    the colorbar. A small linear region surrounds zero.

                Covariance:
                    Original magnetometer covariance block in fT², using RdBu_r
                    and its own symmetric-log normalization, independent of maps.

                Parameters
                ----------
                raw : mne.io.BaseRaw
                    Original recording to inspect input activity, or processed
                    recording to inspect remaining activity.

                    Must have the same time origin and geometry as the recording
                    used to produce diagnostics.

                diagnostics : dict
                    Saved preprocessing diagnostics containing:
                        basis_internal
                        basis_external
                        channel_names
                        do_msss
                        Lin
                        Lout

                    Optional entries:
                        noise_cov
                        run_tmin
                        run_tmax

                noiseCov : mne.Covariance | ndarray | None
                    Noise covariance in physical SI sensor units.

                    An ndarray must follow diagnostics["channel_names"].

                    If None, use the saved covariance. If unavailable, estimate
                    covariance using noiseTmin/noiseTmax in the supplied raw.

                noiseTmin, noiseTmax : float | None
                    Noise interval relative to the beginning of raw.
                    Used only if no covariance is supplied or saved.

                dataLabel : str
                    Figure description.

                nCols : int
                    Number of topomap columns.

                chunkSeconds : float
                    Chunk duration for accumulating temporal statistics.

                spectrumScale : {"log", "linear"}
                    Relative RMS spectrum scale.

                basisMagScale : float | None
                    Magnetometer-row scaling already present in the stored basis.

                    Stored magnetometer rows are divided by this value before
                    fitting. None defaults to 100 for the custom mSSS code and
                    1 for the MNE-returned SSS basis.

                    Set explicitly if your basis construction differs.

                whitenRcond : float
                    Relative eigenvalue cutoff for the standardized noise covariance.

                pinvRcond : float
                    Relative singular-value cutoff for the whitened real design.

                topoLogFloor : float
                    Positive magnitude below which topomap colors become linear
                    around zero. Must lie strictly between 0 and 1.

                    For example, 1e-4 gives logarithmic colors from 0.0001 to 1,
                    separately for positive and negative values.

                show : bool
                    Display the figure.

                Returns
                -------
                fig : matplotlib.figure.Figure
                    Diagnostic figure.

                summary : dict
                    Coefficients, relative weights, displayed maps, physical maps,
                    covariance, and fitting diagnostics.

                Notes
                -----
                This is a joint internal/external diagnostic fit, not a repetition
                of Foster's inverse or its internal-only coefficient calculation.

                For complex S = u + iv and a = x + iy:
                    real(S * a) = u*x - v*y.

                Each stored mode therefore has two real design columns:
                    [real(S), -imag(S)].

                Redundant modes, including conjugate harmonic pairs, receive a
                minimum-norm coefficient allocation. Their individual weights
                are basis-dependent and are not percentages of variance explained.

                For a complex mode, the spatial shape summarizes sensor RMS, with
                signs taken from its dominant real spatial pattern. It need not
                represent an instantaneous field configuration.

                All samples in the saved processed segment are included, matching
                the supplied preprocessing implementation.

                Neither raw nor diagnostics is modified.
                """
                import warnings
                import numpy as np
                import matplotlib.pyplot as plt
                import mne
                from matplotlib.colors import SymLogNorm

                # ---------------------------------------------------------------
                # Helpers.
                # ---------------------------------------------------------------
                def alignCovariance(covariance):
                    """Return dense covariance in the saved basis channel order."""
                    if isinstance(covariance, mne.Covariance):
                        covarianceNames = list(covariance.ch_names)
                        missingNames = [name for name in channelNames if name not in covarianceNames]

                        if missingNames:
                            raise ValueError(f"Noise covariance is missing channels: {missingNames}")

                        channelOrder = [covarianceNames.index(name) for name in channelNames]
                        matrix = np.asarray(covariance["data"])

                        if matrix.ndim == 1:
                            matrix = np.diag(matrix)

                        matrix = matrix[np.ix_(channelOrder, channelOrder)]
                    else:
                        matrix = np.asarray(covariance)

                    if matrix.shape != (nChannels, nChannels):
                        raise ValueError("Noise covariance must match the saved basis channels.")

                    if np.iscomplexobj(matrix):
                        if np.any(matrix.imag != 0):
                            raise ValueError("Sensor noise covariance must be real-valued.")

                        matrix = matrix.real

                    matrix = np.asarray(matrix, dtype=float)

                    if not np.isfinite(matrix).all():
                        raise ValueError("Noise covariance contains NaN or infinite values.")

                    if np.any(np.diag(matrix) <= 0):
                        raise ValueError("Every fitted channel must have positive noise variance.")

                    channelStd = np.sqrt(np.diag(matrix))
                    standardized = matrix / channelStd[:, None] / channelStd[None, :]

                    if not np.allclose(standardized, standardized.T, rtol=1e-7, atol=1e-10):
                        raise ValueError("Noise covariance must be symmetric.")

                    return (matrix + matrix.T) / 2

                def makeSignedRmsMap(spatialVector, coefficientMoment):
                    """Calculate sensor RMS and assign dominant-pattern spatial signs."""
                    eigenvalues, eigenvectors = np.linalg.eigh(coefficientMoment)
                    eigenvalues = np.maximum(eigenvalues, 0.0)

                    spatialParts = np.column_stack([spatialVector.real, -spatialVector.imag])
                    fieldFactor = spatialParts @ (eigenvectors * np.sqrt(eigenvalues)[None, :])
                    sensorRms = np.sqrt(np.sum(fieldFactor**2, axis=1))

                    if not np.any(sensorRms):
                        return sensorRms, 1.0

                    spatialModes, singularValues, _ = np.linalg.svd(fieldFactor, full_matrices=False)
                    referencePattern = spatialModes[:, 0]
                    referenceIndex = np.argmax(np.abs(referencePattern))

                    # The global sign is arbitrary; choose it reproducibly.
                    if referencePattern[referenceIndex] < 0:
                        referencePattern = -referencePattern

                    spatialSigns = np.where(referencePattern >= 0, 1.0, -1.0)
                    dominantFraction = singularValues[0]**2 / np.sum(singularValues**2)

                    return spatialSigns * sensorRms, float(dominantFraction)

                # ---------------------------------------------------------------
                # 1. Validate arguments and align channels.
                # ---------------------------------------------------------------
                if not isinstance(nCols, int) or nCols < 1:
                    raise ValueError("nCols must be a positive integer.")

                if not np.isfinite(chunkSeconds) or chunkSeconds <= 0:
                    raise ValueError("chunkSeconds must be positive and finite.")

                if spectrumScale not in ("linear", "log"):
                    raise ValueError("spectrumScale must be 'linear' or 'log'.")

                if not 0 < whitenRcond < 1:
                    raise ValueError("whitenRcond must lie between zero and one.")

                if not 0 < pinvRcond < 1:
                    raise ValueError("pinvRcond must lie between zero and one.")

                if topoLogFloor is not None and (not np.isfinite(topoLogFloor) or not 0 < topoLogFloor < 1):
                    raise ValueError("topoLogFloor must lie strictly between zero and one.")

                channelNames = list(diagnostics["channel_names"])
                nChannels = len(channelNames)
                missingNames = [name for name in channelNames if name not in raw.ch_names]

                if missingNames:
                    raise ValueError(f"Basis channels are missing from raw: {missingNames}")

                channelPicks = np.array([raw.ch_names.index(name) for name in channelNames])
                basisInfo = mne.pick_info(raw.info, channelPicks, copy=True)

                internalBasis = np.asarray(diagnostics["basis_internal"])
                externalBasis = np.asarray(diagnostics["basis_external"])

                if internalBasis.ndim != 2 or externalBasis.ndim != 2:
                    raise ValueError("Both basis matrices must be two-dimensional.")

                if internalBasis.shape[0] != nChannels or externalBasis.shape[0] != nChannels:
                    raise ValueError("Basis rows must match the saved channel names.")

                if internalBasis.shape[1] == 0:
                    raise ValueError("The internal basis contains no modes.")

                if not np.isfinite(internalBasis).all() or not np.isfinite(externalBasis).all():
                    raise ValueError("The basis contains NaN or infinite values.")

                nInternal = internalBasis.shape[1]
                nExternal = externalBasis.shape[1]
                nModes = nInternal + nExternal

                magPicks = mne.pick_types(basisInfo, meg="mag", ref_meg=False, exclude=[])

                if len(magPicks) < 4:
                    raise ValueError("At least four magnetometers with usable positions are required.")

                magInfo = mne.pick_info(basisInfo, magPicks, copy=True)
                magNames = [channelNames[pick] for pick in magPicks]

                # Match the inclusive run endpoint in the supplied preprocessing code.
                runTmin = diagnostics.get("run_tmin")
                runTmax = diagnostics.get("run_tmax")
                sampleStart = 0 if runTmin is None else int(raw.time_as_index(runTmin, use_rounding=True)[0])
                sampleStop = raw.n_times if runTmax is None else int(raw.time_as_index(runTmax, use_rounding=True)[0]) + 1

                if not 0 <= sampleStart < sampleStop <= raw.n_times:
                    raise ValueError("The saved processed segment is outside this recording.")

                # ---------------------------------------------------------------
                # 2. Obtain physical noise covariance.
                # ---------------------------------------------------------------
                covariance = noiseCov if noiseCov is not None else diagnostics.get("noise_cov")

                if covariance is None:
                    if noiseTmin is None and noiseTmax is None:
                        raise ValueError("Supply noiseCov or specify a suitable noise interval.")

                    covariance = mne.compute_raw_covariance(raw, tmin=noiseTmin, tmax=noiseTmax, picks=channelPicks, method="empirical", rank="info", verbose=False)

                noiseMatrix = alignCovariance(covariance)

                # ---------------------------------------------------------------
                # 3. Undo stored magnetometer-row scaling.
                # ---------------------------------------------------------------
                if basisMagScale is None:
                    basisMagScale = 100.0 if diagnostics["do_msss"] else 1.0

                if not np.isfinite(basisMagScale) or basisMagScale <= 0:
                    raise ValueError("basisMagScale must be positive and finite.")

                physicalBasis = np.array(np.concatenate([internalBasis, externalBasis], axis=1), dtype=complex, copy=True)
                physicalBasis[magPicks] /= basisMagScale

                # ---------------------------------------------------------------
                # 4. Noise whitening.
                #
                # Standardization makes covariance rank selection independent of
                # the differing SI magnitudes of magnetometers and gradiometers.
                # ---------------------------------------------------------------
                noiseStd = np.sqrt(np.diag(noiseMatrix))
                standardizedNoise = noiseMatrix / noiseStd[:, None] / noiseStd[None, :]
                noiseEigenvalues, noiseEigenvectors = np.linalg.eigh(standardizedNoise)
                largestNoiseEigenvalue = float(noiseEigenvalues[-1])

                if largestNoiseEigenvalue <= 0:
                    raise ValueError("Noise covariance has no positive eigenvalues.")

                if noiseEigenvalues[0] < -1e-8 * largestNoiseEigenvalue:
                    raise ValueError("Noise covariance is not positive semidefinite.")

                retainedNoise = noiseEigenvalues > whitenRcond * largestNoiseEigenvalue
                noiseRank = int(np.sum(retainedNoise))
                whitener = (noiseEigenvectors[:, retainedNoise] / np.sqrt(noiseEigenvalues[retainedNoise])).T / noiseStd[None, :]

                if noiseRank < nChannels:
                    warnings.warn(f"Whitening retains {noiseRank} of {nChannels} noise directions; discarded directions are excluded from fitting.", stacklevel=2)

                # ---------------------------------------------------------------
                # 5. Joint real-valued internal/external fit.
                #
                # Columns 2*j and 2*j+1 belong to stored mode j.
                # ---------------------------------------------------------------
                realBasis = np.empty((nChannels, 2 * nModes))
                realBasis[:, 0::2] = physicalBasis.real
                realBasis[:, 1::2] = -physicalBasis.imag
                whitenedBasis = whitener @ realBasis

                leftVectors, singularValues, rightVectorsH = np.linalg.svd(whitenedBasis, full_matrices=False)

                if singularValues.size == 0 or singularValues[0] == 0:
                    raise ValueError("The whitened joint basis has zero rank.")

                singularCutoff = pinvRcond * singularValues[0]
                retainedBasis = singularValues > singularCutoff
                jointRank = int(np.sum(retainedBasis))
                fitDirections = leftVectors[:, retainedBasis]
                coefficientInverse = (rightVectorsH[retainedBasis].T / singularValues[retainedBasis]) @ fitDirections.T
                retainedCondition = float(singularValues[0] / singularValues[retainedBasis][-1])

                activeColumnCount = int(np.sum(np.linalg.norm(whitenedBasis, axis=0) > singularCutoff))

                if jointRank < activeColumnCount:
                    warnings.warn(
                        "The joint real model contains redundant or truncated directions. "
                        "Individual coefficients use a minimum-norm allocation and are "
                        "basis-dependent; conjugate harmonic pairs can be redundant.",
                        stacklevel=2,
                    )

                if retainedCondition > 1e6:
                    warnings.warn(f"The retained whitened design is ill-conditioned ({retainedCondition:.2g}); individual coefficients may be unstable.", stacklevel=2)

                # ---------------------------------------------------------------
                # 6. Accumulate temporal moments over the processed segment.
                # ---------------------------------------------------------------
                chunkSamples = max(1, int(round(chunkSeconds * raw.info["sfreq"])))
                whiteMean = np.zeros(noiseRank)
                whiteScatter = np.zeros((noiseRank, noiseRank))
                sampleCount = 0

                for chunkStart in range(sampleStart, sampleStop, chunkSamples):
                    chunkStop = min(chunkStart + chunkSamples, sampleStop)
                    dataChunk = raw.get_data(picks=channelPicks, start=chunkStart, stop=chunkStop)

                    if not np.isfinite(dataChunk).all():
                        raise ValueError("The selected data contain NaN or infinite values.")

                    whiteChunk = whitener @ dataChunk
                    chunkCount = whiteChunk.shape[1]
                    chunkMean = whiteChunk.mean(axis=1)
                    centeredChunk = whiteChunk - chunkMean[:, None]
                    chunkScatter = centeredChunk @ centeredChunk.T

                    combinedCount = sampleCount + chunkCount
                    meanDifference = chunkMean - whiteMean

                    whiteScatter += chunkScatter + np.outer(meanDifference, meanDifference) * (sampleCount * chunkCount / combinedCount)
                    whiteMean += meanDifference * (chunkCount / combinedCount)
                    sampleCount = combinedCount

                whiteScatter = (whiteScatter + whiteScatter.T) / 2

                # RMS includes both temporal variance and temporal mean.
                whiteSecondMoment = whiteScatter / sampleCount + np.outer(whiteMean, whiteMean)
                coefficientSecondMoment = coefficientInverse @ whiteSecondMoment @ coefficientInverse.T
                coefficientSecondMoment = (coefficientSecondMoment + coefficientSecondMoment.T) / 2

                if not np.isfinite(coefficientSecondMoment).all():
                    raise ValueError("Nonfinite coefficient moments were produced; inspect data, covariance, and conditioning.")

                coefficientDiagonal = np.maximum(np.diag(coefficientSecondMoment), 0.0)
                rmsWeights = np.sqrt(coefficientDiagonal[0::2] + coefficientDiagonal[1::2])

                maxRmsWeight = float(np.max(rmsWeights))
                relativeRmsWeights = rmsWeights / maxRmsWeight if maxRmsWeight > 0 else np.zeros_like(rmsWeights)
                rmsLabels = [f"{weight:.3g}" for weight in relativeRmsWeights]

                # ---------------------------------------------------------------
                # 7. Mode names.
                # ---------------------------------------------------------------
                internalOrder = diagnostics["Lin"]
                externalOrder = diagnostics["Lout"]
                internalLm = [(ell, order) for ell in range(1, internalOrder + 1) for order in range(-ell, ell + 1)]
                externalLm = [(ell, order) for ell in range(1, externalOrder + 1) for order in range(-ell, ell + 1)]
                modeTitles = []

                for modeIndex in range(nInternal):
                    if diagnostics["do_msss"]:
                        description = f"Internal SVD mode {modeIndex + 1}"
                    elif nInternal == len(internalLm):
                        ell, order = internalLm[modeIndex]
                        description = rf"Internal: $\ell={ell},\ m={order}$"
                    else:
                        description = f"Internal mode {modeIndex + 1}"

                    modeTitles.append(f"#{modeIndex + 1}\n{description}")

                for modeIndex in range(nExternal):
                    if nExternal == len(externalLm):
                        ell, order = externalLm[modeIndex]
                        description = rf"External: $\ell={ell},\ m={order}$"
                    else:
                        description = f"External mode {modeIndex + 1}"

                    modeTitles.append(f"#{nInternal + modeIndex + 1}\n{description}")

                # ---------------------------------------------------------------
                # 8. RMS-weighted maps.
                #
                # First normalize each shape to peak magnitude 1.
                # Then multiply by its relative coefficient RMS.
                #
                # Physical maps are preserved separately, but not displayed.
                # ---------------------------------------------------------------
                signedRmsMapsFt = np.zeros((len(magPicks), nModes))
                weightedRmsMaps = np.zeros((len(magPicks), nModes))
                dominantSpatialFraction = np.zeros(nModes)
                missingMagPatterns = []

                for modeIndex in range(nModes):
                    coefficientSlice = slice(2 * modeIndex, 2 * modeIndex + 2)
                    modeMoment = coefficientSecondMoment[coefficientSlice, coefficientSlice]
                    signedMap, dominantFraction = makeSignedRmsMap(physicalBasis[magPicks, modeIndex], modeMoment)

                    signedRmsMapsFt[:, modeIndex] = signedMap * 1e15
                    dominantSpatialFraction[modeIndex] = dominantFraction
                    peakMagnitude = float(np.max(np.abs(signedMap)))

                    if peakMagnitude > 0:
                        weightedRmsMaps[:, modeIndex] = (signedMap / peakMagnitude) * relativeRmsWeights[modeIndex]
                    elif relativeRmsWeights[modeIndex] > 0:
                        missingMagPatterns.append(modeIndex + 1)

                if not np.isfinite(weightedRmsMaps).all():
                    raise ValueError("The displayed topomaps contain nonfinite values.")

                displayedMapPeaks = np.max(np.abs(weightedRmsMaps), axis=0)
                displayedMapRms = np.sqrt(np.mean(weightedRmsMaps**2, axis=0))
                validMapMask = np.ones(nModes, dtype=bool)

                if missingMagPatterns:
                    validMapMask[np.asarray(missingMagPatterns) - 1] = False
                    warnings.warn(f"Modes {missingMagPatterns} have nonzero coefficient RMS but no magnetometer contribution to normalize. Their maps remain zero.", stacklevel=2)

                positiveMapMask = validMapMask & (relativeRmsWeights > 0)

                if not np.allclose(displayedMapPeaks[positiveMapMask] / relativeRmsWeights[positiveMapMask], 1.0, rtol=1e-8, atol=0):
                    raise RuntimeError("Displayed topomap peaks do not match the relative RMS coefficients.")

                mixedModes = np.flatnonzero(dominantSpatialFraction < 0.95)

                if mixedModes.size:
                    warnings.warn(
                        f"{mixedModes.size} modes contain appreciable activity in two real "
                        "spatial patterns. Their maps summarize sensor RMS, with signs "
                        "taken from the dominant spatial pattern.",
                        stacklevel=2,
                    )

                # Adapt the color scale to the smallest nonzero coefficient that
                # actually has a displayed magnetometer pattern.
                visibleWeights = relativeRmsWeights[positiveMapMask]

                if topoLogFloor is None:
                    smallestVisibleRms = float(np.min(visibleWeights)) if visibleWeights.size else 1.0

                    # Put the white/linear region at least one decade below the
                    # weakest component's peak, rounded down to a power of ten.
                    floorExponent = int(np.floor(np.log10(smallestVisibleRms))) - 1
                    topoLogFloor = max(10.0 ** floorExponent, np.nextafter(0.0, 1.0))

                # Compress the central linear region so it does not consume much
                # of the colorbar. Above the threshold, decades are equally spaced.
                mapNorm = SymLogNorm(linthresh=topoLogFloor, linscale=0.1, vmin=-1.0, vmax=1.0, base=10)
                # ---------------------------------------------------------------
                # 9. Restore the independent covariance appearance.
                #
                # Same RdBu_r colormap and percentile-based symmetric-log scale
                # used before the topomap color-scale changes.
                # ---------------------------------------------------------------
                covarianceFt2 = noiseMatrix[np.ix_(magPicks, magPicks)] * 1e30
                covarianceLimit = float(np.max(np.abs(covarianceFt2)))
                covarianceLimit = covarianceLimit if covarianceLimit > 0 else 1.0

                offDiagonalMask = ~np.eye(len(magPicks), dtype=bool)
                offDiagonalValues = np.abs(covarianceFt2[offDiagonalMask])
                nonzeroValues = offDiagonalValues[offDiagonalValues > 0]
                covarianceLinearThreshold = float(np.percentile(nonzeroValues, 25)) if nonzeroValues.size else covarianceLimit * 0.01
                covarianceLinearThreshold = max(covarianceLinearThreshold, covarianceLimit * 1e-6)
                covarianceNorm = SymLogNorm(linthresh=covarianceLinearThreshold, vmin=-covarianceLimit, vmax=covarianceLimit, base=10)

                # ---------------------------------------------------------------
                # 10. Layout.
                # ---------------------------------------------------------------
                nInternalRows = int(np.ceil(nInternal / nCols))
                nExternalRows = int(np.ceil(nExternal / nCols))
                nMapRows = nInternalRows + nExternalRows
                mapRowHeights = [1.0] * nInternalRows

                if nExternal:
                    mapRowHeights += [0.25] + [1.0] * nExternalRows

                mapSectionHeight = 3.2 * nMapRows + (0.8 if nExternal else 0.0)

                fig = plt.figure(figsize=(max(16, 2.7 * nCols), 5.5 + mapSectionHeight), constrained_layout=True)
                outerGrid = fig.add_gridspec(2, 1, height_ratios=[4.5, mapSectionHeight])
                topGrid = outerGrid[0].subgridspec(1, 2, width_ratios=[3.2, 1.4])
                mapGrid = outerGrid[1].subgridspec(len(mapRowHeights), nCols, height_ratios=mapRowHeights)

                spectrumAxis = fig.add_subplot(topGrid[0])
                covarianceAxis = fig.add_subplot(topGrid[1])

                internalColor = "#4477AA"
                externalColor = "#DD8844"
                externalBackground = "#FFF0DC"
                basisNumbers = np.arange(1, nModes + 1)
                barColors = np.array([internalColor] * nInternal + [externalColor] * nExternal)

                # ---------------------------------------------------------------
                # 11. Relative coefficient RMS spectrum.
                # ---------------------------------------------------------------
                if nExternal:
                    spectrumAxis.axvspan(nInternal + 0.5, nModes + 0.5, color=externalBackground, zorder=0)
                    spectrumAxis.axvline(nInternal + 0.5, color="0.4", linewidth=1)

                actualSpectrumScale = spectrumScale
                positiveMask = relativeRmsWeights > 0
                positiveWeights = relativeRmsWeights[positiveMask]

                if spectrumScale == "log" and positiveWeights.size:
                    lowerLimit = max(float(positiveWeights.min() / 2), np.nextafter(0.0, 1.0))
                    spectrumAxis.set_yscale("log", nonpositive="mask")
                    spectrumAxis.bar(basisNumbers[positiveMask], relativeRmsWeights[positiveMask] - lowerLimit, bottom=lowerLimit, color=barColors[positiveMask], width=0.8, zorder=2)
                    spectrumAxis.set_ylim(lowerLimit, 2.0)

                    zeroCount = int(np.sum(~positiveMask))

                    if zeroCount:
                        spectrumAxis.text(0.01, 0.02, f"{zeroCount} zero weights omitted on log axis", transform=spectrumAxis.transAxes, ha="left", va="bottom", fontsize=8)
                else:
                    if spectrumScale == "log":
                        actualSpectrumScale = "linear"
                        warnings.warn("All coefficients have zero RMS; using a linear spectrum.", stacklevel=2)

                    spectrumAxis.bar(basisNumbers, relativeRmsWeights, color=barColors, width=0.8, zorder=2)
                    spectrumAxis.set_ylim(0, 1.15)

                tickStep = max(1, int(np.ceil(nModes / 25)))
                spectrumTicks = np.unique(np.r_[basisNumbers[::tickStep], nInternal, nModes])

                spectrumAxis.set_xticks(spectrumTicks)
                spectrumAxis.set_xlim(0.5, nModes + 0.5)
                spectrumAxis.set_xlabel("Spatial basis function number (# matches topomap titles)")
                spectrumAxis.set_ylabel("Relative RMS coefficient (largest = 1)")
                spectrumAxis.set_title("Noise-whitened joint fit — RMS coefficients", pad=28)
                spectrumAxis.grid(axis="y", which="major", alpha=0.3)
                spectrumAxis.grid(axis="y", which="minor", alpha=0.12)
                spectrumAxis.set_axisbelow(True)
                spectrumAxis.text((nInternal + 1) / 2, 1.02, "Internal", transform=spectrumAxis.get_xaxis_transform(), ha="center", color=internalColor)

                if nExternal:
                    spectrumAxis.text(nInternal + (nExternal + 1) / 2, 1.02, "External", transform=spectrumAxis.get_xaxis_transform(), ha="center", color="#A65F22")

                # ---------------------------------------------------------------
                # 12. Covariance panel: independent normalization and colorbar.
                # ---------------------------------------------------------------
                covarianceImage = covarianceAxis.imshow(covarianceFt2, origin="upper", interpolation="nearest", cmap="RdBu_r", norm=covarianceNorm)
                covarianceTicks = np.unique(np.linspace(0, len(magNames) - 1, min(6, len(magNames))).astype(int))
                covarianceLabels = [magNames[index] for index in covarianceTicks]

                covarianceAxis.set_xticks(covarianceTicks)
                covarianceAxis.set_yticks(covarianceTicks)
                covarianceAxis.set_xticklabels(covarianceLabels, rotation=90, fontsize=7)
                covarianceAxis.set_yticklabels(covarianceLabels, fontsize=7)
                covarianceAxis.set_xlabel("Magnetometer channel")
                covarianceAxis.set_ylabel("Magnetometer channel")
                covarianceAxis.set_title("Noise covariance: magnetometer block\nIndependent symmetric-log scale")

                covarianceColorbar = fig.colorbar(covarianceImage, ax=covarianceAxis, shrink=0.8, pad=0.03)
                covarianceColorbar.set_label("Noise covariance (fT²)")

                # ---------------------------------------------------------------
                # 13. Topomaps, with external modes on a new row.
                # ---------------------------------------------------------------
                mapAxes = []
                dividerAxis = None

                if nExternal:
                    dividerAxis = fig.add_subplot(mapGrid[nInternalRows, :])
                    dividerAxis.set_axis_off()
                    dividerAxis.plot([0.0, 1.0], [0.5, 0.5], transform=dividerAxis.transAxes, color=externalColor, linewidth=1.5)
                    dividerAxis.text(0.5, 0.5, "External basis functions", transform=dividerAxis.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color="#A65F22", bbox={"facecolor": fig.get_facecolor(), "edgecolor": "none", "pad": 5})

                for modeIndex in range(nModes):
                    if modeIndex < nInternal:
                        mapRow = modeIndex // nCols
                        mapCol = modeIndex % nCols
                    else:
                        externalIndex = modeIndex - nInternal
                        mapRow = nInternalRows + 1 + externalIndex // nCols
                        mapCol = externalIndex % nCols

                    mapAxis = fig.add_subplot(mapGrid[mapRow, mapCol])

                    if modeIndex >= nInternal:
                        mapAxis.set_facecolor(externalBackground)

                    mapImage, _ = mne.viz.plot_topomap(
                        weightedRmsMaps[:, modeIndex],
                        magInfo,
                        ch_type="mag",
                        axes=mapAxis,
                        show=False,
                        cmap="RdBu_r",
                        cnorm=mapNorm,
                        contours=0,
                        sensors=True,
                        outlines="head",
                        extrapolate="local",
                        image_interp="linear",
                    )

                    mapAxis.set_axis_on()
                    mapAxis.set_xticks([])
                    mapAxis.set_yticks([])

                    for spine in mapAxis.spines.values():
                        spine.set_visible(False)

                    mapAxis.set_xlabel("Left → right", fontsize=8)
                    mapAxis.set_ylabel("Posterior → anterior", fontsize=8)

                    titleText = f"{modeTitles[modeIndex]}\nRelative RMS = {rmsLabels[modeIndex]}"

                    if modeIndex + 1 in missingMagPatterns:
                        titleText += "\n(no magnetometer pattern)"
                    elif dominantSpatialFraction[modeIndex] < 0.95:
                        titleText += "\n(two-pattern mode)"

                    mapAxis.set_title(titleText, fontsize=9, pad=8)
                    mapAxes.append(mapAxis)

                # ---------------------------------------------------------------
                # 14. Shared signed-log topomap colorbar.
                #
                # Tick labels are actual relative values, not log10(value).
                # Their positions are logarithmically spaced above the floor.
                # ---------------------------------------------------------------
                colorbarAxes = mapAxes + ([dividerAxis] if dividerAxis is not None else [])
                mapColorbar = fig.colorbar(mapImage, ax=colorbarAxes, orientation="horizontal", fraction=0.02, pad=0.02, aspect=60)

                lowestExponent = int(np.ceil(np.log10(topoLogFloor)))
                positiveTicks = 10.0 ** np.arange(lowestExponent, 1)
                colorTicks = np.r_[-positiveTicks[::-1], 0.0, positiveTicks]

                mapColorbar.set_ticks(colorTicks)
                mapColorbar.set_ticklabels([f"{value:.3g}" for value in colorTicks])
                mapColorbar.set_label(f"Relative RMS × peak-normalized spatial pattern (dimensionless)\nSigned logarithmic colors; linear only within ±{topoLogFloor:.3g}")

                segmentText = f"{raw.times[sampleStart]:.2f}–{raw.times[sampleStop - 1]:.2f} s"
                fig.suptitle(f"{dataLabel} — processed segment: {segmentText}\nEach topomap's peak magnitude equals its relative RMS coefficient\nOne shared signed-log color scale; colors do not represent variance explained", fontsize=12)

                summary = {
                    "rmsWeights": rmsWeights,
                    "relativeRmsWeights": relativeRmsWeights,
                    "rmsLabels": rmsLabels,
                    "weightedRmsMaps": weightedRmsMaps,
                    "displayedMapPeaks": displayedMapPeaks,
                    "displayedMapRms": displayedMapRms,
                    "signedRmsMapsFt": signedRmsMapsFt,
                    "dominantSpatialFraction": dominantSpatialFraction,
                    "missingMagPatterns": missingMagPatterns,
                    "modeTitles": modeTitles,
                    "noiseCovarianceSi": noiseMatrix,
                    "whitener": whitener,
                    "noiseRank": noiseRank,
                    "noiseEigenvalues": noiseEigenvalues,
                    "basisMagScale": basisMagScale,
                    "channelNames": channelNames,
                    "magnetometerNames": magNames,
                    "nInternal": nInternal,
                    "nExternal": nExternal,
                    "sampleStart": sampleStart,
                    "sampleStop": sampleStop,
                    "jointRank": jointRank,
                    "retainedCondition": retainedCondition,
                    "singularValues": singularValues,
                    "spectrumScale": actualSpectrumScale,
                    "topoLogFloor": topoLogFloor,
                    "topomapNormalization": "Peak magnitude equals relative RMS coefficient",
                    "dataLabel": dataLabel,
                }

                if show:
                    plt.show()

                return fig, summary

    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # set run and noise boundaries for running mSSS with Fosters
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class getRunAndNoiseBoundaries(pglAction):
        '''
        Get list of tuples representing the beginning and end of runs and noise segments        
        '''
        
        noiseTriggerLabel = Unicode("blank", help="Label for the trigger to identify noise segments")
        triggerSet = Unicode(allow_none=True, default_value=None, help="Trigger set in which triggerLabel is defineds. See configureEvents. Default will look for the triggerLabel and use the first set that it is defined in")
        
        def _run(self, session:pglSession) -> pglSession:
            # resolve the trigger set
            try:
                self.triggerSet = self.resolveTriggerSet(session, self.noiseTriggerLabel, self.triggerSet)
            except Exception as e:
                self.setError(f"Error resolving trigger set: {str(e)}")
                return None

            try:
                runs, noiseSegments = self.extractRunAndNoiseIntervals(session, noiseTriggerLabel=self.noiseTriggerLabel, triggerSet=self.triggerSet)
            except Exception as e:
                self.setError(f"Error extracting or plotting run and noise intervals: {str(e)}")
                return None

            # display them
            self.plotRunAndNoiseIntervals(session, runs, noiseSegments)

            # For each run select a noise segment
            selectedNoiseSegments = self.selectNoiseSegmentsForRuns(runs, noiseSegments)    
            
            # package into mne's derivedParameters
            session.mne.derivedParameters["runAndNoiseIntervals"] = {"runs": runs, "noiseSegments": noiseSegments, "selectedNoiseSegments": selectedNoiseSegments}
            
            return session
    
        
        def resolveTriggerSet(self, session, triggerLabel, triggerSet=None):
            """Find the first trigger set containing triggerLabel, or validate an explicit set."""
            if session is None or session.mne is None or session.mne.eventsID is None:
                raise ValueError("Cannot resolve triggerSet: eventsID is missing. Run configureEvents first.")
            eventsID = session.mne.eventsID
            
            reservedColumns = {"sample", "previousValue", "code", "rawLabel"}
            availableSets = [column for column in eventsID.columns if column not in reservedColumns]

            if triggerSet is not None and triggerSet not in availableSets:
                raise ValueError(f"Unknown triggerSet {triggerSet!r}. Available trigger sets: {availableSets!r}. Trigger sets searched: none.")

            candidateSets = availableSets if triggerSet is None else [triggerSet]
            searchedSets = []

            for setName in candidateSets:
                searchedSets.append(setName)

                if eventsID[setName].eq(triggerLabel).fillna(False).any():
                    if triggerSet is None:
                        pglMessages.message(f"No triggerSet specified. Selected {setName!r}, the first trigger set containing label {triggerLabel!r}. Trigger sets searched: {searchedSets!r}.")
                    else:
                        pglMessages.message(f"Using specified triggerSet {setName!r}, containing label {triggerLabel!r}.")

                    return setName

            raise ValueError(f"Trigger label {triggerLabel!r} was not found in the recorded event labels. Trigger sets searched: {searchedSets!r}.")
        
        def extractRunAndNoiseIntervals(self, session, noiseTriggerLabel, triggerSet):
            """
            Calculate proposed run and noise intervals.

            triggerSet must already be resolved.

            Returns
            -------
            runs : list of tuples
                (start, end) times in seconds relative to the current Raw start.
            noiseSegments : list of tuples
                (start, end) times in seconds relative to the current Raw start.

            Both endpoints are included, matching Raw.crop(tmin=start, tmax=end).

            Runs are the complement of BAD_inter_run annotations.
            Noise starts at a matching trigger and ends one sample before the next
            non-matching trigger, or at recording end. Consecutive matching triggers
            form one segment. Unlabeled triggers count as non-matching.

            Noise intervals are clipped to runs. A noise label remains active across
            excluded gaps until a non-matching trigger, but the gaps are not returned.
            """
            import numpy as np

            # ------------------------------------------------------------------
            # Validate session fields.
            # ------------------------------------------------------------------
            mneData = getattr(session, "mne", None)

            if mneData is None:
                raise ValueError("Cannot extract intervals: session.mne is None. Load raw data and run configureEvents first.")

            raw = getattr(mneData, "raw", None)
            events = getattr(mneData, "events", None)
            eventsID = getattr(mneData, "eventsID", None)

            if raw is None:
                raise ValueError("Cannot extract intervals: session.mne.raw is None. Load raw data first.")
            if events is None:
                raise ValueError("Cannot extract intervals: session.mne.events is None. Run configureEvents first.")
            if eventsID is None:
                raise ValueError("Cannot extract intervals: session.mne.eventsID is None. Run configureEvents first.")
            if triggerSet is None:
                raise ValueError("Cannot extract intervals: triggerSet is None. Call resolveTriggerSet first.")

            reservedColumns = {"sample", "previousValue", "code", "rawLabel"}
            availableSets = [column for column in eventsID.columns if column not in reservedColumns]

            if triggerSet not in availableSets:
                raise ValueError(f"Cannot extract intervals: unknown triggerSet {triggerSet!r}. Available trigger sets: {availableSets!r}.")

            events = np.asarray(events)
            sfreq = float(raw.info["sfreq"])
            nSamples = int(raw.n_times)

            if events.ndim != 2 or events.shape[1] != 3:
                raise ValueError("session.mne.events must have shape (nEvents, 3).")
            if len(events) == 0:
                raise ValueError("Cannot extract intervals: session.mne.events contains no events.")
            if len(eventsID) != len(events):
                raise ValueError("session.mne.eventsID must contain one row per event in session.mne.events.")

            for column in ("sample", "code"):
                if column not in eventsID.columns:
                    raise ValueError(f"session.mne.eventsID is missing required column {column!r}. Run configureEvents again.")

            if not np.array_equal(eventsID["sample"].to_numpy(), events[:, 0]):
                raise ValueError("session.mne.eventsID sample positions do not match session.mne.events. Run configureEvents again.")

            if not np.array_equal(eventsID["code"].to_numpy(), events[:, 2]):
                raise ValueError("session.mne.eventsID trigger codes do not match session.mne.events. Run configureEvents again.")

            eventSamples = events[:, 0].astype(np.int64) - raw.first_samp
            noiseMatches = eventsID[triggerSet].eq(noiseTriggerLabel).fillna(False).to_numpy(dtype=bool)

            if np.any(np.diff(eventSamples) <= 0):
                raise ValueError("session.mne.events must have strictly increasing sample positions.")

            if np.any((eventSamples < 0) | (eventSamples >= nSamples)):
                raise ValueError("Some events are outside session.mne.raw. Update events and their metadata for the current Raw.")

            if not noiseMatches.any():
                raise ValueError(f"Noise trigger label {noiseTriggerLabel!r} was not found in the specified triggerSet {triggerSet!r}.")

            # ------------------------------------------------------------------
            # Read excluded intervals.
            # Attached annotation onsets include raw.first_time.
            # Internally, stops are exclusive; returned endpoints are inclusive.
            # ------------------------------------------------------------------
            badIntervals = []

            for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
                if description != "BAD_inter_run":
                    continue

                start = int(round((onset - raw.first_time) * sfreq))
                stop = int(round((onset + duration - raw.first_time) * sfreq))

                start = max(0, min(nSamples, start))
                stop = max(0, min(nSamples, stop))

                if start < stop:
                    badIntervals.append((start, stop))

            # Merge overlapping or touching excluded intervals.
            mergedBad = []

            for start, stop in sorted(badIntervals):
                if mergedBad and start <= mergedBad[-1][1]:
                    mergedBad[-1] = (mergedBad[-1][0], max(mergedBad[-1][1], stop))
                else:
                    mergedBad.append((start, stop))

            # ------------------------------------------------------------------
            # Runs are the complement of the excluded intervals.
            # Without BAD_inter_run intervals, the entire recording is one run.
            # ------------------------------------------------------------------
            runSamples = []
            cursor = 0

            for start, stop in mergedBad:
                if cursor < start:
                    runSamples.append((cursor, start))
                cursor = stop

            if cursor < nSamples:
                runSamples.append((cursor, nSamples))

            # ------------------------------------------------------------------
            # Merge consecutive matching triggers into continuous noise intervals.
            # ------------------------------------------------------------------
            noiseSamples = []
            noiseStart = None

            for sample, isNoise in zip(eventSamples, noiseMatches):
                if isNoise:
                    if noiseStart is None:
                        noiseStart = int(sample)
                elif noiseStart is not None:
                    noiseSamples.append((noiseStart, int(sample)))
                    noiseStart = None

            if noiseStart is not None:
                noiseSamples.append((noiseStart, nSamples))

            # ------------------------------------------------------------------
            # Clip noise intervals to runs.
            # ------------------------------------------------------------------
            clippedNoise = []

            for runStart, runStop in runSamples:
                for noiseStart, noiseStop in noiseSamples:
                    if noiseStop <= runStart:
                        continue
                    if noiseStart >= runStop:
                        break

                    start = max(runStart, noiseStart)
                    stop = min(runStop, noiseStop)

                    if start < stop:
                        clippedNoise.append((start, stop))

            # Convert to seconds with inclusive endpoints.
            runs = [(start / sfreq, (stop - 1) / sfreq) for start, stop in runSamples]
            noiseSegments = [(start / sfreq, (stop - 1) / sfreq) for start, stop in clippedNoise]

            pglMessages.message(f"Identified {len(runs)} proposed runs and {len(noiseSegments)} noise segments using label {noiseTriggerLabel!r} in triggerSet {triggerSet!r}.")

            return runs, noiseSegments

        def plotRunAndNoiseIntervals(self, session, runs, noiseSegments):
            """
            Plot previously calculated run and noise intervals.

            Inputs are lists of (start, end) tuples in seconds relative to Raw start,
            with both endpoints included.

            Adds the figure to the session report if available; otherwise displays it.
            Returns the figure.
            """
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            mneData = getattr(session, "mne", None)

            if mneData is None:
                raise ValueError("Cannot plot intervals: session.mne is None. Load raw data first.")

            raw = getattr(mneData, "raw", None)

            if raw is None:
                raise ValueError("Cannot plot intervals: session.mne.raw is None. Load raw data first.")

            sfreq = float(raw.info["sfreq"])
            sampleDuration = 1.0 / sfreq
            recordingDuration = raw.n_times / sfreq

            # Add one sample to the widths because the endpoints are inclusive.
            runBars = [(start, end - start + sampleDuration) for start, end in runs]
            noiseBars = [(start, end - start + sampleDuration) for start, end in noiseSegments]

            fig, ax = plt.subplots(figsize=(18, 4))

            ax.broken_barh(runBars, (1.1, 0.6), facecolors="forestgreen", alpha=0.8)
            ax.broken_barh(noiseBars, (0.1, 0.6), facecolors="purple", alpha=0.8)

            for runIndex, (start, end) in enumerate(runs, start=1):
                ax.text((start + end + sampleDuration) / 2, 1.4, str(runIndex), ha="center", va="center", color="white", fontsize=9)

            ax.set_yticks([0.4, 1.4])
            ax.set_yticklabels(["Noise", "Runs"])
            ax.set_ylim(-0.2, 2.0)
            ax.set_xlim(0, recordingDuration)
            ax.set_xlabel("Time from recording start (s)")
            ax.set_title("Proposed runs and noise segments")
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=0.3)

            runPatch = Patch(facecolor="forestgreen", alpha=0.8, label=f"Runs ({len(runs)})")
            noisePatch = Patch(facecolor="purple", alpha=0.8, label=f"Noise segments ({len(noiseSegments)})")
            ax.legend(handles=[runPatch, noisePatch], loc="upper right")

            fig.tight_layout()

            if getattr(mneData, "report", None) is not None:
                mneData.report.add_figure(fig=fig, title="Proposed runs and noise segments")
                plt.show()
                plt.close(fig)
            else:
                plt.show()
                
            return fig
        
        def selectNoiseSegmentsForRuns(self, runs, noiseSegments):
            """Select one noise segment per run, returning an aligned list of tuples or None."""
            selectedNoiseSegments = []

            for runIndex, (runStart, runEnd) in enumerate(runs, start=1):
                containedSegments = [(start, end) for start, end in noiseSegments if runStart <= start and end <= runEnd]

                if containedSegments:
                    selectedSegment = containedSegments[0]
                    reason = "selected an enclosed noise segment" if len(containedSegments) == 1 else f"selected the first of {len(containedSegments)} enclosed noise segments"
                elif noiseSegments:
                    selectedSegment = min(noiseSegments, key=lambda segment: max(runStart - segment[1], segment[0] - runEnd, 0.0))
                    gap = max(runStart - selectedSegment[1], selectedSegment[0] - runEnd, 0.0)
                    reason = f"no enclosed noise segment; selected the closest segment (gap {gap:g} s)"
                else:
                    selectedSegment = None
                    reason = "no noise segments available; selected None"

                selectedNoiseSegments.append(selectedSegment)

                segmentText = "None" if selectedSegment is None else f"({selectedSegment[0]:g}, {selectedSegment[1]:g}) s"
                pglMessages.message(f"Run {runIndex} ({runStart:g}, {runEnd:g}) s: {reason}; noise = {segmentText}.")

            return selectedNoiseSegments

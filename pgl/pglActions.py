################################################################
#   filename: pglActions.py
#    purpose: actions
#         by: JLG
#       date: Sept 12, 2026
################################################################

from .pglMessages import pglMessages
from .pglActionProgress import actionProgress
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
from traitlets import Any, HasTraits, Enum, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
import pandas as pd
from .pglExperiment import pglEventTrial, pglEventSegment
import matplotlib
from tqdm.auto import tqdm
from .pglSettings import pglImportSettings
from .pglPipeline import pglActionCollection

#################################
# Collection of predefined actions
#################################
class pglActions(pglActionCollection):
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # loadSession
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class loadSession(pglAction):
        # settings
        selectedPaths = List(Unicode(), help="Paths of runs selected for loading")
        filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// which can be set if the files are not local")
        sortByAcquisitionTime = Bool(True, help="Load runs in acquisition-time order, oldest first. If False, preserve the chooser's order.")
        
        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglImportSettings=None, settingsName: str=None, experimentName: str=None, subjectID: str=None, sessionName: str=None, runName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None, sortByAcquisitionTime: bool=True) -> None:
            """Choose valid runs and optionally sort oldest first.

            Uses startDateTime when every selected run has a valid,
            timezone-aware timestamp. Otherwise uses startTime for the
            entire selection, assuming those values share a comparable clock.

            Equal timestamps preserve chooser order.
            sortByAcquisitionTime=False preserves chooser order without
            reading acquisition timestamps.
            """
            import json
            import math
            import posixpath
            from datetime import datetime, timezone
            from .pglPipeline import pglActionStatus

            # Clear previous configuration if this action instance is reused.
            self.status = pglActionStatus.INITIALIZED
            self.selectedPaths = []
            self.filesystemPrefix = ""
            self.selectedFilesystem = None

            if type(sortByAcquisitionTime) is not bool:
                raise ValueError("sortByAcquisitionTime must be True or False.")

            self.sortByAcquisitionTime = sortByAcquisitionTime

            filesystem, runList, filesystemPrefix = pglChoose.getSessionRuns(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath)

            if filesystem is None:
                return

            requiredFiles = ("experimentSettings.json", "settings.json", "data.json", "state.json", "pgl.json")
            validPaths = []

            for runPath in runList:
                try:
                    if not filesystem.isdir(runPath):
                        pglMessages.warning(f"Skipping invalid run {runPath!r}: directory does not exist or is not a directory.", level=1)
                        continue

                    missingFiles = [filename for filename in requiredFiles if not filesystem.isfile(posixpath.join(runPath, filename))]

                    if missingFiles:
                        pglMessages.warning(f"Skipping incomplete run {runPath!r}: missing required files: {', '.join(missingFiles)}.", level=1)
                        continue

                except OSError as error:
                    pglMessages.warning(f"Skipping inaccessible run {runPath!r}: {error}", level=1)
                    continue

                validPaths.append(runPath)

            if not validPaths:
                pglMessages.warning("No valid run directories remain; session was not loaded.", level=1)
                return

            if self.sortByAcquisitionTime:
                runTimes = []

                for runPath in validPaths:
                    dataFile = posixpath.join(runPath, "data.json")

                    try:
                        # Read JSON without reconstructing experiment, event, task, or display objects.
                        with filesystem.open(dataFile, "rt", encoding="utf-8") as source:
                            document = json.load(source)

                        if not isinstance(document, dict):
                            raise ValueError("data.json must contain a JSON object.")

                    except (OSError, ValueError) as error:
                        pglMessages.warning(f"Cannot read acquisition metadata for {runPath!r}: {error} Session loading was not configured.", level=1)
                        return

                    acquiredAt = None
                    value = document.get("startDateTime")

                    if isinstance(value, str) and value.strip():
                        try:
                            parsed = datetime.fromisoformat(value.strip())

                            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                                acquiredAt = parsed.astimezone(timezone.utc)

                        except (ValueError, OverflowError):
                            pass

                    runTimes.append({"path": runPath, "startDateTime": acquiredAt, "startTime": document.get("startTime")})

                if all(record["startDateTime"] is not None for record in runTimes):
                    sortField = "startDateTime"
                else:
                    # Never mix calendar timestamps and getSecs() values within the same sort.
                    sortField = "startTime"

                    for record in runTimes:
                        value = record["startTime"]

                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            pglMessages.warning(f"Cannot order run {record['path']!r}: fallback startTime is missing or invalid. Session loading was not configured. Use sortByAcquisitionTime=False to preserve chooser order.", level=1)
                            return

                    pglMessages.warning("Some runs lack usable startDateTime values; sorting all selected runs by startTime instead. This assumes their startTime values share a comparable clock.", level=0)

                # Stable sorting preserves chooser order for equal times.
                runTimes.sort(key=lambda record: record[sortField])
                validPaths = [record["path"] for record in runTimes]

            # Record the actual loading order.
            # Record the actual loading order and source filesystem.
            self.selectedPaths = validPaths
            self.selectedFilesystem = filesystem
            self.filesystemPrefix = filesystemPrefix or ""
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session=None):
            """Load behavioral runs into a new or existing session.

            Args:
                session (pglSession or None): Existing session to append to.
                    If None, create and return a new session.

            Returns:
                pglSession: The new session or the same supplied session.

            Existing runs retain their order. Incoming runs are appended in
            configured order. Repeated loading can append duplicate runs.

            MRI, MNE, checkpoint information, and other existing session
            contents are preserved.
            """
            from .pglSession import pglSession

            if session is not None and not isinstance(session, pglSession):
                raise TypeError("session must be a pglSession or None.")

            if not self.selectedPaths:
                raise ValueError("No runs selected. Configure loadSession before running it.")

            if self.selectedFilesystem is None:
                raise ValueError("No source filesystem configured. Configure loadSession before running it.")

            # Construct all incoming run objects before changing an existing
            # session. Run contents remain lazy, as in normal session loading.
            incomingSession = pglSession(
                filesystem=self.selectedFilesystem,
                filesystemPrefix=self.filesystemPrefix,
                runList=self.selectedPaths
            )

            if session is None:
                return incomingSession

            # Each incoming run retains its own source filesystem.
            # Do not replace the existing session's filesystem or modalities.
            # Assign a new list so trait observers receive the change.
            session.runs = list(session.runs or []) + list(incomingSession.runs or [])

            return session
        
    ################################################################
    # makeStimulusMovie
    ################################################################
    class makeStimulusMovie(pglAction):
        """Generate and attach one stimulus-preview movie per selected run.

        None runIndices selects every behavioral run. Explicit indices are
        zero-based and processed in the supplied order.

        Movies are timestamp-sampled visual previews, not lossless archives
        or independent verification of physical display visibility.

        Existing attachment records and their files remain unchanged until
        every selected movie has been generated successfully.

        Does not open a movie or save a database checkpoint.
        """

        runIndices = List(Int(), default_value=None, allow_none=True, help="Zero-based run indices; None selects all runs.")
        fps = Float(default_value=60.0, allow_none=True, help="Movie sampling rate; None infers the recorded rate.")
        sampling = Enum(["nearest", "previous"], default_value="nearest", help="Recorded presentation timestamp sampling policy.")
        allowAmbiguous = Bool(False, help="Explicitly accept excluded untimed frames and last-flush selection at duplicate timestamps.")
        overwrite = Bool(False, help="Replace existing stimulus-movie attachments, without overwriting their original files.")
        cacheDirectory = Unicode(default_value=None, allow_none=True, help="Local cache override; None uses default database settings.cacheDir.")

        ################################################################
        # Configure generation settings
        ################################################################
        def configure(self, **kwargs):
            self.configureTraits(**kwargs)
            self._validateOptions()
            super().configure()

        def _validateOptions(self):
            """Validate again at execution because traits can be edited later."""
            import math

            if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
                raise ValueError("fps must be positive and finite, or None.")

            if self.runIndices is not None:
                if not self.runIndices: raise ValueError("runIndices must not be empty; use None for all runs.")
                if any(type(index) is not int or index < 0 for index in self.runIndices):
                    raise ValueError("runIndices must contain nonnegative integer indices.")
                if len(set(self.runIndices)) != len(self.runIndices):
                    raise ValueError("runIndices must not contain duplicates.")

            if self.cacheDirectory is not None and not self.cacheDirectory.strip():
                raise ValueError("cacheDirectory must be a nonblank local path or None.")

        ################################################################
        # Resolve the current computer's cache
        ################################################################
        def _cacheRoot(self):
            from pathlib import Path
            from .pglDatabase import pglDatabase

            directory = self.cacheDirectory

            if directory is None:
                settings = pglDatabase.getSettings()
                if settings is None: raise RuntimeError("Database settings are unavailable.")
                directory = settings.cacheDir

            if directory is None or not str(directory).strip():
                raise ValueError("No local cache directory is configured.")
            if "://" in str(directory):
                raise ValueError("Stimulus movie generation requires a local cache directory.")

            return Path(directory).expanduser().resolve() / "stimulusMovies"

        ################################################################
        # Construct a closed renderer; recordingMovie owns open/close
        ################################################################
        @staticmethod
        def _createRenderer():
            from pgl import pgl
            return pgl()

        ################################################################
        # Materialize one recording through its existing filesystem
        ################################################################
        @staticmethod
        def _downloadRecording(run, sourcePath, destination, label):
            """Stream into private local staging before opening the HDF5 file.

            Uses the supplied run filesystem, retaining any verification
            implemented by that filesystem. Does not bypass it with a
            guessed local storage path.
            """
            from .pglProgress import pglProgress

            details = run.filesystem.info(sourcePath)
            size = details.get("size")
            totalBytes = int(size) if size is not None and size > 0 else None
            copiedBytes = 0

            with pglProgress(label) as progress:
                progress(0, "Downloading recording")

                with run.filesystem.open(sourcePath, "rb") as source, destination.open("xb") as target:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block: break
                        if target.write(block) != len(block):
                            raise OSError(f"Incomplete recording write: {destination}")

                        copiedBytes += len(block)
                        fraction = None if totalBytes is None else min(copiedBytes / totalBytes, 0.99)
                        progress(fraction, f"Downloading recording — {copiedBytes / 1024**2:.1f} MiB")

                # Finish after both streams close, including any source-side
                # verification performed on close.
                if totalBytes is not None and copiedBytes != totalBytes:
                    raise ValueError(f"Recording size changed during download: {sourcePath}")

                progress.finish("Recording downloaded")

        ################################################################
        # Generate movies, then attach completed outputs
        ################################################################
        def _run(self, session: pglSession) -> pglSession:
            import posixpath
            import shutil
            import tempfile
            from copy import deepcopy
            from pathlib import Path
            from .pglRecordingFile import pglRecordingFile

            self._validateOptions()

            if not isinstance(session, pglSession):
                self.setError("makeStimulusMovie requires a pglSession.")
                return None

            runs = list(session.runs or [])
            if not runs:
                self.setError("Session has no behavioral runs.")
                return None

            indices = list(range(len(runs))) if self.runIndices is None else list(self.runIndices)
            if any(index >= len(runs) for index in indices):
                self.setError(f"runIndices is outside this session's {len(runs)} runs.")
                return None

            # Preflight every selection before downloading or encoding.
            # Missing source files are not silently skipped or replaced by
            # recordings guessed from another checkpoint.
            jobs = []

            for runIndex in indices:
                run = runs[runIndex]
                name = f"stimulusMovie.run{runIndex:06d}"

                if name in session.derivedFiles and not self.overwrite:
                    self.setError(f"{name!r} is already attached; use overwrite=True to replace it.")
                    return None

                if run.filesystem is None or run.fullDataPath is None:
                    self.setError(f"Run {runIndex} has no recording filesystem source.")
                    return None

                sourcePath = posixpath.join(run.fullDataPath, "recording.h5")

                if not run.filesystem.isfile(sourcePath):
                    self.setError(
                        f"Run {runIndex} does not expose recording.h5 at {sourcePath!r}. "
                        "Its acquisition recording must be made available before movie generation."
                    )
                    return None

                jobs.append({
                    "runIndex": runIndex,
                    "run": run,
                    "name": name,
                    "sourcePath": sourcePath,
                })

            # Snapshot metadata, not payloads. Concurrent session mutation
            # remains unsupported, but detectable changes are not overwritten.
            originalAttachments = deepcopy(session.derivedFiles)
            cacheRoot = self._cacheRoot()
            cacheRoot.mkdir(parents=True, exist_ok=True)
            outputDirectory = Path(tempfile.mkdtemp(prefix="generation-", dir=str(cacheRoot)))
            registrationStarted = False

            try:
                for position, job in enumerate(jobs):
                    runIndex = job["runIndex"]
                    label = f"Stimulus movie {position + 1}/{len(jobs)} — run {runIndex}"
                    pglMessages.message(label)

                    # Keep at most one loaded command recording at a time.
                    # Successful movies stay in outputDirectory; downloaded
                    # source recordings are removed after each run.
                    with tempfile.TemporaryDirectory(prefix=".recording-", dir=str(outputDirectory)) as directory:
                        localRecording = Path(directory) / "recording.h5"
                        self._downloadRecording(job["run"], job["sourcePath"], localRecording, label)
                        recording = pglRecordingFile.load(localRecording)

                        try:
                            renderer = self._createRenderer()
                            outputPath = outputDirectory / f"stimulusMovie.run{runIndex:06d}.mp4"

                            # Always create a new physical file. overwrite
                            # controls attachment replacement, not modification
                            # of an existing attachment's payload.
                            renderer.recordingMovie(
                                recording,
                                outputPath,
                                fps=self.fps,
                                sampling=self.sampling,
                                allowAmbiguous=self.allowAmbiguous,
                                overwrite=False,
                            )

                            if renderer.isOpen() or renderer.renderTarget is not None:
                                raise RuntimeError(f"Run {runIndex}: movie export left renderer resources active.")
                            if not outputPath.is_file() or outputPath.stat().st_size == 0:
                                raise RuntimeError(f"Run {runIndex}: movie export produced no nonempty MP4.")

                            job["outputPath"] = outputPath

                        finally:
                            # recordingMovie owns renderer cleanup. Do not send
                            # additional cleanup commands after a failed exchange.
                            del recording

                if session.derivedFiles != originalAttachments:
                    raise RuntimeError("Session attachments changed during movie generation.")

                # All encoders, renderers, and temporary recording directories
                # have finished before installing any attachment.
                registrationStarted = True

                for job in jobs:
                    session.addDerivedFile(
                        job["name"],
                        job["outputPath"],
                        runIndex=job["runIndex"],
                        mediaType="video/mp4",
                        overwrite=self.overwrite,
                    )

            except BaseException as error:
                # Restore metadata if hashing/registration fails partway.
                # This is not a transaction for arbitrary trait-observer side
                # effects; concurrent mutation is not supported.
                safeToRemove = True

                if registrationStarted:
                    try:
                        session.derivedFiles = originalAttachments
                    except BaseException as rollbackError:
                        safeToRemove = False
                        if hasattr(error, "add_note"):
                            error.add_note(f"Attachment rollback failed; generated files retained: {rollbackError}")

                if safeToRemove:
                    try:
                        shutil.rmtree(outputDirectory)
                    except BaseException as cleanupError:
                        if hasattr(error, "add_note"):
                            error.add_note(f"Generated-file cleanup failed: {cleanupError}")

                raise

            # Successful cache files remain available for display and saving.
            # Replaced attachment files are not deleted: other sessions or
            # applications may still be using them.
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
        
    ################################################################
    # Load a deferred Flywheel MRI session
    ################################################################
    class loadFlywheel(pglAction):
        """Choose Flywheel MRI files using a saved import-settings profile.

        By default, dicom=True selects DICOM files or archives, downloads
        them during execution, and converts each selected source with
        dcm2niix into NIfTI images and JSON sidecars.
        Failed conversions fall back to a matching NIfTI from the same
        Flywheel acquisition. Sources that cannot be loaded are skipped.
        Set dicom=False to select NIfTI files and attach their pinned sources
        without downloading image payloads.

        Use saveRawSession() later to stream acquisition bytes into storage.
        
        One selected Flywheel session returns one PGL session. Multiple
        selected Flywheel sessions return a list of separate PGL sessions in
        selection order. Supplying an existing session requires a single
        Flywheel session and an empty MRI attachment.
        
        History stores the settings name, pinned metadata, and labels.
        It never stores credentials, the SDK client, or caller-local cache
        configuration. Source metadata may identify subjects.

        Restored/copied actions require runtime access supplied again through
        run(flywheel=..., cacheDirectory=...). They do not reconnect silently.
        Analysis replay should normally begin at the saved raw checkpoint,
        excluding this ingestion action through the recorded history boundary.
        """

        settingsName = Unicode("", help="Resolved import-settings profile used to select Flywheel scans")
        selectedScans = List(Dict(), help="Ordered scan names and pinned source metadata")
        experimentName = Unicode("", help="Explicit PGL experiment label")
        subjectID = Unicode("", help="Session subject label")
        dicom = Bool(True, help="Select DICOM files and convert them to NIfTI")
        showScanCounts = Bool(False, help="Show matching file counts in the session chooser. Fetches scan metadata for every session and can be slow.")

        # Runtime resources must not enter action history or pipeline recipes.
        _client = Any(default_value=None, allow_none=True).tag(serialize=False)
        _cacheDirectory = Unicode("").tag(serialize=False)

        # ====================================================================
        def configure(self, settingsName=None, *, experimentName=None, subjectID=None, showScanCounts=False, dicom=True, projectName=None, verbose=False):
            """Connect and choose scans using a saved import-settings profile.

            settingsName=None selects the default import-settings profile.
            The profile supplies the keychain identity and local cacheDir.
            Successful configuration records the resolved profile name.

            showScanCounts=False keeps scan discovery lazy while browsing.
            True fetches acquisition/file metadata for every session in a
            browsed project to display matching file counts. This can be slow.
            By default, dicom=True selects DICOM files/archives for conversion
            during run. Set dicom=False to select existing NIfTIs directly.
            projectName selects the initial project by label or ID.
            verbose=True enables Flywheel project and session browsing messages.

            Cancel or invalid input leaves this action unconfigured.
            Configuration downloads no image or DICOM payload and creates no
            cache directory.

            experimentName is explicit; a Flywheel project is not automatically
            assumed to be a PGL experiment. Unset labels remain unset and are
            subject to the ordinary saveRawSession() metadata checks.
            """
            import os
            from pathlib import Path

            from .pglFlywheel import pglFlywheel
            from .pglMRI import pglMRIScan
            from .pglPipeline import pglActionStatus
            from .pglSession import pglSession
            from .pglSettings import pglSettingsManager

            # Clear earlier configuration and runtime access before starting.
            # Cancellation must not leave an earlier selection ready to run.
            self.status = pglActionStatus.INITIALIZED
            self.error = None
            self.settingsName = ""
            self.selectedScans = []
            self.experimentName = ""
            self.subjectID = ""
            self.dicom = True
            self.showScanCounts = False
            self._client = None
            self._cacheDirectory = ""

            # Validate the browsing option before loading settings or connecting.
            if type(showScanCounts) is not bool:
                pglMessages.warning("Flywheel loading was not configured: showScanCounts must be Boolean.", level=1)
                return
            self.showScanCounts = showScanCounts
            if type(dicom) is not bool:
                pglMessages.warning("Flywheel loading was not configured: set dicom to True or False.")
                return
            self.dicom = dicom
            if type(verbose) is not bool:
                pglMessages.warning("Flywheel loading was not configured: set verbose to True or False.")
                return
            if projectName is not None and not isinstance(projectName, str):
                pglMessages.warning("Flywheel loading was not configured: set projectName to a Flywheel project label or ID, or None.")
                return

            # Validate the requested profile name before loading settings.
            if settingsName is not None:
                if not isinstance(settingsName, str) or not settingsName.strip():
                    pglMessages.warning("Flywheel loading was not configured: settingsName must be a nonempty string or None.", level=0)
                    return

            # Resolve the named profile, or the default when no name is given.
            importSettings = pglSettingsManager.getImportSettings(settingsName=settingsName)
            if importSettings is None: return

            # Validate the profile's cache path and explicit session labels
            # before connecting or opening the selection dialog.
            try:
                cacheText = os.fspath(importSettings.cacheDir)
                if not isinstance(cacheText, str) or not cacheText.strip() or "://" in cacheText or "::" in cacheText or any(character in cacheText for character in ("\x00", "\n", "\r")):
                    raise ValueError("Import settings cacheDir must be an ordinary local path.")

                cachePath = str(Path(cacheText).expanduser().absolute())
                experimentLabel = pglSession._normalizeSessionLabel(experimentName, "experimentName")
                subjectLabel = pglSession._normalizeSessionLabel(subjectID, "subjectID")

            except (TypeError, ValueError) as error:
                pglMessages.warning(f"Flywheel loading was not configured: {error}", level=0)
                return

            # Retrieve the keychain credential using exactly this profile.
            flywheel = pglFlywheel(settings=importSettings)
            if flywheel.fw is None: return

            # Browse and select scans. Keep an empty accepted selection
            # distinct from cancellation for the execution warning.
            selectedScans = flywheel.choose(showScanCounts=self.showScanCounts, dicom=self.dicom, projectName=projectName, verbose=verbose)
            if selectedScans is None: return
            if not selectedScans:
                self.error = ValueError("no scans were selected")
                return
            multipleSessions = len({(scan["projectID"], scan["sessionID"]) for scan in selectedScans}) > 1

            # Pin the selected file versions from each chosen session.
            # This reads metadata only, without creating the cache directory.
            if multipleSessions:
                mri = flywheel._pinSelectedMRI(selectedScans, cacheDirectory=cachePath, dicom=self.dicom)
            else:
                mri = flywheel.toMRI(cacheDirectory=cachePath, dicom=self.dicom)
            if mri is None: return

            # Use the Flywheel subject label unless explicitly overridden.
            if subjectID is None and not multipleSessions:
                subjects = {
                    pglSession._normalizeSessionLabel(scan.sourceMetadata["selection"]["subjectID"], "Flywheel subjectID")
                    for scan in mri.scans
                }
                if len(subjects) != 1:
                    pglMessages.warning("Flywheel loading was not configured: selected scans do not share one subject label.", level=0)
                    return
                subjectLabel = next(iter(subjects))

            # Retain portable metadata, not the temporary MRI collection.
            # Every execution constructs independent working scans.
            records = [{"name": scan.name, "sourceMetadata": pglMRIScan._copyMetadata(scan.sourceMetadata)} for scan in mri.scans]

            # Commit the resolved profile name and accepted scan configuration.
            self.settingsName = importSettings.name
            self.selectedScans = records
            self.experimentName = experimentLabel or ""
            self.subjectID = subjectLabel or ""

            # Keep the connected client and resolved cache path runtime-only.
            # Neither resource enters action history or pipeline recipes.
            self._client = flywheel.fw
            self._cacheDirectory = cachePath
            super().configure()

        # ====================================================================
        def run(self, *args, **kwargs):
            """Import one session or run separate, recorded imports for a batch."""
            if not self.isConfigured():
                if self.error is None: return super().run(*args, **kwargs)
                pglMessages.warning(f"Action {self.name} is not yet configured: {self.error}.")
                return None
            if getattr(self, "_executionInProgress", False): return super().run(*args, **kwargs)

            groups = {}
            try:
                for record in self.selectedScans:
                    selection = record["sourceMetadata"]["selection"]
                    groups.setdefault((selection["projectID"], selection["sessionID"]), []).append(record)
            except (TypeError, KeyError):
                return super().run(*args, **kwargs)
            if len(groups) <= 1: return super().run(*args, **kwargs)
            return self._runBatch(groups, args, kwargs)

        # ====================================================================
        @actionProgress
        def _runBatch(self, groups, args, kwargs):
            """Preserve independent action history and subject identity per session."""
            from .pglActionProgress import currentProgress, progressCall
            from .pglPipeline import pglActionStatus

            selections = [record["sourceMetadata"]["selection"] for group in groups.values() for record in group]
            subjects = {}
            try:
                for key, group in groups.items():
                    labels = {
                        pglSession._normalizeSessionLabel(record["sourceMetadata"]["selection"].get("subjectID"), "Flywheel subjectID")
                        for record in group
                    }
                    if len(labels) != 1: raise ValueError(f"Flywheel session {key[1]!r} has inconsistent subject labels. Refresh the selection and retry.")
                    subjects[key] = next(iter(labels))
                if self.subjectID and len({label for label in subjects.values() if label}) > 1:
                    raise ValueError("Omit subjectID= to preserve each subject's Flywheel ID when importing multiple subjects.")
            except (TypeError, ValueError) as error:
                self.setError(str(error))
                return None

            self._printMultiSessionSelection(selections)
            if (args and args[0] is not None) or kwargs.get("session") is not None:
                self.setError("Omit session= to import multiple Flywheel sessions into separate PGL sessions.")
                return None

            progress = currentProgress()
            totalScans = len(selections)
            completedScans = 0
            results = []
            self.status = pglActionStatus.RUNNING
            self.error = None
            self._executionInProgress = True
            try:
                for index, (key, group) in enumerate(groups.items(), start=1):
                    child = type(self)()
                    child.configureTraits(
                        settingsName=self.settingsName,
                        selectedScans=group,
                        experimentName=self.experimentName,
                        subjectID=self.subjectID or subjects[key] or "",
                        dicom=self.dicom,
                        showScanCounts=self.showScanCounts,
                    )
                    child.status = pglActionStatus.CONFIGURED
                    child.name = self.name
                    child.version = self.version
                    child.keepIntermediates = self.keepIntermediates
                    child._client = self._client
                    child._cacheDirectory = self._cacheDirectory
                    child.cacheDirectory = getattr(self, "cacheDirectory", None)
                    end = completedScans + len(group)
                    description = f"Flywheel session {index}/{len(groups)}"
                    try:
                        result = progressCall(progress, completedScans / totalScans, end / totalScans, description, child.run, *args, **kwargs)
                    except Exception as error:
                        pglMessages.warning(f"Could not import Flywheel session {key[1]!r}: {error}. Check the connection and cache settings before retrying.")
                        result = None
                    finally:
                        self.actionHistory.extend(child.actionHistory)
                    if result is not None: results.append(result)
                    completedScans = end
            except (KeyboardInterrupt, SystemExit) as error:
                self.status = pglActionStatus.FAILED
                self.error = RuntimeError(f"{type(error).__name__}: Flywheel import was interrupted.")
                raise
            finally:
                self._executionInProgress = False

            if not results:
                self.setError("No Flywheel sessions were imported. Check the session errors and retry.")
                return None
            self.status = pglActionStatus.SUCCESS
            pglMessages.print(f"Returned {len(results)} of {len(groups)} selected Flywheel sessions.")
            return results

        # ====================================================================
        def _run(self, session=None, *, flywheel=None, cacheDirectory=None):
            """Attach pinned, unopened MRI sources to a new or existing session.

            None session creates a new session with the configured labels.
            An existing session retains its labels and all other contents.
            Existing MRI data is never silently replaced or merged.

            Runtime overrides allow execution of a restored/copied action
            without repeating interactive selection or repinning files.

            NIfTI mode attaches the pinned source without downloading it.
            DICOM mode streams the pinned source during this call, verifies
            its recorded size/checksum, and converts it with dcm2niix.
            If a source fails, the action tries an existing NIfTI from its
            acquisition and skips the source if neither route succeeds.

            Attaching to an existing session does not establish temporal
            alignment between MRI scans and behavioral runs.
            """
            import os
            from pathlib import Path

            from .pglActionProgress import currentProgress
            from .pglFlywheel import pglFlywheel, pglFlywheelMRIBacking, pglFlywheelSession
            from .pglMRI import pglMRI, pglMRIScan
            from .pglSession import pglSession

            progress = currentProgress()
            if progress is not None: progress(0, "Preparing Flywheel scan sources")

            if session is not None:
                if not isinstance(session, pglSession):
                    self.setError("session must be a pglSession or None.")
                    return None

                if session.mri is not None:
                    self.setError("The supplied session already has MRI data. loadFlywheel will not overwrite or merge it.")
                    return None

            # Use the configured runtime client unless explicitly overridden.
            client = self._client
            if flywheel is not None:
                if not isinstance(flywheel, pglFlywheel) or flywheel.fw is None:
                    self.setError("Supply a working pglFlywheel instance.")
                    return None
                client = flywheel.fw

            # Restored/copied actions must receive runtime access explicitly.
            # The recorded settings name does not trigger an automatic login.
            if client is None:
                self.setError("Flywheel runtime access is missing. Create a connection with pglFlywheel(settingsName=action.settingsName), then use action.run(session=session, flywheel=connection, cacheDirectory=...).")
                return None

            try:
                # Resolve and validate the runtime cache path without creating it.
                cacheValue = self._cacheDirectory if cacheDirectory is None else cacheDirectory
                cacheText = os.fspath(cacheValue)
                if not isinstance(cacheText, str) or not cacheText.strip() or "://" in cacheText or "::" in cacheText or any(character in cacheText for character in ("\x00", "\n", "\r")):
                    raise ValueError("A local cacheDirectory is required.")

                cachePath = str(Path(cacheText).expanduser().absolute())
                if not self.selectedScans:
                    raise ValueError("No pinned scans are configured.")

                # Work with independent metadata copies on every execution.
                records = [pglMRIScan._copyMetadata(record) for record in self.selectedScans]

                # Recheck persisted configuration, including recipes edited
                # through configureTraits() rather than configure().
                sessionKeys = set()
                for record in records:
                    if set(record) != {"name", "sourceMetadata"}:
                        raise ValueError("Invalid configured scan record.")

                    provenance = record["sourceMetadata"]
                    selection = provenance["selection"]
                    fileMetadata = provenance["file"]
                    sessionKeys.add((selection["projectID"], selection["sessionID"]))

                    if fileMetadata["name"] != selection["fileName"]:
                        raise ValueError("Pinned filename differs from selection metadata.")
                    if self.dicom:
                        fileType = selection.get("fileType", "")
                        isDicomType = (
                            isinstance(fileType, str)
                            and fileType.casefold() in {"dicom", "dicom-zip", "dcm"}
                        )
                        if (
                            fileType == "nifti"
                            or (
                                not isDicomType
                                and not pglFlywheelMRIBacking.isDicomFileName(selection["fileName"])
                            )
                        ):
                            raise ValueError("A non-DICOM source was configured for DICOM conversion.")

                # Each action execution attaches MRI from one source session.
                if len(sessionKeys) != 1:
                    raise ValueError("Use run() to import multiple Flywheel sessions into separate PGL sessions.")

                scanRange = progress.section(0.02, 0.95) if progress is not None else None
                if self.dicom:
                    provenance = records[0]["sourceMetadata"]
                    selection = provenance["selection"]
                    sessionMetadata = provenance.get("session", {})
                    projectName = selection.get("projectName") or selection["projectID"]
                    sessionLabel = sessionMetadata.get("label") or selection.get("sessionLabel") or selection.get("sessionName") or "unknown"
                    subjectID = selection.get("subjectID") or self.subjectID or "unknown"
                    savedBounds = selection.get("acquisitionTimeBounds")
                    bounds = pglFlywheelSession._acquisitionBounds(savedBounds) if isinstance(savedBounds, list) else ()
                    selectedOnly = not bounds
                    if selectedOnly:
                        bounds = pglFlywheelSession._acquisitionBounds(
                            record["sourceMetadata"].get("acquisition", {}).get("timestamp") for record in records
                        )
                    pglMessages.printHeader("Flywheel DICOM to NIfTI")
                    pglMessages.print(f"Project: {projectName}")
                    pglMessages.print(f"Session label: {sessionLabel}")
                    pglMessages.print(f"Session ID: {selection['sessionID']}")
                    pglMessages.print(f"Subject ID: {subjectID}")
                    acquired = pglFlywheelSession._localizeTimestamp(sessionMetadata.get("timestamp"), sessionMetadata.get("timezone"), selection["sessionID"])
                    acquisitionTime = pglFlywheelSession._formatAcquisitionTime(acquired) or selection.get("acquisitionTimeText")
                    if acquisitionTime: pglMessages.print(f"Acquisition time: {acquisitionTime}")
                    duration = self._formatSessionDuration(bounds, selectedOnly=selectedOnly)
                    pglMessages.print(f"Session duration: {duration}")
                    pglMessages.print(f"Selected DICOM files or archives: {len(records)}")
                    pglMessages.print()
                    mri = self._convertDicomSources(records, client, cachePath, progress=scanRange)
                    updatedDuration = self._formatSessionDuration(bounds, selectedOnly=selectedOnly, scans=mri.scans)
                    if updatedDuration != duration: pglMessages.print(f"Session duration: {updatedDuration}")
                else:
                    scans = []
                    for index, record in enumerate(records):
                        provenance = record["sourceMetadata"]
                        selection = provenance["selection"]
                        fileMetadata = provenance["file"]
                        if scanRange is not None: scanRange(index / len(records), f"Preparing NIfTI {index + 1}/{len(records)} — {selection['fileName']}")

                        fileType = selection.get("fileType")
                        if (
                            fileType != "nifti"
                            and not (
                                fileType is None
                                and isinstance(selection.get("fileName"), str)
                                and selection["fileName"].lower().endswith((".nii", ".nii.gz"))
                            )
                        ):
                            raise ValueError("A non-NIfTI source was configured for NIfTI loading.")

                        # Reconstruct the recorded source version without
                        # requesting metadata or downloading NIfTI payloads.
                        backing = pglFlywheelMRIBacking(
                            client,
                            selection["acquisitionID"],
                            selection["fileName"],
                            version=fileMetadata["version"],
                            sizeBytes=fileMetadata["size"],
                            sourceHash=fileMetadata["hash"],
                        )

                        scan = pglMRIScan(name=record["name"], sourceMetadata=provenance)
                        if scan.attachBacking(backing, cacheDirectory=cachePath) is None:
                            raise ValueError("Could not attach the configured MRI source.")
                        scans.append(scan)

                    mri = pglMRI()
                    mri.scans = scans

                # Create a new acquisition session with the configured labels.
                # Existing session labels remain authoritative, including labels
                # derived from behavioral runs.
                creatingSession = session is None
                if creatingSession:
                    session = pglSession()
                    session.experimentName = self.experimentName
                    session.subjectID = self.subjectID

            except (TypeError, ValueError, KeyError) as error:
                self.setError(f"Flywheel session was not loaded: {error}")
                return None

            projectID, flywheelSessionID = next(iter(sessionKeys))
            if not mri.scans:
                if progress is not None: progress(0.98, "No Flywheel scans loaded")
                pglMessages.print("No Flywheel scans were loaded. The session has no attached MRI data.")
                return session

            if progress is not None: progress(0.98, f"Attaching {len(mri.scans)} MRI scan(s) to the session")
            destination = "a new session" if creatingSession else "the existing session"
            sourceKind = "NIfTI" if self.dicom else "deferred NIfTI"
            pglMessages.message(f"Attaching {len(mri.scans)} {sourceKind} scan(s) from Flywheel project {projectID!r}, session {flywheelSessionID!r}, to {destination}.")

            if not creatingSession:
                pglMessages.message(f"Preserving {len(session.runs or [])} behavioral run(s), existing labels, and other session contents. Scan-to-run alignment is not performed.")

            # Attach the deferred MRI collection and return the same supplied
            # session, or the new session created above.
            session.add(mri)
            return session

        # ====================================================================
        def _printMultiSessionSelection(self, selectedScans):
            """Summarize a selection that spans multiple Flywheel sessions."""
            from .pglFlywheel import pglFlywheelSession

            grouped = {}
            for scan in selectedScans: grouped.setdefault((scan["projectID"], scan["sessionID"]), []).append(scan)
            sessions = [scans[0] for scans in grouped.values()]
            projects = dict.fromkeys(scan.get("projectName") or scan["projectID"] for scan in sessions)
            labels = [scan.get("sessionLabel") or scan.get("sessionName") or "unknown" for scan in sessions]
            subjects = dict.fromkeys(scan.get("subjectID") or "unknown" for scan in sessions)
            durations = []
            for scan in sessions:
                savedBounds = scan.get("acquisitionTimeBounds", [])
                bounds = pglFlywheelSession._acquisitionBounds(savedBounds) if isinstance(savedBounds, list) else ()
                durations.append(self._formatSessionDuration(bounds, selectedOnly=False))

            pglMessages.printHeader("Flywheel selection across sessions")
            pglMessages.print(f"Projects: {', '.join(projects)}")
            pglMessages.print(f"Session labels: {', '.join(labels)}")
            pglMessages.print(f"Session IDs: {', '.join(scan['sessionID'] for scan in sessions)}")
            pglMessages.print(f"Subject IDs: {', '.join(subjects)}")
            pglMessages.print(f"Acquisition times: {', '.join(scan.get('acquisitionTimeText') or 'unknown' for scan in sessions)}")
            pglMessages.print(f"Session durations: {', '.join(durations)}")
            counts = ", ".join(str(len(scans)) for scans in grouped.values())
            fileLabel = "DICOM files or archives" if self.dicom else "NIfTI files"
            pglMessages.print(f"Selected {fileLabel}: {counts} by session ({len(selectedScans)} total)")

        # ====================================================================
        @staticmethod
        def _formatSessionDuration(bounds, *, selectedOnly, scans=()):
            """Describe the recorded acquisition span and any known scan ends."""
            import math
            from datetime import timedelta

            from .pglFlywheel import pglFlywheelSession

            if not bounds: return "unknown"
            firstStart, lastStart = bounds
            latestEnd = lastStart
            haveFinalDuration = False
            for scan in scans:
                duration = scan.sidecar.get("AcquisitionDuration")
                acquisition = scan.sourceMetadata.get("acquisition", {})
                start = pglFlywheelSession._acquisitionBounds([acquisition.get("timestamp")])
                if not start or (start[0].tzinfo is None) != (firstStart.tzinfo is None): continue
                if not firstStart <= start[0] <= lastStart: continue
                try:
                    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0: continue
                    end = start[0] + timedelta(seconds=duration)
                except OverflowError:
                    continue
                latestEnd = max(latestEnd, end)
                if start[0] == lastStart: haveFinalDuration = True

            if latestEnd == firstStart: return "unknown"
            complete = haveFinalDuration and not selectedOnly
            elapsed = (latestEnd - firstStart).total_seconds()
            span = str(timedelta(seconds=round(elapsed) if complete else int(elapsed)))
            scope = "selected acquisitions only, " if selectedOnly else ""
            if complete: return f"about {span} (first acquisition start to final scan end)"
            endpoint = "including available scan durations" if latestEnd > lastStart else "first to last acquisition start"
            return f"at least {span} ({scope}{endpoint})"

        # ====================================================================
        def _convertDicomSources(self, records, client, cachePath, *, progress=None):
            """Convert DICOM sources or download their Flywheel NIfTI fallback."""
            import re
            import shutil
            import stat
            import tempfile
            import zipfile
            from pathlib import Path, PurePosixPath
            from uuid import uuid4

            from .pglExternalTools import pglDcm2niix
            from .pglFlywheel import pglFlywheelMRIBacking
            from .pglMRI import pglMRI, pglMRIScan

            cacheDirectory = Path(cachePath)
            cacheDirectory.mkdir(parents=True, exist_ok=True)
            if progress is not None: progress(0, "Checking dcm2niix availability")
            tool = pglDcm2niix()
            toolAvailable = tool.checkAvailable(cacheDirectory=cachePath)

            conversionRoot = cacheDirectory / "pgl-dicom-conversions"
            conversionRoot.mkdir(parents=True, exist_ok=True)
            publishedDirectory = conversionRoot / uuid4().hex

            def extractArchive(archivePath, destination):
                """Extract a ZIP without allowing paths or links outside it."""
                destination.mkdir(parents=True, exist_ok=False)
                with zipfile.ZipFile(archivePath) as archive:
                    for member in archive.infolist():
                        memberName = member.filename.replace("\\", "/")
                        memberPath = PurePosixPath(memberName)
                        if (
                            memberPath.is_absolute()
                            or any(part in ("..",) for part in memberPath.parts)
                            or (memberPath.parts and ":" in memberPath.parts[0])
                        ):
                            raise ValueError(f"Unsafe path in DICOM archive: {member.filename!r}.")
                        if not memberPath.parts:
                            if member.is_dir():
                                continue
                            raise ValueError(f"Invalid empty path in DICOM archive: {member.filename!r}.")

                        mode = member.external_attr >> 16
                        kind = stat.S_IFMT(mode)
                        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                            raise ValueError(f"Unsupported special file in DICOM archive: {member.filename!r}.")
                        if stat.S_ISLNK(mode):
                            raise ValueError(f"Symbolic links are not allowed in DICOM archives: {member.filename!r}.")

                        target = destination.joinpath(*memberPath.parts)
                        if member.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue

                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member, "r") as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)

            # ================================================================
            def prepareDicomInput(sourcePath, sourceName, inputDirectory, scanProgress):
                """Expand an outer ZIP and any per-series ZIPs into one tree."""
                if sourceName.casefold().endswith(".zip") or zipfile.is_zipfile(sourcePath):
                    if scanProgress is not None: scanProgress(None, f"Extracting DICOM archive — {sourceName}")
                    extractArchive(sourcePath, inputDirectory)
                else:
                    if scanProgress is not None: scanProgress(None, f"Staging DICOM file — {sourceName}")
                    inputDirectory.mkdir(parents=True, exist_ok=False)
                    shutil.copyfile(sourcePath, inputDirectory / Path(sourceName).name)

                for _ in range(8):
                    nestedArchives = sorted(
                        path for path in inputDirectory.rglob("*")
                        if path.is_file() and path.suffix.casefold() == ".zip"
                    )
                    if not nestedArchives:
                        return

                    if scanProgress is not None: scanProgress(None, f"Extracting nested DICOM archives — {sourceName}")
                    for archivePath in nestedArchives:
                        nestedDirectory = archivePath.with_name(archivePath.name + ".contents")
                        extractArchive(archivePath, nestedDirectory)
                        archivePath.unlink()

                if any(
                    path.is_file() and path.suffix.casefold() == ".zip"
                    for path in inputDirectory.rglob("*")
                ):
                    raise ValueError("DICOM archives are nested too deeply to convert safely.")

            def fileStem(filename):
                for suffix in (".nii.gz", ".nii", ".zip", ".dcm", ".dicom", ".ima"):
                    if filename.casefold().endswith(suffix):
                        return filename[:-len(suffix)]
                return filename

            def cleanBaseName(filename):
                cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", fileStem(filename))
                cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
                return cleaned or "dicom"

            # ================================================================
            def download(backing, destination, sizeBytes, scanProgress, description):
                downloaded = 0
                sizeMiB = sizeBytes / (1024 ** 2)
                if scanProgress is not None:
                    scanProgress(None, f"{description} — 0.0/{sizeMiB:,.1f} MiB — {backing.path}")

                with backing.open() as source, destination.open("xb") as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                        downloaded += len(block)
                        if scanProgress is not None:
                            scanProgress(None, f"{description} — {downloaded / (1024 ** 2):,.1f}/{sizeMiB:,.1f} MiB — {backing.path}")

            def loadScan(niftiPath, provenance):
                scan = pglMRIScan(
                    name=niftiPath.name,
                    localPath=str(niftiPath),
                    sourceMetadata=provenance,
                )
                if scan.data is None:
                    raise ValueError(f"Could not open NIfTI header for {scan.name!r}.")
                return scan

            # ================================================================
            def convertSource(backing, provenance, sourceDirectory, outputDirectory, scanProgress):
                sourceName = backing.path
                sourcePath = sourceDirectory / sourceName
                download(backing, sourcePath, provenance["file"]["size"], scanProgress, "Downloading DICOM")

                inputDirectory = sourceDirectory / "dicom"
                prepareDicomInput(sourcePath, sourceName, inputDirectory, scanProgress)

                if scanProgress is not None: scanProgress(None, f"Running dcm2niix — {sourceName}")
                pglMessages.print(f"Converting Flywheel DICOM {sourceName!r} with dcm2niix.")
                result = tool.runCommand(
                    arguments=[
                        "-z", "y",
                        "-b", "y",
                        "-ba", "n",
                        "-w", "1",
                        "-f", f"{cleanBaseName(sourceName)}_%s",
                        "-o", str(outputDirectory),
                        str(inputDirectory),
                    ],
                    cacheDirectory=cachePath,
                )
                if result is None:
                    raise RuntimeError("dcm2niix did not run successfully.")

                if scanProgress is not None: scanProgress(None, f"Validating converted NIfTI — {sourceName}")
                niftiFiles = sorted(
                    path for path in outputDirectory.iterdir()
                    if path.is_file() and path.name.casefold().endswith((".nii", ".nii.gz"))
                )
                if not niftiFiles:
                    raise ValueError("dcm2niix did not produce a NIfTI file.")

                pairedSidecars = set()
                scans = []
                for niftiPath in niftiFiles:
                    sidecarPath = pglMRIScan._sidecarPath(niftiPath)
                    if not sidecarPath.is_file():
                        raise ValueError(f"dcm2niix did not create the JSON sidecar for {niftiPath.name!r}.")

                    scanProvenance = {
                        **provenance,
                        "conversion": {
                            "software": "dcm2niix",
                            "sourceFile": sourceName,
                            "outputFile": niftiPath.name,
                        },
                    }
                    scan = loadScan(niftiPath, scanProvenance)
                    pglMRIScan._writeSidecar(niftiPath, scan.sidecar)
                    scans.append(scan)
                    pairedSidecars.add(sidecarPath.name)

                jsonFiles = {
                    path.name for path in outputDirectory.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".json"
                }
                if jsonFiles != pairedSidecars:
                    raise ValueError("dcm2niix produced unpaired JSON sidecars.")
                return scans

            # ================================================================
            def downloadFallback(provenance, outputDirectory, scanProgress):
                selection = provenance["selection"]
                acquisitionID = selection["acquisitionID"]
                if scanProgress is not None: scanProgress(None, f"Looking for a Flywheel NIfTI fallback — {selection['fileName']}")
                try:
                    acquisition = client.get_acquisition(acquisitionID)
                    files = list(acquisition.files or [])
                except Exception:
                    raise RuntimeError("Could not read Flywheel acquisition files.") from None

                candidates = [
                    file for file in files
                    if isinstance(getattr(file, "name", None), str)
                    and file.name.casefold().endswith((".nii", ".nii.gz"))
                ]
                if not candidates:
                    raise ValueError("No NIfTI file was found in this Flywheel acquisition.")

                sourceStem = fileStem(selection["fileName"]).casefold()
                matching = [
                    file for file in candidates
                    if fileStem(file.name).casefold() == sourceStem
                ]
                candidates = matching or candidates
                if len(candidates) != 1:
                    raise ValueError("Could not identify a unique NIfTI file in this Flywheel acquisition.")

                file = candidates[0]
                fileMetadata = {
                    "name": file.name,
                    "type": "nifti",
                    "version": getattr(file, "version", None),
                    "size": getattr(file, "size", None),
                    "hash": getattr(file, "hash", None),
                }
                backing = pglFlywheelMRIBacking(
                    client,
                    acquisitionID,
                    file.name,
                    version=fileMetadata["version"],
                    sizeBytes=fileMetadata["size"],
                    sourceHash=fileMetadata["hash"],
                )
                acquisitionLabel = provenance.get("acquisition", {}).get("label")
                fallbackProvenance = {
                    **provenance,
                    "selection": {
                        **selection,
                        "name": f"{acquisitionLabel}: {file.name}" if acquisitionLabel else file.name,
                        "fileName": file.name,
                        "fileType": "nifti",
                    },
                    "versionPinnedAt": "loadFlywheel",
                    "file": fileMetadata,
                    "fallback": {"selection": selection, "file": provenance["file"]},
                }
                pglMessages.print(f"Downloading existing Flywheel NIfTI {file.name!r} for {selection['fileName']!r}.")
                niftiPath = outputDirectory / file.name
                download(backing, niftiPath, fileMetadata["size"], scanProgress, "Downloading NIfTI fallback")
                if scanProgress is not None: scanProgress(None, f"Validating NIfTI fallback — {file.name}")
                return loadScan(niftiPath, fallbackProvenance)

            mri = pglMRI()
            scanCount = len(records)
            with tempfile.TemporaryDirectory(prefix=".pgl-dicom-", dir=str(cacheDirectory)) as temporary:
                publishedDirectory.mkdir()
                try:
                    for index, record in enumerate(records):
                        provenance = record["sourceMetadata"]
                        selection = provenance["selection"]
                        fileMetadata = provenance["file"]
                        sourceName = selection["fileName"]
                        if index: pglMessages.printHeader(fillChar="-")
                        # Advance the overall bar by scans processed. The
                        # download status shows bytes transferred per scan.
                        scanProgress = progress.section(index / scanCount, (index + 1) / scanCount, label=f"Scan {index + 1}/{scanCount}") if progress is not None else None
                        if scanProgress is not None: scanProgress(0, f"Preparing DICOM source — {sourceName}")

                        backing = pglFlywheelMRIBacking(
                            client,
                            selection["acquisitionID"],
                            sourceName,
                            version=fileMetadata["version"],
                            sizeBytes=fileMetadata["size"],
                            sourceHash=fileMetadata["hash"],
                            dicom=True,
                        )
                        sourceDirectory = Path(temporary) / f"source-{index:04d}"
                        sourceDirectory.mkdir()
                        outputDirectory = publishedDirectory / f"scan-{index:04d}"
                        outputDirectory.mkdir()
                        try:
                            if not toolAvailable:
                                raise RuntimeError("dcm2niix is unavailable.")
                            scans = convertSource(backing, provenance, sourceDirectory, outputDirectory, scanProgress)
                        except Exception as error:
                            pglMessages.print(
                                f"DICOM import failed for {sourceName!r} ({error}). "
                                "Checking its Flywheel acquisition for an existing NIfTI."
                            )
                            shutil.rmtree(outputDirectory)
                            outputDirectory.mkdir()
                            try:
                                scan = downloadFallback(provenance, outputDirectory, scanProgress)
                            except Exception as fallbackError:
                                shutil.rmtree(outputDirectory)
                                pglMessages.message(f"Skipping Flywheel scan {sourceName!r}: {fallbackError}")
                                if scanProgress is not None: scanProgress(1, f"Skipped — {sourceName}")
                                continue
                            pglMessages.print(f"Loaded Flywheel NIfTI {scan.name!r} for {sourceName!r}.")
                            scans = [scan]
                        mri.scans = [*mri.scans, *scans]
                        if scanProgress is not None: scanProgress(1, f"Loaded — {sourceName}")
                except BaseException:
                    shutil.rmtree(publishedDirectory)
                    raise

                if not mri.scans:
                    publishedDirectory.rmdir()

            if progress is not None: progress(1, "Summarizing imported MRI scans")
            pglMessages.printHeader("DICOM conversion results", fillChar="-")
            for scan in mri.scans:
                summary = scan.getScanSummary(readHeaders=False)
                tr = summary["trSeconds"]
                trText = "unknown" if tr is None else f"{tr:.6g}s"
                pglMessages.message(
                    f"{scan.name}: dims={summary['scanDims']}, voxelDims={summary['voxelDims']}mm, "
                    f"TR={trText}, qform_code={summary['qform_code']}, "
                    f"sform_code={summary['sform_code']}"
                )

            return mri

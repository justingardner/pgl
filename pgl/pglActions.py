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
import matplotlib
from tqdm.auto import tqdm

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
        sortByAcquisitionTime = Bool(True, help="Load runs in acquisition-time order, oldest first. If False, preserve the chooser's order.")
        
        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, experimentName: str=None, subjectID: str=None, sessionName: str=None, runName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None, sortByAcquisitionTime: bool=True) -> None:
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
            self.selectedPaths = validPaths
            self.filesystemPrefix = filesystemPrefix or ""
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

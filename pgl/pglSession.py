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
from traitlets import HasTraits, Float, Int, Any, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance, observe
from fsspec import AbstractFileSystem
from .pglSettings import pglSettings, pglSettingsManager, pglTraitSettings
from .pglPipeline import pglActionable
from .pglMessages import pglMessages
from types import SimpleNamespace
import pandas as pd
from .pglExperiment import pglTask
try:
    import mne
except ImportError:
    mne = None
from .pglBase import pglBase
from pathlib import Path
from .pglData import pglEpochsDataMatrix
import posixpath
import matplotlib.pyplot as plt
from traitlets import observe
from .pglLazyLoad import pglLazyLoad, pglFsspecBacking

##################################
# pglRun
##################################
class pglRun(pglLazyLoad, pglExperimentBase):

    filesystem = Instance(AbstractFileSystem, allow_none=True, help="Filesystem for serialization").tag(serialize=False)
    fullDataPath = Unicode(allow_none=True, default_value="", help="Full path to data", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Filesystem protocol prefix", visible=False)

    experimentSettings = Instance(pglExperimentSettings, allow_none=True, default_value=None, help="Experiment settings").tag(lazyload=True)
    settings = Instance(pglSettings, allow_none=True, default_value=None, help="Settings used for this run").tag(lazyload=True, loader="_loadSettings")
    data = Instance(pglExperimentData, allow_none=True, default_value=None, help="Experiment data").tag(lazyload=True)
    tasks = List(Instance(pglTaskBase), allow_none=True, default_value=None, help="Tasks from experiment").tag(lazyload=True, loader="_loadTasks")

    _taskCache = Dict(default_value={}, help="Tasks loaded individually by index").tag(serialize=False)

    @property
    def _tasks(self):
        """Nonloading compatibility access for pglExperimentBase.load()."""
        return self._trait_values.get("tasks")

    @_tasks.setter
    def _tasks(self, value):
        self.tasks = value

    @observe("tasks")
    def _tasksChanged(self, change):
        # The resolved list already contains the cached task objects.
        # Once a list is assigned, it becomes the authoritative collection.
        self._taskCache = {}
        
    def getTask(self, taskName='', taskID=None):
        '''
        get the task, if taskName is not set get the task that has the largest number of trials run, preferring
        tasks in which settings.nTrials has been set (i.e. that have been set to run nTrials and have completed those trials)
        '''
        if taskName == '':
            hasNTrialsSet = False
            task = None

            # for each task 
            for t in self.tasks:
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
            matchingTasks = [(iTask, t) for iTask, t in enumerate(self.tasks) if t.taskName == taskName]

            # no match
            if not matchingTasks:
                pglMessages.warning(f"No task matches taskName '{taskName}'")
                return None

            #single match just return it
            if len(matchingTasks) == 1:
                task = matchingTasks[0][1]
                
            # muultiple matches, see if taskID is se
            elif taskID is None:
                pglMessages.warning(f"Multiple tasks match taskName '{taskName}'; specify taskID")
                return None
            else:
                task = next((t for iTask, t in matchingTasks if iTask == taskID), None)
                if task is None:
                    pglMessages.warning(f"No task matches taskName '{taskName}' and taskID {taskID}")
                    return None
        return task

    def getTaskAt(self, taskIndex):
        """Load one task without resolving the full tasks field."""
        if not isinstance(taskIndex, int):
            raise TypeError(f"taskIndex must be an int, got {type(taskIndex).__name__}")

        # Inspect pending status and stored values without accessing self.tasks.
        if not self.lazyIsPending("tasks"):
            tasks = self._trait_values.get("tasks")
            if tasks is None: raise ValueError("Tasks are None and have no deferred backing")
            return tasks[taskIndex]

        experimentSettings = self.experimentSettings
        if experimentSettings is None: raise ValueError("Cannot load tasks without experiment settings")

        taskNames = experimentSettings.tasks
        if taskIndex < 0: taskIndex += len(taskNames)
        if taskIndex < 0 or taskIndex >= len(taskNames):
            raise IndexError(f"Task index {taskIndex} is outside 0 to {len(taskNames) - 1}")

        if taskIndex not in self._taskCache:
            # Use the explicitly attached task backing, not inferred source fields.
            backing = self._lazyRuntime().pending["tasks"]
            taskName = taskNames[taskIndex]
            taskDirectory = pglTaskBase.getTaskDirectoryName(taskID=taskIndex, taskName=taskName)
            taskPath = posixpath.join(backing.path, taskDirectory)

            pglMessages.message(f"Loading task {taskIndex}: {taskPath}", messageType="detailed")
            task = pglTaskBase.load(dataPath=taskPath, filesystem=backing.fs)

            if task is None:
                raise RuntimeError(f"Could not load task {taskIndex} ({taskName!r}) from {taskPath!r}")

            # The legacy task loader can return an object with missing components.
            if task.settings is None or task.state is None or task.data is None:
                raise RuntimeError(f"Task {taskIndex} has missing settings/state/data: {taskPath!r}")

            self._taskCache[taskIndex] = task

        return self._taskCache[taskIndex]

    def __init__(self, fullDataPath=None, filesystem=None, filesystemPrefix=None):
        """Initialize in memory, then attach backing only for an explicit source."""
        super().__init__()

        if fullDataPath is None:
            self.filesystem = filesystem
            self.fullDataPath = ""
            self.filesystemPrefix = filesystemPrefix or ""
            return

        self.filesystem, self.fullDataPath, self.filesystemPrefix = pglBase.validateFilesystem(
            filesystem=filesystem,
            dataPath=fullDataPath,
            filesystemPrefix=filesystemPrefix,
        )
        if self.filesystem is None: raise ValueError(f"Could not resolve run source: {fullDataPath!r}")

        # Replace only the initialization values on this newly constructed run.
        for name in ("experimentSettings", "settings", "data"):
            backing = pglFsspecBacking(self.filesystem, posixpath.join(self.fullDataPath, f"{name}.json"))
            self.lazyBind(name, backing, replace=True)

        # Tasks are backed by a directory/bundle rather than a single JSON file.
        self.lazyBind("tasks", pglFsspecBacking(self.filesystem, self.fullDataPath), replace=True)
             
    def _loadSettings(self, backing):
        """Preserve settings restoration without refreshing machine configuration."""
        settings = pglSettings.load(filename=backing.path, filesystem=backing.fs, refresh=False)
        if settings is None: raise RuntimeError(f"Could not load run settings from {backing.path!r}")
        return settings

    def _loadTasks(self, backing):
        """Resolve all tasks using the same objects loaded individually."""
        experimentSettings = self.experimentSettings
        if experimentSettings is None: raise ValueError("Cannot load tasks without experiment settings")
        return [self.getTaskAt(index) for index in range(len(experimentSettings.tasks))]
                
    def getTaskNames(self):
        '''
        Extracts task names from experimentSettings
        '''
        return(", ".join(self.experimentSettings.tasks))
    
    def display(self, fig=None):
        """
        Display plot of the run.

        If fig is provided, replace its contents with the run plots.
        """
        try:
            nTasks = len(self.tasks)
            nRows = nTasks + 1

            if fig is None:
                fig = plt.figure(
                    figsize=(12, 4 * nRows),
                    constrained_layout=True,
                )
            else:
                fig.clear()
                fig.set_layout_engine("constrained")

            axes = fig.subplots(nRows, 1, squeeze=False)[:, 0]

            # Display experiment
            self.data.display(ax=axes[0])

            # Display tasks
            for iTask, task in enumerate(self.tasks):
                task.display(ax=axes[iTask + 1])

            return fig

        except Exception as e:
            pglMessages.warning(f"error: {e}")
    
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
            pglMessages.warning(f"Could not find {taskName} in experiemnt.\nValid tasks are: {' '.join(t.settings.taskName for t in self.tasks)}")
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
            pglMessages.warning(f"Could not find '{parameterName}' in parameters {[p.settings.name for p in parameters]}")
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
        
    def saveDerivedRun(self, dataPath, filesystem=None, filesystemPrefix=None):
        """Export current behavioral contents to a fresh directory.

        Preserves cached edits and the existing behavioral bundle layout.
        Reads acquisition state only for export when self.state is None.
        All required save failures propagate.

        Not atomic: failed exports may leave partial destination files.
        Does not copy acquisition artifacts or register a checkpoint.
        """
        from .pglSerialize import pglSerialize

        components = {
            "experimentSettings": self.experimentSettings,
            "settings": self.settings,
            "data": self.data,
        }
        for name, component in components.items():
            if component is None: raise ValueError(f"Cannot save run: missing {name}")

        # Preserve state for export without installing it on the analysis run.
        state = self.state
        if state is None:
            if self.filesystem is None or not self.fullDataPath:
                raise ValueError("Cannot save run: state is missing and there is no source")

            statePath = posixpath.join(self.fullDataPath, "state.json")
            with self.filesystem.open(statePath, "rb") as fileHandle:
                state = pglSerialize.fromJSON(fileHandle.read(), filename=statePath)

        if state is None: raise ValueError("Cannot save run: experiment state is missing")
        components["state"] = state

        for name, component in components.items():
            if not callable(getattr(component, "saveStrict", None)):
                raise TypeError(f"Cannot save run {name}: no saveStrict() method")

        taskValues = self.tasks
        if taskValues is None: raise ValueError("Cannot save run: tasks are missing")
        tasks = list(taskValues)
        experimentSettings = components["experimentSettings"]

        if len(tasks) != len(experimentSettings.tasks):
            raise ValueError("Task count does not match experiment settings")

        for index, task in enumerate(tasks):
            if task is None: raise ValueError(f"Cannot save run: task {index} is missing")
            if not callable(getattr(task, "saveBundleStrict", None)):
                raise TypeError(f"Cannot save task {index}: no saveBundleStrict() method")

            expected = pglTaskBase.getTaskDirectoryName(taskID=index, taskName=experimentSettings.tasks[index])
            actual = task._getTaskDirectoryName()
            if actual != expected:
                raise ValueError(f"Task {index} directory mismatch: {actual!r} != {expected!r}")

        filesystem, destination, _ = pglBase.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            requireExists=False,
        )
        if filesystem is None: raise ValueError("Could not resolve destination filesystem")
        if filesystem.exists(destination): raise FileExistsError(f"Derived run destination already exists: {destination}")

        filesystem.makedirs(destination, exist_ok=False)

        for name in ("experimentSettings", "settings", "state", "data"):
            components[name].saveStrict(filename=posixpath.join(destination, f"{name}.json"), filesystem=filesystem)

        for task in tasks:
            task.saveBundleStrict(dataPath=destination, filesystem=filesystem)

        return destination
    
class pglMNEBundleBacking:
    """Deferred FIF bundle with a caller-owned fallback cache.

    directLocalPolicy:
        "materialize":
            Always use verified materialization.

        "immutable":
            Permit verified direct-local access when layout is suitable.
            The caller/storage owner guarantees that all source files and
            their resolved paths remain available and unchanged while MNE
            or any derived reader uses them.

    Construction performs no storage access.

    _localEntrypoint and _localStrategy are diagnostic results of successful
    reader initialization, not substitutes for verification on a new load.
    Reader caching remains the responsibility of pglLazyLoad/getRawAt().
    """

    def __init__(
        self,
        filesystem,
        bundle,
        cacheDirectory,
        *,
        directLocalPolicy="materialize",
    ):
        from copy import deepcopy

        if directLocalPolicy not in ("materialize", "immutable"):
            raise ValueError(
                "directLocalPolicy must be 'materialize' or 'immutable'"
            )

        if cacheDirectory is None or not str(cacheDirectory).strip():
            raise ValueError("An explicit caller-owned cacheDirectory is required")

        self.fs = filesystem
        self.bundle = deepcopy(bundle)
        self.cacheDirectory = str(cacheDirectory)
        self.directLocalPolicy = directLocalPolicy
        self._localEntrypoint = None
        self._localStrategy = None
           
################################################################
# filename: pglMNECache.py
# purpose: Session-owned FIF cache with cross-session hard links
################################################################

import errno
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import tempfile
import time

from contextlib import contextmanager
from pathlib import Path


class pglMNECache:
    """Session-owned verified FIF cache.

    Payload files exist only in session/unregistered bundle directories.
    Other bundles can hard-link identical verified files.

    Cache manifests contain identities, not payload copies or hard links.
    No automatic cleanup or eviction is implemented.

    Callers must keep cached files immutable and available while readers use
    them. The cache must not be writable by untrusted users/processes.
    """

    manifestName = ".pgl-cache.json"

    @staticmethod
    def _identity(path):
        """Hash a regular file without following a leaf symlink."""
        path = Path(path)
        if path.is_symlink(): raise ValueError(f"Cache file must not be a symlink: {path}")

        digest, size = hashlib.sha256(), 0

        with path.open("rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode): raise ValueError(f"Cache path is not a regular file: {path}")

            while True:
                block = source.read(1024 * 1024)
                if not block: break
                digest.update(block)
                size += len(block)

        return {"sha256": digest.hexdigest(), "size_bytes": size}

    @classmethod
    def _verify(cls, path, identity):
        expected = {key: identity[key] for key in ("sha256", "size_bytes")}
        if cls._identity(path) != expected: raise ValueError(f"Cached FIF integrity mismatch: {path}")

    @staticmethod
    @contextmanager
    def _namespaceLock(path, timeout=600):
        """Coordinate cooperating cache writers across processes.

        The persistent lock file contains no payload data.
        Do not delete it while loaders may be running.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        acquired = False

        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()

                def acquire():
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

                def release():
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

            else:
                import fcntl

                def acquire():
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                def release():
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            deadline = time.monotonic() + timeout

            while True:
                try:
                    acquire()
                    acquired = True
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK): raise
                    if time.monotonic() >= deadline: raise TimeoutError(f"Timed out waiting for cache lock: {path}") from error
                    time.sleep(0.05)

            yield

        finally:
            try:
                if acquired: release()
            finally:
                handle.close()

    @classmethod
    def _validateManifest(cls, manifest):
        """Validate the small local cache manifest."""
        if not isinstance(manifest, dict) or set(manifest) != {"version", "kind", "entrypoint", "files"}:
            raise ValueError("Invalid cache manifest fields.")

        if manifest["version"] != 1 or manifest["kind"] not in ("raw", "epochs"): raise ValueError("Unsupported cache manifest.")
        if not isinstance(manifest["files"], list) or not manifest["files"]: raise ValueError("Missing cache file identities.")

        def validateName(name):
            if not isinstance(name, str) or not name: raise ValueError("Invalid cache filename.")
            if any(part in ("", ".", "..") or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in name.split("/")):
                raise ValueError(f"Unsafe cache filename: {name!r}")
            if name.split("/")[0] == cls.manifestName: raise ValueError("Reserved cache manifest filename.")

        validateName(manifest["entrypoint"])
        names = set()

        for record in manifest["files"]:
            if not isinstance(record, dict) or set(record) != {"name", "sha256", "size_bytes"}: raise ValueError("Invalid cache identity.")
            validateName(record["name"])
            if record["name"] in names: raise ValueError("Duplicate cache filename.")
            names.add(record["name"])

            if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None: raise ValueError("Invalid cache SHA-256.")
            if type(record["size_bytes"]) is not int or not 0 <= record["size_bytes"] <= 9223372036854775807: raise ValueError("Invalid cache file size.")

        if manifest["entrypoint"] not in names: raise ValueError("Cache entrypoint is not listed.")
        return manifest

    @classmethod
    def _readManifest(cls, path):
        """Read bounded metadata; cached file bytes are verified separately."""
        if path.is_symlink(): raise ValueError(f"Cache manifest must not be a symlink: {path}")

        limit = 16 * 1024 * 1024
        with path.open("rb") as source: content = source.read(limit + 1)
        if len(content) > limit: raise ValueError("Cache manifest exceeds size limit.")
        return cls._validateManifest(json.loads(content))

    @classmethod
    def _candidateFiles(cls, namespaceRoot):
        """Build an in-memory lookup from published bundle manifests.

        No persistent central index or canonical payload directory is needed.
        Incomplete staging directories are excluded.
        """
        manifests = list(namespaceRoot.glob(f"sessions/session*/mne/*/{cls.manifestName}"))
        manifests.extend(namespaceRoot.glob(f"unregistered/*/{cls.manifestName}"))
        candidates = {}

        for path in sorted(manifests):
            if path.parent.name.startswith("."): continue
            manifest = cls._readManifest(path)

            for record in manifest["files"]:
                identity = (record["sha256"], record["size_bytes"])
                candidate = path.parent.joinpath(*record["name"].split("/"))
                candidates.setdefault(identity, []).append(candidate)

        return candidates

    @classmethod
    def _linkOrCopy(cls, source, target, identity):
        """Link a verified cached file; copy when hard links are unsupported."""
        unsupported = {
            errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK, errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }

        try:
            os.link(source, target)
        except OSError as error:
            if error.errno not in unsupported and getattr(error, "winerror", None) not in (1, 17, 50): raise

            with source.open("rb") as incoming, target.open("xb") as outgoing:
                while True:
                    block = incoming.read(1024 * 1024)
                    if not block: break
                    if outgoing.write(block) != len(block): raise OSError(f"Incomplete cache copy: {target}")

        cls._verify(target, identity)

    @classmethod
    def _download(cls, filesystem, sourcePath, targetPath, identity):
        """Copy source bytes into the private bundle and verify them."""
        digest, size = hashlib.sha256(), 0

        with filesystem.open(sourcePath, "rb") as source, targetPath.open("xb") as target:
            while True:
                block = source.read(1024 * 1024)
                if not block: break
                size += len(block)
                if size > identity["size_bytes"]: raise ValueError(f"FIF payload exceeds recorded size: {sourcePath}")
                digest.update(block)
                if target.write(block) != len(block): raise OSError(f"Incomplete FIF cache write: {sourcePath}")

            target.flush()
            os.fsync(target.fileno())

        if size != identity["size_bytes"] or digest.hexdigest() != identity["sha256"]:
            raise ValueError(f"FIF payload integrity mismatch: {sourcePath}")

        cls._verify(targetPath, identity)

    @classmethod
    def materialize(cls, backing):
        """Publish or reuse a complete verified session-specific FIF bundle.

        Caller must validate the FIF resource description before invoking this.
        Existing published bundles are never silently replaced or repaired.
        """
        validator = getattr(backing.fs, "validateBundleIdentities", None)
        if callable(validator): validator(backing.bundle["root"], backing.bundle["files"])

        manifest = {
            "version": 1,
            "kind": backing.bundle["kind"],
            "entrypoint": backing.bundle["entrypoint"],
            "files": sorted([dict(record) for record in backing.bundle["files"]], key=lambda record: record["name"]),
        }
        cls._validateManifest(manifest)
        manifestBytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fingerprint = hashlib.sha256(manifestBytes).hexdigest()
        context = getattr(backing.fs, "mneCacheContext", None)

        if context is None:
            namespace = "standalone"
        else:
            if not isinstance(context, dict) or set(context) != {"namespace", "sessionID"}: raise ValueError("Invalid MNE cache context.")
            namespace, sessionID = context["namespace"], context["sessionID"]
            if not isinstance(namespace, str) or re.fullmatch(r"[A-Za-z0-9_-]+", namespace) is None: raise ValueError("Invalid cache namespace.")
            if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807: raise ValueError("Invalid cache session ID.")

        namespaceRoot = Path(backing.cacheDirectory).expanduser().resolve() / namespace

        if context is None:
            destination = namespaceRoot / "unregistered" / f"{manifest['kind']}-{fingerprint}"
        else:
            resourceID = posixpath.basename(backing.bundle["root"].rstrip("/"))
            if re.fullmatch(r"r[0-9]{6}", resourceID) is None: resourceID = f"{manifest['kind']}-{fingerprint}"
            destination = namespaceRoot / "sessions" / f"session{sessionID:06d}" / "mne" / resourceID

        with cls._namespaceLock(namespaceRoot / ".cache.lock"):
            if os.path.lexists(destination):
                if destination.is_symlink() or not destination.is_dir(): raise ValueError(f"Invalid cache bundle directory: {destination}")
                if cls._readManifest(destination / cls.manifestName) != manifest: raise ValueError(f"Existing cache bundle has different identities: {destination}")

                for record in manifest["files"]: cls._verify(destination.joinpath(*record["name"].split("/")), record)
                return str(destination.joinpath(*manifest["entrypoint"].split("/")))

            candidates = cls._candidateFiles(namespaceRoot)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".building-", dir=str(destination.parent)))

            try:
                for record in manifest["files"]:
                    target = staging.joinpath(*record["name"].split("/"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    identity = (record["sha256"], record["size_bytes"])
                    reused = False

                    for candidate in candidates.get(identity, []):
                        # Stale metadata can point to a removed file. A present
                        # but corrupt candidate raises instead of hiding damage.
                        if not os.path.lexists(candidate): continue
                        cls._verify(candidate, record)
                        cls._linkOrCopy(candidate, target, record)
                        reused = True
                        break

                    if not reused:
                        sourcePath = posixpath.join(backing.bundle["root"], record["name"])
                        cls._download(backing.fs, sourcePath, target, record)

                    candidates.setdefault(identity, []).append(target)

                with (staging / cls.manifestName).open("xb") as target:
                    if target.write(manifestBytes) != len(manifestBytes): raise OSError("Incomplete cache manifest write.")
                    target.flush()
                    os.fsync(target.fileno())

                # Cooperating processes hold the namespace lock. Publishing
                # occurs only after every payload and the manifest are ready.
                if os.path.lexists(destination): raise FileExistsError(destination)
                staging.rename(destination)

            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

        return str(destination.joinpath(*manifest["entrypoint"].split("/")))     
##################################
# pglMNE
##################################
class pglMNE(pglLazyLoad, pglActionable):
    
    raws = List(Any(), help="Ordered Raw recordings", lazyload=True, loader="_loadRaws", serialize=False)
    _rawCache = Dict(default_value={}, help="Individually resolved Raw recordings").tag(serialize=False)
    _raw = Any(allow_none=True, default_value=None, help="Current single or combined Raw", lazyload=True, loader="_loadRaw", writer="_saveRaw")
    rawReferenceTime = Float(allow_none=True, default_value=None, help='reference time for the raw to compare to session reference time')
    rawFilenames = List(Unicode(allow_none=True), help="filenames of raw (if they exist)")
    rawFilesystemPrefix = List(Unicode(allow_none=True), help="filenames of raw (if they exist)", serialize=False)
    events = Instance(np.ndarray, default_value=None, allow_none=True, help="Canonical events array", lazyload=True, loader="_loadEvents", writer="_saveEvents")
    eventsID = Instance(pd.DataFrame, default_value=None, allow_none=True, help="Event labeling metadata", lazyload=True, loader="_loadEventsID", writer="_saveEventsID")
    epochs = Any(default_value=None, allow_none=True, help="MNE Epochs with event metadata", lazyload=True, loader="_loadEpochs", writer="_saveEpochs")
    evoked = Any(default_value=None, allow_none=True, help="Current MNE Evoked", lazyload=True, loader="_loadEvoked", writer="_saveEvoked")
    ica = Any(default_value=None, allow_none=True, help="Current MNE ICA", lazyload=True, loader="_loadICA", writer="_saveICA")
    report = Any(default_value=None, allow_none=True, help="Analysis report", lazyload=True, loader="_loadReport", writer="_saveReport")
    
    def __init__(self):
        super().__init__()
        if mne is None:
            pglMessages.warning("mne library is not available. Need to add to environment")
            return
        
        # init reprot
        self.report = mne.Report(title="MNE analysis report", verbose=False)
        
    def add(self, data, filename: str = None, filesystemPrefix: str = None):
        '''
        add data to the pglMNE instance. Handles known types like mne raw or epochs
        '''
        import mne
        # save either raw or epochs
        if isinstance(data, mne.io.BaseRaw):
            self.raws.append(data)
            self.rawFilenames.append(filename)
            self.rawFilesystemPrefix.append(filesystemPrefix)
            # this means that our rawAll object is no longer valid
    
    @observe("raws")
    def _rawsChanged(self, change):
        # Once assigned/resolved, the list holds the authoritative objects.
        self._rawCache = {}

    def getRawAt(self, index):
        """Resolve one recording, preserving duplicate-reference identity."""
        if type(index) is not int: raise TypeError("Raw index must be an integer")
        if not self.lazyIsPending("raws"): return self.raws[index]

        backing = self._lazyRuntime().pending["raws"]
        if not isinstance(backing, pglMNERawsBacking):
            raise TypeError("raws requires pglMNERawsBacking")

        count = len(backing.backings)
        if index < 0: index += count
        if index < 0 or index >= count: raise IndexError("Raw index is out of range")

        if index not in self._rawCache:
            reference = backing.backings[index]

            # Duplicate list entries point to the same backing object.
            for cachedIndex, raw in list(self._rawCache.items()):
                if backing.backings[cachedIndex] is reference:
                    self._rawCache[index] = raw
                    return raw

            import mne
            raw = pglMNEIO.loadRaw(reference)
            if not isinstance(raw, mne.io.BaseRaw):
                raise TypeError("Raw loader did not return an MNE BaseRaw")
            self._rawCache[index] = raw

        return self._rawCache[index]

    def _loadRaws(self, backing):
        """Resolve the collection, reusing individually loaded Raw objects."""
        if not isinstance(backing, pglMNERawsBacking):
            raise TypeError("raws requires pglMNERawsBacking")
        return [self.getRawAt(index) for index in range(len(backing.backings))]
    
    @property
    def raw(self):
        """Return current Raw, or resolve only the sole recording in raws."""
        raw = self._raw
        if raw is not None: return raw

        if self.lazyIsPending("raws"):
            backing = self._lazyRuntime().pending["raws"]
            if not isinstance(backing, pglMNERawsBacking):
                raise TypeError("raws requires pglMNERawsBacking")
            count = len(backing.backings)
        else:
            count = len(self.raws)

        if count == 1:
            self._raw = self.getRawAt(0)
            return self._raw

        if count:
            pglMessages.warning("There are multiple raws, run pglActions.mneConcatenate first")
        else:
            pglMessages.warning("No raws have been loaded")

        return None

    @raw.setter
    def raw(self, value):
        """Replace current Raw and cancel any pending current-Raw backing."""
        import mne

        if value is not None and not isinstance(value, mne.io.BaseRaw):
            raise TypeError(f"raw must be an mne.io.BaseRaw, got {type(value).__name__}")
        self._raw = value

    def _loadEvents(self, backing):
        return pglMNEIO.loadEvents(backing)

    def _saveEvents(self, value, destination):
        return pglMNEIO.saveEvents(value, destination.path, filesystem=destination.fs)

    def _loadEventsID(self, backing):
        return pglMNEIO.loadEventsID(backing)

    def _saveEventsID(self, value, destination):
        return pglMNEIO.saveEventsID(value, destination.path, filesystem=destination.fs)

    def _loadRaw(self, backing):
        if isinstance(backing, pglMNERawAliasBacking):
            return self.getRawAt(backing.index)
        return pglMNEIO.loadRaw(backing)
    def _loadEpochs(self, backing):
        return pglMNEIO.loadEpochs(backing)

    def _saveRaw(self, value, destination):
        return pglMNEIO.saveRaw(value, destination.path, filesystem=destination.fs)

    def _saveEpochs(self, value, destination):
        return pglMNEIO.saveEpochs(value, destination.path, filesystem=destination.fs)
        
    def _loadEvoked(self, backing):
        return pglMNEIO.loadEvoked(backing)

    def _saveEvoked(self, value, destination):
        return pglMNEIO.saveEvoked(value, destination.path, filesystem=destination.fs)

    def _loadICA(self, backing):
        return pglMNEIO.loadICA(backing)

    def _saveICA(self, value, destination):
        return pglMNEIO.saveICA(value, destination.path, filesystem=destination.fs)

    def _loadReport(self, backing):
        return pglMNEIO.loadReport(backing)

    def _saveReport(self, value, destination):
        return pglMNEIO.saveReport(value, destination.path, filesystem=destination.fs)
        
    def saveBundle(self, dataPath, filesystem=None):
        return pglMNEBundle.save(self, dataPath, filesystem=filesystem)

    @classmethod
    def loadBundle(cls, dataPath, filesystem=None, *, cacheDirectory, directLocalPolicy="materialize"):
        return pglMNEBundle.load(dataPath, filesystem=filesystem, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

    def validatePicks(self, picks):
        """
        Validate a single MNE pick selector.

        Returns either:

        - An available channel-type selector, such as ``"mag"``, ``"grad"``,
        ``"eeg"``, or ``"eog"``.
        - An exact channel name.
        - The full channel name resolved from a partial sensor name via
        :meth:`lookupSensor`.

        Raises
        ------
        ValueError
            If ``picks`` is empty, is not an available channel type, and cannot
            be resolved to a sensor/channel name.
        """
        if not isinstance(picks, str) or not picks.strip():
            raise ValueError("picks must be a non-empty string")

        picks = picks.strip()

        # First allow an exact channel name, including names which happen to
        # overlap with a possible MNE selector.
        if picks in self.raw.ch_names:
            return picks

        # These are the actual direct channel types available in this recording,
        # e.g. "mag", "grad", "eeg", "eog", "stim", etc.
        availableTypes = self.raw.get_channel_types(
                unique=True,
                only_data_chs=False,
            )

        if picks.lower() in availableTypes:
            return picks.lower()

        # Resolve a partial sensor/channel name using lookupSensor
        sensorName = self.lookupSensor(picks)

        if sensorName is not None:
            return sensorName

        raise ValueError(
            f"Invalid picks value {picks!r}. "
            f"Available channel types: {', '.join(availableTypes)}. "
            f"Use a full channel name or a resolvable partial sensor name."
        )
    
    def isSensor(self, name):
        '''
        check if the name is a sensor
        '''
        if self.lookupSensor(name, verbose=False) is not None:
            return True
        return False
        
    def lookupSensor(self, name, verbose=True):
        '''
        helper function to lookup the full sensor name from a parital match
        
        Args:
            name (str): A parital name of a sensor like R401
            
        Returns:
            str of full name of sensor or None if no match, or no unique match
        '''

        if self.raw is None:
            pglMessages.warning(f"No raw is loaded")
            return None
        
        matchingChannels = [
            channelName
            for channelName in self.raw.ch_names
            if channelName.lower().startswith(name.lower())
        ]

        if not matchingChannels:
            if verbose: pglMessages.warning(f"no matching sensor to: {name}")
            return None

        if len(matchingChannels) > 1:
            if verbose: pglMessages.warning(f"Multiple channels start with {name!r}: {matchingChannels}")
            return None

        # return the match
        return(matchingChannels[0])
    
    def getDataMatrix(self, verbose=True, dropBadData=True):
        """Return epochs as a pglEpochsDataMatrix, or None if import fails."""
        import mne

        if self.epochs is None:
            pglMessages.warning("Cannot create data matrix: no epochs have been loaded or created.")
            return None

        if not isinstance(self.epochs, mne.BaseEpochs):
            pglMessages.warning(f"Cannot create data matrix: expected MNE Epochs, got {type(self.epochs).__name__}.")
            return None

        if len(self.epochs.events) == 0:
            pglMessages.warning("Cannot create data matrix: no epochs are available.")
            return None

        try:
            epochs = self.epochs

            if not epochs.preload:
                # Load a copy to leave the source unchanged and defer threshold
                # rejection so fromEpochs can label bad trials before removing them.
                epochs = epochs.copy()
                rejectCriteria = epochs.reject
                flatCriteria = epochs.flat
                epochs.reject = None
                epochs.flat = None
                epochs.load_data(verbose=False)
                epochs.reject = rejectCriteria
                epochs.flat = flatCriteria

            return pglEpochsDataMatrix.fromEpochs(epochs, verbose=verbose, dropBadData=dropBadData)

        except (ValueError, TypeError, RuntimeError, OSError) as error:
            pglMessages.warning(f"Cannot create data matrix: {error}")
            return None

    def displayReport(self):
        """Save the current report as temporary HTML and open it in a browser.

        Returns the HTML Path, or None if no report is attached.
        Resolves a lazily backed report when necessary.

        The temporary directory is retained so the browser can read/reload it.
        Each call creates a new directory; successful exports are not
        automatically removed by this method.
        """
        import shutil
        from pathlib import Path
        from tempfile import mkdtemp

        report = self.report

        if report is None:
            pglMessages.warning("No MNE report is attached to this session.")
            return None

        directory = Path(mkdtemp(prefix="pgl-mne-report-"))
        filename = directory / "report.html"

        try:
            report.save(str(filename), overwrite=False, open_browser=True)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

        return filename
################################################################
#   filename: pglMNEBundle.py
#    purpose: Save and lazily restore a complete pglMNE bundle
################################################################

import hashlib
import json
import math
import posixpath
import re
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fsspec.core import url_to_fs
from traitlets import HasTraits

from .pglLazyLoad import pglFsspecBacking
from .pglSerialize import pglSerialize, pglShouldReadTraitForSerialize


class pglMNEVerifiedBacking:
    """Single-payload backing with verification at field resolution."""

    def __init__(self, filesystem, path, identity):
        self.fs = filesystem
        self.path = path
        self.identity = dict(identity)

    @contextmanager
    def open(self, mode="rb", **kwargs):
        if mode != "rb": raise ValueError("Verified payload backing supports rb only")

        # Underlying checkpoint-filesystem verification remains active.
        with self.fs.open(self.path, "rb") as source:
            content = source.read(self.identity["size_bytes"] + 1)

        actual = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        expected = {key: self.identity[key] for key in actual}
        if actual != expected: raise ValueError(f"Payload integrity mismatch: {self.path}")

        with BytesIO(content) as stream:
            yield stream


class pglMNEBundle:
    """Object-bundle persistence; no database registration or atomic publication."""

    contract = "pgl.mne.bundle.v1"
    fields = ("epochs", "events", "eventsID", "evoked", "ica", "report")
    formats = {
        "raw": "fif",
        "epochs": "fif",
        "events": "npy",
        "eventsID": "parquet",
        "evoked": "fif",
        "ica": "fif",
        "report": "hdf5",
    }

    @classmethod
    def _metadataNames(cls, obj):
        payloadNames = {"raws", "_raw", *cls.fields}
        return sorted(
            name for name, trait in obj.traits().items()
            if name not in payloadNames and trait.metadata.get("serialize", True) is not False
        )

    @classmethod
    def _checkMetadata(cls, value, active=None):
        """Reject unsupported metadata and cycles before using the PGL encoder."""
        if active is None: active = set()

        if isinstance(value, (Enum, Exception, datetime, Path)): return
        if value is None or isinstance(value, (str, bool, int)): return
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)): raise ValueError("Nonfinite metadata numbers are not supported")
            return
        if isinstance(value, np.integer): return
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufUS":
                raise TypeError(f"Unsupported metadata array dtype: {value.dtype}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError("Nonfinite metadata arrays are not supported")
            return

        identity = id(value)
        if identity in active: raise ValueError("Circular metadata cannot be preserved")
        active.add(identity)

        try:
            if isinstance(value, pglSerialize):
                children = value.toJSONdict()
            elif isinstance(value, HasTraits):
                children = {
                    name: getattr(value, name)
                    for name in value.traits()
                    if pglShouldReadTraitForSerialize(value, name)
                }
            elif isinstance(value, SimpleNamespace):
                children = vars(value)
            elif isinstance(value, dict):
                children = value
            elif isinstance(value, (list, tuple)):
                for child in value: cls._checkMetadata(child, active)
                return
            else:
                raise TypeError(f"Unsupported metadata type: {type(value).__name__}")

            for name, child in children.items():
                if not isinstance(name, str): raise TypeError("Metadata dictionary keys must be strings")
                cls._checkMetadata(child, active)
        finally:
            active.remove(identity)

    @classmethod
    def _encodeMetadata(cls, metadata):
        value = SimpleNamespace(**metadata)
        cls._checkMetadata(value)
        text = pglSerialize.toJSON(value)

        # Detect dropped fields or permissive restoration changing the record.
        restored = pglSerialize.fromJSON(text)
        if not isinstance(restored, SimpleNamespace):
            raise ValueError("Metadata did not restore as a namespace")
        cls._checkMetadata(restored)
        if pglSerialize.toJSON(restored) != text:
            raise ValueError("Metadata did not survive its PGL codec round trip")

        return text

    @staticmethod
    def _validatePath(name):
        if not isinstance(name, str) or not name:
            raise ValueError("Missing bundle path")
        if any(
            part in ("", ".", "..") or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None
            for part in name.split("/")
        ):
            raise ValueError(f"Unsafe bundle path: {name!r}")

    @classmethod
    def _validateDescriptor(cls, descriptor):
        expectedFields = {"contract", "metadata", "raws", "currentRaw", "fields", "resources"}
        if not isinstance(descriptor, dict) or set(descriptor) != expectedFields:
            raise ValueError("Invalid MNE bundle descriptor fields")
        if descriptor["contract"] != cls.contract: raise ValueError("Unsupported MNE bundle contract")
        if not isinstance(descriptor["metadata"], str): raise ValueError("Invalid metadata record")
        if not isinstance(descriptor["raws"], list): raise ValueError("Invalid Raw references")
        if not isinstance(descriptor["fields"], dict) or set(descriptor["fields"]) != set(cls.fields):
            raise ValueError("Invalid analysis-field references")
        if not isinstance(descriptor["resources"], dict): raise ValueError("Invalid resource table")

        resources = descriptor["resources"]
        used = set()

        def checkReference(reference, kind, allowNone=False):
            if reference is None and allowNone: return
            if not isinstance(reference, str) or reference not in resources:
                raise ValueError("Unknown resource reference")
            if resources[reference]["kind"] != kind:
                raise ValueError(f"Resource kind mismatch: {reference}")
            used.add(reference)

        for resourceID, resource in resources.items():
            if not isinstance(resourceID, str) or re.fullmatch(r"r[0-9]{6}", resourceID) is None:
                raise ValueError("Invalid resource ID")
            if not isinstance(resource, dict) or set(resource) != {"kind", "format", "root", "entrypoint", "files"}:
                raise ValueError("Invalid resource record")

            kind = resource["kind"]
            if kind not in cls.formats or resource["format"] != cls.formats[kind]:
                raise ValueError("Invalid resource format")
            if resource["root"] != f"objects/{resourceID}":
                raise ValueError("Invalid resource root")

            cls._validatePath(resource["entrypoint"])
            records = resource["files"]
            if not isinstance(records, list) or not records:
                raise ValueError("Missing resource files")

            names = []
            for record in records:
                if not isinstance(record, dict) or set(record) != {"name", "sha256", "size_bytes"}:
                    raise ValueError("Invalid file identity")
                cls._validatePath(record["name"])
                if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
                    raise ValueError("Invalid SHA256")
                if type(record["size_bytes"]) is not int or not 0 <= record["size_bytes"] <= 9223372036854775807:
                    raise ValueError("Invalid file size")
                names.append(record["name"])

            if names != sorted(set(names)): raise ValueError("Resource files must be unique and sorted")
            if resource["entrypoint"] not in names: raise ValueError("Missing entrypoint")
            if kind not in ("raw", "epochs") and names != [resource["entrypoint"]]:
                raise ValueError("Expected a single-file analysis resource")

        for reference in descriptor["raws"]: checkReference(reference, "raw")
        checkReference(descriptor["currentRaw"], "raw", allowNone=True)
        for name, reference in descriptor["fields"].items():
            checkReference(reference, name, allowNone=True)
        if used != set(resources): raise ValueError("Unreferenced resources in descriptor")

    @classmethod
    def save(cls, obj, dataPath, filesystem=None):
        """Save current contents; write descriptor last. Partial output may remain."""
        import mne

        if type(obj) is not pglMNE:
            raise TypeError("This bundle contract supports pglMNE, not subclasses")
        if dataPath is None or not str(dataPath).strip():
            raise ValueError("A destination is required")

        if filesystem is None: filesystem, destination = url_to_fs(str(dataPath))
        else: destination = str(dataPath)
        if filesystem.exists(destination): raise FileExistsError(destination)

        metadata = {name: getattr(obj, name) for name in cls._metadataNames(obj)}
        metadataText = cls._encodeMetadata(metadata)

        # Do not use obj.raw: its convenience fallback can create an alias.
        raws = list(obj.raws)
        currentRaw = obj._raw
        values = {name: getattr(obj, name) for name in cls.fields}

        for raw in raws + ([] if currentRaw is None else [currentRaw]):
            if not isinstance(raw, mne.io.BaseRaw):
                raise TypeError("Raw references must contain BaseRaw objects")

        descriptor = {
            "contract": cls.contract,
            "metadata": metadataText,
            "raws": [],
            "currentRaw": None,
            "fields": {},
            "resources": {},
        }
        rawIDs = {}

        filesystem.makedirs(destination, exist_ok=False)
        filesystem.makedirs(posixpath.join(destination, "objects"), exist_ok=True)

        def saveResource(kind, value):
            resourceID = f"r{len(descriptor['resources']):06d}"
            relativeRoot = f"objects/{resourceID}"
            target = posixpath.join(destination, relativeRoot)

            if kind == "raw":
                bundle = pglMNEIO.saveRaw(value, target, filesystem=filesystem)
            else:
                bundle = obj.lazySaveField(kind, pglFsspecBacking(filesystem, target))

            descriptor["resources"][resourceID] = {**bundle, "root": relativeRoot}
            return resourceID

        def saveRawReference(raw):
            if id(raw) not in rawIDs: rawIDs[id(raw)] = saveResource("raw", raw)
            return rawIDs[id(raw)]

        descriptor["raws"] = [saveRawReference(raw) for raw in raws]
        if currentRaw is not None:
            descriptor["currentRaw"] = saveRawReference(currentRaw)

        for name, value in values.items():
            descriptor["fields"][name] = None if value is None else saveResource(name, value)

        cls._validateDescriptor(descriptor)
        content = json.dumps(descriptor, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        path = posixpath.join(destination, "mne.json")

        with filesystem.open(path, "xb") as target:
            if target.write(content) != len(content):
                raise OSError("Incomplete MNE descriptor write")

        with filesystem.open(path, "rb") as stored:
            if stored.read(len(content) + 1) != content:
                raise ValueError("MNE descriptor verification failed")

        return destination, descriptor

    @classmethod
    def load(cls, dataPath, filesystem=None, *, cacheDirectory, directLocalPolicy="materialize", maxDescriptorBytes=16 * 1024 * 1024):
        """Restore metadata/backing without reading analysis payloads."""
        if directLocalPolicy not in ("materialize", "immutable"): raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'")
        
        if cacheDirectory is None or not str(cacheDirectory).strip():
            raise ValueError("An explicit caller-owned cache directory is required")
        if type(maxDescriptorBytes) is not int or maxDescriptorBytes <= 0:
            raise ValueError("Invalid descriptor size limit")

        if filesystem is None: filesystem, root = url_to_fs(str(dataPath))
        else: root = str(dataPath)

        with filesystem.open(posixpath.join(root, "mne.json"), "rb") as source:
            content = source.read(maxDescriptorBytes + 1)
        if len(content) > maxDescriptorBytes:
            raise ValueError("MNE descriptor exceeds size limit")

        descriptor = json.loads(content)
        cls._validateDescriptor(descriptor)

        metadata = pglSerialize.fromJSON(descriptor["metadata"])
        if not isinstance(metadata, SimpleNamespace):
            raise ValueError("Invalid metadata namespace")
        if cls._encodeMetadata(vars(metadata)) != descriptor["metadata"]:
            raise ValueError("Metadata restoration changed its encoded contents")

        obj = pglMNE()
        if set(vars(metadata)) != set(cls._metadataNames(obj)):
            raise ValueError("Metadata fields do not match the supported pglMNE schema")

        # Assignment errors propagate rather than being logged and ignored.
        for name, value in vars(metadata).items():
            obj.set_trait(name, value)

        currentMetadata = {name: getattr(obj, name) for name in cls._metadataNames(obj)}
        if cls._encodeMetadata(currentMetadata) != descriptor["metadata"]:
            raise ValueError("Metadata assignment changed stored values")

        obj.raws = []
        obj._raw = None
        for name in cls.fields: obj.set_trait(name, None)

        backings = {}
        for resourceID, resource in descriptor["resources"].items():
            resourceRoot = posixpath.join(root, resource["root"])

            if resource["kind"] in ("raw", "epochs"):
                backings[resourceID] = pglMNEBundleBacking(filesystem, {**resource, "root": resourceRoot}, cacheDirectory, directLocalPolicy=directLocalPolicy)
            else:
                backings[resourceID] = pglMNEVerifiedBacking(
                    filesystem,
                    posixpath.join(resourceRoot, resource["entrypoint"]),
                    resource["files"][0],
                )

        rawReferences = descriptor["raws"]
        obj.lazyBind(
            "raws",
            pglMNERawsBacking([backings[key] for key in rawReferences]),
            replace=True,
        )

        current = descriptor["currentRaw"]
        if current is not None:
            backing = (
                pglMNERawAliasBacking(rawReferences.index(current))
                if current in rawReferences else backings[current]
            )
            obj.lazyBind("_raw", backing, replace=True)

        for name, reference in descriptor["fields"].items():
            if reference is not None:
                obj.lazyBind(name, backings[reference], replace=True)

        return obj
                                                                                 
################################################################
#   filename: pglMNEIO.py
#    purpose: Format-specific MNE bundle I/O through fsspec
################################################################

import hashlib
import posixpath
from pathlib import Path
from tempfile import TemporaryDirectory

from fsspec.core import url_to_fs


class pglMNEIO:
    """Standalone FIF bundle writers, not checkpoint registration."""

    @staticmethod
    def _fileIdentity(fileHandle):
        digest = hashlib.sha256()
        size = 0

        while True:
            block = fileHandle.read(1024 * 1024)
            if not block: break
            digest.update(block)
            size += len(block)

        return {"sha256": digest.hexdigest(), "size_bytes": size}

    @classmethod
    def saveRaw(cls, raw, dataPath, filesystem=None, *, splitSize="2GB", bufferSizeSec=1.0):
        """Save a copy of current Raw contents as a verified FIF bundle."""
        import mne

        if not isinstance(raw, mne.io.BaseRaw): raise TypeError("raw must be an MNE BaseRaw")

        return cls._saveFif(
            raw,
            dataPath,
            filesystem,
            kind="raw",
            entrypoint="data_raw.fif",
            saveOptions={"split_size": splitSize, "buffer_size_sec": bufferSizeSec},
        )

    @classmethod
    def saveEpochs(cls, epochs, dataPath, filesystem=None, *, splitSize="2GB"):
        """Save a copy of current Epochs contents as a verified FIF bundle."""
        import mne

        if not isinstance(epochs, mne.BaseEpochs): raise TypeError("epochs must be an MNE BaseEpochs")

        return cls._saveFif(
            epochs,
            dataPath,
            filesystem,
            kind="epochs",
            entrypoint="data-epo.fif",
            saveOptions={"split_size": splitSize},
        )

    @classmethod
    def _saveFif(cls, value, dataPath, filesystem, *, kind, entrypoint, saveOptions):
        """Stage locally, transfer all files, and verify stored bytes.

        Returns a resource description with relative filenames.
        Failed transfers may leave partial destination output.
        """
        if dataPath is None or not str(dataPath).strip():
            raise ValueError("A destination directory is required")

        if filesystem is None:
            filesystem, destination = url_to_fs(str(dataPath))
        else:
            destination = str(dataPath)

        if filesystem.exists(destination):
            raise FileExistsError(f"MNE bundle destination already exists: {destination}")

        with TemporaryDirectory(prefix="pgl-mne-write-") as temporaryDirectory:
            localRoot = Path(temporaryDirectory)
            localEntry = localRoot / entrypoint

            # Protect the caller from save-time modifications.
            # For preloaded objects, copy() may duplicate data in RAM.
            working = value.copy()
            working.save(str(localEntry), fmt="double", overwrite=False, verbose=False, **saveOptions)

            if not localEntry.is_file():
                raise RuntimeError(f"MNE did not create the expected entrypoint: {entrypoint}")

            localFiles = sorted(path for path in localRoot.rglob("*") if path.is_file())
            records = []

            for localPath in localFiles:
                if localPath.is_symlink(): raise ValueError("Unexpected symlink in staged MNE bundle")
                with localPath.open("rb") as fileHandle: identity = cls._fileIdentity(fileHandle)
                records.append({"name": localPath.relative_to(localRoot).as_posix(), **identity})

            # MNE encoding must succeed before creating remote output.
            filesystem.makedirs(destination, exist_ok=False)

            for record in records:
                localPath = localRoot / record["name"]
                targetPath = posixpath.join(destination, record["name"])
                parentPath = posixpath.dirname(targetPath)
                if parentPath != destination: filesystem.makedirs(parentPath, exist_ok=True)

                with localPath.open("rb") as source, filesystem.open(targetPath, "xb") as target:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block: break
                        if target.write(block) != len(block):
                            raise OSError(f"Incomplete MNE payload write: {targetPath}")

                with filesystem.open(targetPath, "rb") as stored:
                    actual = cls._fileIdentity(stored)

                expected = {key: record[key] for key in ("sha256", "size_bytes")}
                if actual != expected:
                    raise ValueError(f"MNE payload verification failed: {targetPath}")

        return {
            "kind": kind,
            "format": "fif",
            "root": destination,
            "entrypoint": entrypoint,
            "files": records,
        }
        
    @staticmethod
    def _validateFifBundle(backing):
        """Validate the resource description without accessing storage."""
        import re

        bundle = backing.bundle

        if bundle.get("format") != "fif":
            raise ValueError("Expected FIF format")
        if bundle.get("kind") not in ("raw", "epochs"):
            raise ValueError("Unknown FIF bundle kind")

        root = bundle.get("root")
        records = bundle.get("files")
        entrypoint = bundle.get("entrypoint")

        if not isinstance(root, str) or not root:
            raise ValueError("Missing bundle root")
        if not isinstance(records, list) or not records:
            raise ValueError("Missing bundle files")

        def validateName(name):
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid bundle filename")

            if any(
                part in ("", ".", "..")
                or re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None
                for part in name.split("/")
            ):
                raise ValueError(f"Unsafe bundle filename: {name!r}")

        validateName(entrypoint)
        names = set()

        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Invalid FIF file record")

            name = record.get("name")
            validateName(name)

            if name in names:
                raise ValueError(f"Duplicate bundle filename: {name!r}")
            names.add(name)

            size = record.get("size_bytes")
            digest = record.get("sha256")

            if type(size) is not int or not 0 <= size <= 9223372036854775807:
                raise ValueError(f"Invalid bundle file size: {name!r}")

            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(f"Invalid bundle SHA256: {name!r}")

        if entrypoint not in names:
            raise ValueError("Entrypoint is not listed in bundle files")

        return bundle

    @staticmethod
    def _localBundlePaths(backing):
        """Resolve eligible local paths without reading payload bytes."""
        from pathlib import Path
        from fsspec.implementations.local import LocalFileSystem

        resolver = getattr(backing.fs, "localBundlePaths", None)
        if callable(resolver):
            return resolver(backing.bundle["root"], backing.bundle["files"])

        # Explicit support for ordinary local fsspec storage.
        # Other wrappers must provide their own identity-aware resolver.
        if type(backing.fs) is not LocalFileSystem:
            return None

        paths = {}

        for record in backing.bundle["files"]:
            source = Path(
                backing.fs._strip_protocol(
                    posixpath.join(backing.bundle["root"], record["name"])
                )
            ).expanduser()

            if source.is_symlink():
                return None

            source = source.resolve()
            if not source.is_file():
                return None

            paths[record["name"]] = source

        return paths

    @classmethod
    def _verifyLocalFifFiles(cls, backing, paths):
        """Hash actual local files without making a temporary payload copy."""
        import stat

        for record in backing.bundle["files"]:
            path = paths[record["name"]]

            with path.open("rb") as source:
                import os
                details = os.fstat(source.fileno())

                if not stat.S_ISREG(details.st_mode):
                    raise ValueError(f"FIF payload is not a regular file: {path}")

                if details.st_size != record["size_bytes"]:
                    raise ValueError(f"FIF payload size mismatch: {path}")

                actual = cls._fileIdentity(source)

            expected = {
                key: record[key]
                for key in ("sha256", "size_bytes")
            }
            if actual != expected:
                raise ValueError(f"FIF payload integrity mismatch: {path}")

    @staticmethod
    def _directFifLayoutCompatible(backing, paths):
        """Audit forward split references after all bundle files are verified.

        MNE follows NEXT references from the entrypoint. PREV references may
        contain obsolete absolute staging paths and are not traversal targets.
        Conservatively supports flat bundles with relative NEXT filenames.
        """
        import numpy as np
        import posixpath

        try:
            from mne._fiff.open import fiff_open
            from mne._fiff.tree import dir_tree_find
            from mne._fiff.tag import read_tag
            from mne._fiff.constants import FIFF
        except ImportError:
            try:
                from mne.io.open import fiff_open
                from mne.io.tree import dir_tree_find
                from mne.io.tag import read_tag
                from mne.io.constants import FIFF
            except ImportError:
                return False

        names = set(paths)
        if any("/" in name for name in names): return False
        if len({path.parent for path in paths.values()}) != 1: return False
        if any(path.name != name for name, path in paths.items()): return False

        nextFiles = {}

        for name, path in paths.items():
            fid, tree, _ = fiff_open(path, preload=False, verbose=False)
            successors = []

            with fid:
                for node in dir_tree_find(tree, FIFF.FIFFB_REF):
                    role, referenceName = None, None

                    for entry in node.get("directory") or ():
                        if entry.kind == FIFF.FIFF_REF_ROLE: role = int(np.asarray(read_tag(fid, entry.pos).data).item())
                        elif entry.kind == FIFF.FIFF_REF_FILE_NAME: referenceName = read_tag(fid, entry.pos).data

                    # Backward references are not followed from the entrypoint.
                    if role == FIFF.FIFFV_ROLE_PREV_FILE: continue
                    if role != FIFF.FIFFV_ROLE_NEXT_FILE: return False

                    # Every forward target must be a verified sibling file.
                    if not isinstance(referenceName, str): return False
                    if posixpath.basename(referenceName) != referenceName or "\\" in referenceName or referenceName not in names: return False
                    successors.append(referenceName)

            if len(successors) > 1: return False
            nextFiles[name] = successors[0] if successors else None

        # Reject cycles and bundles with members outside the forward chain.
        visited, current = set(), backing.bundle["entrypoint"]

        while current is not None:
            if current in visited: return False
            visited.add(current)
            current = nextFiles[current]

        return visited == names

    @classmethod
    def _resolveFifBundle(cls, backing):
        """Return (entrypoint, strategy); publish neither as a cached reader."""
        cls._validateFifBundle(backing)

        if backing.directLocalPolicy == "immutable":
            paths = cls._localBundlePaths(backing)

            if paths is not None:
                # Integrity failure is NOT an eligibility failure.
                # In particular, do not catch it and materialize instead.
                cls._verifyLocalFifFiles(backing, paths)

                if cls._directFifLayoutCompatible(backing, paths):
                    return (
                        str(paths[backing.bundle["entrypoint"]]),
                        "direct-local",
                    )

        return cls._materializeFifBundle(backing), "materialized"

    @classmethod
    def _localFifPath(cls, backing):
        """Resolve an explicit local reference or a verified FIF bundle."""
        if isinstance(backing, pglMNEBundleBacking):
            return cls._resolveFifBundle(backing)[0]

        protocols = backing.fs.protocol
        protocols = (
            {protocols} if isinstance(protocols, str) else set(protocols)
        )
        if not protocols.intersection({"file", "local"}):
            raise NotImplementedError(
                "Remote FIF requires pglMNEBundleBacking "
                "with an explicit local cache"
            )

        return str(backing.fs._strip_protocol(backing.path))

    @classmethod
    def _loadFif(cls, backing, *, kind):
        """Verify first; record successful initialization only after MNE opens."""
        import mne

        isBundle = isinstance(backing, pglMNEBundleBacking)

        if isBundle:
            # A previous resolved filename is not proof of present integrity.
            # Normal repeated field access is cached by the lazy-load owner.
            backing._localEntrypoint = None
            backing._localStrategy = None

            if backing.bundle.get("kind") != kind:
                raise ValueError(f"Expected a {kind} FIF bundle")

            path, strategy = cls._resolveFifBundle(backing)
        else:
            path = cls._localFifPath(backing)
            strategy = None

        if kind == "raw":
            value = mne.io.read_raw_fif(
                path,
                preload=False,
                on_split_missing="raise",
                verbose=False,
            )
        else:
            value = mne.read_epochs(path, preload=False, verbose=False)

        if isBundle:
            backing._localEntrypoint = path
            backing._localStrategy = strategy

        return value

    @classmethod
    def loadRaw(cls, backing):
        return cls._loadFif(backing, kind="raw")

    @classmethod
    def loadEpochs(cls, backing):
        return cls._loadFif(backing, kind="epochs")

    @classmethod
    def _materializeFifBundle(cls, backing):
        """Create a verified bundle using shared cached objects and hard links.

        Original storage payloads are never hard-linked into this cache.
        Successful cache objects and bundles remain caller-owned.
        """
        from .pglSession import pglMNECache

        cls._validateFifBundle(backing)
        return pglMNECache.materialize(backing)

    @classmethod
    def _saveBytesBundle(cls, content, dataPath, filesystem, *, kind, formatName, entrypoint):
        """Write and verify one encoded payload in a fresh bundle directory."""
        import hashlib
        import posixpath
        from fsspec.core import url_to_fs

        if dataPath is None or not str(dataPath).strip():
            raise ValueError("A destination directory is required")

        if filesystem is None:
            filesystem, destination = url_to_fs(str(dataPath))
        else:
            destination = str(dataPath)

        if filesystem.exists(destination):
            raise FileExistsError(f"Bundle destination already exists: {destination}")

        identity = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

        filesystem.makedirs(destination, exist_ok=False)
        targetPath = posixpath.join(destination, entrypoint)

        with filesystem.open(targetPath, "xb") as target:
            if target.write(content) != len(content):
                raise OSError(f"Incomplete payload write: {targetPath}")

        with filesystem.open(targetPath, "rb") as stored:
            if cls._fileIdentity(stored) != identity:
                raise ValueError(f"Payload verification failed: {targetPath}")

        return {
            "kind": kind,
            "format": formatName,
            "root": destination,
            "entrypoint": entrypoint,
            "files": [{"name": entrypoint, **identity}],
        }

    @staticmethod
    def _validateEvents(events):
        import numpy as np

        if not isinstance(events, np.ndarray):
            raise TypeError("events must be a NumPy array")
        if events.ndim != 2 or events.shape[1] != 3:
            raise ValueError("events must have shape (nEvents, 3)")
        if events.dtype.kind not in ("i", "u"):
            raise TypeError("events must have an integer dtype")

    @classmethod
    def saveEvents(cls, events, dataPath, filesystem=None):
        """Preserve the canonical integer events array without pickle."""
        from io import BytesIO
        import numpy as np

        cls._validateEvents(events)
        buffer = BytesIO()
        np.save(buffer, events, allow_pickle=False)

        return cls._saveBytesBundle(
            buffer.getvalue(), dataPath, filesystem,
            kind="events", formatName="npy", entrypoint="events.npy",
        )

    @classmethod
    def loadEvents(cls, backing):
        """Read through the supplied filesystem, including its integrity checks."""
        from io import BytesIO
        import numpy as np

        with backing.open("rb") as source: content = source.read()
        events = np.load(BytesIO(content), allow_pickle=False)
        cls._validateEvents(events)
        return events

    @staticmethod
    def _eventsIDDtypeRecord(dtype):
        """Record string/object representation; leave other dtypes to Parquet."""
        import numpy as np
        import pandas as pd

        if isinstance(dtype, pd.StringDtype):
            missingValue = dtype.na_value

            if missingValue is pd.NA:
                missingKind = "pd.NA"
            elif isinstance(missingValue, (float, np.floating)) and np.isnan(missingValue):
                missingKind = "nan"
            else:
                raise ValueError("Unsupported eventsID string missing-value representation")

            return {"kind": "string", "storage": dtype.storage, "na_value": missingKind}

        if pd.api.types.is_object_dtype(dtype):
            return {"kind": "object"}

        return {"kind": "native"}

    @staticmethod
    def _eventsIDDtypeFromRecord(record):
        """Decode a restricted dtype record without importing arbitrary types."""
        import numpy as np
        import pandas as pd

        if not isinstance(record, dict):
            raise ValueError("Invalid eventsID dtype metadata")

        if record == {"kind": "native"}:
            return None

        if record == {"kind": "object"}:
            return np.dtype("object")

        if set(record) != {"kind", "storage", "na_value"} or record["kind"] != "string":
            raise ValueError("Unsupported eventsID dtype metadata")

        if not isinstance(record["storage"], str):
            raise ValueError("Invalid eventsID string storage metadata")

        if record["na_value"] not in ("nan", "pd.NA"):
            raise ValueError("Unsupported eventsID string missing-value representation")

        missingValue = np.nan if record["na_value"] == "nan" else pd.NA
        return pd.StringDtype(storage=record["storage"], na_value=missingValue)

    @classmethod
    def _decodeEventsID(cls, content):
        """Decode eventsID and restore explicit per-column and axis dtypes.

        New files record object/StringDtype distinctions independently for
        every data column, the row index, and the column-name index.

        Other dtype representations remain governed by pandas/Parquet and
        must pass strict save-time roundtrip validation.
        """
        import json
        import pandas as pd
        import pyarrow.parquet as pq
        from io import BytesIO

        parquetMetadata = pq.read_schema(BytesIO(content)).metadata or {}
        encodedSchema = parquetMetadata.get(b"pgl.eventsID.schema.v2")

        if encodedSchema is None:
            # Compatibility with the earlier column-index-only metadata.
            legacyMetadata = parquetMetadata.get(b"pgl.eventsID.columns.v1")

            if legacyMetadata is None:
                return pd.read_parquet(BytesIO(content), engine="pyarrow")

            specification = json.loads(legacyMetadata.decode("utf-8"))
            columnsDtype = cls._eventsIDDtypeFromRecord(specification)

            if columnsDtype is None:
                raise ValueError("Invalid legacy eventsID column-index metadata")

            with pd.option_context("future.infer_string", False):
                restored = pd.read_parquet(BytesIO(content), engine="pyarrow")
                restored.columns = pd.Index(restored.columns.tolist(), dtype=columnsDtype, name=restored.columns.name)

            return restored

        schema = json.loads(encodedSchema.decode("utf-8"))

        if not isinstance(schema, dict) or set(schema) != {"version", "data", "index", "columns"}:
            raise ValueError("Invalid eventsID schema metadata")

        if type(schema["version"]) is not int or schema["version"] != 2:
            raise ValueError("Unsupported eventsID schema version")

        if not isinstance(schema["data"], list):
            raise ValueError("Invalid eventsID data-column metadata")

        dataDtypes = [cls._eventsIDDtypeFromRecord(record) for record in schema["data"]]
        indexDtype = cls._eventsIDDtypeFromRecord(schema["index"])
        columnsDtype = cls._eventsIDDtypeFromRecord(schema["columns"])

        with pd.option_context("future.infer_string", False):
            restored = pd.read_parquet(BytesIO(content), engine="pyarrow")

            if isinstance(restored.index, pd.MultiIndex) or isinstance(restored.columns, pd.MultiIndex):
                raise ValueError("eventsID schema v2 does not support MultiIndex axes")

            if len(dataDtypes) != len(restored.columns):
                raise ValueError("eventsID schema does not match the number of data columns")

            # Positional replacement avoids ambiguity and installs the actual
            # restored dtype rather than assigning into an existing dtype.
            for position, dtype in enumerate(dataDtypes):
                if dtype is not None:
                    restored.isetitem(position, restored.iloc[:, position].astype(dtype))

            if indexDtype is not None:
                restored.index = pd.Index(restored.index.tolist(), dtype=indexDtype, name=restored.index.name)

            if columnsDtype is not None:
                restored.columns = pd.Index(restored.columns.tolist(), dtype=columnsDtype, name=restored.columns.name)

        return restored

    @classmethod
    def saveEventsID(cls, eventsID, dataPath, filesystem=None):
        """Save eventsID with explicit string/object dtype metadata.

        Records dtype distinctions for data columns and both axes.
        Does not modify the caller's DataFrame.

        Other representations must survive the native Parquet roundtrip.
        Rejects unsupported/lossy results before writing the payload bundle.
        """
        import json
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from io import BytesIO

        if not isinstance(eventsID, pd.DataFrame):
            raise TypeError("eventsID must be a pandas DataFrame")

        if isinstance(eventsID.index, pd.MultiIndex) or isinstance(eventsID.columns, pd.MultiIndex):
            raise ValueError("eventsID schema v2 does not support MultiIndex axes")

        if not eventsID.columns.is_unique or not all(isinstance(name, str) for name in eventsID.columns):
            raise ValueError("eventsID requires unique string column names")

        if eventsID.attrs:
            raise ValueError("eventsID.attrs persistence is not implemented; refusing to omit it")

        schema = {
            "version": 2,
            "data": [cls._eventsIDDtypeRecord(dtype) for dtype in eventsID.dtypes],
            "index": cls._eventsIDDtypeRecord(eventsID.index.dtype),
            "columns": cls._eventsIDDtypeRecord(eventsID.columns.dtype),
        }

        # Keep pandas' metadata for native dtypes and index reconstruction.
        # Add the explicit string/object distinctions pandas may otherwise infer.
        table = pa.Table.from_pandas(eventsID, preserve_index=None)
        metadata = dict(table.schema.metadata or {})
        metadata[b"pgl.eventsID.schema.v2"] = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        table = table.replace_schema_metadata(metadata)

        buffer = BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        content = buffer.getvalue()

        # Validate using precisely the same decoder as normal loading.
        restored = cls._decodeEventsID(content)

        try:
            pd.testing.assert_frame_equal(eventsID, restored, check_dtype=True, check_index_type=True, check_column_type=True, check_exact=True, check_categorical=True, check_flags=True)
        except AssertionError as error:
            raise ValueError(f"eventsID cannot be preserved exactly by this Parquet codec: {error}") from error

        return cls._saveBytesBundle(content, dataPath, filesystem, kind="eventsID", formatName="parquet", entrypoint="eventsID.parquet")

    @classmethod
    def loadEventsID(cls, backing):
        """Read verified eventsID bytes and restore the recorded schema."""
        with backing.open("rb") as source:
            content = source.read()

        return cls._decodeEventsID(content)
    
    @classmethod
    def _saveAnalysisObject(cls, value, dataPath, filesystem, *, kind):
        """Encode a copied MNE analysis object, then write a verified bundle."""
        from copy import deepcopy
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import mne

        specifications = {
            "evoked": (mne.Evoked, "data-ave.fif", "fif"),
            "ica": (mne.preprocessing.ICA, "data-ica.fif", "fif"),
            "report": (mne.Report, "report.h5", "hdf5"),
        }
        if kind not in specifications: raise ValueError(f"Unknown MNE analysis kind: {kind}")

        expectedClass, filename, formatName = specifications[kind]
        if not isinstance(value, expectedClass):
            raise TypeError(f"{kind} must be an instance of {expectedClass.__name__}")

        with TemporaryDirectory(prefix="pgl-mne-analysis-write-") as directory:
            path = Path(directory) / filename
            working = deepcopy(value) if kind == "report" else value.copy()

            if kind == "evoked":
                mne.write_evokeds(str(path), working, overwrite=False, verbose=False)
            elif kind == "ica":
                working.save(str(path), overwrite=False, verbose=False)
            else:
                working.save(str(path), overwrite=False, open_browser=False, verbose=False)

            if not path.is_file(): raise RuntimeError(f"MNE did not create {filename}")
            content = path.read_bytes()

        return cls._saveBytesBundle(
            content, dataPath, filesystem,
            kind=kind, formatName=formatName, entrypoint=filename,
        )

    @staticmethod
    def _loadAnalysisObject(backing, *, kind):
        """Read through backing storage, then eagerly reconstruct a native object."""
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import mne

        filenames = {
            "evoked": "data-ave.fif",
            "ica": "data-ica.fif",
            "report": "report.h5",
        }
        if kind not in filenames: raise ValueError(f"Unknown MNE analysis kind: {kind}")

        with backing.open("rb") as source: content = source.read()

        with TemporaryDirectory(prefix="pgl-mne-analysis-read-") as directory:
            path = Path(directory) / filenames[kind]
            path.write_bytes(content)

            if kind == "evoked":
                values = mne.read_evokeds(str(path), condition=None, proj=False, verbose=False)
                if len(values) != 1: raise ValueError("Expected exactly one Evoked object")
                return values[0]

            if kind == "ica":
                return mne.preprocessing.read_ica(str(path), verbose=False)

            return mne.open_report(str(path))

    @classmethod
    def saveEvoked(cls, value, dataPath, filesystem=None):
        return cls._saveAnalysisObject(value, dataPath, filesystem, kind="evoked")

    @classmethod
    def loadEvoked(cls, backing):
        return cls._loadAnalysisObject(backing, kind="evoked")

    @classmethod
    def saveICA(cls, value, dataPath, filesystem=None):
        return cls._saveAnalysisObject(value, dataPath, filesystem, kind="ica")

    @classmethod
    def loadICA(cls, backing):
        return cls._loadAnalysisObject(backing, kind="ica")

    @classmethod
    def saveReport(cls, value, dataPath, filesystem=None):
        return cls._saveAnalysisObject(value, dataPath, filesystem, kind="report")

    @classmethod
    def loadReport(cls, backing):
        return cls._loadAnalysisObject(backing, kind="report")

class pglMNERawsBacking:
    """Ordered backing references for individual Raw objects; opens nothing."""

    def __init__(self, backings):
        self.backings = tuple(backings)


class pglMNERawAliasBacking:
    """The current Raw refers to an entry in the owner's raws collection."""

    def __init__(self, index):
        if type(index) is not int or index < 0:
            raise ValueError("Raw alias index must be a nonnegative integer")
        self.index = index

##################################
# pglSession
##################################
class pglSession(pglActionable):

    # filesystem, name and prefix for where the session is loaded from
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    
    # List of all runs
    runs = List(Instance(pglRun), allow_none=True, help="List of all runs")
    
    # mne MEG/EEG data
    mne = Instance(pglMNE, allow_none=True, help="MEG/EEG data in the form of an MNE variable")
    
    def __init__(self, filesystem=None, filesystemPrefix='', runList=[]):
        '''
        Init
        
        Args:
            fullDataPath (str): path to data for session
            filesystem: filesystem where path exists (None for local)
            filesystemPrefix: Prefix like ssh://
            runList: List of paths to runs
        '''
        super().__init__()
        
        self.filesystem = filesystem                
        self.filesystemPrefix = filesystemPrefix
        
        for runPath in runList:
            self.runs.append(pglRun(fullDataPath=runPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix))
            
    def add(self, data):
        '''
        add data to the pglSession, works on known types pglMNE...
        '''
        if isinstance(data,pglMNE): self.mne = data
        else:
            pglMessages.warning(f"Unknown data type: {type(data).__name__}")
            
    def get(self, dataName):
        '''
        Gets the called for data
        '''
        
        # MNE RAW
        if dataName.lower() == "mneraw":
            mne = self.get("mne")
            if mne: return (mne.raw)
            else: None
        # MNE
        elif dataName.lower() == "mne":
            if self.mne:
                return self.mne
            else:
                pglMessages.warning(f"No mne data loaded",level=1)
                return None
        else:
            pglMessages.warning(f"Unkown data type: {dataName}")
            return None
        
    def saveRawSession(self, *, settings=None, settingsName=None, backendID=None, storageOptions=None, sslRootCert=None):
        """Save this session as a parentless acquisition/root checkpoint.

        Supports behavioral-only, MNE-only, and mixed sessions.

        Behavioral runs archive saved acquisition files, not in-memory edits.
        Attached MNE objects are serialized from their current contents into
        a portable bundle. Original vendor files are not automatically archived.

        Uses current or supplied settings and the configured storage destination.

        Returns the completed checkpoint ID, or None when configuration is
        unavailable or connection/password entry is cancelled.
        """
        from .pglStorage import pglStorage

        return pglStorage.saveRawSession(self, settings=settings, settingsName=settingsName, backendID=backendID, storageOptions=storageOptions, sslRootCert=sslRootCert)
    
    def print(self, verbose=False):
        """Print a session overview followed by each run."""
        pglMessages.printHeader("Session")

        protocol = getattr(self.filesystem, "protocol", None)
        if isinstance(protocol, (tuple, list)):
            protocol = ", ".join(protocol)

        information = {
            "Runs": len(self.runs),
            "Filesystem": protocol or "Not set",
            "MEG/EEG data": "Attached" if self.mne is not None else "None",
        }

        labelWidth = max(len(label) for label in information)

        for label, value in information.items():
            pglMessages.print(f"  {label:<{labelWidth}} : {value}")

        for index, run in enumerate(self.runs):
            pglMessages.print()
            pglMessages.printHeader(f"Run {index + 1}", fillChar="-")
            run.print(verbose=verbose)
            
    def display(self):
        """Display every run and its tasks, returning the created figures."""
        if not self.runs:
            pglMessages.warning("No runs have been loaded.")
            return []

        figures = []

        for index, run in enumerate(self.runs):
            fig = run.display()

            if fig is not None:
                fig.suptitle(f"Run {index + 1}")
                figures.append(fig)

        return figures
 
    @classmethod
    def loadSession(cls, sessionID, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, includeDeleted=False, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load a completed checkpoint with deferred backed fields.

        For MNE checkpoints, cacheDirectory overrides settings.cacheDir.
        The default cacheDir is ~/data/pglCache, independent of dataPath.
        Cache directories are created only when materialization is needed;
        successfully materialized files are not automatically deleted.

        directLocalPolicy='immutable' permits verified direct-local access
        under the caller's source-file immutability/lifetime commitment.

        Cache-path, payload, and contract-validation errors propagate.
        includeDeleted permits inspection without undeleting the checkpoint.
        """
        from .pglStorage import pglStorage

        return pglStorage.loadSession(sessionID, settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert, includeDeleted=includeDeleted, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

    def saveSession(
        self,
        *,
        parentSessionID,
        settings=None,
        settingsName=None,
        backendID=None,
        description=None,
        ingestMethod="manual",
        storageOptions=None,
        sslRootCert=None,
    ):
        """Save current session contents and history as a new checkpoint.

        Requires an explicit completed, active parentSessionID.
        Uses current or supplied settings and the configured storage backend.

        Returns the completed checkpoint ID, or None for unavailable
        configuration or connection cancellation.

        Unlike saveRawSession(), this saves current contents and cached edits.
        """
        from .pglStorage import pglStorage

        return pglStorage.saveSession(
            self,
            parentSessionID=parentSessionID,
            settings=settings,
            settingsName=settingsName,
            backendID=backendID,
            description=description,
            ingestMethod=ingestMethod,
            storageOptions=storageOptions,
            sslRootCert=sslRootCert,
        )
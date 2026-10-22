
################################################################
#   filename: pglActionsFixes.py
#    purpose: actions that fix datasets - these should be one-off
#             fixes that are just required for datasests to
#             be migrated to new formats, and not needed
#             for any new sessions.
#         by: JLG
#       date: Oct 12, 2026
################################################################

from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from .pglPipeline import pglAction
from traitlets import HasTraits, Enum, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from .pglExperiment import pglEventSegment, pglEventVolumeTrigger
from .pglSession import pglTaskBase
from .pglStorage import pglStorage
from .pglPipeline import pglActionStatus
import json
import posixpath

class pglActionsFixes:

    class migrateTaskDirectories(pglAction):
        """Migrate all runs in a session to taskNN_taskName directories."""

        dryRun = Bool(default_value=True, help="Preview without modifying files")
        verbose = Bool(default_value=False, help="Verbose information printed")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.version = "1.0"

        def _run(self, session=None):
            """Validate every run, then optionally migrate directories and task IDs.

            Inspects resolved lazy fields without loading legacy task directories.

            dryRun=True:
                Reads source files and reports the plan.
                Does not modify source files or task caches.

            dryRun=False:
                Renames task directories, backs up/replaces task settings, and
                synchronizes taskID on already loaded task objects.

            Only for writable legacy acquisition directories, not immutable
            database checkpoint sources. Execution is not transactional.
            """
            if session is None:
                raise ValueError("No session was supplied.")

            plan = []

            # Read disk directly: legacy task directories cannot yet be loaded
            # through the current task-directory convention.
            for runIndex, run in enumerate(session.runs or []):
                filesystem, runPath, _ = pglStorage.validateFilesystem(filesystem=run.filesystem, dataPath=run.fullDataPath, filesystemPrefix=run.filesystemPrefix)

                if filesystem is None:
                    raise ValueError(f"Could not resolve filesystem for run {runIndex + 1}: {run.fullDataPath!r}")

                experimentPath = posixpath.join(runPath, "experimentSettings.json")

                with filesystem.open(experimentPath, "r") as stream:
                    experimentSettings = json.load(stream)

                taskNames = experimentSettings.get("tasks")

                if not isinstance(taskNames, list) or not all(isinstance(name, str) and name for name in taskNames):
                    raise ValueError(f"Expected a list of nonempty task names in {experimentPath!r}")

                # Read only resolved settings. Pending fields may still have
                # initialization values in _trait_values; those are not authoritative.
                loadedExperimentSettings = None

                if not run.lazyIsPending("experimentSettings"):
                    loadedExperimentSettings = run._trait_values.get("experimentSettings")

                if loadedExperimentSettings is not None and list(loadedExperimentSettings.tasks) != taskNames:
                    raise ValueError(f"In-memory task names differ from disk for {runPath!r}. Resolve those edits before migration.")

                # Inspect the authoritative full list only when it is resolved.
                # Individually loaded tasks remain available through _taskCache.
                loadedTasks = None

                if not run.lazyIsPending("tasks"):
                    loadedTasks = run._trait_values.get("tasks")

                if loadedTasks is not None and len(loadedTasks) != len(taskNames):
                    raise ValueError(f"Cached task count differs from disk for {runPath!r}")

                usedSources = set()

                for taskIndex, taskName in enumerate(taskNames):
                    if "/" in taskName or "\\" in taskName or taskName in (".", ".."):
                        raise ValueError(f"Unsafe task directory name: {taskName!r}")

                    legacyName = taskName if taskIndex == 0 else f"{taskName}Phase{taskIndex:02d}"
                    targetName = pglTaskBase.getTaskDirectoryName(taskID=taskIndex, taskName=taskName)
                    legacyPath = posixpath.join(runPath, legacyName)
                    targetPath = posixpath.join(runPath, targetName)
                    legacyExists = filesystem.exists(legacyPath)
                    targetExists = filesystem.exists(targetPath)

                    if legacyExists and targetExists:
                        raise FileExistsError(f"Both legacy and new task paths exist: {legacyPath!r} and {targetPath!r}. Nothing was changed.")

                    if not legacyExists and not targetExists:
                        raise FileNotFoundError(f"Neither {legacyPath!r} nor {targetPath!r} exists. Nothing was changed.")

                    sourcePath = legacyPath if legacyExists else targetPath

                    if not filesystem.isdir(sourcePath):
                        raise ValueError(f"Expected a task directory: {sourcePath!r}")

                    if sourcePath in usedSources:
                        raise ValueError(f"Multiple tasks map to {sourcePath!r}")

                    usedSources.add(sourcePath)
                    settingsPath = posixpath.join(sourcePath, "settings.json")

                    with filesystem.open(settingsPath, "rb") as stream:
                        originalBytes = stream.read()

                    taskSettings = json.loads(originalBytes)

                    if taskSettings.get("taskSaveName") != taskName:
                        raise ValueError(f"taskSaveName in {settingsPath!r} does not match {taskName!r}")

                    expectedTaskID = taskIndex + 1
                    oldTaskID = taskSettings.get("taskID")
                    updateTaskID = type(oldTaskID) is not int or oldTaskID != expectedTaskID
                    backupName = "settings.json.beforeTaskDirectoryMigration"
                    backupPath = posixpath.join(sourcePath, backupName)

                    if updateTaskID and filesystem.exists(backupPath):
                        with filesystem.open(backupPath, "rb") as stream:
                            backupBytes = stream.read()

                        if backupBytes != originalBytes:
                            raise ValueError(f"Existing backup differs from current settings: {backupPath!r}. Inspect before retrying.")

                    updatedBytes = None

                    if updateTaskID:
                        taskSettings["taskID"] = expectedTaskID
                        updatedBytes = (json.dumps(taskSettings, indent=4, ensure_ascii=False, allow_nan=True) + "\n").encode("utf-8")

                    # Preserve individually loaded and fully resolved task objects.
                    # No access to run.tasks occurs here.
                    cachedTasks = []
                    cachedTask = run._taskCache.get(taskIndex)

                    if cachedTask is not None:
                        cachedTasks.append(cachedTask)

                    if loadedTasks is not None:
                        loadedTask = loadedTasks[taskIndex]

                        if loadedTask is not None and all(loadedTask is not task for task in cachedTasks):
                            cachedTasks.append(loadedTask)

                    for task in cachedTasks:
                        if task.settings is None or task.settings.taskSaveName != taskName:
                            raise ValueError(f"Cached task {taskIndex} does not match disk for {runPath!r}")

                    change = {
                        "runIndex": runIndex,
                        "runPath": runPath,
                        "taskIndex": taskIndex,
                        "taskName": taskName,
                        "source": sourcePath,
                        "destination": targetPath,
                        "rename": sourcePath != targetPath,
                        "oldTaskID": oldTaskID,
                        "newTaskID": expectedTaskID,
                        "updateTaskID": updateTaskID,
                    }

                    plan.append((run, filesystem, change, originalBytes, updatedBytes, backupName, cachedTasks))

            self.changes = [entry[2] for entry in plan]

            for _, _, change, _, _, _, _ in plan:
                operations = []

                if change["rename"]:
                    operations.append(f"rename {posixpath.basename(change['source'])} -> {posixpath.basename(change['destination'])}")

                if change["updateTaskID"]:
                    operations.append(f"taskID {change['oldTaskID']!r} -> {change['newTaskID']}")

                description = "; ".join(operations) if operations else "already migrated"

                if operations or self.verbose:
                    pglMessages.print(f"{change['runPath']} | Task {change['taskIndex'] + 1}: {description}")

            if self.dryRun:
                pglMessages.message("Preview only: no files, directories, or task caches were changed.")
                return session

            for run, filesystem, change, originalBytes, updatedBytes, backupName, cachedTasks in plan:
                sourcePath = change["source"]
                targetPath = change["destination"]

                # Recheck the settings immediately before modifying this task.
                with filesystem.open(posixpath.join(sourcePath, "settings.json"), "rb") as stream:
                    if stream.read() != originalBytes:
                        raise ValueError(f"Settings changed after validation: {sourcePath!r}")

                if change["rename"]:
                    if filesystem.exists(targetPath):
                        raise FileExistsError(f"Destination appeared after validation: {targetPath!r}")

                    filesystem.mv(sourcePath, targetPath, recursive=True)

                if change["updateTaskID"]:
                    settingsPath = posixpath.join(targetPath, "settings.json")
                    backupPath = posixpath.join(targetPath, backupName)

                    if not filesystem.exists(backupPath):
                        with filesystem.open(backupPath, "wb") as stream:
                            stream.write(originalBytes)

                    with filesystem.open(settingsPath, "wb") as stream:
                        stream.write(updatedBytes)

                # Preserve loaded objects and other edits; synchronize only taskID.
                for task in cachedTasks:
                    task.settings.taskID = change["newTaskID"]

                # Remove failed-load entries left by older getTaskAt versions.
                if change["taskIndex"] in run._taskCache and run._taskCache[change["taskIndex"]] is None:
                    del run._taskCache[change["taskIndex"]]

            pglMessages.message("Task directory migration complete for all runs in the session.")
            return session
    
    
    class migrateParameterDirectories(pglAction):
        """Move flat task parameter bundles into their named subdirectories.

        Preview may inspect checkpoint-backed sessions.

        Execution supports ordinary local acquisition directories only. The
        caller must own these writable sources; registered checkpoint payloads
        must never be passed through a different filesystem to bypass protection.

        No JSON contents are rewritten. Execution is not transactional.
        """

        dryRun = Bool(default_value=True, help="Preview without modifying files")
        verbose = Bool(default_value=False, help="Print tasks without legacy parameter bundles")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.version = "1.0"

        def _run(self, session=None):
            """Validate all candidate bundles, then optionally relocate their files."""
            import json
            import posixpath
            from fsspec.implementations.local import LocalFileSystem

            if session is None:
                raise ValueError("No session was supplied.")

            if not self.dryRun and session.sessionID is not None:
                raise ValueError("Do not modify saved checkpoint sources. Create a separate session over writable acquisition directories or exported copies.")

            componentNames = ("settings.json", "state.json", "data.json")
            plan = []
            seenSources = set()

            for runIndex, run in enumerate(session.runs or []):
                filesystem = run.filesystem
                runPath = run.fullDataPath

                if filesystem is None or not runPath:
                    raise ValueError(f"Run {runIndex}: no explicit filesystem/source path.")

                if not self.dryRun:
                    # Deliberately exclude checkpoint wrappers and remote filesystems.
                    if type(filesystem) is not LocalFileSystem:
                        raise ValueError(f"Run {runIndex}: execution currently requires an ordinary LocalFileSystem.")

                    # Use a freshly constructed acquisition session. Do not silently
                    # discard loaded task objects or their possible working edits.
                    loadedTasks = None if run.lazyIsPending("tasks") else run._trait_values.get("tasks")

                    if loadedTasks is not None or run._taskCache:
                        raise ValueError(f"Run {runIndex}: tasks are already resolved/cached. Create a fresh session over the acquisition directories before migration.")

                experimentPath = posixpath.join(runPath, "experimentSettings.json")

                with filesystem.open(experimentPath, "rb") as stream:
                    experimentSettings = json.loads(stream.read())

                if not isinstance(experimentSettings, dict):
                    raise ValueError(f"Expected an object in {experimentPath!r}.")

                taskNames = experimentSettings.get("tasks")

                if not isinstance(taskNames, list) or not all(isinstance(name, str) and name for name in taskNames):
                    raise ValueError(f"Expected nonempty task names in {experimentPath!r}.")

                for taskIndex, taskName in enumerate(taskNames):
                    if taskName in (".", "..") or any(character in taskName for character in ("/", "\\", "\x00")):
                        raise ValueError(f"Unsafe task name: {taskName!r}")

                    # Accept either directory convention, but never guess between
                    # two existing copies of the task.
                    currentName = pglTaskBase.getTaskDirectoryName(taskID=taskIndex, taskName=taskName)
                    legacyName = taskName if taskIndex == 0 else f"{taskName}Phase{taskIndex:02d}"
                    candidatePaths = list(dict.fromkeys(posixpath.join(runPath, name) for name in (currentName, legacyName)))
                    existingPaths = [path for path in candidatePaths if filesystem.exists(path)]

                    if len(existingPaths) != 1:
                        raise ValueError(f"Run {runIndex}, task {taskIndex}: expected exactly one task directory; found {existingPaths!r}.")

                    taskPath = existingPaths[0]

                    if not filesystem.isdir(taskPath):
                        raise ValueError(f"Expected a task directory: {taskPath!r}")

                    parameterRoot = posixpath.join(taskPath, "parameters")

                    if not filesystem.isdir(parameterRoot):
                        raise ValueError(f"Missing parameter directory: {parameterRoot!r}")

                    componentPaths = {name: posixpath.join(parameterRoot, name) for name in componentNames}
                    existingComponents = [name for name, path in componentPaths.items() if filesystem.exists(path)]

                    if not existingComponents:
                        if self.verbose:
                            pglMessages.print(f"{taskPath}: no flat parameter bundle")
                        continue

                    if len(existingComponents) != len(componentNames):
                        raise ValueError(f"Incomplete flat parameter bundle at {parameterRoot!r}: found {existingComponents!r}.")

                    contents = {}

                    for componentName, componentPath in componentPaths.items():
                        if not filesystem.isfile(componentPath):
                            raise ValueError(f"Expected a component file: {componentPath!r}")

                        with filesystem.open(componentPath, "rb") as stream:
                            contents[componentName] = stream.read()

                        # Validate JSON structure without instantiating saved classes.
                        decoded = json.loads(contents[componentName])

                        if not isinstance(decoded, dict):
                            raise ValueError(f"Expected a JSON object in {componentPath!r}.")

                    parameterSettings = json.loads(contents["settings.json"])

                    if parameterSettings.get("__class__") != "pglParameterSettings":
                        raise ValueError(f"Unrecognized flat settings record at {parameterRoot!r}; refusing to infer a parameter bundle.")

                    parameterName = parameterSettings.get("name")

                    if not isinstance(parameterName, str) or not parameterName.strip():
                        raise ValueError(f"Missing parameter name at {parameterRoot!r}.")

                    if parameterName in (".", "..") or any(character in parameterName for character in ("/", "\\", "\x00")):
                        raise ValueError(f"Unsafe parameter name: {parameterName!r}")

                    destination = posixpath.join(parameterRoot, parameterName)

                    if filesystem.exists(destination):
                        raise FileExistsError(f"Parameter destination already exists: {destination!r}. Inspect rather than merge automatically.")

                    # Avoid planning the same source twice when runs are duplicated.
                    sourceKey = (id(filesystem), parameterRoot)

                    if sourceKey in seenSources:
                        raise ValueError(f"Duplicate migration source: {parameterRoot!r}")

                    seenSources.add(sourceKey)

                    change = {
                        "runIndex": runIndex,
                        "taskIndex": taskIndex,
                        "taskName": taskName,
                        "parameterName": parameterName,
                        "source": parameterRoot,
                        "destination": destination,
                        "files": list(componentNames),
                    }

                    plan.append((filesystem, change, contents))

            self.changes = [change for _, change, _ in plan]

            for _, change, _ in plan:
                pglMessages.print(f"{change['source']} -> {change['destination']} ({', '.join(componentNames)})")

            if self.dryRun:
                pglMessages.message(f"Preview only: {len(plan)} flat parameter bundle(s); no source files or task caches changed.")
                return session

            # Recheck all planned sources before beginning any filesystem changes.
            for filesystem, change, contents in plan:
                if filesystem.exists(change["destination"]):
                    raise FileExistsError(f"Destination appeared after validation: {change['destination']!r}")

                for componentName, expectedBytes in contents.items():
                    sourcePath = posixpath.join(change["source"], componentName)

                    with filesystem.open(sourcePath, "rb") as stream:
                        if stream.read() != expectedBytes:
                            raise ValueError(f"Source changed after validation: {sourcePath!r}")

            for filesystem, change, contents in plan:
                destination = change["destination"]
                filesystem.makedirs(destination, exist_ok=False)

                # Preserve exact bytes. A failure may leave a partially moved
                # bundle; subsequent preflight refuses to merge it silently.
                for componentName, expectedBytes in contents.items():
                    sourcePath = posixpath.join(change["source"], componentName)
                    targetPath = posixpath.join(destination, componentName)
                    filesystem.mv(sourcePath, targetPath)

                    with filesystem.open(targetPath, "rb") as stream:
                        if stream.read() != expectedBytes:
                            raise ValueError(f"Migration verification failed: {targetPath!r}")

            pglMessages.message(f"Parameter-directory migration complete: {len(plan)} bundle(s).")
            return session
    ##################################################################
    # class pglActionRecreateExperimentDataFromTasks
    ##################################################################
    class pglActionRecreateExperimentDataFromTasks(pglAction):
        """
        Recreate experiment start/end times and volume events from task data.

        This is a fixer for sessions that were run when pglExperimentData was not being saved correctly

        Updates the session in memory; does not save files.
        """

        class Settings(pglTraitSettings):

            class ChooseRun(pglTraitSettings):

                class ChooseTaskName(pglTraitSettings):
                    taskName = List(Unicode(), default_value=[], help="Task in run", visible=False)
                    taskIndex = Int(help="Zero-based task index", visible=False)

                runName = Unicode(help="Name of run", visible=False)
                taskNames = List(Instance(ChooseTaskName), default_value=[], settingsListKey="taskName", traitDisplayName="Select task", multiSelect=True, maxRowsVisible=2, help="Select exactly one task for this run")

            TR = Float(1.0, help="TR used for frame acquisition")
            nVols = Int(0, help="Number of volumes; 0 reconstructs through the end of the task")
            runList = List(Instance(ChooseRun), default_value=[], settingsListKey="runName", traitDisplayName="Run", help="Runs to reconstruct")
            taskIndexList = List(Int(), default_value=[], help="Selected task index for each run", visible=False)

        settings = Instance(Settings, allow_none=True, default_value=None, help="Reconstruction settings")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.version = "1.0"
            self.session = None
            self._configuredRuns = []

        def configure(self, session):
            """Choose a TR and exactly one source task for each run."""
            import math
            import posixpath
            from .pglDialog import pglDialogs
            from .pglPipeline import pglActionStatus

            self.status = pglActionStatus.INITIALIZED
            self.error = None
            self.session = session
            self._configuredRuns = list(session.runs)
            self.settings = self.Settings()

            for run in self._configuredRuns:
                experimentSettings = run.experimentSettings
                if experimentSettings is None:
                    pglMessages.warning(f"Cannot configure reconstruction: experiment settings could not be loaded for {run.fullDataPath!r}.")
                    return

                if not experimentSettings.tasks:
                    pglMessages.warning(f"Cannot configure reconstruction: no tasks are configured for {run.fullDataPath!r}.")
                    return

                chooseRun = self.Settings.ChooseRun()
                chooseRun.runName = posixpath.basename(run.fullDataPath.rstrip("/"))

                for taskIndex, taskName in enumerate(experimentSettings.tasks):
                    chooseTask = self.Settings.ChooseRun.ChooseTaskName()
                    chooseTask.taskName = [f"{taskIndex + 1}: {taskName}"]
                    chooseTask.taskIndex = taskIndex
                    chooseTask.isSelected = taskIndex == 0
                    chooseRun.taskNames.append(chooseTask)

                self.settings.runList.append(chooseRun)

            if not self._configuredRuns:
                pglMessages.warning("Cannot configure reconstruction: the session has no runs.")
                return

            self.settings = pglDialogs.traitsDialog(self.settings)
            if self.settings is None:
                pglMessages.message("Experiment data reconstruction cancelled.")
                return

            if not math.isfinite(self.settings.TR) or self.settings.TR <= 0:
                pglMessages.warning("Cannot configure reconstruction: TR must be finite and greater than zero.")
                return

            if self.settings.nVols < 0:
                pglMessages.warning("Cannot configure reconstruction: nVols must be zero or greater.")
                return

            if len(self.settings.runList) != len(self._configuredRuns):
                pglMessages.warning("Cannot configure reconstruction: the run selection does not match the session.")
                return

            selectedIndices = []
            for chooseRun in self.settings.runList:
                selectedTasks = [task for task in chooseRun.taskNames if task.isSelected]
                if len(selectedTasks) != 1:
                    pglMessages.warning(f"Select exactly one task for run {chooseRun.runName!r}; found {len(selectedTasks)} selected tasks.")
                    return
                selectedIndices.append(selectedTasks[0].taskIndex)

            self.settings.taskIndexList = selectedIndices
            self.status = pglActionStatus.CONFIGURED

        def _makeVolumeEvents(self, startTime, stopTime, TR, currentVolumeNum):
            """Generate evenly spaced triggers between two inclusive endpoints."""
            import math

            if not all(math.isfinite(value) for value in (startTime, stopTime, TR)):
                raise ValueError("Trigger endpoints and TR must be finite.")

            if TR <= 0:
                raise ValueError("TR must be greater than zero.")

            duration = stopTime - startTime
            if duration < 0:
                raise ValueError(f"Trigger interval ends before it starts: {startTime} -> {stopTime}")

            # A segment can begin exactly at the preceding trigger.
            if duration == 0:
                return [startTime]

            nTRs = round(duration / TR)
            if nTRs == 0:
                raise ValueError(f"Trigger interval {duration:.6f}s is too short for TR={TR:.6f}s; cannot infer a positive number of intervals.")

            actualTR = duration / nTRs
            slop = actualTR - TR
            if abs(slop) > 0.1 * TR:
                pglMessages.warning(f"Warning for {nTRs} volumes beginning at {currentVolumeNum}: spacing requires {slop:.3f}s of slop ({100 * abs(slop) / TR:.1f}% of TR)", level=1)

            times = [startTime + i * actualTR for i in range(nTRs + 1)]
            times[0] = startTime
            times[-1] = stopTime
            return times

        def _run(self):
            """Build replacements for all runs before modifying experiment data."""
            import math
            import numpy as np

            if self.session is None or self.settings is None:
                raise ValueError("The action has not been configured with a session and settings.")

            runs = list(self.session.runs)
            if len(runs) != len(self._configuredRuns) or any(run is not configuredRun for run, configuredRun in zip(runs, self._configuredRuns)):
                raise ValueError("Session runs changed after configuration; configure the action again.")

            if len(self.settings.taskIndexList) != len(runs):
                raise ValueError("Expected exactly one selected task index per run.")

            if not math.isfinite(self.settings.TR) or self.settings.TR <= 0:
                raise ValueError("TR must be finite and greater than zero.")

            if self.settings.nVols < 0:
                raise ValueError("nVols must be zero or greater.")

            replacements = []

            for runIndex, run in enumerate(runs):
                taskIndex = self.settings.taskIndexList[runIndex]
                task = run.getTaskAt(taskIndex)

                if task is None or task.settings is None or task.data is None:
                    raise ValueError(f"Cannot reconstruct {run.fullDataPath!r}: task {taskIndex} is missing settings or data.")

                runData = run.data
                runSettings = run.settings
                if runData is None:
                    raise ValueError(f"Cannot reconstruct {run.fullDataPath!r}: experiment data could not be loaded.")
                if runSettings is None:
                    raise ValueError(f"Cannot reconstruct {run.fullDataPath!r}: run settings could not be loaded.")

                if not runSettings.startOnVolumeTrigger:
                    pglMessages.warning(f"{run.fullDataPath}: run did not start on a volume trigger; alignment of volumes to task is not guaranteed.")

                taskStartTime = task.data.startTime
                taskEndTime = task.data.endTime
                if not all(math.isfinite(value) for value in (taskStartTime, taskEndTime)) or taskEndTime < taskStartTime:
                    raise ValueError(f"Invalid task start/end times for {run.fullDataPath!r}, task {taskIndex}.")

                waitUntilVolumeTriggerSegments = [i for i, value in enumerate(task.settings.waitUntilVolumeTrigger) if value]
                segments = [event for event in task.data.events if isinstance(event, pglEventSegment) and event.segmentNum in waitUntilVolumeTriggerSegments]

                volumeTriggers = []
                startTime = taskStartTime

                for segment in segments:
                    volumeTriggers += self._makeVolumeEvents(startTime, segment.timestamp, self.settings.TR, len(set(volumeTriggers)))
                    startTime = segment.timestamp

                if self.settings.nVols > 0:
                    endTime = taskStartTime + self.settings.nVols * self.settings.TR
                else:
                    endTime = taskEndTime

                # Preserve the original tail-generation rule.
                if endTime - startTime > self.settings.TR:
                    endTime = startTime + round((endTime - startTime) / self.settings.TR) * self.settings.TR
                    volumeTriggers += self._makeVolumeEvents(startTime, endTime, self.settings.TR, len(set(volumeTriggers)))

                volumeTriggers = sorted(set(volumeTriggers))
                if self.settings.nVols > 0:
                    volumeTriggers = volumeTriggers[:self.settings.nVols]

                if not volumeTriggers:
                    raise ValueError(f"No volume triggers could be reconstructed for {run.fullDataPath!r}; existing experiment data was not changed.")

                if self.settings.nVols > 0 and len(volumeTriggers) != self.settings.nVols:
                    raise ValueError(f"Reconstructed {len(volumeTriggers)} triggers for {run.fullDataPath!r}, but requested {self.settings.nVols}; existing experiment data was not changed.")

                differences = np.diff(volumeTriggers)
                meanText = f"{np.mean(differences):.3f}" if len(differences) else "n/a"
                stdText = f"{np.std(differences, ddof=1):.3f}" if len(differences) > 1 else "n/a"
                pglMessages.message(f"{run.fullDataPath} | nTriggers: {len(volumeTriggers)} Mean: {meanText}, SD: {stdText}")

                events = [event for event in runData.events if not isinstance(event, pglEventVolumeTrigger)]
                for triggerTime in volumeTriggers:
                    event = pglEventVolumeTrigger()
                    event.timestamp = triggerTime
                    events.append(event)

                events.sort(key=lambda event: event.timestamp)
                replacements.append((runData, taskStartTime, taskEndTime, events))

            # Apply only after every run's replacement has been constructed.
            for runData, startTime, endTime, events in replacements:
                runData.startTime = startTime
                runData.endTime = endTime
                runData.events = events

            return self.session
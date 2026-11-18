
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
from .pglPipeline import pglAction, pglActionCollection
from traitlets import HasTraits, Enum, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from .pglExperiment import pglEventSegment, pglEventVolumeTrigger
from .pglSession import pglTaskBase
from .pglStorage import pglStorage
from .pglPipeline import pglActionStatus
import json
import posixpath
from .pglSession import pglSession
import numpy as np

class pglActionsFixes(pglActionCollection):

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
        
    ################################################################
    # Fix behavioral MRI volume-trigger times
    ################################################################
    class fixVolumeTimes(pglAction):
        """Repair canonical volume-trigger events independently in each run.

        Estimates volume time from the median recorded interval.

        Automatic repairs:
            addMissing inserts inferred missing triggers between retained anchors.
            adjustTiming corrects isolated timing errors when predictions from
            both original neighboring triggers agree sufficiently closely.
            Persistent timing shifts are not propagated or regularized.

        A final gap greater than 1.5 times the median causes its final trigger
        to be removed as a suspected manual ending, unless the gap is within
        100 ms of twice the median. A near-double final gap permits one missing
        trigger to be inserted when addMissing is enabled.
        Automatic ending removal is independent of addMissing/adjustTiming.

        Boundary edits follow automatic repairs:
            Positive startVolumes/endVolumes add evenly spaced triggers.
            Negative values remove triggers from that boundary.

        With nVols:
            If neither boundary edit is supplied, adjust the end to reach
            nVols. Otherwise require the explicitly edited count to match.
            Explicit zero is a supplied boundary edit.

        Replaces selected run.data objects only after all plans validate.
        Original acquisition files and acquisition state remain unchanged.
        Non-volume events retain their objects and relative ordering.
        Repair history is stored on data.volumeTimingRepairs.

        Boundary additions/removals are not idempotent: running the action
        again applies those requested edits again.
        """

        runIndices = List(Int(), default_value=None, allow_none=True, help="Zero-based run indices; None selects all behavioral runs.")
        discrepancyMs = Float(20.0, min=0.0, help="Minimum discrepancy in milliseconds for automatic correction.")
        shortIntervalMs = Float(5.0, min=0.0, help="Warn when consecutive recorded volume triggers are closer than this many milliseconds. Zero disables the warning.")
        addMissing = Bool(True, help="Insert inferred missing volume-trigger events.")
        adjustTiming = Bool(True, help="Correct early or late volume-trigger timestamps.")
        nVols = Int(default_value=None, allow_none=True, min=0, help="Required final number of volumes in each selected run.")
        startVolumes = Int(default_value=None, allow_none=True, help="Add volumes at the beginning; negative values remove volumes.")
        endVolumes = Int(default_value=None, allow_none=True, help="Add volumes at the end; negative values remove volumes.")

        ################################################################
        # Configure repair settings
        ################################################################
        def configure(self, **kwargs):
            from .pglPipeline import pglActionStatus

            self.status = pglActionStatus.INITIALIZED
            self.error = None

            # Accept a single zero-based run index or a collection of indices.
            if "runIndices" in kwargs and kwargs["runIndices"] is not None:
                runIndices = kwargs["runIndices"]
                if type(runIndices) is int: runIndices = [runIndices]
                if not isinstance(runIndices, (list, tuple)) or any(type(index) is not int for index in runIndices):
                    raise TypeError("runIndices must be an integer, a list/tuple of integers, or None.")
                kwargs["runIndices"] = list(runIndices)

            # Reject Boolean counts before traitlets can interpret them as integers.
            for name in ("nVols", "startVolumes", "endVolumes"):
                if name in kwargs and kwargs[name] is not None and type(kwargs[name]) is not int:
                    raise TypeError(f"{name} must be an integer or None.")

            if "discrepancyMs" in kwargs and isinstance(kwargs["discrepancyMs"], bool):
                raise TypeError("discrepancyMs must be a number, not Boolean.")

            for name in ("addMissing", "adjustTiming"):
                if name in kwargs and type(kwargs[name]) is not bool:
                    raise TypeError(f"{name} must be Boolean.")

            self.configureTraits(**kwargs)
            self._validateOptions()
            super().configure()

        ################################################################
        # Warn about unusually close recorded volume triggers
        ################################################################
        def _warnShortVolumeIntervals(self, times, runIndex):
            import math

            if not math.isfinite(self.shortIntervalMs) or self.shortIntervalMs < 0:
                raise ValueError("shortIntervalMs must be finite and nonnegative.")
            if self.shortIntervalMs == 0: return

            # Inspect original intervals, not timestamps generated by repairs.
            for index in range(1, len(times)):
                intervalMs = (times[index] - times[index - 1]) * 1000.0
                if 0 <= intervalMs < self.shortIntervalMs:
                    pglMessages.warning(f"Run {runIndex}: triggers {index} → {index + 1} are only {intervalMs:.3f} ms apart (warning threshold: {self.shortIntervalMs:g} ms). Possible duplicate or spurious trigger; review before relying on automatic repairs.", level=1)
                            
        ################################################################
        # Validate settings again at execution
        ################################################################
        def _validateOptions(self):
            import math

            if not math.isfinite(self.discrepancyMs) or self.discrepancyMs < 0:
                raise ValueError("discrepancyMs must be finite and nonnegative.")

            if self.runIndices is not None:
                if not self.runIndices: raise ValueError("runIndices must not be empty; use None for all runs.")
                if any(type(index) is not int or index < 0 for index in self.runIndices):
                    raise ValueError("runIndices must contain nonnegative integer indices.")
                if len(set(self.runIndices)) != len(self.runIndices):
                    raise ValueError("runIndices must not contain duplicates.")

            for name in ("nVols", "startVolumes", "endVolumes"):
                value = getattr(self, name)
                if value is not None and type(value) is not int:
                    raise TypeError(f"{name} must be an integer or None.")

            if self.nVols is not None and self.nVols < 0:
                raise ValueError("nVols must be nonnegative.")

        ################################################################
        # Merge repaired volumes without reordering other recorded events
        ################################################################
        @staticmethod
        def _mergeVolumeEvents(originalEvents, volumeEvents):
            import math
            from numbers import Real

            mergedEvents = []
            volumeIndex = 0

            # Source events may arrive out of timestamp order. Preserve their
            # relative order rather than globally sorting the acquisition log.
            for event in originalEvents:
                if event.type == "volumeTrigger": continue
                timestamp = getattr(event, "timestamp", None)

                if isinstance(timestamp, Real) and math.isfinite(float(timestamp)):
                    while volumeIndex < len(volumeEvents) and volumeEvents[volumeIndex].timestamp <= timestamp:
                        mergedEvents.append(volumeEvents[volumeIndex])
                        volumeIndex += 1

                mergedEvents.append(event)

            mergedEvents.extend(volumeEvents[volumeIndex:])
            return mergedEvents

        ################################################################
        # Prepare one run without changing its current contents
        ################################################################
        def _prepareRun(self, run, runIndex):
            import math
            from copy import deepcopy
            from numbers import Real
            from .pglExperiment import pglEventVolumeTrigger

            data = run.data
            if data is None: raise ValueError(f"Run {runIndex}: experiment data is unavailable.")

            originalEvents = list(data.events)
            volumeRecords = [(index, event) for index, event in enumerate(originalEvents) if event.type == "volumeTrigger"]
            if len(volumeRecords) < 2:
                raise ValueError(f"Run {runIndex}: at least two volume triggers are required to estimate volume time.")

            times = []
            for eventIndex, event in volumeRecords:
                timestamp = event.timestamp
                if isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(float(timestamp)):
                    raise ValueError(f"Run {runIndex}: invalid volume timestamp at event index {eventIndex}.")
                times.append(float(timestamp))

            # Inspect original intervals before changing any timestamps.
            self._warnShortVolumeIntervals(times, runIndex)

            intervals = np.diff(times)
            if not np.isfinite(intervals).all() or np.any(intervals <= 0):
                raise ValueError(f"Run {runIndex}: volume intervals must be finite and strictly positive.")

            volumeTime = float(np.median(intervals))
            threshold = self.discrepancyMs / 1000.0
            changes = []
            volumes = []
            correctedTimes = list(times)
            correctedEvents = [deepcopy(event) for _, event in volumeRecords]

            # Use the smaller integer at an exact half-interval tie.
            def nearestMultiple(value):
                return max(1, int(math.ceil(value - 0.5)))

            def worthFixing(error):
                return abs(error) > 1e-9 and abs(error) >= threshold - 1e-9

            def addVolume(timestamp, reason):
                timestamp = float(timestamp)
                if not math.isfinite(timestamp): raise ValueError(f"Run {runIndex}: generated a nonfinite timestamp.")
                event = pglEventVolumeTrigger(timestamp=timestamp)
                changes.append({"operation": "insert", "reason": reason, "timestamp": timestamp})
                return event

            # Classify the original final gap before correcting its neighbors.
            # Near-double gaps permit one missing volume; other long final gaps
            # are treated as ending with a manually supplied trigger.
            finalInterval = float(intervals[-1])
            finalMultiple = nearestMultiple(finalInterval / volumeTime)
            closeToDouble = finalMultiple == 2 and abs(finalInterval - 2 * volumeTime) <= 0.100 + 1e-9
            removeFinalTrigger = finalInterval > 1.5 * volumeTime and not closeToDouble

            ################################################################
            # Identify isolated timing errors using original neighbors
            ################################################################
            candidates = {}

            # Both neighbors must predict approximately the same timestamp.
            # Cap their permitted disagreement at 10% of the median TR.
            supportTolerance = max(1e-9, min(threshold, 0.1 * volumeTime))

            if self.adjustTiming:
                for position in range(1, len(times) - 1):
                    # Do not use a suspected manual ending as a timing anchor.
                    if removeFinalTrigger and position + 1 == len(times) - 1: continue

                    leftInterval = times[position] - times[position - 1]
                    rightInterval = times[position + 1] - times[position]

                    # Very short intervals may indicate duplicate/spurious events.
                    if min(leftInterval, rightInterval) * 1000.0 < self.shortIntervalMs: continue

                    leftMultiple = nearestMultiple(leftInterval / volumeTime)
                    rightMultiple = nearestMultiple(rightInterval / volumeTime)
                    leftError = leftInterval - leftMultiple * volumeTime
                    rightError = rightInterval - rightMultiple * volumeTime

                    # An isolated late trigger makes a long interval followed
                    # by a short one; an early trigger produces the reverse.
                    if leftError * rightError >= 0: continue
                    if not worthFixing(leftError) or not worthFixing(rightError): continue

                    expectedFromLeft = times[position - 1] + leftMultiple * volumeTime
                    expectedFromRight = times[position + 1] - rightMultiple * volumeTime
                    if abs(expectedFromLeft - expectedFromRight) > supportTolerance + 1e-9: continue

                    expectedTime = (expectedFromLeft + expectedFromRight) / 2.0
                    if not worthFixing(times[position] - expectedTime): continue
                    if not times[position - 1] < expectedTime < times[position + 1]: continue

                    candidates[position] = {
                        "timestamp": expectedTime,
                        "leftTimestamp": times[position - 1],
                        "rightTimestamp": times[position + 1],
                        "expectedFromLeft": expectedFromLeft,
                        "expectedFromRight": expectedFromRight,
                    }

            # Adjacent candidates depend on anchors that would themselves move.
            # Retain them rather than arbitrarily choosing which trigger is wrong.
            ambiguousPositions = {
                position for position in candidates
                if position - 1 in candidates or position + 1 in candidates
            }

            for position, candidate in candidates.items():
                if position in ambiguousPositions: continue

                expectedTime = candidate["timestamp"]
                correctedTimes[position] = expectedTime
                correctedEvents[position].timestamp = expectedTime
                changes.append({
                    "operation": "adjust",
                    "originalEventIndex": volumeRecords[position][0],
                    "originalTriggerNumber": position + 1,
                    "originalTimestamp": times[position],
                    "timestamp": expectedTime,
                    "leftTimestamp": candidate["leftTimestamp"],
                    "rightTimestamp": candidate["rightTimestamp"],
                    "expectedFromLeft": candidate["expectedFromLeft"],
                    "expectedFromRight": candidate["expectedFromRight"],
                    "shiftMs": (expectedTime - times[position]) * 1000.0,
                })

            ################################################################
            # Infer missing events and remove a suspected manual ending
            ################################################################
            unresolvedIntervals = []
            volumes.append(correctedEvents[0])

            for position in range(1, len(times)):
                isFinal = position == len(times) - 1

                # Removal is independent of addMissing and adjustTiming.
                # Never manufacture missing volumes across this ending gap.
                if isFinal and removeFinalTrigger:
                    changes.append({
                        "operation": "remove",
                        "reason": "suspectedManualEnd",
                        "originalEventIndex": volumeRecords[position][0],
                        "originalTriggerNumber": position + 1,
                        "timestamp": times[position],
                        "precedingTimestamp": times[position - 1],
                        "interval": finalInterval,
                        "expectedInterval": volumeTime,
                    })
                    continue

                leftTime = correctedTimes[position - 1]
                rightTime = correctedTimes[position]
                interval = rightTime - leftTime
                if interval <= 0:
                    raise ValueError(f"Run {runIndex}: isolated corrections produced overlapping trigger times.")

                multiple = nearestMultiple(interval / volumeTime)
                nMissing = multiple - 1
                timingError = interval - multiple * volumeTime

                # Internal gaps can contain multiple missing volumes.
                # The final gap permits only one, supported by the original gap.
                allowMissing = not isFinal or (nMissing == 1 and closeToDouble)

                if self.addMissing and nMissing and allowMissing and worthFixing(interval - volumeTime):
                    # Interpolate between retained anchors without shifting
                    # the subsequent recorded trigger sequence.
                    for missingIndex in range(1, multiple):
                        missingTime = leftTime + interval * missingIndex / multiple
                        volumes.append(addVolume(missingTime, "missing"))

                if worthFixing(timingError):
                    unresolvedIntervals.append({
                        "firstTrigger": position,
                        "secondTrigger": position + 1,
                        "interval": interval,
                        "expectedInterval": multiple * volumeTime,
                        "errorMs": timingError * 1000.0,
                    })

                volumes.append(correctedEvents[position])

            ################################################################
            # Apply explicit boundary changes after automatic repairs
            ################################################################
            def editBoundary(count, atStart, reason):
                nonlocal volumes
                if not count: return

                if count < 0:
                    nRemove = -count
                    if nRemove > len(volumes):
                        raise ValueError(f"Run {runIndex}: {reason} removes {nRemove} volumes but only {len(volumes)} remain.")

                    removed = volumes[:nRemove] if atStart else volumes[-nRemove:]
                    for event in removed:
                        changes.append({"operation": "remove", "reason": reason, "timestamp": float(event.timestamp)})

                    volumes = volumes[nRemove:] if atStart else volumes[:-nRemove]
                    return

                if not volumes:
                    raise ValueError(f"Run {runIndex}: cannot add boundary volumes without a remaining trigger to anchor them.")

                boundaryTime = float(volumes[0].timestamp if atStart else volumes[-1].timestamp)
                if atStart:
                    added = [addVolume(boundaryTime - step * volumeTime, reason) for step in range(count, 0, -1)]
                    volumes = added + volumes
                else:
                    added = [addVolume(boundaryTime + step * volumeTime, reason) for step in range(1, count + 1)]
                    volumes.extend(added)

            editBoundary(self.startVolumes, True, "startVolumes")
            editBoundary(self.endVolumes, False, "endVolumes")

            if self.nVols is not None:
                explicitBoundary = self.startVolumes is not None or self.endVolumes is not None

                if explicitBoundary and len(volumes) != self.nVols:
                    raise ValueError(f"Run {runIndex}: explicit boundary edits produce {len(volumes)} volumes, conflicting with nVols={self.nVols}.")

                if not explicitBoundary:
                    editBoundary(self.nVols - len(volumes), False, "nVols")

            finalTimes = np.asarray([event.timestamp for event in volumes], dtype=float)
            if not np.isfinite(finalTimes).all() or np.any(np.diff(finalTimes) <= 0):
                raise ValueError(f"Run {runIndex}: requested repairs produce overlapping or reversed volume timestamps.")

            report = {
                "runIndex": runIndex,
                "method": "isolatedNeighborSupport",
                "volumeTime": volumeTime,
                "anchorTime": times[0],
                "originalCount": len(volumeRecords),
                "finalCount": len(volumes),
                "removedFinalTrigger": removeFinalTrigger,
                "finalRecordedInterval": finalInterval,
                "discrepancyMs": self.discrepancyMs,
                "shortIntervalMs": self.shortIntervalMs,
                "supportToleranceMs": supportTolerance * 1000.0,
                "addMissing": self.addMissing,
                "adjustTiming": self.adjustTiming,
                "nVols": self.nVols,
                "startVolumes": self.startVolumes,
                "endVolumes": self.endVolumes,
                "ambiguousTriggers": [position + 1 for position in sorted(ambiguousPositions)],
                "unresolvedIntervals": unresolvedIntervals,
                "changes": changes,
            }

            # No-op runs retain their original data object and event identities.
            # Changed runs receive a replacement only after all runs validate.
            replacement = data
            if changes:
                history = getattr(data, "volumeTimingRepairs", [])
                if not isinstance(history, list):
                    raise ValueError(f"Run {runIndex}: volumeTimingRepairs must be a list.")

                replacement = deepcopy(data)
                replacement.events = self._mergeVolumeEvents(originalEvents, volumes)
                replacement.volumeTimingRepairs = deepcopy(history) + [report]

            return {"run": run, "original": data, "replacement": replacement, "report": report}

        ################################################################
        # Report the expected timing reference and planned changes
        ################################################################
        @staticmethod
        def _printPlan(report):
            changes = report["changes"]
            nInserted = sum(change["operation"] == "insert" for change in changes)
            nAdjusted = sum(change["operation"] == "adjust" for change in changes)
            nRemoved = sum(change["operation"] == "remove" for change in changes)

            pglMessages.message(f"Run {report['runIndex']}: median volume time={report['volumeTime']:.6f} s; correction threshold={report['discrepancyMs']:g} ms.")

            if report["adjustTiming"]:
                pglMessages.message(f"  Align isolated triggers to the midpoint of their original neighbors' median-TR predictions, only when those predictions agree within {report['supportToleranceMs']:g} ms.")
            else:
                pglMessages.message("  Recorded-trigger timestamp adjustment is disabled.")

            if report["addMissing"]:
                pglMessages.message("  Infer missing-trigger counts from median-TR multiples; interpolate their timestamps between retained anchors.")
            else:
                pglMessages.message("  Missing-trigger insertion is disabled.")

            pglMessages.message("  Persistent timing shifts are not propagated or regularized.")
            pglMessages.message("  Remove a final trigger when its original gap exceeds 1.5 × median TR, unless the gap is within 100 ms of 2 × median TR.")
            pglMessages.message(f"  Volumes: {report['originalCount']} → {report['finalCount']}; insert={nInserted}, adjust={nAdjusted}, remove={nRemoved}.")

            if report["originalCount"] == 2:
                pglMessages.warning("Only one recorded interval is available; missing, mistimed, or manual ending triggers cannot be independently established.", level=1)

            if report["removedFinalTrigger"]:
                finalInterval = report["finalRecordedInterval"]
                pglMessages.message(f"  Suspected manual ending: original final gap={finalInterval:.6f} s ({finalInterval / report['volumeTime']:.3f} × median TR).")
                pglMessages.message("  Plan removes that final trigger without filling its gap. Explicit boundary/count edits apply afterward.")

            if report["ambiguousTriggers"]:
                pglMessages.warning(f"Run {report['runIndex']}: adjacent correction candidates were retained because their anchors are ambiguous. Original trigger numbers: {report['ambiguousTriggers']}.", level=1)

            unresolved = report["unresolvedIntervals"]
            if unresolved:
                reason = "insufficient support for an isolated correction" if report["adjustTiming"] else "timestamp adjustment disabled"
                pglMessages.warning(f"Run {report['runIndex']}: {len(unresolved)} interval timing discrepancies remain before boundary edits ({reason}). These were not fixed and may still be flagged by print()/display().", level=1)

                for record in unresolved:
                    pglMessages.message(f"  Retained trigger interval {record['firstTrigger']} → {record['secondTrigger']}: {record['interval']:.6f} s; expected={record['expectedInterval']:.6f} s; error={record['errorMs']:+.3f} ms.")

            for change in changes:
                if change["operation"] == "adjust":
                    pglMessages.message(f"  Adjust trigger {change['originalTriggerNumber']}: {change['originalTimestamp']:.6f} → {change['timestamp']:.6f} s ({change['shiftMs']:+.1f} ms).")
                    pglMessages.message(f"    Neighbor predictions: {change['expectedFromLeft']:.6f} s from preceding trigger, {change['expectedFromRight']:.6f} s from following trigger; using their midpoint.")
                elif change["operation"] == "remove" and change["reason"] == "suspectedManualEnd":
                    pglMessages.message(f"  Remove original trigger {change['originalTriggerNumber']} at {change['timestamp']:.6f} s (suspected manual ending; preceding gap={change['interval']:.6f} s).")
                else:
                    pglMessages.message(f"  {change['operation'].capitalize()} volume at {change['timestamp']:.6f} s ({change['reason']}).")

          
        ################################################################
        # Validate all runs before installing repaired data
        ################################################################
        def _run(self, session: pglSession) -> pglSession:
            self._validateOptions()

            if not isinstance(session, pglSession):
                self.setError("fixVolumeTimes requires a pglSession.")
                return None

            runs = list(session.runs or [])
            if not runs:
                self.setError("Session has no behavioral runs.")
                return None

            indices = list(range(len(runs))) if self.runIndices is None else list(self.runIndices)
            if any(index >= len(runs) for index in indices):
                self.setError(f"runIndices is outside this session's {len(runs)} runs.")
                return None

            # Existing behavioral matrices are snapshots, not live event views.
            # Refuse rather than silently leave one inconsistent with repaired events.
            if "behavior" in session.dataMatrices:
                self.setError("Fix volume times before constructing the session's behavior matrix. An attached behavior matrix would become stale.")
                return None

            plans = []
            try:
                for runIndex in indices:
                    plans.append(self._prepareRun(runs[runIndex], runIndex))

                # Aliased run/data objects would make independent edits ambiguous.
                if len({id(plan["run"]) for plan in plans}) != len(plans):
                    raise ValueError("Selected runs contain duplicate run objects.")
                if len({id(plan["original"]) for plan in plans}) != len(plans):
                    raise ValueError("Selected runs share the same experiment data object.")

            except (ValueError, TypeError) as error:
                self.setError(f"Volume repair refused: {error}")
                return None

            for plan in plans: self._printPlan(plan["report"])

            # Restore original data references if an assignment/observer fails.
            # This cannot roll back arbitrary external observer side effects.
            installed = []
            try:
                for plan in plans:
                    if plan["replacement"] is plan["original"]: continue
                    installed.append(plan)
                    plan["run"].data = plan["replacement"]

            except BaseException as error:
                for plan in reversed(installed):
                    try:
                        plan["run"].data = plan["original"]
                    except BaseException as rollbackError:
                        if hasattr(error, "add_note"): error.add_note(f"Could not restore run data: {rollbackError}")
                raise

            pglMessages.message("Volume-time repair complete. Changes are in memory; use saveSession() to publish a derived checkpoint.")
            return session
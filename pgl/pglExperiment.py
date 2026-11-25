################################################################
#   filename: pglExperiment.py
#    purpose: Experiment class which handles timing, parameter randomization
#             subject response, synchronizing with measurement hardware, as well
#             as saving experimental data
#         by: JLG
#       date: September 2, 2025
################################################################

#############
# Import modules
#############
from datetime import date as Date, datetime
from IPython.display import clear_output
import numpy as np
import random
import math
from dataclasses import dataclass, field
from pathlib import Path
from IPython.display import display, HTML
import ipywidgets as widgets
from traitlets import Float, TraitError, TraitError, observe, Instance, Int, Unicode, Dict, validate, Bool, Tuple
from .pglParameter import pglParameter, pglParameterBlock
from .pglEvent import pglEvent
from .pglSerialize import pglSerialize
from traitlets import List
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import numpy.typing as npt
from enum import Enum
from .pglTimestamp import pglTimestamp
from .pglSettings import pglDisplaySettings, pglTraitSettings, pglStateDataSettings
from .pglMessages import pglMessages
import fsspec
import posixpath
from .pglStorage import pglStorage
from .pglDialog import pglDialogs
from types import SimpleNamespace
from .pglDevice import pglDigitalIODevice
from .pglPlots import pglTimelinePlot

#######################
# for returning stats
#######################
@dataclass
class Stats:
    mean: float
    median: float
    std: float
    min: float
    max: float
    last: float

##############################################s
# Experiment base class
##############################################
class pglExperimentBase(pglStateDataSettings):
    '''
    Base class for pglExperiment which runs experiments
    and pglExperimentAnalysis which is used for loading and
    analyzing experimental data. This class handles loading
    and saving settings, state and data.
    '''
    def __init__(self):
        # initialize variables
        self.settings = None
        self.state = None
        self.data = None
        self.experimentSettings = None
        self.pgl = None
        self.eyeTracker = None
        self.nTasks = 0
    
    @classmethod
    def isValidExperimentPath(cls, verbose=False, fullDataPath=None, settings=None,
                            settingsName=None, experimentName=None, subjectID=None,
                            sessionName=None, runName=None, filesystem=None,
                            filesystemPrefix=None, dataPath=None):
        """Check required experiment/task files, allowing an optional recording."""
        from .pglChoose import pglChoose

        filesystem, fullDataPath, _ = pglChoose.getExperimentPath(
            fullDataPath=fullDataPath,
            settings=settings,
            settingsName=settingsName,
            experimentName=experimentName,
            subjectID=subjectID,
            sessionName=sessionName,
            runName=runName,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
            dataPath=dataPath,
        )
        if filesystem is None: return False

        sep = filesystem.sep
        requiredExperimentFiles = ["data.json", "experimentSettings.json", "settings.json", "pgl.json", "state.json"]
        requiredFiles = ["data.json", "settings.json", "state.json"]

        # Check top-level required JSON files.
        for fileName in requiredExperimentFiles:
            filePath = f"{fullDataPath}{sep}{fileName}"
            if not filesystem.isfile(filePath):
                if verbose: pglMessages.message(f"Missing file: {filePath}")
                return False

        # Permit the explicitly named recording artifact alongside task dirs.
        for item in filesystem.ls(fullDataPath, detail=False):
            fileName = item.rsplit(sep, 1)[-1]
            if fileName.endswith(".json"): continue
            if fileName == "recording.h5" and filesystem.isfile(item): continue

            if not filesystem.isdir(item):
                if verbose: pglMessages.message(f"Unexpected non-directory artifact: {item}")
                return False

            # Check task files only if any task component exists.
            taskFilesFound = [name for name in requiredFiles if filesystem.isfile(f"{item}{sep}{name}")]
            if not taskFilesFound: continue

            for name in requiredFiles:
                filePath = f"{item}{sep}{name}"
                if not filesystem.isfile(filePath):
                    if verbose: pglMessages.message(f"Missing file in task directory: {filePath}")
                    return False

            # Parameters directory may be empty.
            parametersPath = f"{item}{sep}parameters"
            if not filesystem.isdir(parametersPath):
                if verbose: pglMessages.message(f"No parameters directory in {item}")
                return False

            for subItem in filesystem.ls(parametersPath, detail=False):
                if not filesystem.isdir(subItem): continue
                for name in requiredFiles:
                    filePath = f"{subItem}{sep}{name}"
                    if not filesystem.isfile(filePath):
                        if verbose: pglMessages.message(f"Missing parameter file: {filePath}")
                        return False

        return True

    @classmethod
    def load(cls, settings=None, dataPath=None, experimentName=None, subjectID=None, sessionName=None, runName=None, fullDataPath=None, filesystem=None, filesystemPrefix=None):
        '''
        Load the experiment settings, state and data.         
        '''
        # get the experiment directory
        from .pglChoose import pglChoose
        filesystem, experimentPath, filesystemPrefix = pglChoose.getExperimentPath(settings=settings, dataPath=dataPath, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, fullDataPath=fullDataPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix)
        if filesystem is None:
            if dataPath is not None:
                pglMessages.message(f"Could not locate experiment directory {dataPath}")
            return
 
        # call parent class to load the experiment data, settings, and state
        pglMessages.message(f"(pglExperiment:load) Loading experimentdata from: {experimentPath}")
        obj = super().load(dataPath=experimentPath, filesystem=filesystem,filesystemPrefix=filesystemPrefix, loadAsClass=cls)
        if obj is None:
            if experimentPath is not None:
                pglMessages.warning(f"Unable to load {experimentPath}")
            return None

        # keep where we were loaded from
        obj.filesystem = filesystem
        obj.fullDataPath = experimentPath
        obj.filesystemPrefix = filesystemPrefix
        
        # if we have
        # load experiment settings
        experimentSettingsPath = f"{experimentPath}{filesystem.sep}experimentSettings.json"
        if filesystem.exists(experimentSettingsPath):
            obj.experimentSettings = pglSerialize.load(experimentSettingsPath, filesystem=filesystem)
            if obj.experimentSettings is None:
                pglMessages.warning(f"Could not load experiment settings for {experimentSettingsPath}")
        else:
            pglMessages.warning(f"No experimentSettings.json found for: {experimentSettingsPath}",level=1)
        
        # load pgl state
        pglStatePath = f"{experimentPath}{filesystem.sep}pgl.json"
        if filesystem.exists(pglStatePath):
            obj.pglState = pglSerialize.load(pglStatePath, filesystem=filesystem)
            if obj.pglState is None:
                pglMessages.warning(f"Could not load pgl state for {pglStatePath}")
                return None
        else:
            pglMessages.warning(f"No pgl.json found for: {pglStatePath}", level=1)            
            
        # load all the tasks
        if obj.experimentSettings:
            # initialize tasks, this is only relevant for pglRun
            # becuase it lazy-loads tasks, and this will prevent
            # it from trying to do so
            if hasattr(obj,'_tasks'): obj._tasks = []
            
            for iTask, taskName in enumerate(obj.experimentSettings.tasks):
                # get the task directory
                taskDirName = pglTask.getTaskDirectoryName(iTask,taskName)
                taskPath = f"{experimentPath}{filesystem.sep}{taskDirName}"

                # load the task data
                if filesystem.isdir(taskPath):
                    # load the task
                    task = pglTask.load(dataPath=taskPath, filesystem=filesystem)
                
                    if task is None:
                        pglMessages.warning(f"Could not load task {taskName}", level=1)
                    else:
                        # add the task to the experiment
                        obj.addTask(task, addToTaskList=False)
                        pass
                else:
                    pglMessages.warning(f"Could not find task {taskName}: {taskPath}")
        
        # return the created object
        return obj
       
    def addTask(self, task, addToTaskList=True, addPhase=False):
        '''
        Add a task to the experiment.
        
        Args:
            addPhase (bool): If True, the task will be added as a new phase. If False, it will be added to the current phase.
        '''
        # give it a reference to pgl and experiment
        task.pgl = self.pgl
        task.e = self
        self.nTasks += 1
        task.settings.taskID = self.nTasks

        # if we want to add the task as a new phase
        if addPhase and self.tasks is not None:            
            maxPhaseNum = max(
                (task.settings.phaseNum for task in self.tasks
                if task.settings.phaseNum is not None),
                default=-1
            )
            task.settings.phaseNum = maxPhaseNum + 1

        # set whether to save eye tracker info
        if self.eyeTracker is not None:
            task.settings.saveEyeTracker = True

        # add the task
        self.tasks.append(task)
        
        # save in experimentSettings
        if addToTaskList and task.settings is not None:
            self.experimentSettings.tasks.append(task.settings.taskSaveName)

    def display(self):
        '''
        Display a timeline of experiment events.
        '''
        # display experiment data
        self.data.display(self)
        
        # display task data
        if hasattr(self, "tasks"):
            for task in self.tasks:
                task.display()   

    def print(self, verbose=True, discrepancyMs=100.0):
        '''
        Print a summary of the experiment events.
        Warn and skip summaries whose required contents could not be loaded.

        Report possible missing volumes and interval timing discrepancies
        in red. Suspected manual ending triggers are reported separately.
        Detailed intervals are printed when verbose=True.
        '''
        from .pglTimestamp import pglTimestamp

        timestamp = pglTimestamp()
        runPath = getattr(self, "fullDataPath", "(unknown run path)")

        # Use ANSI red without requiring a color argument in pglMessages.print().
        def printRed(message):
            pglMessages.print(f"\033[31m{message}\033[0m")

        # check required experiment contents before accessing their fields
        experimentSettings = self.experimentSettings
        if experimentSettings is None:
            pglMessages.warning(f"Skipping experiment summary for {runPath!r}: experiment settings could not be loaded.", level=1)
            return

        # print experiment identifiers even if the data is missing
        pglMessages.print(f"Experiment: {experimentSettings.experimentName} | Subject ID: {experimentSettings.subjectID}")
        pglMessages.print(f"Session: {experimentSettings.sessionName} | Run: {experimentSettings.runName}")

        data = self.data
        if data is None:
            pglMessages.warning(f"Skipping remaining experiment summary for {runPath!r}: experiment data could not be loaded.", level=1)
            return

        if data.startDateTime: pglMessages.print(f"{data.startDateTime}")

        diagnostics = data.getVolumeTimingDiagnostics(discrepancyMs=discrepancyMs)

        if diagnostics["valid"]:
            pglMessages.print(f"Duration: {timestamp.formatDuration(self.experimentDuration())}")
        else:
            printRed(f"MRI volume timing unavailable: {diagnostics['error']}")

        numVols = diagnostics["nVolumes"]
        if numVols != 0:
            pglMessages.print(f"Number of volume triggers: {numVols}")

        if diagnostics["median"] is not None:
            pglMessages.print(f"Median time between triggers: {diagnostics['median']:.3f}s")
            pglMessages.print(f"Mean ± std time between triggers: {diagnostics['mean']:.3f} ± {diagnostics['std']:.6f}s")

            manualFinal = diagnostics["suspectedManualFinal"]
            if manualFinal is not None:
                pglMessages.print(f"Suspected manual final trigger: trigger {manualFinal['secondTrigger']}, gap={manualFinal['interval']:.3f}s; excluded from missing-volume and timing-discrepancy counts.")
                pglMessages.print("Duration and interval statistics above still include this final trigger.")

            # List missing-volume gaps separately from timing errors.
            if diagnostics["missingIntervals"]:
                printRed(f"Possible missing volumes = {diagnostics['nMissingVolumes']}")
                if verbose:
                    for record in diagnostics["missingIntervals"]:
                        printRed(f"  Trigger {record['firstTrigger']} → {record['secondTrigger']}: interval={record['interval']:.3f}s; possibly {record['nMissing']} missing volume(s)")

            timingIntervals = diagnostics["timingIntervals"]
            if timingIntervals:
                nShort = sum(record["errorMs"] < 0 for record in timingIntervals)
                nLong = sum(record["errorMs"] > 0 for record in timingIntervals)
                printRed(f"Interval timing discrepancies (≥{discrepancyMs:g} ms) = {len(timingIntervals)}: {nShort} shorter, {nLong} longer than expected")
                pglMessages.print("These flag unusual intervals, not proven individual timestamp errors. Isolated-trigger repair may deliberately retain them.")

                if verbose:
                    for record in timingIntervals:
                        printRed(f"  Trigger {record['firstTrigger']} → {record['secondTrigger']}: interval={record['interval']:.3f}s; expected={record['expectedInterval']:.3f}s; difference={record['errorMs']:+.1f} ms")
            else:
                pglMessages.print(f"No interval timing discrepancies ≥{discrepancyMs:g} ms detected, excluding any suspected manual final trigger.")

            # With two triggers there is only one interval, so no independent
            # timing baseline exists against which to detect a delayed trigger.
            if numVols == 2:
                pglMessages.print("Only one MRI volume interval is available; missing or mistimed volumes cannot be assessed reliably.")

        # accessing the tasks property may trigger loading
        try:
            tasks = getattr(self, "tasks", [])
        except RuntimeError as error:
            pglMessages.warning(f"Skipping task summary for {runPath!r}: {error}", level=1)
            return

        if tasks is None:
            pglMessages.warning(f"Skipping task summary for {runPath!r}: tasks are unavailable.", level=1)
            return

        # print task data
        for task in tasks:
            pglMessages.print(f"taskName: {task.settings.taskName} (nTrials={task.data.nTrials})")
            if verbose:
                pglMessages.printHeader()
                task.print()
    
    def experimentDuration(self,data=None):
        '''
        Return the total time of the experiment in seconds.
        '''
        # work on passed in data or self data
        if data is None:
            data = self.data
        # check for no data
        if data is None or data.startTime is None or data.endTime is None:
            return 0
        # check to see if this has volumes recorded
        if data.getNumEvents(type="volumeTrigger") > 1:        
            # get the timestamps of the first and last volume triggers
            volumeTimestamps = [e.timestamp for e in data.events if e.type == "volumeTrigger"]
            volumeTR = np.median(np.diff(volumeTimestamps))
            # return the difference between the first and last timestamp
            # because the experiment type as recorded by endTime and startTIme
            # will typically record longer until the experimenter pressed the ESC key to end
            # add one TR to account for the last volume trigger
            return volumeTimestamps[-1] - volumeTimestamps[0] + volumeTR
        else:
            # return timestamps for end compared to start
            return data.endTime - data.startTime

    def getNearestVolumeTrigger(self, event=None, direction="nearest"):
        """Return a zero-based volume-trigger index for an event.

        direction:
            nearest: Closest timestamp; ties select the earlier trigger.
            before: Last trigger at or before the event.
            after: First trigger at or after the event.

        Returns None for a missing event, no triggers, or no trigger on the
        requested side.

        Indices refer to volumeTrigger events in recorded order, not positions
        in the complete events list. Trigger timestamps must increase strictly.

        A behavioral trigger index is not automatically an MRI array index:
        missing/extra triggers and discarded MRI volumes require explicit
        correspondence.
        """
        import math
        from bisect import bisect_left, bisect_right
        from numbers import Real

        if direction not in ("nearest", "before", "after"):
            raise ValueError("direction must be nearest, before, or after.")
        if event is None: return None

        def validTimestamp(value):
            return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and math.isfinite(float(value))

        timestamp = getattr(event, "timestamp", None)
        if not validTimestamp(timestamp):
            raise ValueError("Event timestamp must be a finite number.")

        data = self.data
        if data is None:
            raise ValueError("Experiment data is unavailable.")

        times = [entry.timestamp for entry in data.events if entry.type == "volumeTrigger"]
        if not times: return None

        if any(not validTimestamp(value) for value in times):
            raise ValueError("Volume-trigger timestamps must be finite numbers.")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("Volume-trigger timestamps must increase strictly in recorded order.")

        if direction == "before":
            index = bisect_right(times, timestamp) - 1
            return index if index >= 0 else None

        if direction == "after":
            index = bisect_left(times, timestamp)
            return index if index < len(times) else None

        index = bisect_left(times, timestamp)
        if index == 0: return 0
        if index == len(times): return len(times) - 1

        return index - 1 if timestamp - times[index - 1] <= times[index] - timestamp else index
    
    def getMainTask(self, *, verbose=False):
        """Return the explicitly marked or heuristically selected main task.

        Selection order:
            Exactly one task marked isMainTask: return it.
            Only one task available: return it.
            Otherwise: select the unique task with the most recorded STARTs.

        Multiple explicit main tasks, tied trial counts, missing event data,
        or no recorded trials warn and return None.

        Heuristic selection is reported. verbose additionally reports
        explicit and single-task selection. Task settings are never changed.

        Accessing self.tasks may lazy-load saved task metadata/events.
        Does not load stimulus pixels or MRI payloads.
        """
        from numbers import Integral

        if type(verbose) is not bool:
            raise TypeError("verbose must be Boolean.")

        tasks = list(getattr(self, "tasks", None) or [])

        if not tasks:
            pglMessages.warning("Cannot select a main task: no tasks are available.", level=1)
            return None

        def taskLabel(index, task):
            """Use the zero-based list index, not historical taskID values."""
            settings = getattr(task, "settings", None)
            name = getattr(settings, "taskSaveName", None) or getattr(settings, "taskName", None) or "(unnamed task)"
            name = str(name).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
            return f"task {index} ({name})"

        # Explicit intent takes precedence over every heuristic.
        mainTasks = [(index, task) for index, task in enumerate(tasks) if getattr(getattr(task, "settings", None), "isMainTask", False)]

        if len(mainTasks) == 1:
            index, task = mainTasks[0]
            if verbose: pglMessages.message(f"Selected {taskLabel(index, task)}: isMainTask=True.")
            return task

        if len(mainTasks) > 1:
            candidates = ", ".join(taskLabel(index, task) for index, task in mainTasks)
            pglMessages.warning(f"Multiple tasks have isMainTask=True: {candidates}. Select one explicitly with taskName.", level=1)
            return None

        if len(tasks) == 1:
            if verbose: pglMessages.message(f"Selected {taskLabel(0, tasks[0])}: the only task.")
            return tasks[0]

        # Count distinct recorded trial STARTs, not configured nTrials or the
        # runtime currentTrial counter. Duplicate starts must not inflate counts.
        counts = []

        for index, task in enumerate(tasks):
            data = getattr(task, "data", None)
            events = getattr(data, "events", None)

            if events is None:
                pglMessages.warning(f"Cannot infer the main task: {taskLabel(index, task)} has no available event data. Select taskName explicitly.", level=1)
                return None

            trialNums = set()

            for event in events:
                if not isinstance(event, pglEventTrial) or event.eventType != pglEventTrial.boundaryType.START.value: continue

                trialNum = event.trialNum
                if isinstance(trialNum, (bool, np.bool_)) or not isinstance(trialNum, Integral) or trialNum < 0:
                    pglMessages.warning(f"Cannot infer the main task: {taskLabel(index, task)} has an invalid recorded trial index {trialNum!r}.", level=1)
                    return None

                if trialNum in trialNums:
                    pglMessages.warning(f"Cannot infer the main task: {taskLabel(index, task)} has multiple START events for trial {trialNum}.", level=1)
                    return None

                trialNums.add(int(trialNum))

            counts.append(len(trialNums))

        maxTrials = max(counts)
        winners = [index for index, count in enumerate(counts) if count == maxTrials]

        if maxTrials == 0 or len(winners) != 1:
            summary = "; ".join(f"{taskLabel(index, task)}: {counts[index]} trials started" for index, task in enumerate(tasks))
            reason = "no recorded trial STARTs" if maxTrials == 0 else "tied highest trial counts"
            pglMessages.warning(f"Cannot infer the main task: {reason}. {summary}. Select taskName explicitly.", level=1)
            return None

        index = winners[0]
        pglMessages.message(f"Selected {taskLabel(index, tasks[index])}: most recorded trial STARTs ({maxTrials}); no task is marked main.")
        return tasks[index]

    def getTrialsInfo(self, parameterName=None, taskName=None, *, segmentNum=None, includeBlank=False, direction="nearest"):
        """Group recorded trial information by stimulus or parameter.

        parameterName=None groups by stimulus; otherwise groups by parameter.
        taskName=None uses getMainTask(). Explicit names match taskName first,
        then taskSaveName, case-insensitively.

        segmentNum=None uses trial START. A nonnegative segmentNum uses that
        segment's recorded START within each trial. Missing segment starts are
        omitted and reported; trial START is never substituted.

        Common result fields:
            taskName, taskSaveName, groupBy, segmentNum, direction
            groupValues, groupNames, nGroups
            trialNums: Zero-based behavioral trial indices per group.
            volumes: Zero-based behavioral trigger indices, or None.
            startTimes: Selected onset seconds relative to task start.
            onsetTimestamps: Selected onsets in the recorded clock.
            timeOrigin: Task start timestamp used for startTimes.
            nTrials: Included trial count per group.
            nTrialsTotal: Total included trials.
            omittedTrialNums: Trial indices grouped by omission reason.

        Parameter fields:
            parameterName, parameterValues, parameter

        Stimulus fields:
            stimulusNums, stimulusIds, stimulusNames
            sourceInfo, imageIndices, nStimuli

        Parameter groups preserve configured validValues order and empty groups.
        Stimulus groups include observed stimuli in inventory order, with the
        optional blank group last. Stimulus inventory indices are run-local;
        stimulus IDs/provenance must establish correspondence across runs.

        Planned but unstarted trials are excluded. Started but incomplete trials
        remain included when their selected onset and grouping data exist.
        Missing volume matches do not remove trials.

        Reads metadata/events, never stimulus pixels or MRI payloads.
        Behavioral trigger indices are not automatically MRI array indices.
        Recorded segment STARTs mark task transitions, not verified display
        onset times. This method does not infer stimulus durations.

        Selection failures warn and return None. Invalid arguments or
        inconsistent recorded data raise.
        """
        from copy import deepcopy
        from numbers import Integral, Real
        from types import SimpleNamespace

        from .pglParameter import pglParameterBlock
        from .pglImage import pglStimulusDatabase, blankStimulusNum

        if parameterName is not None and (not isinstance(parameterName, str) or not parameterName.strip()):
            raise ValueError("parameterName must be a nonblank string or None.")
        if taskName is not None and (not isinstance(taskName, str) or not taskName.strip()):
            raise ValueError("taskName must be a nonblank string or None.")

        if segmentNum is not None:
            if isinstance(segmentNum, (bool, np.bool_)) or not isinstance(segmentNum, Integral):
                raise TypeError("segmentNum must be an integer or None.")
            segmentNum = int(segmentNum)
            if segmentNum < 0:
                pglMessages.warning("segmentNum must be nonnegative; segment indices are zero-based.", level=1)
                return None

        if type(includeBlank) is not bool:
            raise TypeError("includeBlank must be Boolean.")
        if parameterName is not None and includeBlank:
            raise ValueError("includeBlank applies only to stimulus grouping.")
        if direction not in ("nearest", "before", "after"):
            raise ValueError("direction must be nearest, before, or after.")

        # Explicit task selection bypasses main-task inference.
        if taskName is None:
            task = self.getMainTask()
            if task is None: return None
        else:
            tasks = list(self.tasks or [])
            selectedName = taskName.casefold()
            matches = [task for task in tasks if task.settings.taskName.casefold() == selectedName]

            if not matches:
                matches = [task for task in tasks if task.settings.taskSaveName.casefold() == selectedName]

            if len(matches) != 1:
                available = ", ".join(task.settings.taskSaveName for task in tasks) or "(none)"
                pglMessages.warning(f"Expected one task matching {taskName!r}; found {len(matches)}. Available tasks: {available}", level=1)
                return None

            task = matches[0]

        if segmentNum is not None and segmentNum >= task.settings.nSegments:
            pglMessages.warning(f"segmentNum {segmentNum} is outside this task's {task.settings.nSegments} configured segments; indices are zero-based.", level=1)
            return None

        def validTimestamp(value):
            return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and math.isfinite(float(value))

        def validIndex(value):
            return not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral) and value >= 0

        def collectParameters(parameterList):
            """Flatten nested parameter blocks without changing their order."""
            parameters = []
            for parameter in parameterList:
                if isinstance(parameter, pglParameterBlock):
                    parameters.extend(collectParameters(parameter.settings.parameters))
                else:
                    parameters.append(parameter)
            return parameters

        def newGroup():
            return {"trialNums": [], "volumes": [], "startTimes": [], "onsetTimestamps": []}

        if task.data is None or not validTimestamp(task.data.startTime):
            raise ValueError("Task data requires a finite startTime.")

        parameter = None
        database = None
        validValues = []
        groups = {}

        if parameterName is not None:
            parameters = collectParameters(task.parameters)
            matches = [parameter for parameter in parameters if parameter.settings.name == parameterName]

            if len(matches) != 1:
                available = [parameter.settings.name for parameter in parameters]
                pglMessages.warning(f"Expected one parameter named {parameterName!r}; found {len(matches)}. Available parameters: {available}", level=1)
                return None

            parameter = matches[0]
            validValues = list(parameter.settings.validValues)

            # Ambiguous configured groups are invalid even if no trial has
            # yet used the duplicated value.
            for index, value in enumerate(validValues):
                if any(np.array_equal(value, previous) for previous in validValues[:index]):
                    raise ValueError("Parameter validValues contain ambiguous duplicate values.")

            groups = {index: newGroup() for index in range(len(validValues))}

        else:
            database = task.stimulusDatabase
            if not isinstance(database, pglStimulusDatabase):
                pglMessages.warning("The selected task has no pglStimulusDatabase.", level=1)
                return None

        # Index recorded START events once. Never accidentally use a trial END.
        # Segment events have no trialNum: associate them through event order,
        # not timestamp sorting, since trial/segment starts may share timestamps.
        starts, segmentStarts = {}, {}
        activeTrial = None
        lastBoundaryTime = None

        for event in task.data.events:
            if isinstance(event, pglEventTrial):
                if event.eventType == pglEventTrial.boundaryType.START.value:
                    if not validIndex(event.trialNum):
                        raise ValueError("Recorded trial indices must be nonnegative integers.")

                    trialIndex = int(event.trialNum)
                    if trialIndex in starts:
                        raise ValueError(f"Trial {trialIndex} has multiple START events.")
                    if not validTimestamp(event.timestamp):
                        raise ValueError(f"Trial {trialIndex} has an invalid START timestamp.")

                    if segmentNum is not None:
                        if lastBoundaryTime is not None and event.timestamp < lastBoundaryTime:
                            raise ValueError("Trial/segment boundary timestamps decrease in recorded event order.")
                        lastBoundaryTime = event.timestamp

                    starts[trialIndex] = event
                    activeTrial = trialIndex

                elif event.eventType == pglEventTrial.boundaryType.END.value:
                    # Historical task shutdown can record a trial END whose
                    # index is the next-trial counter. Do not use it as an onset
                    # or require its index to match the last recorded START.
                    activeTrial = None

            elif segmentNum is not None and isinstance(event, pglEventSegment):
                if event.eventType != pglEventSegment.boundaryType.START.value: continue
                if not validIndex(event.segmentNum):
                    raise ValueError("Recorded segment indices must be nonnegative integers.")
                if activeTrial is None:
                    raise ValueError(f"Segment {event.segmentNum} START has no preceding active trial START.")
                if not validTimestamp(event.timestamp):
                    raise ValueError(f"Trial {activeTrial}, segment {event.segmentNum} has an invalid START timestamp.")
                if lastBoundaryTime is not None and event.timestamp < lastBoundaryTime:
                    raise ValueError("Trial/segment boundary timestamps decrease in recorded event order.")

                lastBoundaryTime = event.timestamp
                if event.segmentNum != segmentNum: continue

                if activeTrial in segmentStarts:
                    raise ValueError(f"Trial {activeTrial} has multiple START events for segment {segmentNum}.")

                segmentStarts[activeTrial] = event

        omittedTrialNums = {
            "missingSegmentStart": [],
            "missingParameter": [],
            "unconfiguredParameterValue": [],
            "blankStimulus": [],
        }

        for trialIndex, trialEvent in sorted(starts.items()):
            onsetEvent = trialEvent if segmentNum is None else segmentStarts.get(trialIndex)

            if onsetEvent is None:
                omittedTrialNums["missingSegmentStart"].append(trialIndex)
                continue

            if parameter is not None:
                if trialIndex >= len(task.data.params) or parameterName not in task.data.params[trialIndex]:
                    omittedTrialNums["missingParameter"].append(trialIndex)
                    continue

                trialValue = task.data.params[trialIndex][parameterName]
                matchingGroups = [index for index, value in enumerate(validValues) if np.array_equal(trialValue, value)]

                if not matchingGroups:
                    omittedTrialNums["unconfiguredParameterValue"].append(trialIndex)
                    continue

                groupIndex = matchingGroups[0]

            else:
                if not 0 <= trialIndex < database.nTrials:
                    raise ValueError(f"Trial {trialIndex} has no corresponding stimulus trial.")

                trial = database.getTrial(trialIndex)
                if trial.trialNum != trialIndex:
                    raise ValueError("Stimulus trial numbering differs from sequence position.")

                stimulusNum = trial.stimulusNum
                if isinstance(stimulusNum, (bool, np.bool_)) or not isinstance(stimulusNum, Integral):
                    raise ValueError("Stimulus indices must be integers.")
                stimulusNum = int(stimulusNum)

                if trial.isBlank:
                    if stimulusNum != blankStimulusNum or trial.stimulus is not None:
                        raise ValueError("Blank trial has inconsistent stimulus metadata.")
                    if not includeBlank:
                        omittedTrialNums["blankStimulus"].append(trialIndex)
                        continue
                else:
                    if not 0 <= stimulusNum < len(database.stimuli):
                        raise ValueError("Trial references an invalid stimulus index.")
                    if trial.stimulus is not database.stimuli[stimulusNum]:
                        raise ValueError("Trial stimulus differs from the current inventory.")

                groupIndex = stimulusNum
                if groupIndex not in groups: groups[groupIndex] = newGroup()

            # The matcher returns zero-based behavioral trigger indices.
            # Retain None: absence of a trigger match must not erase a trial.
            group = groups[groupIndex]
            group["trialNums"].append(trialIndex)
            group["volumes"].append(self.getNearestVolumeTrigger(onsetEvent, direction=direction))
            group["startTimes"].append(float(onsetEvent.timestamp - task.data.startTime))
            group["onsetTimestamps"].append(float(onsetEvent.timestamp))

        if omittedTrialNums["missingSegmentStart"]:
            count = len(omittedTrialNums["missingSegmentStart"])
            pglMessages.warning(f"Omitted {count} started trial(s) without a recorded START for segment {segmentNum}; indices are available in omittedTrialNums.", level=1)

        omissionLabels = {
            "missingParameter": f"missing parameter {parameterName!r}",
            "unconfiguredParameterValue": f"parameter {parameterName!r} value outside configured validValues",
        }

        for reason, label in omissionLabels.items():
            if omittedTrialNums[reason]:
                pglMessages.warning(f"Omitted {len(omittedTrialNums[reason])} started trial(s): {label}. Trial indices are available in omittedTrialNums.", level=1)

        # Excluding blanks is an intentional selection, not a warning.
        # Their indices are still retained in omittedTrialNums for auditing.
        if parameter is not None:
            groupIndices = list(range(len(validValues)))
            groupValues = deepcopy(validValues)
            groupNames = [str(value) for value in validValues]
            extra = {
                "parameterName": parameter.settings.name,
                "parameterValues": deepcopy(validValues),
                "parameter": parameter,
            }

        else:
            # Inventory order, with the optional blank group last.
            groupIndices = sorted(groups, key=lambda index: (index == blankStimulusNum, index))
            stimulusIds, stimulusNames, sourceInfo, imageIndices = [], [], [], []

            for stimulusNum in groupIndices:
                if stimulusNum == blankStimulusNum:
                    stimulusIds.append(None)
                    stimulusNames.append("Blank")
                    sourceInfo.append({})
                    imageIndices.append(None)
                    continue

                stimulus = database.stimuli[stimulusNum]
                provenance = deepcopy(stimulus.sourceInfo)
                stimulusIds.append(stimulus.stimulusId or None)
                stimulusNames.append(stimulus.name)
                sourceInfo.append(provenance)
                imageIndices.append(provenance.get("imageIndex", getattr(stimulus, "imageIndex", None)))

            # Missing IDs remain missing, never replaced with run-local indices.
            # Cross-run condition construction must validate their identities.
            groupValues = list(stimulusIds)
            groupNames = list(stimulusNames)
            extra = {
                "stimulusNums": list(groupIndices),
                "stimulusIds": stimulusIds,
                "stimulusNames": stimulusNames,
                "sourceInfo": sourceInfo,
                "imageIndices": imageIndices,
                "nStimuli": len(groupIndices),
            }

        trialNums = [groups[index]["trialNums"] for index in groupIndices]
        nTrials = [len(indices) for indices in trialNums]

        return SimpleNamespace(
            taskName=task.settings.taskName,
            taskSaveName=task.settings.taskSaveName,
            groupBy="parameter" if parameter is not None else "stimulus",
            segmentNum=segmentNum,
            direction=direction,
            groupValues=groupValues,
            groupNames=groupNames,
            nGroups=len(groupIndices),
            trialNums=trialNums,
            volumes=[groups[index]["volumes"] for index in groupIndices],
            startTimes=[groups[index]["startTimes"] for index in groupIndices],
            onsetTimestamps=[groups[index]["onsetTimestamps"] for index in groupIndices],
            timeOrigin=float(task.data.startTime),
            nTrials=nTrials,
            nTrialsTotal=sum(nTrials),
            omittedTrialNums=omittedTrialNums,
            **extra,
        )
    
##############################################s
# Experiment class
##############################################
class pglExperiment(pglExperimentBase):
    '''
    Experiment class which handles timing, parameter randomization,
    subject response, synchronizing with measurement hardware etc
    '''
    def __init__(self, pgl=None, settingsName=None, settings=None, displayName=None, displaySettings=None, subjectID="s0000", experimentName=None, sessionName=None, runName=None):
        '''
        Initialize the pglExperiment class.

        Args:
            pgl (pgl): An instance of the pgl class.
            settingsName (str): Settings name; defaults are used when neither
                settingsName nor settings is supplied.
            settings (pglSettings): Explicit settings, superseding settingsName.
            displayName (str): Display name incorporated into the settings.
            displaySettings (pglDisplaySettings): Explicit display settings,
                superseding displayName.
            subjectID (str): Subject identifier. Unconventional naming warns
                but does not prevent initialization.
            experimentName (str): Experiment name.
            sessionName (str): Session name; defaults to session_YYYY-MM-DD.
            runName (str): Run name; defaults to run_HH-MM-SS.

        Genuine initialization failures propagate to the caller instead of
        silently returning a partially initialized experiment.
        '''
        # Initialize the base object and runtime defaults.
        super().__init__()
        self.isInitialized = False
        self.tasks = []

        # Set to flush the screen during acquisition.
        self.flush = True

        # Clear the text screen.
        #clear_output(wait=True)

        # Missing dependencies are actual initialization failures.
        if pgl is None:
            raise ValueError("Need to pass in a valid pgl instance.")

        # Save pgl.
        self.pgl = pgl

        # Load settings.
        self.settings = pgl.getSettings(
            settingsName=settingsName,
            settings=settings,
            displaySettings=displaySettings,
            displayName=displayName,
        )
        if self.settings is None:
            raise RuntimeError("Could not initialize experiment: no settings were loaded.")

        # Initialize experiment state and data.
        self.state = pglExperimentState()
        self.data = pglExperimentData()

        # Get experiment settings.
        self.experimentSettings = pglExperimentSettings()
        if experimentName:
            self.experimentSettings.experimentName = experimentName

        # Default session name is session_YYYY-MM-DD.
        self.experimentSettings.sessionName = (
            sessionName or f"session_{datetime.now().strftime('%Y-%m-%d')}"
        )

        # Default run name is run_HH-MM-SS.
        self.experimentSettings.runName = (
            runName or f"run_{datetime.now().strftime('%H-%M-%S')}"
        )

        # Naming-convention warnings do not interrupt initialization.
        self.experimentSettings.subjectID = subjectID

        # Mark successful initialization only after all setup completes.
        self.isInitialized = True

        
    def __repr__(self):
        return f"<pglExperiment: {len(self.task)} phases>"
    
    def initScreen(self, backgroundColor=-1, recordFlush=True, recordCommands=False, estimatedDuration=600, startRecording=True):
        '''
        Initialize the screen, start recording, and configure tasks.

        Args:
            backgroundColor: RGB background color; -1 uses settings.
            recordFlush: Retain flush presentation timing.
            recordCommands: Begin command buffering immediately.
            estimatedDuration: Initial recording capacity estimate in seconds.

        Movie timing is retained independently of the two recording flags.
        Existing recordings must be explicitly saved/exported and cleared.
        '''

        # Temporal batch calibration collects its own frame results.
        # False skips recording-session creation only, not display/device setup.
        if type(startRecording) is not bool:
            raise TypeError("startRecording must be Boolean.")

        
        # Never operate on an incompletely initialized experiment.
        if not getattr(self, "isInitialized", False):
            pglMessages.warning(
                "Experiment initialization did not complete. "
                "Create a new experiment before opening the screen or running."
            )
            return

        from .pglKeyboardMouse import pglKeyboardMouse


        if self.settings is None: raise RuntimeError("No settings found to open screen.")
        if not isinstance(recordFlush, bool) or not isinstance(recordCommands, bool):
            raise TypeError("recordFlush and recordCommands must be Boolean.")
        if isinstance(estimatedDuration, (bool, np.bool_)):
            raise ValueError("estimatedDuration must be a positive number of seconds.")
        estimatedDuration = float(estimatedDuration)
        if not math.isfinite(estimatedDuration) or estimatedDuration <= 0:
            raise ValueError("estimatedDuration must be finite and positive.")

        # Check before screen cleanup: never discard or disturb an old session.
        if getattr(self, "_recordingId", None) is not None:
            raise RuntimeError("This experiment already owns a recording. Create a new experiment for another run.")
        if getattr(self.pgl, "_recordingState", None) in ("active", "stopped"):
            raise RuntimeError("Save/export the previous recording, then call pgl.recordingClear().")
        if self.pgl.commandRecording or self.pgl.flushRecording:
            raise RuntimeError("Stop and export existing capture before initializing this experiment.")
        if self.pgl.commandRecordingMetadata is not None or self.pgl._presentationLog is not None:
            raise RuntimeError("Export existing recording data, then call pgl.recordingClear().")
        if getattr(self.pgl, "_moviePlaybacks", []) or getattr(self.pgl, "_movieDrawSamples", []):
            raise RuntimeError("Export existing movie data, then call pgl.recordingClear().")

        # Initialize recovery bookkeeping before task configuration can fail.
        self._runSavePath = None
        self._runStarted = False
        self._recordingOutcome = "error"
        self.data.runErrors = []
        self.data.saveStatus = "notAttempted"
        self.data.cleanupStatus = "notAttempted"
        self.data.eyeTrackerSaveStatus = "notAttempted"
        self.state.openScreen = False

        # Get background color. Only the scalar sentinel means "use settings".
        if isinstance(backgroundColor, (int, float)) and backgroundColor == -1:
            backgroundColor = self.settings.backgroundColor

        try:
            # Get screen parameters.
            if len(self.settings.displays) < 1:
                raise RuntimeError(f"Settings {self.settings.name} is not associated with a display.")
            self.state.display = self.settings.displays[0]

            # Make PGL quieter.
            self.setVerbose()

            # Preserve the existing screen-initialization cleanup behavior.
            self.pgl.cleanUp()

            # Save original screen resolution before changing it.
            self.state.originalScreenResolution = self.pgl.getResolution(self.state.display.currentDisplayNum)

            if self.state.display.uuid == "windowed":
                # Open the screen.
                self.pgl.open(0, screenWidth=self.state.display.windowSize[0], screenHeight=self.state.display.windowSize[1], backgroundColor=backgroundColor)
                if not self.pgl.isOpen(): raise RuntimeError("Failed to open windowed screen.")
                self.pgl.setWindowFrameInDisplay(
                    0,
                    screenX=self.state.display.windowPosition[0],
                    screenY=self.state.display.windowPosition[1],
                    screenWidth=self.state.display.windowSize[0],
                    screenHeight=self.state.display.windowSize[1],
                )
            else:
                if self.state.display.currentDisplayNum == -1:
                    raise RuntimeError(f"Display {self.state.display.name} is not connected.")

                # Compare the current display mode with the desired mode.
                if self.state.display.displayModes:
                    displayMode = self.state.display.displayModes[0]
                    if displayMode == self.state.originalScreenResolution:
                        pglMessages.message(f"Display mode settings match current mode: {self.state.originalScreenResolution}")
                    else:
                        self.pgl.setResolutionUsingDisplayModeSettings(self.state.display.currentDisplayNum, displayMode)
                        self.state.screenResolution = self.pgl.getResolution()
                        pglMessages.message(
                            f"Changing screen resolution to: {self.state.screenResolution[0]} x "
                            f"{self.state.screenResolution[1]} {self.state.screenResolution[2]}Hz "
                            f"{self.state.screenResolution[3]}bits from: "
                            f"{self.state.originalScreenResolution[0]} x "
                            f"{self.state.originalScreenResolution[1]} "
                            f"{self.state.originalScreenResolution[2]}Hz "
                            f"{self.state.originalScreenResolution[3]}bits"
                        )

                # Open the screen.
                self.pgl.open(whichScreen=self.state.display.currentDisplayNum, backgroundColor=backgroundColor)

            if not self.pgl.isOpen(): raise RuntimeError("Failed to open screen.")

            # Set visual angle coordinates.
            self.pgl.visualAngle(self.state.display.displayDistance, self.state.display.displaySize[0], self.state.display.displaySize[1])

            # Flip left-right and/or up-down if specified in settings.
            if self.state.display.flipLeftRight: self.pgl.flipLeftRight()
            if self.state.display.flipUpDown: self.pgl.flipUpDown()

            # Set the gamma table.
            self.setGamma(self.settings, self.state.display)

            # Add a keyboard device if not already loaded.
            keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse)
            if not keyboardDevices:
                keyboardMouse = pglKeyboardMouse(eatKeys=None)
                self.pgl.devicesAdd(keyboardMouse)

                # Check whether the keyboard listener is running.
                if not keyboardMouse.isRunning():
                    pglMessages.warning(
                        "Accessibility permission not granted for keyboard/mouse access.\n"
                        "On macOS, enable your terminal or application in Privacy & Security > Accessibility."
                    )
                    raise RuntimeError("Keyboard/mouse listener could not start.")
            else:
                # If already loaded, use the existing keyboard device.
                keyboardMouse = keyboardDevices[0]
                if not keyboardMouse.isRunning(): keyboardMouse.start()

            # Clear pending keyboard/mouse events and retain the device.
            keyboardMouse.clear()
            self.keyboardMouse = keyboardMouse

            # A comma-separated response list supports keys such as space/F1.
            # Without commas, each character specifies a response key.
            # Share compact-character / named-key syntax with special eat keys.
            self.responseKeysList = keyboardMouse.parseKeyNames(self.settings.responseKeys)

            # Empty or None control keys are disabled.
            def getControlKeyCode(key, settingName):
                if not key: return None
                keyCode = keyboardMouse.charToKeyCode(key)
                if keyCode is None: raise ValueError(f"Unknown {settingName}: {key!r}")
                return int(keyCode)

            self.state.responseKeyCodesList = [keyboardMouse.charToKeyCode(k) for k in self.responseKeysList]
            self.state.startKeyCode = getControlKeyCode(self.settings.startKey, "startKey")
            self.state.endKeyCode = getControlKeyCode(self.settings.endKey, "endKey")
            self.state.volumeTriggerKeyCode = getControlKeyCode(self.settings.volumeTriggerKey, "volumeTriggerKey")

            # Manual pre-start still requires a keyboard key.
            # Validate scanner synchronization after digital devices initialize.
            if self.settings.manualPreStart and self.state.startKeyCode is None:
                raise ValueError("manualPreStart requires a startKey.")
            
            # Apply Experiment, Special, and All; clear unchecked modes.
            self._configureKeyboardSuppression()
            
            # Wait half a second for the Metal app to initialize.
            self.pgl.waitSecs(0.5)

            # Flush the screen to get rid of transients.
            self.pgl.flush()
            self.pgl.flush()

            # Initialize eye tracker.
            try:
                self.initEyeTracker()
            except Exception as error:
                pglMessages.warning(f"Could not initialize eye tracker: {error}")

            # Display device status.
            self.pgl.deviceStatus()

            # Set up digital IO and other devices.
            self.initDigitalIO()
            self.initDevices()
            
            # Require a usable trigger source, not merely a selected device.
            self._validateVolumeTriggerSource()

            # Start before configure(), which may create stimulus resources.
            if startRecording:
                self._startExperimentRecording(estimatedDuration, recordFlush, recordCommands)

            # Record mask creation so command replay can recreate the stencil.
            # Visual-angle coordinates and display flips are already configured.
            if not self.initStencil(backgroundColor=backgroundColor):
                pglMessages.warning("Could not initialize the experiment stencil.")
                return

            # Bind original source snapshots before configuring any task.
            # Source problems must not prevent acquisition or partial saving.
            sourceErrors = []
            for task in self.tasks:
                self._attemptRunAction(
                    f"capture source for task {task.settings.taskName}",
                    task.captureSource,
                    sourceErrors,
                    reportProgress=False
                )

            # A failed configuration is not a successfully initialized task.
            # Propagate it through the same partial-save recovery path.
            for task in self.tasks: task.configure(self)
            
            self.state.openScreen = True

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            self.state.runFinishedWithError = True
            self._recordingOutcome = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
            self._recordRunError("screen initialization", error)

            errors = []
            self._attemptRunAction("record initialization end time", lambda: setattr(self.data, "endTime", self.pgl.getSecs()), errors)

            try:
                if getattr(self, "_recordingId", None) is not None:
                    # Recording has started: save available task/recording data
                    # before renderer or device cleanup.
                    self._finalizeRun()
                else:
                    # No recording was started. Attempt independent cleanup.
                    self._attemptRunAction("close experiment screen",lambda: self.endScreen(clearScreen=False, resetStencil=False),errors)
                    self._attemptRunAction("close digital IO", self.endDigitalIO, errors)
                    self._attemptRunAction("close experiment devices", self.endDevices, errors)
            except (Exception, KeyboardInterrupt, SystemExit) as finalizationError:
                self._recordRunError("initialization finalization", finalizationError)
            finally:
                self.state.openScreen = False

            raise
        
    def setGamma(self, settings, display):
        '''
        set the gamma table based on settings
        
        Args:
            settings (pglSettings): Used for determining what gamma to use
            display (pglDisplaySettings): Used to determine what displayNum to set the gamma on
        '''
        # no gamma correction asked for
        if settings.calibrateForGamma == 0.0:
            return
        
        # No calibration
        if display.luminanceCalibration[0] == "None":
            if settings.calibrateForGamma is not None and settings.calibrateForGamma[0] != 0.0:
                pglMessages.warning(f"Gamma is set to {settings.calibrateForGamma[0]} but no calibration found for display {display.name}")
                return
        
        # save the original gamma table
        self.state.originalGammaTable = self.pgl.getGammaTable(display.currentDisplayNum)

        # set the gamma to whatever is at the top of the calibrateForGamma list
        gamma = settings.calibrateForGamma[0]
        display.setGamma(self.pgl, gamma=gamma)
        
        # save the gamma table
        self.state.gammaTable = self.pgl.getGammaTable(display.currentDisplayNum)
        
    ################################################################
    # Restore and close the screen with independent cleanup attempts
    ################################################################
    def endScreen(self, clearScreen=True, resetStencil=True):
        """Restore display settings and close this experiment's screen.

        clearScreen=False avoids a final blank frame.
        resetStencil=False avoids stencil commands after an interrupted exchange.
        Failures propagate after the other cleanup operations are attempted.
        """
        from .pglKeyboardMouse import pglKeyboardMouse

        errors = []

        # Stop each keyboard listener independently.
        try:
            keyboardDevices = self.pgl.devicesGet(pglKeyboardMouse) or []
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            keyboardDevices = []
            errors.append(error)
            self._recordRunError("find keyboard devices", error)

        for keyboardDevice in keyboardDevices:
            self._attemptRunAction("stop keyboard/mouse device", keyboardDevice.stop, errors)

        # Disable the experiment mask even when the window remains open.
        # Recovery paths must not send commands into an interrupted exchange.
        if resetStencil:
            self._attemptRunAction("disable experiment stencil", self.endStencil, errors)
        elif not self.settings.closeScreenOnEnd and getattr(self, "_experimentStencilNumber", None) is not None:
            pglMessages.warning(
                "Renderer recovery skipped stencil reset. The open display may "
                "still have masking enabled; close/reopen it before reuse."
            )

        if self.settings.closeScreenOnEnd:
            # Normal callers can request a final blank. Recovery callers skip
            # this because the renderer's command stream may be interrupted.
            if clearScreen:
                def clearDisplay():
                    self.pgl.clearScreen(self.settings.backgroundColor)
                    self.pgl.flush()

                self._attemptRunAction("clear screen", clearDisplay, errors)

            # Restore each display setting even if another operation fails.
            if self.state.originalGammaTable is not None:
                self._attemptRunAction(
                    "restore gamma table",
                    lambda: self.pgl.setGammaTable(
                        self.state.display.currentDisplayNum,
                        rgbGammaTable=self.state.originalGammaTable,
                    ),
                    errors,
                )

            if self.state.screenResolution is not None and self.state.originalScreenResolution is not None:
                if self.state.screenResolution != self.state.originalScreenResolution:
                    self._attemptRunAction(
                        "restore screen resolution",
                        lambda: self.pgl.setResolution(
                            self.state.display.currentDisplayNum,
                            screenResolution=self.state.originalScreenResolution,
                        ),
                        errors,
                    )

            def closeDisplay():
                if self.pgl.close() is False: raise RuntimeError("PGL could not close its renderer.")
                self._experimentStencilNumber = None
                self.state.openScreen = False

            self._attemptRunAction("close renderer connection", closeDisplay, errors)

        # Restore local clearing behavior even when recovery skipped renderer
        # commands, or when disabling the stencil failed.
        if getattr(self, "_stencilClearModeChanged", False):
            self.pgl.clearScreenDrawQuad = self._stencilPreviousDrawQuad
            self._stencilClearModeChanged = False

        self._attemptRunAction("restore verbosity", self.endVerbose, errors)

        if errors: raise errors[0]
        return True
                

    def setEatAllKeys(self, eatAllKeys=False):
        '''
        Sets whether to eat all keys. If False, any keys specified by setEatKeys will still be eaten
        Args: 
            eatAllKeys: bool (whether to eat all keys or not)
        '''
        return self.pgl.setEatAllKeys(eatAllKeys)

    def setEatKeys(self, eatKeys=""):
        '''
        Args: 
            eatKeys (list): list of characters to eat. e.g. ['return','esc','1']
        '''
        return self.pgl.setEatKeys(eatKeys)
    
    ################################################################
    # Run with protected saving and finalization
    ################################################################
    def run(self):
        """Run using the startup repair snapshot; preserve partial saving."""
        from copy import deepcopy

        if not getattr(self, "isInitialized", False):
            pglMessages.warning(
                "Experiment initialization did not complete. "
                "Create a new experiment before opening the screen or running."
            )
            return

        if getattr(self, "_recordingReleased", False):
            raise RuntimeError(
                "This experiment's recording was already saved and released. "
                "Create a new experiment."
            )

        if not self.state.openScreen:
            self.state.runFinishedWithError = True
            pglMessages.warning(
                "Screen is not open. Call initScreen() before running the experiment."
            )
            return

        if getattr(self, "_runStarted", False):
            raise RuntimeError(
                "This experiment has already run. Create a new experiment for another run."
            )

        metadata = self._getExperimentRecording()
        if metadata is not None and self.pgl._recordingState != "active":
            raise RuntimeError("The experiment recording is already stopped.")
        if metadata is None and getattr(self.pgl, "_recordingState", None) == "active":
            raise RuntimeError("PGL has an active recording not owned by this experiment.")

        self._runStarted = True
        if not hasattr(self, "_runSavePath"): self._runSavePath = None
        if not hasattr(self.data, "runErrors"): self.data.runErrors = []

        self.data.saveStatus = "notAttempted"
        self.data.cleanupStatus = "notAttempted"
        self.data.eyeTrackerSaveStatus = (
            "notAttempted" if self.eyeTracker is not None else "notApplicable"
        )
        self.state.runFinishedWithError = True
        self._recordingOutcome = "error"

        try:
            acquisitionConfig = getattr(self, "_labJackTriggerConfig", None)
            if acquisitionConfig is None:
                repairConfig = {"repairMissingVolumeTriggers": False}
                requestedRepairConfig = None
            else:
                snapshot = getattr(self, "_labJackRepairConfig", None)
                if snapshot is None:
                    raise RuntimeError(
                        "LabJack acquisition has no repair-policy snapshot. "
                        "Initialize a new experiment with the updated initDevices()."
                    )

                requestedRepairConfig = deepcopy(snapshot)
                repairConfig = deepcopy(snapshot)

                # Preserve the user's request, but repair cannot operate
                # without counter verification.
                repairConfig["repairMissingVolumeTriggers"] = (
                    repairConfig["repairMissingVolumeTriggers"]
                    and acquisitionConfig["verifyTriggerCounter"]
                )

            self.data.volumeTriggerRepairRequested = requestedRepairConfig
            self._run(**repairConfig)

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            self.state.runFinishedWithError = True
            self._recordingOutcome = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "error"
            )
            self._recordRunError("acquisition", error)

            errors = []
            self._attemptRunAction(
                "record experiment end time",
                lambda: setattr(self.data, "endTime", self.pgl.getSecs()),
                errors,
            )

            try:
                self._finalizeRun()
            except (Exception, KeyboardInterrupt, SystemExit) as finalizationError:
                try:
                    pglMessages.warning(
                        f"Additional finalization failure: "
                        f"{type(finalizationError).__name__}: {finalizationError}"
                    )
                except Exception:
                    pass
            raise

        else:
            self.state.runFinishedWithError = False
            self._recordingOutcome = "completed"
            self._finalizeRun()

    ################################################################
    # Store errors as plain serializable metadata
    ################################################################
    def _recordRunError(self, operation, error):
        """Record an error without retaining its live traceback object."""
        import traceback

        # Error reporting itself must not replace the original failure.
        try:
            errorInfo = {
                "operation": operation,
                "module": type(error).__module__,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
            if not hasattr(self.data, "runErrors"): self.data.runErrors = []
            self.data.runErrors.append(errorInfo)
        except Exception:
            pass

        try:
            pglMessages.warning(f"{operation}: {type(error).__name__}: {error}")
        except Exception:
            pass

    ################################################################
    # Attempt an independent save or cleanup operation
    ################################################################
    def _attemptRunAction(self, operation, action, errors, reportProgress=True):
        """Report routine progress only during saving; always report failures."""
        import time

        started = time.monotonic()

        # Only save() enables detailed progress. Initialization and cleanup
        # remain quiet unless an operation fails.
        showProgress = (
            reportProgress
            and getattr(self, "_saveProgressActive", False)
        )

        if showProgress:
            self._reportRunProgress(f"START: {operation}")

        try:
            result = action()
            if result is False:
                raise RuntimeError(f"{operation} returned False.")

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            # The interrupted operation is not resumed. Remaining independent
            # operations are still attempted, and the caller later raises.
            errors.append(error)
            elapsed = time.monotonic() - started
            outcome = (
                "INTERRUPTED"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "FAILED"
            )

            # Always report failures, including during initialization/cleanup.
            self._reportRunProgress(
                f"{outcome}: {operation} ({elapsed:.2f}s): "
                f"{type(error).__name__}: {error}"
            )

            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                self._reportRunProgress(
                    "This operation did not return successfully. "
                    "Continuing remaining independent attempts."
                )

            self._recordRunError(operation, error)
            return False

        if showProgress:
            self._reportRunProgress(
                f"DONE: {operation} ({time.monotonic() - started:.2f}s)"
            )

        return True

    ################################################################
    # Report finalization separately from the main local save pass
    ################################################################
    def _finalizeRun(self):
        """Finalize acquisition with separate save and cleanup reporting."""
        #self._reportRunProgress(
        #    "FINALIZATION STARTED: stop capture, save locally, then clean up. "
        #    "Final metadata will be updated after cleanup."
        #)

        try:
            self._finalizeRunContents()

        except (Exception, KeyboardInterrupt, SystemExit):
            self._reportRunProgress(
                "FINALIZATION DID NOT COMPLETE SUCCESSFULLY."
            )
            self._reportRunProgress(
                f"Local save: {getattr(self.data, 'saveStatus', 'unknown')} | "
                f"Cleanup: {getattr(self.data, 'cleanupStatus', 'unknown')} | "
                f"Eye tracker: {getattr(self.data, 'eyeTrackerSaveStatus', 'unknown')}"
            )

            if getattr(self.data, "saveStatus", None) != "complete":
                self._reportSaveRecovery()
            else:
                self._reportRunProgress(
                    "The local save is marked complete. "
                    "Review the failed stage: a cleanup or eye-tracker failure "
                    "does not by itself mean the local data save failed."
                )

            raise

        self._reportRunProgress(
            f"Experiment finished run and saved to: {getattr(self, '_runSavePath', None)}"
        )
    ################################################################
    # Finalize successful or interrupted acquisition
    ################################################################
    ################################################################
    # Finalize successful or interrupted acquisition
    ################################################################
    def _finalizeRunContents(self):
        """Stop owned capture, attempt local saving, then clean up devices.

        No task/result display is performed here.

        Local saving is attempted before screen/device cleanup and eye-tracker
        transfer. Final experiment metadata is updated after those operations.
        """
        errors = []

        def stopRecording():
            metadata = self._getExperimentRecording()
            if metadata is not None:
                self.pgl.recordingStop(
                    outcome=getattr(self, "_recordingOutcome", "error")
                )

        ################################################################
        # Stop capture and attempt local saving first
        ################################################################

        # Stop before any save or cleanup command can enter the recording.
        self._attemptRunAction(
            "stop experiment recording",
            stopRecording,
            errors,
        )

        # Attempt local saving before potentially blocking device operations.
        # The save() wrapper reports whether this save pass completed.
        if not self._attemptRunAction(
            "save experiment",
            lambda: self.save(includeEyeTracker=False),
            errors,
        ):
            self.data.saveStatus = "failed"

        ################################################################
        # Attempt screen and device cleanup after the local save attempt
        ################################################################

        # Local saving has been attempted. Avoid extra drawing commands during
        # cleanup, especially after an interrupted renderer exchange.
        #
        # If closeScreenOnEnd=False, this does not blank the final stimulus.
        # The raw save above remains independent of ending calibration.
        # Stop/drain and finalize timing while the owned connection is open.
        self._attemptRunAction(
            "finalize LabJack clock timing",
            self._finalizeLabJackClockTiming,
            errors,
            reportProgress=False,
        )

        cleanupErrorCount = len(errors)

        # Normal completion disables masking without presenting another frame.
        # Error recovery avoids additional renderer exchanges.
        self._attemptRunAction(
            "close experiment screen",
            lambda: self.endScreen(
                clearScreen=False,
                resetStencil=not self.state.runFinishedWithError,
            ),
            errors,
        )
        self._attemptRunAction(
            "close digital IO",
            self.endDigitalIO,
            errors,
        )
        self._attemptRunAction(
            "close experiment devices",
            self.endDevices,
            errors,
        )

        self.data.cleanupStatus = (
            "failed" if len(errors) > cleanupErrorCount else "complete"
        )

        ################################################################
        # Retrieve eye-tracker data after essential local save attempts
        ################################################################
        if self.eyeTracker is not None:
            eyeTrackerErrorCount = len(errors)
            dataPath = getattr(self, "_runSavePath", None)

            if dataPath is not None:
                def saveEyeTracker():
                    filename = dataPath / self.settings.eyetracker[0].lower()
                    return self.eyeTracker.save(filename)

                self._attemptRunAction(
                    "save eye tracker",
                    saveEyeTracker,
                    errors,
                )

            else:
                self._attemptRunAction(
                    "stop eye tracker",
                    self.eyeTracker.stop,
                    errors,
                )

            self.data.eyeTrackerSaveStatus = (
                "failed" if len(errors) > eyeTrackerErrorCount
                else "complete" if dataPath is not None
                else "notSaved"
            )

        ################################################################
        # Persist final outcome metadata without repeating large saves
        ################################################################

        # These writes update cleanup/eye-tracker outcomes without repeating
        # task, recording, or EDF saves.
        #
        # Their interruption safety depends on the implementation of
        # saveStrict(); successful return alone does not prove fsync durability.
        dataPath = getattr(self, "_runSavePath", None)

        if dataPath is not None:
            if not self._attemptRunAction(
                "update final experiment state",
                lambda: self.state.saveStrict(dataPath / "state.json"),
                errors,
            ):
                self.data.saveStatus = "failed"

            if not self._attemptRunAction(
                "update final experiment data",
                lambda: self.data.saveStrict(dataPath / "data.json"),
                errors,
            ):
                self.data.saveStatus = "failed"

        # run() preserves an earlier acquisition error. Otherwise this exposes
        # the first finalization failure after all independent attempts.
        if errors:
            raise errors[0]
    
    def _run(
        self, *,
        repairMissingVolumeTriggers=True,
        volumeTriggerMedianWindow=10,
        volumeTriggerMinIntervals=3,
    ):
        """Run acquisition with cumulative task timing and frame association."""
        from numbers import Integral

        if type(repairMissingVolumeTriggers) is not bool:
            raise TypeError("repairMissingVolumeTriggers must be Boolean.")

        for name, value in (
            ("volumeTriggerMedianWindow", volumeTriggerMedianWindow),
            ("volumeTriggerMinIntervals", volumeTriggerMinIntervals),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if volumeTriggerMinIntervals > volumeTriggerMedianWindow:
            raise ValueError("Minimum interval history cannot exceed its window.")

        self.state.runFinishedWithError = True
        if not self.state.openScreen:
            pglMessages.warning("Screen is not open. Call initScreen() first.")
            return

        self._volumeRepairRuntime = {
            "enabled": repairMissingVolumeTriggers,
            "medianWindow": int(volumeTriggerMedianWindow),
            "minIntervals": int(volumeTriggerMinIntervals),
            "lastTrustedAnchor": None,
            "handledMismatchIds": set(),
            "acceptanceBoundary": None,
        }
        if not hasattr(self.data, "volumeTriggerRepairAudit"):
            self.data.volumeTriggerRepairAudit = []

        self.data.volumeTriggerRepairConfiguration = {
            "enabled": repairMissingVolumeTriggers,
            "medianWindow": int(volumeTriggerMedianWindow),
            "minIntervals": int(volumeTriggerMinIntervals),
            "preStartPolicy": "doNotRetroactivelyChangeAcceptanceBoundary",
        }

        self._initializeTaskTiming()

        if self.eyeTracker is not None: self.eyeTracker.start()

        self.state.phaseNums = sorted({
            task.settings.phaseNum for task in self.tasks
            if task.settings.phaseNum is not None
        }) or None
        self.state.currentPhaseIndex = 0
        self.state.experimentDone = False
        self.state.experimentStarted = False
        self.state.volumeNumber = 0

        manualPreStart = bool(self.settings.manualPreStart)
        manualPreStartVolumes = 0
        ignoreInitialVolumes = self.settings.ignoreInitialVolumes

        if self.settings.eyetracker is not None: self.calibrateEyeTracker()
        self._validateVolumeTriggerSource()

        diagnosticTypes = {
            "digitalInputCounterWindow",
            "digitalInputMismatch",
            "digitalInputRecovery",
        }

        def matchesKey(event, keyCode, keydownOnly=False):
            return (
                keyCode is not None
                and event.type == "keyboard"
                and event.keyCode == keyCode
                and (not keydownOnly or event.eventType == "keydown")
            )

        def retainPolledEvents():
            events = list(self.pgl.poll())
            self.data.events.extend(events)
            return events

        def acceptMeasured(event):
            self._recordVolumeTrigger(event)
            if self._volumeRepairRuntime["acceptanceBoundary"] is None:
                self._volumeRepairRuntime["acceptanceBoundary"] = event.timestamp

        def processDiagnostics(events):
            for event in events:
                if event.type in diagnosticTypes:
                    self._consumeVolumeCounterDiagnostic(event)

        # Process retained evidence without accepting pre-gate volumes.
        processDiagnostics(list(self.data.events))

        for device in getattr(self, "_experimentInputDevices", []):
            pending = list(device.poll())
            self.data.events.extend(pending)
            processDiagnostics(pending)

        try:
            ################################################################
            # Wait for manual/start-on-volume gates
            ################################################################
            if self.state.startKeyCode is not None or self.settings.startOnVolumeTrigger:
                while not self.state.experimentStarted:
                    if manualPreStart:
                        self.pgl.text(
                            f"Press {self.settings.startKey} key to make experiment start",
                            y=0,
                        )
                    elif self.settings.startOnVolumeTrigger:
                        screen = self.settings.startScreen[0]
                        if screen in ("blank", "fixation cross"):
                            self.pgl.clearScreen(self.settings.backgroundColor)
                            if screen == "fixation cross": self.pgl.fixationCross()
                        elif screen == "waiting message":
                            self.pgl.text("Waiting for volume trigger to start experiment", y=0)
                    else:
                        self.pgl.text(
                            f"Press {self.settings.startKey} key to start experiment",
                            y=0,
                        )

                    # Retain a recent presentation for the first task prediction.
                    self._flushTaskFrame()
                    events = retainPolledEvents()

                    if any(matchesKey(event, self.state.endKeyCode) for event in events):
                        self.state.experimentStarted = True
                        self.state.experimentDone = True
                        processDiagnostics(events)
                        break

                    # Physical controls use timestamp order. Diagnostics retain
                    # their original delivery order.
                    controls = [
                        event for event in events
                        if event.type not in diagnosticTypes
                    ]
                    controls.sort(key=lambda event: (
                        event.timestamp
                        if getattr(event, "timestamp", None) is not None
                        else float("inf")
                    ))

                    for event in controls:
                        if matchesKey(event, self.state.startKeyCode, keydownOnly=True):
                            if manualPreStart:
                                manualPreStart = False
                                pglMessages.message(
                                    f"Manual pre-start ended after {manualPreStartVolumes} volumes.",
                                    messageType="experiment",
                                )
                            else:
                                self.state.experimentStarted = True

                        if not self.settings.startOnVolumeTrigger: continue
                        if not self._isVolumeTrigger(event): continue

                        if manualPreStart:
                            manualPreStartVolumes += 1
                        elif ignoreInitialVolumes > 0:
                            ignoreInitialVolumes -= 1
                        else:
                            if not self.state.experimentStarted:
                                pglMessages.message(
                                    f"Ignored {self.settings.ignoreInitialVolumes} initial volumes.",
                                    messageType="experiment",
                                )
                            self.state.experimentStarted = True
                            acceptMeasured(event)

                    processDiagnostics(events)
            else:
                self.state.experimentStarted = True

            if self.state.experimentDone:
                self.data.endTime = self.pgl.getSecs()
                self._reportRunProgress(
                    "Experiment ended before tasks started; proceeding to save."
                )
                return

            ################################################################
            # Start tasks without backdating transition timestamps
            ################################################################
            pglMessages.message("Experiment started.", messageType="experiment")
            self.data.startTime = self.pgl.getSecs()
            self.data.startDateTime = datetime.now().astimezone().isoformat()

            if self._volumeRepairRuntime["acceptanceBoundary"] is None:
                self._volumeRepairRuntime["acceptanceBoundary"] = self.data.startTime

            if not self.state.phaseNums:
                raise ValueError("Experiment needs at least one task with a phaseNum.")

            self._prepareTaskFrame(self.pgl.getSecs())
            self.startPhase(phaseNum=self.state.phaseNums[0])

            while not self.state.experimentDone:
                events = retainPolledEvents()

                if any(matchesKey(event, self.state.endKeyCode) for event in events):
                    self.state.experimentDone = True
                    processDiagnostics(events)

                    stopTime = self.pgl.getSecs()
                    errors = []
                    for task in self.currentTasks:
                        self._attemptRunAction(
                            f"stop task {task.settings.taskName}",
                            lambda task=task: task.end(timestamp=stopTime, reason="userStopped"),
                            errors,
                            reportProgress=False,
                        )
                    if errors: raise errors[0]
                    continue

                taskEvents = []
                for event in events:
                    if event.type in diagnosticTypes: continue
                    if self._isVolumeTrigger(event):
                        acceptMeasured(event)
                    else:
                        taskEvents.append(event)

                # Retained source edges are available before late repair placement.
                processDiagnostics(events)

                subjectResponses = [
                    (self.state.responseKeyCodesList.index(event.keyCode), event.timestamp)
                    for event in taskEvents
                    if (
                        event.type == "keyboard"
                        and event.eventType == "keydown"
                        and event.keyCode in self.state.responseKeyCodesList
                    )
                ]

                updateTime = self.pgl.getSecs()
                self._prepareTaskFrame(updateTime)

                phaseDone = False
                for task in self.currentTasks:
                    task.update(
                        updateTime=updateTime,
                        subjectResponses=subjectResponses,
                        phaseNum=self.state.phaseNum,
                        tasks=self.currentTasks,
                        events=taskEvents,
                    )
                    if task.done(): phaseDone = True

                if self.flush: self._flushTaskFrame()

                if phaseDone:
                    phaseEndTime = self.pgl.getSecs()
                    errors = []

                    for task in self.currentTasks:
                        self._attemptRunAction(
                            f"end phase task {task.settings.taskName}",
                            lambda task=task: task.end(
                                timestamp=phaseEndTime, reason="phaseEnded",
                            ),
                            errors,
                            reportProgress=False,
                        )
                    if errors: raise errors[0]

                    if self.state.currentPhaseIndex >= len(self.state.phaseNums) - 1:
                        self.state.experimentDone = True
                    else:
                        self.state.currentPhaseIndex += 1
                        self._prepareTaskFrame(self.pgl.getSecs())
                        self.startPhase(
                            phaseNum=self.state.phaseNums[self.state.currentPhaseIndex]
                        )

            self.data.endTime = self.pgl.getSecs()

        except (Exception, KeyboardInterrupt, SystemExit):
            # Preserve partial lifecycle evidence without invoking callbacks
            # that might send drawing commands into an interrupted exchange.
            failureTime = self.pgl.getSecs()

            for task in self.tasks:
                if not task._taskActive or task.data.endTime is not None:
                    continue

                if task._trialActive:
                    event = pglEventTrial(
                        task.state.currentTrial,
                        failureTime,
                        eventType=pglEventTrial.boundaryType.END,
                    )
                    event.completed = False
                    event.endReason = "acquisitionError"
                    task.data.events.append(event)
                    task._trialActive = False

                task.data.endTime = failureTime
                task.data.endReason = "acquisitionError"
                task.data.endCallbacksSkipped = True
                task._taskActive = False

            raise

        finally:
            self._finishPendingTaskTiming()
    
    def initDigitalIO(self):
        """
        Initialize experiment-owned digital hardware once per physical role.

        LabJack output and LabJack digital trigger share one connection.
        Opening an input-only device does not configure output pins.
        """
        from .pglLabJack import pglLabJack
        from .pglSettings import pglSettingsManager

        self.endDigitalIO()

        self.state.digitalIODevices = []
        self.state.triggerWords = {}
        self.state.triggers = {}
        self.settings.digitalIOWords = []
        self.settings.digitalIOChannels = []
        self._experimentInputDevices = []

        selectedNames = list(self.settings.digitalIO)
        needsLabJackInput = any(
            name.casefold() == "labjack digital trigger"
            for name in selectedNames
        )

        # Resolve the preset before opening the experiment-owned connection.
        self._labJackPresetSnapshot = None
        needsLabJack = any(
            name.casefold() in {"labjack", "labjack digital trigger"}
            for name in selectedNames
        )
        if needsLabJack:
            if self.settings.labJack:
                self._labJackPresetSnapshot = self.settings.getLabJackPreset()
            else:
                from copy import deepcopy
                preset = pglSettingsManager.getLabJackSettings()
                if preset is None:
                    raise RuntimeError("No saved default LabJack preset is available.")
                self._labJackPresetSnapshot = deepcopy(preset)
            self._labJackPresetSnapshot.getClockCalibrationConfig()

        requestedNames = []
        for name in selectedNames:
            hardwareName = "LabJack" if name.casefold() == "labjack digital trigger" else name
            if not any(existing.casefold() == hardwareName.casefold() for existing in requestedNames):
                requestedNames.append(hardwareName)

        for deviceName in requestedNames:
            normalizedName = deviceName.casefold()

            if normalizedName == "labjack":
                existingDevices = self.pgl.devicesGet(pglLabJack)
                if any(device.isActive for device in existingDevices):
                    raise RuntimeError(
                        "A LabJack is already registered outside this experiment. "
                        "Stop, close, and unregister it before initialization."
                    )

                labJack = pglLabJack(
                    **self._labJackPresetSnapshot.getClockCalibrationConfig()
                )

                # Retain ownership immediately for initialization recovery.
                if labJack.isActive:
                    self.state.digitalIODevices.append(labJack)
                    pglMessages.message("LabJack initialized")
                elif needsLabJackInput:
                    raise RuntimeError("The selected LabJack scanner-trigger device did not initialize")
                else:
                    pglMessages.message("LabJack did not initialize")

            elif normalizedName == "datapixx":
                from .pglVPixx import pglDataPixx
                dataPixx = pglDataPixx()

                if dataPixx.isActive:
                    self.state.digitalIODevices.append(dataPixx)
                    pglMessages.message("DataPixx initialized")
                else:
                    pglMessages.message("DataPixx did not initialize")

            else:
                pglMessages.warning(f"Unknown digitalIO device: {deviceName}")
        

    def getDigitalIODevice(self, digitalIODevice = None):
        '''
        Returns the digitalIODevice that has been initialized or None if there is no digialIODevice
        Without any arguments returns the 1st in the list if there is more than one (digialIODevices are
        selected in pgl.settings()
        
        Args:
            digitalIODevice (pglDigitalIODevice): Type of digitalIODevice to return (e.g. pglLabJack). If there
                is no match, then None will be returned. 
        '''
        
        # no digitalIODevice specified, return the first in list
        if digitalIODevice is None:
            return next(iter(self.state.digitalIODevices), None)

        # look for device and return it if specified
        return next((device for device in self.state.digitalIODevices if isinstance(device, digitalIODevice)),None)
        
    def configureTriggers(self, channels, groupName=None, digitalIODevice=None, pulseLen=5):
        """
        Configure named triggers on the digital IO device.

        Args:
            channels (dict, int, str, or list): A dictionary mapping trigger keys to
                channel numbers or physical pin names, or unnamed channel values.
                Integer values specify logical channel numbers. Physical pin names
                receive automatically allocated logical IDs. Unnamed entries receive
                keys such as "trigger00", based on their logical channel number.
            groupName (str or None): Hardware bank used for numeric channel values.
            digitalIODevice (type or None): Device class to select; None uses the first device.
            pulseLen (int): Pulse duration in milliseconds.
        """
        if isinstance(channels, dict):
            entries = list(channels.items())
            explicitKeys = True
        elif isinstance(channels, (int, str)):
            entries = [(None, channels)]
            explicitKeys = False
        elif isinstance(channels, (list, tuple, range)):
            entries = [(None, value) for value in channels]
            explicitKeys = False
        else:
            raise TypeError("channels must be a dictionary, integer, string, or sequence of channel values")

        if not entries:
            return

        # Validate input before allocating IDs or changing hardware.
        for channelKey, value in entries:
            if explicitKeys and (not isinstance(channelKey, str) or not channelKey):
                pglMessages.warning("Each channelKey must be a nonempty string")
                return
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                pglMessages.warning("Each channel must be an integer or a physical pin name")
                return
            if isinstance(value, int) and value < 0:
                pglMessages.warning("Channel numbers must be nonnegative")
                return
            if isinstance(value, str) and not value:
                pglMessages.warning("Physical pin names must be nonempty")
                return

        device = self.getDigitalIODevice(digitalIODevice=digitalIODevice)
        if device is None:
            pglMessages.warning("No matching digitalIO device has been found. Make sure that devices have been configured in pgl.settings() and are properly connected")
            return

        # Reserve existing IDs and explicitly requested IDs before allocating new ones.
        usedChannels = {config.channel for config in self.settings.digitalIOChannels}
        usedChannels.update(device.digitalChannels)
        usedChannels.update(value for _, value in entries if isinstance(value, int))
        nextChannel = max(usedChannels, default=-1) + 1

        existingKeys = {config.channelKey for config in self.settings.digitalIOChannels}
        existingKeys.update(self.state.triggers)

        # Build and validate all metadata before configuring hardware.
        channelSettings = []
        for channelKey, value in entries:
            if isinstance(value, str):
                channel = nextChannel
                channelName = value
                nextChannel += 1
            else:
                channel = value
                channelName = None

            if channelKey is None:
                channelKey = f"trigger{channel:02d}"

            if channelKey in existingKeys:
                pglMessages.warning(f"Trigger key already configured: {channelKey}")
                return

            existingKeys.add(channelKey)
            config = pglDigitalIOChannel(channelKey=channelKey, channel=channel, channelName=channelName, groupName=groupName, pulseLen=pulseLen, deviceType=device.deviceType)
            channelSettings.append(config)

        existingChannels = set(device.digitalChannels)
        newChannels = [config.channel for config in channelSettings]

        if len(newChannels) != len(set(newChannels)) or existingChannels.intersection(newChannels):
            pglMessages.warning("Logical channel numbers must be unique on each device")
            return

        for config in channelSettings:
            if not device.setupDigitalOutput(channel=config.channel, channelName=config.channelName, groupName=config.groupName, pulseLen=config.pulseLen):
                pglMessages.warning(f"Failed to configure trigger {config.channelKey!r}")
                return

            self.settings.digitalIOChannels.append(config)
            self.state.triggers[config.channelKey] = (config, device)
    
    def configureTriggerWords(self, wordDict: dict, digitalIODevice=None, channels=None, pulseLen=5):
        """
        Configure named trigger words.

        Args:
            wordDict (dict[str, int]): Word names mapped to unsigned integer values.
            digitalIODevice (type or None): Device class to select; None uses the first device.
            channels (list[int | str] or None): Configured logical channel numbers or
                physical channel names, ordered from lowest to highest bit.
                None uses all configured channels in logical channel number order.
            pulseLen (int): Pulse duration in milliseconds, stored with each word.
        """
        device = self.getDigitalIODevice(digitalIODevice=digitalIODevice)
        if device is None:
            pglMessages.warning("No matching digitalIO device has been found. Make sure that devices have been configured in pgl.settings() and are properly connected")
            return

        if not isinstance(wordDict, dict):
            pglMessages.warning("wordDict must be a dictionary mapping word names to integer values")
            return

        configuredChannels = [config for config, configuredDevice in self.state.triggers.values() if configuredDevice is device]

        if channels is None:
            selectedChannels = sorted(configuredChannels, key=lambda config: config.channel)
        else:
            channels = [channels] if isinstance(channels, (int, str)) else list(channels)
            selectedChannels = []

            for channel in channels:
                if isinstance(channel, bool) or not isinstance(channel, (int, str)):
                    pglMessages.warning("Each channel must be an integer channel number or a string channel name")
                    return

                if isinstance(channel, int):
                    matches = [config for config in configuredChannels if config.channel == channel]
                else:
                    matches = [config for config in configuredChannels if config.channelName == channel]

                if not matches:
                    pglMessages.warning(f"Channel {channel!r} is not configured for {device.deviceType}")
                    return
                if len(matches) > 1:
                    pglMessages.warning(f"Channel {channel!r} matches multiple configured entries")
                    return

                selectedChannels.append(matches[0])

        channelNumbers = [config.channel for config in selectedChannels]

        if not channelNumbers:
            pglMessages.warning("At least one configured channel is required for trigger words")
            return
        if len(channelNumbers) != len(set(channelNumbers)):
            pglMessages.warning("Each logical channel can appear only once in a trigger word")
            return

        # Check the device's actual configuration before its setup method is called.
        missingChannels = set(channelNumbers) - device.digitalChannels.keys()
        if missingChannels:
            pglMessages.warning(f"Channels are not initialized on the selected device: {sorted(missingChannels)}")
            return

        maxValue = (1 << len(channelNumbers)) - 1
        existingKeys = {word.key for word in self.settings.digitalIOWords}
        existingKeys.update(self.state.triggerWords)

        # Validate all metadata before changing the device's word configuration.
        wordSettings = []
        for key, value in wordDict.items():
            if not isinstance(key, str) or not key:
                pglMessages.warning("Each word key must be a nonempty string")
                return
            if key in existingKeys:
                pglMessages.warning(f"Trigger word already configured for {device.deviceType}: {key}")
                return
            if isinstance(value, bool) or not isinstance(value, int):
                pglMessages.warning(f"Value for word '{key}' must be an integer")
                return
            if not 0 <= value <= maxValue:
                pglMessages.warning(f"Value for word '{key}' must be between 0 and {maxValue}")
                return

            word = pglDigitalIOWord(key=key, value=value, pulseLen=pulseLen, deviceType=device.deviceType)
            wordSettings.append(word)

        # check for existing configuration - 
        hasConfiguredWords = any(wordDevice is device for _, wordDevice in self.state.triggerWords.values())
        if hasConfiguredWords and channelNumbers != device.wordDigitalChanels:
            pglMessages.warning("Cannot change word channels while trigger words are configured on this device")
            return
        
        # configure the ouptut channels for writing words
        device.setupDigitalOutputWord(channels=channelNumbers)

        # and keep the settings and state 
        for word in wordSettings:
            self.settings.digitalIOWords.append(word)
            self.state.triggerWords[word.key] = (word, device)
    
    def sendTriggerWord(self, key, task=None):
        """Send a configured trigger word by its unique key."""
        target = self.state.triggerWords.get(key)
        
        if target is None:
            pglMessages.message(f"Trigger word {key!r} has not been configured")
            # setup trigger event
            e = pglEventOutputTrigger(triggerName=key, triggerWord=None, timestamp=pglTimestamp.getSecs(), triggerSent=False)
        else:
            # get the word and device to send
            word, device = target
            # send it
            timestamp = device.digitalOutputWord(word.value)
            if timestamp is None: 
                pglMessages.warning(f"Device did not return a timestamp",level=1)
                # setup trigger event
                e = pglEventOutputTrigger(triggerName=key, triggerWord=word.value, timestamp=pglTimestamp.getSecs(), triggerSent=False)
            else:
                # setup trigger event
                e = pglEventOutputTrigger(triggerName=key, triggerWord=word.value, timestamp=timestamp, triggerSent=True)

        # add event to events
        self.data.events.append(e)
        if task is not None: task.data.events.append(e)

    def sendTrigger(self, key, task=None):
        """Send a configured trigger; return its timestamp or None."""
        target = self.state.triggers.get(key)
        if target is None:
            pglMessages.message(f"Trigger {key!r} has not been configured")
            e = pglEventOutputTrigger(triggerName=key, triggerValue=None,
                                    timestamp=pglTimestamp.getSecs(), triggerSent=False)
            timestamp = None
        else:
            config, device = target
            timestamp = device.digitalOutputPulse(config.channel)
            if timestamp is None:
                pglMessages.warning("Device did not return a timestamp", level=1)
                e = pglEventOutputTrigger(triggerName=key, triggerValue=config.channel,
                                        timestamp=pglTimestamp.getSecs(), triggerSent=False)
            else:
                e = pglEventOutputTrigger(triggerName=key, triggerValue=config.channel,
                                        timestamp=timestamp, triggerSent=True)

        self.data.events.append(e)
        if task is not None:
            task.data.events.append(e)

        return timestamp    
    
    def endDigitalIO(self):
        """Stop shared inputs and close each experiment-owned hardware device."""
        errors = []
        failedDevices = []

        # Input shutdown must precede closing the shared hardware connection.
        # A reported acquisition error must not prevent other cleanup attempts.
        self._attemptRunAction("stop experiment input devices", self.endDevices, errors)

        # One device failure must not prevent closing the others.
        for digitalIODevice in list(self.state.digitalIODevices):
            closed = self._attemptRunAction(
                "close digital IO device",
                digitalIODevice.close,
                errors,
            )

            if not closed:
                failedDevices.append(digitalIODevice)
                continue

            # Successful close means this object must no longer be polled.
            self.pgl.devicesRemove(digitalIODevice)
            self._experimentInputDevices = [
                device for device in getattr(self, "_experimentInputDevices", [])
                if device is not digitalIODevice
            ]

        # Retain failed devices so cleanup can be retried.
        self.state.digitalIODevices = failedDevices
        if not failedDevices:
            self.state.triggers = {}
            self.state.triggerWords = {}

        if errors: raise errors[0]
        return True
            
    def initDevices(self):
        """Snapshot LabJack acquisition and experiment repair policy together."""
        from copy import deepcopy
        from .pglLabJack import pglLabJack

        self.endDevices()
        self._experimentInputDevices = []
        self._labJackTriggerConfig = None
        self._labJackAcquisitionId = None
        self._labJackRepairConfig = None

        for deviceName in self.settings.devices:
            pglMessages.warning(
                f"Input device is not implemented yet: {deviceName}", level=1,
            )

        selectedNames = {name.casefold() for name in self.settings.digitalIO}
        if "labjack digital trigger" not in selectedNames: return

        preset = getattr(self, "_labJackPresetSnapshot", None)
        if preset is None:
            raise RuntimeError("LabJack preset was not resolved before device construction.")
        preset = deepcopy(preset)
        config = preset.getDigitalReadConfig()
        repairConfig = preset.getVolumeTriggerRepairConfig()

        labJack = self.getDigitalIODevice(pglLabJack)
        if labJack is None or not labJack.isActive:
            raise RuntimeError(
                "LabJack digital trigger was selected, but no initialized "
                "LabJack is available"
            )

        # Both snapshots come from the same preset. Mid-run edits affect neither.
        self._labJackTriggerConfig = deepcopy(config)
        self._labJackRepairConfig = deepcopy(repairConfig)
        self.data.labJackAcquisition = {
            "presetName": preset.name,
            "presetUUID": preset.uuid,
            "requested": deepcopy(config),
            "volumeTriggerRepair": deepcopy(repairConfig),
            "clockCalibration": preset.getClockCalibrationConfig(),
            "clockEvents": labJack.getClockEvents(),
            "actualScanRate": None,
            "streamBufferSizeBytes": None,
            "acquisitionId": None,
            "deviceUUID": str(labJack.uuid),
            "serialNumber": getattr(labJack, "serialNumber", None),
        }

        # Track ownership before startup so partial initialization can stop.
        self._experimentInputDevices.append(labJack)
        actualRate = labJack.startDigitalRead(**config)

        if not np.isfinite(actualRate) or actualRate <= 0:
            raise RuntimeError(f"Invalid actual LabJack scan rate: {actualRate}")

        status = labJack.getDigitalReadStatus()
        acquisitionId = status.get("acquisitionId")
        if not isinstance(acquisitionId, str) or not acquisitionId:
            raise RuntimeError("LabJack startup returned no acquisition identity.")

        self._labJackAcquisitionId = acquisitionId
        self._retainLabJackClockEvents(labJack)
        self.data.labJackAcquisition.update({
            "acquisitionId": acquisitionId,
            "actualScanRate": float(actualRate),
            "streamBufferSizeBytes": status.get("streamBufferSizeBytes"),
        })

        previousHistory = getattr(self, "_verifiedVolumeTriggerHistory", None)
        if previousHistory is not None:
            if not hasattr(self, "_previousVerifiedVolumeTriggerHistories"):
                self._previousVerifiedVolumeTriggerHistories = []
            self._previousVerifiedVolumeTriggerHistories.append(previousHistory)

        self._verifiedVolumeTriggerHistory = {
            "source": (acquisitionId, config["channels"], config["edge"]),
            "lastVerifiedTrigger": None,
            "intervals": [],
            "handledWindowIds": set(),
        }

        if not np.isclose(actualRate, config["scanRate"], rtol=1e-6, atol=0):
            pglMessages.message(
                f"LabJack requested {config['scanRate']:g} Hz; "
                f"using device rate {actualRate:g} Hz for timestamps."
            )

        if not self.pgl.devicesAdd(labJack):
            raise RuntimeError("Could not register LabJack input for polling")

        pglMessages.message(
            f"LabJack scanner-trigger input initialized with preset "
            f"{preset.name!r}; counter verification="
            f"{config['verifyTriggerCounter']}, automatic restart="
            f"{config['autoRestartOnStreamOverflow']}"
        )
            

    def endDevices(self):
        """
        Stop owned inputs, retain queued events, snapshot, then unregister.

        Drain only after both worker and stream have stopped. A latched
        acquisition error must not prevent retaining already-queued events.

        Shutdown-drained events are retained as source evidence only:
        no additional accepted volumes or retrospective task updates.
        """
        errors = []
        failedDevices = []
        devices = list(getattr(self, "_experimentInputDevices", []))

        for device in devices:
            self._attemptRunAction(
                "stop experiment digital input",
                device.stopDigitalRead,
                errors,
            )

            try:
                status = device.getDigitalReadStatus()
            except Exception as error:
                errors.append(error)
                self._recordRunError("inspect digital input shutdown", error)
                failedDevices.append(device)
                continue

            if status["running"] or status["streamActive"]:
                # Do not race a live producer or drop ownership.
                failedDevices.append(device)
                continue

            try:
                # LabJack.poll() atomically drains the complete queue before
                # exposing a retained failure on a subsequent empty poll.
                # One call is sufficient now that the producer has stopped.
                pending = list(device.poll())

                if pending:
                    self.data.events.extend(pending)

                    if not hasattr(self.data, "digitalInputShutdownDrains"):
                        self.data.digitalInputShutdownDrains = []

                    self.data.digitalInputShutdownDrains.append({
                        "deviceUUID": str(device.uuid),
                        "acquisitionId": status.get("acquisitionId"),
                        "drainedTimestamp": pglTimestamp.getSecs(),
                        "experimentEndTime": getattr(self.data, "endTime", None),
                        "eventCount": len(pending),
                        # Retain direct references as well as data.events.
                        # Later chronological insertion cannot invalidate
                        # this association through shifted list indices.
                        "events": list(pending),
                        "processing": "retainedSourceOnly",
                        "reason": "drainedAfterAcquisitionLoop",
                    })

            except Exception as error:
                # poll() can expose the same fatal acquisition error already
                # reported by stopDigitalRead(). Do not hide new drain errors.
                if status.get("error") is None or status.get("pendingEvents", 0):
                    errors.append(error)
                    self._recordRunError("drain stopped digital input", error)
                    failedDevices.append(device)
                    continue

            # Snapshot while the device is still available to the experiment.
            # The final data.json write persists the drained events and
            # post-shutdown diagnostics after the initial local save.
            try:
                self.pgl.devicesRemove(device)
            except Exception as error:
                errors.append(error)
                self._recordRunError("unregister stopped digital input", error)
                failedDevices.append(device)

        self._attemptRunAction(
            "snapshot digital input diagnostics",
            self._snapshotDigitalInputStatus,
            errors,
        )

        self._experimentInputDevices = failedDevices
        if errors: raise errors[0]
        return True
        
    def initEyeTracker(self):
        '''Initialize eye tracker if we have an eye tracker.'''
        # load eye tracker settings
        self.eyeTrackerSettings = self.pgl.getEyeTrackerSettings(settings=self.settings)
        
        if self.settings.eyetracker[0] == "Eyelink":
            from .pglEyelink import pglEyelink
            # set edf filename to current date (note it has to be 8.3 characters
            # since SR Research has progressed from the days of DOS)
            self.settings.edfFilename = f"{datetime.now().strftime('%Y%m%d')}"

            # init the eyeink
            pglMessages.message(f"(pglExperiment) Initialize Eyelink with filename: {self.settings.edfFilename}",messageType='experiment')
            self.eyeTracker = pglEyelink(pgl=self.pgl, edfFilename=self.settings.edfFilename)  

            # check if running
            if self.eyeTracker.eyelink is None:
                self.eyeTracker = None      

            # get from settings
            nEyelinkCalibrationPoints = self.eyeTrackerSettings.nEyelinkCalibrationPoints
            eyelinkCalibrationProportion = self.eyeTrackerSettings.eyelinkCalibrationProportion
 
            if self.eyeTracker is not None:
                self.eyeTracker.setCustomCalibrationPoints(margin=eyelinkCalibrationProportion, numPoints=nEyelinkCalibrationPoints)

        elif self.settings.eyetracker[0] == "None":
            self.eyeTracker = None
        else:
            pglMessages.warning(f"Unknown eye tracker type {self.settings.eyetracker[0]}")
            self.eyeTracker = None


    def calibrateEyeTracker(self):
        '''
        Run eye tracker calibration if we have an eye tracker and it is not calibrated yet.
        '''
        if self.eyeTracker is not None:
            # wait for key press to calibrate
            self.state.waitingForCalibration = True
            self.state.runCalibration = False
            # FIX: These could be exposed 
            self.settings.calibrateKey = 'space'
            self.settings.calibrateKeyCode = self.keyboardMouse.charToKeyCode(self.settings.calibrateKey)
            self.settings.skipCalibrationKey = 'return'
            self.settings.skipCalibrationKeyCode = self.keyboardMouse.charToKeyCode(self.settings.skipCalibrationKey)
            # instructions
            displayText = f"Press {self.settings.calibrateKey} to calibrate eye tracker. {self.settings.skipCalibrationKey} to skip."
            pglMessages.message(displayText, messageType='experiment')

            k = self.pgl.devicesGetKeyboard()            
            eatKeys = k.eatKeyCodes

            # eat relevant keys
            self.pgl.setEatKeys(keyChars=[self.settings.calibrateKey, self.settings.skipCalibrationKey])

            # wait till we get a response
            while self.state.waitingForCalibration:
                self.pgl.text(displayText, y=0)
                # flush to display text
                self.pgl.flush()
                # poll for events
                events = self.pgl.poll()
                self.data.events.extend(events)

                # see if we have a match to startKey
                if [e for e in events if e.type == "keyboard" and e.eventType == "keydown"and e.keyCode == self.settings.calibrateKeyCode]:
                    self.state.waitingForCalibration = False
                    self.state.runCalibration = True
                elif [e for e in events if e.type == "keyboard" and e.eventType == "keydown" and e.keyCode == self.settings.skipCalibrationKeyCode]:
                    self.state.waitingForCalibration = False
                    self.state.runCalibration = False
                    pglMessages.message("Skipping eye tracker calibration.",messageType='experiment')

            # reset eat keys
            self.pgl.setEatKeys(eatKeys)    

            # if we should run calibration, then do it
            if self.state.runCalibration:
                self.eyeTracker.calibrate()
                # and restart saving data, as calibrate seems to stop it
                self.eyeTracker.start()


    def saveEyeTrackerEvent(self, eventType="segment", taskID=None, trialNum=None, segmentNum=None, timestamp=None, phaseNum=None):
        '''Save an eye tracker event for synchronization. This is called by tasks during updates if settings.saveEyeTracker is True.'''
        self.eyeTracker.sendMessage(f"pgl: {eventType} taskID={taskID} trialNum={trialNum} segmentNum={segmentNum} timestamp={timestamp} phaseNum={phaseNum}")

    def startPhase(self, phaseNum=0):
        """Start available tasks, skipping empty phases without index confusion."""
        self.state.currentPhaseIndex = self.state.phaseNums.index(phaseNum)

        while self.state.currentPhaseIndex < len(self.state.phaseNums):
            self.state.phaseNum = self.state.phaseNums[self.state.currentPhaseIndex]

            pglMessages.message(
                f"Starting phase: {self.state.currentPhaseIndex + 1}/{len(self.state.phaseNums)}",
                messageType='experiment',
            )

            candidates = [
                task for task in self.tasks
                if (
                    task.data.endTime is None
                    and (
                        task.settings.phaseNum is None
                        or task.settings.phaseNum == self.state.phaseNum
                    )
                )
            ]

            self.currentTasks = []
            startTime = self.pgl.getSecs()

            for task in candidates:
                if task.start(startTime) == False:
                    pglMessages.warning(
                        f"(pglExperiment:startPhase) Task "
                        f"{task.settings.taskName} did not start."
                    )
                    continue
                self.currentTasks.append(task)

            if self.currentTasks: return

            self.state.currentPhaseIndex += 1

        self.currentTasks = []
        self.state.experimentDone = True

    def isAbortedRun(self):
        """Decide whether this run should be saved under trash.

        Explicitly marked main tasks take precedence. If none are marked,
        all tasks are candidates. A run is retained if any candidate has
        enough recorded trial starts.

        Finite tasks require their configured nTrials. Infinite tasks retain
        the existing minimum of ten trials.

        This is a trial-count heuristic, not a completion or data-quality
        check. It does not change acquisition error reporting.
        """
        # Background tasks cannot disqualify a run when main tasks are marked.
        mainTasks = [task for task in self.tasks if getattr(task.settings, "isMainTask", False)]
        candidates = mainTasks if mainTasks else list(self.tasks)

        if not candidates:
            pglMessages.message("Detected aborted run: no tasks are available for trial-count classification.", messageType='experiment', emphasize=True)
            return True

        shortfalls = []

        for task in candidates:
            # Count recorded trial starts rather than interpreting currentTrial,
            # which also changes when a finite task completes.
            nTrialsRun = task.data.nTrials
            configuredTrials = task.settings.nTrials
            requiredTrials = 10 if np.isinf(configuredTrials) else configuredTrials

            # Any qualifying candidate is sufficient to retain the run.
            if nTrialsRun >= requiredTrials: return False

            shortfalls.append(f"{task.settings.taskSaveName}: {nTrialsRun}/{requiredTrials:g} trials started")

        # Report failure only after checking every candidate.
        candidateType = "main tasks" if mainTasks else "candidate tasks"
        pglMessages.message(f"Detected aborted run: none of the {candidateType} reached its trial threshold. " + "; ".join(shortfalls), messageType='experiment', emphasize=True)
        return True
    
    ################################################################
    # Save experiment components independently
    ################################################################
    def _saveRunContents(self, includeEyeTracker=True):
        """Save available run contents, continuing after component failures.

        Repeated calls reuse this run's directory. Each acquisition task saves
        its original defining module or notebook cell alongside its data.

        Experiment recording buffers are automatically released only after
        HDF5 and its JSON reference are successfully saved.

        The bundle is not atomic, although recording.h5 is published through
        its own temporary-file replacement.
        """
        errors = []
        dataPath = getattr(self, "_runSavePath", None)

        ################################################################
        # Choose and retain this run's destination
        ################################################################
        if dataPath is None:
            # Failed aborted-run classification must not prevent saving.
            aborted = False
            if self.settings.saveAbortedRunsToTrash:
                try:
                    aborted = self.isAbortedRun()
                except (Exception, KeyboardInterrupt, SystemExit) as error:
                    errors.append(error)
                    self._recordRunError("classify aborted run", error)

            runName = Path(self.experimentSettings.runName)
            if aborted: runName = Path("trash") / runName

            basePath = (
                Path(self.settings.dataPath).expanduser()
                / self.experimentSettings.experimentSaveName
                / self.experimentSettings.subjectID
                / self.experimentSettings.sessionName
                / runName
            )

            # Never overwrite a previous run. Retry directory-name collisions.
            dataPath = basePath
            suffix = datetime.now().strftime("%H-%M-%S-%f")
            attempt = 0
            while True:
                try:
                    dataPath.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    attempt += 1
                    dataPath = basePath.parent / f"{basePath.name}_{suffix}_{attempt}"

            # Retain the destination immediately for discovery and retries.
            self._runSavePath = dataPath
            self.fullDataPath = str(dataPath)

        pglMessages.message(f"Saving experiment data to: {dataPath}", messageType='experiment')
        self.data.saveStatus = "inProgress"


        ################################################################
        # Save essential experiment components before device operations
        ################################################################
        # Capture diagnostics before saving, without waiting for device shutdown.
        self._attemptRunAction("snapshot digital input diagnostics", self._snapshotDigitalInputStatus, errors)

        components = (
            ("experiment data", self.data, "data.json"),
            ("experiment state", self.state, "state.json"),
            ("experiment settings", self.experimentSettings, "experimentSettings.json"),
            ("settings", self.settings, "settings.json"),
        )

        for label, component, filename in components:
            self._attemptRunAction(
                f"save {label}",
                lambda component=component, filename=filename: component.saveStrict(dataPath / filename),
                errors,
            )

        # Preserve the parent's optional traits.json convention.
        if set(self.traits()) != set(pglTraitSettings().traits()):
            self._attemptRunAction(
                "save experiment traits",
                lambda: pglSerialize.saveStrict(self, filename=dataPath / "traits.json"),
                errors,
            )

        ################################################################
        # Save each task's source, data, state, settings, and parameters
        ################################################################
        for taskIndex, task in enumerate(self.tasks):
            try:
                taskPath = dataPath / task._getTaskDirectoryName()
                taskPath.mkdir(parents=True, exist_ok=True)
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                errors.append(error)
                self._recordRunError(f"create task {taskIndex + 1} directory", error)
                continue

            # Write the definition-time snapshot, not the current source file
            # or latest notebook cell. A source failure must not prevent saving
            # this task's acquired data or its remaining components.
            self._attemptRunAction(
                f"save task {taskIndex + 1} source",
                lambda task=task, taskPath=taskPath: task.saveSourceStrict(taskPath),
                errors,
            )

            # Attempt each task component independently.
            for componentName in ("data", "state", "settings"):
                self._attemptRunAction(
                    f"save task {taskIndex + 1} {componentName}",
                    lambda task=task, componentName=componentName, taskPath=taskPath:
                        getattr(task, componentName).saveStrict(taskPath / f"{componentName}.json"),
                    errors,
                )

            # Stimulus export is independent of task JSON and parameter saving.
            # Failures are recorded, and savePendingData() can retry this step.
            self._attemptRunAction(
                f"save task {taskIndex + 1} stimuli",
                lambda task=task, taskPath=taskPath: task.saveStimuliStrict(taskPath),
                errors
            )

            # Parameters have their own settings/state/data bundles, including
            # nested parameter bundles and saved random-number-generator state.
            parameterPath = taskPath / "parameters"
            if self._attemptRunAction(
                f"create task {taskIndex + 1} parameters directory",
                lambda parameterPath=parameterPath: parameterPath.mkdir(parents=True, exist_ok=True),
                errors,
            ):
                for parameterIndex, parameter in enumerate(task.parameters):
                    self._attemptRunAction(
                        f"save task {taskIndex + 1} parameter {parameterIndex + 1}",
                        lambda parameter=parameter, parameterPath=parameterPath:
                            parameter.saveBundleStrict(dataPath=parameterPath),
                        errors,
                    )

        ################################################################
        # Save cached PGL state and the experiment recording
        ################################################################

        # Cached PGL state does not require a renderer query.
        self._attemptRunAction("save PGL state", lambda: self.pgl.saveStrict(dataPath / "pgl.json"), errors)

        # HDF5 failure must not prevent saving the other experiment files.
        # Repeated saves after release reuse the existing recording artifact.
        if getattr(self, "_recordingId", None) is not None:
            saved = self._attemptRunAction(
                "save experiment recording",
                lambda: self._saveExperimentRecording(dataPath),
                errors,
            )

            if not saved:
                reference = dict(getattr(self.data, "recording", {}))
                reference["recordingId"] = self._recordingId
                reference["saveStatus"] = "failed"
                self.data.recording = reference

        ################################################################
        # Optionally retrieve eye-tracker data
        ################################################################

        # Direct save() callers retain EDF retrieval. run() postpones it until
        # local saving and screen/device cleanup have been attempted.
        if includeEyeTracker and self.eyeTracker is not None:
            def saveEyeTracker():
                filename = dataPath / self.settings.eyetracker[0].lower()
                return self.eyeTracker.save(filename)

            saved = self._attemptRunAction("save eye tracker", saveEyeTracker, errors)
            self.data.eyeTrackerSaveStatus = "complete" if saved else "failed"

        ################################################################
        # Persist save status and recording reference before releasing RAM
        ################################################################
        self.data.saveStatus = "failed" if errors else "complete"
        referenceSaved = self._attemptRunAction(
            "save final save status",
            lambda: self.data.saveStrict(dataPath / "data.json"),
            errors,
        )
        if not referenceSaved: self.data.saveStatus = "failed"

        # Release only after data.json successfully records a complete HDF5
        # reference. Other component failures do not invalidate that artifact.
        reference = getattr(self.data, "recording", {})
        if (
            referenceSaved
            and getattr(self, "_recordingId", None) is not None
            and reference.get("saveStatus") == "complete"
        ):
            if not self._attemptRunAction(
                "release saved experiment recording",
                self._releaseExperimentRecording,
                errors,
            ):
                # Buffer-release failure is separate from artifact success:
                # do not relabel the successfully written HDF5 as failed.
                self.data.saveStatus = "failed"
                self._attemptRunAction(
                    "save recording release failure",
                    lambda: self.data.saveStrict(dataPath / "data.json"),
                    errors,
                )

        # Report the first failure only after all independent saves were tried.
        if errors: raise errors[0]
        return dataPath
    
    def getLastRun(self, task=None, allRuns=False):
        """Load saved runs or their matching tasks.

        Args:
            task (pglTask, optional): A task from self.tasks. When provided,
                return the task at the same index in each selected run.
            allRuns (bool): Return all matches, newest first, instead of
                only the latest run's match.

        Returns:
            When allRuns=False:
                pglRun, pglTask, or None if no match exists.
            When allRuns=True:
                A list of pglRun or pglTask objects, or an empty list.
                Tasks missing from a saved run are omitted.
        """
        try:
            from .pglSession import pglRun

            emptyResult = [] if allRuns else None

            # Resolve the requested task's position in the current experiment.
            taskIndex = None
            if task is not None:
                taskIndex = next(
                    (index for index, currentTask in enumerate(self.tasks)
                    if currentTask is task),
                    None,
                )

                if taskIndex is None:
                    pglMessages.warning(
                        f"Could not find matching task for {task.settings.taskName}"
                    )
                    return emptyResult

            # Directory containing this session's saved runs.
            dataPath = (
                Path(self.settings.dataPath).expanduser()
                / self.experimentSettings.experimentSaveName
                / self.experimentSettings.subjectID
                / self.experimentSettings.sessionName
            )

            if not dataPath.is_dir():
                return emptyResult

            requiredFiles = (
                "experimentSettings.json",
                "settings.json",
                "state.json",
                "data.json",
            )

            def isRunDirectory(directory):
                return directory.is_dir() and all(
                    (directory / fileName).is_file()
                    for fileName in requiredFiles
                )

            # Look for runDirs
            runDirs = (
                directory
                for directory in dataPath.iterdir()
                if isRunDirectory(directory)
            )

            runs = [
                pglRun(fullDataPath=runDir)
                for runDir in runDirs
            ]

            if not runs:
                return emptyResult

            runs.sort(key=lambda run: run.data.startTime, reverse=True)

            # Only inspect the latest run unless all runs were requested.
            selectedRuns = runs if allRuns else runs[:1]

            if task is None:
                return selectedRuns if allRuns else selectedRuns[0]

            matchingTasks = []
            for run in selectedRuns:
                savedTask = run.getTaskAt(taskIndex)
                if savedTask is not None:
                    matchingTasks.append(savedTask)

            if allRuns:
                return matchingTasks

            return matchingTasks[0] if matchingTasks else None
            
        except Exception as e:
            pglMessages.warning(f"Unable to load last run: {e}")

    def getLastRunParameters(self, task=None):
        '''
        Get the parameters from the last run
        '''
        lastRun = self.getLastRun(task)
        if lastRun: 
            return lastRun.parameters
        else:
            None
            
    def printLastRuns(self, task):
        '''
        displays last runs in session
        eex
        Args:
            task (pglTask): If set will display how many trials were run in specified task
            
        '''
        try:
            self.setVerbose(verbose=False)
            tasks = self.getLastRun(task=task, allRuns=True)

            self.setVerbose()
            if tasks: self.pgl.printHeader("previous runs from this session", messageType='experiment')
                        
            for iRun, task in enumerate(tasks):
                self.setVerbose()
                pglMessages.print(f"{iRun:2d}:{task.settings.taskSaveName} nTrials={task.state.currentTrial:3d} ran {pglTimestamp.formatDuration(self.pgl.getSecs()-task.data.startTime)} ago", messageType='experiment')
                self.setVerbose(verbose=False)
            
            self.setVerbose()
            if tasks: self.pgl.printHeader(messageType='experiment')

            
        except Exception as e:
            pglMessages.warning(f"Unable to load last run: {e}")

    def setVerbose(self, verbose=True):
        if not self.settings.verbose:
            # only let experiment and parameter message through (warnings will also print)
            if verbose:
                pglMessages.setEnabledTypes(['experiment','parameter'])
            else:
                pglMessages.setEnabledTypes(set())
            self.pgl.verbose=False

    def endVerbose(self):
        if not self.settings.verbose:
            # Restore verbose settings
            self.pgl.verbose=True
            pglMessages.setEnabledTypes(None)
            
    ################################################################
    # Start this experiment's recording
    ################################################################
    def _startExperimentRecording(self, estimatedDuration, recordFlush, recordCommands):
        """Start capture before task configuration and bind it to this experiment."""
        if getattr(self, "_recordingId", None) is not None:
            raise RuntimeError("This experiment already owns a recording. Create a new experiment for another run.")

        self.pgl.recordingStart(
            estimatedDuration=estimatedDuration,
            recordFlush=recordFlush,
            recordCommands=recordCommands,
        )

        metadata = self.pgl._recordingMetadata
        self._recordingId = metadata["recordingId"]

        # Store identifiers, not a reference to the live experiment.
        metadata["experiment"] = {
            "experimentUUID": self.uuid,
            "experimentName": self.experimentSettings.experimentName,
            "experimentSaveName": self.experimentSettings.experimentSaveName,
            "subjectID": self.experimentSettings.subjectID,
            "sessionName": self.experimentSettings.sessionName,
            "runName": self.experimentSettings.runName,
        }

        # Filename is published only after a successful HDF5 write.
        self.data.recording = {
            "recordingId": self._recordingId,
            "filename": None,
            "formatVersion": None,
            "saveStatus": "notAttempted",
        }

    ################################################################
    # Verify recording ownership without exporting the buffers
    ################################################################
    def _getExperimentRecording(self):
        """Return owned live metadata, or None after automatic release.

        A released experiment must never access a later experiment's buffers.
        Its saved artifact is handled separately by _saveExperimentRecording().
        """
        recordingId = getattr(self, "_recordingId", None)
        if recordingId is None: return None
        if getattr(self, "_recordingReleased", False): return None

        metadata = getattr(self.pgl, "_recordingMetadata", None)
        if not isinstance(metadata, dict) or metadata.get("recordingId") != recordingId:
            raise RuntimeError("This experiment's recording was cleared or replaced; refusing to use another recording.")

        if metadata.get("experiment", {}).get("experimentUUID") != self.uuid:
            raise RuntimeError("Recording ownership does not match this experiment.")

        if getattr(self.pgl, "_recordingState", None) not in ("active", "stopped"):
            raise RuntimeError("This experiment's recording session is unavailable.")

        return metadata
    
    ################################################################
    # Save the owned recording into the run directory
    ################################################################
    def _saveExperimentRecording(self, dataPath):
        """Save owned capture or reuse its already-published artifact.

        Buffers are released separately, only after data.json successfully
        persists the recording reference.
        """
        import h5py
        from .pglRecordingFile import pglRecordingFile

        filename = (Path(dataPath) / "recording.h5").resolve()

        if getattr(self, "_recordingReleased", False):
            # Do not inspect, save, or clear the current PGL recording here:
            # it may now belong to another experiment.
            savedPath = getattr(self, "_recordingSavedPath", None)
            reference = getattr(self, "_recordingSavedReference", None)

            if savedPath is None or filename != Path(savedPath):
                raise RuntimeError("Released recording cannot be saved to a different run directory.")
            if not isinstance(reference, dict) or reference.get("recordingId") != self._recordingId:
                raise RuntimeError("Released recording has no valid saved reference.")
            if not filename.is_file():
                raise FileNotFoundError(f"Saved recording is missing: {filename}")

            # Read only the small root metadata, not the recording arrays.
            with h5py.File(filename, "r") as file:
                if file.attrs.get("formatName") != pglRecordingFile.formatName:
                    raise RuntimeError("Saved recording artifact has an unexpected format.")
                if file.attrs.get("formatVersion") != reference["formatVersion"]:
                    raise RuntimeError("Saved recording artifact has an unexpected version.")
                metadata = pglRecordingFile._readMetadata(file)

            if (metadata.get("recordingId") != self._recordingId
                    or metadata.get("experiment", {}).get("experimentUUID") != self.uuid):
                raise RuntimeError("Saved recording artifact does not belong to this experiment.")

            self.data.recording = dict(reference)
            return filename

        metadata = self._getExperimentRecording()
        if metadata is None: return None

        if self.pgl._recordingState != "stopped":
            raise RuntimeError("Stop acquisition before saving the experiment recording.")
        if self.pgl.commandRecording or self.pgl.flushRecording:
            raise RuntimeError("Low-level capture was restarted after session finalization.")

        # Add task-run boundaries without changing the raw recording clock.
        metadata["experimentStartTime"] = self.data.startTime
        metadata["experimentEndTime"] = self.data.endTime

        self.pgl.recordingSave(filename)

        # Small reference only: recording buffers never enter experiment JSON.
        reference = {
            "recordingId": metadata["recordingId"],
            "filename": "recording.h5",
            "formatVersion": pglRecordingFile.formatVersion,
            "saveStatus": "complete",
        }
        self.data.recording = dict(reference)

        # Retain only publication bookkeeping, not a copy of captured data.
        self._recordingSavedPath = filename
        self._recordingSavedReference = dict(reference)
        return filename
        
    ################################################################
    # release experiment recording
    ################################################################
    def _releaseExperimentRecording(self):
        """Release this experiment's saved buffers after reference publication.

        Called by save() only after data.json was successfully written.
        Never release an active, replaced, or foreign recording.
        """
        if getattr(self, "_recordingId", None) is None: return
        if getattr(self, "_recordingReleased", False): return

        reference = getattr(self.data, "recording", {})
        if (reference.get("recordingId") != self._recordingId
                or reference.get("saveStatus") != "complete"):
            raise RuntimeError("Cannot release a recording without a successful saved reference.")

        metadata = self._getExperimentRecording()
        if metadata is None or self.pgl._recordingState != "stopped":
            raise RuntimeError("Cannot release an unavailable or active experiment recording.")
        if self.pgl.commandRecording or self.pgl.flushRecording:
            raise RuntimeError("Cannot release recording while low-level capture is active.")

        self.pgl.recordingClear()
        self._recordingReleased = True
        
    ################################################################
    # Report save/cleanup progress without relying on message filters
    ################################################################
    def _reportRunProgress(self, message):
        """Print timestamped progress without allowing output errors to abort saving."""
        try:
            print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)
        except (Exception, KeyboardInterrupt):
            # A failed output channel must not prevent the actual save.
            # An interrupt delivered during this brief print is ignored.
            pass
        
    ################################################################
    # Explain how to retry without rerunning acquisition
    ################################################################
    def _reportSaveRecovery(self):
        """Report the destination and recovery instructions for the live object."""
        dataPath = getattr(self, "_runSavePath", None)
        saveStatus = getattr(self.data, "saveStatus", "unknown")

        self._reportRunProgress(f"Current local save status: {saveStatus}")
        self._reportRunProgress(
            f"Run directory: {dataPath if dataPath is not None else 'not allocated yet'}"
        )
        self._reportRunProgress(
            "Experiment/task data objects remain attached to this experiment. "
            "Keep this object and kernel alive; do not rerun acquisition."
        )

        if getattr(self, "_recordingReleased", False):
            self._reportRunProgress(
                "Recording buffers were released after the recording and its "
                "JSON reference were reported saved. Retry uses recording.h5."
            )
        elif getattr(self, "_recordingId", None) is not None:
            self._reportRunProgress(
                "Recording release is not confirmed. Retain the PGL object too; "
                "do not call recordingClear(). Recording availability will be "
                "checked when retrying."
            )

        self._reportRunProgress(
            "After this cell returns, retry local saving with "
            "e.savePendingData() — replace e with your experiment variable."
        )
        self._reportRunProgress(
            "This retry excludes eye-tracker transfer and device cleanup."
        )

    ################################################################
    # Public save entry point with explicit completion reporting
    ################################################################
    def save(self, includeEyeTracker=True):
        """Enable detailed progress during saving, then restore quiet cleanup."""
        previousProgress = getattr(self, "_saveProgressActive", False)

        try:
            self._saveProgressActive = True

            # Decorative output must not prevent saving.
            try:
                print(
                    "\n"
                    "================================================================\n"
                    "  SAVING EXPERIMENT DATA\n"
                    "================================================================",
                    flush=True,
                )
            except (Exception, KeyboardInterrupt):
                pass

            self._reportRunProgress(
                "SAVE STARTED "
                f"(eye-tracker transfer {'included if present' if includeEyeTracker else 'excluded'})."
            )

            try:
                dataPath = self._saveRunContents(
                    includeEyeTracker=includeEyeTracker
                )

            except (Exception, KeyboardInterrupt, SystemExit):
                # Also catch failures outside the per-component helper,
                # such as interruption while creating the run directory.
                self.data.saveStatus = "failed"
                self._reportRunProgress(
                    "SAVE CALL FAILED OR WAS INTERRUPTED. "
                    "Some components may have saved successfully; "
                    "the complete requested save is not confirmed."
                )
                self._reportSaveRecovery()
                raise

            self._reportRunProgress(f"SAVE PASS COMPLETE: {dataPath}")

            if not includeEyeTracker:
                self._reportRunProgress(
                    "Local save operations returned successfully. "
                    "Eye-tracker transfer was not included."
                )

            # Only print the finished header after a successful save pass.
            try:
                print(
                    "\n"
                    "================================================================\n"
                    "  FINISHED SAVING EXPERIMENT DATA\n"
                    "================================================================",
                    flush=True,
                )
            except (Exception, KeyboardInterrupt):
                pass

            return dataPath

        finally:
            # Restore even after a failed/interrupted save so subsequent
            # cleanup does not emit routine START/DONE messages.
            self._saveProgressActive = previousProgress
    
    ################################################################
    # Retry local saving without rerunning tasks or device cleanup
    ################################################################
    def savePendingData(self):
        """
        Retry local saving after acquisition/finalization has stopped.

        Call only after the original run cell has returned.
        Does not recalibrate clocks, correct timestamps, run task-ending hooks,
        display results, close devices, or retrieve eye-tracker data.

        Clock persistence status is reconciled only after a successful save.
        Earlier failures and every retry attempt remain retained.
        """
        from uuid import uuid4

        errors = []
        clockRecord = getattr(self.data, "labJackClockFinalization", None)
        retry = None

        if isinstance(clockRecord, dict):
            if not hasattr(self.data, "labJackClockSaveRetryAudit"):
                self.data.labJackClockSaveRetryAudit = []

            retry = {
                "retryId": str(uuid4()),
                "startedTimestamp": pglTimestamp.getSecs(),
                "completedTimestamp": None,
                "previousCorrectedSaveStatus": clockRecord.get("correctedSaveStatus"),
                "status": "inProgress",
                "failureReason": None,
                "calibrationRepeated": False,
                "eventCorrectionRepeated": False,
            }
            self.data.labJackClockSaveRetryAudit.append(retry)

        def stopOwnedRecording():
            metadata = self._getExperimentRecording()
            if metadata is not None and self.pgl._recordingState == "active":
                return self.pgl.recordingStop(
                    outcome=getattr(self, "_recordingOutcome", "error")
                )

        try:
            # A stop failure must not prevent attempting experiment/task saves.
            self._attemptRunAction(
                "stop owned recording for save retry",
                stopOwnedRecording,
                errors,
            )

            dataPath = self.save(includeEyeTracker=False)

            if errors:
                self._reportRunProgress(
                    "The save pass returned, but the recording-stop attempt failed. "
                    "Review the error before considering recovery complete."
                )
                raise errors[0]

            if retry is not None:
                retry["completedTimestamp"] = pglTimestamp.getSecs()
                retry["status"] = "complete"
                clockRecord["correctedSaveStatus"] = "complete"
                clockRecord["lastSuccessfulSaveRetryId"] = retry["retryId"]

                # save() already saved all components. Persist the reconciliation
                # afterward without repeating task or recording saves.
                self.data.saveStrict(Path(dataPath) / "data.json")

            return dataPath

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            if retry is not None:
                retry["completedTimestamp"] = pglTimestamp.getSecs()
                retry["status"] = "failed"
                retry["failureReason"] = f"{type(error).__name__}: {error}"
                clockRecord["correctedSaveStatus"] = "failed"

            self.data.saveStatus = "failed"
            self._recordRunError("save pending data retry", error)
            self._reportRunProgress(
                "Save retry did not complete. Pending data and retry diagnostics "
                "remain in memory. Some files may have been written; the final "
                "on-disk status is not confirmed."
            )
            raise
    
    ################################################################
    # initStencil
    ################################################################
    def initStencil(self, backgroundColor=None):
        """
        Create and select the experiment's stencil using plane 1.

        Aperture:
            (circle union polygon) minus exclusion polygon

        If neither inclusion shape is enabled, an enabled exclusion polygon
        is removed from the full screen. With all shapes disabled, no mask
        is created.

        Coordinates are visual-angle units. Symbolic x/y values resolve to
        half the screen width/height in degrees.

        Call after visual-angle setup and after command recording starts.
        """
        self._experimentStencilNumber = None

        # Preserve the caller's clearing behavior for experiment cleanup.
        self._stencilPreviousDrawQuad = self.pgl.clearScreenDrawQuad
        self._stencilClearModeChanged = False

        # No selection, or the explicit no-stencil option.
        presets = getattr(self.settings, "stencil", [])
        if not presets: return True

        preset = presets[0]
        if preset.uuid == "pgl-stencil-none": return True

        # Invalid trait values warn and reset to their defaults.
        preset.validateSettings()

        useCircle = preset.useCircle
        usePolygon = preset.usePolygon
        useExcludePolygon = preset.useExcludePolygon
        hasInclusion = useCircle or usePolygon

        if not hasInclusion and not useExcludePolygon: return True

        if backgroundColor is None:
            backgroundColor = self.settings.backgroundColor

        # Resolve screen-edge symbols in visual-angle coordinates.
        halfWidth = float(self.pgl.screenWidth.deg) / 2
        halfHeight = float(self.pgl.screenHeight.deg) / 2

        if not all(np.isfinite(value) and value > 0 for value in (halfWidth, halfHeight)):
            pglMessages.warning("Cannot create stencil without valid visual-angle screen dimensions.")
            return False

        edgeValues = {
            "x": halfWidth,
            "-x": -halfWidth,
            "y": halfHeight,
            "-y": -halfHeight,
        }

        # Resolve and triangulate before starting any stencil exchange.
        def preparePolygon(text, label):
            coordinates = preset.parsePolygonCoordinates(text)
            if coordinates is None:
                pglMessages.warning(f"Could not resolve {label} coordinates.")
                return None

            vertices = np.array([
                [
                    edgeValues[value] if isinstance(value, str) else value
                    for value in pair
                ]
                for pair in coordinates
            ], dtype=np.float64)

            triangles = self.pgl.triangulatePolygon(vertices)
            if triangles is None:
                pglMessages.warning(f"Could not prepare {label} for stencil '{preset.name}'.")

            return triangles

        # Prepare the circle.
        radius = None
        if useCircle:
            radius = preset.parseRadius(preset.radius)
            if radius is None:
                pglMessages.warning("Could not resolve the stencil circle radius.")
                return False

            if isinstance(radius, str): radius = edgeValues[radius]

        # Prepare the inclusion polygon.
        polygonTriangles = None
        if usePolygon:
            polygonTriangles = preparePolygon(
                preset.polygonCoordinates,
                "inclusion polygon",
            )
            if polygonTriangles is None: return False

        # Prepare the exclusion polygon.
        excludeTriangles = None
        if useExcludePolygon:
            excludeTriangles = preparePolygon(
                preset.excludePolygonCoordinates,
                "exclusion polygon",
            )
            if excludeTriangles is None: return False

        # Establish a render-pass boundary before creating the stencil.
        if self.pgl.stencilSelect(0) is None: return False
        self.pgl.flush()

        stencilNumber = 1

        if hasInclusion:
            # Initialize to blocked everywhere, then write the included union.
            if self.pgl.stencilCreateBegin(stencilNumber) is None: return False

            if useCircle:
                self.pgl.circle(
                    radius=radius,
                    x=preset.centerX,
                    y=preset.centerY,
                    numSegments=180,
                    fill=True,
                    color=1,
                )

            if polygonTriangles is not None:
                # One real triangle and one degenerate triangle per quad.
                self.pgl.quad(
                    polygonTriangles[:, [0, 1, 2, 0], :],
                    color=1,
                )

            if self.pgl.stencilCreateEnd() is None: return False

            if useExcludePolygon:
                # Start a second pass that writes zero into the selected bit.
                # Do NOT call stencilCreateBegin() here: its full-screen
                # initialization would overwrite the union we just created.
                if not self.pgl.s.writeCommand("mglStartStencilCreation"):
                    return False

                ackTime = self.pgl.s.readAck()
                if ackTime is None: return False

                self.pgl.s.write(np.uint32(stencilNumber))
                self.pgl.s.write(np.uint32(1))  # Inverted creation writes zero.
                self.pgl.commandResults = self.pgl.s.readCommandResults(ackTime)

                if self.pgl.commandResults is None: return False
                success = self.pgl.commandResults.get("success")
                if success is None or not np.all(np.asarray(success) != 0):
                    pglMessages.warning("Could not start stencil exclusion pass.")
                    return False

        else:
            # Exclusion-only mask: initialize to allowed everywhere and
            # leave creation configured to write zero inside the polygon.
            if self.pgl.stencilCreateBegin(stencilNumber, invert=True) is None:
                return False

        # Remove the exclusion polygon from the included region.
        if useExcludePolygon:
            self.pgl.quad(
                excludeTriangles[:, [0, 1, 2, 0], :],
                color=1,
            )

            if self.pgl.stencilCreateEnd() is None: return False

        # All creation passes are finished, with no stencil selected.
        # Future passes start black everywhere, including excluded regions.
        if not self.pgl.clearScreen(0, drawQuad=False): return False

        # Track ownership before selecting the experiment mask.
        self._experimentStencilNumber = stencilNumber
        if self.pgl.stencilSelect(stencilNumber) is None: return False

        # Ordinary task clearScreen() calls now paint only inside the aperture.
        self.pgl.clearScreenDrawQuad = True
        self._stencilClearModeChanged = True

        if not self.pgl.clearScreen(backgroundColor): return False

        self.pgl.flush()
        return True
    
    ################################################################
    # endStencil
    ################################################################
    def endStencil(self):
        """
        Disable the experiment stencil and restore its clearing default.

        Does not flush or change the last presented frame.
        Only call when renderer communication is safe.
        """
        if getattr(self, "_experimentStencilNumber", None) is not None:
            if self.pgl.isOpen():
                if self.pgl.stencilSelect(0) is None: return False

            self._experimentStencilNumber = None

        # Restore the Python-side default without presenting another frame.
        if getattr(self, "_stencilClearModeChanged", False):
            self.pgl.clearScreenDrawQuad = self._stencilPreviousDrawQuad
            self._stencilClearModeChanged = False

        return True
    
    ################################################################
    # Validate scanner-trigger availability
    ################################################################
    def _validateVolumeTriggerSource(self):
        """Allow a live digital worker to recover between hardware streams."""
        selectedNames = {name.casefold() for name in self.settings.digitalIO}
        digitalSelected = "labjack digital trigger" in selectedNames
        config = getattr(self, "_labJackTriggerConfig", None)

        if digitalSelected:
            if config is None:
                raise RuntimeError("LabJack trigger was selected but has no acquisition configuration")

            devices = [
                device for device in getattr(self, "_experimentInputDevices", [])
                if device.deviceType == "LabJack"
            ]
            if len(devices) != 1:
                raise RuntimeError("LabJack scanner input requires exactly one initialized input device")

            status = devices[0].getDigitalReadStatus()
            if status["error"] is not None:
                raise RuntimeError(f"LabJack scanner input failed: {status['error']}")
            if not status["running"]:
                raise RuntimeError("LabJack scanner input worker is not running")
            if not status["streamActive"] and not status.get("recovering", False):
                raise RuntimeError("LabJack scanner input is neither acquiring nor recovering")

        elif config is not None:
            raise RuntimeError(
                "Digital-IO selection changed after LabJack acquisition started. "
                "Create a new experiment rather than changing an active source."
            )

        if self.settings.startOnVolumeTrigger and self.state.volumeTriggerKeyCode is None and not digitalSelected:
            raise ValueError("startOnVolumeTrigger requires a volumeTriggerKey or an initialized digital trigger device.")

    ################################################################
    # Match keyboard or configured digital scanner triggers
    ################################################################
    def _isVolumeTrigger(self, event):
        """
        Match keyboard triggers or the active digital acquisition.

        Uses the immutable acquisition snapshot, not current preset contents.
        Stale digital events remain available in data.events but cannot count
        as scanner volumes for a different acquisition.

        This does not deduplicate a physical pulse delivered through both
        keyboard and digital interfaces.
        """
        if event.type == "keyboard":
            return (
                self.state.volumeTriggerKeyCode is not None
                and event.keyCode == self.state.volumeTriggerKeyCode
                and event.eventType == "keydown"
            )

        if event.type != "digitalInput": return False
        if getattr(event, "deviceType", None) != "LabJack": return False

        config = getattr(self, "_labJackTriggerConfig", None)
        acquisitionId = getattr(self, "_labJackAcquisitionId", None)
        if config is None or not acquisitionId: return False

        return (
            getattr(event, "acquisitionId", None) == acquisitionId
            and getattr(event, "channel", None) == config["channels"]
            and getattr(event, "eventType", None) == config["edge"]
        )

    ################################################################
    # Snapshot digital acquisition diagnostics
    ################################################################
    def _snapshotDigitalInputStatus(self):
        """
        Retain independent acquisition diagnostics without hardware reads.

        Preserve prior snapshots after unregistering devices.
        """
        from copy import deepcopy

        snapshots = deepcopy(getattr(self.data, "digitalInputDiagnostics", {}))
        devices = list(getattr(self, "_experimentInputDevices", []))

        for device in self.state.digitalIODevices:
            if not any(existing is device for existing in devices):
                devices.append(device)

        errors = []

        for device in devices:
            getStatus = getattr(device, "getDigitalReadStatus", None)
            if not callable(getStatus): continue

            deviceKey = str(device.uuid)

            try:
                status = getStatus()
                if status.get("scanRate") is None and status.get("error") is None:
                    continue

                config = getattr(self, "_labJackTriggerConfig", None) or {}

                snapshots[deviceKey] = {
                    "deviceType": device.deviceType,
                    "deviceDescription": device.deviceDescription,
                    "serialNumber": getattr(device, "serialNumber", None),
                    "snapshotTimestamp": pglTimestamp.getSecs(),
                    "configuredChannel": config.get("channels"),
                    "configuredEdge": config.get("edge"),
                    "requestedConfiguration": deepcopy(config),
                    "acquisition": deepcopy(getattr(self.data, "labJackAcquisition", {})),
                    "status": status,
                }

            except Exception as error:
                record = dict(snapshots.get(deviceKey, {}))
                record["snapshotError"] = f"{type(error).__name__}: {error}"
                snapshots[deviceKey] = record
                errors.append(error)

        self.data.digitalInputDiagnostics = snapshots
        if errors: raise errors[0]
        return True

    ################################################################
    # Record an accepted scanner trigger without changing its source event
    ################################################################
    def _recordVolumeTrigger(self, event, *, reconstruction=None):
        """
        Record an accepted trigger in chronological volume-trigger order.

        Original source events retain their existing order and contents.
        This method does not decide pre-start/ignored-volume eligibility;
        callers must apply those gates before recording.

        reconstruction=None:
            Record an existing measured source event.

        reconstruction=dict:
            Record an estimated volume using explicitly supplied evidence.
            event remains the source diagnostic, not a fabricated digital edge.

        Returns:
            The newly recorded pglEventVolumeTrigger.

        Updates the current volume count only. Does not replay task updates
        or cause a stimulus transition at the estimated timestamp.
        """
        from copy import deepcopy
        from numbers import Real, Integral

        def validTimestamp(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Real)
                and np.isfinite(value)
            )

        if reconstruction is not None:
            if not isinstance(reconstruction, dict):
                raise TypeError("reconstruction must be a dictionary or None.")

            required = {
                "timestamp", "mismatchId", "missingEdgeOrdinal",
                "medianInterval", "supportingObservations",
            }
            missing = required.difference(reconstruction)
            if missing:
                raise ValueError(f"Missing reconstruction evidence: {sorted(missing)}")

            mismatchId = reconstruction["mismatchId"]
            ordinal = reconstruction["missingEdgeOrdinal"]
            interval = reconstruction["medianInterval"]

            if not isinstance(mismatchId, str) or not mismatchId:
                raise ValueError("mismatchId must be a nonempty string.")
            if (
                isinstance(ordinal, (bool, np.bool_))
                or not isinstance(ordinal, Integral)
                or ordinal < 0
            ):
                raise ValueError("missingEdgeOrdinal must be a nonnegative integer.")
            if not validTimestamp(interval) or interval <= 0:
                raise ValueError("medianInterval must be finite and positive.")
            if not isinstance(reconstruction["supportingObservations"], dict):
                raise TypeError("supportingObservations must be a dictionary.")

            timestamp = reconstruction["timestamp"]
        else:
            timestamp = getattr(event, "timestamp", None)

        if not validTimestamp(timestamp):
            raise ValueError("Volume-trigger timestamp must be finite.")

        timestamp = float(timestamp)
        volumeEntries = [
            (index, entry)
            for index, entry in enumerate(self.data.events)
            if entry.type == "volumeTrigger"
        ]

        # Never sort away corrupt existing data.
        previousTimestamp = None
        for _, entry in volumeEntries:
            existingTimestamp = getattr(entry, "timestamp", None)
            if not validTimestamp(existingTimestamp):
                raise ValueError("Existing volume-trigger timestamp is invalid.")
            if previousTimestamp is not None and existingTimestamp <= previousTimestamp:
                raise ValueError("Existing volume-trigger timestamps are not strictly increasing.")
            if existingTimestamp == timestamp:
                raise ValueError("A volume trigger already exists at this timestamp.")
            previousTimestamp = existingTimestamp

            if reconstruction is not None and (
                getattr(entry, "mismatchId", None) == mismatchId
                and getattr(entry, "missingEdgeOrdinal", None) == ordinal
            ):
                raise ValueError("This missing edge has already been reconstructed.")

        from uuid import uuid4
        volumeEvent = pglEventVolumeTrigger(timestamp=timestamp)
        volumeEvent.eventId = str(uuid4())
        volumeEvent.sourceType = event.type

        # Preserve source identity for measured and reconstructed events.
        for fieldName in ("deviceType", "deviceUUID", "serialNumber", "acquisitionId", "channel"):
            if hasattr(event, fieldName):
                setattr(volumeEvent, fieldName, deepcopy(getattr(event, fieldName)))

        if reconstruction is not None:
            volumeEvent.edge = getattr(event, "edge", None)
            volumeEvent.isEstimated = True
            volumeEvent.timestampMethod = "counterConfirmedMedianInterval"
            volumeEvent.reconstructionReason = "counterEdgeMismatch"
            volumeEvent.mismatchId = mismatchId
            volumeEvent.missingEdgeOrdinal = int(ordinal)
            volumeEvent.medianInterval = float(interval)
            volumeEvent.supportingObservations = deepcopy(
                reconstruction["supportingObservations"]
            )
            volumeEvent.reconstructionRecordedTimestamp = pglTimestamp.getSecs()
            volumeEvent.diagnosticTimestamp = getattr(event, "timestamp", None)

            # An estimated trigger has no measured sample location.
            volumeEvent.sampleIndex = None
            volumeEvent.sampleTime = None
            volumeEvent.scanRate = None
            volumeEvent.segmentId = None
            volumeEvent.sampleIndexScope = None

        elif event.type == "digitalInput":
            if getattr(event, "deviceType", None) == "LabJack":
                sourceId = getattr(event, "eventId", None)
                if not isinstance(sourceId, str) or not sourceId:
                    raise ValueError("Measured LabJack source has no stable event ID.")
                volumeEvent.sourceEventId = sourceId
                for name in ("labJackTimestamp", "clockMappingId", "deviceTimestampMethod"):
                    setattr(volumeEvent, name, deepcopy(getattr(event, name, None)))
            volumeEvent.edge = event.eventType
            for fieldName in (
                "sampleIndex", "scanRate", "sampleTime", "timestampMethod",
                "readCompletedTimestamp", "segmentId", "sampleIndexScope",
            ):
                setattr(volumeEvent, fieldName, deepcopy(getattr(event, fieldName, None)))
            volumeEvent.isEstimated = bool(getattr(event, "isEstimated", False))

        else:
            volumeEvent.isEstimated = False

        # Insert before the next volume event, not by globally sorting events.
        # The relative order of all existing entries is preserved.
        insertIndex = next(
            (index for index, entry in volumeEntries if entry.timestamp > timestamp),
            len(self.data.events),
        )

        self.data.events.insert(insertIndex, volumeEvent)
        self.state.volumeNumber += 1
        return volumeEvent
        
    def _getKeyboardSuppressionConfig(self):
        """
        Resolve selective suppression and the independent all-key flag.

        Response keys and special keys share parseKeyNames() syntax.
        Disabled special-key text is retained but not interpreted.

        Validate the complete request before changing listener configuration.
        """
        keyboard = self.keyboardMouse
        keyCodes = []

        def addKeyCode(value, label):
            if value is None: return
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Invalid key code for {label}: {value!r}")
            if not 0 <= value <= 65535:
                raise ValueError(f"Key code outside the supported range for {label}: {value!r}")
            keyCodes.append(int(value))

        if self.settings.eatKeys:
            for index, keyCode in enumerate(self.state.responseKeyCodesList):
                if keyCode is None:
                    raise ValueError(f"Response key {index + 1} has no valid key code")
                addKeyCode(keyCode, f"response key {index + 1}")

            for fieldName in ("startKeyCode", "endKeyCode", "volumeTriggerKeyCode"):
                addKeyCode(getattr(self.state, fieldName), fieldName)

        if self.settings.eatSpecialKeys:
            keyNames = keyboard.parseKeyNames(self.settings.specialKeysToEat)

            for keyName in keyNames:
                keyCode = keyboard.charToKeyCode(keyName)
                if keyCode is None:
                    raise ValueError(f"Unknown special key to eat: {keyName!r}")
                addKeyCode(keyCode, keyName)

        return {
            "keyCodes": sorted(set(keyCodes)),
            "eatAllKeys": bool(self.settings.eatAllKeys),
        }
        
    ################################################################
    # Apply keyboard suppression settings
    ################################################################
    def _configureKeyboardSuppression(self):
        """Validate and explicitly replace selective and all-key suppression."""
        config = self._getKeyboardSuppressionConfig()
        keyboard = self.keyboardMouse

        if not keyboard.isRunning():
            raise RuntimeError("Cannot configure suppression: keyboard listener is not running")

        try:
            # Clear the previous all-key mode before replacing selective keys.
            keyboard.setEatAllKeys(False)
            keyboard.setEatKeys(keyCodes=config["keyCodes"])
            keyboard.setEatAllKeys(config["eatAllKeys"])

        except Exception:
            # Attempt both releases without replacing the original error.
            for release in (
                lambda: keyboard.setEatAllKeys(False),
                lambda: keyboard.setEatKeys(keyCodes=[]),
            ):
                try:
                    release()
                except Exception as cleanupError:
                    pglMessages.warning(f"Could not release keyboard suppression: {cleanupError}")
            raise

        return config
    
    @staticmethod
    def _estimateMissingVolumeTimes(
        missingCount, measuredIntervals, anchorTimestamp,
        lowerTimestamp, upperTimestamp, measuredTimestamps,
        medianWindow=10, minIntervals=3,
    ):
        """
        Estimate counter-confirmed missing volume timestamps.

        measuredIntervals:
            Chronologically ordered eligible measured intervals only.
            The caller must exclude intervals spanning scan gaps, missing
            counter increments, estimated events, or different sources.

        anchorTimestamp:
            Trusted measured trigger immediately preceding the missing
            sequence. This helper handles a consecutive missing sequence;
            it does not assign missing edges among interspersed observations.

        lowerTimestamp, upperTimestamp:
            Exclusive supported bounds for the missing sequence.

        measuredTimestamps:
            All measured trigger timestamps in the relevant placement region,
            including later-delivered observations.

        Returns:
            Dictionary containing timestamps or an explicit unresolved reason.

        This helper neither mutates events nor updates the volume count.
        Sample indices and sample times are never manufactured.
        """
        import math
        from numbers import Integral, Real
        from statistics import median

        def validTimestamp(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Real)
                and math.isfinite(float(value))
            )

        def validInteger(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Integral)
            )

        if not validInteger(missingCount) or missingCount < 1:
            raise ValueError("missingCount must be a positive counter-confirmed integer.")
        if not validInteger(medianWindow) or medianWindow < 1:
            raise ValueError("medianWindow must be a positive integer.")
        if not validInteger(minIntervals) or not 1 <= minIntervals <= medianWindow:
            raise ValueError("minIntervals must be between 1 and medianWindow.")

        result = {
            "resolved": False,
            "reason": None,
            "missingCount": int(missingCount),
            "timestamps": [],
            "medianInterval": None,
            "intervalsUsed": [],
            "timestampMethod": "counterConfirmedMedianInterval",
        }

        def unresolved(reason):
            result["reason"] = reason
            return result

        if not all(validTimestamp(value) for value in (
            anchorTimestamp, lowerTimestamp, upperTimestamp,
        )):
            return unresolved("invalidTimestampBounds")

        anchorTimestamp = float(anchorTimestamp)
        lowerTimestamp = float(lowerTimestamp)
        upperTimestamp = float(upperTimestamp)

        if not anchorTimestamp <= lowerTimestamp < upperTimestamp:
            return unresolved("inconsistentTimestampBounds")

        intervals = list(measuredIntervals)[-int(medianWindow):]
        if any(not validTimestamp(value) or value <= 0 for value in intervals):
            return unresolved("invalidMeasuredIntervalHistory")

        result["intervalsUsed"] = [float(value) for value in intervals]
        if len(intervals) < minIntervals:
            return unresolved("insufficientMeasuredIntervalHistory")

        intervalEstimate = float(median(intervals))
        result["medianInterval"] = intervalEstimate

        observedTimes = list(measuredTimestamps)
        if any(not validTimestamp(value) for value in observedTimes):
            return unresolved("invalidMeasuredTriggerTimestamp")
        observedTimes = sorted(float(value) for value in observedTimes)

        # This placement rule is valid only for a consecutive missing sequence.
        # A measured trigger between the anchor and the final estimate would
        # require a different, explicitly supported assignment of missing edges.
        finalEstimate = anchorTimestamp + int(missingCount) * intervalEstimate
        firstEstimate = anchorTimestamp + intervalEstimate

        if not math.isfinite(finalEstimate):
            return unresolved("nonfiniteEstimatedTimestamp")
        if firstEstimate <= lowerTimestamp or finalEstimate >= upperTimestamp:
            return unresolved("estimatesOutsideSupportedWindow")

        if any(anchorTimestamp < timestamp <= finalEstimate for timestamp in observedTimes):
            return unresolved("measuredTriggerIntersectsMissingSequence")

        # Bound the count by the supported window before allocating a list.
        timestamps = [
            anchorTimestamp + index * intervalEstimate
            for index in range(1, int(missingCount) + 1)
        ]

        if any(
            not math.isfinite(timestamp)
            or not lowerTimestamp < timestamp < upperTimestamp
            for timestamp in timestamps
        ):
            return unresolved("estimatesOutsideSupportedWindow")

        if any(right <= left for left, right in zip(
            [anchorTimestamp] + timestamps, timestamps,
        )):
            return unresolved("estimatedTimestampsNotStrictlyIncreasing")

        result.update({
            "resolved": True,
            "reason": "counterEdgeMismatch",
            "timestamps": timestamps,
        })
        return result
    
    def _updateVerifiedVolumeTriggerHistory(
        self, event, *, acquisitionId, medianWindow=10,
    ):
        """
        Maintain experiment-side history from settled counter diagnostics.

        Call for diagnostic events regardless of whether scanner volumes are
        currently accepted by the experiment's start/ignore gates.

        Eligible intervals require:
            - Two individually verified measured rising edges.
            - Same acquisition, channel, edge, and timing segment.
            - Exactly one counter increment between the verified edges.
            - No intervening reported mismatch or scan discontinuity.

        Multi-edge matched windows are conservatively excluded because their
        individual counter-to-edge assignments have not been established.

        Original events are not modified. No volume events are inserted.
        """
        from copy import deepcopy
        from numbers import Integral, Real

        if not isinstance(acquisitionId, str) or not acquisitionId:
            raise ValueError("acquisitionId must be a nonempty string.")
        if (
            isinstance(medianWindow, (bool, np.bool_))
            or not isinstance(medianWindow, Integral)
            or medianWindow < 1
        ):
            raise ValueError("medianWindow must be a positive integer.")

        config = getattr(self, "_labJackTriggerConfig", None)
        if config is None:
            raise RuntimeError("No LabJack acquisition configuration is available.")

        source = (acquisitionId, config["channels"], config["edge"])
        history = getattr(self, "_verifiedVolumeTriggerHistory", None)

        if history is None:
            history = {
                "source": source,
                "lastVerifiedTrigger": None,
                "intervals": [],
                "handledWindowIds": set(),
            }
            self._verifiedVolumeTriggerHistory = history
        elif history["source"] != source:
            raise RuntimeError(
                "Verified trigger history belongs to another acquisition. "
                "Initialize fresh history when starting a new acquisition."
            )

        def result(outcome, reason):
            return {
                "outcome": outcome,
                "reason": reason,
                "intervals": deepcopy(history["intervals"]),
                "lastVerifiedTrigger": deepcopy(history["lastVerifiedTrigger"]),
            }

        def breakContinuity(reason):
            # Retain earlier trusted intervals; do not bridge this boundary.
            history["lastVerifiedTrigger"] = None
            return result("continuityBroken", reason)

        if getattr(event, "acquisitionId", None) != acquisitionId:
            return result("ignored", "foreignAcquisition")
        if getattr(event, "deviceType", None) != "LabJack":
            return result("ignored", "foreignDevice")

        eventType = getattr(event, "type", None)
        if eventType == "digitalInputRecovery":
            if config["channels"] not in getattr(event, "channels", []):
                return result("ignored", "foreignChannel")
            return breakContinuity("streamRecovery")

        if eventType not in ("digitalInputCounterWindow", "digitalInputMismatch"):
            return result("ignored", "notCounterEvidence")

        if (
            getattr(event, "channel", None) != config["channels"]
            or getattr(event, "edge", None) != config["edge"]
        ):
            return result("ignored", "foreignSource")

        windowId = getattr(event, "windowId", None)
        if not isinstance(windowId, str) or not windowId:
            return breakContinuity("missingWindowIdentity")
        if windowId in history["handledWindowIds"]:
            return result("ignored", "duplicateWindow")
        history["handledWindowIds"].add(windowId)

        comparison = getattr(event, "comparison", None)
        if not isinstance(comparison, dict):
            return breakContinuity("invalidComparison")

        if eventType == "digitalInputMismatch":
            return breakContinuity("counterMismatch")

        if (
            comparison.get("outcome") != "matched"
            or comparison.get("windowSettled") is not True
            or comparison.get("counterContinuityVerified") is not True
            or comparison.get("missingEdgeCount") != 0
        ):
            return breakContinuity("unverifiedWindow")

        if (
            getattr(event, "continuousScanTimeline", None) is not True
            or getattr(event, "gapRecords", None) != []
        ):
            return breakContinuity("scanDiscontinuity")

        observations = getattr(event, "measuredObservations", None)
        if (
            not isinstance(observations, list)
            or len(observations) != 1
            or comparison.get("counterIncrement") != 1
            or comparison.get("measuredEdgeCount") != 1
        ):
            return breakContinuity("individualEdgeAssignmentUnavailable")

        before = getattr(event, "beforeObservation", None)
        after = getattr(event, "afterObservation", None)
        measured = observations[0]

        def integer(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Integral)
            )

        def finiteTime(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Real)
                and np.isfinite(value)
            )

        for observation in (before, measured, after):
            if not isinstance(observation, dict):
                return breakContinuity("invalidObservation")
            if (
                not integer(observation.get("segmentId"))
                or observation["segmentId"] < 0
                or not integer(observation.get("sampleIndex"))
                or observation["sampleIndex"] < 0
                or not finiteTime(observation.get("timestamp"))
                or not integer(observation.get("counter"))
                or not 0 <= observation["counter"] <= 0xFFFFFFFF
            ):
                return breakContinuity("invalidObservation")

        if not (
            before["segmentId"] == measured["segmentId"] == after["segmentId"]
            and before["sampleIndex"] < measured["sampleIndex"] < after["sampleIndex"]
            and before["timestamp"] < measured["timestamp"] < after["timestamp"]
            and before.get("inputHigh") is False
            and measured.get("inputHigh") is True
            and after.get("inputHigh") is False
            and after["counter"] - before["counter"] == 1
            and comparison.get("counterBefore") == before["counter"]
            and comparison.get("counterAfter") == after["counter"]
        ):
            return breakContinuity("inconsistentObservationWindow")

        # Use the settled endpoint's count to identify the verified edge.
        # The counter read at the HIGH sample can be offset by scan ordering.
        current = {
            "timestamp": float(measured["timestamp"]),
            "sampleIndex": int(measured["sampleIndex"]),
            "segmentId": int(measured["segmentId"]),
            "settledCounter": int(after["counter"]),
            "windowId": windowId,
        }
        previous = history["lastVerifiedTrigger"]

        if previous is not None and current["timestamp"] <= previous["timestamp"]:
            return breakContinuity("outOfOrderVerification")

        eligible = (
            previous is not None
            and current["segmentId"] == previous["segmentId"]
            and current["sampleIndex"] > previous["sampleIndex"]
            and current["settledCounter"] == previous["settledCounter"] + 1
        )

        history["lastVerifiedTrigger"] = current
        if not eligible:
            return result("baseline", "verifiedMeasuredEdge")

        history["intervals"].append({
            "interval": current["timestamp"] - previous["timestamp"],
            "firstTimestamp": previous["timestamp"],
            "secondTimestamp": current["timestamp"],
            "firstWindowId": previous["windowId"],
            "secondWindowId": current["windowId"],
            "segmentId": current["segmentId"],
        })

        # Only the working median history is bounded. Original diagnostic
        # events remain in data.events and are not removed by this method.
        history["intervals"] = history["intervals"][-int(medianWindow):]
        return result("intervalAdded", "consecutiveCounterVerifiedMeasuredEdges")
    
    def _consumeVolumeCounterDiagnostic(self, event):
        """
        Consume counter evidence during pre-start or active acquisition.

        Original diagnostic events must already be retained in data.events.

        Conservative reconstruction:
            - Same active acquisition/channel/edge.
            - Settled, counter-confirmed positive deficit.
            - No measured edges interspersed inside the mismatch window.
            - Trusted measured anchor and sufficient earlier interval history.
            - Estimates fit the observation bounds.
            - Estimates occur after the established acceptance boundary.
            - No duplicate volume timestamps or repeated mismatch handling.

        Never retroactively opens a start gate or changes task timestamps.
        """
        from copy import deepcopy
        from numbers import Integral, Real

        eventType = getattr(event, "type", None)
        if eventType not in (
            "digitalInputCounterWindow",
            "digitalInputMismatch",
            "digitalInputRecovery",
        ):
            return False

        acquisitionId = getattr(self, "_labJackAcquisitionId", None)
        config = getattr(self, "_labJackTriggerConfig", None)
        runtime = getattr(self, "_volumeRepairRuntime", None)
        if runtime is None:
            raise RuntimeError("Volume-repair runtime has not been initialized.")

        if (
            not acquisitionId
            or config is None
            or getattr(event, "acquisitionId", None) != acquisitionId
            or getattr(event, "deviceType", None) != "LabJack"
        ):
            return True  # Retained source event, but not this acquisition.

        if eventType == "digitalInputRecovery":
            if config["channels"] not in getattr(event, "channels", []):
                return True
        elif (
            getattr(event, "channel", None) != config["channels"]
            or getattr(event, "edge", None) != config["edge"]
        ):
            return True

        # Keep trusted anchors separately: history breaks adjacency at a gap,
        # but an earlier measured anchor may still support bounded placement.
        snapshot = self._updateVerifiedVolumeTriggerHistory(
            event,
            acquisitionId=acquisitionId,
            medianWindow=runtime["medianWindow"],
        )

        if eventType == "digitalInputCounterWindow":
            if snapshot["outcome"] in ("baseline", "intervalAdded"):
                anchor = snapshot["lastVerifiedTrigger"]
                if anchor is not None:
                    runtime["lastTrustedAnchor"] = deepcopy(anchor)
            return True

        if eventType == "digitalInputRecovery":
            return True

        mismatchId = getattr(event, "mismatchId", None)
        audit = {
            "acquisitionId": acquisitionId,
            "mismatchId": mismatchId,
            "diagnosticTimestamp": getattr(event, "timestamp", None),
            "handledTimestamp": pglTimestamp.getSecs(),
            "status": "unresolved",
            "reason": None,
            "insertedTimestamps": [],
            "evidence": {
                "comparison": deepcopy(getattr(event, "comparison", None)),
                "beforeObservation": deepcopy(getattr(event, "beforeObservation", None)),
                "afterObservation": deepcopy(getattr(event, "afterObservation", None)),
                "measuredObservations": deepcopy(getattr(event, "measuredObservations", None)),
                "gapRecords": deepcopy(getattr(event, "gapRecords", None)),
            },
        }
        self.data.volumeTriggerRepairAudit.append(audit)

        def unresolved(reason):
            audit["reason"] = reason
            return True

        if not isinstance(mismatchId, str) or not mismatchId:
            return unresolved("missingMismatchIdentity")
        if mismatchId in runtime["handledMismatchIds"]:
            audit["status"] = "duplicate"
            return unresolved("mismatchAlreadyHandled")
        runtime["handledMismatchIds"].add(mismatchId)

        if not runtime["enabled"]:
            return unresolved("reconstructionDisabled")

        comparison = getattr(event, "comparison", None)
        count = getattr(event, "missingEdgeCount", None)
        if (
            not isinstance(comparison, dict)
            or comparison.get("outcome") != "missingEdges"
            or comparison.get("windowSettled") is not True
            or comparison.get("counterContinuityVerified") is not True
            or isinstance(count, (bool, np.bool_))
            or not isinstance(count, Integral)
            or count <= 0
            or comparison.get("missingEdgeCount") != count
        ):
            return unresolved("missingCountNotVerified")

        if getattr(event, "measuredObservations", None) != []:
            return unresolved("missingSequenceContainsMeasuredEdges")

        def finiteTime(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Real)
                and np.isfinite(value)
            )

        before = getattr(event, "beforeObservation", None)
        after = getattr(event, "afterObservation", None)
        anchor = runtime["lastTrustedAnchor"]

        if not isinstance(before, dict) or not isinstance(after, dict):
            return unresolved("missingObservationBounds")
        if anchor is None:
            return unresolved("noTrustedMeasuredAnchor")

        lower = before.get("timestamp")
        upper = after.get("timestamp")
        if not finiteTime(lower) or not finiteTime(upper) or lower >= upper:
            return unresolved("invalidObservationBounds")

        # The counter at the preceding boundary must still be the anchor's
        # settled count. Otherwise missing edges may precede this window.
        if (
            anchor["timestamp"] > lower
            or anchor["settledCounter"] != before.get("counter")
        ):
            return unresolved("anchorDoesNotPrecedeMissingSequence")

        # Cross-restart bounds are host-clock estimates, not a repaired scan
        # timeline. Retain that distinction in the reconstruction evidence.
        audit["boundsMethod"] = (
            "continuousSampleTimeline"
            if getattr(event, "continuousScanTimeline", False)
            else "segmentedHostClockEstimates"
        )

        # Only interval history preceding the missing window is eligible.
        intervalRecords = [
            record for record in snapshot["intervals"]
            if record["secondTimestamp"] <= lower
        ]
        measuredTimes = [
            entry.timestamp
            for entry in self.data.events
            if (
                getattr(entry, "type", None) == "digitalInput"
                and getattr(entry, "acquisitionId", None) == acquisitionId
                and getattr(entry, "channel", None) == config["channels"]
                and getattr(entry, "eventType", None) == config["edge"]
                and finiteTime(getattr(entry, "timestamp", None))
            )
        ]

        estimate = self._estimateMissingVolumeTimes(
            missingCount=int(count),
            measuredIntervals=[record["interval"] for record in intervalRecords],
            anchorTimestamp=anchor["timestamp"],
            lowerTimestamp=lower,
            upperTimestamp=upper,
            measuredTimestamps=measuredTimes,
            medianWindow=runtime["medianWindow"],
            minIntervals=runtime["minIntervals"],
        )
        audit["estimate"] = deepcopy(estimate)
        if not estimate["resolved"]:
            return unresolved(estimate["reason"])

        timestamps = estimate["timestamps"]
        boundary = runtime["acceptanceBoundary"]
        if boundary is None or any(timestamp <= boundary for timestamp in timestamps):
            return unresolved("repairWouldChangePreStartOrIgnoredVolumeBoundary")
        if not self.state.experimentStarted or self.state.experimentDone:
            return unresolved("experimentNotAcceptingVolumes")

        # Preflight the entire insertion before changing events or counts.
        existingVolumes = [
            entry for entry in self.data.events
            if entry.type == "volumeTrigger"
        ]
        existingTimes = [entry.timestamp for entry in existingVolumes]
        if (
            any(not finiteTime(value) for value in existingTimes)
            or any(right <= left for left, right in zip(existingTimes, existingTimes[1:]))
        ):
            return unresolved("existingVolumeTimelineInvalid")
        if any(timestamp in existingTimes for timestamp in timestamps):
            return unresolved("estimatedTimestampDuplicatesVolume")
        if any(getattr(entry, "mismatchId", None) == mismatchId for entry in existingVolumes):
            return unresolved("mismatchAlreadyRepresentedByVolumes")

        evidence = {
            **deepcopy(audit["evidence"]),
            "boundsMethod": audit["boundsMethod"],
            "anchor": deepcopy(anchor),
            "intervalRecords": deepcopy(intervalRecords),
        }

        try:
            for ordinal, timestamp in enumerate(timestamps):
                self._recordVolumeTrigger(
                    event,
                    reconstruction={
                        "timestamp": timestamp,
                        "mismatchId": mismatchId,
                        "missingEdgeOrdinal": ordinal,
                        "medianInterval": estimate["medianInterval"],
                        "supportingObservations": evidence,
                    },
                )
                audit["insertedTimestamps"].append(timestamp)
        except Exception as error:
            # Do not discard any successfully inserted events on failure.
            audit["status"] = "partialFailure"
            audit["reason"] = f"{type(error).__name__}: {error}"
            raise

        audit["status"] = "reconstructed"
        audit["reason"] = "counterEdgeMismatch"
        pglMessages.warning(
            f"Reconstructed {len(timestamps)} counter-confirmed missing volume "
            f"trigger(s) using a median interval of "
            f"{estimate['medianInterval']:.6g} seconds. "
            "Their timestamps are estimated; earlier stimulus transitions "
            "have not been changed.",
            level=2,
        )
        return True
    
    def _initializeTaskTiming(self):
        """Initialize lightweight frame prediction without renderer queries."""
        self._lastTaskPresentation = None
        self._taskFlushIndex = 0
        self._taskFrameTiming = None

        frameRate = float(self.pgl.frameRate) if self.flush else None
        if self.flush and (not math.isfinite(frameRate) or frameRate <= 0):
            raise ValueError("Visual task timing requires a finite positive cached frameRate.")

        self._taskFrameInterval = 1.0 / frameRate if self.flush else None

        self.data.taskTiming = {
            "mode": "nearestRefresh" if self.flush else "hostClock",
            "frameRate": frameRate,
            "prediction": "lastPresentationPlusCachedRefreshInterval",
            "clockBasis": "sameAsGetSecs",
            "flushIndexScope": "experimentManagedRunFlushes",
            "flushAttempts": 0,
            "invalidPresentationResults": [],
        }

    def _prepareTaskFrame(self, updateTime):
        """Predict a refresh from the last valid presentation when available.

        Missing presentation results do not erase a previously valid reference.
        Before the first valid reference, use host-clock scheduling without
        a half-frame allowance. No renderer query is performed here.
        """
        if not self.flush or self._lastTaskPresentation is None:
            self._taskFrameTiming = {
                "mode": "hostClock",
                "predictedPresentation": None,
                "frameInterval": self._taskFrameInterval,
            }
            return

        interval = self._taskFrameInterval
        previous = self._lastTaskPresentation
        predicted = previous + interval

        # Advance the refresh grid past the current host time.
        if predicted <= updateTime:
            missed = math.floor((updateTime - predicted) / interval) + 1
            predicted += missed * interval

        self._taskFrameTiming = {
            "mode": "nearestRefresh",
            "predictedPresentation": float(predicted),
            "frameInterval": interval,
        }
        
    def _flushTaskFrame(self):
        """Flush once, accepting optional presentation timing.

        A zero/absent presentation timestamp is not a failed flush.
        Preserve the last valid presentation for refresh prediction.

        Boundary events belong only to this flush attempt. Missing actual
        timestamps remain None and are never borrowed from a later frame.
        """
        from copy import deepcopy
        from numbers import Real

        def numericScalar(value):
            """Normalize a finite scalar or single-element NumPy array."""
            if isinstance(value, np.ndarray):
                if value.size != 1 or value.dtype.kind not in "iuf":
                    return None
                value = value.item()

            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                return None

            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                return None

            return value if math.isfinite(value) else None

        flushIndex = self._taskFlushIndex
        self._taskFlushIndex += 1
        self.data.taskTiming["flushAttempts"] = self._taskFlushIndex

        # Snapshot before sending: never associate these starts with a later flush.
        associations = []
        for task in self.tasks:
            pending = list(task._pendingPresentationEvents)
            task._pendingPresentationEvents.clear()
            if pending: associations.append((task, pending))

        returnedTime = None
        commandResults = None
        presentedTime = None
        status = "flushException"
        failureDetails = None

        try:
            returnedTime = self.pgl.flush()
            commandResults = getattr(self.pgl, "commandResults", None)

            if not isinstance(commandResults, dict):
                status = "missingCommandResults"
                raise RuntimeError(
                    f"Flush {flushIndex} returned no command-result dictionary."
                )

            success = numericScalar(commandResults.get("success"))
            if success is None or success <= 0:
                status = "unsuccessfulFlush"
                raise RuntimeError(
                    f"Flush {flushIndex} did not report success: "
                    f"{commandResults.get('success')!r}"
                )

            candidate = numericScalar(returnedTime)

            if returnedTime is None or candidate == 0:
                # Expected: successful command, presentation time unavailable.
                status = "presentationUnavailable"

            elif candidate is None or candidate < 0:
                status = "invalidPresentation"

            elif (
                self._lastTaskPresentation is not None
                and candidate <= self._lastTaskPresentation
            ):
                status = "nonIncreasingPresentation"

            else:
                presentedTime = candidate
                status = "presented"
                self._lastTaskPresentation = presentedTime

            # Missing/invalid timing leaves the previous valid reference intact.
            return presentedTime

        except BaseException as error:
            failureDetails = f"{type(error).__name__}: {error}"
            raise

        finally:
            for task, pending in associations:
                task._resolveTaskPresentation(
                    pending, presentedTime, status, flushIndex,
                )

            # Count expected missing timestamps without copying a large
            # command-result dictionary for every ordinary unavailable result.
            counts = self.data.taskTiming.setdefault("presentationStatusCounts", {})
            counts[status] = counts.get(status, 0) + 1

            if status not in ("presented", "presentationUnavailable"):
                self.data.taskTiming["invalidPresentationResults"].append({
                    "flushIndex": flushIndex,
                    "status": status,
                    "error": failureDetails,
                    "returnedPresentation": deepcopy(returnedTime),
                    "commandResults": deepcopy(commandResults),
                })

    def _finishPendingTaskTiming(self):
        """Mark boundaries left without a flush; never manufacture presentation."""
        for task in self.tasks:
            pending = list(task._pendingPresentationEvents)
            task._pendingPresentationEvents.clear()
            task._resolveTaskPresentation(
                pending, None, "runEndedBeforeFlush", None,
            )

    def _retainLabJackClockEvents(self, device):
        """Append independent clock records by stable ID, without sorting events."""
        snapshots = device.getClockEvents()
        existing = {}
        for event in self.data.events:
            if event.type not in ("labJackClockCalibration", "labJackClockMapping"):
                continue
            eventId = getattr(event, "eventId", None)
            if eventId in existing:
                raise ValueError("Duplicate retained LabJack clock-event ID.")
            existing[eventId] = event

        for event in snapshots:
            previous = existing.get(event.eventId)
            if previous is not None:
                if previous.toJSON(strict=True) != event.toJSON(strict=True):
                    raise ValueError("Retained clock-event ID has conflicting contents.")
                continue
            self.data.events.append(event)
            existing[event.eventId] = event
        return True

    def _correctLabJackEventTimes(self, device, mapping, status):
        """
        Preflight and atomically apply supported event-field corrections in memory.

        No event-list sorting, acceptance changes, or volume-count changes.
        Original reconstruction audits and driver snapshots remain raw evidence.
        Caller must archive/persist data and retain the mapping events.
        """
        from copy import deepcopy
        from numbers import Integral, Real
        from uuid import uuid4

        audit = {
            "auditId": str(uuid4()),
            "mappingId": mapping.eventId,
            "method": "deviceSampleCoordinatePostRun-v1",
            "status": "preflight",
            "failureReason": None,
            "proposals": [],
            "exclusions": [],
            "estimatedVolumePolicy": "retainOriginalEstimateAndReportExclusion",
            "diagnosticSnapshotPolicy": "retainOriginalLiveEvidence",
            "volumeOrderPolicy": "rejectBatchIfExistingVolumeOrderWouldNotIncrease",
        }
        if not hasattr(self.data, "labJackClockCorrectionAudit"):
            self.data.labJackClockCorrectionAudit = []
        self.data.labJackClockCorrectionAudit.append(audit)

        def finite(value):
            return (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, Real)
                and math.isfinite(float(value))
            )

        def getField(target, name, default=None):
            return target.get(name, default) if isinstance(target, dict) else getattr(target, name, default)

        def setField(target, name, value):
            if isinstance(target, dict): target[name] = value
            else: setattr(target, name, value)

        def delField(target, name):
            if isinstance(target, dict): target.pop(name, None)
            elif hasattr(target, name): delattr(target, name)

        proposals = []
        seenTargets = set()
        proposedTimes = {}
        sourceEvents = {}
        acquisitionId = status.get("acquisitionId")
        segments = {}

        try:
            if mapping.status != "validConditional" or mapping.purpose != "postRun":
                raise ValueError("Mapping is not an applicable post-run mapping.")
            if mapping.connectionId != device.connectionId:
                raise ValueError("Mapping belongs to a different connection.")
            if not status.get("clockValid", False):
                raise ValueError("Connection clock continuity is invalid.")
            if not acquisitionId or acquisitionId != getattr(self, "_labJackAcquisitionId", None):
                raise ValueError("Acquisition identity does not match this experiment.")

            for segment in status.get("timingSegments", []):
                segmentId = segment["segmentId"]
                if segmentId in segments:
                    raise ValueError("Duplicate timing-segment identity.")
                segments[segmentId] = segment

            # Source identity is explicit. Never associate by timestamp equality.
            for event in self.data.events:
                if (
                    event.type == "digitalInput"
                    and getattr(event, "deviceType", None) == "LabJack"
                    and getattr(event, "acquisitionId", None) == acquisitionId
                ):
                    eventId = getattr(event, "eventId", None)
                    if not isinstance(eventId, str) or not eventId:
                        raise ValueError("Measured digital input has no source-event ID.")
                    if eventId in sourceEvents:
                        raise ValueError("Duplicate source-event ID in experiment events.")
                    sourceEvents[eventId] = event

            def stageSample(target, path, isObservation=False):
                if id(target) in seenTargets:
                    return
                seenTargets.add(id(target))

                segmentId = getField(target, "segmentId")
                sampleIndex = getField(target, "sampleIndex")
                if (
                    isinstance(sampleIndex, (bool, np.bool_))
                    or not isinstance(sampleIndex, Integral)
                    or sampleIndex < 0
                    or isinstance(segmentId, (bool, np.bool_))
                    or not isinstance(segmentId, Integral)
                    or segmentId not in segments
                ):
                    raise ValueError(f"{path}: missing/invalid measured sample coordinate.")

                segment = segments[segmentId]
                clockSegment = segment["clock"]
                actualRate = segment["scanRate"]
                if not finite(actualRate) or actualRate <= 0:
                    raise ValueError(f"{path}: invalid segment scan rate.")

                if not isObservation:
                    eventRate = getField(target, "scanRate")
                    if eventRate != actualRate:
                        raise ValueError(f"{path}: event/segment scan-rate mismatch.")

                # Keep the integer origin separate until adding the sample offset.
                originTicks = clockSegment["streamStartTicks"]
                if isinstance(originTicks, (bool, np.bool_)) or not isinstance(originTicks, Integral):
                    raise ValueError(f"{path}: invalid stream-start ticks.")

                oldMappingId = clockSegment["mappingId"]
                ticks = int(originTicks) + int(sampleIndex) * 40_000_000 / actualRate
                timing = device.getClockTiming(ticks, mappingId=mapping.eventId)
                originalTiming = device.getClockTiming(ticks, mappingId=oldMappingId)

                if timing["outsideObservationSpan"]:
                    raise ValueError(f"{path}: sample is outside the fitted observation span.")
                newTime = timing["timestamp"]
                currentTime = getField(target, "timestamp")
                liveTime = getField(target, "liveTimestamp", currentTime)
                if not finite(newTime) or not finite(currentTime) or not finite(liveTime):
                    raise ValueError(f"{path}: nonfinite timestamp.")

                # Numerical consistency tolerance, not an accuracy allowance.
                tolerance = max(
                    1e-9,
                    16 * math.ulp(float(liveTime)),
                    16 * math.ulp(float(originalTiming["timestamp"])),
                )
                if abs(liveTime - originalTiming["timestamp"]) > tolerance:
                    raise ValueError(f"{path}: original timestamp conflicts with sample provenance.")

                currentMappingId = getField(target, "clockMappingId")
                if currentMappingId not in (None, oldMappingId, mapping.eventId):
                    raise ValueError(f"{path}: another correction mapping is already attached.")
                if not isObservation and currentMappingId is None:
                    raise ValueError(f"{path}: measured event has no mapping reference.")

                if currentMappingId == mapping.eventId:
                    if abs(currentTime - newTime) > tolerance:
                        raise ValueError(f"{path}: existing correction is inconsistent.")
                    audit["exclusions"].append({
                        "path": path, "reason": "alreadyCorrectedWithThisMapping",
                    })
                    proposedTimes[id(target)] = newTime
                    return

                fields = {
                    "timestamp": float(newTime),
                    "clockMappingId": mapping.eventId,
                    "timestampMethod": "postRunCalibratedDeviceSample",
                    "labJackTimestamp": ticks / 40_000_000,
                    "deviceTimestampMethod": "streamStartStampPlusSampleOffset",
                    "clockCorrectionSeconds": float(newTime - liveTime),
                }
                # Preserve original live fields on FIRST correction only.
                if getField(target, "liveTimestamp") is None:
                    fields["liveTimestamp"] = float(liveTime)
                    fields["liveClockMappingId"] = oldMappingId
                    fields["liveTimestampMethod"] = getField(target, "timestampMethod")

                proposals.append((target, fields))
                proposedTimes[id(target)] = float(newTime)
                audit["proposals"].append({
                    "path": path,
                    "originalLiveTimestamp": float(liveTime),
                    "proposedTimestamp": float(newTime),
                    "correctionSeconds": float(newTime - liveTime),
                    "clockErrorLowerSeconds": timing["clockErrorLowerSeconds"],
                    "clockErrorUpperSeconds": timing["clockErrorUpperSeconds"],
                    "boundStatus": timing["boundStatus"],
                })

            def stageObservationTree(value, path):
                # Only called on explicit counter-observation fields.
                # Diagnostic host timestamps are never passed here.
                if isinstance(value, dict):
                    if all(name in value for name in (
                        "segmentId", "sampleIndex", "timestamp", "counter", "inputHigh",
                    )):
                        stageSample(value, path, isObservation=True)
                    for name, child in list(value.items()):
                        if isinstance(child, (dict, list)):
                            stageObservationTree(child, f"{path}.{name}")
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        stageObservationTree(child, f"{path}[{index}]")

            def stageEvent(event, path):
                if getattr(event, "deviceType", None) != "LabJack":
                    return
                if getattr(event, "acquisitionId", None) != acquisitionId:
                    audit["exclusions"].append({"path": path, "reason": "foreignAcquisition"})
                    return

                if event.type == "volumeTrigger" and getattr(event, "isEstimated", False):
                    audit["exclusions"].append({
                        "path": path,
                        "reason": "estimatedVolumeRetainedWithOriginalReconstruction",
                    })
                    return

                if event.type == "volumeTrigger":
                    sourceId = getattr(event, "sourceEventId", None)
                    source = sourceEvents.get(sourceId)
                    if source is None:
                        raise ValueError(f"{path}: measured volume has no unique raw source.")
                    for name in ("segmentId", "sampleIndex", "scanRate", "channel"):
                        if getattr(event, name, None) != getattr(source, name, None):
                            raise ValueError(f"{path}: source/volume provenance mismatch.")
                    stageSample(event, path)

                elif event.type == "digitalInput":
                    stageSample(event, path)

                elif event.type in ("digitalInputCounterWindow", "digitalInputMismatch"):
                    for name in (
                        "beforeObservation", "afterObservation",
                        "measuredObservations", "gapRecords",
                    ):
                        stageObservationTree(getattr(event, name, None), f"{path}.{name}")

            owners = [("experiment", self.data)] + [
                (f"task[{index}]", task.data)
                for index, task in enumerate(self.tasks)
            ]
            for ownerName, data in owners:
                for index, event in enumerate(data.events):
                    stageEvent(event, f"{ownerName}.events[{index}]")

            # Shutdown records can contain the same objects or independent copies.
            for drainIndex, drain in enumerate(
                getattr(self.data, "digitalInputShutdownDrains", [])
            ):
                for index, event in enumerate(drain.get("events", [])):
                    stageEvent(event, f"shutdownDrains[{drainIndex}].events[{index}]")

            # Check each owner's existing volume order. No global sorting.
            for ownerName, data in owners:
                previous = None
                for event in data.events:
                    if event.type != "volumeTrigger":
                        continue
                    timestamp = proposedTimes.get(id(event), event.timestamp)
                    if not finite(timestamp) or (previous is not None and timestamp <= previous):
                        raise ValueError(f"{ownerName}: corrected volume order is invalid or ambiguous.")
                    previous = timestamp

            # Source/volume copies must receive identical corrected times.
            for event in self.data.events:
                if (
                    event.type == "volumeTrigger"
                    and getattr(event, "deviceType", None) == "LabJack"
                    and getattr(event, "acquisitionId", None) == acquisitionId
                    and not getattr(event, "isEstimated", False)
                ):
                    source = sourceEvents[event.sourceEventId]
                    if proposedTimes[id(event)] != proposedTimes[id(source)]:
                        raise ValueError("Source and measured volume corrections differ.")

        except Exception as error:
            audit["status"] = "rejected"
            audit["failureReason"] = f"{type(error).__name__}: {error}"
            return audit

        # Commit only after the complete preflight. Roll back fields on failure.
        backups = []
        try:
            for target, fields in proposals:
                contents = target if isinstance(target, dict) else vars(target)
                backup = {
                    name: (name in contents, deepcopy(contents.get(name)))
                    for name in fields
                }
                backups.append((target, backup))
                for name, value in fields.items():
                    setField(target, name, value)
        except BaseException:
            for target, backup in reversed(backups):
                for name, (existed, value) in backup.items():
                    if existed: setField(target, name, value)
                    else: delField(target, name)
            audit["status"] = "commitFailedRolledBack"
            raise

        corrections = [
            record["correctionSeconds"] for record in audit["proposals"]
        ]
        audit.update({
            "status": "applied" if proposals else "noChanges",
            "affectedTargetCount": len(proposals),
            "minimumCorrectionSeconds": min(corrections) if corrections else None,
            "maximumCorrectionSeconds": max(corrections) if corrections else None,
            "maximumAbsoluteCorrectionSeconds": (
                max(abs(value) for value in corrections) if corrections else None
            ),
            "historicalAcceptanceDecisionsChanged": False,
        })
        return audit

    def _finalizeLabJackClockTiming(self):
        """
        Run once after the initial raw-save attempt and before device closure.

        Ordinary calibration/correction failure is retained and warned, not made
        a prerequisite for saving the rest of the run.
        """
        from uuid import uuid4

        previous = getattr(self.data, "labJackClockFinalization", None)
        if previous is not None:
            # Save retries do not acquire new observations or repeat correction.
            return True

        record = {
            "status": "starting",
            "failureReason": None,
            "mappingId": None,
            "rawArchiveDirectory": None,
            "correctedSaveStatus": "notAttempted",
        }
        self.data.labJackClockFinalization = record

        acquisitionId = getattr(self, "_labJackAcquisitionId", None)
        if not acquisitionId:
            record["status"] = "notApplicable"
            return True

        devices = [
            device for device in self.state.digitalIODevices
            if getattr(device, "deviceType", None) == "LabJack"
            and callable(getattr(device, "calibrateClockEnding", None))
        ]
        if len(devices) != 1:
            record["status"] = "failed"
            record["failureReason"] = "Expected exactly one owned calibrated LabJack."
            pglMessages.warning(record["failureReason"], level=2)
            return True

        device = devices[0]
        try:
            if not np.isfinite(self.data.endTime) or self.data.endTime <= 0:
                raise ValueError("Experiment end time has not been established.")

            # endDevices retains queued source events without accepting volumes.
            shutdownErrors = []
            self._attemptRunAction(
                "stop and drain acquisition before clock calibration",
                self.endDevices, shutdownErrors, reportProgress=False,
            )
            record["shutdownErrors"] = [
                f"{type(error).__name__}: {error}" for error in shutdownErrors
            ]
            status = device.getDigitalReadStatus()
            if status["running"] or status["streamActive"] or status.get("pendingEvents", 0):
                raise RuntimeError("Acquisition did not stop and drain completely.")

            # This collection occurs while the SAME owned handle remains open.
            mapping = device.calibrateClockEnding()
            record["mappingId"] = mapping.eventId
            self._retainLabJackClockEvents(device)

            # Retain raw diagnostics explicitly as historical live evidence.
            self._snapshotDigitalInputStatus()
            self.data.digitalInputDiagnosticsTiming = "originalLiveClockEvidence"
            self.data.volumeTriggerRepairAuditTiming = "originalOnlineReconstructionEvidence"

            # Preserve independently readable raw snapshots before corrections.
            dataPath = getattr(self, "_runSavePath", None)
            if dataPath is None:
                raise RuntimeError("No run destination is available for raw timing archive.")

            archive = Path(dataPath) / ("labJack-live-timing-" + str(uuid4()))
            archive.mkdir(parents=False, exist_ok=False)
            record["rawArchiveDirectory"] = str(archive)

            owners = [("experiment-data.json", self.data)] + [
                (f"task-{index}-data.json", task.data)
                for index, task in enumerate(self.tasks)
            ]
            for fileName, owner in owners:
                encoded = owner.toJSON(strict=True)
                with (archive / fileName).open("x", encoding="utf-8") as file:
                    written = file.write(encoded)
                    if written != len(encoded):
                        raise OSError("Incomplete raw timing archive write.")

            status = device.getDigitalReadStatus()
            audit = self._correctLabJackEventTimes(device, mapping, status)
            record["correctionAuditId"] = audit["auditId"]
            record["status"] = audit["status"]

            if audit["status"] == "rejected":
                record["failureReason"] = audit["failureReason"]
                pglMessages.warning(
                    "LabJack post-run clock correction was rejected. Original "
                    f"timestamps were retained. {audit['failureReason']} "
                    "Calibration evidence and proposed corrections are retained.",
                    level=2,
                )

        except (Exception, KeyboardInterrupt, SystemExit) as error:
            record["status"] = "failed"
            record["failureReason"] = f"{type(error).__name__}: {error}"
            self._recordRunError("LabJack clock finalization", error)
            pglMessages.warning(
                "LabJack clock finalization did not complete. Review the retained "
                "clock finalization/correction audit; saving and cleanup will "
                f"continue. {type(error).__name__}: {error}",
                level=2,
            )

        finally:
            # Preserve ending/rejected records even when fitting or correction fails.
            try:
                self._retainLabJackClockEvents(device)
            except Exception as error:
                record["clockEventRetentionFailure"] = f"{type(error).__name__}: {error}"
                self._recordRunError("retain LabJack clock events", error)

            # Save all event owners, not only experiment data.json.
            # Failed saves remain retryable through existing savePendingData().
            dataPath = getattr(self, "_runSavePath", None)
            if dataPath is not None:
                saveErrors = []
                for index, task in enumerate(self.tasks):
                    self._attemptRunAction(
                        f"save final timing for task {index + 1}",
                        lambda task=task: task.data.saveStrict(
                            Path(dataPath) / task._getTaskDirectoryName() / "data.json"
                        ),
                        saveErrors, reportProgress=False,
                    )

                record["correctedSaveStatus"] = "failed" if saveErrors else "complete"
                self._attemptRunAction(
                    "save final LabJack timing evidence",
                    lambda: self.data.saveStrict(Path(dataPath) / "data.json"),
                    saveErrors, reportProgress=False,
                )
                if saveErrors:
                    record["correctedSaveStatus"] = "failed"
                    self.data.saveStatus = "failed"

        # Report persistence failure only after all timing saves were attempted.
        # The outer _attemptRunAction retains this error and continues cleanup.
        if record["correctedSaveStatus"] == "failed":
            raise RuntimeError(
                "Final LabJack timing data were not completely saved. "
                "Pending evidence remains in memory; use savePendingData() "
                "after shutdown returns. See retained run errors."
            )

        return True
    
##############################################
# Settings for pglTask
##############################################
class pglTaskSettings(pglTraitSettings):
    taskName = Unicode("Default task", help="Name of the task")
    taskSaveName = Unicode("defaultTask", help="Name to use when saving task data (defaults to camelCase version of taskName)")    
    phaseNum = Int(default_value=None, allow_none=True, help="Phase number for the task. Set to None if this should run in all phases")
    seglen = List(Float(), help="List of segment lengths in seconds.")
    segmin = List(Float(), help="Minimum length of a segment.")
    segmax = List(Float(), help="Maximum length of a segment.")
    waitUntilVolumeTrigger = List(Bool(), help="List of nSegments where if set to true will run through the segment length and then wait for a volume trigger to continue.")
    nSegments = Int(help="Number of segments in the task.")
    nTrials = Float(np.inf, help="Number of trials to run for.")
    # old way of doing this - config replaces
    fixedParameters = Dict(default_value={}, help="Dictionary of fixed parameters for the task.")
    config = Instance(SimpleNamespace,args=(),kw={},help="Place for configuration variables")
    saveEyeTracker = Bool(False, help="Whether to save eye tracker events this task (if we have an eye tracker).")    
    taskID = Int(0, help="Numeric identifier for the task, used for pglExperiment to keep track of tasks.")
    verbose = Bool(True, help="Print routine task messages and trial parameters. Does not suppress warnings or errors.")
    trialPrintNewline = Bool(True, help="Print a newline after each trial's parameters.")
    isMainTask = Bool(False, help="Mark a main task for analysis selection and aborted-run classification. Analysis requires one marked task; with none marked, it may infer the main task from recorded trial counts.")
    
    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    # observe changes to taskName and if taskSaveName is not set
    # set taskSaveName to a camelCase version of taskName
    @observe("taskName")
    def toCamelCase(self, change) -> None:
        if self.taskSaveName == "" or self.taskSaveName == "defaultTask":
            # split taskName into words
            words = change['new'].strip().split()
            if not words:
                return
            # convert to camelCase and save as taskSaveName
            firstWord = words[0][0].lower() + words[0][1:] if words[0] else ""
            restWords = "".join(word[0].upper() + word[1:] if word else "" for word in words[1:])
            self.taskSaveName = firstWord + restWords
        
    # observe changes in seglen, segmin, segmax to keep them in sync
    @observe("seglen", "segmin", "segmax")
    def _updateSegments(self, change):

        # hold off on trait notifications while we update
        with self.hold_trait_notifications():

            # if seglen change, then just make seming and segmax the same as seglen
            if change["name"] == "seglen":
                self.segmin = list(self.seglen)
                self.segmax = list(self.seglen)

            elif change["name"] == "segmin":
                # if segmax is longer than semin, truncate it
                if len(self.segmax) > len(self.segmin):
                    self.segmax = self.segmax[:len(self.segmin)]
                
                # if segmax is shorter than segmin, extend it
                if len(self.segmax) < len(self.segmin):
                    self.segmax += self.segmin[len(self.segmax):]
                
                # ensure segmax is not less than segmin
                for i, (minVal, maxVal) in enumerate(zip(change['new'], self.segmax)):
                    self.segmax[i] = max(minVal, maxVal)
                    
                # set seglen to average of segmin/segmax
                self.seglen = [(minVal + maxVal) / 2.0 for minVal, maxVal in zip(self.segmin, self.segmax)]

            elif change["name"] == "segmax":
                # if segmin is longer than semax, truncate it
                if len(self.segmin) > len(self.segmax):
                    self.segmin = self.segmin[:len(self.segmax)]
                
                # if segmin is shorter than segmax, extend it
                if len(self.segmin) < len(self.segmax):
                    self.segmin += self.segmax[len(self.segmin):]
                
                # ensure segmax is not less than segmin
                for i, (minVal, maxVal) in enumerate(zip(change['new'], self.segmin)):
                    self.segmin[i] = min(minVal, maxVal)

                # set seglen to average of segmin/segmax
                self.seglen = [(minVal + maxVal) / 2.0 for minVal, maxVal in zip(self.segmin, self.segmax)]
        
        self.nSegments = len(self.seglen)
        
        # make length of waitUntilVolumeTrigger same as nSegments
        self.waitUntilVolumeTrigger = (self.waitUntilVolumeTrigger + [False] * self.nSegments)[:self.nSegments]
    
    @observe("waitUntilVolumeTrigger")
    def _updateWaitUntilVolumeTrigger(self, change):
        # make same length as seglen
        self.waitUntilVolumeTrigger = (self.waitUntilVolumeTrigger + [False] * self.nSegments)[:self.nSegments]
    '''
    Settings for pglTask
    '''
    def __init__(self):
        super().__init__()
        
    def updateTraitsFromDict(self, data, filename="<dict>", typeConverter=None):
        """
        Override to convert parameter dicts to pglParameter instances.
        Only converts items in the 'parameters' list, not other dict values.
        """
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # ONLY convert the 'parameters' key specifically
        if 'parameters' in data and isinstance(data['parameters'], list):
            converted_params = []
            for item in data['parameters']:
                if isinstance(item, dict):
                    # Extract the two required positional arguments
                    name = item.get('name', 'unnamed')
                    validValues = item.get('validValues', [])
                    
                    # Create pglParameter with those two args
                    param = pglParameter(name, validValues)
                    
                    # Update any other attributes that might be stored
                    # (like blockNum, currentTrial, etc. from your serialized data)
                    for key, value in item.items():
                        if key not in ['name', 'validValues'] and hasattr(param, key):
                            setattr(param, key, value)
                    
                    converted_params.append(param)
                elif isinstance(item, pglParameter):
                    # Already correct type
                    converted_params.append(item)
                else:
                    pglMessages.warning(f"Warning: Unexpected type in parameters: {type(item)}",level=1)
                    converted_params.append(item)
            data['parameters'] = converted_params
        
        # Call parent implementation to handle ALL other traits normally
        pglTraitSettings.updateTraitsFromDict(self, data, filename, typeConverter)
##############################################
# State for pglTask
##############################################
class pglTaskState(pglTraitSettings):
    phaseNum = Int(default_value=None, allow_none=True, help="Current experiment phase number.")
    currentTrial = Int(default_value=0, help="Current trial number.")
    currentSegment = Int(default_value=0, help="Current segment number.")
    subjectResponses = List(Int(), help="List of subject response key codes.")
    gotResponse = Bool(default_value=False, help="whetehr the subject has responded or not")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

##############################################
# State for pglTask
##############################################
class pglTaskData(pglTraitSettings):
    startTime = Float(default_value=None, allow_none=True, help="Task start time.")
    endTime = Float(default_value=None, allow_none=True, help="Task end time.")
    events = List(Instance(pglEvent), help="List of task events.")
    params = List(Dict(), help="List of task parameter dictionaries.")
    trialVariables=List(Dict(),help="List of task variable dictionaries. One for each trial, with any variables computed or discovered by task")
    responseMapping = Dict(default_value={True: ("Correct", "green"), False: ("Incorrect", "red")}, help="response mapping for handleSubjectResponses")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    @property
    def nTrials(self):
        '''
        Compute number of trials that have been run
        '''
        nTrials = 0
        for event in self.events:
            # if we find a new trial event, reset the beginning time
            if isinstance(event, pglEventTrial):
                if event.eventType == "start":
                    nTrials += 1
        return nTrials

    def display(self, taskName="task", responseMapping=None, nTotalTrials=None, ax=None):
        '''
        Display the experiment data.
        '''
        # use responseMapping for displaying subject responses
        if responseMapping is None:
            responseMapping = self.responseMapping

        # get trial timestamps
        trialTimestamps = np.array([e.timestamp for e in self.events if isinstance(e, pglEventTrial)])
        if len(trialTimestamps) <= 2:
            pglMessages.message("Insufficient trial events found to display.")
            return

        # get the max trial length
        maxTrialLength = np.diff(trialTimestamps[:-1]).max()

        # init timeline
        timeline = pglTimelinePlot(ax=ax, startTime=0, endTime=maxTrialLength)

        # init a dict for counting the number of different responseTypes found in the events
        responseCounts = {respType: 0 for respType in responseMapping}

        # for each event, add to timeline
        trialStart = None
        gotResponse = False
        gotOutputTrigger = False
        gotUnsentTrigger = False
        nTrials = 0

        for event in self.events:
            # if we find a new trial event, reset the beginning time
            if isinstance(event, pglEventTrial):
                trialStart = event.timestamp
                if event.eventType == "start":
                    nTrials += 1
            elif trialStart is not None:
                # display segment events
                if isinstance(event, pglEventSegment) and event.eventType == pglEventSegment.boundaryType.START.value:
                    timeline.addTriangleMarker(time=event.timestamp - trialStart, color='blue', label=f'{event.segmentNum}', direction='up')
                # display subject response events
                elif isinstance(event, pglEventSubjectResponse):
                    gotResponse = True
                    label, color = responseMapping.get(event.responseType, ('?', 'gray'))
                    timeline.addTriangleMarker(time=event.timestamp - trialStart, color=color, label=label[0], direction='down')
                    # update response counts
                    if event.responseType in responseCounts:
                        responseCounts[event.responseType] += 1
                # display output trigger events
                elif isinstance(event, pglEventOutputTrigger):
                    if event.timestamp is not None:
                        value = getattr(event, "value", None)
                        label = f"{value}" if value is not None else 'x'
                        sent = getattr(event, "triggerSent", True)
                        if sent:
                            gotOutputTrigger = True
                        else:
                            gotUnsentTrigger = True
                        # display triangle
                        timeline.addTriangleMarker(time=event.timestamp - trialStart, color='purple' if sent else 'orange', label=label, direction='down')

        # compute duration
        if self.endTime is not None and self.startTime is not None:
            duration = pglTimestamp.formatDuration(self.endTime-self.startTime)
        else:
            duration = ""

        if nTotalTrials is not None and not np.isinf(nTotalTrials):
            trials = f"{nTrials}/{int(nTotalTrials)}"
        else:
            trials = f"{nTrials}"

        timeline.setTitle(f"{taskName}: {trials} trials, {duration}")

        # Display legend.
        legend = [{'label': 'Segment', 'color': 'blue'}]

        if gotResponse:
            totalResponses = sum(responseCounts.values())
            for respType, (label, color) in responseMapping.items():
                count = responseCounts.get(respType, 0)
                if count != 0:
                    percent = count / totalResponses * 100 if totalResponses > 0 else 0
                    legend.append({'label': f'{label} (n={count}: {percent:.1f}%)', 'color': color})

        if gotOutputTrigger:
            legend.append({'label': 'Output trigger', 'color': 'purple'})

        if gotUnsentTrigger:
            legend.append({'label': 'Output trigger', 'color': 'orange'})

        timeline.addLegend(legend)
        if ax is None: timeline.show()
    
##############################################
# Task base 
##############################################
class pglTaskBase(pglTraitSettings):
    settings = Instance(pglTaskSettings, allow_none=True, help="Settings for the task")
    state = Instance(pglTaskState, allow_none=True, help="State of task")
    data = Instance(pglTaskData, allow_none=True, help="Data of task")
    parameters = List(Instance(pglParameter), default_value=[], help="parameters of task")
    
    # Keep live image caches and HDF5 handles out of task JSON.
    stimulusDatabase = Instance(f"{__package__}.pglImage.pglStimulusDatabase",allow_none=True,default_value=None,serialize=False,help="Stimulus database associated with this task")
    
    # Runtime-only location of saved acquisition source. Reading is deferred
    # until printSource() is called; these references are never serialized.
    _taskSourceFilesystem = Instance(fsspec.AbstractFileSystem, allow_none=True, default_value=None).tag(serialize=False)
    _taskSourcePath = Unicode(default_value=None, allow_none=True).tag(serialize=False)
  
    def save(self, dataPath, filesystem=None, filesystemPrefix=None):
        '''
        Save current task contents, parameters, and attached stimuli.

        dataPath is the destination run directory. Stimulus export currently
        requires a local filesystem; tasks without stimuli retain remote support.
        Does not read or copy original acquisition source files.
        '''
        filesystem, dataPath, _ = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            create=True
        )
        if filesystem is None:
            raise ValueError("Could not resolve the task destination filesystem")

        taskPath = posixpath.join(dataPath, self._getTaskDirectoryName())
        filesystem.makedirs(taskPath, exist_ok=True)

        self.settings.save(filename=posixpath.join(taskPath, "settings.json"), filesystem=filesystem)
        self.state.save(filename=posixpath.join(taskPath, "state.json"), filesystem=filesystem)
        self.data.save(filename=posixpath.join(taskPath, "data.json"), filesystem=filesystem)

        parameterPath = posixpath.join(taskPath, "parameters")
        filesystem.makedirs(parameterPath, exist_ok=True)

        for parameter in self.parameters:
            parameter.save(dataPath=parameterPath, filesystem=filesystem)

        # Do not serialize the live database; export its portable bundle instead.
        self.saveStimuliStrict(taskPath, filesystem=filesystem)

        pglMessages.message(f"Saved task {self.settings.taskName} to {taskPath}")
        return taskPath

    def saveBundleStrict(self, dataPath, filesystem=None, filesystemPrefix=None):
        '''
        Strictly save task components, parameters, and attached stimuli.

        dataPath is the destination run directory. Errors propagate;
        previously written output may remain.
        '''
        components = {}
        for name in ("settings", "state", "data"):
            component = getattr(self, name, None)
            if component is None: raise ValueError(f"Cannot save task: missing {name}")
            if not callable(getattr(component, "saveStrict", None)):
                raise TypeError(f"Cannot save task {name}: no saveStrict() method")
            components[name] = component

        parameters = list(self.parameters)
        for parameter in parameters:
            if not callable(getattr(parameter, "saveBundleStrict", None)):
                raise TypeError(f"Cannot save parameter {type(parameter).__name__}: no saveBundleStrict() method")

        taskDirectory = self._getTaskDirectoryName()
        filesystem, dataPath, _ = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            create=True
        )
        if filesystem is None: raise ValueError("Could not resolve task destination filesystem")

        taskPath = posixpath.join(dataPath, taskDirectory)
        filesystem.makedirs(taskPath, exist_ok=True)

        for name, component in components.items():
            component.saveStrict(filename=posixpath.join(taskPath, f"{name}.json"), filesystem=filesystem)

        parameterPath = posixpath.join(taskPath, "parameters")
        filesystem.makedirs(parameterPath, exist_ok=True)

        for parameter in parameters:
            parameter.saveBundleStrict(dataPath=parameterPath, filesystem=filesystem)

        # Strict export failures propagate to the caller.
        self.saveStimuliStrict(taskPath, filesystem=filesystem)
        return taskPath
    
    def saveStimuliStrict(self, taskPath, filesystem=None, filesystemPrefix=None):
        '''
        Export this task's stimulus database into its task directory.

        Tasks without a stimulus database require no export. The current
        image exporter supports local destinations only; remote destinations
        raise explicitly rather than being interpreted as local paths.

        Errors propagate so experiment saving can report and retry failures.
        '''
        if self.stimulusDatabase is None: return None

        saveRun = getattr(self.stimulusDatabase, "saveRun", None)
        if not callable(saveRun):
            raise TypeError("Attached stimulus database does not implement saveRun()")

        filesystem, taskPath, _ = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=str(taskPath),
            filesystemPrefix=filesystemPrefix
        )
        if filesystem is None:
            raise ValueError("Could not resolve the stimulus export destination")

        from fsspec.implementations.local import LocalFileSystem
        if not isinstance(filesystem, LocalFileSystem):
            raise NotImplementedError("Stimulus export currently supports local task directories only")

        return saveRun(taskPath)

    def _getTaskDirectoryName(self):
        """Return the directory name used to save this task."""

        #taskDirectoryName = f"task{self.settings.taskID:02d}_{self.settings.taskSaveName}"
        taskDirectoryName = self.getTaskDirectoryName(taskID=self.settings.taskID-1,taskName=self.settings.taskSaveName)

        return taskDirectoryName
    
    @classmethod
    def getTaskDirectoryName(cls, taskID, taskName):
        
        taskDirectoryName = f"task{taskID+1:02d}_{taskName}"
        return taskDirectoryName
    
    @classmethod
    def load(cls, dataPath, filesystem=None):
        '''
        Load task components, optional stimuli, and deferred source access.

        Stimulus manifests load immediately; image pixels remain lazy-loaded.
        Older tasks without a stimulus directory remain supported.

        Saved source is never executed. Source text is read only when
        printSource() is called.
        '''
        pglMessages.message(f"Loading task data from: {dataPath}", messageType="detailed")

        # Resolve the filesystem and task directory.
        filesystem, taskPath, filesystemPrefix = pglStorage.validateFilesystem(filesystem, dataPath)
        if filesystem is None:
            pglMessages.warning(f"Could not resolve task filesystem: {dataPath}")
            return None

        ################################################################
        # Load settings, state, and data
        ################################################################
        try:
            settings = pglSerialize.load(filename=posixpath.join(taskPath, "settings.json"), filesystem=filesystem)
            state = pglSerialize.load(filename=posixpath.join(taskPath, "state.json"), filesystem=filesystem)
            data = pglSerialize.load(filename=posixpath.join(taskPath, "data.json"), filesystem=filesystem)
        except Exception as error:
            pglMessages.warning(f"Could not load task data from {taskPath}: {error}")
            return None

        ################################################################
        # Load parameter bundles
        ################################################################
        parameters = []
        parametersDir = posixpath.join(taskPath, "parameters")

        try:
            for parameterDir in filesystem.ls(parametersDir, detail=False):
                parameter = pglParameter.from_file(parameterDir, filesystem=filesystem)
                if parameter is None:
                    pglMessages.warning(f"Skipping parameter that failed to load: {parameterDir}")
                    continue
                parameters.append(parameter)
        except Exception as error:
            pglMessages.warning(f"Could not load task parameters from {parametersDir}: {error}")
            return None

        ################################################################
        # Construct the task and attach its original source location
        ################################################################
        obj = cls()
        obj.settings = settings
        obj.state = state
        obj.data = data
        obj.parameters = parameters

        # Runtime backing only; do not capture or execute analysis-class source.
        obj._taskSourceFilesystem = filesystem
        obj._taskSourcePath = str(taskPath)

        ################################################################
        # Restore an optional self-contained stimulus bundle
        ################################################################
        stimulusPath = posixpath.join(taskPath, "stimuli")
        if filesystem.exists(stimulusPath):
            from .pglImage import pglImageDatabase

            # An incomplete/corrupt bundle is not equivalent to an older task
            # with no stimulus bundle. Preserve the task data but report failure.
            try:
                obj.stimulusDatabase = pglImageDatabase.fromRun(
                    taskDirectory=taskPath,
                    filesystem=filesystem,
                    filesystemPrefix=filesystemPrefix
                )
            except Exception as error:
                pglMessages.warning(f"Could not load task stimuli from {stimulusPath}: {error}")
                raise

        return obj

    def display(self, ax=None):
        '''
        Display the task data
        '''
        self.data.display(taskName=self.settings.taskName, nTotalTrials=self.settings.nTrials, ax=ax)
    
    def print(self):
        """Print task metadata and zero-based trial details.

        Includes stimulus names when a stimulus database is attached.
        Reads stimulus metadata only, never image pixels.

        Trial times are relative to task start. Volume indices are shown
        when this task has an experiment/run reference supporting matching.
        Counts remain counts; trial and volume indices are zero-based.
        """
        from .pglTimestamp import pglTimestamp

        if self.data.startTime is None:
            pglMessages.message(f"Task {self.settings.taskName} was never run")
            return

        if self.data.endTime is None:
            pglMessages.message(f"Task {self.settings.taskName} has no endTime")
            return

        # Report recorded trial starts, not the current-trial state counter.
        pglMessages.print(f"Task: {self.settings.taskName} | Trials: {self.data.nTrials}")
        pglMessages.print(f"Duration={pglTimestamp.formatDuration(self.data.endTime - self.data.startTime)} | startTime={self.data.startTime} | endTime={self.data.endTime}")
        pglMessages.print("Trial and volume indices are zero-based.")

        # Print segment configuration.
        pglMessages.print(f"seglen={self.settings.seglen}")
        pglMessages.print(f"waitUntilVolumeTrigger={self.settings.waitUntilVolumeTrigger}")

        # Print fixed parameters.
        pglMessages.print("\n".join(f"{key}={value}" for key, value in self.settings.fixedParameters.items()))
        pglMessages.print("-" * 40)

        # Print parameter names.
        for parameter in self.parameters:
            pglMessages.print(f"{parameter.settings.name}")

        # Index START events only. Multiple starts are reported as ambiguous
        # rather than silently choosing one or accidentally selecting an END.
        startsByTrial = {}
        for event in self.data.events:
            if isinstance(event, pglEventTrial) and event.eventType == pglEventTrial.boundaryType.START.value:
                startsByTrial.setdefault(event.trialNum, []).append(event)

        database = self.stimulusDatabase
        experiment = getattr(self, "e", None)
        volumeMatcher = getattr(experiment, "getNearestVolumeTrigger", None)

        def singleLine(value):
            """Keep names and parameter values on one display line."""
            return str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")

        # Preserve one row per recorded parameter entry. Planned stimulus
        # trials beyond this collection are not presented as acquired trials.
        for trialIndex, params in enumerate(self.data.params):
            starts = startsByTrial.get(trialIndex, [])
            trialEvent = starts[0] if len(starts) == 1 else None

            if not starts:
                timingLabel = "no recorded START"
            elif len(starts) > 1:
                timingLabel = "multiple recorded START events"
            elif trialEvent.timestamp is None:
                timingLabel = "START timestamp unavailable"
            else:
                trialStart = trialEvent.timestamp - self.data.startTime
                timingLabel = f"{trialStart:.2f}s"

            label = f"Trial {trialIndex} at {timingLabel}"

            if trialEvent is not None and trialEvent.timestamp is not None and callable(volumeMatcher):
                volumeIndex = volumeMatcher(trialEvent)
                if volumeIndex is not None:
                    label += f" (vol={volumeIndex})"

            details = []

            # getTrial() retrieves the presentation record without loading img.
            if database is not None:
                if trialIndex < database.nTrials:
                    stimulusTrial = database.getTrial(trialIndex)

                    if stimulusTrial.trialNum != trialIndex:
                        stimulusName = "(trial numbering mismatch)"
                    elif stimulusTrial.isBlank:
                        stimulusName = "Blank"
                    elif stimulusTrial.stimulus is None:
                        stimulusName = "(missing stimulus)"
                    else:
                        stimulusName = stimulusTrial.stimulus.name or "(unnamed stimulus)"

                    details.append(f"stimulus={singleLine(stimulusName)}")
                else:
                    details.append("stimulus=(no corresponding stimulus trial)")

            details.extend(f"{key}={singleLine(value)}" for key, value in params.items())
            pglMessages.print(label + (": " + ", ".join(details) if details else ""))
                
    def printSource(self):
        """Print and return the task's original defining module or notebook cell.

        Acquisition tasks use their captured source snapshot.
        Loaded tasks read saved source through their attached filesystem.

        Source is verified against its recorded SHA-256 before printing.
        Missing source warns and returns None; malformed metadata, integrity
        failures, and storage-read errors raise.

        Never executes source or reconstructs historical source from the
        currently installed task implementation.
        """
        import hashlib
        import json

        ################################################################
        # Loaded tasks must use their saved acquisition artifacts
        ################################################################
        if self._taskSourcePath is not None:
            filesystem = self._taskSourceFilesystem
            if filesystem is None: raise RuntimeError("Task source has a saved path but no filesystem.")

            sourcePath = posixpath.join(self._taskSourcePath, "taskSource.py")
            metadataPath = posixpath.join(self._taskSourcePath, "taskSource.json")
            hasSource = filesystem.isfile(sourcePath)
            hasMetadata = filesystem.isfile(metadataPath)

            # Older runs may legitimately have no source artifacts.
            if not hasSource and not hasMetadata:
                pglMessages.warning(f"No saved task source is available in {self._taskSourcePath}")
                return None

            # A source file without provenance cannot be hash-verified.
            if not hasMetadata: raise ValueError(f"Task source metadata is missing: {metadataPath}")

            with filesystem.open(metadataPath, "rt", encoding="utf-8") as sourceFile:
                metadata = json.load(sourceFile)

            if not isinstance(metadata, dict): raise ValueError(f"Invalid task source metadata: {metadataPath}")

            # Failed capture is an explicit recorded outcome, not an excuse
            # to inspect the current Python environment for replacement code.
            if metadata.get("captureStatus") == "failed":
                if hasSource: raise ValueError(f"Task source exists despite failed capture metadata: {sourcePath}")
                pglMessages.warning(f"Task source capture failed: {metadata.get('captureError', 'Unknown error')}")
                return None

            if metadata.get("captureStatus") != "complete":
                raise ValueError(f"Unknown task source capture status: {metadataPath}")
            if not hasSource: raise FileNotFoundError(f"Recorded task source is missing: {sourcePath}")

            with filesystem.open(sourcePath, "rb") as sourceFile:
                sourceBytes = sourceFile.read()

        ################################################################
        # Acquisition tasks use their original in-memory snapshot
        ################################################################
        else:
            snapshot = getattr(self, "_taskSourceSnapshot", None)

            # captureSource() only binds the definition-time class snapshot.
            # It does not inspect current module files or notebook history.
            if snapshot is None:
                captureSource = getattr(self, "captureSource", None)
                if callable(captureSource): snapshot = captureSource()

            if snapshot is None:
                pglMessages.warning("No captured task source or saved source location is available.")
                return None

            metadata = snapshot["metadata"]
            source = snapshot["source"]

            if metadata.get("captureStatus") == "failed":
                pglMessages.warning(f"Task source capture failed: {metadata.get('captureError', 'Unknown error')}")
                return None

            if metadata.get("captureStatus") != "complete" or not isinstance(source, str):
                raise ValueError("Invalid captured task source snapshot.")

            sourceBytes = source.encode("utf-8")

        ################################################################
        # Verify before displaying; return the same text for convenience
        ################################################################
        expectedHash = metadata.get("sha256")
        actualHash = hashlib.sha256(sourceBytes).hexdigest()
        if not isinstance(expectedHash, str) or actualHash != expectedHash:
            raise ValueError(f"Task source SHA-256 mismatch for {metadata.get('className', 'unknown task')}.")

        source = sourceBytes.decode("utf-8")
        print(source, end="" if source.endswith("\n") else "\n")
        return source
                              
##############################################
# Task class
##############################################
class pglTask(pglTaskBase):
    # this are set every trial, which allows us to
    # randomize the length of each segment (based on segmin/segmax)
    # or jump segment, by dynamically changing from Inf to current time
    _thisTrialSeglen = []
    
    # reference to pgl, set by pglExperiment when added
    pgl = None
    
    '''
    Class representing a task in the experiment. For example, a fixation task. Or
    a stimulus task which controls when and what stimuli are presented
    '''
    
    def __init__(self, pgl=None, phaseNum=0):
        self.pgl = pgl
        self.settings = pglTaskSettings()
        self.state = pglTaskState()
        self.data = pglTaskData()
        self.parameters: List[pglParameter] = []

        # Default segment length and phase.
        self.settings.seglen = [1.0]
        self.settings.phaseNum = phaseNum

        self.tasks = None
        self.e = None
        self.waitUntilVolumeTrigger = False

        # Instance-owned runtime timing and lifecycle state.
        self._thisTrialSeglen = []
        self._taskActive = False
        self._trialActive = False
        self._completedTrials = 0
        self._taskEnding = False
        self._segmentJumpRequested = False
        self._segmentAwaitingPresentation = False
        self._segmentScheduledStart = None
        self._segmentStartEvent = None
        self._pendingTrialStart = None
        self._pendingPresentationEvents = []

    def start(self, startTime):
        """Public task-start entry point; subclasses override startTask()."""
        return self._startTask(startTime)

    def _startTask(self, startTime):
        """Initialize the task once, invoke its callback, then start a trial."""
        if self.data.endTime is not None: return False
        if self._taskActive: return True

        self.data.startTime = startTime
        self._taskActive = True
        self.state.currentTrial = -1

        if self.startTask(startTime) == False:
            self._endTask(timestamp=startTime, reason="startRejected")
            return False

        # The callback may explicitly end the task.
        if self.data.endTime is not None: return False

        if self.settings.nTrials <= 0:
            self._endTask(timestamp=startTime, reason="completed")
            return False

        self._startTrial(startTime)
        return self.data.endTime is None

    def startTrial(self, startTime):
        """Subclass callback before segment zero starts.

        Parameters, trialVariables, and finalized segment durations are ready.
        Override modifyTrialSegments() to change the segment durations.

        Do not advance the trial or call _startTrial() from this callback.
        """
        pass

    def _startTrial(self, startTime, *, scheduledStart=None):
        """Initialize trial bookkeeping and timing, then invoke callbacks."""
        from numbers import Real

        if self.data.endTime is not None: return
        if self._trialActive:
            raise RuntimeError("Cannot start a trial while another trial is active.")

        self.state.currentTrial += 1
        self.state.currentSegment = -1
        self.state.trialStartTime = startTime
        self._trialActive = True

        trialEvent = pglEventTrial(self.state.currentTrial, startTime)
        self.data.events.append(trialEvent)
        self._pendingTrialStart = trialEvent

        if self.settings.saveEyeTracker:
            self.e.saveEyeTrackerEvent(eventType="trial", taskID=self.settings.taskID, trialNum=self.state.currentTrial, segmentNum=-1, timestamp=startTime, phaseNum=self.settings.phaseNum)

        ################################################################
        # Generate parameters using this task's verbosity
        ################################################################
        self.data.params.append({})
        self.currentParams = self.data.params[-1]

        for parameter in self.parameters:
            # Apply before get(), including nested parameter blocks.
            parameter.setVerbose(self.settings.verbose)
            self.currentParams.update(parameter.get())

        self.data.trialVariables.append({})

        ################################################################
        # Randomize and finalize this trial's segment durations
        ################################################################
        if not (len(self.settings.segmin) == len(self.settings.segmax) == len(self.settings.waitUntilVolumeTrigger) == self.settings.nSegments):
            raise ValueError("Segment bounds and scanner-wait flags must match settings.nSegments.")

        proposedSeglen = [
            float("inf") if math.isinf(minVal) or math.isinf(maxVal)
            else random.uniform(minVal, maxVal)
            for minVal, maxVal in zip(self.settings.segmin, self.settings.segmax)
        ]
        modifiedSeglen = self.modifyTrialSegments(proposedSeglen)

        if not isinstance(modifiedSeglen, (list, tuple, np.ndarray)):
            raise TypeError("modifyTrialSegments() must return a list, tuple, or one-dimensional NumPy array.")
        if isinstance(modifiedSeglen, np.ndarray) and modifiedSeglen.ndim != 1:
            raise ValueError("modifyTrialSegments() must return a one-dimensional sequence.")
        if not 1 <= len(modifiedSeglen) <= self.settings.nSegments:
            raise ValueError("modifyTrialSegments() must return between one and settings.nSegments lengths.")

        trialSeglen = []
        for segmentIndex, duration in enumerate(modifiedSeglen):
            if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real):
                raise TypeError(f"Segment {segmentIndex} duration must be a real number.")

            duration = float(duration)
            if math.isnan(duration) or duration < 0:
                raise ValueError(f"Segment {segmentIndex} duration must be nonnegative and not NaN.")

            trialSeglen.append(duration)

        self._thisTrialSeglen = trialSeglen

        if self.settings.verbose:
            pglMessages.print(f"({self.settings.taskName}) Trial {self.state.currentTrial + 1}: ", end='', messageType='experiment')
            for name, value in self.currentParams.items(): pglMessages.print(f'{name}={value}', end=' ', messageType='experiment')
            pglMessages.print("", end="\n" if self.settings.trialPrintNewline else " ", messageType='experiment')

        # Subclasses see parameters and finalized durations before segment zero.
        self.startTrial(startTime)
        if self.data.endTime is not None: return

        self._startSegment(startTime, scheduledStart=scheduledStart)

    def _endTrial(self, endTime, *, reason="completed"):
        """Record one trial END and invoke the subclass endTrial() callback."""
        if not self._trialActive: return

        # Close bookkeeping before the callback to prevent reentrant duplicates.
        self._trialActive = False
        self.waitUntilVolumeTrigger = False

        event = pglEventTrial(
            self.state.currentTrial,
            endTime,
            eventType=pglEventTrial.boundaryType.END,
        )
        event.completed = reason == "completed"
        event.endReason = reason
        self.data.events.append(event)

        if event.completed: self._completedTrials += 1

        # This timestamp is a lifecycle boundary, not a measured visual offset.
        self.endTrial(endTime)

    def end(self, *, timestamp=None, reason="stopped"):
        """Public task-end entry point; subclasses override endTask()."""
        return self._endTask(timestamp=timestamp, reason=reason)

    def _endTask(self, *, timestamp=None, reason="stopped"):
        """Close the active trial and task exactly once, even if callbacks fail."""
        if self.data.endTime is not None or self._taskEnding: return
        if not self._taskActive: return

        if timestamp is None: timestamp = self.pgl.getSecs()

        self._taskEnding = True
        errors = []

        try:
            # Publish task termination before invoking callbacks.
            self.data.endTime = timestamp
            self.data.endReason = reason
            self._taskActive = False

            try:
                self._endTrial(timestamp, reason=reason)
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                errors.append(error)

            # Never attach an unpresented start to some later unrelated frame.
            pending = list(self._pendingPresentationEvents)
            self._pendingPresentationEvents.clear()
            self._resolveTaskPresentation(
                pending, None, "taskEndedBeforeFlush", None,
            )

            if self._pendingTrialStart is not None:
                self._pendingTrialStart.presentationTimestamp = None
                self._pendingTrialStart.presentationStatus = "taskEndedBeforeSegmentStart"
                self._pendingTrialStart = None

            try:
                self.endTask()
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                errors.append(error)

            pglMessages.message(f"Ending task {self.settings.taskName}", messageType='experiment')

        finally:
            self._taskEnding = False

        if errors: raise errors[0]

    def done(self, trialNum=None):
        """Query completion without invoking callbacks or changing task state."""
        if self.data.endTime is not None: return True
        if trialNum is not None: return trialNum >= self.settings.nTrials
        return self._completedTrials >= self.settings.nTrials
    
    def update(self, updateTime, subjectResponses, phaseNum, tasks, events):
        """Process input, advance at most once, then draw the current segment."""
        if self.data.endTime is not None: return

        self.tasks = tasks
        self.state.subjectResponses = []
        self.state.phaseNum = phaseNum

        self.handleEvents(events)
        if self.data.endTime is not None: return

        # Give a newly started visual segment a flush opportunity before
        # considering another transition. Responses and drawing still run.
        if not self._segmentAwaitingPresentation:
            duration = self._thisTrialSeglen[self.state.currentSegment]
            deadline = self._segmentScheduledStart + duration
            jumpRequested = self._segmentJumpRequested or duration == 0

            if jumpRequested:
                self._startSegment(updateTime)

            elif self.waitUntilVolumeTrigger:
                if self.e.state.volumeNumber > self.lastVolumeNumber:
                    # Intentional scanner synchronization resets the schedule.
                    self._startSegment(updateTime)

            elif math.isfinite(deadline):
                frame = getattr(self.e, "_taskFrameTiming", None)
                visualTiming = bool(
                    self.e is not None
                    and getattr(self.e, "flush", False)
                    and frame is not None
                    and frame["mode"] == "nearestRefresh"
                )
                waitForTrigger = self.settings.waitUntilVolumeTrigger[self.state.currentSegment]

                if visualTiming and not waitForTrigger:
                    checkTime = (
                        frame["predictedPresentation"]
                        + frame["frameInterval"] / 2
                    )
                else:
                    checkTime = updateTime

                # Numerical tolerance only—not an additional frame allowance.
                tolerance = max(
                    1e-9,
                    4 * math.ulp(float(checkTime)),
                    4 * math.ulp(float(deadline)),
                )

                if checkTime >= deadline - tolerance:
                    if waitForTrigger:
                        self.waitUntilVolumeTrigger = True
                        self.lastVolumeNumber = self.e.state.volumeNumber
                    else:
                        # Do not reset to the late/early actual transition.
                        self._startSegment(updateTime, scheduledStart=deadline)

        if self.data.endTime is not None: return

        for subjectResponse, timestamp in subjectResponses:
            self.state.subjectResponses.append(subjectResponse)
            responseType = self.handleSubjectResponse(subjectResponse, timestamp)

            if responseType is not None:
                self.data.events.append(
                    pglEventSubjectResponse(
                        response=subjectResponse,
                        timestamp=timestamp,
                        responseType=responseType,
                    )
                )

            if self.data.endTime is not None: return

        self.updateScreen()
    
    def _startSegment(self, updateTime, *, scheduledStart=None):
        """Advance one segment and associate its start with the next flush.

        scheduledStart=None resets the schedule for an initial start,
        explicit jump, or scanner-trigger transition.

        A supplied scheduledStart preserves the cumulative intended timeline.

        Presentation association does not require an existing presentation
        reference. Refresh prediction does require one.

        No segment END events are generated.
        """
        if self.data.endTime is not None: return

        # Preserve elapsed duration for the existing explicit-zero jump convention.
        if (
            self.state.currentSegment >= 0
            and self._thisTrialSeglen[self.state.currentSegment] == 0
        ):
            self._thisTrialSeglen[self.state.currentSegment] = (
                updateTime - self.state.segmentStartTime
            )

        if self.state.currentSegment + 1 >= len(self._thisTrialSeglen):
            self._endTrial(updateTime)

            # An endTrial callback may explicitly stop the task.
            if self.data.endTime is not None: return

            if self._completedTrials >= self.settings.nTrials:
                self._endTask(timestamp=updateTime, reason="completed")
            else:
                self._startTrial(updateTime, scheduledStart=scheduledStart)
            return

        self.state.currentSegment += 1
        self.state.segmentStartTime = updateTime
        self.waitUntilVolumeTrigger = False
        self._segmentJumpRequested = False

        frame = getattr(self.e, "_taskFrameTiming", None)

        # Associate starts with experiment-managed flushes even before a
        # presentation timestamp is available for refresh prediction.
        trackPresentation = bool(
            self.e is not None and getattr(self.e, "flush", False)
        )
        predictRefresh = bool(
            trackPresentation
            and frame is not None
            and frame["mode"] == "nearestRefresh"
        )

        predicted = frame["predictedPresentation"] if predictRefresh else None
        resetSchedule = scheduledStart is None

        # Until the flush returns, use a provisional origin for a reset.
        # Ordinary transitions always retain their cumulative scheduled origin.
        if resetSchedule:
            scheduledStart = predicted if predicted is not None else updateTime

        self._segmentScheduledStart = float(scheduledStart)

        segmentEvent = pglEventSegment(self.state.currentSegment, updateTime)
        segmentEvent.trialNum = self.state.currentTrial
        self.data.events.append(segmentEvent)
        self._segmentStartEvent = segmentEvent

        boundaryEvents = [segmentEvent]
        if self._pendingTrialStart is not None:
            boundaryEvents.insert(0, self._pendingTrialStart)
            self._pendingTrialStart = None

        for event in boundaryEvents:
            event.scheduleReset = resetSchedule

            # Preserve the provisional origin separately for auditing.
            event.provisionalScheduledTimestamp = float(scheduledStart)
            event.scheduledTimestamp = (
                None if resetSchedule and trackPresentation
                else float(scheduledStart)
            )

            event.predictedPresentationTime = predicted
            event.presentationTimestamp = None
            event.presentationStatus = (
                "pending" if trackPresentation else "notTracked"
            )
            event.presentationFlushIndex = None
            event.presentationError = None

            event.scheduleBasis = (
                "pendingPresentation" if resetSchedule and trackPresentation
                else "hostClockReset" if resetSchedule
                else "cumulative"
            )

        # Give every newly started visual segment one flush opportunity before
        # advancing again, regardless of whether timing is returned.
        self._segmentAwaitingPresentation = trackPresentation
        if trackPresentation:
            self._pendingPresentationEvents.extend(boundaryEvents)

        if self.settings.saveEyeTracker:
            self.e.saveEyeTrackerEvent(
                eventType="segment",
                taskID=self.settings.taskID,
                trialNum=self.state.currentTrial,
                segmentNum=self.state.currentSegment,
                timestamp=updateTime,
                phaseNum=self.settings.phaseNum,
            )

        self.startSegment(updateTime)

    def _resolveTaskPresentation(self, events, presentedTime, status, flushIndex):
        """Resolve boundary associations for their own flush attempt only.

        Actual presentation timestamps are optional. Missing values remain
        None; predictions and host-clock fallbacks are labeled separately.

        Only explicit schedule resets can establish a new timing origin.
        Ordinary timed boundaries retain the cumulative schedule.
        """
        for event in events:
            event.presentationTimestamp = presentedTime
            event.presentationStatus = status
            event.presentationFlushIndex = flushIndex

            if event.scheduleReset:
                if presentedTime is not None:
                    origin = presentedTime
                    event.scheduleBasis = "actualPresentationReset"

                elif event.predictedPresentationTime is not None:
                    # Refresh-grid estimate, not a measured presentation.
                    origin = event.predictedPresentationTime
                    event.scheduleBasis = "predictionFallback"

                else:
                    # No refresh reference was available for this transition.
                    # Preserve its actual Python transition as the timing origin.
                    origin = event.timestamp
                    event.scheduleBasis = "hostClockFallback"

                event.scheduledTimestamp = float(origin)

                if event is self._segmentStartEvent:
                    self._segmentScheduledStart = float(origin)

            event.presentationError = (
                float(presentedTime - event.scheduledTimestamp)
                if (
                    presentedTime is not None
                    and event.scheduledTimestamp is not None
                )
                else None
            )

            if event is self._segmentStartEvent:
                # A completed flush opportunity is sufficient. Do not wait
                # indefinitely for optional presentation timing.
                self._segmentAwaitingPresentation = False

    def jumpSegment(self):
        """Request an intentional transition and reset timing at the next start."""
        if not self._trialActive or self.data.endTime is not None: return
        self._segmentJumpRequested = True
        self._thisTrialSeglen[self.state.currentSegment] = 0
    
    def configure(self, e):
        '''
        Configure gets called on tasks, just after initScreen runs so that tasks can use
        any information needed from the pglExperiment to initialize, 
        '''
        pass

    def startTask(self, startTime):
        '''
        Can be overridden in subclasses to add custom behavior when the task starts.
        Must return True to run the task as intended, or False to abort running the task
        '''
        return True
    
    def startSegment(self, updateTime):
        '''
        Called exactly once, each time a new segment genuinely starts.
        Override in subclasses to add custom per-segment behavior
        (e.g. loading a stimulus). Base implementation does nothing.
        '''    
        pass
            
    def endTrial(self, endTime):
        '''
        End a trial.
        '''

    def addParameter(self, param):
        '''
        Add a parameter to the task.
        '''
        self.parameters.append(param)

    def handleSubjectResponse(self, response, updateTime) -> None:
        '''
        Handle subject responses. To handle subject responses, override this method
        If you provide a return value (e.g. 1 or 0, or 'correct'/'incorrect'), then
        that value will be stored in the pglEventSubjectResponse event.
        '''
        pass
    
    def handleEvents(self, events) -> None:
        '''
        Handle keyboard/mouse events. For subclasses that need to handle keyboard or mouse
        events (for example to handle typing text), subclass this method.
        '''
        pass
    
    def updateScreen(self):
        '''
        Update the screen.
        '''
        pass
    
    def endTask(self):
        '''
        subclass overrideable function that gets called for any end of task processing
        '''
        pass
            
    def modifyTrialSegments(self, seglen):
        """Modify randomized segment lengths before the first segment starts.

        currentParams already contains this trial's parameter values.
        seglen is a fresh list; changing it does not change task settings.

        Return a nonempty sequence of nonnegative durations, optionally
        shortened. Positive infinity is allowed for indefinite segments.
        Adding more segments than configured is not supported.

        Scanner-wait flags remain associated with their original indices.
        """
        return seglen
        
    ################################################################
    # Automatically capture source when an acquisition task is defined
    ################################################################
    def __init_subclass__(cls, **kwargs):
        """Capture the defining module or notebook cell for each task class."""
        super().__init_subclass__(**kwargs)

        import inspect

        currentFrame = inspect.currentframe()
        definingFrame = None

        try:
            definingFrame = currentFrame.f_back if currentFrame is not None else None

            # Skip cooperative subclass hooks and locate the defining caller.
            while definingFrame is not None:
                if (
                    definingFrame.f_globals.get("__name__") == cls.__module__
                    and definingFrame.f_code.co_name != "__init_subclass__"
                ):
                    break
                definingFrame = definingFrame.f_back

            cls._captureClassSource(definingFrame=definingFrame)

        finally:
            # Do not retain execution frames or their notebook variables.
            del definingFrame
            del currentFrame

    ################################################################
    # Capture a class-specific source snapshot
    ################################################################
    @classmethod
    def _captureClassSource(cls, definingFrame=None):
        """Cache the defining source on this particular task class.

        Notebook source comes from the defining execution frame's source
        cache, not the latest notebook-history entry.

        Module source includes the complete module, including imports and
        module-level helpers. Imported dependencies are not bundled.

        Capture failures are recorded and warned about without preventing
        class definition or acquisition.
        """
        import hashlib
        import inspect
        import linecache
        import sys

        # Only inspect this class's dictionary. Never reuse a parent's source.
        existingSnapshot = cls.__dict__.get("_taskClassSourceSnapshot")
        if existingSnapshot is not None: return existingSnapshot

        metadata = {
            "filename": "taskSource.py",
            "module": cls.__module__,
            "className": cls.__qualname__,
            "sourceScope": None,
            "sourceFile": None,
            "capturedDateTime": datetime.now().astimezone().isoformat(),
            "sha256": None,
            "captureStatus": "failed",
            "captureError": None,
        }
        source = None

        try:
            try:
                from IPython import get_ipython
                shell = get_ipython()
            except ImportError:
                shell = None

            # Imported modules execute in their own namespace, even when
            # imported from a notebook.
            isNotebookSource = (
                shell is not None
                and definingFrame is not None
                and definingFrame.f_globals is shell.user_ns
            )

            if isNotebookSource:
                sourceFile = definingFrame.f_code.co_filename
                metadata["sourceScope"] = "notebookCell"
                metadata["sourceFile"] = sourceFile

                # Retrieve the complete cell associated with this definition.
                source = "".join(
                    linecache.getlines(sourceFile, definingFrame.f_globals)
                )
                if not source:
                    raise OSError(
                        f"IPython source cache has no defining cell for "
                        f"{sourceFile!r}."
                    )

            else:
                module = sys.modules.get(cls.__module__)
                if module is None:
                    raise OSError(
                        f"Could not locate defining module {cls.__module__!r}."
                    )

                metadata["sourceScope"] = "module"
                metadata["sourceFile"] = inspect.getsourcefile(module)

                # Include imports and helper functions from the same module.
                source = inspect.getsource(module)
                if not source:
                    raise OSError("The defining module has no available source.")

            metadata["sha256"] = hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest()
            metadata["captureStatus"] = "complete"

        except Exception as error:
            source = None
            metadata["captureError"] = f"{type(error).__name__}: {error}"

            # Reporting a source problem must not prevent task definition.
            try:
                pglMessages.warning(
                    f"Could not capture source for "
                    f"{cls.__module__}.{cls.__qualname__}: {error}"
                )
            except Exception:
                pass

        snapshot = {
            "source": source,
            "metadata": metadata,
        }

        # Assignment creates an entry on this class rather than its parent.
        cls._taskClassSourceSnapshot = snapshot
        return snapshot

    ################################################################
    # Bind the class snapshot to this acquisition task
    ################################################################
    def captureSource(self):
        """Bind this instance to its definition-time source snapshot.

        Never recover source from current notebook history or current module
        files during saving: those may no longer match the original class.
        """
        import copy

        snapshot = getattr(self, "_taskSourceSnapshot", None)
        if snapshot is not None: return snapshot

        # Do not silently inherit a parent class's source snapshot.
        classSnapshot = type(self).__dict__.get("_taskClassSourceSnapshot")
        if classSnapshot is None:
            raise RuntimeError(
                f"No definition-time source snapshot exists for "
                f"{type(self).__module__}.{type(self).__qualname__}. "
                "Reload its module or rerun its defining notebook cell. "
                "If an intermediate class overrides __init_subclass__(), "
                "it must call super().__init_subclass__()."
            )

        self._taskSourceSnapshot = copy.deepcopy(classSnapshot)
        return self._taskSourceSnapshot

    ################################################################
    # Save the original source with this task's acquisition data
    ################################################################
    def saveSourceStrict(self, taskPath):
        """Write source and provenance into a local acquisition task directory.

        taskPath is the task directory, not the parent run directory.

        Failed source capture still writes explanatory metadata, then raises.
        The experiment's independent-save logic can continue saving task data.
        """
        import hashlib
        import json

        snapshot = self.captureSource()
        metadata = snapshot["metadata"]
        source = snapshot["source"]

        taskPath = Path(taskPath)
        taskPath.mkdir(parents=True, exist_ok=True)

        if metadata["captureStatus"] == "complete" and source is not None:
            sourceBytes = source.encode("utf-8")

            # Detect accidental modification of the cached snapshot.
            if hashlib.sha256(sourceBytes).hexdigest() != metadata["sha256"]:
                raise ValueError(
                    f"Cached task source hash mismatch for "
                    f"{metadata['className']}."
                )

            # Write exactly the bytes used for the recorded hash.
            (taskPath / "taskSource.py").write_bytes(sourceBytes)

        (taskPath / "taskSource.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        if metadata["captureStatus"] != "complete" or source is None:
            raise RuntimeError(
                f"Task source was not captured for {metadata['className']}: "
                f"{metadata['captureError']}"
            )

        return taskPath / "taskSource.py"

################################################################
# Capture the base acquisition task itself.
# Subclasses capture their own source through __init_subclass__().
################################################################
pglTask._captureClassSource()

##############################################
# small class to carry information about digital IO words
##############################################
class pglDigitalIOWord(pglTraitSettings):
    # channel
    key = Unicode("", help='String that can be used by sendTriggerWord to set word value')
    value = Int(help="Word that will be sent")
    pulseLen = Int(help="length of pulse")
    deviceType = Unicode("", help="type of device that the channel was configured on")
    
##############################################
# small class to carry information about digitalIOChannels
##############################################
class pglDigitalIOChannel(pglTraitSettings):

    channelKey = Unicode(help="Unique trigger name associated with the logical channel")
    channel = Int(help="Channel number that has been configured")
    channelName = Unicode(allow_none=True, default_value=None, help="Name of channel if available")
    groupName = Unicode(allow_none=True, help='Name of channel group')
    pulseLen = Int(help="length of pulse that has been configured")
    deviceType = Unicode("", help="type of device that the channel was configured on")
    
##############################################
# Settings for pglExperiment
##############################################
class pglExperimentSettings(pglTraitSettings):
    experimentName = Unicode("Default experiment", help="Name of the experiment")
    sessionName = Unicode("", help="Session name of experiment")
    runName = Unicode("", help="Name of run of experiment")
    experimenterName = Unicode("", help="Name of experimenter who ran in experiment")
    experimentSaveName = Unicode("defaultExperiment", help="Name to use when saving experiment data (defaults to camelCase version of experimentName)")
    subjectID = Unicode("s0000", help="Identifier for the subject participating in the experiment.")
    tasks = List(Unicode(), default_value=[], help="Task names")
    digitalIOChannels = List(Instance(pglDigitalIOChannel), help='List of digitalIO channesl that have been configured')
    digitalIOWords = List(Instance(pglDigitalIOWord), default_value=[], help='List of words, keys and associated values that have been configured')

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    # observe changes to experimentName and if experimentSaveName is not set
    # set experimentSaveName to a camelCase version of experimentName
    @observe("experimentName")
    def toCamelCase(self, change) -> None:
        if self.experimentSaveName == "" or self.experimentSaveName == "defaultExperiment":
            # split experimentName into words
            words = change['new'].strip().split()
            if not words:
                return
            # convert to camelCase and save as experimentSaveName
            self.experimentSaveName = words[0].lower() + "".join(word.capitalize() for word in words[1:])
    
    @validate("subjectID")
    def _validateSubjectID(self, proposal):
        """Warn about unconventional subject IDs without rejecting the assignment."""
        import re

        value = proposal["value"]

        # Accept a nonempty prefix containing at least one nondigit,
        # followed by one or more digits.
        # Preserve the supplied ID, including capitalization and leading zeros.
        if not re.fullmatch(r".*[^0-9][0-9]+", value):
            pglMessages.warning(
                f"Unconventional subjectID {value!r}. Expected a text prefix "
                "followed by one or more digits, such as 's1', 'subject00001', "
                "or 'sdbp_nsdmini_0001'. Continuing with the supplied subjectID.",
                level=2,
            )

        return value
##############################################
# Data for pglExperiment
##############################################
class pglExperimentData(pglTraitSettings):
    startTime = Float(0.0, help="Time in secs of start of experiment")
    startDateTime = Unicode("", help="Experiment start date and time in ISO format")
    endTime = Float(0.0, help="Time in secs of end of experiment")
    events = List(Instance(pglEvent), default_value=[], help="List of events from experiment")

    # make sure that any settings that the experimenter writes into settings get saved
    _serializeUnregisteredFields = True

    def __repr__(self):
        return f"pglExperimentData(startTime={self.startTime}, endTime={self.endTime}, {len(self.events)} events)"
    
    def getNumEvents(self, type=None, eventType=None, keyChar=None):
        # filter for type
        if type is None:
            return len(self.events)
        filteredEvents = [event for event in self.events if event.type == type]
        # filter for events
        if eventType is not None:
            filteredEvents = [event for event in filteredEvents if event.eventType == eventType]
        # filter for keyChar
        if keyChar is not None:
            filteredEvents = [event for event in filteredEvents if getattr(event, "keyChar", None) == keyChar]
        return len(filteredEvents)
    
    def display(self, e=None, ax=None, discrepancyMs=100.0):
        '''
        Display the experiment data.

        Use the same volume-timing diagnostics as the printed summary.
        Put MRI volume time and diagnostic counts on the second title line.
        Suspected manual final triggers remain visible but are excluded
        from missing-volume and interval-discrepancy counts.
        '''
        import math
        from numbers import Real

        if len(self.events) == 0:
            pglMessages.print("(pglExperimentData) No events to display.")
            return

        diagnostics = self.getVolumeTimingDiagnostics(discrepancyMs=discrepancyMs)

        # get info from experiment if provided
        if e is not None:
            self.volumeTriggerKey = e.settings.volumeTriggerKey
        else:
            #settings = pglSettingsManager.getSettings()
            #self.volumeTriggerKey = settings.volumeTriggerKey
            self.volumeTriggerKey = "`"

        # Invalid event timestamps cannot be plotted. Keep the original events
        # untouched and report how many markers were omitted.
        def validTimestamp(value):
            return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and math.isfinite(float(value))

        if not validTimestamp(self.startTime) or not validTimestamp(self.endTime):
            pglMessages.warning("Cannot display experiment events: startTime or endTime is invalid.", level=1)
            return

        # Get the time at which start the timeline, if there is a keyboard
        # event that happens before the start of the experiment (like when the experimenter
        # hits space to start the experiment), then adjust the start time to show that as a negative time)
        startTime = 0
        firstKeydownEvent = next((event for event in self.events if event.type == "keyboard" and event.eventType == "keydown" and validTimestamp(event.timestamp)), None)
        if firstKeydownEvent is not None:
            if firstKeydownEvent.timestamp < self.startTime:
                startTime = firstKeydownEvent.timestamp - self.startTime

        # Include added pre-start volumes and appended ending volumes in the
        # visible timeline without modifying acquisition start/end times.
        volumeTimes = [float(event.timestamp) - self.startTime for event in self.events if event.type == "volumeTrigger" and validTimestamp(event.timestamp)]
        if volumeTimes: startTime = min(startTime, min(volumeTimes))
        endTime = max(self.endTime - self.startTime, 10)
        if volumeTimes: endTime = max(endTime, max(volumeTimes))

        # track number of volumes
        nVols = diagnostics["nVolumes"] - int(diagnostics["suspectedManualFinal"] is not None)
        nKeys = 0
        nOutputTriggers = 0
        nUnsentTriggers = 0
        nInvalidTimestamps = 0

        # init timeline
        timeline = pglTimelinePlot(ax=ax, startTime=startTime, endTime=endTime)

        # for each event, add to timeline
        for event in self.events:
            if event.type not in ("keyboard", "volumeTrigger", "outputTrigger"): continue

            if not validTimestamp(event.timestamp):
                nInvalidTimestamps += 1
                continue

            if event.type == "keyboard":
                if event.eventType == "keydown":
                    if event.keyChar != self.volumeTriggerKey:
                        timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='green', label=f'{event.keyChar}', direction='down')
                        nKeys += 1
                elif (event.keyChar == "escape"):
                    timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='red', label=f'{event.keyChar}', direction='down')
                    nKeys += 1
            elif event.type == "volumeTrigger":
                timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='blue', direction='up')
            elif event.type == "outputTrigger":
                value = getattr(event, "value", None)
                label = f"{value}" if value is not None else 'x'
                sent = getattr(event, "triggerSent", True)
                if sent:
                    nOutputTriggers += 1
                else:
                    nUnsentTriggers += 1
                # set marker
                timeline.addTriangleMarker(time=event.timestamp - self.startTime, color='purple' if sent else 'orange', label=label, direction='up')

        if nInvalidTimestamps:
            pglMessages.warning(f"Skipped {nInvalidTimestamps} event markers with invalid timestamps.", level=1)

        if self.startDateTime != "":
            dateTime = datetime.fromisoformat(self.startDateTime)
            readableTime = dateTime.strftime("%I:%M %p").lstrip("0")
            title = f"Experiment Events: {dateTime:%b. %d, %Y} at {readableTime}"
        else:
            title = "Experiment Events"

        # Keep all volume-timing information together on the second title line.
        # Detection is shared with print(); do not duplicate its calculations here.
        if not diagnostics["valid"]:
            title += "\nMRI volume timing unavailable"
            pglMessages.warning(diagnostics["error"], level=1)
        elif diagnostics["median"] is not None:
            details = [f"MRI volume time: {diagnostics['median']:.3f}s (std={diagnostics['std']:.3f}s, min={diagnostics['min']:.3f})s max={diagnostics['max']:.3f})s"]

            if diagnostics["nMissingVolumes"]:
                details.append(f"\nMissing volumes={diagnostics['nMissingVolumes']}")

            if diagnostics["timingIntervals"]:
                details.append(f"\nInterval discrepancies ≥{discrepancyMs:g} ms={len(diagnostics['timingIntervals'])}")

            if diagnostics["suspectedManualFinal"] is not None:
                details.append("\nSuspected manual final trigger=1 (excluded from diagnostic counts)")

            title += "\n" + "; ".join(details)

        timeline.setTitle(title)

        legend = [{'label': f'Keypress (n={nKeys})', 'color': 'green'}]

        if nVols > 0:
            legend.append({'label': f'Volumes (nVolumes={nVols})', 'color': 'blue'})

        if nOutputTriggers > 0:
            legend.append({'label': f'Output triggers (n={nOutputTriggers})', 'color': 'purple'})

        if nUnsentTriggers > 0:
            legend.append({'label': f'Unsent triggers (n={nUnsentTriggers})', 'color': 'orange'})

        timeline.addLegend(legend)

        if ax is None: timeline.show()
    
    def getTriggerStats(self):
        '''
        Get the median time between volume triggers.
        '''
        # get all volume trigger events
        volumeTriggerEvents = [event for event in self.events if event.type == "volumeTrigger"]
        # get the timestamps of the volume trigger events
        timestamps = [event.timestamp for event in volumeTriggerEvents]
        # get the differences between the timestamps
        diffs = np.diff(timestamps)
        # return the median of the differences
        return Stats(
            mean=np.mean(diffs) if len(diffs) > 0 else None,
            median=np.median(diffs) if len(diffs) > 0 else None,
            std=np.std(diffs) if len(diffs) > 0 else None,
            min=np.min(diffs) if len(diffs) > 0 else None,
            max=np.max(diffs) if len(diffs) > 0 else None,
            last=diffs[-1] if len(diffs) > 0 else None
        )
        
    ################################################################
    # Shared MRI volume-timing diagnostics
    ################################################################
    def getVolumeTimingDiagnostics(self, discrepancyMs=100.0):
        """Describe trigger intervals without modifying events.

        Missing volumes are inferred from the nearest positive integer
        multiple of the median interval. Exact ties use the smaller multiple.

        A final gap longer than 1.5 times the median is treated as a possible
        manual ending trigger unless it is within 100 ms of twice the median.

        mean and median retain their original raw-interval definitions.
        rawStd also describes all raw intervals.

        std, min, max, and adjustedMean use intervals normalized as:
            interval / (nMissing + 1)
        Suspected manual final gaps are excluded from these statistics.
        Each observed gap contributes one normalized interval, not fabricated
        individual intervals for the missing triggers.

        Normalization estimates average spacing within a gap; it does not
        recover individual missing-trigger timing or measure its jitter.
        """
        import math
        import numpy as np

        if isinstance(discrepancyMs, bool) or not isinstance(discrepancyMs, (int, float)):
            raise TypeError("discrepancyMs must be a number.")
        if not math.isfinite(discrepancyMs) or discrepancyMs < 0:
            raise ValueError("discrepancyMs must be finite and nonnegative.")

        volumeEvents = [event for event in self.events if event.type == "volumeTrigger"]
        result = {
            "nVolumes": len(volumeEvents),
            "discrepancyMs": discrepancyMs,
            "valid": True,
            "error": None,
            "median": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "rawStd": None,
            "adjustedMean": None,
            "adjustedIntervals": [],
            "nMissingVolumes": 0,
            "missingIntervals": [],
            "timingIntervals": [],
            "suspectedManualFinal": None,
        }

        if len(volumeEvents) < 2: return result

        # Preserve recorded order; do not sort away duplicate or reversed timestamps.
        try:
            times = np.asarray([event.timestamp for event in volumeEvents], dtype=float)
            intervals = np.diff(times)
            if not np.isfinite(times).all() or not np.isfinite(intervals).all() or np.any(intervals <= 0):
                raise ValueError("Invalid, duplicate, or reversed volume-trigger timestamps.")
        except (TypeError, ValueError, OverflowError) as error:
            result["valid"] = False
            result["error"] = str(error)
            return result

        medianVolumeTime = float(np.median(intervals))
        result["median"] = medianVolumeTime

        # Preserve raw statistics and the existing detection baseline.
        result["mean"] = float(np.mean(intervals))
        result["rawStd"] = float(np.std(intervals))
        threshold = discrepancyMs / 1000.0
        adjustedIntervals = []

        for intervalIndex, interval in enumerate(intervals):
            # Choose the nearest positive multiple; exact ties prefer
            # the smaller multiple rather than infer an extra missing volume.
            multiple = max(1, int(np.ceil(interval / medianVolumeTime - 0.5)))
            expectedInterval = multiple * medianVolumeTime
            timingError = float(interval - expectedInterval)

            record = {
                "firstTrigger": intervalIndex + 1,
                "secondTrigger": intervalIndex + 2,
                "interval": float(interval),
                "expectedInterval": expectedInterval,
                "errorMs": timingError * 1000.0,
                "nMissing": multiple - 1,
            }

            # The final trigger may be manual after acquisition stopped.
            # Only infer one missing volume when this gap is close to 2x.
            if intervalIndex == len(intervals) - 1:
                closeToDouble = multiple == 2 and abs(interval - 2 * medianVolumeTime) <= 0.100 + 1e-9

                if interval > 1.5 * medianVolumeTime and not closeToDouble:
                    result["suspectedManualFinal"] = record
                    continue

                record["nMissing"] = 1 if closeToDouble else 0

            # Normalize using the missing count actually assigned above,
            # including the existing special handling of the final interval.
            adjustedInterval = float(interval / (record["nMissing"] + 1))
            record["adjustedInterval"] = adjustedInterval
            adjustedIntervals.append(adjustedInterval)

            if record["nMissing"]:
                result["nMissingVolumes"] += record["nMissing"]
                result["missingIntervals"].append(record)

            # Missing-volume gaps are not themselves timing errors.
            # Compare against their inferred expected interval instead.
            if abs(timingError) > 1e-9 and abs(timingError) >= threshold - 1e-9:
                result["timingIntervals"].append(record)

        result["adjustedIntervals"] = adjustedIntervals
        if adjustedIntervals:
            result["adjustedMean"] = float(np.mean(adjustedIntervals))
            result["std"] = float(np.std(adjustedIntervals))
            result["min"] = float(np.min(adjustedIntervals))
            result["max"] = float(np.max(adjustedIntervals))

        return result

    def getLabJackClockReport(self):
        """
        Describe retained LabJack timing evidence without hardware or mutation.

        Conditional error summaries come from correction audit proposals.
        Rejected proposals are never described as applied corrections.
        Audit target counts may include raw inputs, volume events, counter
        observations, and independent copies; they are not pulse counts.
        """
        from copy import deepcopy
        from numbers import Real
        import math

        def finite(value):
            return (
                not isinstance(value, bool)
                and isinstance(value, Real)
                and math.isfinite(float(value))
            )

        recordsById = {}
        for event in self.events:
            if event.type not in (
                "labJackClockCalibration", "labJackClockMapping",
            ):
                continue
            eventId = getattr(event, "eventId", None)
            recordsById.setdefault(eventId, []).append(event)

        mappingReports = []
        calibrationReports = []
        recordIssues = []

        for eventId, records in recordsById.items():
            if not isinstance(eventId, str) or not eventId:
                recordIssues.append({
                    "eventId": eventId,
                    "reason": "missingClockRecordIdentity",
                })
            elif len(records) != 1:
                recordIssues.append({
                    "eventId": eventId,
                    "reason": "duplicateClockRecordIdentity",
                    "count": len(records),
                })

            for event in records:
                if event.type == "labJackClockCalibration":
                    calibrationReports.append({
                        "eventId": eventId,
                        "connectionId": getattr(event, "connectionId", None),
                        "phase": getattr(event, "phase", None),
                        "status": getattr(event, "status", None),
                        "actualDurationSeconds": getattr(event, "actualDuration", None),
                        "observationCount": len(getattr(event, "observations", [])),
                    })
                    continue

                referenceIssues = []
                calibrationIds = list(getattr(event, "calibrationEventIds", []))
                for calibrationId in calibrationIds:
                    matches = recordsById.get(calibrationId, [])
                    if len(matches) != 1:
                        referenceIssues.append({
                            "calibrationEventId": calibrationId,
                            "reason": "missing" if not matches else "ambiguous",
                        })
                    elif matches[0].type != "labJackClockCalibration":
                        referenceIssues.append({
                            "calibrationEventId": calibrationId,
                            "reason": "wrongRecordType",
                        })
                    elif getattr(matches[0], "connectionId", None) != getattr(
                        event, "connectionId", None
                    ):
                        referenceIssues.append({
                            "calibrationEventId": calibrationId,
                            "reason": "foreignConnection",
                        })

                parameters = getattr(event, "parameters", None) or {}
                scale = parameters.get("clockScale")
                scaleBounds = parameters.get("feasibleClockScale")
                if scaleBounds is None:
                    region = parameters.get("affineRegion")
                    if (
                        isinstance(region, list) and region
                        and all(
                            isinstance(point, (list, tuple))
                            and len(point) == 2
                            and finite(point[1])
                            for point in region
                        )
                    ):
                        scaleBounds = [
                            1 + min(point[1] for point in region),
                            1 + max(point[1] for point in region),
                        ]

                validBounds = (
                    isinstance(scaleBounds, (list, tuple))
                    and len(scaleBounds) == 2
                    and all(finite(value) for value in scaleBounds)
                    and 0 < scaleBounds[0] <= scaleBounds[1]
                )
                usableReferences = (
                    not referenceIssues
                    and isinstance(eventId, str) and bool(eventId)
                    and len(records) == 1
                )
                applicable = (
                    getattr(event, "status", None) == "validConditional"
                    and usableReferences
                )

                mappingReports.append({
                    "mappingId": eventId,
                    "purpose": getattr(event, "purpose", None),
                    "status": getattr(event, "status", None),
                    "method": getattr(event, "method", None),
                    "calibrationEventIds": calibrationIds,
                    "referenceIssues": referenceIssues,
                    "selectedScaleDeviationPpm": (
                        (float(scale) - 1) * 1e6
                        if applicable and finite(scale) and scale > 0 else None
                    ),
                    "feasibleScaleDeviationPpm": (
                        [(float(value) - 1) * 1e6 for value in scaleBounds]
                        if applicable and validBounds else None
                    ),
                    "failureReason": (
                        getattr(event, "diagnostics", {}) or {}
                    ).get("failureReason"),
                })

        audits = []
        for sourceAudit in getattr(self, "labJackClockCorrectionAudit", []):
            proposals = sourceAudit.get("proposals", [])
            corrections = []
            lowerErrors, upperErrors = [], []
            invalidProposalCount = 0

            for proposal in proposals:
                correction = proposal.get("correctionSeconds")
                lower = proposal.get("clockErrorLowerSeconds")
                upper = proposal.get("clockErrorUpperSeconds")
                if finite(correction):
                    corrections.append(float(correction))
                else:
                    invalidProposalCount += 1
                if finite(lower) and finite(upper) and lower <= upper:
                    lowerErrors.append(float(lower))
                    upperErrors.append(float(upper))

            exclusions = {}
            for exclusion in sourceAudit.get("exclusions", []):
                reason = exclusion.get("reason", "unspecified")
                exclusions[reason] = exclusions.get(reason, 0) + 1

            mappingId = sourceAudit.get("mappingId")
            matching = recordsById.get(mappingId, [])
            mappingResolved = (
                len(matching) == 1
                and matching[0].type == "labJackClockMapping"
            )

            audits.append({
                "auditId": sourceAudit.get("auditId"),
                "mappingId": mappingId,
                "mappingRecordResolved": mappingResolved,
                "status": sourceAudit.get("status", "unknown"),
                "failureReason": sourceAudit.get("failureReason"),
                "affectedTargetCount": sourceAudit.get("affectedTargetCount"),
                "proposalCount": len(proposals),
                "invalidCorrectionValueCount": invalidProposalCount,
                "correctionRangeSeconds": (
                    [min(corrections), max(corrections)] if corrections else None
                ),
                "maximumAbsoluteCorrectionSeconds": (
                    max(abs(value) for value in corrections) if corrections else None
                ),
                # Envelope over recorded target-specific intervals, not a
                # newly inferred interval for every point throughout the run.
                "recordedTargetClockErrorEnvelopeSeconds": (
                    [min(lowerErrors), max(upperErrors)] if lowerErrors else None
                ),
                "targetsWithRecordedClockBounds": len(lowerErrors),
                "exclusionsByReason": exclusions,
                "estimatedVolumePolicy": sourceAudit.get("estimatedVolumePolicy"),
                "historicalAcceptanceDecisionsChanged": sourceAudit.get(
                    "historicalAcceptanceDecisionsChanged", False
                ),
            })

        finalization = deepcopy(getattr(self, "labJackClockFinalization", None))
        present = bool(recordsById or audits or finalization)
        return {
            "availability": "retainedEvidence" if present else "unavailable",
            "finalization": finalization,
            "calibrations": calibrationReports,
            "mappings": mappingReports,
            "recordIssues": recordIssues,
            "correctionAudits": audits,
            "saveRetryAudit": deepcopy(getattr(self, "labJackClockSaveRetryAudit", [])),
            "uncertaintyBasis": "recordedPerTargetConditionalAffineIntervals",
            "unknownContributions": [
                "hostTimestampUncertainty",
                "withinRunNonlinearClockVariation",
            ],
            "totalTimingBoundAvailable": False,
        }

    def printLabJackClockReport(self):
        """Print saved clock/correction evidence directly; return its report."""
        report = self.getLabJackClockReport()
        print("LabJack experiment clock report")

        if report["availability"] == "unavailable":
            print(
                "Clock-calibration/correction evidence is unavailable. "
                "This recording may predate calibrated device timing."
            )
            print("Timing uncertainty is unknown, not zero.")
            return report

        finalization = report["finalization"]
        if finalization:
            print(
                f"Finalization: {finalization.get('status', 'unknown')} | "
                f"timing persistence: "
                f"{finalization.get('correctedSaveStatus', 'unknown')}"
            )
            if finalization.get("failureReason"):
                print(f"Finalization issue: {finalization['failureReason']}")

        for calibration in report["calibrations"]:
            duration = calibration["actualDurationSeconds"]
            durationText = (
                f"{duration * 1000:.2f} ms"
                if isinstance(duration, (int, float)) else "unavailable"
            )
            print(
                f"  Collection {calibration['phase']}: {calibration['status']}, "
                f"{calibration['observationCount']} observations, "
                f"{durationText} actual [{calibration['eventId']}]"
            )

        for issue in report["recordIssues"]:
            print(f"RECORD ISSUE: {issue}")

        for mapping in report["mappings"]:
            print(
                f"Mapping {mapping['mappingId']}: "
                f"{mapping['purpose']} / {mapping['status']}"
            )
            for issue in mapping["referenceIssues"]:
                print(f"  UNRESOLVED: {issue}")
            if mapping["failureReason"]:
                print(f"  Failure: {mapping['failureReason']}")
            if mapping["selectedScaleDeviationPpm"] is not None:
                print(
                    f"  Selected scale deviation: "
                    f"{mapping['selectedScaleDeviationPpm']:+.3f} ppm"
                )
            bounds = mapping["feasibleScaleDeviationPpm"]
            if bounds is not None:
                print(
                    f"  Conditional supported scale deviation: "
                    f"[{bounds[0]:+.3f}, {bounds[1]:+.3f}] ppm"
                )

        for audit in report["correctionAudits"]:
            status = audit["status"]
            applied = status == "applied"
            label = "Applied" if applied else "Proposed / not applied"
            if status == "noChanges":
                label = "No changes in this attempt"
            print(f"Correction audit {audit['auditId']}: {status}")
            print(
                f"  Mapping: {audit['mappingId']} "
                f"(reference {'resolved' if audit['mappingRecordResolved'] else 'UNRESOLVED'})"
            )
            print(
                f"  {label}: {audit['proposalCount']} targets. "
                "Targets can include copies and counter observations, not just pulses."
            )

            values = audit["correctionRangeSeconds"]
            if values is not None:
                print(
                    f"  {label} correction range: "
                    f"[{values[0] * 1000:+.6f}, {values[1] * 1000:+.6f}] ms; "
                    f"maximum absolute: "
                    f"{audit['maximumAbsoluteCorrectionSeconds'] * 1000:.6f} ms"
                )
            bounds = audit["recordedTargetClockErrorEnvelopeSeconds"]
            if bounds is not None:
                print(
                    f"  Envelope of recorded target-specific conditional clock "
                    f"errors: [{bounds[0] * 1e6:+.2f}, "
                    f"{bounds[1] * 1e6:+.2f}] µs "
                    f"({audit['targetsWithRecordedClockBounds']} targets)"
                )
            else:
                print("  Recorded target-specific clock bounds: unavailable.")

            if audit["failureReason"]:
                print(f"  Rejection/failure: {audit['failureReason']}")
            for reason, count in audit["exclusionsByReason"].items():
                print(f"  Excluded: {reason} ({count})")
            if audit["invalidCorrectionValueCount"]:
                print(
                    f"  Invalid/missing correction values: "
                    f"{audit['invalidCorrectionValueCount']}"
                )

        if not report["correctionAudits"]:
            print("No recorded correction attempt.")

        print(
            "Conditional clock-model intervals only—not total event-timing accuracy. "
            "The target envelope is not a bound on unobserved within-run drift."
        )
        print("Unknown: " + ", ".join(report["unknownContributions"]))
        print(
            "Sampling/edge localization, channel timing, and presentation "
            "uncertainty remain separate where applicable."
        )
        return report
    
##############################################
# State for pglExperiment
##############################################
class pglExperimentState(pglTraitSettings):
    phaseNum = Int(default_value=None, allow_none=True, help="Current experiment phase number.")
    phaseNums = List(trait=Int(), default_value=None, allow_none=True, help="List of experiment phase numbers.")
    currentPhaseIndex = Int(default_value=0, help="Index into phaseNums for the current phase.")

    openScreen = Bool(default_value=False, help="Whether the experiment screen is currently open.")
    runFinishedWithError = Bool(default_value=False, help="Whether experiment acquisition ended with an error.")
    volumeNumber = Int(default_value=0, help="Current scanner-volume number.")
    experimentStarted = Bool(default_value=False, help="Whether the experiment has started.")
    experimentDone = Bool(default_value=False, help="Whether the experiment has completed.")

    # None means the corresponding control key is disabled.
    responseKeyCodesList = List(Int(), help="List of subject response key codes.")
    startKeyCode = Int(default_value=None, allow_none=True, help="Key code used to begin the experiment; None disables the key.")
    endKeyCode = Int(default_value=None, allow_none=True, help="Key code used to end the experiment; None disables the key.")
    volumeTriggerKeyCode = Int(default_value=None, allow_none=True, help="Scanner-volume trigger key code; None disables the key.")

    originalGammaTable = Tuple(List, List, List, default_value=None, allow_none=True, help="Original display gamma table: (red, green, blue).")
    gammaTable = Tuple(List, List, List, default_value=None, allow_none=True, help="Current display gamma table: (red, green, blue).")
    display = Instance(pglDisplaySettings, default_value=None, allow_none=True, help="Current display settings.")
    originalScreenResolution = Tuple(Int(), Int(), Int(), Int(), default_value=None, allow_none=True, help="Original screen resolution.")
    screenResolution = Tuple(Int(), Int(), Int(), Int(), default_value=None, allow_none=True, help="Current screen resolution.")

    digitalIODevices = List(Instance(pglDigitalIODevice), help="List of digital IO devices the experiment is using.")
    triggerWords = Dict(help="Runtime lookup of trigger word keys to configurations and devices.")
    triggers = Dict(help="Runtime lookup of trigger keys to configurations and devices.")

    # Make sure any variables added by experimenter code get saved.
    _serializeUnregisteredFields = True

#################################################################
# Events that specify trial timing
#################################################################
class pglEventTrial(pglEvent):

    class boundaryType(Enum):
        START = 'start'
        END = 'end'

    def __init__(self, trialNum=None, timestamp=None, eventType=None):
        super().__init__(type="trial")

        # handle default
        if eventType is None:
            eventType = self.boundaryType.START
            
        # set attributes
        self.trialNum = trialNum
        self.timestamp = timestamp
        self.eventType = eventType.value

    def print(self):
        pglMessages.message(f"Trial {self.eventType} at: {self.timestamp}")
        
#################################################################
# Events that specify segment timing
#################################################################
class pglEventSegment(pglEvent):

    class boundaryType(Enum):
        START = 'start'
        END = 'end'

    def __init__(self, segmentNum = None, timestamp=None, eventType=None):
        super().__init__(type="segment")

        # handle default
        if eventType is None:
            eventType = self.boundaryType.START
        
        # set attributes
        self.segmentNum = segmentNum
        self.eventType = eventType.value
        self.timestamp = timestamp

    def print(self):
        pglMessages.message(f"Segment {self.segmentNum} {self.eventType} at: {self.timestamp}")
        

#################################################################
# Events that specify subject response
#################################################################
class pglEventSubjectResponse(pglEvent):
    
    def __init__(self, response=None, timestamp=None, responseType=None):
        super().__init__(type="subjectResponse")
        
        # set attributes
        self.response = response
        self.timestamp = timestamp
        self.responseType = responseType

#################################################################
# Events that specifys mri volume trigger
#################################################################
class pglEventVolumeTrigger(pglEvent):
    
    def __init__(self, timestamp=None):
        super().__init__(type="volumeTrigger")
        
        # set attributes
        self.timestamp = timestamp

#################################################################
# Events that specifys a trigger was sent
#################################################################
class pglEventOutputTrigger(pglEvent):
    
    def __init__(self, triggerName=None, triggerValue=None, triggerWord=None, timestamp=None, triggerSent=True):
        super().__init__(type="outputTrigger")
        
        # set attributes
        self.timestamp = timestamp
        self.triggerName = triggerName
        self.triggerSent = triggerSent

        # set value
        if triggerValue is not None:
            self.value = triggerValue
            self.triggerWord = False
        elif triggerWord is not None:
            self.value = triggerWord
            self.triggerWord = True
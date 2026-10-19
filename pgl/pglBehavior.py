################################################################
#   filename: pglBehavior.py
#    purpose: Generic behavioral source extraction.
#         by: JLG
#       date: October 19, 2026
################################################################


import math
import re
from numbers import Integral, Real

import numpy as np
import xarray as xr

from .pglData import pglDataMatrix


class pglBehavior:
    """Behavioral extraction utilities; no database or payload publication."""

    # Development contract: do not treat this as the final behavioral schema.
    sourceContract = "pgl.behavior.source.v0"

    eventFieldNames = {
        "type": "eventType",
        "eventType": "eventSubtype",
        "timestamp": "eventTime",
        "trialNum": "eventSourceTrialNum",
        "segmentNum": "eventSegmentNum",
        "response": "eventResponse",
        "responseType": "eventResponseType",
        "triggerName": "eventTriggerName",
        "value": "eventTriggerValue",
        "triggerSent": "eventTriggerSent",
        "triggerWord": "eventTriggerWord",
    }

    @staticmethod
    def _fieldName(prefix, sourceName):
        """Produce camelCaps while retaining a separate original-name map."""
        if not isinstance(sourceName, str) or not sourceName:
            raise TypeError("Recorded field names must be nonempty strings.")

        parts = re.findall(r"[A-Za-z0-9]+", sourceName)

        if not parts:
            raise ValueError(f"Cannot generate a column name for {sourceName!r}.")

        suffix = "".join(part[0].upper() + part[1:] for part in parts)
        return prefix + suffix

    @staticmethod
    def _eventFields(event):
        """Read public stored fields without calling event methods.

        Trait fields marked serialize=False and private runtime fields are
        excluded. Unregistered public fields are included.
        """
        values = {name: value for name, value in vars(event).items() if not name.startswith("_")}

        traitsMethod = getattr(event, "traits", None)

        if callable(traitsMethod):
            for name, trait in traitsMethod().items():
                if name.startswith("_") or trait.metadata.get("serialize", True) is False:
                    values.pop(name, None)
                    continue

                values[name] = getattr(event, name)

        return values

    @staticmethod
    def _scalar(value, context):
        """Normalize supported scientific scalars without stringifying objects."""
        if value is None:
            return None

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        if isinstance(value, Integral):
            value = int(value)

            if not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
                raise ValueError(f"{context}: integer is outside signed int64 range.")

            return value

        if isinstance(value, Real):
            return float(value)

        if isinstance(value, (str, np.str_)):
            return str(value)

        raise TypeError(f"{context}: unsupported scientific value of type {type(value).__name__}. Expected a numeric, Boolean, or string scalar, or None.")

    @classmethod
    def _encodeColumn(cls, values, context):
        """Encode one scalar column and return arrays plus its logical type.

        Missing integer/Boolean/string entries use a fill value and a separate
        presence mask. A recorded floating NaN remains present.

        Mixed Boolean/numeric and string/numeric fields are rejected.
        """
        values = [cls._scalar(value, f"{context}, row {index}") for index, value in enumerate(values)]
        present = np.asarray([value is not None for value in values], dtype=bool)
        available = [value for value in values if value is not None]

        if not available:
            return np.full(len(values), np.nan, dtype=np.float64), present, "unknown"

        kinds = set()

        for value in available:
            if isinstance(value, bool):
                kinds.add("boolean")
            elif isinstance(value, int):
                kinds.add("integer")
            elif isinstance(value, float):
                kinds.add("float")
            else:
                kinds.add("string")

        if kinds == {"boolean"}:
            array = np.asarray([False if value is None else value for value in values], dtype=bool)
            logicalType = "boolean"

        elif kinds == {"integer"}:
            array = np.asarray([0 if value is None else value for value in values], dtype=np.int64)
            logicalType = "integer"

        elif kinds <= {"integer", "float"}:
            # Do not silently round large recorded integers during promotion.
            for value in available:
                if isinstance(value, int) and int(float(value)) != value:
                    raise ValueError(f"{context}: integer {value} cannot be represented exactly in a float64 column.")

            array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
            logicalType = "float"

        elif kinds == {"string"}:
            width = max(1, max(len(value) for value in available))
            array = np.asarray(["" if value is None else value for value in values], dtype=f"<U{width}")
            logicalType = "string"

        else:
            raise TypeError(f"{context}: incompatible scalar types {sorted(kinds)}.")

        return array, present, logicalType

    @classmethod
    def _addColumns(cls, dataset, dimension, records, fieldNames, *, sourceGroup):
        """Extract scalar, vector, and explicitly adapted structured fields.

        pglTraitSettings objects may implement extractAsDataFrameColumn():
            dict: named scalar components.
            "unsupported": no supported analysis representation.

        Unsupported value representations produce an explicitly marked string
        column. Missingness remains separate from the unsupported marker.

        Structural naming errors and exceptions inside custom adapters raise.
        All variables are prepared before they are added to the Dataset.
        """
        from .pglSettings import pglTraitSettings
        from .pglMessages import pglMessages

        prepared = {}
        coordinates = {}
        unsupportedReports = []

        # Reserve generated names, including masks which might not be needed
        # for this particular set of records.
        reserved = set(dataset.variables) | set(dataset.dims)

        def reserve(names, context):
            if len(names) != len(set(names)):
                raise ValueError(f"{context}: generated column names collide.")

            collisions = reserved.intersection(names)

            if collisions:
                raise ValueError(f"{context}: column-name collision: {sorted(collisions)}.")

            reserved.update(names)

        def putVariable(name, array, dimensions, present, attributes):
            attributes = dict(attributes)

            if not present.all():
                maskName = name + "Present"
                attributes["presenceVariable"] = maskName
                prepared[maskName] = xr.DataArray(present, dims=(dimension,), attrs={"role": "presenceMask", "forVariable": name})

            prepared[name] = xr.DataArray(array, dims=dimensions, attrs=attributes)

        def putUnsupported(columnName, values, attributes, reason):
            present = np.asarray([value is not None for value in values], dtype=bool)
            array = np.asarray(["unsupported" if isPresent else "" for isPresent in present], dtype="<U11")
            sourceTypes = sorted({f"{type(value).__module__}.{type(value).__qualname__}" for value in values if value is not None})

            attributes = {
                **attributes,
                "logicalType": "string",
                "extractionStatus": "unsupported",
                "sourceTypes": sourceTypes,
                "unsupportedReason": reason,
            }

            putVariable(columnName, array, (dimension,), present, attributes)
            unsupportedReports.append(f"{columnName}: {reason}")

        names = list(fieldNames.values())

        if len(names) != len(set(names)):
            raise ValueError(f"{sourceGroup}: multiple source fields map to the same column name.")

        # Reserve all top-level fields before considering structured components.
        for columnName in names:
            reserve([columnName, columnName + "Present", columnName + "Component"], sourceGroup)

        for sourceName, columnName in fieldNames.items():
            values = [record.get(sourceName) for record in records]
            available = [value for value in values if value is not None]
            context = f"{sourceGroup}.{sourceName}"

            attributes = {
                "sourceGroup": sourceGroup,
                "sourceField": sourceName,
                "extractionStatus": "supported",
            }

            hasStructuredObject = any(isinstance(value, pglTraitSettings) for value in available)
            hasVector = any(isinstance(value, (tuple, list, np.ndarray)) for value in available)

            # --------------------------------------------------------------
            # Structured objects: use an explicit analysis adapter.
            # --------------------------------------------------------------
            if hasStructuredObject:
                if not all(isinstance(value, pglTraitSettings) for value in available):
                    putUnsupported(columnName, values, attributes, "Mixed structured-object and other value representations.")
                    continue

                projected = []
                componentNames = {}
                unsupportedReason = None

                for rowIndex, value in enumerate(values):
                    if value is None:
                        projected.append(None)
                        continue

                    adapter = getattr(value, "extractAsDataFrameColumn", None)

                    if not callable(adapter):
                        result = "unsupported"
                    else:
                        # Adapter bugs must remain visible rather than silently
                        # replacing data with an unsupported marker.
                        try:
                            result = adapter()
                        except Exception as error:
                            raise RuntimeError(f"{context}, row {rowIndex}: extractAsDataFrameColumn() failed.") from error

                    if isinstance(result, str) and result == "unsupported":
                        unsupportedReason = "At least one object has no supported analysis adapter."
                        break

                    if not isinstance(result, dict) or not result:
                        raise TypeError(f"{context}, row {rowIndex}: adapter must return a nonempty dictionary or 'unsupported'.")

                    for componentName in result:
                        if not isinstance(componentName, str) or not componentName:
                            raise TypeError(f"{context}, row {rowIndex}: adapter component names must be nonempty strings.")

                        componentNames.setdefault(componentName, None)

                    projected.append(dict(result))

                if unsupportedReason is not None:
                    putUnsupported(columnName, values, attributes, unsupportedReason)
                    continue

                componentMapping = {componentName: cls._fieldName(columnName, componentName) for componentName in componentNames}
                generatedNames = []

                for componentColumn in componentMapping.values():
                    generatedNames.extend((componentColumn, componentColumn + "Present"))

                # Naming ambiguity is structural, not an unsupported value.
                reserve(generatedNames, context)

                encodedComponents = {}

                try:
                    for componentName, componentColumn in componentMapping.items():
                        componentValues = [None if record is None else record.get(componentName) for record in projected]
                        encodedComponents[componentName] = cls._encodeColumn(componentValues, f"{context}.{componentName}")
                except (TypeError, ValueError) as error:
                    putUnsupported(columnName, values, attributes, str(error))
                    continue

                # Always retain parent presence for structured objects, even
                # when every current row contains an object.
                parentMaskName = columnName + "Present"
                parentPresent = np.asarray([value is not None for value in values], dtype=bool)

                prepared[parentMaskName] = xr.DataArray(
                    parentPresent,
                    dims=(dimension,),
                    attrs={
                        **attributes,
                        "role": "structuredPresence",
                        "forField": columnName,
                        "componentFields": componentMapping,
                    },
                )

                for componentName, componentColumn in componentMapping.items():
                    array, present, logicalType = encodedComponents[componentName]
                    componentAttributes = {
                        **attributes,
                        "logicalType": logicalType,
                        "sourceComponent": componentName,
                        "structuredField": columnName,
                        "parentPresenceVariable": parentMaskName,
                    }
                    putVariable(componentColumn, array, (dimension,), present, componentAttributes)

                continue

            # --------------------------------------------------------------
            # Fixed-length numeric vectors.
            # --------------------------------------------------------------
            if hasVector:
                try:
                    vectorLength = None
                    normalized = []

                    for rowIndex, value in enumerate(values):
                        if value is None:
                            normalized.append(None)
                            continue

                        if not isinstance(value, (tuple, list, np.ndarray)):
                            raise TypeError(f"{context}, row {rowIndex}: mixed scalar and vector values.")

                        if isinstance(value, np.ndarray) and value.ndim != 1:
                            raise ValueError(f"{context}, row {rowIndex}: expected a one-dimensional vector.")

                        components = list(value)

                        if not components:
                            raise ValueError(f"{context}, row {rowIndex}: empty vectors are not supported.")

                        if vectorLength is None:
                            vectorLength = len(components)
                        elif len(components) != vectorLength:
                            raise ValueError(f"{context}, row {rowIndex}: inconsistent vector length.")

                        normalizedComponents = []

                        for componentIndex, component in enumerate(components):
                            component = cls._scalar(component, f"{context}, row {rowIndex}, component {componentIndex}")

                            if component is None or isinstance(component, str):
                                raise TypeError(f"{context}: vector components must be numeric or Boolean.")

                            normalizedComponents.append(component)

                        normalized.append(normalizedComponents)

                    flattened = []

                    for value in normalized:
                        flattened.extend([None] * vectorLength if value is None else value)

                    flatArray, _, elementType = cls._encodeColumn(flattened, context)
                    array = flatArray.reshape(len(values), vectorLength)
                    present = np.asarray([value is not None for value in normalized], dtype=bool)

                except (TypeError, ValueError) as error:
                    putUnsupported(columnName, values, attributes, str(error))
                    continue

                componentDimension = columnName + "Component"
                coordinates[componentDimension] = xr.DataArray(
                    np.arange(vectorLength, dtype=np.int64),
                    dims=(componentDimension,),
                    attrs={"meaning": "Zero-based position in the recorded vector"},
                )

                vectorAttributes = {
                    **attributes,
                    "logicalType": "vector",
                    "elementLogicalType": elementType,
                    "componentDimension": componentDimension,
                    "sourceRepresentation": "Fixed-length numeric vector; Python container type not retained",
                }

                putVariable(columnName, array, (dimension, componentDimension), present, vectorAttributes)
                continue

            # --------------------------------------------------------------
            # Scalars, including a nonfatal fallback for unsupported objects.
            # --------------------------------------------------------------
            try:
                array, present, logicalType = cls._encodeColumn(values, context)
            except (TypeError, ValueError) as error:
                putUnsupported(columnName, values, attributes, str(error))
                continue

            putVariable(columnName, array, (dimension,), present, {**attributes, "logicalType": logicalType})

        for coordinateName, coordinate in coordinates.items():
            dataset.coords[coordinateName] = coordinate

        for variableName, variable in prepared.items():
            dataset[variableName] = variable

        if unsupportedReports:
            details = "; ".join(unsupportedReports[:8])

            if len(unsupportedReports) > 8:
                details += f"; plus {len(unsupportedReports) - 8} additional field(s)"

            pglMessages.warning(f"{sourceGroup}: {len(unsupportedReports)} field(s) marked 'unsupported'. {details}",level=1)

    @staticmethod
    def _dictionaryFields(records, context):
        """Collect field names in deterministic first-appearance order."""
        names = {}

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"{context}, row {index}: expected a dictionary.")

            for name in record:
                if not isinstance(name, str) or not name:
                    raise TypeError(f"{context}, row {index}: field names must be nonempty strings.")

                names.setdefault(name, None)

        return list(names)

    @staticmethod
    def _label(value, context):
        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"{context}: expected a string or None.")

        return value

    @staticmethod
    def _trialNumber(value, context):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{context}: trial number must be an integer.")

        return int(value)

    @staticmethod
    def _trialTime(value, context):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{context}: trial START requires a finite numeric timestamp.")

        return float(value)

    @classmethod
    def extractSession(cls, session, *, name="behavior"):
        """Extract current session contents into an in-memory pglDataMatrix.

        Reads all runs/tasks and their behavioral data. Does not access MNE,
        save the session, publish a matrix, or claim checkpoint equivalence.

        Initial strict contract:
            One sequentially numbered START per params/trialVariables entry.
            Recorded scientific fields must contain supported scalar values.

        Relevant settings/configuration export is intentionally not part of
        this first implementation.
        """
        trialRecords = []
        parameterRecords = []
        variableRecords = []
        eventRecords = []
        eventContexts = []

        runMetadata = []
        taskMetadata = []

        def appendEvent(fields, *, runIndex, taskIndex, trialIndex, sourceIndex, scope, association):
            eventContexts.append({
                "eventRunIndex": runIndex,
                "eventTaskIndex": taskIndex,
                "eventTrialIndex": trialIndex,
                "eventSourceIndex": sourceIndex,
                "eventScope": scope,
                "eventAssociation": association,
            })
            eventRecords.append(fields)

        for runIndex, run in enumerate(session.runs or []):
            experimentSettings = run.experimentSettings
            runData = run.data

            if experimentSettings is None or runData is None:
                raise ValueError(f"Run {runIndex}: missing experimentSettings or data.")

            labels = {
                field: cls._label(getattr(experimentSettings, field, None), f"Run {runIndex}.{field}")
                for field in ("subjectID", "sessionName", "runName", "experimentName")
            }

            runMetadata.append({"runIndex": runIndex, **labels})

            # Run-level events are preserved without guessing trial association.
            for sourceIndex, event in enumerate(runData.events):
                fields = cls._eventFields(event)
                appendEvent(fields, runIndex=runIndex, taskIndex=-1, trialIndex=-1, sourceIndex=sourceIndex, scope="run", association="runUnassigned")

            # Respect the authoritative working task list when already loaded.
            if run.lazyIsPending("tasks"):
                taskCount = len(experimentSettings.tasks)
            else:
                taskValues = run._trait_values.get("tasks")

                if taskValues is None:
                    raise ValueError(f"Run {runIndex}: tasks are unavailable.")

                taskCount = len(taskValues)

            for taskIndex in range(taskCount):
                task = run.getTaskAt(taskIndex)

                if task is None or task.settings is None or task.data is None:
                    raise ValueError(f"Run {runIndex}, task {taskIndex}: missing task settings or data.")

                settings = task.settings
                data = task.data
                context = f"Run {runIndex}, task {taskIndex}"

                taskLabels = {
                    field: cls._label(getattr(settings, field, None), f"{context}.{field}")
                    for field in ("taskName", "taskSaveName")
                }

                taskMetadata.append({"runIndex": runIndex, "taskIndex": taskIndex, **taskLabels})

                # Copy dictionaries so the result does not alias source records.
                params = list(data.params)
                variables = list(data.trialVariables)
                cls._dictionaryFields(params, f"{context}.params")
                cls._dictionaryFields(variables, f"{context}.trialVariables")

                fieldsList = [cls._eventFields(event) for event in data.events]
                starts = [fields for fields in fieldsList if fields.get("type") == "trial" and fields.get("eventType") == "start"]

                if len(starts) != len(params) or len(starts) != len(variables):
                    raise ValueError(f"{context}: {len(starts)} trial STARTs, {len(params)} parameter records, and {len(variables)} trial-variable records do not align.")

                activeTrialIndex = -1
                activeTrialNum = None
                trialPosition = 0

                for sourceIndex, fields in enumerate(fieldsList):
                    isTrial = fields.get("type") == "trial"
                    boundary = fields.get("eventType")
                    association = "taskContext" if activeTrialIndex >= 0 else "outsideTrial"

                    if isTrial and boundary == "start":
                        trialNum = cls._trialNumber(fields.get("trialNum"), f"{context}, event {sourceIndex}")
                        trialTime = cls._trialTime(fields.get("timestamp"), f"{context}, event {sourceIndex}")

                        # Current recorder numbers STARTs consecutively from zero.
                        # Refuse ambiguous parameter joins rather than guess.
                        if trialNum != trialPosition:
                            raise ValueError(f"{context}: START trialNum {trialNum} does not match parameter position {trialPosition}.")

                        activeTrialIndex = len(trialRecords)
                        activeTrialNum = trialNum
                        association = "trialStart"

                        trialRecords.append({
                            **labels,
                            **taskLabels,
                            "runIndex": runIndex,
                            "taskIndex": taskIndex,
                            "sourceTrialNum": trialNum,
                            "trialStartTime": trialTime,
                            "trialStartEventIndex": len(eventRecords),
                        })

                        parameterRecords.append(dict(params[trialPosition]))
                        variableRecords.append(dict(variables[trialPosition]))
                        trialPosition += 1

                    elif isTrial and boundary == "end":
                        if activeTrialIndex >= 0:
                            trialNum = cls._trialNumber(fields.get("trialNum"), f"{context}, event {sourceIndex}")

                            if trialNum != activeTrialNum:
                                raise ValueError(f"{context}: END trialNum {trialNum} does not match active trial {activeTrialNum}.")

                            association = "trialEnd"
                        else:
                            association = "outsideTrial"

                    appendEvent(fields, runIndex=runIndex, taskIndex=taskIndex, trialIndex=activeTrialIndex, sourceIndex=sourceIndex, scope="task", association=association)

                    # Preserve events after termination without assigning them
                    # silently to the preceding trial.
                    if isTrial and boundary == "end":
                        activeTrialIndex = -1
                        activeTrialNum = None

        dataset = xr.Dataset(coords={"trial": np.arange(len(trialRecords), dtype=np.int64), "event": np.arange(len(eventRecords), dtype=np.int64)})

        trialFields = [
            "subjectID", "sessionName", "runName", "experimentName",
            "taskName", "taskSaveName", "runIndex", "taskIndex",
            "sourceTrialNum", "trialStartTime", "trialStartEventIndex",
        ]
        contextFields = [
            "eventRunIndex", "eventTaskIndex", "eventTrialIndex",
            "eventSourceIndex", "eventScope", "eventAssociation",
        ]

        cls._addColumns(dataset, "trial", trialRecords, {field: field for field in trialFields}, sourceGroup="trialContext")
        cls._addColumns(dataset, "event", eventContexts, {field: field for field in contextFields}, sourceGroup="eventContext")

        parameterNames = cls._dictionaryFields(parameterRecords, "params")
        variableNames = cls._dictionaryFields(variableRecords, "trialVariables")
        eventNames = cls._dictionaryFields(eventRecords, "events")

        parameterMapping = {field: cls._fieldName("parameter", field) for field in parameterNames}
        variableMapping = {field: cls._fieldName("variable", field) for field in variableNames}

        # Include familiar fields even when absent, so the event schema is
        # inspectable on sessions without responses or triggers.
        eventMapping = dict(cls.eventFieldNames)

        for field in eventNames:
            if field not in eventMapping:
                eventMapping[field] = cls._fieldName("event", field)

        cls._addColumns(dataset, "trial", parameterRecords, parameterMapping, sourceGroup="params")
        cls._addColumns(dataset, "trial", variableRecords, variableMapping, sourceGroup="trialVariables")
        cls._addColumns(dataset, "event", eventRecords, eventMapping, sourceGroup="events")

        for field in ("trialStartTime", "eventTime"):
            dataset[field].attrs.update({"units": "seconds", "clockScope": "source run acquisition clock", "reference": "original recorded timestamp"})

        dataset["eventTrialIndex"].attrs.update({"referencesDimension": "trial", "missingRelationship": -1})
        dataset["trialStartEventIndex"].attrs.update({"referencesDimension": "event"})
        dataset["eventTaskIndex"].attrs.update({"indexScope": "within source run", "missingRelationship": -1})
        dataset["taskIndex"].attrs["indexScope"] = "within source run"

        dataset.attrs.update({
            "behaviorContract": cls.sourceContract,
            "runSources": runMetadata,
            "taskSources": taskMetadata,
            "parameterFieldNames": parameterMapping,
            "trialVariableFieldNames": variableMapping,
            "eventFieldNames": eventMapping,
            "configurationIncluded": False,
            "eventOrdering": "run order; run events first; then task order and recorded task-event order",
            "trialOrdering": "run order; task order; recorded trial START order",
        })

        matrix = pglDataMatrix(dataset, name=name)

        checkpointID = getattr(session, "sessionID", None)

        matrix.provenance = {
            "construction": "pglBehavior.extractSession",
            "behaviorContract": cls.sourceContract,
            "sourceState": "workingSession",
            "sourceCheckpointID": None if checkpointID is None else int(checkpointID),
            "checkpointEquivalenceVerified": False,
        }

        return matrix
    
    @classmethod
    def _requireBehaviorData(cls, matrix):
        """Validate the behavioral container without reading scientific arrays."""
        data = matrix.requireData()

        if not isinstance(data, xr.Dataset):
            raise TypeError("Behavioral access requires an xarray.Dataset.")

        if data.attrs.get("behaviorContract") != cls.sourceContract:
            raise ValueError(f"Expected behavioral contract {cls.sourceContract!r}.")

        requiredFields = {
            "runIndex", "taskIndex", "trialStartTime", "trialStartEventIndex",
            "eventType", "eventTime", "eventTrialIndex", "eventScope",
            "eventResponse", "eventResponseType",
        }

        missingFields = requiredFields.difference(data.variables)

        if missingFields or "trial" not in data.dims or "event" not in data.dims:
            raise ValueError(f"Incomplete behavioral source schema; missing fields: {sorted(missingFields)}.")

        return data

    @staticmethod
    def _behaviorValues(data, variableName):
        """Read one scalar field, restoring None from its presence mask.

        Recorded floating NaNs remain floating NaNs, not absent values.
        Unsupported markers remain explicitly marked values.
        """
        variable = data[variableName]

        if variable.ndim != 1:
            raise ValueError(f"{variableName}: expected a one-dimensional field.")

        values = variable.values.tolist()
        maskName = variable.attrs.get("presenceVariable")

        if maskName is not None:
            mask = data[maskName]

            if mask.dims != variable.dims or mask.dtype.kind != "b":
                raise ValueError(f"{variableName}: invalid presence mask {maskName!r}.")

            values = [value if present else None for value, present in zip(values, mask.values.tolist())]

        return values

    @classmethod
    def _selectBehaviorTrials(cls, data, *, taskName=None, taskIndex=None, taskSelection=None):
        """Resolve tasks independently per run and return trial positions.

        taskName matches taskSaveName exactly.
        Runs missing a requested task raise rather than disappearing silently.
        Without explicit selection, only single-task runs resolve automatically.
        """
        if taskSelection not in (None, "all"):
            raise ValueError("taskSelection must be None or 'all'.")

        selectionCount = sum(value is not None for value in (taskName, taskIndex, taskSelection))

        if selectionCount > 1:
            raise ValueError("Specify only one of taskName, taskIndex, or taskSelection.")

        if taskName is not None and (not isinstance(taskName, str) or not taskName):
            raise TypeError("taskName must be a nonempty taskSaveName string.")

        if taskIndex is not None:
            if isinstance(taskIndex, (bool, np.bool_)) or not isinstance(taskIndex, Integral) or taskIndex < 0:
                raise ValueError("taskIndex must be a nonnegative integer.")

        runSources = data.attrs.get("runSources")
        taskSources = data.attrs.get("taskSources")

        if not isinstance(runSources, list) or not isinstance(taskSources, list):
            raise ValueError("Behavioral source metadata must include runSources and taskSources.")

        selectedTasks = set()

        for runSource in runSources:
            runIndex = runSource["runIndex"]
            candidates = [source for source in taskSources if source["runIndex"] == runIndex]

            if taskSelection == "all":
                matches = candidates
            elif taskName is not None:
                matches = [source for source in candidates if source.get("taskSaveName") == taskName]
            elif taskIndex is not None:
                matches = [source for source in candidates if source["taskIndex"] == taskIndex]
            else:
                matches = candidates

            if taskSelection != "all" and len(matches) != 1:
                available = [(source["taskIndex"], source.get("taskSaveName")) for source in candidates]
                raise ValueError(f"Run {runIndex}: selection resolved to {len(matches)} tasks; specify an unambiguous selection. Available tasks: {available}.")

            for source in matches:
                selectedTasks.add((source["runIndex"], source["taskIndex"]))

        runIndices = cls._behaviorValues(data, "runIndex")
        taskIndices = cls._behaviorValues(data, "taskIndex")
        positions = [position for position, identity in enumerate(zip(runIndices, taskIndices)) if identity in selectedTasks]

        return np.asarray(positions, dtype=np.int64)

    @staticmethod
    def _behaviorTrialLatency(responseTime, trialStartTime):
        """Calculate seconds relative to trial start without clipping negatives.

        Returns the latency and an explicit diagnostic status.
        This helper is the single timing implementation for this increment.
        """
        if responseTime is None:
            return None, "missingResponseTimestamp"

        if trialStartTime is None:
            return None, "missingTrialStartTimestamp"

        if not math.isfinite(float(responseTime)) or not math.isfinite(float(trialStartTime)):
            return None, "nonfiniteTimestamp"

        latency = float(responseTime) - float(trialStartTime)
        return latency, "negativeLatency" if latency < 0 else "ok"

    @classmethod
    def getTrialData(cls, matrix, *, taskName=None, taskIndex=None, taskSelection=None, responsePolicy="first", responseReference="trial", includeAllResponses=False, includeTriggers=True, includeConfiguration=False):
        """Return an independent trial-only xarray projection.

        Reads matrix arrays as needed. Does not mutate the source matrix or
        publish records/payloads.

        Initial timing support is trial-relative only. Trigger expansion and
        configuration projection are explicitly deferred.
        """
        from copy import deepcopy

        data = cls._requireBehaviorData(matrix)

        if responsePolicy not in ("first", "last"):
            raise ValueError("responsePolicy must be 'first' or 'last'.")

        if responseReference != "trial":
            raise NotImplementedError("This increment supports responseReference='trial' only.")

        if includeConfiguration:
            raise NotImplementedError("Configuration projection is not implemented.")

        positions = cls._selectBehaviorTrials(data, taskName=taskName, taskIndex=taskIndex, taskSelection=taskSelection)

        # Retain all trial-aligned fields, including presence masks and vectors.
        # Never include the independent event dimension in the trial projection.
        trialFields = []

        for variableName, variable in data.data_vars.items():
            if "trial" in variable.dims:
                if "event" in variable.dims or variable.dims[0] != "trial":
                    raise ValueError(f"{variableName}: unsupported trial-field dimensions {variable.dims}.")

                trialFields.append(variableName)

        trialData = data[trialFields].isel(trial=positions).copy(deep=True)
        trialData.attrs = deepcopy(data.attrs)

        trialIDs = trialData.coords["trial"].values.tolist()

        if len(trialIDs) != len(set(trialIDs)):
            raise ValueError("Trial coordinates must identify unique source rows.")

        trialPositions = {trialID: position for position, trialID in enumerate(trialIDs)}
        trialStarts = cls._behaviorValues(trialData, "trialStartTime")

        eventTypes = cls._behaviorValues(data, "eventType")
        eventScopes = cls._behaviorValues(data, "eventScope")
        eventTrials = cls._behaviorValues(data, "eventTrialIndex")
        eventTimes = cls._behaviorValues(data, "eventTime")
        eventIDs = data.coords["event"].values.tolist()

        # Refuse to imply that requested trigger expansion has been performed.
        # This conservative guard also recognizes named triggers of unfamiliar
        # event types. Trigger semantics will receive their own implementation.
        if includeTriggers:
            triggerNames = cls._behaviorValues(data, "eventTriggerName") if "eventTriggerName" in data else [None] * len(eventTypes)
            hasTriggers = any(name is not None or "trigger" in str(eventType).lower() for name, eventType in zip(triggerNames, eventTypes))

            if hasTriggers:
                raise NotImplementedError("Trigger expansion is not implemented yet. Use includeTriggers=False for a response-only projection; all trigger records remain in the source matrix.")

        for variableName in ("eventResponse", "eventResponseType", "eventTime"):
            if data[variableName].attrs.get("extractionStatus") != "supported":
                raise ValueError(f"{variableName}: response projection requires a supported encoding.")

        responses = cls._behaviorValues(data, "eventResponse")
        responseTypes = cls._behaviorValues(data, "eventResponseType")
        responseGroups = [[] for _ in trialIDs]
        timingDiagnostics = []

        # Iteration preserves recorded event order. No timestamp sorting.
        for eventPosition, eventType in enumerate(eventTypes):
            if eventType != "subjectResponse" or eventScopes[eventPosition] != "task":
                continue

            trialPosition = trialPositions.get(eventTrials[eventPosition])

            if trialPosition is None:
                continue

            latency, status = cls._behaviorTrialLatency(eventTimes[eventPosition], trialStarts[trialPosition])

            responseGroups[trialPosition].append({
                "response": responses[eventPosition],
                "responseType": responseTypes[eventPosition],
                "responseTime": latency,
                "responseEventIndex": eventIDs[eventPosition],
            })

            if status != "ok":
                timingDiagnostics.append({
                    "trialIndex": trialIDs[trialPosition],
                    "eventIndex": eventIDs[eventPosition],
                    "status": status,
                })

        responseFields = ("response", "responseType", "responseTime", "responseEventIndex")
        maximumResponses = max((len(group) for group in responseGroups), default=0)
        projectedFields = ["responseCount", *responseFields]

        if includeAllResponses:
            for responseNumber in range(1, maximumResponses + 1):
                projectedFields.extend(f"{field}{responseNumber}" for field in responseFields)

        responseRecords = []

        for group in responseGroups:
            record = {"responseCount": len(group)}
            selectedResponse = group[0 if responsePolicy == "first" else -1] if group else {}

            for field in responseFields:
                record[field] = selectedResponse.get(field)

            if includeAllResponses:
                for responseNumber, responseRecord in enumerate(group, start=1):
                    for field in responseFields:
                        record[f"{field}{responseNumber}"] = responseRecord[field]

            responseRecords.append(record)

        # Encode summaries directly: incompatible values must raise, not become
        # unsupported markers silently introduced by the projection.
        reserved = set(trialData.variables) | set(trialData.dims)

        for field in projectedFields:
            generatedNames = {field, field + "Present"}
            collisions = reserved.intersection(generatedNames)

            if collisions:
                raise ValueError(f"Response projection column collision: {sorted(collisions)}.")

            reserved.update(generatedNames)
            values = [record.get(field) for record in responseRecords]
            array, present, logicalType = cls._encodeColumn(values, f"responseProjection.{field}")
            attributes = {"sourceGroup": "responseProjection", "logicalType": logicalType, "extractionStatus": "supported"}

            if not present.all():
                maskName = field + "Present"
                trialData[maskName] = xr.DataArray(present, dims=("trial",), attrs={"role": "presenceMask", "forVariable": field})
                attributes["presenceVariable"] = maskName

            if field.startswith("responseTime"):
                attributes.update({"units": "seconds", "reference": "trialStartTime"})

            if field.startswith("responseEventIndex"):
                attributes.update({"referencesDimension": "event", "referenceScope": "sourceMatrix"})

            trialData[field] = xr.DataArray(array, dims=("trial",), attrs=attributes)

        # These indices still identify events in the comprehensive source,
        # not rows in a nonexistent event dimension of this trial-only result.
        trialData["trialStartEventIndex"].attrs["referenceScope"] = "sourceMatrix"

        trialData.attrs.update({
            "behaviorView": "trial",
            "viewSpecification": {
                "taskName": taskName,
                "taskIndex": None if taskIndex is None else int(taskIndex),
                "taskSelection": taskSelection,
                "responsePolicy": responsePolicy,
                "responseReference": responseReference,
                "responseUnits": "seconds",
                "includeAllResponses": bool(includeAllResponses),
                "includeTriggers": bool(includeTriggers),
                "includeConfiguration": False,
                "inferenceRules": "none",
            },
            "responseTimingDiagnostics": timingDiagnostics,
            "relationshipIndexScope": "Original comprehensive source matrix; trial coordinates retained",
            "triggerProjectionStatus": "noTriggerRecords" if includeTriggers else "excludedByView",
        })

        return trialData

    @classmethod
    def getBehaviorTable(cls, matrix, **options):
        """Convert the shared trial projection to an independent pandas table.

        Materializes selected trial arrays in memory. Masks become pandas
        missing values; vectors become numbered component columns.

        Column order is explicit and independent of xarray/Zarr variable
        enumeration order. Does not modify source arrays or publish anything.
        """
        from copy import deepcopy

        import pandas as pd

        trialData = cls.getTrialData(matrix, **options)
        columns = {}
        columnMetadata = {}

        # Reserve scalar names before generating vector-component names.
        scalarNames = {
            name for name, variable in trialData.data_vars.items()
            if variable.dims == ("trial",) and variable.attrs.get("role") not in ("presenceMask", "structuredPresence")
        }

        def addColumn(columnName, values, present, logicalType, metadata):
            if columnName in columns:
                raise ValueError(f"Pandas projection column collision: {columnName!r}.")

            values = np.array(values, copy=True)
            missing = ~present

            if logicalType == "integer":
                result = pd.array(values, dtype="Int64")
                result[missing] = pd.NA
            elif logicalType == "boolean":
                result = pd.array(values, dtype="boolean")
                result[missing] = pd.NA
            elif logicalType == "string":
                result = pd.array(values, dtype="string")
                result[missing] = pd.NA
            elif logicalType in ("float", "unknown"):
                result = values.astype(np.float64, copy=True)
                result[missing] = np.nan
            else:
                raise TypeError(f"{columnName}: unsupported logical type {logicalType!r}.")

            columns[columnName] = result
            columnMetadata[columnName] = deepcopy(metadata)

        for variableName, variable in trialData.data_vars.items():
            if variable.attrs.get("role") in ("presenceMask", "structuredPresence"):
                continue

            maskName = variable.attrs.get("presenceVariable")
            present = np.ones(trialData.sizes["trial"], dtype=bool) if maskName is None else trialData[maskName].values.astype(bool, copy=True)
            logicalType = variable.attrs.get("logicalType")
            metadata = {**variable.attrs, "sourceVariable": variableName}

            if variable.dims == ("trial",):
                addColumn(variableName, variable.values, present, logicalType, metadata)
                continue

            componentDimension = variable.attrs.get("componentDimension")

            if logicalType != "vector" or variable.dims != ("trial", componentDimension):
                raise ValueError(f"{variableName}: no pandas encoding for dimensions {variable.dims}.")

            values = variable.values

            for componentIndex in range(variable.sizes[componentDimension]):
                columnName = f"{variableName}Component{componentIndex}"

                if columnName in scalarNames:
                    raise ValueError(f"Vector expansion collides with scalar field {columnName!r}.")

                componentMetadata = {
                    **metadata,
                    "logicalType": variable.attrs["elementLogicalType"],
                    "sourceComponentIndex": componentIndex,
                }

                addColumn(columnName, values[:, componentIndex], present, variable.attrs["elementLogicalType"], componentMetadata)

        contextFields = [
            "subjectID", "sessionName", "runName", "experimentName",
            "taskName", "taskSaveName", "runIndex", "taskIndex",
            "sourceTrialNum", "trialStartTime", "trialStartEventIndex",
        ]
        contextOrder = {name: index for index, name in enumerate(contextFields)}
        responseFields = ["response", "responseType", "responseTime", "responseEventIndex"]
        responseOrder = {name: index for index, name in enumerate(responseFields)}

        def columnOrder(columnName):
            metadata = columnMetadata[columnName]
            sourceGroup = metadata.get("sourceGroup")
            sourceVariable = metadata["sourceVariable"]
            componentIndex = metadata.get("sourceComponentIndex", -1)

            if columnName in contextOrder:
                return (0, contextOrder[columnName])

            if sourceGroup == "params":
                return (1, sourceVariable, componentIndex, columnName)

            if sourceGroup == "trialVariables":
                return (2, sourceVariable, componentIndex, columnName)

            if sourceGroup == "responseProjection":
                if columnName == "responseCount":
                    return (3, 0, 0)

                if columnName in responseOrder:
                    return (3, 1, responseOrder[columnName])

                match = re.fullmatch(r"(responseEventIndex|responseTime|responseType|response)([1-9][0-9]*)", columnName)

                if match is not None:
                    return (3, 2, int(match.group(2)), responseOrder[match.group(1)])

                return (3, 3, columnName)

            return (4, sourceVariable, componentIndex, columnName)

        orderedNames = sorted(columns, key=columnOrder)
        orderedColumns = {name: columns[name] for name in orderedNames}

        table = pd.DataFrame(orderedColumns, index=pd.Index(trialData.coords["trial"].values.copy(), name="trial"))
        table.attrs = deepcopy(trialData.attrs)
        table.attrs["columnMetadata"] = {name: columnMetadata[name] for name in orderedNames}
        table.attrs["pandasEncoding"] = {
            "vectorColumns": "Zero-based ComponentN suffix; see columnMetadata",
            "missingValues": "Nullable integer/Boolean/string; NaN for floating columns",
            "floatingNaN": "Recorded NaN and absent floating values both display as NaN; consult xarray presence masks to distinguish",
            "columnOrdering": "Context; alphabetical parameters; alphabetical trial variables; response summaries and numerically ordered responses; alphabetical remaining fields",
        }

        return table
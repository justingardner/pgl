from __future__ import annotations

import inspect
import uuid

from dataclasses import dataclass, field
from functools import wraps
from typing import Any


# -------------------------------------------------------------------------
# pglData
# -------------------------------------------------------------------------
class pglData:
    """
    Marker/base class for data that can move through a pgl pipeline.

    All classes that represent pipeline artifacts should inherit from this.

    Examples:
        class pglSession(pglData):
            ...

        class pglRawData(pglData):
            ...

        class pglMotionEstimate(pglData):
            ...
    """

    # These fields are assigned automatically by pglActionRun when needed.
    dataId: str
    producerAction: pglAction | None
    producerActionId: str | None
    producerRunId: str | None
    producerReturnType: Any | None


# -------------------------------------------------------------------------
# pglActionRunRecord
# -------------------------------------------------------------------------
@dataclass
class pglActionRunRecord:
    """
    One actual invocation of an action.

    An action object can potentially run more than once.  Each invocation
    gets its own record, preserving the full dataflow history.
    """

    runId: str
    action: pglAction

    # Keys are the run() parameter names.
    # Values are the values supplied to those parameters.
    inputData: dict[str, Any] = field(default_factory=dict)

    # Flat list of pglData objects returned by the action.
    outputData: list[pglData] = field(default_factory=list)

    # The return annotation declared on run(), if any.
    returnType: Any | None = None

    error: Exception | None = None


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def iterPglData(value: Any):
    """
    Yield all pglData objects contained in value.

    Supports direct pglData values, lists, tuples, sets, and dictionaries.

    Examples:
        pglSession
        [pglSession, pglSession]
        {"rawData": pglRawData}
        (pglSession, pglMotionEstimate)
    """

    if isinstance(value, pglData):
        yield value

    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iterPglData(item)

    elif isinstance(value, dict):
        for item in value.values():
            yield from iterPglData(item)


def containsPglData(value: Any) -> bool:
    """
    Return True when value directly or recursively contains pglData.
    """
    return any(iterPglData(value))


def ensurePglDataMetadata(data: pglData) -> None:
    """
    Ensure a pglData object has standard provenance attributes.

    This is deliberately lazy.  Your individual pglData subclasses do not
    need custom initialization merely to support pipeline tracking.
    """

    if not hasattr(data, "dataId"):
        data.dataId = str(uuid.uuid4())

    if not hasattr(data, "producerAction"):
        data.producerAction = None

    if not hasattr(data, "producerActionId"):
        data.producerActionId = None

    if not hasattr(data, "producerRunId"):
        data.producerRunId = None

    if not hasattr(data, "producerReturnType"):
        data.producerReturnType = None


# -------------------------------------------------------------------------
# pglActionRun decorator
# -------------------------------------------------------------------------
def pglActionRun(runMethod):
    """
    Decorate a pglAction subclass's run() method.

    The subclass writes a normal run() method:

        class MotionCorrectAction(pglAction):

            @pglActionRun
            def run(
                self,
                session: pglSession,
                rawData: pglRawData,
            ) -> pglSession:
                return self.doMotionCorrection(session, rawData)

    The decorator automatically:

    1. Records pglData arguments using their declared parameter names.
    2. Calls the actual algorithm.
    3. Tags returned pglData objects with their producer action/run.
    4. Creates and stores a pglActionRunRecord.
    """

    signature = inspect.signature(runMethod)
    returnType = signature.return_annotation

    @wraps(runMethod)
    def wrapped(self: pglAction, *args, **kwargs):
        self.status = pglActionStatus.RUNNING
        self.error = None

        runRecord = pglActionRunRecord(
            runId=str(uuid.uuid4()),
            action=self,
            returnType=returnType,
        )

        # These expose the latest invocation in the same spirit as your
        # original inputData/outputData fields.
        self.inputData = runRecord.inputData
        self.outputData = runRecord.outputData

        self.runRecords.append(runRecord)

        try:
            # Match supplied positional/keyword arguments to the parameter
            # names declared in the child's run() signature.
            boundArguments = signature.bind(self, *args, **kwargs)
            boundArguments.apply_defaults()

            # Parameter names become the input names in the dataflow graph.
            #
            # For example:
            #
            #   def run(self, session, rawData):
            #
            # gives:
            #
            #   inputData["session"]
            #   inputData["rawData"]
            #
            for inputName, value in boundArguments.arguments.items():
                if inputName == "self":
                    continue

                if containsPglData(value):
                    runRecord.inputData[inputName] = value

                    for data in iterPglData(value):
                        ensurePglDataMetadata(data)

            # Run the actual subclass algorithm.
            result = runMethod(self, *args, **kwargs)

            # Find all pglData returned by the action.
            returnedData = list(iterPglData(result))

            # An output should generally be a new artifact, not one of the
            # same input objects mutated in-place.  Otherwise provenance
            # becomes ambiguous: was this object produced upstream or here?
            inputDataIds = {
                id(data)
                for inputValue in runRecord.inputData.values()
                for data in iterPglData(inputValue)
            }

            for data in returnedData:
                if id(data) in inputDataIds:
                    raise RuntimeError(
                        f"{self.name}.run() returned an object that was also "
                        "provided as an input. Pipeline provenance becomes "
                        "ambiguous for in-place mutation. Return a new "
                        "pglData artifact instead."
                    )

            # Avoid registering the same returned object more than once if,
            # for example, it appears twice in a nested return structure.
            seenDataIds = set()

            for data in returnedData:
                if id(data) in seenDataIds:
                    continue

                seenDataIds.add(id(data))
                ensurePglDataMetadata(data)

                data.producerAction = self
                data.producerActionId = self.actionId
                data.producerRunId = runRecord.runId
                data.producerReturnType = returnType

                runRecord.outputData.append(data)

            self.status = pglActionStatus.COMPLETED
            return result

        except Exception as error:
            self.error = error
            runRecord.error = error
            self.status = pglActionStatus.FAILED
            raise

    return wrapped


# -------------------------------------------------------------------------
# pglAction
# -------------------------------------------------------------------------
class pglAction(pglTraitSettings):
    """
    Base class for pipeline actions.

    Subclasses should define a normal decorated run() method:

        class MyAction(pglAction):

            @pglActionRun
            def run(self, session: pglSession) -> pglSession:
                ...
    """

    name = Unicode("", help="Name of action")

    status = Instance(
        pglActionStatus,
        help="Action status",
    )

    error = Instance(
        Exception,
        allow_none=True,
        default_value=None,
        help="Error raised while running this action",
    )

    settings = Instance(
        pglTraitSettings,
        allow_none=True,
        default_value=None,
        help="Settings for this action",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.actionId = str(uuid.uuid4())

        self.name = self.__class__.__name__
        self.status = pglActionStatus.INITIALIZED
        self.error = None

        self.version = "0.0"

        # Latest action invocation only.
        #
        # inputData:
        #     {
        #         "session": <pglSession>,
        #         "rawData": <pglRawData>,
        #     }
        #
        # outputData:
        #     [
        #         <pglSession>,
        #     ]
        self.inputData: dict[str, Any] = {}
        self.outputData: list[pglData] = []

        # Full invocation history.
        self.runRecords: list[pglActionRunRecord] = []

    def configure(self) -> None:
        """
        Configure this action.
        """
        self.status = pglActionStatus.CONFIGURED

    def print(self) -> None:
        """
        Print action summary.
        """
        print(
            f"Action: {self.name} "
            f"status: {self.status.name} "
            f"runs: {len(self.runRecords)}"
        )
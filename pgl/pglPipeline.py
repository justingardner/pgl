################################################################
#   filename: pglPipeline.py
#    purpose: Pipeline 
#         by: JLG
#       date: Aug 1, 2026
################################################################

#############
# Import
#############
from .pglActionProgress import actionProgress, operationProgress, currentProgress, progressCall, statusOutput, noProgressBar, daskProgress
from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from datetime import datetime
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from enum import Enum, auto
from pathlib import Path
from typing import Annotated
from .pglTimestamp import pglTimestamp
from datetime import datetime
import inspect
from copy import deepcopy
from contextvars import ContextVar
from uuid import uuid4
from .pglSerialize import pglSerialize

# Track the active execution, not the action instance. ContextVar keeps
# independent execution contexts separate and supports nested action calls.
_activeActionID = ContextVar("pglActiveActionID", default=None)

class pglActionConsoleCapture:
    """Tee Python console output to live streams and persistent local files.

    Files are flushed on every write. Original streams are never closed.

    Nested actions on the same thread are supported: child output also passes
    through the parent's tee. Concurrent action capture on different threads
    is rejected because Python stream redirection is process-global.

    Unrelated background-thread output may appear in the transcript.

    Does not intercept native file-descriptor writes, rich notebook display()
    output, or handlers holding previously cached stream references.
    External-command logging remains the responsibility of tool adapters.
    """

    from threading import RLock

    _captureLock = RLock()

    class _Tee:
        """A text stream forwarding output without owning the live stream."""

        def __init__(self, display, logfile):
            from threading import RLock

            self.display = display
            self.logfile = logfile
            self.captureErrors = []
            self._lock = RLock()

        def _recordError(self, error):
            message = f"{type(error).__name__}: {error}"
            if message not in self.captureErrors: self.captureErrors.append(message)

        def write(self, text):
            if not isinstance(text, str):
                raise TypeError("Action console capture expects text.")

            with self._lock:
                # Keep diagnostic files useful even if the display later fails.
                # Disk errors do not silently masquerade as complete capture.
                try:
                    written = self.logfile.write(text)
                    if written != len(text):
                        raise OSError("Incomplete action transcript write.")
                    self.logfile.flush()
                except Exception as error:
                    self._recordError(error)

                if self.display is not None:
                    self.display.write(text)
                    self.display.flush()

            return len(text)

        def flush(self):
            with self._lock:
                try:
                    self.logfile.flush()
                except Exception as error:
                    self._recordError(error)

                if self.display is not None:
                    self.display.flush()

        def isatty(self):
            return bool(self.display is not None and getattr(self.display, "isatty", lambda: False)())

        def writable(self):
            return True

        def fileno(self):
            from io import UnsupportedOperation

            # Exposing a descriptor would let callers bypass Python capture.
            raise UnsupportedOperation("Action console tee has no file descriptor.")

        @property
        def encoding(self):
            return getattr(self.display, "encoding", None) or "utf-8"

        @property
        def errors(self):
            return getattr(self.display, "errors", None) or "replace"

        @property
        def closed(self):
            return self.logfile.closed

    def __init__(self, cacheDirectory=None):
        self.cacheDirectory = cacheDirectory
        self.directory = None
        self.paths = {}
        self.streams = {}
        self._stack = None
        self._locked = False

    def __enter__(self):
        import os
        import sys
        import tempfile
        from contextlib import ExitStack, redirect_stderr, redirect_stdout
        from pathlib import Path

        if not self._captureLock.acquire(blocking=False):
            raise RuntimeError(
                "Another thread is running an action with console capture. "
                "Run actions serially or in separate processes."
            )
        self._locked = True
        stack = ExitStack()

        try:
            if self.cacheDirectory is None:
                root = Path.home() / ".cache" / "pgl" / "actionLogs"
            else:
                text = os.fspath(self.cacheDirectory)
                if (
                    not isinstance(text, str) or not text.strip()
                    or "://" in text or "::" in text
                    or any(character in text for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError("Action log cacheDirectory must be an ordinary local path.")
                root = Path(text).expanduser().absolute() / "actionLogs"

            root.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="python-", dir=root))
            self.paths = {
                "stdout": self.directory / "stdout.log",
                "stderr": self.directory / "stderr.log",
            }

            stdoutFile = stack.enter_context(self.paths["stdout"].open("x", encoding="utf-8"))
            stderrFile = stack.enter_context(self.paths["stderr"].open("x", encoding="utf-8"))

            self.streams = {
                "stdout": self._Tee(sys.stdout, stdoutFile),
                "stderr": self._Tee(sys.stderr, stderrFile),
            }

            # ExitStack restores streams before closing their transcript files.
            stack.enter_context(redirect_stdout(self.streams["stdout"]))
            stack.enter_context(redirect_stderr(self.streams["stderr"]))

            self._stack = stack
            return self

        except BaseException:
            try:
                stack.close()
            finally:
                self._captureLock.release()
                self._locked = False
            raise

    def __exit__(self, exceptionType, exceptionValue, traceback):
        try:
            for stream in self.streams.values():
                try:
                    stream.flush()
                except Exception as error:
                    stream._recordError(error)

            try:
                self._stack.close()
            except Exception as error:
                # Preserve the original scientific exception/outcome. Stream
                # restoration callbacks still run through ExitStack.
                for stream in self.streams.values():
                    stream._recordError(error)
        finally:
            if self._locked:
                self._captureLock.release()
                self._locked = False

        return False

    @property
    def captureErrors(self):
        return {
            name: list(stream.captureErrors)
            for name, stream in self.streams.items()
        }

########################
# action status
########################
class pglActionStatus(Enum):
    INITIALIZED = auto()
    VALIDATED = auto()
    CONFIGURED = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    
class pglActionHistory(pglTraitSettings):
    """Record one action attempt, its configuration, source, and execution identity.

    Configuration and source are captured before scientific execution.
    update() changes only execution outcome and timing.

    Legacy entries retain absent identity/source fields when reserialized.
    """

    actionID = Unicode(default_value=None, allow_none=True, help="UUID identifying this execution attempt")
    parentActionID = Unicode(default_value=None, allow_none=True, help="UUID of the containing action execution")
    actionName = Unicode(help="Name of action")
    actionVersion = Unicode(help="Version of action")
    actionStatus = Instance(pglActionStatus, help="Status of action")
    actionError = Instance(Exception, allow_none=True, default_value=None, help="Error raised while running this action")
    actionSettings = Dict(default_value={}, help="Snapshot of the action's configuration traits")
    actionCode = Dict(default_value={}, help="Versioned action class identity and inspect source snapshot; empty for legacy history")
    runDateTime = Unicode(help="Date and time when the action was run")
    startTime = Float(help="Start time of the action in seconds")
    endTime = Float(help="End time of the action in seconds")
    runDuration = Float(help="Time taken to run the action in seconds")
    actionInput = Dict(default_value={}, help="Explicit input provenance when recorded; empty for legacy or unspecified inputs")
    actionEnvironment = Dict(default_value={}, help="Runtime environment captured for this execution; empty when not recorded")
    actionInvocations = List(Dict(), help="External-tool invocation attempts recorded during this execution")
    actionLogs = Dict(default_value={}, help="Archived command logs indexed by invocation ID and stdout/stderr")
    actionApplication = Dict(default_value={}, help="Preparation/application outcome for this execution; empty for ordinary or legacy actions")
    actionProducts = List(Dict(), help="Products recorded by this execution, with committed/uncommitted outcome")
    actionProductInputs = List(Dict(), help="Explicit product selectors and exact committed products resolved during execution")

    def __init__(self, action=None):
        """Create an execution snapshot, or an empty restoration instance."""
        super().__init__()
        self.actionStatus = pglActionStatus.INITIALIZED

        # Restoration must not fabricate execution identity or input provenance.
        if action is None:
            return

        self.actionID = str(uuid4())
        self.parentActionID = _activeActionID.get()
        self.runDateTime = datetime.now().astimezone().isoformat()
        self.actionName = action.name
        self.actionStatus = action.status
        self.actionVersion = action.version
        self.actionSettings = action._captureHistorySettings()

        # Prepared by pipeline.run(). Ordinary actions retain an empty record.
        # A supplied working object must not be described as an unchanged
        # checkpoint merely because it still has a sessionID.
        self.actionInput = deepcopy(getattr(action, "_pipelineExecutionInput", {}))
        self.actionCode = self.captureActionCode(action)

        self.startTime = pglTimestamp.getSecs()
        self.endTime = self.startTime
        self.runDuration = 0.0

    @staticmethod
    def captureActionCode(action):
        """Capture inspect source and class identity without executing source.

        The hash identifies the exact captured UTF-8 source text, not the
        complete runtime environment or all inherited/imported code.
        """
        import hashlib
        import inspect

        actionClass = type(action)
        snapshot = {
            "version": 1,
            "module": actionClass.__module__,
            "qualname": actionClass.__qualname__,
            "captureMethod": "inspect.getsource",
            "status": "unavailable",
            "source": None,
            "sha256": None,
            "error": None,
        }

        try:
            source = inspect.getsource(actionClass)

            if not isinstance(source, str) or not source.strip():
                raise ValueError("inspect returned no usable class source.")

            snapshot["source"] = source
            snapshot["sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            snapshot["status"] = "captured"

        except Exception as error:
            # Notebook/generated classes may have no retrievable source.
            # Keep the class identity and an explicit diagnostic.
            snapshot["error"] = f"{type(error).__name__}: {error}"

        return snapshot

    def update(self, action):
        """Finalize outcome, timing, application, and scientific product records.

        Configuration and source snapshots remain unchanged.
        Resolved dependencies are retained even when execution later fails.
        Products from unsuccessful executions remain uncommitted.

        This is metadata finalization, not scientific or filesystem rollback.
        """
        from .pglExternalTools import getExecutionRecorder

        self.actionStatus = action.status
        self.actionError = action.error
        self.endTime = pglTimestamp.getSecs()
        self.runDuration = self.endTime - self.startTime

        application = getattr(action, "_executionApplication", None)
        if application is not None:
            self.actionApplication = deepcopy(application)

        recorder = getExecutionRecorder()
        if recorder is not None and recorder.actionID == self.actionID:
            self.actionProductInputs = recorder.exportProductInputs()
            outcome = "committed" if action.status == pglActionStatus.SUCCESS else "uncommitted"
            self.actionProducts = [
                {**record, "recordStatus": outcome}
                for record in recorder.exportProducts()
            ]
            
    def toJSONdict(self, type="all"):
        """Preserve absent optional fields in previously saved history.

        New provenance fields must not appear in legacy history merely
        because the installed class now declares them.
        """
        data = super().toJSONdict(type=type)
        originalFields = getattr(self, "_originalHistoryFields", None)

        collectionFields = {
            "actionCode", "actionInput", "actionEnvironment",
            "actionInvocations", "actionApplication", "actionProducts",
            "actionProductInputs", "actionLogs",
        }

        for name in (
            "actionID", "parentActionID", "actionCode", "actionInput",
            "actionEnvironment", "actionInvocations", "actionApplication",
            "actionProducts", "actionProductInputs", "actionLogs",
        ):
            value = data.get(name)
            empty = not value if name in collectionFields else value is None

            if empty and (originalFields is None or name not in originalFields):
                data.pop(name, None)

        return data
        
    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore history without creating new execution provenance."""
        # Validate only the new representation. Legacy records retain
        # their original fields and are never assigned inferred identities.
        inputRecord = data.get("actionInput", {})
        if isinstance(inputRecord, dict) and inputRecord.get("kind") == "actionInputs":
            cls.validateActionInputs(inputRecord)

        obj = cls()
        obj.updateTraitsFromDict(data, filename=filename)

        # Runtime-only bookkeeping preserves explicitly present null/empty
        # fields as well as older documents where those fields were absent.
        obj._originalHistoryFields = frozenset(data)
        return obj

    def __repr__(self):
        """Return a short status description."""
        return f"{self.actionName} status: {self.actionStatus.name}"

    def toString(self):
        """Return a readable execution description."""
        if self.actionStatus.value > pglActionStatus.CONFIGURED.value:
            runDateTime = datetime.fromisoformat(self.runDateTime).strftime("%H:%M:%S %m/%d/%Y")
            return f"{self.actionName} ran at {runDateTime} with status: {self.actionStatus.name} duration: {pglTimestamp.formatDuration(self.runDuration)}"

        return f"{self.actionName} status: {self.actionStatus.name}"

    def print(self):
        """Print the execution description without displaying full source."""
        pglMessages.print(self.toString())
        
    ################################################################
    # Preserve command logs and attach them to a history recipient
    ################################################################
    def _preserveLogs(self, owner, *, cacheDirectory=None):
        """Copy command logs into a persistent cache and attach their files.

        Invocation provenance remains unchanged. actionLogs supplies separate
        attachment references and records unavailable archives explicitly.

        Called after command execution and invocation finalization.
        Multiple history recipients reuse the same archived bytes.

        Ordinary archival errors warn without changing scientific status.
        Objects without the attachment interface retain local archive references.
        """
        import hashlib
        import os
        import shutil
        import tempfile
        from pathlib import Path

        from .pglExternalTools import pglExternalTool

        logs = deepcopy(self.actionLogs)
        canAttach = callable(getattr(owner, "addDerivedFile", None)) and hasattr(owner, "derivedFiles")

        def archiveRoot():
            """Use an explicit action cache or a persistent per-user default."""
            if cacheDirectory is None:
                return Path.home() / ".cache" / "pgl" / "actionLogs"

            text = os.fspath(cacheDirectory)
            if (
                not isinstance(text, str) or not text.strip()
                or "://" in text or "::" in text
                or any(character in text for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("Log cacheDirectory must be an ordinary local path.")

            return Path(text).expanduser().absolute() / "actionLogs"

        for invocation in self.actionInvocations:
            invocationID = invocation.get("invocationID")
            invocationResult = invocation.get("result")
            if not isinstance(invocationID, str) or not isinstance(invocationResult, dict): continue

            # Generic command records contain their own logs. Enclosing
            # scientific records need not duplicate their child log files.
            commandLogs = invocationResult.get("logs")
            if not isinstance(commandLogs, dict): continue

            streamRecords = logs.setdefault(invocationID, {})

            for stream in ("stdout", "stderr"):
                original = commandLogs.get(stream)
                if not isinstance(original, dict): continue

                attachmentName = f"action.{self.actionID}.invocation.{invocationID}.{stream}"
                previous = streamRecords.get(stream, {})
                record = {
                    "attachmentName": attachmentName,
                    "originalPath": original.get("localPath"),
                    "archiveStatus": "unavailable",
                }
                streamRecords[stream] = record

                try:
                    # Another recipient may already have archived these bytes.
                    sourceText = previous.get("localPath") if previous.get("archiveStatus") == "archived" else original.get("localPath")
                    if not isinstance(sourceText, str) or not sourceText:
                        raise ValueError("The invocation has no usable log path.")

                    source = Path(sourceText)
                    identity = pglExternalTool._fileIdentity(source)

                    # Compare with execution-time identity when it was captured.
                    for key in ("sha256", "size_bytes"):
                        if key in original and identity[key] != original[key]:
                            raise ValueError("Log bytes differ from their execution-time identity.")

                    root = archiveRoot()
                    root.mkdir(parents=True, exist_ok=True)

                    # Hashing the logical name avoids interpreting identifiers
                    # as directory components. Include content identity too.
                    nameHash = hashlib.sha256(attachmentName.encode("utf-8")).hexdigest()
                    destination = root / f"{nameHash}-{identity['sha256']}.log"

                    if destination.exists():
                        if pglExternalTool._fileIdentity(destination) != identity:
                            raise ValueError("An existing archived log has a different identity.")
                    else:
                        # Publish only a complete, verified copy. An interrupted
                        # copy must not look like an archived log.
                        temporaryPath = None
                        try:
                            with tempfile.NamedTemporaryFile(prefix=".pgl-log-", dir=root, delete=False) as temporary:
                                temporaryPath = Path(temporary.name)
                                with source.open("rb") as handle: shutil.copyfileobj(handle, temporary)
                                temporary.flush()
                                os.fsync(temporary.fileno())

                            if pglExternalTool._fileIdentity(temporaryPath) != identity:
                                raise ValueError("Log changed while being archived.")

                            os.replace(temporaryPath, destination)
                        finally:
                            if temporaryPath is not None: temporaryPath.unlink(missing_ok=True)

                    record.update({
                        **identity,
                        "localPath": str(destination),
                        "archiveStatus": "archived",
                    })

                    # An archived file and an attachment are distinct:
                    # history can be shared by several scientific recipients.
                    if canAttach:
                        attachment = owner.derivedFiles.get(attachmentName)
                        if attachment is None:
                            owner.addDerivedFile(attachmentName, destination, mediaType="text/plain")
                            attachment = owner.derivedFiles[attachmentName]

                        if any(attachment.get(key) != identity[key] for key in ("sha256", "size_bytes")):
                            raise ValueError("The log attachment has a different identity.")

                except Exception as error:
                    # If copying succeeded but attachment failed, preserve that
                    # distinction and the path to the successfully archived file.
                    record["error"] = f"{type(error).__name__}: {error}"
                    pglMessages.warning(
                        f"Could not fully preserve {stream} for invocation {invocationID}: {error}",
                        level=1,
                    )

        self.actionLogs = logs
        
    def _preserveConsoleLogs(self, capture, owners):
        """Record and attach closed Python stdout/stderr transcripts.

        Uses the existing actionLogs layout, so history(showLogs=True) works
        without a separate viewer or fabricated external-command invocation.

        Called only after capture exits: file identities must describe closed,
        finalized files, not transcripts still receiving output.

        Local files remain available even when attachment fails. Incomplete
        capture is recorded explicitly rather than labeled fully archived.

        Does not alter scientific outcome, products, or failure flags.
        Does not replace existing external-command log records.
        """
        from copy import deepcopy
        from pathlib import Path

        from .pglExternalTools import pglExternalTool

        logs = deepcopy(self.actionLogs)
        consoleKey = f"action:{self.actionID}:console"

        if consoleKey in logs:
            raise ValueError("Python console logs are already finalized for this execution.")

        captureErrors = capture.captureErrors
        streams = {}
        owners = list(owners)

        # Readable attachment names; execution identity stays in consoleKey.
        import re
        actionLabel = re.sub(r"[^A-Za-z0-9_-]+", "_", self.actionName or "action").strip("_") or "action"
        usedNames = {
            name for owner in owners
            for name in getattr(owner, "derivedFiles", {})
        }
        for logStreams in logs.values():
            if not isinstance(logStreams, dict): continue
            for logRecord in logStreams.values():
                if isinstance(logRecord, dict) and logRecord.get("attachmentName"):
                    usedNames.add(logRecord["attachmentName"])

        for stream in ("stdout", "stderr"):
            baseName = f"{actionLabel}_{stream}"
            attachmentName = f"{baseName}.log"
            suffix = 2
            while attachmentName in usedNames:
                attachmentName = f"{baseName}_v{suffix}.log"
                suffix += 1
            usedNames.add(attachmentName)
            record = {
                "attachmentName": attachmentName,
                "displayName": attachmentName,
                "archiveStatus": "unavailable",
            }
            streams[stream] = record

            try:
                path = Path(capture.paths[stream])
                record["originalPath"] = str(path)
                record["localPath"] = str(path)

                identity = pglExternalTool._fileIdentity(path)
                record.update(identity)

                errors = list(captureErrors.get(stream, []))
                if not errors:
                    record["archiveStatus"] = "archived"

                # Each owner receives its own attachment record referencing
                # the same immutable finalized transcript.
                for owner in owners:
                    if not callable(getattr(owner, "addDerivedFile", None)): continue
                    if not hasattr(owner, "derivedFiles"): continue

                    try:
                        attachment = owner.derivedFiles.get(attachmentName)
                        if attachment is None:
                            owner.addDerivedFile(attachmentName, path, mediaType="text/plain")
                            attachment = owner.derivedFiles.get(attachmentName)

                        if attachment is None or any(
                            attachment.get(key) != value for key, value in identity.items()
                        ):
                            raise ValueError("Console attachment identity differs from its transcript.")

                    except Exception as error:
                        errors.append(f"Attachment to {type(owner).__name__}: {type(error).__name__}: {error}")

                if errors:
                    record["error"] = "; ".join(errors)

            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"

        logs[consoleKey] = streams
        self.actionLogs = logs

        # These warnings happen after transcript closure. They cannot safely be
        # appended to files whose identities have already been registered.
        for stream, record in streams.items():
            if record.get("error"):
                pglMessages.warning(
                    f"Action {self.actionID} {stream} log preservation: {record['error']}",
                    level=1,
                )


    @staticmethod
    def captureActionInputs(args, kwargs, *, replayInput=None):
        """Capture supplied working inputs before scientific execution.

        Inspect direct arguments and built-in list/tuple/dict containers.
        Do not traverse arbitrary objects, resolve payloads, or read defaults.

        Saved references come only from storage-established session binding.
        They identify a saved source, not the complete current working state.

        String dictionary keys retain their existing path representation.
        Other dictionary keys use {"mappingKey": key}; the existing PGL codec
        preserves supported key types, including tuples.

        Configuration-only dictionaries impose no key-serialization requirement.
        A key on a path to a scientific input must be supported by the PGL codec.

        Cyclic container back-references are skipped on the active traversal path.
        Shared containers reached through separate argument paths are inspected
        separately. This records finite simple paths, not infinite cyclic paths.

        Pipeline replay metadata remains separate inside actionInput.
        Does not modify arguments, history, or database preferences.
        """
        from .pglDatabase import pglDatabase
        from .pglSession import pglSession

        inputs = []
        activeContainers = set()

        def visit(value, path):
            if isinstance(value, pglActionable) and not isinstance(value, pglAction):
                reference = (
                    pglDatabase._getBoundCheckpointReference(value)
                    if isinstance(value, pglSession) else None
                )
                inputs.append({
                    "path": deepcopy(path),
                    "objectClass": f"{type(value).__module__}.{type(value).__qualname__}",
                    "state": "workingObject",
                    "checkpoint": reference,
                })
                return

            # Do not invoke custom collection iteration or inspect arbitrary
            # attributes merely to discover provenance.
            if type(value) not in (list, tuple, dict): return

            identity = id(value)
            if identity in activeContainers: return

            activeContainers.add(identity)
            try:
                if type(value) is dict:
                    for key, child in value.items():
                        component = key if type(key) is str else {"mappingKey": key}
                        visit(child, [*path, component])
                else:
                    for index, child in enumerate(value):
                        visit(child, [*path, index])
            finally:
                activeContainers.remove(identity)

        for index, value in enumerate(args):
            visit(value, ["args", index])
        for name, value in kwargs.items():
            visit(value, ["kwargs", name])

        record = {
            "version": 2,
            "kind": "actionInputs",
            "inputs": inputs,
            "replayInput": deepcopy({} if replayInput is None else replayInput),
        }
        pglActionHistory.validateActionInputs(record)
        return record

    @staticmethod
    def validateActionInputs(record):
        """Validate new input provenance without connecting or changing records.

        Legacy string-keyed paths remain valid. Typed mapping-key components
        delegate key validation and encoding to the existing PGL codec.

        The PGL serializer—not plain json.dumps—is the persistence contract for
        records containing tuple keys. Do not coerce keys into display strings.

        Legacy actionInput records remain governed by their existing contracts.
        """
        import json
        from types import SimpleNamespace
        from uuid import UUID

        from .pglDatabase import pglDatabase
        from .pglSerialize import pglSerialize, _pglValidateMappingKey

        if type(record) is not dict or set(record) != {"version", "kind", "inputs", "replayInput"}:
            raise ValueError("Invalid action-input provenance fields.")
        if type(record["version"]) is not int or record["version"] != 2 or record["kind"] != "actionInputs":
            raise ValueError("Unsupported action-input provenance version or kind.")
        if type(record["inputs"]) is not list:
            raise ValueError("Action inputs must be a list.")

        seenPaths = set()
        for item in record["inputs"]:
            if type(item) is not dict or set(item) != {"path", "objectClass", "state", "checkpoint"}:
                raise ValueError("Invalid working-input provenance fields.")
            if item["state"] != "workingObject":
                raise ValueError("Action inputs must describe working objects.")

            path = item["path"]
            if type(path) is not list or len(path) < 2:
                raise ValueError("Action input requires an argument path.")
            if path[0] == "args":
                if type(path[1]) is not int or path[1] < 0:
                    raise ValueError("Invalid positional argument path.")
            elif path[0] == "kwargs":
                if type(path[1]) is not str:
                    raise ValueError("Invalid keyword argument path.")
            else:
                raise ValueError("Unknown action argument path root.")

            for position, component in enumerate(path[2:], start=2):
                if type(component) is str: continue
                if type(component) is int and component >= 0: continue

                if type(component) is not dict or set(component) != {"mappingKey"}:
                    raise ValueError("Unsupported action argument path component.")

                key = component["mappingKey"]
                if type(key) is str:
                    raise ValueError("String mapping keys must use the existing direct path representation.")

                _pglValidateMappingKey(key, f"actionInput.path[{position}].mappingKey")

            # Encode through the shared codec so typed keys cannot collide with
            # string keys or sequence indices. No arbitrary scientific values
            # are serialized here.
            encodedPath = pglSerialize.toJSON(SimpleNamespace(path=path), strict=True)
            pathKey = json.dumps(json.loads(encodedPath), sort_keys=True, separators=(",", ":"), allow_nan=False)
            if pathKey in seenPaths:
                raise ValueError("Duplicate action input argument path.")
            seenPaths.add(pathKey)

            if not isinstance(item["objectClass"], str) or not item["objectClass"].strip():
                raise ValueError("Action input requires an object class label.")

            reference = item["checkpoint"]
            if reference is None: continue
            if type(reference) is not dict or set(reference) != {"databaseID", "sessionID", "databaseName"}:
                raise ValueError("Invalid action checkpoint reference.")

            sessionID = reference["sessionID"]
            if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
                raise ValueError("Action checkpoint sessionID must be a nonnegative PostgreSQL bigint.")

            databaseID = reference["databaseID"]
            if not isinstance(databaseID, str):
                raise ValueError("Action checkpoint databaseID must be a UUID string.")
            try:
                parsed = UUID(databaseID)
            except (ValueError, AttributeError):
                raise ValueError("Invalid action checkpoint databaseID.") from None
            if str(parsed) != databaseID or parsed.int == 0:
                raise ValueError("Action checkpoint databaseID must be canonical and nonzero.")

            label = reference["databaseName"]
            if (
                not isinstance(label, str) or not label.strip()
                or any(character in label for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("Invalid action checkpoint databaseName label.")

        replay = record["replayInput"]
        if type(replay) is not dict:
            raise ValueError("Pipeline replay input must be a dictionary.")

        if replay:
            if replay == {"version": 1, "kind": "workingObject"}:
                if type(replay["version"]) is not int:
                    raise ValueError("Invalid working-object replay input version.")
            else:
                pglDatabase._validatePipelineReplayInput(replay)

        return record

    @staticmethod
    def getRecordedReplayInput(record):
        """Read pipeline replay metadata without rewriting historical records.

        Legacy histories store replay metadata directly in actionInput.
        New histories preserve it inside the version-2 actionInputs record.
        Ordinary working-source references are never promoted to replay inputs.
        """
        if type(record) is not dict:
            raise ValueError("Action input must be a dictionary.")

        if record.get("kind") == "actionInputs":
            pglActionHistory.validateActionInputs(record)
            return deepcopy(record["replayInput"])

        return deepcopy(record)

           

########################
# class pglActionable
########################
class pglActionable(pglTraitSettings):
    """Scientific object carrying history and runtime checkpoint state."""

    actionHistory = List(Instance(pglActionHistory), help="History of actions that have been run")
    _failedActionIDs = List(Unicode(), help="Failed executions that may have modified this working object").tag(serialize=False)
    _checkpointBaseline = Dict(default_value={}, help="Storage-managed checkpoint identity and history boundary").tag(serialize=False)

    def _recordFailedAction(self, actionID):
        """Remember a failed attempt without changing scientific history.

        Later successful actions do not clear an earlier failure.
        """
        if not isinstance(actionID, str) or not actionID:
            raise ValueError("A failed action must have a nonempty execution ID.")

        if actionID not in self._failedActionIDs:
            self._failedActionIDs = [*self._failedActionIDs, actionID]

    def history(self, actionIndex=None, *, verbose=False, showProducts=False, showLogs=False, tail=None, cacheDirectory=None):
        """List history or inspect one action and its saved command logs.

        history():
            Numbered action list with archived-log counts.

        history(showProducts=True):
            Include aligned product-kind summaries beneath each action.

        history(verbose=True):
            Include bounded configuration settings for each action and
            technical product-summary metadata when showProducts=True.

        history(index):
            Detailed action information, including available log references.
            Negative indices count from the end, just like a Python list:
            history(-1) selects the last action; history(-2) the preceding one.
            The displayed history index remains nonnegative.

        history(index, showLogs=True, tail=None):
            Retrieve and print logs. tail optionally limits each log to its
            final N lines. None prints the complete contents.

        Legacy history(True/False) remains a positional verbose setting.
        Listing and product summaries read history metadata only.
        No session/run overview is printed.
        Only showLogs requests log payload retrieval.
        """
        import shlex
        from collections import deque
        from pathlib import Path
        from pprint import pformat

        # Preserve the previous history(verbose) positional calling convention.
        if type(actionIndex) is bool:
            verbose = verbose or actionIndex
            actionIndex = None

        if actionIndex is not None and type(actionIndex) is not int:
            raise TypeError("actionIndex must be an integer or None.")
        if any(type(value) is not bool for value in (verbose, showProducts, showLogs)):
            raise TypeError("verbose, showProducts, and showLogs must be Boolean.")
        if tail is not None and (type(tail) is not int or tail <= 0):
            raise ValueError("tail must be a positive integer or None.")
        if showLogs and actionIndex is None:
            raise ValueError("Select a history index when requesting log contents.")

        # Accept Python-style negative indexing, then normalize for display.
        # Out-of-bounds selection warns and returns without printing history.
        if actionIndex is not None:
            historyCount = len(self.actionHistory)
            if not -historyCount <= actionIndex < historyCount:
                validRange = f"-{historyCount} through {historyCount - 1}" if historyCount else "(history is empty)"
                pglMessages.warning(f"actionIndex {actionIndex} is outside this object's history. Valid indices: {validRange}.", level=1)
                return
            if actionIndex < 0: actionIndex += historyCount
            
        pglMessages.printHeader("action history")

        if self._failedActionIDs:
            pglMessages.warning(
                "This working object has failed action attempts and is blocked from checkpoint saving. "
                "Reload a saved checkpoint before continuing with a saveable analysis.",
                level=2,
            )

        if not self.actionHistory:
            pglMessages.message(f"{type(self).__name__} has no history")
            return

        def printSettings(entry):
            """Print bounded configuration summaries without expanding huge records.

            Named record lists show counts and a few names, not their nested
            source metadata. Other values use depth-limited representations.
            Stored configuration remains unchanged.
            """
            from itertools import islice
            from reprlib import Repr

            formatter = Repr()
            formatter.maxlevel = 2
            formatter.maxlist = 4
            formatter.maxtuple = 4
            formatter.maxdict = 3
            formatter.maxset = 4
            formatter.maxfrozenset = 4
            formatter.maxstring = 100
            formatter.maxother = 100
            formatter.maxlong = 40

            if not entry.actionSettings:
                pglMessages.print("    (No saved configuration parameters)")
                return

            for name, value in sorted(entry.actionSettings.items()):
                if isinstance(value, (list, tuple)) and value and all(isinstance(item, dict) for item in value):
                    # Scan selectors carry complete source snapshots for replay.
                    # Print only their readable labels, never those snapshots.
                    labels = []
                    for item in value[:3]:
                        label = item.get("name") or item.get("label")
                        labels.append(self._productDisplayText(label, 70) if isinstance(label, str) else "(unnamed record)")

                    valueText = f"{len(value)} record(s); names=" + ", ".join(labels)
                    if len(value) > 3: valueText += f", +{len(value) - 3} more"

                elif isinstance(value, dict):
                    # A shallow key inventory avoids exposing large nested
                    # records merely to explain what a setting contains.
                    keys = [self._productDisplayText(key, 40) for key in islice(value, 4)]
                    valueText = f"{len(value)} field(s); keys=" + ", ".join(keys) if value else "{}"
                    if len(value) > 4: valueText += f", +{len(value) - 4} more"

                else:
                    valueText = formatter.repr(value)
                    if isinstance(value, (list, tuple, set, frozenset)) and len(value) > 4:
                        valueText = f"{len(value)} items; {valueText}"

                # Keep every setting on one bounded line, including strings.
                pglMessages.print(f"    {name} = {self._productDisplayText(valueText, 240)}")

        widths = self._productSummaryWidths() if showProducts else None

        if actionIndex is None:
            for index, entry in enumerate(self.actionHistory):
                pglMessages.print(self._historyActionLine(index, entry))
                if showProducts: self._printProductRows(entry, widths=widths, verbose=verbose)
                if verbose: printSettings(entry)
            return

        entry = self.actionHistory[actionIndex]
        pglMessages.print(self._historyActionLine(actionIndex, entry))
        if showProducts: self._printProductRows(entry, widths=widths, verbose=verbose)

        pglMessages.print(f"Execution ID: {entry.actionID}")
        if entry.parentActionID: pglMessages.print(f"Parent execution: {entry.parentActionID}")
        if entry.actionError is not None: pglMessages.print(f"Error: {entry.actionError}")

        pglMessages.printHeader("configuration")
        printSettings(entry)

        if entry.actionApplication:
            pglMessages.printHeader("application")
            pglMessages.print(pformat(entry.actionApplication, sort_dicts=False))

        pglMessages.printHeader("input products")
        if entry.actionProductInputs:
            for productInput in entry.actionProductInputs:
                pglMessages.print(pformat(productInput, sort_dicts=False))
        else:
            pglMessages.print("(No recorded product dependencies)")

        pglMessages.printHeader("output products")
        if entry.actionProducts:
            for product in entry.actionProducts:
                pglMessages.print(
                    f"{product.get('name')} [{product.get('kind')}] "
                    f"ID={product.get('productID')} "
                    f"status={product.get('recordStatus', 'unspecified')}"
                )
        else:
            pglMessages.print("(No recorded products)")

        pglMessages.printHeader("external commands")
        if not entry.actionInvocations:
            pglMessages.print("(No recorded external invocations)")

        for invocation in entry.actionInvocations:
            invocationID = invocation.get("invocationID", "(unknown)")
            invocationResult = invocation.get("result", {})
            executable = invocationResult.get("resolvedExecutable") if isinstance(invocationResult, dict) else None

            # Show the actual executable when recorded. Invocation IDs remain
            # the authoritative association between commands and log files.
            label = str(executable) if executable else invocation.get("operation", "invocation")
            pglMessages.print(f"{invocationID}: {label} — {invocation.get('status', 'unknown')}")

            # Some recorders retain command specifications in a nested field.
            specification = invocation.get("specification", invocation)
            if isinstance(specification, dict):
                arguments = specification.get("arguments")
                commandName = executable or specification.get("requestedExecutable")
                if commandName and isinstance(arguments, (list, tuple)):
                    pglMessages.print(f"    {shlex.join([str(commandName), *map(str, arguments)])}")

        pglMessages.printHeader("log files")
        records = list(self._historyLogRecords(entry))
        if not records:
            pglMessages.print("(No archived logs; older history may contain only original paths in actionInvocations.)")
            return

        attachments = getattr(self, "derivedFiles", {})
        getFile = getattr(self, "getDerivedFile", None)

        for invocationID, stream, record in records:
            attachmentName = record.get("attachmentName")
            attachment = attachments.get(attachmentName)
            archived = record.get("archiveStatus") == "archived"

            pglMessages.print(f"{invocationID} — {stream}")
            pglMessages.print(f"    Attachment: {attachmentName}")
            pglMessages.print(f"    Archive status: {record.get('archiveStatus', 'unknown')}")
            pglMessages.print(f"    Attached to this object: {'yes' if attachment is not None else 'no'}")
            if "size_bytes" in record: pglMessages.print(f"    Size: {record['size_bytes']} bytes")
            if record.get("originalPath"): pglMessages.print(f"    Original path: {record['originalPath']}")
            if record.get("localPath"): pglMessages.print(f"    Original archive location: {record['localPath']}")
            if record.get("error"): pglMessages.print(f"    Archival diagnostic: {record['error']}")

            if not showLogs: continue

            if not archived:
                pglMessages.print("    Log contents unavailable: archival did not complete.")
                continue

            try:
                if callable(getFile):
                    # Never silently substitute a stale filesystem path for a
                    # missing attachment on a session.
                    if attachment is None:
                        raise ValueError("The log attachment is absent from this object.")
                    if any(attachment.get(key) != record.get(key) for key in ("sha256", "size_bytes")):
                        raise ValueError("The attachment identity differs from action history.")

                    path = getFile(attachmentName, cacheDirectory=cacheDirectory)
                    if path is None: raise ValueError("The log attachment could not be retrieved.")

                else:
                    # An action object may carry history but no attachment
                    # interface. Its local archive is still inspectable.
                    from .pglExternalTools import pglExternalTool

                    path = Path(record["localPath"])
                    expected = {key: record[key] for key in ("sha256", "size_bytes")}
                    if pglExternalTool._fileIdentity(path) != expected:
                        raise ValueError("The local archive identity differs from history.")

                pglMessages.printHeader(f"{stream}: {invocationID}")

                if record.get("size_bytes") == 0:
                    pglMessages.print("(Empty log)")
                    continue

                # Stream full logs line by line; retain only the requested tail.
                with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
                    lines = handle if tail is None else deque(handle, maxlen=tail)
                    for line in lines:
                        pglMessages.print(line.rstrip("\r\n"))

            except Exception as error:
                pglMessages.warning(
                    f"Could not display {stream} for invocation {invocationID}: {error}",
                    level=1,
                )
    
    ################################################################
    # Find a committed named scientific product
    ################################################################
    def getProduct(self, name=None, *, productID=None):
        """Get a committed product by name, kind, history index, or exact ID.

        Integer:
            Zero-based actionHistory index. Select from that execution's products.

        String:
            Exact product names take precedence and retain existing explicit
            supersession semantics. Otherwise interpret the string as a kind.

        Multiple candidates:
            Open the existing item-selection dialog and require one selection.
            Cancellation returns None.

        Explicit productID:
            Resolve that exact product without a dialog.

        Does not load payloads, rerun producers, or record scientific consumption.
        Scientific actions should continue using pglProductSelector directly.
        """
        from numbers import Integral

        try:
            if productID is not None:
                if name is not None: raise ValueError("Supply either name or productID, not both.")
                return pglProductSelector(productID=productID)._resolve(self)

            if isinstance(name, bool):
                raise TypeError("Supply a product name, kind, or integer history index—not a Boolean.")

            historyIndex = None
            if isinstance(name, Integral):
                historyIndex = int(name)
                if not 0 <= historyIndex < len(self.actionHistory):
                    raise ValueError(f"History index {historyIndex} is outside this object's action history.")
            elif not isinstance(name, str) or not name.strip():
                raise ValueError("Supply a product name, kind, integer history index, or productID.")

            candidates = [
                (index, entry, record)
                for index, entry in enumerate(self.actionHistory)
                if entry.actionStatus == pglActionStatus.SUCCESS
                for record in entry.actionProducts
                if record.get("recordStatus") == "committed"
            ]

            if historyIndex is not None:
                candidates = [item for item in candidates if item[0] == historyIndex]
                selectionLabel = f"action {historyIndex}"
            else:
                # Preserve established name/supersession semantics rather than
                # interpreting an existing name as a kind.
                if any(record.get("name") == name for _, _, record in candidates):
                    return pglProductSelector(name)._resolve(self)

                candidates = [item for item in candidates if item[2].get("kind") == name]
                selectionLabel = f"kind {name!r}"

            matches = []
            seenIDs = set()
            for index, entry, record in candidates:
                identifier = record.get("productID")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("A matching committed product lacks its productID.")
                if identifier in seenIDs: continue
                seenIDs.add(identifier)
                matches.append((index, entry, record))

            if not matches:
                if historyIndex is not None:
                    raise ValueError(f"Action history entry {historyIndex} has no committed products from a successful execution.")
                raise ValueError(f"No committed product has name or kind {name!r}.")

            if len(matches) == 1:
                selected = matches[0][2]
            else:
                from .pglDialog import pglDialogs

                labels = []
                for index, entry, record in matches:
                    productName = " ".join(str(record.get("name") or "(unnamed)").split())
                    kind = " ".join(str(record.get("kind") or "unspecified").split())
                    actionName = " ".join(str(entry.actionName).split())
                    dateTime = " ".join(str(entry.runDateTime).split())
                    labels.append(f"{productName} [{kind}] — action {index}: {actionName} — {dateTime}")

                positions = pglDialogs.selectItems(labels, title=f"Select one product — {selectionLabel}")
                if positions is None or len(positions) == 0: return None
                if len(positions) != 1:
                    pglMessages.warning("Select exactly one product.", level=1)
                    return None

                selected = matches[positions[0]][2]

            # Exact-ID resolution retains ordinary provenance checks and owner binding.
            return pglProductSelector(productID=selected["productID"])._resolve(self)

        except Exception as error:
            pglMessages.warning(f"Derived product was not selected: {error}")
            return None
    
    def getPipeline(self, name=None):
        """Extract an independent recipe from this object's recorded history.

        Does not edit history, load checkpoints, or execute actions.
        Extraction does not establish a default replay input.
        """
        return pglPipeline.fromHistory(self, name=name)
    
    ################################################################
    # Shared history/product presentation
    ################################################################
    @staticmethod
    def _productDisplayText(value, limit=120):
        """Keep metadata on one display line without printing large values."""
        text = " ".join(str(value).split())
        return text if len(text) <= limit else text[:limit - 3] + "..."

    @staticmethod
    def _historyLogRecords(entry):
        """Yield recorded log references without opening their files."""
        for invocationID, streams in entry.actionLogs.items():
            for stream in ("stdout", "stderr"):
                record = streams.get(stream)
                if isinstance(record, dict): yield invocationID, stream, record

    def _historyActionLine(self, index, entry):
        """Use exactly the same execution line in history and product views."""
        records = list(self._historyLogRecords(entry))
        archivedCount = sum(record.get("archiveStatus") == "archived" for _, _, record in records)
        unavailableCount = len(records) - archivedCount
        suffix = f" | {archivedCount} archived log(s)" if records else ""
        if unavailableCount: suffix += f", {unavailableCount} unavailable"
        return f"{index}: {entry.toString()}{suffix}"

    def _productSummaryRows(self, entry, *, verbose=False):
        """Summarize products with human-readable information first.

        Normal output:
            Names, backend, operation, descriptive scalar metadata, and
            explicitly recorded scientific counts.

        verbose=True:
            Append record-status counts, artifact-role counts, and remaining
            metadata. Large metadata values are bounded for display.

        Uncommitted/unspecified outcomes and unavailable logs remain visible
        regardless of verbosity.

        Counts describe recorded products and artifact references, not
        inferred numbers of scientific arrays. No payloads are accessed.
        """
        from collections import Counter

        if type(verbose) is not bool:
            raise TypeError("verbose must be Boolean.")

        groups = {}
        for product in entry.actionProducts:
            kind = self._productDisplayText(product.get("kind") or "unspecified")
            groups.setdefault(kind, []).append(product)

        # Deliberately prioritize meaningful fields rather than printing
        # arbitrary metadata keys in alphabetical order.
        usefulKeys = (
            "backend", "operation", "description", "summary",
            "nScans", "nRuns", "nVolumes", "nMatrices", "nTransforms",
            "units", "coordinateSystem",
        )
        rows = []

        def summarizeValues(values, *, limit=90):
            """Show distinct recorded values without expanding large collections."""
            texts = list(dict.fromkeys(self._productDisplayText(value, limit) for value in values))
            text = ", ".join(texts[:3])
            if len(texts) > 3: text += f", +{len(texts) - 3} more"
            return text

        for kind, products in groups.items():
            names = [product.get("name") or "(unnamed)" for product in products]
            details = [f"names={summarizeValues(names, limit=50)}"]
            metadataRecords = [product.get("metadata", {}) for product in products]
            displayedKeys = set()

            for key in usefulKeys:
                values = [metadata[key] for metadata in metadataRecords if key in metadata and metadata[key] is not None]
                if not values: continue

                # Ordinary output stays compact and avoids expanding nested
                # records, arrays, identities, and other machine metadata.
                if not all(type(value) in (str, bool, int, float) for value in values): continue

                valueText = summarizeValues(values)
                if len(values) != len(products): valueText += f" ({len(values)}/{len(products)} records)"
                details.append(f"{key}={valueText}")
                displayedKeys.add(key)

            statuses = Counter(product.get("recordStatus") or "unspecified" for product in products)

            # Never conceal unsuccessful or incompletely recorded outcomes.
            exceptionalStatuses = {status: count for status, count in statuses.items() if status != "committed"}
            if exceptionalStatuses:
                details.append("status: " + ", ".join(f"{self._productDisplayText(status)}={count}" for status, count in exceptionalStatuses.items()))

            if entry.actionStatus != pglActionStatus.SUCCESS and statuses.get("committed"):
                details.append("WARNING: committed records belong to a non-successful action")

            if verbose:
                details.append("records: " + ", ".join(f"{self._productDisplayText(status)}={count}" for status, count in statuses.items()))

                roles = Counter(role for product in products for role in product.get("artifacts", {}))
                if roles:
                    details.append("artifacts: " + ", ".join(f"{self._productDisplayText(role)}={count}" for role, count in roles.items()))

                # Remaining metadata belongs at the end, and only in the
                # explicitly requested technical view.
                remainingKeys = set().union(*(set(metadata) for metadata in metadataRecords)) - displayedKeys
                for key in sorted(remainingKeys):
                    values = [metadata[key] for metadata in metadataRecords if key in metadata]
                    details.append(f"{self._productDisplayText(key, 40)}={summarizeValues(values)}")

            rows.append((kind, len(products), "; ".join(details)))

        logRecords = list(self._historyLogRecords(entry))
        if logRecords:
            streams = Counter(stream for _, stream, _ in logRecords)
            archivedCount = sum(record.get("archiveStatus") == "archived" for _, _, record in logRecords)
            unavailableCount = len(logRecords) - archivedCount

            details = [", ".join(f"{stream}={count}" for stream, count in streams.items())]
            if unavailableCount: details.append(f"unavailable={unavailableCount}")
            if verbose: details.append(f"archived={archivedCount}")

            rows.append(("logs", len(logRecords), "; ".join(details)))

        return rows

    def _productSummaryWidths(self):
        """Calculate shared column widths across this object's entire history."""
        rows = [row for entry in self.actionHistory for row in self._productSummaryRows(entry)]
        kindWidth = max([len("Kind"), *(len(kind) for kind, _, _ in rows)])
        countWidth = max([1, *(len(str(count)) for _, count, _ in rows)])
        return kindWidth, countWidth

    def _printProductRows(self, entry, *, widths, details=False, verbose=False):
        """Print readable recorded products without opening scientific payloads.

        Exact-ID retrieval commands preserve historical product selection.
        UUIDs remain available for identity, not as primary artifact labels.
        """
        import mimetypes
        from pathlib import Path

        if type(verbose) is not bool or type(details) is not bool:
            raise TypeError("verbose and details must be Boolean.")

        kindWidth, countWidth = widths
        rows = self._productSummaryRows(entry, verbose=verbose)
        for kind, count, description in rows:
            pglMessages.print(f"    {kind:<{kindWidth}}  {count:>{countWidth}}  {description}")

        if not details:
            if rows:
                indices = [index for index, candidate in enumerate(self.actionHistory) if candidate is entry]
                if len(indices) == 1: pglMessages.print(f"    Inspect/get: session.printDataProducts({indices[0]})")
            return

        derivedFiles = getattr(self, "derivedFiles", {})
        imageExtensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        imageTypes = {"image/png", "image/jpeg", "image/tiff", "image/bmp"}

        for product in entry.actionProducts:
            name = self._productDisplayText(product.get("name") or "(unnamed)")
            kind = self._productDisplayText(product.get("kind") or "unspecified")
            status = self._productDisplayText(product.get("recordStatus") or "unspecified")
            identifier = product.get("productID")
            committed = entry.actionStatus == pglActionStatus.SUCCESS and product.get("recordStatus") == "committed"
            canSelect = committed and isinstance(identifier, str) and bool(identifier.strip())
            metadata = product.get("metadata", {})

            pglMessages.print(f"        {name} [{kind}] | {status}")
            if canSelect:
                pglMessages.print(f"            Get: product = session.getProduct(productID={identifier!r})")
                pglMessages.print("            Display: product.display()")
            else:
                pglMessages.print("            Get: unavailable — no committed selectable record")

            labels = {
                item["role"]: item["relativePath"]
                for item in metadata.get("supportArtifacts", [])
                if isinstance(item, dict) and "role" in item and "relativePath" in item
            }

            for role, artifact in product.get("artifacts", {}).items():
                attachmentName = artifact.get("attachmentName") or ""
                attachment = derivedFiles.get(attachmentName) or {}
                label = labels.get(role, role)
                recordedType = artifact.get("mediaType") or attachment.get("mediaType")
                inferredType = mimetypes.guess_type(str(label))[0]
                typeText = recordedType or (f"{inferredType} (inferred)" if inferredType else "type not recorded")
                size = artifact.get("size_bytes")
                sizeText = f"{size:,} bytes" if type(size) is int else "size not recorded"
                pglMessages.print(f"            {self._productDisplayText(label)} | {typeText} | {sizeText}")

                if canSelect:
                    pglMessages.print(f"                File: product.getFile(role={role!r})")
                    mediaType = str(recordedType or "").split(";", 1)[0].strip().lower()
                    extension = Path(str(label)).suffix.lower() or Path(attachmentName).suffix.lower()
                    if mediaType in imageTypes or (mediaType in ("", "application/octet-stream") and extension in imageExtensions):
                        pglMessages.print(f"                Show: product.displayArtifacts(role={role!r})")

                if verbose:
                    pglMessages.print(f"                Role: {role}")
                    pglMessages.print(f"                Attachment: {self._productDisplayText(attachmentName)}")

            # MRI-owned analysis matrices and declared replacement matrices.
            matrices = metadata.get("matrices", [])
            replacements = metadata.get("imageReplacements", {})
            if isinstance(replacements, dict): matrices = [*matrices, *replacements.get("stages", [])]

            mri = getattr(self, "mri", None)
            mriMatrices = getattr(mri, "dataMatrices", {}) if mri is not None else getattr(self, "dataMatrices", {})
            for declaration in matrices:
                if not isinstance(declaration, dict) or not declaration.get("matrixName"): continue
                matrixName = declaration["matrixName"]
                shape = declaration.get("shape")
                dimensions = declaration.get("dimensions")
                dtype = declaration.get("dtype")
                retained = matrixName in mriMatrices
                description = []
                if shape is not None: description.append(f"shape={tuple(shape)}")
                if dimensions is not None: description.append(f"dimensions={tuple(dimensions)}")
                if dtype is not None: description.append(f"dtype={dtype}")
                description.append("attached" if retained else "not attached here")
                pglMessages.print(f"            {self._productDisplayText(matrixName)} | matrix | {'; '.join(description)}")
                if retained:
                    container = "session.mri.dataMatrices" if mri is not None else "session.dataMatrices"
                    pglMessages.print(f"                Matrix: {container}[{matrixName!r}]")
                if verbose: pglMessages.print(f"                UUID: {declaration.get('matrixUUID', '(not recorded)')}")

        # Log records retain their execution/invocation identities independently.
        for invocationID, stream, record in self._historyLogRecords(entry):
            attachmentName = record.get("attachmentName")
            status = self._productDisplayText(record.get("archiveStatus") or "unspecified")
            label = record.get("displayName") or attachmentName or f"{entry.actionName}_{stream}.log"
            pglMessages.print(f"        {self._productDisplayText(label)} | text/plain | {status}")
            if record.get("archiveStatus") == "archived" and isinstance(attachmentName, str) and attachmentName:
                pglMessages.print(f"            File: session.getDerivedFile({attachmentName!r})")
            if verbose: pglMessages.print(f"            Invocation: {self._productDisplayText(invocationID)}")

    def _printRecordedProducts(self, actionIndex=None, *, details=False):
        """Print only producing executions, retaining original history indices."""
        if actionIndex is not None:
            if type(actionIndex) is not int: raise TypeError("actionIndex must be an integer or None.")
            if not 0 <= actionIndex < len(self.actionHistory): raise ValueError("actionIndex is outside this object's history.")

        pglMessages.printHeader("data products" if details else "products")
        pglMessages.print("Recorded outputs: product counts are records; artifact counts are file references.")

        if self._failedActionIDs:
            pglMessages.warning("This working object contains failed action attempts; uncommitted outputs are not usable committed products.", level=2)

        widths = self._productSummaryWidths()
        entries = enumerate(self.actionHistory) if actionIndex is None else [(actionIndex, self.actionHistory[actionIndex])]
        printed = False

        for index, entry in entries:
            if not self._productSummaryRows(entry): continue
            if printed: pglMessages.print()
            pglMessages.print(self._historyActionLine(index, entry))
            self._printProductRows(entry, widths=widths, details=details)
            printed = True

        if not printed:
            pglMessages.print("(No recorded products or command logs)")

    def printProducts(self, actionIndex=None):
        """Print producing actions with one aligned summary per product kind.

        Uses the same action lines and indices as history().
        Includes recorded command logs and explicitly labels product outcomes.

        Historical versions remain visible. This is not a current-attachment
        inventory, and missing provenance is never inferred from filenames.
        Reads history metadata only; does not retrieve or verify payload bytes.
        """
        self._printRecordedProducts(actionIndex, details=False)

    def printDataProducts(self, actionIndex=None):
        """Print product summaries plus individual names, IDs, and artifact roles.

        Optionally select one zero-based history index.
        Includes recorded log attachment names without opening their files.
        Products and attachments without recorded history are not inferred.
        """
        self._printRecordedProducts(actionIndex, details=True)
    
################################################################
# Portable scientific-product selector
################################################################
class pglProductSelector(pglSerialize):
    """Explicit, portable dependency on a named or pinned product.

    Name selection:
        Resolves the named committed product in the execution's input object.

    Pinned selection:
        Resolves an exact product UUID. Useful for deliberate reuse, but that
        UUID must exist in the input supplied during replay.

    kind:
        Optional expected semantic product type.

    Construction does not search history or access payloads.
    A missing product is diagnosed only when explicitly resolved.

    Examples:
        pglProductSelector("motion", kind="rigidMotion")
        pglProductSelector(productID=identifier, kind="rigidMotion")
    """

    def __init__(self, name=None, *, productID=None, kind=None):
        self.productName = name
        self.productID = productID
        self.kind = kind
        self._validate()

    def _validate(self):
        """Require one explicit selection mode and portable string values."""
        if (self.productName is None) == (self.productID is None):
            raise ValueError("Supply exactly one product name or productID.")

        for field, value in (
            ("productName", self.productName),
            ("productID", self.productID),
            ("kind", self.kind),
        ):
            if value is None: continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonblank string or None.")
            if any(character in value for character in ("\x00", "\n", "\r")):
                raise ValueError(f"{field} must not contain NUL or newlines.")

    def toJSONdict(self, type="all"):
        """Serialize intent, never resolved runtime objects."""
        self._validate()
        return {
            "schemaVersion": 1,
            "productName": self.productName,
            "productID": self.productID,
            "kind": self.kind,
        }

    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore selection intent without accessing scientific products."""
        import builtins

        expected = {"schemaVersion", "productName", "productID", "kind"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Invalid product-selector fields.")

        # The serializer's type argument shadows Python's built-in type().
        # Require an actual integer, not Boolean True.
        if builtins.type(data["schemaVersion"]) is not int or data["schemaVersion"] != 1:
            raise ValueError("Unsupported product-selector version.")

        return cls(
            data["productName"],
            productID=data["productID"],
            kind=data["kind"],
        )

    def resolve(self, owner):
        """Resolve a product and record its consuming owner's saved source.

        Reads product/history metadata only, not payload files.

        Inside an action:
            Records the selector, exact product record, and optional
            database-qualified checkpoint of the supplied session owner.
            Failures raise internally so scientific execution stops.

        Outside an action:
            Returns the product for inspection without recording consumption.
            Ordinary failures warn and return None.

        The owner checkpoint identifies the saved source beneath its working
        state. It does not establish that this product was saved there.
        Unbound owners have no checkpoint reference; defaults are never used
        to infer their database.

        Original product metadata and historical producer references remain
        unchanged. No producer is automatically run or payload regenerated.
        """
        from .pglExternalTools import getExecutionRecorder

        recorder = getExecutionRecorder()

        try:
            product = self._resolve(owner)

            if recorder is not None and recorder.actionID is not None:
                from .pglDatabase import pglDatabase
                from .pglSession import pglSession

                checkpoint = (
                    pglDatabase._getBoundCheckpointReference(owner)
                    if isinstance(owner, pglSession) else None
                )
                recorder.recordProductInput(
                    self.toJSONdict(),
                    product.record,
                    checkpoint=checkpoint,
                )

            return product

        except Exception as error:
            if recorder is not None and recorder.actionID is not None:
                raise

            pglMessages.warning(f"Product dependency could not be resolved: {error}")
            return None

    def _resolve(self, owner):
        """Resolve committed metadata by exact ID or explicit named version.

        Named versions must form one complete, unbranched supersession chain.
        No timestamp, history-order, or newest-product heuristic is used.

        Pinned IDs continue to resolve historical versions unchanged.
        """
        self._validate()
        if not isinstance(owner, pglActionable):
            raise TypeError("Product resolution requires an object carrying PGL history.")

        matches = []

        for entry in owner.actionHistory:
            if entry.actionStatus != pglActionStatus.SUCCESS: continue

            for record in entry.actionProducts:
                if record.get("recordStatus") != "committed": continue

                selected = (
                    record.get("name") == self.productName
                    if self.productName is not None else
                    record.get("productID") == self.productID
                )
                if not selected: continue

                if record.get("producerActionID") != entry.actionID:
                    raise ValueError("Product producer link disagrees with its history entry.")

                matches.append(record)

        label = self.productName if self.productName is not None else self.productID

        if not matches:
            raise ValueError(
                f"Required product {label!r} is unavailable. Run its producing "
                "action first or explicitly select a different input."
            )

        if self.productID is not None:
            # Pinned selection never follows a supersession link.
            if len(matches) != 1:
                raise ValueError(f"Product ID {label!r} occurs more than once in history.")
            record = matches[0]

        else:
            recordsByID = {}
            for candidate in matches:
                identifier = candidate.get("productID")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError(f"Product {label!r} has an invalid product ID.")
                if identifier in recordsByID:
                    raise ValueError(f"Product ID {identifier!r} occurs more than once in history.")
                recordsByID[identifier] = candidate

            predecessors = {}
            successors = {}

            for identifier, candidate in recordsByID.items():
                revision = candidate.get("metadata", {}).get("_productRevision")
                if revision is None: continue

                if (
                    not isinstance(revision, dict)
                    or set(revision) != {"schemaVersion", "supersedesProductID"}
                    or type(revision["schemaVersion"]) is not int
                    or revision["schemaVersion"] != 1
                ):
                    raise ValueError(f"Product {identifier!r} has invalid revision metadata.")

                previousID = revision["supersedesProductID"]
                if not isinstance(previousID, str) or not previousID.strip():
                    raise ValueError("A superseded product ID must be a nonblank string.")
                if previousID == identifier:
                    raise ValueError("A product cannot supersede itself.")
                if previousID not in recordsByID:
                    raise ValueError(
                        f"Product {identifier!r} supersedes an unavailable committed "
                        f"version of {label!r}: {previousID!r}."
                    )

                previous = recordsByID[previousID]
                if candidate.get("kind") != previous.get("kind"):
                    raise ValueError("Product versions must retain the same semantic kind.")
                if previousID not in candidate.get("inputProductIDs", []):
                    raise ValueError("A superseded version must be recorded as an input product.")
                if previousID in successors:
                    raise ValueError(
                        f"Product {label!r} has competing replacement versions. "
                        "Select an explicit productID."
                    )

                predecessors[identifier] = previousID
                successors[previousID] = identifier

            # A current version is one that no other committed version replaces.
            currentIDs = set(recordsByID) - set(successors)
            if len(currentIDs) != 1:
                identities = ", ".join(recordsByID)
                raise ValueError(
                    f"Product {label!r} is ambiguous or has a cyclic version chain. "
                    f"Select an explicit productID: {identities}"
                )

            currentID = next(iter(currentIDs))
            visited = set()
            identifier = currentID

            while identifier is not None:
                if identifier in visited:
                    raise ValueError(f"Product {label!r} has a cyclic version chain.")
                visited.add(identifier)
                identifier = predecessors.get(identifier)

            # Reject unrelated duplicate names and disconnected version cycles.
            if visited != set(recordsByID):
                raise ValueError(
                    f"Product {label!r} contains unrelated or disconnected versions. "
                    "Select an explicit productID."
                )

            record = recordsByID[currentID]

        # Kind is validation, never a hidden tiebreaker between duplicate names.
        if self.kind is not None and record.get("kind") != self.kind:
            raise ValueError(
                f"Product {label!r} has kind {record.get('kind')!r}; "
                f"expected {self.kind!r}."
            )

        return pglDerivedProduct(record, owner=owner)

    def __repr__(self):
        selection = f"name={self.productName!r}" if self.productName is not None else f"productID={self.productID!r}"
        return f"pglProductSelector({selection}, kind={self.kind!r})"


################################################################
# Recorded derived-product view
################################################################
class pglDerivedProduct:
    """Independent product metadata with a weak runtime owner reference.

    Persisted provenance remains in the owning object's action history.
    This view does not serialize its owner or keep the session alive.

    The view remains pinned to its original product ID and record, not
    whichever product later happens to have the same logical name.

    A remembered sessionID is only a candidate checkpoint for reloading.
    It does not establish that unsaved working products exist there.

    Payload access remains through session.getDerivedFile(), including its
    integrity checks and deferred storage retrieval.
    """
    # Runtime display adapters are not part of serialized product metadata.
    _displayAdapters = {}
    
    def __init__(self, record, *, owner=None):
        self._record = deepcopy(record)
        self._ownerReference = None
        self._lastSessionID = None
        if owner is not None: self._bindOwner(owner)

    @property
    def productID(self): return self._record["productID"]

    @property
    def name(self): return self._record["name"]

    @property
    def kind(self): return self._record["kind"]

    @property
    def producerActionID(self): return self._record["producerActionID"]

    @property
    def metadata(self): return deepcopy(self._record["metadata"])

    @property
    def artifacts(self): return deepcopy(self._record["artifacts"])

    @property
    def record(self): return deepcopy(self._record)

    ################################################################
    # Bind runtime access without retaining the owning session
    ################################################################
    def _bindOwner(self, owner):
        """Bind a weak reference and refresh the last-known checkpoint ID."""
        import weakref

        if not isinstance(owner, pglActionable):
            raise TypeError("Product owner must carry PGL action history.")

        self._ownerReference = weakref.ref(owner)
        self._rememberCheckpoint(owner)

    def _rememberCheckpoint(self, owner):
        """Remember runtime checkpoint identity without claiming membership."""
        identifier = getattr(owner, "sessionID", None)
        if type(identifier) is int and 0 <= identifier <= 9223372036854775807:
            self._lastSessionID = identifier

    @property
    def lastKnownSessionID(self):
        """Return the last observed checkpoint ID, not verified product storage.

        Refreshes from a live owner. If that owner was saved and released
        without this view being accessed again, the remembered ID may be older.
        """
        owner = self._ownerReference() if self._ownerReference is not None else None
        if owner is not None: self._rememberCheckpoint(owner)
        return self._lastSessionID

    ################################################################
    # Resolve runtime ownership
    ################################################################
    def _requireOwner(self, owner=None):
        """Return an explicit or live owner, without automatically loading one."""
        if owner is None and self._ownerReference is not None:
            owner = self._ownerReference()

        if owner is not None:
            if not isinstance(owner, pglActionable):
                raise TypeError("Product owner must carry PGL action history.")
            self._rememberCheckpoint(owner)
            return owner

        if self._lastSessionID is None:
            raise RuntimeError(
                "The owning session is no longer available or was not bound. "
                "No checkpoint ID is known. Retrieve this product from a live "
                "session; unsaved results cannot be restored from the database."
            )

        raise RuntimeError(
            "The owning session is no longer available. "
            f"Its last known checkpoint was session {self._lastSessionID}.\n"
            "To explicitly reload and verify this product, use:\n"
            "    session = product.reloadSession()\n"
            "Or load the checkpoint directly with:\n"
            f"    session = pglDatabase.loadSession({self._lastSessionID})\n"
            "The product may be absent if it was created after that checkpoint."
        )

    ################################################################
    # Verify the exact product/producer relationship
    ################################################################
    def _findProducingAction(self, owner):
        """Resolve provenance by execution and product identity, never by name."""
        matches = [
            entry for entry in owner.actionHistory
            if entry.actionID == self.producerActionID
        ]
        if len(matches) != 1:
            raise ValueError("Producing execution is missing or ambiguous in this object's history.")

        entry = matches[0]
        records = [
            record for record in entry.actionProducts
            if record.get("productID") == self.productID
        ]
        if len(records) != 1 or records[0] != self._record:
            raise ValueError("Product metadata does not match the producing execution's record.")

        return entry

    ################################################################
    # Find the recorded producing execution
    ################################################################
    def getProducingAction(self, owner=None):
        """Return the producing history entry using the bound owner by default.

        An explicit owner remains supported for compatibility.
        Does not reconstruct an action or execute scientific code.

        Missing ownership or inconsistent provenance warns and returns None.
        """
        try:
            owner = self._requireOwner(owner)
            return self._findProducingAction(owner)

        except Exception as error:
            pglMessages.warning(f"Product producer could not be resolved: {error}")
            return None

    ################################################################
    # Resolve a declared payload through its existing attachment
    ################################################################
    def getFile(self, session=None, role=None, *, cacheDirectory=None):
        """Return a verified local artifact path using the bound session.

        Preferred:
            product.getFile("matrices")
            product.getFile(role="matrices")

        Existing explicit-session calls remain supported:
            product.getFile(session, "matrices")

        Checks producer correspondence and attachment identity before
        delegating byte verification/retrieval to getDerivedFile().

        Never reloads a session, regenerates a payload, or substitutes another
        product automatically. Ordinary failures warn and return None.
        """
        try:
            # Preserve the previous (session, role) interface while allowing
            # the normal single-argument call to specify only the artifact role.
            if isinstance(session, str) and role is None:
                role, session = session, None

            if not isinstance(role, str) or not role.strip():
                raise ValueError("Supply a nonblank artifact role.")

            session = self._requireOwner(session)
            self._findProducingAction(session)

            if role not in self._record["artifacts"]:
                raise ValueError(f"Product {self.name!r} has no artifact role {role!r}.")

            artifact = self._record["artifacts"][role]
            attachment = session.derivedFiles.get(artifact["attachmentName"])
            if attachment is None:
                raise ValueError("The product payload is not attached; it may have been discarded.")

            for field in ("sha256", "size_bytes"):
                if attachment.get(field) != artifact[field]:
                    raise ValueError("The attachment identity differs from the recorded product.")

            path = session.getDerivedFile(
                artifact["attachmentName"],
                cacheDirectory=cacheDirectory,
            )
            if path is None: raise ValueError("The registered product payload is unavailable.")
            return path

        except Exception as error:
            pglMessages.warning(f"Product file could not be resolved: {error}")
            return None

    ################################################################
    # Explicitly reload and verify the owning checkpoint
    ################################################################
    def reloadSession(self, sessionID=None, **loadOptions):
        """Load a checkpoint and rebind only if this exact product is present.

        sessionID:
            None uses lastKnownSessionID.
            An explicit ID allows selecting a later checkpoint that contains
            the product when the remembered checkpoint predates its creation.

        loadOptions:
            Forwarded to pglDatabase.loadSession(), for example settingsName,
            cacheDirectory, or storageOptionsByBackend.

        Returns the loaded session after exact product-record verification.
        The caller must retain it because this view holds only a weak reference:

            session = product.reloadSession(settingsName="desktop")

        Missing products, changed records, and ordinary load failures warn and
        return None without replacing the existing owner reference.

        Does not search checkpoint ancestry, select a same-named replacement,
        rerun estimation, or materialize the product payload.
        """
        try:
            identifier = self.lastKnownSessionID if sessionID is None else sessionID
            if type(identifier) is not int or not 0 <= identifier <= 9223372036854775807:
                raise ValueError("No usable checkpoint ID is known. Supply sessionID explicitly.")

            from .pglDatabase import pglDatabase

            session = pglDatabase.loadSession(identifier, **loadOptions)
            if session is None:
                pglMessages.warning(f"Product session was not reloaded: checkpoint {identifier} could not be loaded.")
                return None

            # Use the raising metadata resolver to avoid duplicate warnings.
            # Product names are not sufficiently precise for rebinding.
            try:
                restored = pglProductSelector(productID=self.productID)._resolve(session)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Checkpoint {identifier} does not expose this exact committed "
                    f"product ({self.productID}). It may have been created after "
                    "that checkpoint or never saved. No replacement was selected."
                ) from error

            if restored.record != self._record:
                raise ValueError(
                    f"Checkpoint {identifier} contains a different record for "
                    "this product ID. Refusing to rebind."
                )

            self._findProducingAction(session)
            self._bindOwner(session)

            # The requested checkpoint ID remains known even if a loader or
            # test double does not populate session.sessionID.
            self._lastSessionID = identifier
            return session

        except Exception as error:
            pglMessages.warning(f"Product session was not reloaded: {type(error).__name__}: {error}")
            return None
        
    ################################################################
    # Display a recorded scientific product
    ################################################################
    def display(self, **options):
        """Display a saved product using registered, action-specific, or default code.

        Precedence:
            Explicit runtime kind adapter.
            Installed producing action's displayProduct class method.
            Base action display (existing motion adapters or generic saved images).

        Resolves classes through the existing trusted action resolver. Saved source
        text is never executed and no action instance is constructed.

        Legacy history without class identity uses the generic/base implementation.
        """
        import inspect

        try:
            owner = self._requireOwner()
            entry = self._findProducingAction(owner)
            adapter = type(self)._displayAdapters.get(self.kind)

            if adapter is None:
                actionClass = pglAction
                code = entry.actionCode
                if isinstance(code, dict) and code.get("module") and code.get("qualname"):
                    resolved = pglPipeline._resolveActionClass(code["module"], code["qualname"])
                    if resolved is not None:
                        actionClass = resolved
                    else:
                        pglMessages.warning("Producing action class unavailable; using the default product display.", level=1)
                adapter = actionClass.displayProduct

            try:
                inspect.signature(adapter).bind(self, **options)
            except TypeError as error:
                raise ValueError(f"Invalid display options for {self.kind!r}: {error}") from error

            return adapter(self, **options)

        except Exception as error:
            pglMessages.warning(f"Product display stopped: {type(error).__name__}: {error}")
            return None
        
        
    ################################################################
    # Print
    ################################################################
    def print(self, verbose=False):
        """Print the producing action and readable product information.

        Uses the owner's history line when the owner is still available.
        Reads recorded metadata only; never retrieves scientific payloads.

        verbose=True adds provenance IDs, attachment names, and hashes.
        """
        if type(verbose) is not bool: raise TypeError("verbose must be Boolean.")

        record = self._record
        metadata = record.get("metadata", {})
        artifacts = record.get("artifacts", {})
        text = pglActionable._productDisplayText

        pglMessages.printHeader(f"Product: {text(self.name)}")

        # Show the same producing-action line used by session.history().
        # Do not automatically reload a released owner merely for printing.
        owner = self._ownerReference() if self._ownerReference is not None else None

        if owner is None:
            pglMessages.print("Producing action: history unavailable because the owning object is no longer available.")
        else:
            try:
                entry = self._findProducingAction(owner)
                indices = [index for index, candidate in enumerate(owner.actionHistory) if candidate is entry]
                if len(indices) != 1: raise ValueError("Producing history entry is ambiguous.")
                pglMessages.print(owner._historyActionLine(indices[0], entry))
            except (ValueError, KeyError) as error:
                pglMessages.print(f"Producing action: could not verify history — {text(error)}")

        pglMessages.print()

        information = {
            "Name": self.name,
            "Kind": self.kind,
        }

        # Put useful scientific descriptions before technical provenance.
        usefulKeys = (
            "backend", "operation", "description", "summary",
            "nScans", "nRuns", "nVolumes", "nMatrices", "nTransforms",
            "units", "coordinateSystem",
        )
        for key in usefulKeys:
            if key in metadata: information[key] = metadata[key]

        # Always expose an unsuccessful or unspecified product outcome.
        status = record.get("recordStatus", "unspecified")
        if verbose or status != "committed": information["Status"] = status

        information["Artifact files"] = len(artifacts)
        if artifacts: information["File roles"] = ", ".join(artifacts)

        if verbose:
            information["Product ID"] = self.productID
            information["Producer action ID"] = self.producerActionID

            for key, value in metadata.items():
                if key not in usefulKeys: information[f"metadata.{key}"] = value

            if record.get("inputProductIDs"): information["Input product IDs"] = record["inputProductIDs"]
            if record.get("invocationIDs"): information["Invocation IDs"] = record["invocationIDs"]

        labelWidth = max(len(label) for label in information)

        for label, value in information.items():
            pglMessages.print(f"    {label:<{labelWidth}} : {text(value, 240)}")

        # Attachment bookkeeping belongs only in the explicit technical view.
        if verbose and artifacts:
            pglMessages.print()
            pglMessages.printHeader("Artifact details")
            roleWidth = max(len(text(role)) for role in artifacts)

            for role, artifact in artifacts.items():
                size = artifact.get("size_bytes")
                sizeText = f"{size:,} bytes" if type(size) is int else "size not recorded"
                pglMessages.print(f"    {text(role):<{roleWidth}}  {sizeText}")
                pglMessages.print(f"        Attachment: {text(artifact.get('attachmentName'))}")
                pglMessages.print(f"        SHA-256: {text(artifact.get('sha256'))}")
        
    ################################################################
    # Register a display implementation for a scientific product kind
    ################################################################
    @classmethod
    def registerDisplayAdapter(cls, kind, adapter, *, overwrite=False):
        """Register a trusted runtime display function for a product kind.

        The adapter is called as:
            adapter(product, **options)

        The product carries its weak owner reference. The adapter handles
        kind-specific validation, verified payload access, and presentation.

        Built-in displays for rigidMotion and runAlignment need no explicit
        registration. Replacing either requires overwrite=True, as does
        replacing an existing custom registration.

        Registration does not execute the adapter or access scientific data.
        Functions are runtime configuration, never serialized into products.

        Returns True on registration.
        Invalid registrations warn at the default level and return False.
        """
        import inspect

        try:
            if (
                not isinstance(kind, str)
                or not kind.strip()
                or any(character in kind for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("Product kind must be a nonblank string without NUL or newlines.")

            if not callable(adapter):
                raise TypeError("Display adapter must be callable.")

            if type(overwrite) is not bool:
                raise TypeError("overwrite must be Boolean.")

            existing = cls._displayAdapters
            if (kind in existing or kind in ("rigidMotion", "runAlignment")) and not overwrite:
                raise ValueError(
                    f"A display adapter already exists for {kind!r}; "
                    "use overwrite=True to replace it."
                )

            # Verify that the callable exposes an inspectable signature.
            # Its actual options are checked by display() before invocation.
            inspect.signature(adapter)

            cls._displayAdapters = {**existing, kind: adapter}
            return True

        except Exception as error:
            pglMessages.warning(
                f"Display adapter was not registered: "
                f"{type(error).__name__}: {error}"
            )
            return False

    def displayArtifacts(self, role=None, *, cacheDirectory=None, show=True):
        """Display saved raster images through verified artifact retrieval.

        With role=None, displays all supported image artifacts.
        Explicit role selects one image. Unsupported files remain retrievable
        through getFile(); they are never executed or unpickled for display.

        Returns Matplotlib figures. show=False leaves presentation to the caller.
        """
        from pathlib import Path
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        if type(show) is not bool: raise TypeError("show must be Boolean.")
        owner = self._requireOwner()
        self._findProducingAction(owner)
        artifacts = self._record.get("artifacts", {})
        labels = {
            item["role"]: item["relativePath"]
            for item in self.metadata.get("supportArtifacts", [])
            if isinstance(item, dict) and "role" in item and "relativePath" in item
        }
        imageTypes = {"image/png", "image/jpeg", "image/tiff", "image/bmp"}
        imageExtensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        if role is not None and (not isinstance(role, str) or role not in artifacts):
            raise ValueError(f"Unknown image artifact role: {role!r}.")

        selected = []
        for artifactRole, artifact in artifacts.items():
            if role is not None and artifactRole != role: continue
            attachmentName = artifact.get("attachmentName", "")
            attachment = owner.derivedFiles.get(attachmentName) or {}
            mediaType = artifact.get("mediaType") or attachment.get("mediaType") or ""
            mediaType = str(mediaType).split(";", 1)[0].strip().lower()
            label = labels.get(artifactRole, artifactRole)

            # Recorded media type is authoritative when specific; filename fallback
            # supports attachments with missing or generic binary media types.
            extension = Path(label).suffix.lower() or Path(attachmentName).suffix.lower()
            supported = mediaType in imageTypes or (
                mediaType in ("", "application/octet-stream") and extension in imageExtensions
            )
            if supported: selected.append((artifactRole, label))
            elif role is not None: raise ValueError(f"{role!r} is not a supported raster image; use getFile().")

        if not selected:
            pglMessages.message(f"Product {self.name!r} has no supported saved image artifacts.")
            return []

        figures = []
        for artifactRole, label in selected:
            figure = None
            try:
                path = self.getFile(role=artifactRole, cacheDirectory=cacheDirectory)
                if path is None: raise ValueError("Image artifact could not be retrieved.")
                image = mpimg.imread(path)
                height, width = image.shape[:2]
                figure, axis = plt.subplots(figsize=(10, max(2, min(14, 10 * height / width))))
                axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
                axis.set_axis_off()
                axis.set_title(f"{self.name} — {Path(label).name}")
                figure.tight_layout()
                if show: plt.show(block=False)
                figures.append(figure)
            except Exception as error:
                if figure is not None: plt.close(figure)
                pglMessages.warning(f"Could not display {label!r}: {type(error).__name__}: {error}", level=1)

        return figures
 
########################
# class pglAction
########################
class pglAction(pglActionable):
    name = Unicode("", help="Name of action")
    status = Instance(pglActionStatus, help="action status")
    error = Instance(Exception, allow_none=True, default_value=None, help="error")
    keepIntermediates = Bool(False, help="Keep superseded MRI image attachments after this action. False releases unpinned intermediates for scans replaced by this execution; files and existing checkpoints are not deleted.")
    # settings for the action, required to be a pglTraitSettings. Subclass should override this
    #settings = Instance(pglTraitSettings, allow_none=True, default_value=None, help='Settings for this action')
    version = Unicode("", help='Verion of pglAction')
    
    # init action
    #-----------------
    def __init__(self, **kwargs):
        """Initialize action state and apply explicitly supplied configuration.

        Empty construction does not call configure(), because subclasses may
        perform interactive selection there. This also keeps construction
        suitable for metadata deserialization.
        """
        super().__init__()

        self.name = type(self).__name__
        self.status = pglActionStatus.INITIALIZED
        self.version = "0.0"
        self.error = None

        # Explicit constructor configuration follows the same path as an
        # ordinary configure() call, including subclass-specific behavior.
        if kwargs:
            self.configure(**kwargs)
            
    # configure
    ################################
    def configure(self, **kwargs) -> None:

        # set traitlets
        self.configureTraits(**kwargs)     
        
        # set status
        self.status = pglActionStatus.CONFIGURED

    @classmethod
    def execute(cls, *args, **kwargs):
        """Route configuration and execution arguments with explicit validation.

        Scientific inputs come from _run() for ordinary actions and
        _estimate() for deferred actions.

        Names accepted by both configuration and execution are passed to both,
        preserving existing behavior for settings such as deferApply.
        """
        action = cls()

        configureNames = cls._getKeywordNames(action.configure) | set(action.traits())
        runNames = action._getRunKeywordNames()
        validNames = configureNames | runNames

        if args:
            pglMessages.warning(
                f"{cls.__name__}.execute() requires keyword arguments, but received "
                f"{len(args)} positional argument(s). Pass arguments as name=value."
            )
            action.printConfiguration()
            pglMessages.print(f"Accepted execute arguments: {', '.join(sorted(validNames)) or '(none)'}")
            return None

        unknownNames = set(kwargs) - validNames
        if unknownNames:
            pglMessages.warning(
                f"{cls.__name__}.execute() got unknown argument(s): "
                f"{', '.join(sorted(unknownNames))}. Argument names are case-sensitive."
            )
            action.printConfiguration()
            pglMessages.print(f"Accepted execute arguments: {', '.join(sorted(validNames)) or '(none)'}")
            return None

        configureArgs = {name: value for name, value in kwargs.items() if name in configureNames}
        runArgs = {name: value for name, value in kwargs.items() if name in runNames}

        action.configure(**configureArgs)
        return action.run(**runArgs)

    @staticmethod
    def _getKeywordNames(method):
        """Return explicitly declared parameters accepting keyword arguments."""
        return {
            name
            for name, parameter in inspect.signature(method).parameters.items()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
                
    def configureTraits(self, **kwargs) -> None:
        validTraits = self.traits()

        unknownKeys = set(kwargs) - set(validTraits)
        if unknownKeys:
            unknownText = ", ".join(sorted(unknownKeys))
            errorMessage = f"{type(self).__name__}.configure() got unknown trait argument(s): {unknownText}"
            pglMessages.warning(errorMessage)
            raise TypeError(errorMessage)

        for name, value in kwargs.items():
            self.set_trait(name, value)
        
    def isConfigured(self):
        '''
        check whether the action is configured or not
        '''
        return self.status.value >= pglActionStatus.CONFIGURED.value
        
    @actionProgress
    def run(self, *args, **kwargs):
        """Execute with provenance, live console output, and saved transcripts.

        Python stdout/stderr are displayed live and flushed to separate local
        files throughout execution, finalization, retention, and presentation.

        Closed transcripts are recorded in actionLogs and attached to history
        recipients supporting derived files. External-command logs retain their
        existing independent capture and archival mechanism.

        Successful actions attach history to their returned actionable.
        Failed actions attach history to directly supplied actionables and any
        actionable result returned before failure. Failure marks these objects
        unsafe for checkpoint saving; later success does not clear that state.

        Nested actions have separate recorders and transcripts. Child console
        output also appears in the parent's transcript. Concurrent action
        capture on different threads is rejected.

        Ordinary scientific exceptions warn and return None. Interruptions are
        recorded and propagated. Tracebacks are printed live and saved.

        Log setup failure prevents scientific execution. Archival failures
        warn without replacing the scientific outcome. Files remain local for
        diagnosis; partial capture is identified explicitly.

        Scientific mutations are not rolled back. Python capture does not
        intercept native file-descriptor writes or rich notebook display().
        """
        import traceback

        from .pglExternalTools import pglExecutionRecorder

        if getattr(self, "_executionInProgress", False):
            pglMessages.warning(
                f"Action {self.name} is already running. Use a separate action instance for another execution."
            )
            return None

        if not self.isConfigured():
            pglMessages.warning(f"Action {self.name} is not yet configured.")
            return None

        inputActionables = []

        # Inspect direct arguments only. Discovering recipients must not load
        # scientific fields, tasks, arrays, or deferred storage.
        for value in (*args, *kwargs.values()):
            if isinstance(value, pglActionable) and value is not self and all(value is not existing for existing in inputActionables):
                inputActionables.append(value)

        previousStatus = self.status
        previousError = self.error
        self._executionInProgress = True

        entry = None
        result = None
        contextToken = None
        logRecipients = []
        capture = pglActionConsoleCapture(getattr(self, "cacheDirectory", None))

        def appendEntry(owner):
            """Preserve external logs and attach this exact history object."""
            if owner is not self and all(owner is not existing for existing in logRecipients):
                logRecipients.append(owner)

                try:
                    entry._preserveLogs(owner, cacheDirectory=getattr(self, "cacheDirectory", None))
                except Exception as error:
                    pglMessages.warning(
                        f"Could not preserve all external-command logs for action {self.name}: {error}",
                        level=1,
                    )

            # Settings equality can compare UUIDs. History attachment requires
            # object identity, not equality with a different execution snapshot.
            if owner is not self and all(existing is not entry for existing in owner.actionHistory):
                owner.actionHistory.append(entry)

        try:
            with capture:
                print(f"Action console logs: {capture.directory}", flush=True)

                self.error = None
                self.status = pglActionStatus.RUNNING

                try:
                    # Capture input bindings before scientific execution.
                    # No settings/default-database lookup is performed.
                    inputRecord = pglActionHistory.captureActionInputs(args, kwargs, replayInput=getattr(self, "_pipelineExecutionInput", {}))
                    entry = pglActionHistory(self)
                    entry.actionInput = inputRecord
                    recorder = pglExecutionRecorder(actionID=entry.actionID)
                    entry.actionEnvironment = recorder.exportEnvironment()
                    self.actionHistory.append(entry)

                except Exception as error:
                    self.status = previousStatus
                    self.error = previousError
                    traceback.print_exc()
                    pglMessages.warning(
                        f"Action {self.name} was not started: could not capture execution provenance: {error}"
                    )
                    return None

                except BaseException:
                    self.status = previousStatus
                    self.error = previousError
                    traceback.print_exc()
                    raise

                contextToken = _activeActionID.set(entry.actionID)

                with recorder.activate():
                    try:
                        result = self._run(*args, **kwargs)
                        recorder.requireComplete()

                        if self.status == pglActionStatus.FAILED or self.error is not None:
                            self.status = pglActionStatus.FAILED

                            if self.error is None:
                                self.error = RuntimeError(
                                    f"Action {self.name} reported failure without an error description."
                                )

                        elif self.status == pglActionStatus.SKIPPED:
                            pass

                        elif result is None:
                            self.error = RuntimeError(
                                f"Action {self.name} returned None without reporting failure or an explicit skip."
                            )
                            self.status = pglActionStatus.FAILED
                            pglMessages.warning(str(self.error))

                        else:
                            self.status = pglActionStatus.SUCCESS

                    except (KeyboardInterrupt, SystemExit) as error:
                        self.error = RuntimeError(
                            f"{type(error).__name__}: action execution was interrupted."
                        )
                        self.status = pglActionStatus.FAILED
                        traceback.print_exc()
                        raise

                    except Exception as error:
                        self.error = error
                        self.status = pglActionStatus.FAILED
                        traceback.print_exc()
                        pglMessages.warning(f"Error running action {self.name}: {error}")

                    finally:
                        # Finalize tool provenance and product commitment using
                        # the original runner's scientific outcome rules.
                        recorder.close()
                        entry.actionEnvironment = recorder.exportEnvironment()
                        entry.actionInvocations = recorder.exportInvocations()

                        if self.status == pglActionStatus.FAILED:
                            for owner in inputActionables:
                                owner._recordFailedAction(entry.actionID)

                            if isinstance(result, pglActionable) and result is not self:
                                result._recordFailedAction(entry.actionID)

                        entry.update(self)

                        if self.status == pglActionStatus.SUCCESS and isinstance(result, pglActionable):
                            for owner in inputActionables:
                                for failedActionID in owner._failedActionIDs:
                                    result._recordFailedAction(failedActionID)

                            appendEntry(result)

                        else:
                            for owner in inputActionables:
                                appendEntry(owner)

                            if isinstance(result, pglActionable):
                                appendEntry(result)

                        # Preserve external-command logs even without a session.
                        if not logRecipients:
                            try:
                                entry._preserveLogs(self, cacheDirectory=getattr(self, "cacheDirectory", None))
                            except Exception as error:
                                pglMessages.warning(f"Could not preserve action logs: {error}", level=1)

                if self.status == pglActionStatus.FAILED:
                    return None

                if self.status == pglActionStatus.SUCCESS:
                    if not isinstance(result, pglActionable):
                        pglMessages.warning(
                            f"Action {self.name} returned type {type(result).__name__}; "
                            "execution history is retained on the action and directly supplied input actionables.",
                            level=1,
                        )

                    # Preserve the existing retention lifecycle. This runs
                    # outside this execution's recorder, but inside its tee.
                    self._finalizeImageRetention(result)

                    try:
                        self._afterSuccess(result)
                    except Exception as error:
                        traceback.print_exc()
                        pglMessages.warning(
                            f"Action {self.name} succeeded, but presenting its results failed: "
                            f"{type(error).__name__}: {error}",
                            level=1,
                        )

                return result

        finally:
            # The capture context has exited: original streams are restored
            # and files are closed before identities are calculated.
            try:
                if entry is not None and capture.paths:
                    owners = []

                    for owner in [self, *inputActionables, *logRecipients, result]:
                        if not isinstance(owner, pglActionable): continue
                        if any(owner is existing for existing in owners): continue
                        if not any(existing is entry for existing in owner.actionHistory): continue
                        owners.append(owner)

                    try:
                        entry._preserveConsoleLogs(capture, owners)
                    except Exception as error:
                        pglMessages.warning(
                            f"Could not attach Python console logs for action {self.name}: "
                            f"{type(error).__name__}: {error}. "
                            f"Local transcripts remain at {capture.directory}.",
                            level=1,
                        )

            finally:
                if contextToken is not None:
                    _activeActionID.reset(contextToken)

                self._executionInProgress = False
    
    def _run(self):
        '''
        Subclass overrides this function to implement the action. 
        '''
        pass
    
    def setError(self, errorString: str | None = None, e: Exception | None = None) -> None:
        # set error status
        self.status = pglActionStatus.FAILED
        
        # store the exception
        if errorString:self.error = Exception(errorString)
        if e: self.error = e
        
        # warn if both are set
        if errorString and e:
            pglMessages.warning("Both errorString and exception passed, only keeping exception", level=1)
            
        # print the error message
        pglMessages.warning(f"{self.error}",callerNameDepth=2)

    def print(self, verbose=False):
        """Print action status and optionally its configuration parameters."""
        pglMessages.print(f"Action: {self.name} status: {self.status.name}")
        if verbose:
            self.printConfiguration()
   
    def printConfiguration(self, frameworkTraits=False):
        """Print configuration traits, configure arguments, and run arguments.

        Deferred-action run arguments include scientific inputs declared by
        _estimate(), not the internal _apply() hook.
        """
        traits = self.traits()
        baseTraitNames = set(pglAction.class_traits())
        configureParameters = inspect.signature(self.configure).parameters
        runNames = self._getRunKeywordNames()

        pglMessages.printHeader("Configuration parameters")

        # Show action-specific traits, excluding framework bookkeeping.
        configurationTraits = {name: trait for name, trait in traits.items() if name not in baseTraitNames}
        for name, trait in sorted(configurationTraits.items()):
            pglMessages.print(f"{name} = {getattr(self, name)!r} | {trait.help}")

        # Also show explicitly declared configure arguments.
        explicitNames = self._getKeywordNames(self.configure)
        for name in sorted(explicitNames - set(configurationTraits)):
            parameter = configureParameters[name]
            valueText = "required" if parameter.default is inspect.Parameter.empty else repr(parameter.default)
            pglMessages.print(f"{name} = {valueText}")

        if not configurationTraits and not explicitNames:
            pglMessages.print("(No action-specific configuration parameters)")

        if runNames:
            pglMessages.printHeader("Run arguments")
            for name in sorted(runNames):
                pglMessages.print(name)

        # These remain accepted by the existing configureTraits implementation.
        if frameworkTraits:
            frameworkNames = set(traits) & baseTraitNames
            if frameworkNames:
                pglMessages.printHeader("Framework traits")
                for name in sorted(frameworkNames):
                    pglMessages.print(f"{name}: {traits[name].help}")
    
    def _captureHistorySettings(self):
        """Capture ordinary action configuration independently of runtime state.

        Pipelines override this to record their prepared execution recipe.
        Capturing ordinary defaults here is intentional provenance behavior.
        """
        frameworkTraits = set(pglAction.class_traits())
        configurationNames = [name for name, trait in self.traits().items() if name not in frameworkTraits and trait.metadata.get("serialize", True)]
        return deepcopy({name: getattr(self, name) for name in sorted(configurationNames)})

    ################################################################
    # Discover public execution arguments
    ################################################################
    def _getRunKeywordNames(self):
        """Return explicitly declared execution arguments.

        Ordinary actions declare scientific inputs in _run().
        Deferred actions declare them in _estimate(); their _run() method
        coordinates preparation and application through generic forwarding.

        _apply() is an internal hook, not an additional public input interface.
        """
        methods = [self.run, self._run]

        if isinstance(self, pglDeferredAction):
            methods.append(self._estimate)

        return set().union(*(self._getKeywordNames(method) for method in methods))
    
    ################################################################
    # Register this execution's named scientific product
    ################################################################
    def registerProduct(self, name, kind, *, session, files=None, inputProductIDs=None, metadata=None, invocationIDs=None, supersedesProductID=None, productID=None):
        """Register an output using already attached session payloads.

        Called inside scientific execution after output validation/attachment.

        files maps artifact roles to session.derivedFiles attachment names.
        Input file identities are copied from registered attachment records.
        This method does not rehash files.

        supersedesProductID:
            Optional exact ID of the current committed version of this name.
            Registers a replacement without modifying the historical product.
            The replacement becomes current only if this action succeeds.

        A replacement must retain the previous product's name and kind.
        Its predecessor is automatically recorded as an exact dependency.

        productID:
            Optional preallocated canonical UUID for complete metadata/payload
            preparation before registration. None generates a fresh UUID.
            Must not occur in destination history, including failed attempts.

        Named product supersession is distinct from an MRI image replacement.
        Image replacement associations are declared explicitly in metadata.

        This method does not merge scientific records or request confirmation.
        The owning scientific action must prepare the complete replacement.
        """
        from uuid import UUID, uuid4

        from .pglExternalTools import getExecutionRecorder

        recorder = getExecutionRecorder()

        if (
            not getattr(self, "_executionInProgress", False)
            or recorder is None
            or not self.actionHistory
            or recorder.actionID != self.actionHistory[-1].actionID
        ):
            raise RuntimeError("registerProduct() must be called by its owning action during execution.")

        if not isinstance(session, pglActionable) or not hasattr(session, "derivedFiles"):
            raise TypeError("Product payloads require a session with derivedFiles.")

        # Allocate before validation so the same identity is checked against
        # history and passed unchanged to the execution recorder.
        if productID is None:
            productID = str(uuid4())
        else:
            if not isinstance(productID, str):
                raise TypeError("productID must be a canonical UUID string or None.")
            try:
                canonicalID = str(UUID(productID))
            except (ValueError, AttributeError):
                raise ValueError("productID must be a canonical UUID string.") from None
            if canonicalID != productID:
                raise ValueError("productID must use canonical lowercase UUID formatting.")

        if any(
            record.get("productID") == productID
            for entry in session.actionHistory
            for record in entry.actionProducts
        ):
            raise ValueError("productID already occurs in destination history.")

        # Include previous attempts made by a reused action instance.
        if any(
            record.get("productID") == productID
            for entry in self.actionHistory
            for record in entry.actionProducts
        ):
            raise ValueError("productID already occurs in this action's history.")

        files = {} if files is None else files
        metadata = {} if metadata is None else deepcopy(metadata)
        inputProductIDs = [] if inputProductIDs is None else deepcopy(inputProductIDs)

        if type(files) is not dict:
            raise TypeError("files must map artifact roles to attachment names.")
        if type(metadata) is not dict:
            raise TypeError("metadata must be a dictionary.")
        if type(inputProductIDs) is not list:
            raise TypeError("inputProductIDs must be a list.")
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in inputProductIDs):
            raise ValueError("Input product identities must be nonblank strings.")
        if len(set(inputProductIDs)) != len(inputProductIDs):
            raise ValueError("Input product identities must not contain duplicates.")
        if productID in inputProductIDs:
            raise ValueError("A product cannot identify itself as an upstream input.")

        # Revision links are framework-owned. Callers supply the predecessor
        # through supersedesProductID, not through arbitrary metadata.
        if "_productRevision" in metadata:
            raise ValueError("Use supersedesProductID rather than supplying _productRevision metadata.")

        existing = [
            record
            for entry in session.actionHistory
            if entry.actionStatus == pglActionStatus.SUCCESS
            for record in entry.actionProducts
            if record.get("recordStatus") == "committed"
            and record.get("name") == name
        ]

        previousSelector = None

        if supersedesProductID is None:
            if existing:
                raise ValueError(
                    f"Product {name!r} already exists. Explicitly supersede its "
                    "current version or choose another outputName."
                )

        else:
            if not isinstance(supersedesProductID, str) or not supersedesProductID.strip():
                raise ValueError("supersedesProductID must be a nonblank string or None.")

            # Resolve the current named version, not just any historical record.
            current = pglProductSelector(name, kind=kind)._resolve(session)
            if current.productID != supersedesProductID:
                raise ValueError(
                    f"Product {name!r} has changed or the supplied predecessor is "
                    "not its current committed version."
                )

            previousSelector = pglProductSelector(productID=supersedesProductID, kind=kind)
            metadata["_productRevision"] = {
                "schemaVersion": 1,
                "supersedesProductID": supersedesProductID,
            }
            if supersedesProductID not in inputProductIDs:
                inputProductIDs.append(supersedesProductID)

        artifacts = {}

        for role, attachmentName in files.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Artifact roles must be nonblank strings.")
            if not isinstance(attachmentName, str) or not attachmentName.strip():
                raise ValueError("Artifact names must be nonblank strings.")

            attachment = session.derivedFiles.get(attachmentName)
            if attachment is None:
                raise ValueError(f"Product attachment {attachmentName!r} is not registered.")

            artifacts[role] = {
                "attachmentName": attachmentName,
                "sha256": attachment["sha256"],
                "size_bytes": attachment["size_bytes"],
            }

        # Record the exact predecessor as consumed, preserving dependency
        # provenance even if later scientific execution fails.
        if previousSelector is not None: previousSelector.resolve(session)

        return recorder.registerProduct(
            name,
            kind,
            artifacts=artifacts,
            inputProductIDs=inputProductIDs,
            metadata=metadata,
            invocationIDs=invocationIDs,
            productID=productID,
        )
        
    ################################################################
    # Optional presentation after successful execution
    ################################################################
    def _afterSuccess(self, result):
        """Perform optional presentation after history/products are committed.

        Called only when scientific execution completed successfully.
        Ordinary actions need not override this hook.

        Use for plots or presentation of already-validated results.
        Do not perform scientific processing, register products, mutate
        scientific contents, or launch recorded tool commands here.

        Scientific validation belongs inside _run()/_estimate()/_apply().
        Ordinary hook errors warn at level=1 without changing scientific
        SUCCESS. KeyboardInterrupt/SystemExit still propagate.

        The hook's return value is ignored.
        """
        pass
    
    def _finalizeImageRetention(self, result):
        """Release eligible image attachments after successful history commit.

        Separate from _afterSuccess(), which is presentation-only.

        Acts only when this execution committed image replacement declarations.
        Limits cleanup to acquisitions replaced by THIS execution. Historical
        stages of other scans are protected for this call.

        keepIntermediates=True skips automatic release. Persistent stage pins
        always remain protected by the MRI retention implementation.

        Scientific success is not revoked by an ordinary retention failure.
        Validation failures leave attachments unchanged and produce a warning.
        Unexpected observer errors during assignment may occur after attachment
        replacement; no general rollback of observer side effects is claimed.

        Does not delete files, close readers, mutate historical products, or
        alter saved checkpoints. Nested execution defers cleanup rather than
        releasing data while an enclosing action remains active.
        """
        from .pglExternalTools import getExecutionRecorder
        from .pglSession import pglSession

        if self.status != pglActionStatus.SUCCESS or self.keepIntermediates:
            return
        if not self.actionHistory:
            return

        entry = self.actionHistory[-1]
        replacementProducts = [
            record
            for record in entry.actionProducts
            if record.get("recordStatus") == "committed"
            and "imageReplacements" in record.get("metadata", {})
        ]
        if not replacementProducts:
            return

        try:
            if not isinstance(result, pglSession) or result.mri is None:
                raise ValueError("Image replacement actions must return their owning MRI session.")

            if all(existing is not entry for existing in result.actionHistory):
                raise ValueError("The producing execution has not been attached to the returned session.")

            # An enclosing recorder means this is a nested action. Its caller
            # may still need older arrays or may fail; do not clean up early.
            if getExecutionRecorder() is not None:
                pglMessages.warning(
                    "Automatic image retention was deferred because an enclosing "
                    "action is still executing. Intermediates remain attached.",
                    level=1,
                )
                return

            mri = result.mri
            if mri._getSession() is not result:
                raise ValueError("MRI ownership differs from the returned session.")

            # Reuse complete committed-declaration validation before reading
            # stage associations or deciding which attachments may be released.
            stages, bindings = mri._getCommittedImageStages()
            productIDs = {record["productID"] for record in replacementProducts}
            affectedScans = {
                stage["scanUUID"]
                for stage in stages
                if bindings[stage["stageID"]]["productID"] in productIDs
            }

            if not affectedScans:
                raise ValueError("Committed replacement products declare no resolvable image stages.")

            # The public retention operation spans the MRI collection. Protect
            # attached historical stages belonging to unaffected acquisitions.
            # Already-released stages must not be passed as requested keep IDs.
            keepStageIDs = [
                stage["stageID"]
                for stage in stages
                if stage["scanUUID"] not in affectedScans
                and stage["matrixName"] in mri.dataMatrices
            ]

            released = mri.releaseImageIntermediates(keepStageIDs=keepStageIDs)
            count = len(released["releaseMatrixNames"])

            if count:
                statusOutput(
                    f"Released {count} superseded MRI image attachment(s). "
                    "Pinned stages, current images, cache files, and saved checkpoints are preserved."
                )

        except Exception as error:
            pglMessages.warning(
                f"Action {self.name} succeeded, but automatic image retention "
                f"could not complete: {type(error).__name__}: {error}. "
                "Inspect MRI attachments before saving; no payload files were deleted.",
                level=1,
            )

    @classmethod
    def displayProduct(cls, product, **options):
        """Default saved-product presentation; subclasses may extend with super().

        Uses installed display code, never executes saved source, and does not
        instantiate/configure actions or run scientific processing.
        """
        if product.kind in ("rigidMotion", "runAlignment") and "role" not in options:
            from .pglMRI import pglMRITransforms
            adapter = {
                "rigidMotion": pglMRITransforms.displayMotionProduct,
                "runAlignment": pglMRITransforms.displayRunAlignmentProduct,
            }[product.kind]
            return adapter(product, **options)

        return product.displayArtifacts(**options)

########################
# class pglDeferredAction
########################
class pglDeferredAction(pglAction):
    """Action with distinct preparation and application phases.

    Subclasses implement:
        _estimate(...):
            Prepare or estimate operations, register their products, and
            return the scientific working object.

        _apply(prepared):
            Apply this execution's explicitly identified operations and
            return the resulting scientific working object.

    Subclasses should not override _run(): it coordinates these phases.

    Default behavior estimates and applies.
    deferApply=True estimates/registers without application.

    Deferral is a persistent action-specific setting, so pipeline copies,
    serialization, and history extraction retain it through existing code.

    A run-time override is captured as the effective configuration for that
    execution, then the action's configured default is restored.

    No MRI-specific behavior is implemented here. Subclasses are responsible
    for explicit product dependencies, geometry validation, and output
    publication. Application must never search for arbitrary pending products.

    This base does not implement a public apply() operation. Applying products
    later belongs to a separate recorded action with explicit inputs.
    """

    deferApply = Bool(False, help="Prepare operations without applying them; False prepares and applies.")

    ################################################################
    # Run using the effective application policy
    ################################################################
    def run(self, *args, deferApply=None, **kwargs):
        """Prepare and optionally apply, using the ordinary action runner.

        deferApply=None uses the configured trait value.
        An explicit Boolean overrides it for this execution only.

        The effective value is present before history captures actionSettings.
        Restoring the configured default afterward does not modify history.

        Invalid overrides warn and return None before scientific execution.
        Ordinary scientific exceptions are handled by pglAction.run().
        KeyboardInterrupt and SystemExit propagate after bookkeeping.
        """
        if getattr(self, "_executionInProgress", False):
            pglMessages.warning(
                f"Action {self.name} is already running. Use another action instance."
            )
            return None

        if deferApply is not None and type(deferApply) is not bool:
            pglMessages.warning("deferApply must be True, False, or None.")
            return None

        configuredDefault = self.deferApply
        self.__dict__.pop("_executionApplication", None)

        try:
            if deferApply is not None:
                self.deferApply = deferApply

            return super().run(*args, **kwargs)

        finally:
            # The history entry has its own configuration/outcome snapshots.
            # Runtime phase state must not leak into the next attempt.
            self.__dict__.pop("_executionApplication", None)
            self.deferApply = configuredDefault

    ################################################################
    # Coordinate preparation and application
    ################################################################
    def _run(self, *args, **kwargs):
        """Run the two phases without creating nested action executions."""
        self._executionApplication = {
            "schemaVersion": 1,
            "deferApply": self.deferApply,
            "preparationStatus": "running",
            "applicationStatus": "notStarted",
        }

        try:
            prepared = self._estimate(*args, **kwargs)
        except BaseException:
            self._executionApplication["preparationStatus"] = "failed"
            raise

        if self.status == pglActionStatus.FAILED or self.error is not None:
            self._executionApplication["preparationStatus"] = "failed"
            return prepared

        if self.status == pglActionStatus.SKIPPED:
            self._executionApplication["preparationStatus"] = "skipped"
            return prepared

        if not isinstance(prepared, pglActionable) or isinstance(prepared, pglAction):
            self._executionApplication["preparationStatus"] = "failed"
            raise TypeError("_estimate() must return a scientific pglActionable, not an action or None.")

        self._executionApplication["preparationStatus"] = "success"

        if self.deferApply:
            self._executionApplication["applicationStatus"] = "pending"
            return prepared

        self._executionApplication["applicationStatus"] = "running"

        try:
            applied = self._apply(prepared)
        except BaseException:
            self._executionApplication["applicationStatus"] = "failed"
            raise

        if self.status == pglActionStatus.FAILED or self.error is not None:
            self._executionApplication["applicationStatus"] = "failed"
            return applied

        if self.status == pglActionStatus.SKIPPED:
            self._executionApplication["applicationStatus"] = "skipped"
            return applied

        if not isinstance(applied, pglActionable) or isinstance(applied, pglAction):
            self._executionApplication["applicationStatus"] = "failed"
            raise TypeError("_apply() must return a scientific pglActionable, not an action or None.")

        self._executionApplication["applicationStatus"] = "applied"
        return applied

    ################################################################
    # Scientific preparation hook
    ################################################################
    def _estimate(self, *args, **kwargs):
        """Subclasses estimate/register operations and return a working object."""
        raise NotImplementedError(f"{type(self).__name__} must implement _estimate().")

    ################################################################
    # Scientific application hook
    ################################################################
    def _apply(self, prepared):
        """Subclasses apply explicitly identified operations to prepared."""
        raise NotImplementedError(f"{type(self).__name__} must implement _apply().")
                    
class pglActionCollection:
    """Container for nested action classes, with discoverable help."""
    @classmethod
    def help(cls, *args, **kwargs):
        """Alias for print(), forwarding all arguments unchanged."""
        return cls.print(*args, **kwargs)
    
    @classmethod
    def print(cls, actionName=None, verbose=False, alphabetical=False):
        """Print grouped actions, or full information for one named action."""
        import inspect

        if actionName is not None and not isinstance(actionName, str):
            pglMessages.warning("actionName must be a string or None.", level=2)
            return

        frameworkTraits = set(pglAction.class_traits())

        # Collect inherited definitions first, then apply subclass overrides.
        members = {}
        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if name in members:
                    del members[name]
                members[name] = value

        # Header markers divide subsequent actions into groups.
        groups = []
        groupHeader = None
        groupActions = []

        for name, value in members.items():
            if name.startswith("_actionHeader_") and isinstance(value, str):
                if groupActions:
                    groups.append((groupHeader, groupActions))
                groupHeader = value
                groupActions = []
            elif inspect.isclass(value) and issubclass(value, pglAction) and value is not pglAction:
                groupActions.append((name, value))

        if groupActions:
            groups.append((groupHeader, groupActions))

        # Selecting an action always enables full information.
        if actionName is not None:
            availableNames = [name for _, actions in groups for name, _ in actions]
            if actionName not in availableNames:
                pglMessages.warning(f"No action named {actionName!r} in {cls.__name__}. Available actions: {', '.join(availableNames) or '(none)'}.", level=2)
                return

            groups = [(header, [(name, actionClass) for name, actionClass in actions if name == actionName]) for header, actions in groups]
            groups = [(header, actions) for header, actions in groups if actions]
            verbose = True

        if not groups:
            pglMessages.print("No actions defined.")
            return

        for groupHeader, actions in groups:
            if groupHeader is not None:
                pglMessages.printHeader(groupHeader)

            if alphabetical:
                actions = sorted(actions, key=lambda item: item[0])

            for actionIndex, (name, actionClass) in enumerate(actions):
                documentation = inspect.cleandoc(actionClass.__doc__ or "")
                description = " ".join(documentation.split("\n\n", 1)[0].split()) or "No description available."

                if verbose and actionIndex:
                    pglMessages.print()

                if actionName is not None:
                    pglMessages.print(f"{name}:")
                    for line in (documentation or "No description available.").splitlines():
                        pglMessages.print(f"  {line}")
                else:
                    pglMessages.print(f"{name}: {description}")

                if not verbose:
                    continue

                configurationTraits = {traitName: trait for traitName, trait in actionClass.class_traits().items() if traitName not in frameworkTraits}

                if configurationTraits:
                    pglMessages.print("  Configuration traits:")
                    for traitName, trait in sorted(configurationTraits.items()):
                        defaultText = repr(trait.default_value)
                        helpText = trait.help or "No help available."
                        pglMessages.print(f"    {traitName} ({type(trait).__name__}, default={defaultText}): {helpText}")

                hasArguments = False
                seenRunArguments = set()

                for methodName in ("configure", "run", "_estimate" if issubclass(actionClass, pglDeferredAction) else "_run"):
                    method = getattr(actionClass, methodName)
                    arguments = []

                    for parameterName, parameter in inspect.signature(method).parameters.items():
                        if parameterName in ("self", "cls"):
                            continue
                        if parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                            continue

                        if methodName in ("run", "_run"):
                            if parameterName in seenRunArguments:
                                continue
                            seenRunArguments.add(parameterName)

                        valueText = "required" if parameter.default is inspect.Parameter.empty else f"default={parameter.default!r}"
                        annotationText = "" if parameter.annotation is inspect.Parameter.empty else f", type={inspect.formatannotation(parameter.annotation)}"
                        positionalText = ", positional-only" if parameter.kind is inspect.Parameter.POSITIONAL_ONLY else ""
                        arguments.append(f"    {parameterName}: {valueText}{annotationText}{positionalText}")

                    if arguments:
                        hasArguments = True
                        heading = "Configure arguments" if methodName == "configure" else "Run arguments"
                        pglMessages.print(f"  {heading}:")
                        for argument in arguments:
                            pglMessages.print(argument)

                if not configurationTraits and not hasArguments:
                    pglMessages.print("  No action-specific parameters.")
                    
########################
# pglPipeline
########################
class pglPipeline(pglAction):
    """An editable recipe containing independently configured actions.

    Step numbers are zero-based current positions.
    Step keys are stable identities retained when steps move or are configured.

    Nested pipelines are single steps unless insertion explicitly requests
    expansion. Editing never runs actions or calls interactive configuration.

    Execution, history extraction, and persistence are connected separately.
    """

    actions = List(Instance(pglAction), help="Ordered configured actions")
    stepKeys = List(Unicode(), help="Stable identities corresponding to actions")
    description = Unicode(default_value=None, allow_none=True, help="Description of this recipe")
    pipelineID = Int(default_value=None, allow_none=True, help="Saved pipeline checkpoint ID").tag(serialize=False)
    stepSources = Dict(default_value={}, help="Original execution identity and source snapshot keyed by recipe step key")
    replayInput = Dict(default_value={}, help="Optional storage-managed checkpoint input and history boundary for default replay")
    
    def __init__(self, actions=None, *, name=None, description=None):
        """Construct a recipe, copying supplied actions without execution."""
        super().__init__()

        if name is not None:
            self._validatePipelineName(name)
            self.name = name

        if description is not None and (not isinstance(description, str) or "\x00" in description):
            raise ValueError("description must be a string without NUL, or None.")

        self.description = description

        if actions is not None:
            if not isinstance(actions, (list, tuple)):
                raise TypeError("actions must be a list or tuple of configured actions.")

            # Build the entire collection before installing anything.
            copies = [self._copyConfiguredAction(action) for action in actions]
            self.actions = copies
            self.stepKeys = [str(uuid4()) for action in copies]

        self.status = pglActionStatus.CONFIGURED

    @staticmethod
    def _validatePipelineName(name):
        """Require a usable display name."""
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            raise ValueError("Pipeline name must be a nonempty string without NUL.")

    @staticmethod
    def _configurationNames(action):
        """Return persistent action-specific configuration trait names.

        Framework bookkeeping and explicitly nonserialized runtime traits are
        not scientific configuration. Dynamic configuration defaults are
        deliberately captured when constructing a configured recipe.
        """
        frameworkNames = set(pglAction.class_traits())
        return sorted(name for name, trait in action.traits().items() if name not in frameworkNames and trait.metadata.get("serialize", True))

    @classmethod
    def _copyConfiguredAction(cls, action, activeObjects=None):
        """Copy configuration into a fresh action without interactive configure.

        Nested recipes receive fresh step keys while retaining independently
        copied associations with the historical executions they came from.
        """
        if not isinstance(action, pglAction):
            raise TypeError("Pipeline steps must be pglAction instances.")

        activeObjects = set() if activeObjects is None else activeObjects
        objectID = id(action)

        if objectID in activeObjects:
            raise ValueError("A pipeline cannot contain a recursive action structure.")

        activeObjects.add(objectID)

        try:
            if isinstance(action, pglPipeline):
                if type(action) is not pglPipeline:
                    raise TypeError("Pipeline subclass copying requires an explicit recipe adapter.")

                action._validateStructure()
                copied = pglPipeline(name=action.name, description=action.description)
                copied.version = action.version
                copied.actions = [cls._copyConfiguredAction(child, activeObjects) for child in action.actions]
                copied.stepKeys = [str(uuid4()) for child in copied.actions]
                copied.stepSources = {newKey: deepcopy(action.stepSources[oldKey]) for oldKey, newKey in zip(action.stepKeys, copied.stepKeys) if oldKey in action.stepSources}
                return copied

            if action.status not in (pglActionStatus.CONFIGURED, pglActionStatus.SUCCESS, pglActionStatus.FAILED, pglActionStatus.SKIPPED):
                raise ValueError(f"Action {action.name!r} must be configured and not running before it can be inserted.")

            settings = deepcopy({name: getattr(action, name) for name in cls._configurationNames(action)})
            copied = type(action)()
            copied.configureTraits(**settings)
            copied.name = action.name
            copied.version = action.version
            copied.error = None
            copied.actionHistory = []
            copied._failedActionIDs = []
            copied._checkpointBaseline = {}
            copied.status = pglActionStatus.CONFIGURED
            return copied

        finally:
            activeObjects.remove(objectID)

    def _validateStructure(self):
        """Validate recipe bookkeeping without inspecting scientific payloads."""
        if len(self.actions) != len(self.stepKeys):
            raise ValueError("Pipeline action/step-key counts differ. Use the editing methods rather than directly changing the action list.")

        if any(not isinstance(key, str) or not key for key in self.stepKeys):
            raise ValueError("Pipeline step keys must be nonempty strings.")

        if len(set(self.stepKeys)) != len(self.stepKeys):
            raise ValueError("Pipeline step keys must be unique within a pipeline.")

        if set(self.stepSources) - set(self.stepKeys):
            raise ValueError("Pipeline source metadata refers to nonexistent steps.")
    
    def _requireEditable(self):
        """Do not edit a recipe while that pipeline is executing."""
        if getattr(self, "_executionInProgress", False):
            raise RuntimeError("Cannot edit a pipeline while it is running.")

        self._validateStructure()

    def _resolveStep(self, selector):
        """Resolve a current position or unique exact action name."""
        self._validateStructure()

        if type(selector) is int:
            if not 0 <= selector < len(self.actions):
                raise IndexError(f"Pipeline step {selector} is out of range; this pipeline has {len(self.actions)} steps.")

            return selector

        if not isinstance(selector, str):
            raise TypeError("An action selector must be a zero-based integer or an action name.")

        matches = [index for index, action in enumerate(self.actions) if action.name == selector]

        if not matches:
            available = ", ".join(f"[{index}] {action.name}" for index, action in enumerate(self.actions))
            raise ValueError(f"No action named {selector!r}. Available steps: {available or '(none)'}.")

        if len(matches) > 1:
            available = ", ".join(f"[{index}] {self.actions[index].name}" for index in matches)
            raise ValueError(f"Action name {selector!r} is ambiguous: {available}. Use the displayed step number.")

        return matches[0]

    def get(self, selector):
        """Return a step belonging to this working pipeline.

        Nested edits through this object change this pipeline's nested recipe,
        not the original pipeline from which it was copied.
        """
        return self.actions[self._resolveStep(selector)]

    def configure(self, selector=None, **kwargs):
        """Edit one step's configuration without executing custom configure().

        pipeline.configure() marks this recipe configured.
        pipeline.configure(selector, **settings) updates a selected step.

        Only declared persistent action-specific traits can be changed here.
        Interactive configure arguments are not replayable configuration;
        their resulting trait values must be edited instead.

        Configure nested children through pipeline.get(selector).configure(...).
        """
        self._requireEditable()

        if selector is None:
            if kwargs:
                raise TypeError("Specify an action number or name when editing step configuration.")

            self.status = pglActionStatus.CONFIGURED
            return self

        index = self._resolveStep(selector)
        action = self.actions[index]

        if isinstance(action, pglPipeline):
            raise TypeError("Configure a nested pipeline's steps through pipeline.get(selector).configure(childSelector, **settings).")

        validNames = set(self._configurationNames(action))
        unknownNames = set(kwargs) - validNames

        if unknownNames:
            raise TypeError(f"Unknown or nonpersistent configuration for {action.name!r}: {', '.join(sorted(unknownNames))}. Available settings: {', '.join(sorted(validNames)) or '(none)'}.")

        if not kwargs:
            return self

        # Validate on an independent candidate. A trait validation failure must
        # not leave half the requested settings installed on the current step.
        candidate = self._copyConfiguredAction(action)
        candidate.configureTraits(**deepcopy(kwargs))
        candidate.status = pglActionStatus.CONFIGURED

        actions = list(self.actions)
        actions[index] = candidate
        self.actions = actions

        # The configured step keeps its identity and position.
        self.status = pglActionStatus.CONFIGURED
        return self

    def insert(self, action, *, before=None, after=None, expand=False):
        """Insert independent configured copies, retaining source associations.

        Nested pipelines remain one step unless expand=True.
        Expansion copies immediate children only.
        """
        self._requireEditable()

        if type(expand) is not bool:
            raise ValueError("expand must be a boolean.")

        if before is not None and after is not None:
            raise ValueError("Supply before or after, not both.")

        if action is self:
            raise ValueError("A pipeline cannot insert itself.")

        if before is not None:
            destination = self._resolveStep(before)
        elif after is not None:
            destination = self._resolveStep(after) + 1
        else:
            destination = len(self.actions)

        if expand:
            if not isinstance(action, pglPipeline):
                raise TypeError("expand=True requires a pipeline.")

            action._validateStructure()
            sources = list(action.actions)
        else:
            sources = [action]

        copies = [self._copyConfiguredAction(source) for source in sources]

        if not copies:
            return self

        keys = [str(uuid4()) for copied in copies]
        sourceMetadata = dict(self.stepSources)

        if expand:
            for oldKey, newKey in zip(action.stepKeys, keys):
                if oldKey in action.stepSources:
                    sourceMetadata[newKey] = deepcopy(action.stepSources[oldKey])

        self.actions = [*self.actions[:destination], *copies, *self.actions[destination:]]
        self.stepKeys = [*self.stepKeys[:destination], *keys, *self.stepKeys[destination:]]
        self.stepSources = sourceMetadata
        self.status = pglActionStatus.CONFIGURED
        return self

    def delete(self, selector):
        """Remove a working-recipe step and its source association."""
        self._requireEditable()
        index = self._resolveStep(selector)
        removedKey = self.stepKeys[index]

        self.actions = [*self.actions[:index], *self.actions[index + 1:]]
        self.stepKeys = [*self.stepKeys[:index], *self.stepKeys[index + 1:]]
        self.stepSources = {key: value for key, value in self.stepSources.items() if key != removedKey}
        self.status = pglActionStatus.CONFIGURED
        return self

    def move(self, selector, *, before=None, after=None):
        """Move a step while retaining its identity and configured object.

        Resolve both selectors against the original list before removing
        anything. Moving relative to itself is a no-op.
        """
        self._requireEditable()

        if (before is None) == (after is None):
            raise ValueError("Supply exactly one of before or after.")

        source = self._resolveStep(selector)
        target = self._resolveStep(before if before is not None else after)

        if source == target:
            return self

        pairs = list(zip(self.actions, self.stepKeys))
        moving = pairs.pop(source)

        # Removing a preceding source shifts the original target left.
        targetAfterRemoval = target - 1 if source < target else target
        destination = targetAfterRemoval if before is not None else targetAfterRemoval + 1
        pairs.insert(destination, moving)

        self.actions = [action for action, key in pairs]
        self.stepKeys = [key for action, key in pairs]
        self.status = pglActionStatus.CONFIGURED
        return self

    def print(self, verbose=False):
        """Print the numbered recipe, including nested pipelines.

        Numbers are local to each pipeline level.
        Verbose output includes settings and stable step keys.
        """
        self._validateStructure()
        pglMessages.printHeader("Pipeline")
        pglMessages.print(f"Name: {self.name}")

        if self.pipelineID is not None:
            pglMessages.print(f"Saved pipeline ID: {self.pipelineID} (working recipe may contain edits)")

        if self.description:
            pglMessages.print(f"Description: {self.description}")

        pglMessages.print(f"Steps: {len(self.actions)}")

        def printSteps(pipeline, indent, activeObjects):
            objectID = id(pipeline)

            if objectID in activeObjects:
                raise ValueError("Cannot print a recursive pipeline structure.")

            activeObjects.add(objectID)

            try:
                pipeline._validateStructure()

                if not pipeline.actions:
                    pglMessages.print(f"{indent}(no steps)")

                for index, action in enumerate(pipeline.actions):
                    nested = isinstance(action, pglPipeline)
                    suffix = " [pipeline]" if nested else ""
                    pglMessages.print(f"{indent}[{index}] {action.name}{suffix}")

                    if verbose:
                        pglMessages.print(f"{indent}    stepKey = {pipeline.stepKeys[index]}")

                    if nested:
                        printSteps(action, indent + "    ", activeObjects)
                    elif verbose:
                        names = self._configurationNames(action)

                        if not names:
                            pglMessages.print(f"{indent}    (No configuration parameters)")

                        for name in names:
                            pglMessages.print(f"{indent}    {name} = {getattr(action, name)!r}")

            finally:
                activeObjects.remove(objectID)

        printSteps(self, "  ", set())

    # +++++++++++++++++++++++++++++++ run +++++++++++++++++++++++++++++++++++
    def run(self, *, session=None, sessionID=None, start=None, end=None, allowCodeMismatch=False, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy="materialize"):
        """Run selected recipe steps on a working object or saved checkpoint.

        User-facing pipeline execution coordinator. Prepares an independent
        recipe snapshot, resolves its input, then invokes pglAction.run()
        for execution identity, history recording, and failure bookkeeping.

        Input selection:
            session supplies an existing scientific working object.
            sessionID loads an exact active checkpoint through pglDatabase.
            With neither, use the verified default recorded in replayInput.

        Explicit session/sessionID overrides the default for this execution.
        Only default replay requires the loaded checkpoint identity and
        history to match the previously recorded replay input.

        start/end select an inclusive range of top-level steps using the
        existing step selectors. Recipe and selection validation occur before
        checkpoint loading. Nested pipelines must receive a working object
        explicitly rather than load a new checkpoint.

        Database loading options apply only to checkpoint inputs.
        Resolving a checkpoint reads metadata and establishes lazy payload
        backings; selected actions may subsequently load scientific arrays.

        Returns:
            The resulting scientific actionable on success.
            None if input loading returns no session or action execution fails
            through the ordinary action runner.

        Preparation, input-validation, and source-mismatch errors propagate.
        Scientific mutations are not rolled back.
        No session or dataset checkpoint is saved automatically.

        Usage:
            result = pipeline.run(sessionID=42, settingsName="desktop")
            result = pipeline.run(session=workingSession)
            result = pipeline.run()
        """
        from .pglDatabase import pglDatabase

        if getattr(self, "_executionInProgress", False) or getattr(self, "_pipelineExecutionPrepared", False):
            raise RuntimeError("This pipeline is already preparing or executing a run.")

        if session is not None and sessionID is not None:
            raise ValueError("Supply session or sessionID, not both.")

        if type(allowCodeMismatch) is not bool:
            raise ValueError("allowCodeMismatch must be a boolean.")

        first, last = self._resolveRunRange(start, end)
        expectedInput = None

        if session is None and sessionID is None:
            if not self.replayInput:
                raise ValueError("This pipeline has no verified default input. Supply session or sessionID explicitly.")

            expectedInput = deepcopy(self.replayInput)
            pglDatabase._validatePipelineReplayInput(expectedInput)
            sessionID = expectedInput["sessionID"]

        if sessionID is not None:
            if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
                raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

            if directLocalPolicy not in ("materialize", "immutable"):
                raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'.")

            if _activeActionID.get() is not None:
                raise ValueError("Nested pipeline execution must receive session explicitly, not load a checkpoint.")

        else:
            if not isinstance(session, pglActionable) or isinstance(session, pglAction):
                raise TypeError("session must be a scientific pglActionable object, not an action.")

            databaseOptions = (settings, settingsName, storageOptionsByBackend, sslRootCert, cacheDirectory)

            if any(value is not None for value in databaseOptions) or directLocalPolicy != "materialize":
                raise ValueError("Database loading options apply only to checkpoint inputs.")

        self._pipelineExecutionPrepared = True

        try:
            recipeJSON = self.toJSON(strict=True)
            executionRecipe = pglPipeline.fromJSON(recipeJSON, strict=True)
            executionRecipe._checkExecutionSources(first, last, allowCodeMismatch=allowCodeMismatch)

            if sessionID is not None:
                pglMessages.message(f"Pipeline {self.name!r}: loading input checkpoint {sessionID}...")
                session, inputRecord = pglDatabase._loadPipelineInput(sessionID, settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

                if session is None:
                    pglMessages.warning(f"Pipeline {self.name!r} was not run: input checkpoint {sessionID} could not be loaded.", level=2)
                    return None

                # Explicit overrides intentionally select another input.
                # Default replay must match the recorded checkpoint boundary.
                pglDatabase._validatePipelineReplayInput(inputRecord, expected=expectedInput)
                inputLabel = f"checkpoint {sessionID}"

            else:
                inputRecord = {"version": 1, "kind": "workingObject"}
                inputLabel = "the supplied working object"

            snapshot = {
                "pipelineExecutionVersion": 2,
                "recipeJSON": recipeJSON,
                "selection": {"start": first, "end": last, "stepKeys": list(executionRecipe.stepKeys[first:last + 1])},
                "allowCodeMismatch": allowCodeMismatch,
                "inputMode": "providedObject",
            }

            self._pipelineExecutionRecipe = executionRecipe
            self._pipelineExecutionSnapshot = snapshot
            self._pipelineExecutionInput = inputRecord
            self.status = pglActionStatus.CONFIGURED

            pglMessages.message(f"Running pipeline {self.name!r}: steps [{first}] through [{last}] on {inputLabel}.")
            return super().run(session=session)

        finally:
            self.__dict__.pop("_pipelineExecutionRecipe", None)
            self.__dict__.pop("_pipelineExecutionSnapshot", None)
            self.__dict__.pop("_pipelineExecutionInput", None)
            self._pipelineExecutionPrepared = False
            
    def _run(self, session):
        """Execute the prepared selection, stopping at the first failed step.

        Called only by the ordinary action runner after run() preparation.
        Each child uses that same runner and records this execution as parent.
        """
        recipe = getattr(self, "_pipelineExecutionRecipe", None)
        snapshot = getattr(self, "_pipelineExecutionSnapshot", None)

        if recipe is None or snapshot is None:
            raise RuntimeError("Pipeline execution resources are missing.")

        selection = snapshot["selection"]
        working = session

        for index in range(selection["start"], selection["end"] + 1):
            action = recipe.actions[index]
            pglMessages.message(f"Pipeline {self.name!r}: [{index}] {action.name}")

            if isinstance(action, pglPipeline):
                result = action.run(session=working, allowCodeMismatch=snapshot["allowCodeMismatch"])
            else:
                result = action.run(session=working)

            if action.status == pglActionStatus.FAILED:
                self.setError(f"Pipeline step [{index}] {action.name} failed: {action.error}")
                return None

            # Skipped steps have no universal output contract yet. Refuse to
            # assume they left a suitable scientific state for later steps.
            if action.status != pglActionStatus.SUCCESS:
                self.setError(f"Pipeline step [{index}] {action.name} did not complete successfully; status={action.status.name}.")
                return None

            if not isinstance(result, pglActionable) or isinstance(result, pglAction):
                self.setError(f"Pipeline step [{index}] {action.name} did not return a scientific actionable.")
                return None

            working = result

        return working
            
    def save(self, *args, **kwargs):
        """Reserve save() for storage-managed pipeline persistence."""
        raise NotImplementedError("Database pipeline saving is not implemented yet.")

    @classmethod
    def load(cls, *args, **kwargs):
        """Reserve load() for storage-managed pipeline reconstruction."""
        raise NotImplementedError("Database pipeline loading is not implemented yet.")
    
    # ++++++++++++++++++++++++++ _resolveActionClass +++++++++++++++++++++++++
    @staticmethod
    def _resolveActionClass(moduleName, qualname):
        """Resolve a recorded action identity, loading built-in actions on demand.

        Called by recipe restoration, history extraction, and serialization.

        Already-loaded modules:
            Resolve the exact recorded class, including nested action classes.

        Unloaded modules:
            Automatically import immediate pglActions* modules within this
            pipeline module's package. No per-module registry is required.
            External action packages must be explicitly imported by the caller.

        Only ordinary module/class namespaces are traversed. Class aliases,
        mismatched identities, and objects that are not pglAction classes
        are rejected.

        Imports execute installed module initialization code. This policy
        trusts the installed PGL package and Python import environment; it
        is not a sandbox. Saved source text is never executed.

        Does not instantiate actions, configure them, or call scientific run
        methods. Resolution and import errors emit a pglMessages.warning
        and return None. Callers must handle None before instantiating an action.

        Examples:
            # Public usage: required built-in action modules load automatically.
            pipeline = session.getPipeline(name="reprocessing")
            if pipeline is not None:
                pipeline.print(verbose=True)

            # Serialized recipe restoration uses the same resolver.
            restored = pglPipeline.fromJSON(recipeJSON)

            # External actions require an explicit trusted import first.
            import myLab.actions
            restored = pglPipeline.fromJSON(recipeJSON)
        """
        import sys
        from importlib import import_module
        from types import ModuleType

        try:
            if not isinstance(moduleName, str) or not moduleName or not isinstance(qualname, str) or not qualname:
                raise ValueError("Action identity requires a module and qualified class name.")

            moduleParts = moduleName.split(".")
            classParts = qualname.split(".")

            # Validate the complete identity before triggering any import.
            # Locally defined classes containing <locals> are not addressable.
            if any(not part.isidentifier() for part in moduleParts):
                raise ValueError(f"Invalid action module name: {moduleName!r}.")

            if any(not part.isidentifier() for part in classParts):
                raise ValueError(f"Action class is not addressable through its module: {moduleName}.{qualname}")

            current = sys.modules.get(moduleName)

            if current is None:
                parentPackage, _, moduleLeaf = moduleName.rpartition(".")

                # Trust the installed package's existing action-module convention,
                # not arbitrary packages named by serialized history.
                # Exact parent matching excludes similarly named external packages
                # and nested modules outside this convention.
                isBuiltinActionModule = (
                    bool(__package__)
                    and parentPackage == __package__
                    and moduleLeaf.startswith("pglActions")
                )

                if not isBuiltinActionModule:
                    raise ValueError(
                        f"Action module {moduleName!r} is not loaded and is outside "
                        "the built-in action-module namespace. Import the trusted "
                        "action package before loading this pipeline."
                    )

                # Import failures are reported by the warning handler below.
                current = import_module(moduleName)

            if not isinstance(current, ModuleType):
                raise TypeError(f"Action module {moduleName!r} is not a Python module.")

            # Resolve nested collection/action classes without invoking dynamic
            # attribute lookup or falling back to display-name searches.
            for index, part in enumerate(classParts):
                if index and not isinstance(current, type):
                    raise ValueError(f"Action class is not addressable through its module: {moduleName}.{qualname}")

                current = vars(current).get(part)

                if current is None:
                    raise ValueError(f"Action class is unavailable: {moduleName}.{qualname}")

            if not isinstance(current, type) or not issubclass(current, pglAction):
                raise TypeError(f"{moduleName}.{qualname} is not a pglAction class.")

            if current.__module__ != moduleName or current.__qualname__ != qualname:
                raise ValueError(f"Installed action identity does not match {moduleName}.{qualname}.")

            return current

        except Exception as error:
            # Fail gracefully without swallowing KeyboardInterrupt or SystemExit.
            pglMessages.warning(
                f"Could not resolve action {moduleName!r}.{qualname!r}: "
                f"{type(error).__name__}: {error}. Returning None."
            )
            return None

    def _toRecipeRecord(self, activeObjects=None):
        """Select recipe metadata without encoding individual setting values.

        Child actions are represented by class identity and configuration,
        never by their complete runtime objects.

        pglSerialize handles the resulting values recursively.
        """
        if type(self) is not pglPipeline:
            raise TypeError("Pipeline subclasses require an explicit recipe adapter.")

        activeObjects = set() if activeObjects is None else activeObjects
        objectID = id(self)

        if objectID in activeObjects:
            raise ValueError("Cannot serialize a recursive pipeline.")

        if getattr(self, "_executionInProgress", False):
            raise RuntimeError("Cannot serialize a pipeline while it is running.")

        self._validateStructure()
        self._validatePipelineName(self.name)
        activeObjects.add(objectID)

        try:
            steps = []

            for index, (action, stepKey) in enumerate(zip(self.actions, self.stepKeys)):
                path = f"pipeline[{index}]({action.name})"

                if getattr(action, "_executionInProgress", False):
                    raise RuntimeError(f"{path}: cannot serialize a running action.")

                if isinstance(action, pglPipeline):
                    steps.append({"stepKey": stepKey, "stepType": "pipeline", "recipe": action._toRecipeRecord(activeObjects)})
                    continue

                if action.status not in (pglActionStatus.CONFIGURED, pglActionStatus.SUCCESS, pglActionStatus.FAILED, pglActionStatus.SKIPPED):
                    raise ValueError(f"{path}: action must be configured before its recipe can be saved.")

                actionClass = type(action)
                resolvedClass = self._resolveActionClass(actionClass.__module__, actionClass.__qualname__)

                if resolvedClass is not actionClass:
                    raise ValueError(f"{path}: action belongs to a stale or replaced class definition.")

                # Capture configuration defaults, but exclude framework traits,
                # execution history, and explicitly nonpersistent fields.
                settings = {name: getattr(action, name) for name in self._configurationNames(action)}
                steps.append({"stepKey": stepKey, "stepType": "action", "module": actionClass.__module__, "qualname": actionClass.__qualname__, "name": action.name, "version": action.version, "settings": settings})

            return {"recipeVersion": 1, "name": self.name, "description": self.description, "version": self.version, "steps": steps}

        finally:
            activeObjects.remove(objectID)

    @classmethod
    def _fromRecipeRecord(cls, record):
        """Restore a recipe whose setting values were decoded by pglSerialize.

        Construct fresh actions and assign configuration traits directly.
        Do not call interactive configure() methods or scientific _run().

        Strict pglSerialize restoration verifies that the resulting recipe
        re-encodes without changing or dropping serialized metadata.
        """
        expectedFields = {"recipeVersion", "name", "description", "version", "steps"}

        if not isinstance(record, dict) or set(record) != expectedFields:
            raise ValueError("Invalid pipeline recipe fields.")

        if type(record["recipeVersion"]) is not int or record["recipeVersion"] != 1:
            raise ValueError("Unsupported pipeline recipe version.")

        cls._validatePipelineName(record["name"])

        if record["description"] is not None and (not isinstance(record["description"], str) or "\x00" in record["description"]):
            raise ValueError("Invalid pipeline description.")

        if not isinstance(record["version"], str) or not isinstance(record["steps"], list):
            raise ValueError("Invalid pipeline version or step collection.")

        actions = []
        stepKeys = []

        for index, step in enumerate(record["steps"]):
            path = f"pipeline[{index}]"

            if not isinstance(step, dict) or step.get("stepType") not in ("action", "pipeline"):
                raise ValueError(f"{path}: invalid step record.")

            stepKey = step.get("stepKey")

            if not isinstance(stepKey, str) or not stepKey or stepKey in stepKeys:
                raise ValueError(f"{path}: missing or duplicate step key.")

            if step["stepType"] == "pipeline":
                if set(step) != {"stepKey", "stepType", "recipe"}:
                    raise ValueError(f"{path}: invalid nested pipeline fields.")

                action = cls._fromRecipeRecord(step["recipe"])

            else:
                expectedStepFields = {"stepKey", "stepType", "module", "qualname", "name", "version", "settings"}

                if set(step) != expectedStepFields or not isinstance(step["settings"], dict):
                    raise ValueError(f"{path}: invalid action fields.")

                if not isinstance(step["name"], str) or not isinstance(step["version"], str):
                    raise ValueError(f"{path}: invalid action name or version.")

                actionClass = cls._resolveActionClass(step["module"], step["qualname"])

                if issubclass(actionClass, pglPipeline):
                    raise ValueError(f"{path}: pipeline actions must use nested recipe records.")

                # Empty construction avoids the kwargs-driven configure() path.
                # Action constructors must themselves remain lightweight.
                action = actionClass()
                configurationNames = set(cls._configurationNames(action))
                suppliedNames = set(step["settings"])

                if suppliedNames != configurationNames:
                    missing = sorted(configurationNames - suppliedNames)
                    unknown = sorted(suppliedNames - configurationNames)
                    raise ValueError(f"{path}: installed configuration schema differs; missing={missing}, unknown={unknown}.")

                # Assign through trait validation without reopening selectors or
                # invoking custom interactive configure() methods.
                try:
                    action.configureTraits(**deepcopy(step["settings"]))
                except Exception as error:
                    raise ValueError(f"{path}: could not restore action configuration: {error}") from error

                action.name = step["name"]
                action.version = step["version"]
                action.error = None
                action.actionHistory = []
                action._failedActionIDs = []
                action._checkpointBaseline = {}
                action.status = pglActionStatus.CONFIGURED

            actions.append(action)
            stepKeys.append(stepKey)

        restored = pglPipeline(name=record["name"], description=record["description"])
        restored.version = record["version"]

        # Install directly: the public constructor copies inserted actions and
        # assigns new keys, whereas restoration must preserve recorded keys.
        restored.actions = actions
        restored.stepKeys = stepKeys
        restored.status = pglActionStatus.CONFIGURED
        restored._validateStructure()
        return restored
    
    # +++++++++++++++++++++++++++ toJSONdict ++++++++++++++++++++++++++++++++
    def toJSONdict(self, type="all"):
        """Build the recipe document used by pglSerialize.

        Called during recipe serialization and execution-snapshot preparation.
        Includes recipe configuration, historical source associations, and an
        optional top-level checkpoint replay input.

        pglSerialize handles recursive encoding of the returned values.
        Database-specific replay metadata is structurally validated through
        pglDatabase without connecting to PostgreSQL or reading payloads.

        Nested execution receives its parent's working object. Only the
        top-level recipe's replayInput is included in this document.

        Returns a new document without modifying the recipe or its history.
        Structural and serialization-preparation errors propagate.
        """
        from .pglDatabase import pglDatabase

        recipe = self._toRecipeRecord()
        sources = {}

        # +++++++++++++++++++++++++++ collect +++++++++++++++++++++++++++++++
        def collect(pipeline, path):
            """Copy source associations for each nested recipe location."""
            if pipeline.stepSources:
                sources[path] = deepcopy(pipeline.stepSources)

            for index, action in enumerate(pipeline.actions):
                if isinstance(action, pglPipeline):
                    collect(action, f"{path}/{index}")

        collect(self, "root")
        document = {"recipe": recipe}

        if sources:
            document["sourceMetadataVersion"] = 1
            document["stepSourcesByPipeline"] = sources

        if self.replayInput:
            pglDatabase._validatePipelineReplayInput(self.replayInput)
            document["replayInput"] = deepcopy(self.replayInput)

        return document

    # ++++++++++++++++++++++++++ fromJSONdict +++++++++++++++++++++++++++++++
    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore a recipe document and optional checkpoint replay metadata.

        Called by pglSerialize after decoding the document's values.
        Delegates recipe/source restoration to _fromRecipeDocument() and
        checkpoint-input structure validation to pglDatabase.

        Does not load the referenced checkpoint, verify current database
        availability, execute scientific actions, or connect to PostgreSQL.

        Returns a reconstructed pipeline with an independent replay-input
        record. Invalid documents and restoration errors propagate.
        """
        from .pglDatabase import pglDatabase

        if not isinstance(data, dict):
            raise ValueError("Invalid serialized pipeline document.")

        document = dict(data)
        hasReplayInput = "replayInput" in document
        replayInput = document.pop("replayInput", None)

        if hasReplayInput:
            pglDatabase._validatePipelineReplayInput(replayInput)

        restored = cls._fromRecipeDocument(document, type=type, filename=filename)

        if hasReplayInput:
            restored.replayInput = deepcopy(replayInput)

        return restored

    @classmethod
    def _fromRecipeDocument(cls, data, type="all", filename=None):
        """Restore recipe structure and optional historical source associations."""
        if not isinstance(data, dict):
            raise ValueError("Invalid serialized pipeline document.")

        legacy = set(data) == {"recipe"}
        expectedFields = {"recipe", "sourceMetadataVersion", "stepSourcesByPipeline"}

        if not legacy and set(data) != expectedFields:
            raise ValueError("Invalid serialized pipeline fields.")

        if not legacy and (data["sourceMetadataVersion"].__class__ is not int or data["sourceMetadataVersion"] != 1):
            raise ValueError("Unsupported pipeline source metadata version.")

        restored = cls._fromRecipeRecord(data["recipe"])

        if legacy:
            return restored

        sources = data["stepSourcesByPipeline"]

        if not isinstance(sources, dict) or not sources:
            raise ValueError("Pipeline source metadata must be a nonempty mapping.")

        pipelines = {}

        def collect(pipeline, path):
            pipelines[path] = pipeline

            for index, action in enumerate(pipeline.actions):
                if isinstance(action, pglPipeline):
                    collect(action, f"{path}/{index}")

        collect(restored, "root")

        for path, records in sources.items():
            if path not in pipelines or not isinstance(records, dict) or not records:
                raise ValueError(f"Invalid source metadata for pipeline path {path!r}.")

            for key, record in records.items():
                if not isinstance(record, dict) or set(record) != {"historyIndex", "actionID", "actionCode", "actionSettings"}:
                    raise ValueError(f"Invalid source metadata for step {key!r}.")

                if record["historyIndex"].__class__ is not int or record["historyIndex"] < 0:
                    raise ValueError(f"Invalid source history index for step {key!r}.")

                if record["actionID"] is not None and not isinstance(record["actionID"], str):
                    raise ValueError(f"Invalid source execution ID for step {key!r}.")

                if not isinstance(record["actionCode"], dict) or not isinstance(record["actionSettings"], dict):
                    raise ValueError(f"Invalid source configuration or code record for step {key!r}.")

            pipelines[path].stepSources = deepcopy(records)
            pipelines[path]._validateStructure()

        return restored

    def toJSON(self, type="all", filename=None, *, strict=True):
        """Delegate recipe encoding to pglSerialize, strictly by default."""
        return super().toJSON(type=type, filename=filename, strict=strict)

    @classmethod
    def fromJSON(cls, jsonString, filename=None, loadAsClass=None, *, strict=True):
        """Delegate recipe decoding to pglSerialize and verify the result."""
        result = super().fromJSON(jsonString, filename=filename, loadAsClass=loadAsClass, strict=strict)

        if not isinstance(result, cls):
            raise TypeError("Serialized document did not restore a pipeline.")

        return result
    
    # +++++++++++++++++++++++++++ fromHistory +++++++++++++++++++++++++++++++
    @classmethod
    def fromHistory(cls, actionable, *, name=None):
        """Extract an independent recipe and any recorded checkpoint input.

        Called by pglActionable.getPipeline() and the database extraction
        coordinator when no saved checkpoint baseline is available.

        Finds the earliest successful top-level pipeline execution carrying
        explicit checkpoint-input metadata. Verifies the corresponding
        in-memory history prefix against its recorded fingerprints before
        excluding that prefix from the executable recipe.

        Histories without this evidence remain extractable but have no default
        replay input. A checkpoint is never inferred from sessionID alone.

        Uses pglDatabase for checkpoint-input structure validation and the
        checkpoint history encoding contract. Does not query PostgreSQL,
        read archived checkpoint files, or verify current source availability.

        Returns:
            An independent pglPipeline, possibly with replayInput attached.
            None on extraction or provenance-validation failure, with an
            informative pglMessages.warning.

        Does not modify source history, copy scientific payloads, or execute
        actions. KeyboardInterrupt and SystemExit propagate.

        Current database correspondence is checked later when default replay
        loads its checkpoint.
        """
        import hashlib
        import json

        from .pglDatabase import pglDatabase

        pipelineName = "extracted-pipeline" if name is None else name

        try:
            if not isinstance(actionable, pglActionable):
                raise TypeError("Pipeline extraction requires a pglActionable object.")

            history = list(actionable.actionHistory)
            candidates = []

            for index, entry in enumerate(history):
                if entry.parentActionID is not None or entry.actionStatus != pglActionStatus.SUCCESS:
                    continue

                replayInput = pglActionHistory.getRecordedReplayInput(entry.actionInput)
                if replayInput.get("kind") == "checkpoint":
                    candidates.append((index, entry))

            if not candidates:
                pipeline = cls._fromHistoryRecipe(actionable, name=name)

                # The extractor has already emitted the failure warning.
                if pipeline is None:
                    return None

                pglMessages.message("Pipeline extracted without a verified default input; supply session or sessionID when running it.")
                return pipeline

            sourceIndex, sourceEntry = candidates[0]
            inputRecord = pglActionHistory.getRecordedReplayInput(sourceEntry.actionInput)
            pglDatabase._validatePipelineReplayInput(inputRecord)
            boundary = inputRecord["historyCount"]

            if boundary > sourceIndex:
                raise ValueError("Recorded replay boundary lies after the execution that supplied it.")

            # Checkpoint inputs are currently recorded by pipeline executions.
            # Reject another class rather than inferring its replay semantics.
            code = sourceEntry.actionCode

            if not isinstance(code, dict) or not code.get("module") or not code.get("qualname"):
                raise ValueError("Checkpoint-backed execution has no recorded action class identity.")

            actionClass = cls._resolveActionClass(code["module"], code["qualname"])

            if actionClass is None:
                raise ValueError(
                    f"history[{sourceIndex}]({sourceEntry.actionName}): could not "
                    "resolve the checkpoint-backed execution class. "
                    "See the preceding resolver warning for details."
                )

            if actionClass is not pglPipeline:
                raise ValueError("Automatic replay input extraction currently requires a recorded pglPipeline execution.")

            for index, expectedHash in enumerate(inputRecord["historyHashes"]):
                document = json.loads(history[index].toJSON())
                actualHash = hashlib.sha256(pglDatabase.encodeRawManifest(document)).hexdigest()

                if actualHash != expectedHash:
                    raise ValueError(f"History entry {index} does not match the recorded checkpoint input. Cannot safely exclude the checkpoint history prefix.")

            # Use a lightweight history view rather than copying scientific data.
            # The extractor reads historical entries without editing them.
            view = pglActionable()
            view.actionHistory = list(history[boundary:])
            view._failedActionIDs = list(actionable._failedActionIDs)
            pipeline = cls._fromHistoryRecipe(view, name=name)

            # Do not attach replay metadata or traverse a failed extraction.
            # The extractor has already emitted the failure warning.
            if pipeline is None:
                return None

            # ++++++++++++++++++++ restoreHistoryIndices +++++++++++++++++++++++
            def restoreHistoryIndices(recipe):
                """Translate extracted suffix indices back to full-history positions."""
                for record in recipe.stepSources.values():
                    record["historyIndex"] += boundary

                for action in recipe.actions:
                    if isinstance(action, pglPipeline):
                        restoreHistoryIndices(action)

            restoreHistoryIndices(pipeline)
            pipeline.replayInput = inputRecord

            pglMessages.message(f"Pipeline default input: checkpoint {inputRecord['sessionID']}; excluded {boundary} history entries already represented by that checkpoint.")
            return pipeline

        except Exception as error:
            # Do not return a recipe whose replay provenance failed validation.
            pglMessages.warning(
                f"Cannot extract pipeline {pipelineName!r}: "
                f"{type(error).__name__}: {error} "
                "No pipeline was returned; source history was not modified. Returning None.",
                level=2,
            )
            return None
    
    # +++++++++++++++++++++++++ _fromHistoryRecipe ++++++++++++++++++++++++++
    @classmethod
    def _fromHistoryRecipe(cls, actionable, *, name=None):
        """Extract successful execution trees into an independent recipe.

        Pipeline executions become nested recipe steps. Their recorded
        selection determines which steps are retained, and child history
        verifies what actually executed.

        Failed/skipped top-level attempts are omitted with their descendants.
        Missing parents, duplicate IDs, cycles, and inconsistent successful
        execution records are rejected rather than guessed.

        Extraction failures emit pglMessages.warning and return None for the
        entire pipeline. No partial recipe is returned and missing settings
        are never filled from installed defaults.

        Does not modify history, load checkpoints, or establish a replay input.
        KeyboardInterrupt and SystemExit propagate.
        """
        import json
        from types import SimpleNamespace
        from .pglSerialize import pglSerialize

        pipelineName = "extracted-pipeline" if name is None else name

        try:
            if not isinstance(actionable, pglActionable):
                raise TypeError("Pipeline extraction requires a pglActionable object.")

            history = list(actionable.actionHistory)

            if any(entry.actionStatus == pglActionStatus.RUNNING for entry in history):
                raise ValueError("Cannot extract a pipeline while its history contains running actions.")

            cls._validatePipelineName(pipelineName)

            # Index executions before following parent relationships. Children
            # normally appear before their parents because history records completion.
            indexByID = {}

            for index, entry in enumerate(history):
                actionID = entry.actionID

                if actionID is None:
                    continue

                if not isinstance(actionID, str) or not actionID:
                    raise ValueError(f"history[{index}]: invalid execution ID.")

                if actionID in indexByID:
                    raise ValueError(f"history[{index}]: duplicate execution ID {actionID!r}.")

                indexByID[actionID] = index

            childrenByIndex = {index: [] for index in range(len(history))}
            parentByIndex = {}
            roots = []

            for index, entry in enumerate(history):
                parentID = entry.parentActionID

                if parentID is None:
                    roots.append(index)
                    continue

                if not isinstance(parentID, str) or parentID not in indexByID:
                    raise ValueError(f"history[{index}]: parent execution {parentID!r} is missing; cannot reconstruct the execution tree.")

                parentIndex = indexByID[parentID]

                if parentIndex == index:
                    raise ValueError(f"history[{index}]: an execution cannot be its own parent.")

                parentByIndex[index] = parentIndex
                childrenByIndex[parentIndex].append(index)

            # Check all records, including failed branches, before extraction.
            # Do not allow an orphaned cycle to disappear merely because it has no root.
            finished = set()

            for startIndex in range(len(history)):
                current = startIndex
                chain = set()

                while current not in finished:
                    if current in chain:
                        raise ValueError("Action history contains a cycle in execution ancestry.")

                    chain.add(current)

                    if current not in parentByIndex:
                        break

                    current = parentByIndex[current]

                finished.update(chain)

            def metadataText(value):
                """Compare values through the shared strict metadata representation."""
                encoded = pglSerialize.toJSON(SimpleNamespace(value=value), strict=True)
                return json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"), allow_nan=False)

            def sourceRecord(index):
                """Preserve original provenance independently of editable settings."""
                entry = history[index]
                return {"historyIndex": index, "actionID": entry.actionID, "actionCode": deepcopy(entry.actionCode), "actionSettings": deepcopy(entry.actionSettings)}

            def resolveClass(index):
                """Resolve an action class or stop extraction with history context."""
                entry = history[index]
                code = entry.actionCode
                path = f"history[{index}]({entry.actionName})"

                if not isinstance(code, dict) or not code.get("module") or not code.get("qualname"):
                    raise ValueError(f"{path}: this legacy entry has no recorded action class identity. It cannot be reconstructed safely from its display name alone.")

                actionClass = cls._resolveActionClass(code["module"], code["qualname"])

                # The resolver warns and returns None on failure. Do not pass
                # that result to issubclass() or attempt to instantiate it.
                if actionClass is None:
                    raise ValueError(
                        f"{path}: could not resolve recorded action class "
                        f"{code['module']}.{code['qualname']}. "
                        "See the preceding resolver warning for details."
                    )

                return actionClass

            def restoreOrdinaryAction(index, actionClass):
                """Restore recorded traits without interactive configuration."""
                entry = history[index]
                path = f"history[{index}]({entry.actionName})"

                if childrenByIndex[index]:
                    raise ValueError(f"{path}: this ordinary action contains child executions. An explicit extraction adapter is needed; refusing to flatten them.")

                action = actionClass()
                expectedNames = set(cls._configurationNames(action))
                suppliedNames = set(entry.actionSettings)

                if suppliedNames != expectedNames:
                    missing = sorted(expectedNames - suppliedNames)
                    unknown = sorted(suppliedNames - expectedNames)
                    raise ValueError(
                        f"{path}: installed configuration schema differs; "
                        f"missing={missing}, unknown={unknown}. "
                        "'missing' lists settings required by the installed action "
                        "but absent from the recorded configuration; 'unknown' lists "
                        "recorded settings no longer supported by the installed action. "
                        "Use a compatible action implementation or explicitly migrate "
                        "the recipe; defaults will not be substituted."
                    )

                try:
                    action.configureTraits(**deepcopy(entry.actionSettings))
                except Exception as error:
                    raise ValueError(f"{path}: could not restore configuration: {error}") from error

                action.name = entry.actionName
                action.version = entry.actionVersion
                action.status = pglActionStatus.CONFIGURED
                action.error = None
                action.actionHistory = []
                action._failedActionIDs = []
                action._checkpointBaseline = {}

                restoredSettings = {field: getattr(action, field) for field in cls._configurationNames(action)}

                if metadataText(restoredSettings) != metadataText(entry.actionSettings):
                    raise ValueError(f"{path}: installed action changed the recorded configuration during extraction.")

                currentCode = pglActionHistory.captureActionCode(action)
                recordedCode = entry.actionCode

                if recordedCode.get("status") == "captured" and currentCode.get("status") == "captured" and recordedCode.get("sha256") != currentCode.get("sha256"):
                    pglMessages.warning(f"{path}: installed source differs from recorded source. Extraction is allowed for editing; execution still requires an explicit mismatch override.", level=2)

                return action

            def restorePipeline(index):
                """Rebuild only the recorded selection and verify its child runs."""
                entry = history[index]
                path = f"history[{index}]({entry.actionName})"
                snapshot = entry.actionSettings
                expectedFields = {"pipelineExecutionVersion", "recipeJSON", "selection", "allowCodeMismatch", "inputMode"}

                if set(snapshot) != expectedFields or type(snapshot["pipelineExecutionVersion"]) is not int or snapshot["pipelineExecutionVersion"] != 2:
                    raise ValueError(f"{path}: unsupported pipeline execution snapshot; expected version 2 with recipeJSON.")

                if not isinstance(snapshot["recipeJSON"], str) or type(snapshot["allowCodeMismatch"]) is not bool:
                    raise ValueError(f"{path}: malformed pipeline execution snapshot.")

                if snapshot["inputMode"] != "providedObject":
                    raise ValueError(f"{path}: unsupported recorded pipeline input mode.")

                selection = snapshot["selection"]

                if not isinstance(selection, dict) or set(selection) != {"start", "end", "stepKeys"}:
                    raise ValueError(f"{path}: invalid recorded selection.")

                first = selection["start"]
                last = selection["end"]

                try:
                    recorded = pglPipeline.fromJSON(snapshot["recipeJSON"], strict=True)
                except Exception as error:
                    raise ValueError(f"{path}: could not restore the recorded pipeline recipe: {error}") from error

                if recorded is None:
                    raise ValueError(f"{path}: restoration of the recorded pipeline recipe returned None.")

                if type(first) is not int or type(last) is not int or not 0 <= first <= last < len(recorded.actions):
                    raise ValueError(f"{path}: recorded selection is outside the recipe.")

                selectedKeys = recorded.stepKeys[first:last + 1]

                if selection["stepKeys"] != selectedKeys:
                    raise ValueError(f"{path}: selected step identities differ from the recorded recipe.")

                if recorded.name != entry.actionName or recorded.version != entry.actionVersion:
                    raise ValueError(f"{path}: pipeline identity differs from its execution history.")

                childIndices = childrenByIndex[index]

                if len(childIndices) != len(selectedKeys):
                    raise ValueError(f"{path}: successful pipeline has {len(childIndices)} child records but its selection contains {len(selectedKeys)} steps.")

                restoredActions = []
                restoredSources = {}

                # The current runner is sequential. Direct children therefore have
                # the same completion order as their selected recipe steps.
                for offset, childIndex in enumerate(childIndices):
                    childEntry = history[childIndex]
                    expectedAction = recorded.actions[first + offset]
                    stepKey = selectedKeys[offset]

                    if childEntry.actionStatus != pglActionStatus.SUCCESS:
                        raise ValueError(f"{path}: successful pipeline contains a non-successful child at history[{childIndex}].")

                    actualAction = restoreExecution(childIndex)

                    if type(actualAction) is not type(expectedAction) or actualAction.name != expectedAction.name or actualAction.version != expectedAction.version:
                        raise ValueError(f"{path}: child history[{childIndex}] differs from selected step {first + offset}.")

                    if isinstance(actualAction, pglPipeline):
                        # The current runner executes nested pipelines in full.
                        # A partial or changed child recipe must not pass silently.
                        actualSettings = actualAction._toRecipeRecord()
                        expectedSettings = expectedAction._toRecipeRecord()
                    else:
                        actualSettings = {field: getattr(actualAction, field) for field in cls._configurationNames(actualAction)}
                        expectedSettings = {field: getattr(expectedAction, field) for field in cls._configurationNames(expectedAction)}

                    if metadataText(actualSettings) != metadataText(expectedSettings):
                        raise ValueError(f"{path}: child history[{childIndex}] configuration differs from the selected recipe step.")

                    restoredActions.append(actualAction)
                    restoredSources[stepKey] = sourceRecord(childIndex)

                extracted = pglPipeline(name=recorded.name, description=recorded.description)
                extracted.version = recorded.version
                extracted.actions = restoredActions
                extracted.stepKeys = list(selectedKeys)
                extracted.stepSources = restoredSources
                extracted._validateStructure()
                return extracted

            def restoreExecution(index):
                """Dispatch by recorded class identity, never by display name."""
                actionClass = resolveClass(index)

                if issubclass(actionClass, pglPipeline):
                    if actionClass is not pglPipeline:
                        raise TypeError(f"history[{index}]: pipeline subclasses require an explicit extraction adapter.")

                    return restorePipeline(index)

                return restoreOrdinaryAction(index, actionClass)

            actions = []
            stepKeys = []
            stepSources = {}
            omitted = []

            # Only root executions become top-level recipe steps. Descendants are
            # incorporated by restorePipeline(), never appended a second time.
            for index in roots:
                entry = history[index]

                if entry.actionStatus != pglActionStatus.SUCCESS:
                    omitted.append((index, entry.actionName, entry.actionStatus.name))
                    continue

                action = restoreExecution(index)
                stepKey = str(uuid4())
                actions.append(action)
                stepKeys.append(stepKey)
                stepSources[stepKey] = sourceRecord(index)

            pipeline = cls(name=pipelineName)
            pipeline.actions = actions
            pipeline.stepKeys = stepKeys
            pipeline.stepSources = stepSources
            pipeline._validateStructure()

            if omitted:
                details = ", ".join(f"[{index}] {actionName}: {status}" for index, actionName, status in omitted)
                pglMessages.warning(f"Extraction omitted non-successful top-level attempts and their child executions: {details}. Partial effects from those attempts are not reproduced.", level=2)

            if actionable._failedActionIDs:
                pglMessages.warning("The source working object contains unresolved failed attempts. Run the edited recipe on a clean input, not the failed working object.", level=2)

            return pipeline

        except Exception as error:
            # Stop the entire extraction; never return a partially restored recipe.
            # KeyboardInterrupt and SystemExit are intentionally not caught.
            pglMessages.warning(
                f"Cannot extract pipeline {pipelineName!r}: "
                f"{type(error).__name__}: {error} "
                "No pipeline was returned; source history was not modified. Returning None.",
                level=2,
            )
            return None
    
    def _resolveRunRange(self, start=None, end=None):
        """Resolve an inclusive top-level range before any scientific execution."""
        self._validateStructure()

        if not self.actions:
            raise ValueError("Cannot run an empty pipeline.")

        first = 0 if start is None else self._resolveStep(start)
        last = len(self.actions) - 1 if end is None else self._resolveStep(end)

        if first > last:
            raise ValueError(f"Pipeline start [{first}] is after end [{last}].")

        return first, last
    
    def _checkExecutionSources(self, first, last, *, allowCodeMismatch=False, path="pipeline"):
        """Check source expectations for selected steps and nested recipes.

        This compares inspect snapshots, not complete dependency environments.
        Unselected steps are not checked for execution compatibility.
        """
        import hashlib

        self._validateStructure()

        for index in range(first, last + 1):
            action = self.actions[index]
            stepKey = self.stepKeys[index]
            stepPath = f"{path}[{index}]({action.name})"
            source = self.stepSources.get(stepKey)

            if source is not None:
                recorded = source.get("actionCode")

                if not isinstance(recorded, dict):
                    raise ValueError(f"{stepPath}: malformed recorded code metadata.")

                if recorded.get("module") != type(action).__module__ or recorded.get("qualname") != type(action).__qualname__:
                    raise ValueError(f"{stepPath}: action class differs from its historical source association.")

                if recorded.get("status") == "captured":
                    sourceText = recorded.get("source")
                    expectedHash = recorded.get("sha256")

                    if not isinstance(sourceText, str) or not isinstance(expectedHash, str) or hashlib.sha256(sourceText.encode("utf-8")).hexdigest() != expectedHash:
                        raise ValueError(f"{stepPath}: recorded source snapshot has an invalid hash.")

                    installed = pglActionHistory.captureActionCode(action)
                    matches = installed.get("status") == "captured" and installed.get("sha256") == expectedHash

                    if not matches:
                        reason = "installed source differs" if installed.get("status") == "captured" else "installed source cannot be inspected"

                        if not allowCodeMismatch:
                            raise ValueError(f"{stepPath}: {reason}. Review the code, then explicitly use allowCodeMismatch=True if this execution is intentional.")

                        pglMessages.warning(f"{stepPath}: {reason}; proceeding with the recorded allowCodeMismatch=True override.", level=2)

                else:
                    pglMessages.warning(f"{stepPath}: historical source was unavailable; code equivalence cannot be verified.", level=0)

            if isinstance(action, pglPipeline):
                if not action.actions:
                    raise ValueError(f"{stepPath}: nested pipeline is empty.")

                action._checkExecutionSources(0, len(action.actions) - 1, allowCodeMismatch=allowCodeMismatch, path=stepPath)
                
    def _captureHistorySettings(self):
        """Return the prepared recipe and selection for this execution.

        Preparation happens before the ordinary action runner marks this
        pipeline running. No live child action objects enter parent history.
        """
        snapshot = getattr(self, "_pipelineExecutionSnapshot", None)

        if snapshot is None:
            raise RuntimeError("Pipeline execution was not prepared through pipeline.run().")

        return deepcopy(snapshot)

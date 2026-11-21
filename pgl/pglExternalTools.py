################################################################
#   filename: pglExternalTools.py
#    purpose: External-tool execution context and provenance
#         by: JLG
#       date: Nov 15, 2026
################################################################

#############
# Imports
#############
from .pglActionProgress import actionProgress, operationProgress, currentProgress, progressCall, statusOutput, noProgressBar, daskProgress
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import math
import time
from uuid import uuid4


# Track execution ownership without passing a recorder through every call.
# Nested actions temporarily replace this value and then restore their parent.
_activeExecutionRecorder = ContextVar("pglActiveExecutionRecorder", default=None)


################################################################
# Get the current execution recorder
################################################################
def getExecutionRecorder(): return _activeExecutionRecorder.get()


################################################################
# Validate and copy portable provenance
################################################################
def _copyProvenance(value):
    """Return independent JSON-compatible metadata without coercing values.

    Accepts built-in dictionaries, lists, strings, finite numbers, Booleans,
    and None. Rejects cycles, non-string keys, and arbitrary runtime objects.

    Validation happens before records are installed in a recorder.
    """
    active = set()

    def validate(item):
        if item is None or type(item) in (str, bool, int): return

        if type(item) is float:
            if not math.isfinite(item): raise ValueError("Provenance numbers must be finite.")
            return

        if type(item) not in (dict, list):
            raise TypeError(f"Unsupported provenance type: {type(item).__name__}.")

        identity = id(item)
        if identity in active: raise ValueError("Provenance must not contain circular references.")

        active.add(identity)

        try:
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str: raise TypeError("Provenance dictionary keys must be strings.")
                    validate(child)
            else:
                for child in item: validate(child)
        finally:
            active.remove(identity)

    validate(value)
    return deepcopy(value)


################################################################
# Capture baseline runtime information
################################################################
def getRuntimeEnvironment():
    """Capture baseline runtime metadata without launching external tools.

    Reads OS information and installed Python distribution metadata.
    Does not import those distributions, inspect credentials, record the
    hostname/username, dump environment variables, or create files.

    Installed distribution versions are an inventory, not proof of which
    packages were imported or of their exact source contents.

    Tool builds, effective tool environments, containers, GPU libraries,
    and source-control state require additional capture in later adapters.
    Their absence is explicitly recorded rather than inferred.

    Every call takes a fresh snapshot. No process-wide inventory cache hides
    packages installed or changed between executions.
    """
    import importlib.metadata
    import platform
    import struct
    import sys

    environment = {
        "schemaVersion": 1,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "operatingSystem": {},
        "python": {
            "version": sys.version,
            "implementation": sys.implementation.name,
            "cacheTag": sys.implementation.cache_tag,
            "pointerBits": struct.calcsize("P") * 8,
        },
        "packages": {
            "status": "captured",
            "distributions": [],
            "errors": [],
        },
        "notCaptured": [
            "External executable identities and builds",
            "Effective external-tool environment and execution controls",
            "Container image identity",
            "GPU and numerical backend runtime details",
            "Source-control revision and local modifications",
        ],
    }

    # Do not use platform.platform() or record uname.node: the hostname is
    # unnecessary for scientific reproducibility and may identify a machine.
    try:
        system = platform.uname()
        environment["operatingSystem"] = {
            "status": "captured",
            "system": system.system,
            "release": system.release,
            "version": system.version,
            "machine": system.machine,
        }

        if system.system == "Darwin":
            environment["operatingSystem"]["macOSVersion"] = platform.mac_ver()[0]

        if system.system == "Linux":
            try:
                release = platform.freedesktop_os_release()
                environment["operatingSystem"]["distribution"] = {
                    key: release[key]
                    for key in ("ID", "ID_LIKE", "NAME", "VERSION_ID", "VERSION", "PRETTY_NAME")
                    if key in release
                }
            except Exception as error:
                environment["operatingSystem"]["distribution"] = {
                    "status": "unavailable",
                    "errorType": type(error).__name__,
                }

    except Exception as error:
        environment["operatingSystem"] = {
            "status": "unavailable",
            "errorType": type(error).__name__,
        }

    packages = environment["packages"]

    try:
        for distribution in importlib.metadata.distributions():
            try:
                name = distribution.metadata.get("Name")
                version = distribution.version

                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Distribution has no usable name.")
                if not isinstance(version, str) or not version.strip():
                    raise ValueError("Distribution has no usable version.")

                # Retain repeated distribution records rather than silently
                # selecting one of several installations with the same name.
                packages["distributions"].append({"name": name, "version": version})

            except Exception as error:
                packages["errors"].append({
                    "errorType": type(error).__name__,
                    "description": "Could not identify an installed distribution.",
                })

    except Exception as error:
        packages["status"] = "partial" if packages["distributions"] else "unavailable"
        packages["errors"].append({
            "errorType": type(error).__name__,
            "description": "Could not complete the installed distribution inventory.",
        })

    if packages["errors"] and packages["status"] == "captured":
        packages["status"] = "partial"

    packages["distributions"].sort(key=lambda item: (item["name"].casefold(), item["version"]))
    return _copyProvenance(environment)


################################################################
# Execution recorder
################################################################
class pglExecutionRecorder:
    """Collect portable provenance for one action or standalone execution.

    actionID:
        UUID of the owning action execution, or None for standalone use.

    Construction captures baseline environment information unless an explicit
    environment snapshot is supplied.

    activate() installs this recorder only for the enclosed execution context.
    close() freezes recording and marks unfinished invocations incomplete.

    Records are private. Export methods return independent copies so callers
    cannot accidentally mutate execution provenance through returned values.

    This initial implementation records invocation metadata only. It does not
    execute commands, publish files, or implement the artifact registry.

    Concurrent mutation of one recorder is unsupported. Worker threads and
    processes will need explicit execution-context/record-merging support.
    """

    def __init__(self, actionID=None, *, environment=None):
        if actionID is not None:
            if not isinstance(actionID, str) or not actionID.strip():
                raise ValueError("actionID must be a nonblank string or None.")

        snapshot = getRuntimeEnvironment() if environment is None else _copyProvenance(environment)
        if type(snapshot) is not dict: raise TypeError("environment must be a dictionary.")

        self._actionID = actionID
        self._environment = snapshot
        self._invocations = []
        self._openInvocations = {}
        self._active = False
        self._closed = False

    @property
    def actionID(self): return self._actionID

    @property
    def isClosed(self): return self._closed

    ################################################################
    # Activate recording for this execution context
    ################################################################
    @contextmanager
    def activate(self):
        """Activate this recorder and restore the previous one on every exit."""
        self._requireOpen()
        if self._active: raise RuntimeError("This recorder is already active.")

        token = _activeExecutionRecorder.set(self)
        self._active = True

        try:
            yield self
        finally:
            try:
                _activeExecutionRecorder.reset(token)
            finally:
                self._active = False

    def _requireOpen(self):
        if self._closed: raise RuntimeError("This execution recorder is closed.")

    ################################################################
    # Begin an invocation before attempting execution
    ################################################################
    def beginInvocation(self, specification):
        """Record an invocation attempt and return its UUID.

        specification is a JSON-compatible dictionary prepared by a tool
        adapter. It will describe executable identity, parameters, arguments,
        declared inputs/outputs, and relevant effective environment.

        The caller must redact secrets before supplying metadata. This generic
        recorder cannot infer which arbitrary argument values are sensitive.
        """
        self._requireOpen()
        specification = _copyProvenance(specification)
        if type(specification) is not dict: raise TypeError("specification must be a dictionary.")

        invocationID = str(uuid4())
        record = {
            "schemaVersion": 1,
            "invocationID": invocationID,
            "actionID": self._actionID,
            "specification": specification,
            "status": "running",
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "endedAt": None,
            "durationSeconds": None,
            "result": {},
            "error": None,
        }

        self._invocations.append(record)
        self._openInvocations[invocationID] = (record, time.perf_counter())
        return invocationID

    ################################################################
    # Complete an invocation
    ################################################################
    def finishInvocation(self, invocationID, *, status="success", result=None, error=None):
        """Finalize one attempt after execution and output validation.

        status is success, failed, or interrupted.
        result holds portable execution/output details.
        error is an explicitly prepared, JSON-compatible diagnostic dictionary.

        Tool adapters decide whether a result is successful. A subprocess
        exiting zero alone does not establish scientific output validity.
        """
        self._requireOpen()

        if status not in ("success", "failed", "interrupted"):
            raise ValueError("Invocation status must be success, failed, or interrupted.")

        if not isinstance(invocationID, str) or invocationID not in self._openInvocations:
            raise ValueError("Invocation is unknown or has already been finalized.")

        result = {} if result is None else _copyProvenance(result)
        error = None if error is None else _copyProvenance(error)

        if type(result) is not dict: raise TypeError("result must be a dictionary.")
        if error is not None and type(error) is not dict: raise TypeError("error must be a dictionary or None.")
        if status == "success" and error is not None:
            raise ValueError("A successful invocation must not have an error record.")

        record, started = self._openInvocations[invocationID]
        record.update({
            "status": status,
            "endedAt": datetime.now(timezone.utc).isoformat(),
            "durationSeconds": max(0.0, time.perf_counter() - started),
            "result": result,
            "error": error,
        })
        del self._openInvocations[invocationID]

    ################################################################
    # Validate completion before reporting action success
    ################################################################
    def requireComplete(self):
        """Reject an execution that left tool attempts without outcomes."""
        self._requireOpen()
        if self._openInvocations:
            raise RuntimeError(f"{len(self._openInvocations)} tool invocation(s) were not finalized.")

    ################################################################
    # Freeze recording
    ################################################################
    def close(self):
        """Freeze records, retaining unfinished attempts as incomplete.

        Incomplete means no verified outcome was supplied. It does not claim
        that an external process was stopped or that its files were removed.
        """
        if self._closed: return

        for record, started in self._openInvocations.values():
            record.update({
                "status": "incomplete",
                "endedAt": datetime.now(timezone.utc).isoformat(),
                "durationSeconds": max(0.0, time.perf_counter() - started),
                "error": {
                    "kind": "missingOutcome",
                    "message": "Recording ended before this invocation supplied a verified outcome.",
                },
            })

        self._openInvocations.clear()
        self._closed = True

    ################################################################
    # Export independent metadata snapshots
    ################################################################
    def exportEnvironment(self): return deepcopy(self._environment)

    def exportInvocations(self): return deepcopy(self._invocations)
    
    ################################################################
    # Register a named scientific output
    ################################################################
    def registerProduct(self, name, kind, *, artifacts=None, inputProductIDs=None, metadata=None, invocationIDs=None, productID=None):
        """Record a named product produced by this execution.

        Does not copy files, attach payloads, or establish scientific validity.

        artifacts:
            Named references to files already attached to the working session.
            Each entry contains attachmentName, sha256, and size_bytes.

        inputProductIDs:
            Exact upstream product identities consumed to produce this product.

        metadata:
            Portable scientific associations such as scan UUIDs, original
            volume indices, reference identities, and transform conventions.

        invocationIDs:
            Producing tool invocations from this recorder. None selects all
            successful invocations recorded so far. Supply an explicit list
            when a product uses only a subset.

        productID:
            Optional preallocated canonical UUID. Allows metadata and payloads
            to identify their producing product before registration.
            None generates a new UUID as before.

            This recorder checks uniqueness within its execution. The owning
            action checks for collisions in the destination session history.

        Returns an independent product record. Metadata is copied at
        registration; editing the returned record does not update provenance.

        Action completion determines whether the recorded product is committed
        or uncommitted. A successful subprocess alone does not commit a product.
        """
        import re
        from uuid import UUID, uuid4

        self._requireOpen()

        if self._actionID is None:
            raise ValueError("Scientific product registration requires an action execution.")

        for fieldName, value in (("name", name), ("kind", kind)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Product {fieldName} must be a nonblank string.")
            if any(character in value for character in ("\x00", "\n", "\r")):
                raise ValueError(f"Product {fieldName} must not contain NUL or newlines.")

        products = getattr(self, "_products", [])
        if any(record["name"] == name for record in products):
            raise ValueError(f"This execution already registered product {name!r}.")

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

        if any(record["productID"] == productID for record in products):
            raise ValueError("This execution already registered the supplied productID.")

        artifacts = {} if artifacts is None else _copyProvenance(artifacts)
        metadata = {} if metadata is None else _copyProvenance(metadata)
        inputProductIDs = [] if inputProductIDs is None else _copyProvenance(inputProductIDs)

        if type(artifacts) is not dict or type(metadata) is not dict:
            raise TypeError("artifacts and metadata must be dictionaries.")
        if type(inputProductIDs) is not list:
            raise TypeError("inputProductIDs must be a list.")

        for identifier in inputProductIDs:
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Input product identities must be nonblank strings.")
        if len(set(inputProductIDs)) != len(inputProductIDs):
            raise ValueError("Input product identities must not contain duplicates.")
        if productID in inputProductIDs:
            raise ValueError("A product cannot identify itself as an upstream input.")

        for role, artifact in artifacts.items():
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Artifact roles must be nonblank strings.")
            if type(artifact) is not dict or set(artifact) != {"attachmentName", "sha256", "size_bytes"}:
                raise ValueError("Artifact references require attachmentName, sha256, and size_bytes.")

            attachmentName = artifact["attachmentName"]
            if not isinstance(attachmentName, str) or not attachmentName.strip():
                raise ValueError("Artifact attachmentName must be nonblank.")
            if not isinstance(artifact["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None:
                raise ValueError("Artifact identity requires a lowercase SHA-256.")
            if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] < 0:
                raise ValueError("Artifact size_bytes must be a nonnegative integer.")

        successfulInvocations = {
            record["invocationID"]
            for record in self._invocations
            if record["status"] == "success"
        }

        if invocationIDs is None:
            invocationIDs = [
                record["invocationID"]
                for record in self._invocations
                if record["status"] == "success"
            ]
        else:
            invocationIDs = _copyProvenance(invocationIDs)
            if type(invocationIDs) is not list:
                raise TypeError("invocationIDs must be a list or None.")
            if any(not isinstance(identifier, str) for identifier in invocationIDs):
                raise ValueError("Invocation identities must be strings.")
            if len(set(invocationIDs)) != len(invocationIDs):
                raise ValueError("Invocation identities must not contain duplicates.")
            if set(invocationIDs) - successfulInvocations:
                raise ValueError("Producing invocations must be successful calls from this execution.")

        record = {
            "schemaVersion": 1,
            "productID": productID,
            "name": name,
            "kind": kind,
            "producerActionID": self._actionID,
            "producerInvocationIDs": list(invocationIDs),
            "inputProductIDs": inputProductIDs,
            "artifacts": artifacts,
            "metadata": metadata,
        }

        self._products = [*products, record]
        return deepcopy(record)

    ################################################################
    # Export product metadata independently of payloads
    ################################################################
    def exportProducts(self):
        """Return independent product records, including before finalization."""
        return deepcopy(getattr(self, "_products", []))
    
    ################################################################
    # Record an explicitly resolved scientific dependency
    ################################################################
    def recordProductInput(self, selector, product):
        """Record exact product metadata consumed through an explicit selector.

        Called by selector resolution, not manually by each scientific action.
        Copies metadata so later caller edits cannot change provenance.

        Does not read or verify payload bytes. Artifact access must still use
        the recorded identities and the storage verification interface.
        """
        self._requireOpen()
        if self._actionID is None:
            raise ValueError("Product consumption recording requires an action execution.")

        selector = _copyProvenance(selector)
        product = _copyProvenance(product)

        if type(selector) is not dict or type(product) is not dict:
            raise TypeError("Product selection and record must be dictionaries.")
        if product.get("recordStatus") != "committed":
            raise ValueError("Only committed products can be consumed.")

        for field in ("productID", "producerActionID", "name", "kind"):
            if not isinstance(product.get(field), str) or not product[field].strip():
                raise ValueError(f"Consumed product has no usable {field}.")

        record = {
            "schemaVersion": 1,
            "selector": selector,
            "product": product,
        }
        inputs = getattr(self, "_productInputs", [])

        # Repeated use of the same selector/product is one declared dependency.
        # Different selectors resolving to the same product remain explicit.
        if record not in inputs:
            self._productInputs = [*inputs, record]

    ################################################################
    # Export resolved product dependencies
    ################################################################
    def exportProductInputs(self):
        """Return independent input records without reading payloads."""
        return deepcopy(getattr(self, "_productInputs", []))
    
################################################################
# External-tool execution result
################################################################
class pglToolResult:
    """One external-command result with its execution provenance.

    Log files remain available after execution, including failed execution.
    The caller owns their retention until managed artifact integration exists.

    provenance returns an independent snapshot. Local paths are diagnostic
    runtime locations, not portable checkpoint payload references.
    """

    def __init__(self, invocationID, *, logDirectory=None):
        self.invocationID = invocationID
        self.returnCode = None
        self.status = "running"
        self.logDirectory = logDirectory
        self.stdoutPath = None
        self.stderrPath = None
        self._provenance = {}

    @property
    def provenance(self): return deepcopy(self._provenance)

    def print(self):
        """Print a short execution summary without dumping command output."""
        from .pglMessages import pglMessages

        pglMessages.print(f"Tool execution: {self.status}, return code: {self.returnCode}")
        if self.stdoutPath is not None: pglMessages.print(f"stdout: {self.stdoutPath}")
        if self.stderrPath is not None: pglMessages.print(f"stderr: {self.stderrPath}")

    def saveProvenance(self, filename):
        """Write provenance to a new JSON file without overwriting."""
        import json
        from pathlib import Path

        path = Path(filename).expanduser().absolute()
        content = json.dumps(self._provenance, sort_keys=True, indent=2, allow_nan=False)

        with path.open("x", encoding="utf-8") as target:
            target.write(content)

        return path
    
    @property
    def outputs(self):
        """Return independent declared output records, or an empty dictionary."""
        invocation = self._provenance.get("invocation", {})
        return deepcopy(invocation.get("result", {}).get("outputs", {}))


################################################################
# External-tool execution error
################################################################
class pglToolExecutionError(RuntimeError):
    """Execution failure carrying the failed result and retained log paths."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


################################################################
# Shared external-tool runner
################################################################
class pglExternalTool:
    """Execute external commands with automatic execution provenance.

    executable:
        Optional default command name or executable path.

    directory:
        Optional explicit installation executable directory. Bare command
        names are resolved only within this directory when it is supplied.

    Package adapters will add version probes, command definitions, scientific
    output validation, and installation procedures.

    This base runner does not:
        Interpret arbitrary arguments as scientific inputs/outputs.
        Validate scientific results.
        Install missing tools.
        Execute through a shell.
        Register logs or outputs as session-derived artifacts.

    Commands and log contents may contain sensitive information. Do not pass
    credentials in arguments. Package adapters must implement redaction before
    supporting commands with sensitive parameters.

    Concurrent commands must not share one active recorder. Parallel execution
    requires explicit per-execution contexts and record aggregation.
    """

    # Record only relevant, explicitly selected environment controls.
    # Never dump the complete environment, PATH, or authentication variables.
    recordedEnvironmentNames = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "LANG",
        "LC_ALL",
        "LC_NUMERIC",
    )
    # Subclasses opt into automatic registration by declaring toolName.
    # Intermediate adapter bases can omit it.
    toolName = None
    toolLabel = None
    multipleInstallations = True
    resolvedLocationField = None

    configurationFields = {
        "directory": {
            "type": "path",
            "default": "",
            "label": "Executable directory",
            "help": "Directory containing the executables. Leave blank for automatic discovery.",
        },
    }
    _toolRegistry = {}

    def __init_subclass__(cls, **kwargs):
        """Register a named adapter when its module is imported.

        Does not instantiate the tool, discover executables, or install anything.
        Intermediate subclasses must declare their own toolName to register.

        Reloading the same module/class replaces its previous class object.
        A different class attempting to use that name is rejected.
        """
        import re

        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("toolName")
        if name is None: return

        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
            raise ValueError("toolName must be a lowercase identifier such as 'afni' or 'glmsingle'.")

        previous = pglExternalTool._toolRegistry.get(name)
        if previous is not None and previous is not cls:
            previousIdentity = (previous.__module__, previous.__qualname__)
            currentIdentity = (cls.__module__, cls.__qualname__)
            if previousIdentity != currentIdentity:
                raise ValueError(f"Tool name {name!r} is already registered by another adapter.")

        pglExternalTool._toolRegistry[name] = cls

    @classmethod
    def registeredTools(cls):
        """Return registered adapter classes without probing installations."""
        if cls is pglExternalTool:
            return dict(sorted(pglExternalTool._toolRegistry.items()))

        name = cls.__dict__.get("toolName")
        if name is None or pglExternalTool._toolRegistry.get(name) is not cls:
            raise ValueError("This adapter is not registered under its own toolName.")

        return {name: cls}


    @staticmethod
    def _toolSettingsPath(configFile=None):
        """Resolve the shared configuration path without reading tool payloads.

        configFile is primarily useful for isolated tests.
        Ordinary use follows the existing PGL settings-directory mechanism.
        """
        from pathlib import Path

        from .pglSettings import pglSettingsManager

        if configFile is not None:
            return Path(configFile).expanduser().absolute()

        directory = pglSettingsManager.getPGLSettingsDir()
        if directory is None:
            raise OSError("Could not access the PGL settings directory.")

        return Path(directory) / "tools.json"

    @staticmethod
    def _writeToolSettings(path, document):
        """Atomically replace configuration after complete JSON encoding.

        Does not install tools, modify scientific settings, or delete entries.
        Concurrent configuration writers are unsupported.
        """
        import json
        import os
        import tempfile

        document = _copyProvenance(document)
        content = (
            json.dumps(document, sort_keys=True, indent=4, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(content) > 1024 * 1024:
            raise ValueError("Tool settings exceed the supported 1 MiB size.")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporaryPath = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix=".tools-",
                suffix=".json",
                dir=path.parent,
                delete=False,
            ) as target:
                from pathlib import Path

                temporaryPath = Path(target.name)
                if target.write(content) != len(content):
                    raise OSError("Incomplete tool-settings write.")
                target.flush()
                os.fsync(target.fileno())

            os.replace(temporaryPath, path)
            temporaryPath = None
        finally:
            if temporaryPath is not None: temporaryPath.unlink(missing_ok=True)

    @staticmethod
    def _readToolSettings(path):
        """Read tool configuration, migrating legacy entries in memory.

        V1:
            tools[toolName] = configuration values

        V2:
            tools[toolName] = {
                "defaultInstallation": profile name,
                "installations": {profile name: configuration values},
            }

        Reading never rewrites an existing file. V1 entries become a profile
        named "default", preserving every saved configuration field.

        Unknown tool entries and unknown configuration fields are preserved.
        Malformed documents raise instead of being replaced with defaults.
        """
        import json

        if not path.exists():
            return {"schemaVersion": 2, "tools": {}}

        def uniqueKeys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate tool-settings key: {key!r}")
                result[key] = value
            return result

        def rejectConstant(value):
            raise ValueError(f"Nonfinite tool-settings value: {value}")

        with path.open("rb") as source:
            content = source.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("Tool settings exceed 1 MiB.")

        document = _copyProvenance(json.loads(
            content,
            object_pairs_hook=uniqueKeys,
            parse_constant=rejectConstant,
        ))

        if (
            type(document) is not dict
            or set(document) != {"schemaVersion", "tools"}
            or type(document["schemaVersion"]) is not int
            or document["schemaVersion"] not in (1, 2)
            or type(document["tools"]) is not dict
        ):
            raise ValueError("Unsupported tool-settings document.")

        def requireName(value, label):
            if (
                not isinstance(value, str) or not value.strip()
                or any(character in value for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError(f"{label} must be a nonblank name without NUL or newlines.")

        tools = {}

        for toolName, entry in document["tools"].items():
            requireName(toolName, "Tool name")
            if type(entry) is not dict:
                raise ValueError("Tool configuration entries must be dictionaries.")

            if document["schemaVersion"] == 1:
                entry = {
                    "defaultInstallation": "default",
                    "installations": {"default": entry},
                }

            if set(entry) != {"defaultInstallation", "installations"}:
                raise ValueError(f"Invalid installation-profile record for {toolName!r}.")

            installations = entry["installations"]
            if type(installations) is not dict or not installations:
                raise ValueError(f"Tool {toolName!r} requires at least one installation profile.")

            for profileName, values in installations.items():
                requireName(profileName, "Installation name")
                if type(values) is not dict:
                    raise ValueError("Installation values must be dictionaries.")

            defaultName = entry["defaultInstallation"]
            requireName(defaultName, "Default installation")
            if defaultName not in installations:
                raise ValueError("Default installation does not identify a saved profile.")

            tools[toolName] = entry

        return {"schemaVersion": 2, "tools": tools}

    @classmethod
    def _settingsFromValues(cls, values):
        """Validate adapter-declared fields without constructing GUI classes.

        Returns a lightweight namespace for compatibility with adapter checks.
        Its configNames identifies only supported fields. Unknown saved fields
        remain in storage but are not treated as executable configuration.

        No Qt imports, tool probes, or optional package imports occur here.
        """
        import re
        from types import SimpleNamespace

        if type(values) is not dict:
            raise TypeError("Tool configuration must be a dictionary.")

        fields = cls.configurationFields
        if type(fields) is not dict or not fields:
            raise ValueError("configurationFields must be a nonempty dictionary.")
        if type(cls.multipleInstallations) is not bool:
            raise TypeError("multipleInstallations must be Boolean.")

        resolved = {}

        for name, definition in fields.items():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
                or name in ("name", "toolName", "configNames")
            ):
                raise ValueError(f"Invalid configuration field: {name!r}")

            if type(definition) is not dict or set(definition) != {"type", "default", "label", "help"}:
                raise ValueError(f"Invalid configuration definition for {name!r}.")
            if definition["type"] not in ("text", "path", "bool"):
                raise ValueError(f"Unsupported configuration type for {name!r}.")
            if not isinstance(definition["label"], str) or not definition["label"].strip():
                raise ValueError("Configuration labels must be nonblank strings.")
            if not isinstance(definition["help"], str):
                raise ValueError("Configuration help must be a string.")

            # Validate defaults as well as supplied values.
            for value in (definition["default"], values.get(name, definition["default"])):
                if definition["type"] == "bool":
                    if type(value) is not bool:
                        raise TypeError(f"{name} must be Boolean.")
                elif (
                    not isinstance(value, str)
                    or any(character in value for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError(f"{name} must be a string without NUL or newlines.")

            resolved[name] = values.get(name, definition["default"])

        return SimpleNamespace(
            name=cls.toolLabel or cls.toolName,
            toolName=cls.toolName,
            configNames=tuple(fields),
            **resolved,
        )

    @classmethod
    def _getInstallationProfiles(cls, document):
        """Return saved profiles or a virtual default without writing settings."""
        cls.registeredTools()
        if cls is pglExternalTool:
            raise ValueError("Select a tool adapter.")

        entry = document["tools"].get(cls.toolName)
        if entry is None:
            entry = {
                "defaultInstallation": "default",
                "installations": {"default": {}},
            }

        if not cls.multipleInstallations and len(entry["installations"]) != 1:
            raise ValueError(f"{cls.toolLabel or cls.toolName} supports only one configuration.")

        return _copyProvenance(entry)

    @classmethod
    def getConfiguration(cls, *, installation=None, configFile=None):
        """Return supported fields from the selected installation profile.

        installation=None selects the saved default, or the virtual automatic
        default when no configuration has been saved.

        An unknown explicit profile raises. Never silently selects another
        installation or rewrites preferences.
        """
        document = cls._readToolSettings(cls._toolSettingsPath(configFile))
        entry = cls._getInstallationProfiles(document)

        if installation is not None and (not isinstance(installation, str) or not installation.strip()):
            raise ValueError("installation must be a nonblank profile name or None.")

        selected = entry["defaultInstallation"] if installation is None else installation
        if selected not in entry["installations"]:
            raise ValueError(f"Unknown installation {selected!r} for {cls.toolName!r}.")

        settings = cls._settingsFromValues(entry["installations"][selected])
        return _copyProvenance({
            name: getattr(settings, name)
            for name in settings.configNames
        })

    @classmethod
    def configure(cls, *, configFile=None):
        """Edit tool profiles using the existing schema-driven PGL dialog.

        Top-level call:
            Select registered tools, then edit their configurations.

        Subclass call:
            Edit only that tool.

        Existing nonblank values always take precedence. When the adapter
        declares resolvedLocationField, a blank field can be prefilled from
        that profile's matching, successful cached validation.

        Prefilling does not run probes or save settings. All edits are saved
        together only after every dialog is accepted. Cancel changes nothing.

        A discovered path accepted with OK becomes an explicit configuration.
        That configuration change can invalidate its previous status cache;
        validate=True establishes a fresh check for the explicit selection.

        Unknown fields survive for profiles retaining their names. Renaming
        profiles while removing profiles with unknown fields is refused rather
        than guessing how to transfer unsupported configuration.

        Does not install software, change Python environments, or launch tools.
        """
        from copy import deepcopy

        from .pglDialog import pglDialogs, pglToolDialog
        from .pglMessages import pglMessages

        adapters = cls.registeredTools()
        if not adapters:
            pglMessages.warning("No tool adapters are registered.", level=1)
            return None

        path = cls._toolSettingsPath(configFile)
        original = cls._readToolSettings(path)
        updated = deepcopy(original)
        selectedNames = list(adapters)

        if cls is pglExternalTool:
            labels = [
                f"{adapter.toolLabel or name} [{name}]"
                for name, adapter in adapters.items()
            ]
            selected = pglDialogs.selectItems(labels, title="Choose tools to configure")
            if selected is None: return None
            selectedNames = [selectedNames[index] for index in selected]

        for toolName in selectedNames:
            adapter = adapters[toolName]
            entry = adapter._getInstallationProfiles(original)
            fields = deepcopy(adapter.configurationFields)
            locationField = adapter.resolvedLocationField

            if locationField is not None:
                if (
                    not isinstance(locationField, str)
                    or locationField not in fields
                    or fields[locationField]["type"] != "path"
                ):
                    raise ValueError(
                        f"{toolName}: resolvedLocationField must identify a declared path field."
                    )

            profiles = []
            for profileName, values in entry["installations"].items():
                settings = adapter._settingsFromValues(values)
                profiles.append({
                    "name": profileName,
                    "values": {
                        name: getattr(settings, name)
                        for name in settings.configNames
                    },
                })

            # Read cached observations only when they could supply a blank
            # location. getToolReports(validate=False) performs no tool probes.
            needsSuggestion = (
                locationField is not None
                and any(not profile["values"][locationField].strip() for profile in profiles)
            )

            if needsSuggestion:
                reports = adapter.getToolReports(validate=False, configFile=path)
                reportsByProfile = {
                    report["installation"]: report
                    for report in reports
                }

                for profile in profiles:
                    if profile["values"][locationField].strip(): continue

                    report = reportsByProfile.get(profile["name"])
                    if report is None: continue
                    if report["reportSource"] != "cached" or report["available"] is not True:
                        continue

                    location = report.get("location")
                    if not isinstance(location, str) or not location.strip():
                        continue

                    # This changes only the document passed to the dialog.
                    # The original settings and status cache remain untouched.
                    profile["values"][locationField] = location

                fields[locationField]["help"] += (
                    " A matching previous successful check may supply a suggested "
                    "path. OK saves the displayed path; clear it to retain "
                    "automatic discovery. Cancel saves nothing."
                )

            request = {
                "schemaVersion": 1,
                "toolName": toolName,
                "label": adapter.toolLabel or toolName,
                "multipleInstallations": adapter.multipleInstallations,
                "fields": fields,
                "profiles": profiles,
                "defaultInstallation": entry["defaultInstallation"],
            }
            edited = pglDialogs.toolDialog(request)
            if edited is None: return None

            # Validate again at the configuration boundary. The dialog edits
            # profiles/values, not the adapter's identity or schema.
            edited = pglToolDialog.validate(edited)
            for field in ("schemaVersion", "toolName", "label", "multipleInstallations", "fields"):
                if edited[field] != request[field]:
                    raise ValueError(f"Tool dialog changed protected field {field!r}.")

            newProfiles = {}
            editedNames = {profile["name"] for profile in edited["profiles"]}
            oldNames = set(entry["installations"])

            removedWithUnknown = [
                name for name in oldNames - editedNames
                if set(entry["installations"][name]) - set(fields)
            ]
            if removedWithUnknown and editedNames - oldNames:
                raise ValueError(
                    "Cannot safely rename/copy profiles while removing profiles "
                    f"with unsupported fields: {removedWithUnknown}. "
                    "Keep their names, or delete them in a separate edit."
                )

            for profile in edited["profiles"]:
                adapter._settingsFromValues(profile["values"])
                previous = entry["installations"].get(profile["name"], {})
                newProfiles[profile["name"]] = {
                    **previous,
                    **profile["values"],
                }

            updated["tools"][toolName] = {
                "defaultInstallation": edited["defaultInstallation"],
                "installations": newProfiles,
            }

        if cls._readToolSettings(path) != original:
            raise ValueError("Tool configuration changed while editing. Reopen the dialog.")

        cls._writeToolSettings(path, updated)
        pglMessages.message(f"Saved tool installation profiles: {path}")
        return deepcopy(updated["tools"])
    
    @classmethod
    def print(cls, *, validate=False, verbose=False, timeout=10.0, cacheDirectory=None, configFile=None):
        """Print registered tools using fresh or persistently saved checks.

        print():
            Display cached status/version/location without launching probes.

        print(validate=True):
            Recheck installations and replace the stored observations.
            A failed check replaces an older successful result.

        Configuration remains separate from observed status. Cached results
        are labeled explicitly and are never claimed to be fresh verification.

        Returns the displayed reports. No installation occurs.
        """
        import builtins

        if type(verbose) is not bool:
            raise TypeError("verbose must be Boolean.")

        reports = cls.getToolReports(
            validate=validate,
            timeout=timeout,
            cacheDirectory=cacheDirectory,
            configFile=configFile,
        )
        path = cls._toolSettingsPath(configFile)

        def singleLine(value, limit=None):
            text = " ".join(str(value).split())
            return text if limit is None or len(text) <= limit else text[:limit - 1] + "…"

        headings = ("Tool", "Status", "Check", "Version", "Resolved location")
        rows = [
            (
                singleLine(report["label"], 24),
                singleLine(report["status"], 24),
                report["reportSource"],
                singleLine(report["version"] or "—", 48),
                singleLine(report["location"] or "—", 100),
            )
            for report in reports
        ]

        builtins.print(f"\nTool configuration: {path}")
        builtins.print(f"Saved checks: {path.with_name(f'{path.stem}.status.json')}")

        if rows:
            widths = [
                max(len(headings[index]), *(len(row[index]) for row in rows))
                for index in range(4)
            ]

            builtins.print()
            builtins.print(
                "  ".join(headings[index].ljust(widths[index]) for index in range(4))
                + "  " + headings[4]
            )
            for row in rows:
                builtins.print(
                    "  ".join(row[index].ljust(widths[index]) for index in range(4))
                    + "  " + row[4]
                )
        else:
            builtins.print("No registered tool adapters.")

        if any(report["reportSource"] == "cached" for report in reports):
            builtins.print("\nCached = last recorded check, not a new availability test.")

        for report in reports:
            if report["diagnostic"]:
                builtins.print(f"\n{report['label']}: {singleLine(report['diagnostic'])}")

            if verbose:
                builtins.print(f"\n{report['label']} [{report['toolName']}]")
                builtins.print(f"  Checked: {report['checkedAt'] or 'Not checked for this configuration'}")

                for field, value in report["configuration"].items():
                    automatic = value is None or isinstance(value, str) and not value.strip()
                    text = "Automatic/default" if automatic else repr(value)
                    builtins.print(f"  {field}: {singleLine(text)}")

                if report["version"]:
                    builtins.print(f"  Full version: {singleLine(report['version'])}")
                if report["location"]:
                    builtins.print(f"  Location: {singleLine(report['location'])}")

        if cls is pglExternalTool:
            document = cls._readToolSettings(path)
            unknown = sorted(set(document["tools"]) - set(cls.registeredTools()))
            if unknown:
                builtins.print("\nSaved configuration without a loaded adapter:")
                for name in unknown:
                    builtins.print(f"  {name}")

        return reports

    def __init__(self, executable=None, *, directory=None, installation=None):
        """Select executable configuration without launching or installing tools.

        Selection:
            Explicit executable path or directory.
            Explicit named installation.
            Saved default installation.
            Automatic discovery when no path is configured.

        Named installation and explicit path selection cannot be combined.
        Existing instances retain their selection after settings are edited.
        """
        import os
        from pathlib import Path

        if executable is not None:
            executable = os.fspath(executable)
            if not isinstance(executable, str) or not executable.strip() or "\x00" in executable:
                raise ValueError("executable must be a nonblank command or path without NUL.")

        explicitExecutable = False
        if executable is not None:
            expanded = os.path.expanduser(executable)
            explicitExecutable = os.path.isabs(expanded) or bool(os.path.dirname(expanded))

        if installation is not None:
            if not isinstance(installation, str) or not installation.strip():
                raise ValueError("installation must be a nonblank profile name.")
            if directory is not None or explicitExecutable:
                raise ValueError("Select an installation profile or an explicit path, not both.")

        self.configurationSource = "automatic"
        self.installation = None

        adapter = type(self)
        toolName = adapter.__dict__.get("toolName")
        registered = toolName is not None and pglExternalTool._toolRegistry.get(toolName) is adapter

        if installation is not None and not registered:
            raise ValueError("Named installations require a registered tool adapter.")

        if directory is None and not explicitExecutable and registered:
            document = adapter._readToolSettings(adapter._toolSettingsPath())
            entry = adapter._getInstallationProfiles(document)
            selected = entry["defaultInstallation"] if installation is None else installation

            if selected not in entry["installations"]:
                raise ValueError(f"Unknown installation {selected!r} for {toolName!r}.")

            settings = adapter._settingsFromValues(entry["installations"][selected])
            self.installation = selected

            savedDirectory = getattr(settings, "directory", "")
            if savedDirectory.strip():
                directory = savedDirectory
                self.configurationSource = "profile" if installation is not None else "saved"
            elif installation is not None:
                self.configurationSource = "profile"

        if directory is not None:
            directory = os.fspath(directory)
            if (
                not isinstance(directory, str) or not directory.strip()
                or any(character in directory for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("directory must be a nonblank local path without NUL or newlines.")

            directory = str(Path(directory).expanduser().absolute())
            if self.configurationSource == "automatic":
                self.configurationSource = "explicit"

        if explicitExecutable:
            self.configurationSource = "explicit"

        self.executable = executable
        self.directory = directory
        
    ################################################################
    # Identify executable or log bytes
    ################################################################
    @staticmethod
    def _fileIdentity(filename, *, showProgress=None):
        """Hash a regular local file without loading its contents into RAM.

        showProgress:
            None: report automatically for files of at least 16 MiB.
            True: report regardless of file size.
            False: remain quiet.

        Reporting includes:
            Start and completion messages.
            Byte-level progress through pglProgress.
            Text updates every 10 seconds, captured by the action console tee.

        Existing callers require no changes. Small transform/log files remain
        quiet by default; large acquisition checks become visible.

        Reads the complete file in 1 MiB blocks. Does not modify the source.
        Detects size/mtime changes during hashing, but does not provide a
        filesystem snapshot or protection against all concurrent modification.
        """
        import hashlib
        import os
        import stat
        import time
        from contextlib import ExitStack
        from pathlib import Path

        from .pglProgress import pglProgress

        if showProgress is not None and type(showProgress) is not bool:
            raise TypeError("showProgress must be Boolean or None.")

        path = Path(filename)
        digest = hashlib.sha256()
        size = 0

        # Open first so file validation and size describe the actual handle.
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Expected a regular file.")

            expectedSize = before.st_size
            report = expectedSize >= 16 * 1024 * 1024 if showProgress is None else showProgress
            label = str(path).replace("\r", "\\r").replace("\n", "\\n")
            started = time.monotonic()
            lastBarUpdate = started
            lastTextUpdate = started

            with ExitStack() as resources:
                progress = None

                if report:
                    # Plain text is intentional: notebook widgets alone do not
                    # become part of the saved Python stdout transcript.
                    statusOutput(
                        f"Verifying SHA-256: {label}\n"
                        f"  Reading {expectedSize / (1024 ** 2):,.1f} MiB",
                        flush=True,
                    )
                    progress = resources.enter_context(
                        pglProgress(f"Hashing {path.name}")
                    )
                    progress(0, "Reading file bytes for SHA-256 verification")

                while True:
                    block = source.read(1024 * 1024)
                    if not block: break

                    digest.update(block)
                    size += len(block)

                    if size > expectedSize:
                        raise ValueError(f"File grew during hashing: {label}")

                    if report:
                        now = time.monotonic()

                        # Limit UI updates; keep hashing independent of display
                        # refresh speed and avoid flooding the notebook.
                        if now - lastBarUpdate >= 0.5:
                            progress(
                                size / max(expectedSize, 1),
                                f"Hashed {size / (1024 ** 2):,.1f} / "
                                f"{expectedSize / (1024 ** 2):,.1f} MiB",
                            )
                            lastBarUpdate = now

                        if now - lastTextUpdate >= 10:
                            elapsed = now - started
                            percent = 100 * size / max(expectedSize, 1)
                            rate = size / (1024 ** 2) / max(elapsed, 0.001)
                            statusOutput(
                                f"  SHA-256 {percent:5.1f}%: "
                                f"{size / (1024 ** 2):,.1f} / "
                                f"{expectedSize / (1024 ** 2):,.1f} MiB; "
                                f"{elapsed:,.1f}s elapsed; {rate:,.1f} MiB/s",
                                flush=True,
                            )
                            lastTextUpdate = now

                after = os.fstat(source.fileno())
                if (
                    size != expectedSize
                    or after.st_size != expectedSize
                    or after.st_mtime_ns != before.st_mtime_ns
                ):
                    raise ValueError(f"File changed during hashing: {label}")

                if report:
                    elapsed = time.monotonic() - started
                    progress.finish(f"SHA-256 complete — {size / (1024 ** 2):,.1f} MiB")
                    statusOutput(
                        f"SHA-256 complete: {path.name} — "
                        f"{size / (1024 ** 2):,.1f} MiB in {elapsed:,.1f}s",
                        flush=True,
                    )

        return {"sha256": digest.hexdigest(), "size_bytes": size}

    ################################################################
    # Resolve the actual executable
    ################################################################
    def _resolveExecutable(self, commandName, environment):
        """Resolve without silently falling back from an explicit installation."""
        import os
        import shutil
        from pathlib import Path

        expanded = os.path.expanduser(commandName)
        containsPath = os.path.isabs(expanded) or bool(os.path.dirname(expanded))

        if containsPath:
            candidate = str(Path(expanded).absolute())
            resolved = shutil.which(candidate)
        elif self.directory is not None:
            candidate = str(Path(self.directory) / expanded)
            resolved = shutil.which(candidate)
        else:
            resolved = shutil.which(expanded, path=environment.get("PATH", os.defpath))

        if resolved is None:
            location = f" in {self.directory!r}" if self.directory is not None and not containsPath else ""
            raise FileNotFoundError(f"Executable {commandName!r} was not found or is not executable{location}.")

        resolved = Path(resolved).resolve()

        # Windows command scripts need shell semantics. Do not silently invoke
        # a command processor when the public contract promises shell=False.
        if os.name == "nt" and resolved.suffix.lower() in (".bat", ".cmd"):
            raise ValueError("Windows batch files require an explicit, separately supported shell adapter.")

        return resolved

    ################################################################
    # Run a command with user-facing failure handling
    ################################################################
    def runCommand(self, commandName=None, *, arguments=None, cwd=None, timeout=None, environment=None, cacheDirectory=None):
        """Execute a command, warning instead of raising for freeform failures.

        Returns:
            pglToolResult on success.
            None on an ordinary failure outside an action.

        Inside an action:
            Propagates failures internally to pglAction.run(), which warns,
            marks the action failed, and returns None. This immediately stops
            the scientific action rather than continuing after a failed tool.

        Outside an action:
            Warns and returns None. Subsequent caller statements are not
            automatically stopped; callers should check the returned value.

        lastResult:
            Retains the most recent successful or failed pglToolResult.
            None when validation/preparation failed before a result existed.
            Replaced at the start of each call; use action history or retain
            results explicitly when inspecting multiple executions.

        KeyboardInterrupt and SystemExit propagate intentionally.

        Arguments:
            commandName: executable name/path; None uses the configured one.
            arguments: argument list, never a shell command string.
            cwd: working directory; None uses the current directory.
            timeout: subprocess timeout in seconds; None disables the limit.
            environment: explicit string environment overrides.
            cacheDirectory: optional parent directory for retained logs.

        Generic command success does not establish scientific output validity.
        This wrapper does not install tools or generate missing dependencies.

        Examples:
            result = tool.runCommand(arguments=["--version"])
            if result is not None:
                result.print()

            # After a failed freeform call:
            if tool.lastResult is not None:
                tool.lastResult.print()
        """
        from .pglMessages import pglMessages

        self.lastResult = None

        try:
            self.lastResult = self._runCommand(
                commandName,
                arguments=arguments,
                cwd=cwd,
                timeout=timeout,
                environment=environment,
                cacheDirectory=cacheDirectory,
            )
            return self.lastResult

        except Exception as error:
            if isinstance(error, pglToolExecutionError):
                self.lastResult = error.result

            recorder = getExecutionRecorder()

            # Stop _run() immediately. The owning action runner emits the
            # warning and performs its existing failure/history bookkeeping.
            # Do not warn here as well, producing duplicate diagnostics.
            if recorder is not None and recorder.actionID is not None:
                raise

            # Freeform use has no action boundary to catch the exception.
            # Retain the failed result and logs, but return without a traceback.
            pglMessages.warning(
                f"External command was not completed: "
                f"{type(error).__name__}: {error}",
            )
            return None

    ################################################################
    # Execute one command
    ################################################################
    def _runCommand(self, commandName=None, *, arguments=None, cwd=None, timeout=None, environment=None, cacheDirectory=None, streamOutput=False):
        """Execute a command and automatically retain its provenance.

        commandName:
            Command name/path; None uses this instance's executable.

        arguments:
            List/tuple of strings or path-like arguments, excluding the
            executable. No shell parsing, wildcard expansion, or quoting
            removal is performed.

        cwd:
            Working directory; None uses the caller's current directory.

        timeout:
            Maximum subprocess execution time in seconds; None has no limit.
            Does not limit environment capture, executable hashing, or log
            hashing. Process creation itself may not be interruptible.

        environment:
            Explicit string environment overrides. Effective values of only
            the allowlisted variables enter provenance.

        cacheDirectory:
            Optional local parent directory for retained execution logs.
            None uses the OS temporary directory. Successful and failed logs
            are retained; no automatic cache eviction is performed.

        streamOutput:
            Display newly written stdout/stderr while the command runs.
            Complete original bytes remain in separate retained log files.
            Output is displayed as written; no percentage or ETA is inferred.
            Ordering between stdout and stderr is not guaranteed.
            Child-side buffering can delay visible output.

        Returns:
            pglToolResult on zero exit status.

        Raises:
            pglToolExecutionError on launch failure, timeout, or nonzero exit.
            The exception carries the failed result.
            KeyboardInterrupt/SystemExit are recorded and propagated.

        A zero exit status is command success, not scientific validation.

        Timeout/interruption cleanup stops and waits for the direct child.
        Descendant-process cleanup is not guaranteed by this initial runner;
        adapters launching process trees need an explicit lifecycle strategy.
        """
        import codecs
        import math
        import os
        import subprocess
        import sys
        import tempfile
        import time
        from contextlib import ExitStack
        from pathlib import Path

        commandName = self.executable if commandName is None else commandName
        if commandName is None: raise ValueError("Supply commandName or configure a default executable.")

        commandName = os.fspath(commandName)
        if not isinstance(commandName, str) or not commandName.strip() or "\x00" in commandName:
            raise ValueError("commandName must be a nonblank string without NUL.")

        if arguments is None: arguments = []
        if not isinstance(arguments, (list, tuple)):
            raise TypeError("arguments must be a list or tuple, not a shell command string.")

        argumentList = []
        for argument in arguments:
            value = os.fspath(argument) if isinstance(argument, os.PathLike) else argument
            if not isinstance(value, str) or "\x00" in value:
                raise TypeError("Each argument must be a string or path-like value without NUL.")
            argumentList.append(value)

        if timeout is not None:
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite, or None.")
        if type(streamOutput) is not bool:
            raise TypeError("streamOutput must be Boolean.")

        if environment is None: environment = {}
        if type(environment) is not dict: raise TypeError("environment must be a dictionary or None.")

        for name, value in environment.items():
            if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
                raise ValueError("Environment variable names must be nonempty strings without '=' or NUL.")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("Environment variable values must be strings without NUL.")

        effectiveEnvironment = dict(os.environ)
        effectiveEnvironment.update(environment)
        workingDirectory = Path.cwd() if cwd is None else Path(cwd).expanduser().absolute()

        cacheRoot = None
        if cacheDirectory is not None:
            cacheText = os.fspath(cacheDirectory)
            if (
                not isinstance(cacheText, str)
                or not cacheText.strip()
                or "://" in cacheText
                or "::" in cacheText
                or any(character in cacheText for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("cacheDirectory must be an ordinary local path.")
            cacheRoot = Path(cacheText).expanduser().absolute()

        specification = {
            "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
            "requestedExecutable": commandName,
            "installationDirectory": self.directory,
            "arguments": argumentList,
            "workingDirectory": str(workingDirectory),
            "timeoutSeconds": timeout,
            "streamOutput": streamOutput,
            "environment": {
                name: effectiveEnvironment[name]
                for name in self.recordedEnvironmentNames
                if name in effectiveEnvironment
            },
            "unrecordedEnvironmentOverrideNames": sorted(
                name for name in environment if name not in self.recordedEnvironmentNames
            ),
            "dependencyCapture": {
                "status": "incomplete",
                "reason": "Generic command: scientific inputs, outputs, implicit resources, and tool build are not yet interpreted.",
            },
            "processCleanupScope": "directChild",
        }

        # Freeform calls retain equivalent provenance without fabricating an
        # action execution or installing a lasting active context.
        recorder = getExecutionRecorder()
        standalone = recorder is None
        if standalone: recorder = pglExecutionRecorder()

        invocationID = recorder.beginInvocation(specification)
        result = pglToolResult(invocationID)
        executionDetails = {
            "resolvedExecutable": None,
            "executableIdentity": None,
            "returnCode": None,
            "logs": {},
        }
        executionError = None
        failureRecord = None

        try:
            if not workingDirectory.is_dir():
                raise NotADirectoryError(f"Working directory is unavailable: {workingDirectory}")

            # Resolve and identify before launch. This identifies the executable
            # file, not all its shared libraries or script dependencies.
            executable = self._resolveExecutable(commandName, effectiveEnvironment)
            executionDetails["resolvedExecutable"] = str(executable)
            executionDetails["executableIdentity"] = self._fileIdentity(executable)

            if cacheRoot is not None: cacheRoot.mkdir(parents=True, exist_ok=True)

            logDirectory = Path(tempfile.mkdtemp(
                prefix="pgl-tool-",
                dir=None if cacheRoot is None else str(cacheRoot),
            ))
            result.logDirectory = logDirectory
            result.stdoutPath = logDirectory / "stdout.log"
            result.stderrPath = logDirectory / "stderr.log"

            # Both streams remain file-backed. Live display reads those files
            # without replacing or modifying the recorded bytes.
            with ExitStack() as stack:
                stdout = stack.enter_context(result.stdoutPath.open("xb"))
                stderr = stack.enter_context(result.stderrPath.open("xb"))
                readers = []

                if streamOutput:
                    for path, destination in (
                        (result.stdoutPath, sys.stdout),
                        (result.stderrPath, sys.stderr),
                    ):
                        readers.append({
                            "file": stack.enter_context(path.open("rb", buffering=0)),
                            "decoder": codecs.getincrementaldecoder("utf-8")(errors="replace"),
                            "destination": destination,
                            "displayFailed": False,
                        })

                def displayText(reader, text):
                    """Do not fail a scientific command because display broke."""
                    if not text or reader["displayFailed"]: return

                    try:
                        reader["destination"].write(text)
                        reader["destination"].flush()
                    except Exception as error:
                        reader["displayFailed"] = True
                        executionDetails.setdefault("displayErrors", []).append({
                            "errorType": type(error).__name__,
                            "message": str(error),
                        })

                def displayAvailable(final=False):
                    """Read bounded chunks, including partial lines and UTF-8."""
                    for reader in readers:
                        handle = reader["file"]

                        # During execution, bound each pass so noisy output
                        # cannot indefinitely delay timeout/cancellation checks.
                        # At completion, drain the bytes currently in the file.
                        remaining = (
                            max(0, os.fstat(handle.fileno()).st_size - handle.tell())
                            if final else 65536
                        )

                        while remaining > 0:
                            chunk = handle.read(min(65536, remaining))
                            if not chunk: break
                            remaining -= len(chunk)
                            displayText(reader, reader["decoder"].decode(chunk))

                        if final:
                            displayText(reader, reader["decoder"].decode(b"", final=True))

                # stdin is closed so unattended tools cannot hang on a hidden
                # prompt; interactive tools need a separate adapter.
                command = [str(executable), *argumentList]
                process = subprocess.Popen(
                    command,
                    cwd=str(workingDirectory),
                    env=effectiveEnvironment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                )

                try:
                    if streamOutput:
                        deadline = None if timeout is None else time.monotonic() + timeout

                        while process.poll() is None:
                            if deadline is not None and time.monotonic() >= deadline:
                                raise subprocess.TimeoutExpired(command, timeout)

                            displayAvailable()
                            delay = 0.1 if deadline is None else min(0.1, max(0.0, deadline - time.monotonic()))
                            time.sleep(delay)

                        displayAvailable(final=True)

                    else:
                        process.wait(timeout=timeout)

                finally:
                    # Always stop and reap the direct child before closing or
                    # hashing its logs, including on timeout or interruption.
                    try:
                        if process.poll() is None: process.kill()
                    finally:
                        process.wait()
                        result.returnCode = process.returncode
                        executionDetails["returnCode"] = process.returncode

            if result.returnCode != 0:
                raise RuntimeError(f"Command exited with return code {result.returnCode}.")

            result.status = "success"

        except BaseException as error:
            executionError = error
            result.status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            failureRecord = {
                "errorType": type(error).__name__,
                "message": str(error),
                "timedOut": isinstance(error, subprocess.TimeoutExpired),
            }

        finally:
            # Logs survive scientific failure. If hashing a log fails, retain
            # its diagnostic path and explicitly report unavailable identity.
            for name, path in (("stdout", result.stdoutPath), ("stderr", result.stderrPath)):
                if path is None: continue

                record = {"localPath": str(path)}
                try:
                    record.update(self._fileIdentity(path))
                    record["identityStatus"] = "captured"
                except Exception as error:
                    record["identityStatus"] = "unavailable"
                    record["errorType"] = type(error).__name__

                    if executionError is None:
                        executionError = error
                        result.status = "failed"
                        failureRecord = {
                            "errorType": type(error).__name__,
                            "message": "Could not establish the identity of an execution log.",
                            "timedOut": False,
                        }

                executionDetails["logs"][name] = record

            recorder.finishInvocation(
                invocationID,
                status=result.status,
                result=executionDetails,
                error=failureRecord,
            )

            if standalone: recorder.close()

            # A result contains only this invocation, not unrelated calls made
            # by the same action. Action history receives the complete list.
            invocation = next(
                record for record in recorder.exportInvocations()
                if record["invocationID"] == invocationID
            )
            result._provenance = {
                "schemaVersion": 1,
                "environment": recorder.exportEnvironment(),
                "invocation": invocation,
            }

        if executionError is not None:
            if not isinstance(executionError, Exception):
                raise executionError

            logHint = f" Logs: {result.logDirectory}" if result.logDirectory is not None else ""
            raise pglToolExecutionError(
                f"{commandName!r} failed: {failureRecord['message']}{logHint}",
                result,
            ) from executionError

        return result
    
    @classmethod
    def getStatus(cls, *, configuration=None, timeout=10.0, cacheDirectory=None):
        """Return the common installation-status report.

        Subclasses override this method using their existing discovery and
        validation mechanisms. The base implementation does not run commands
        or assume that a configured path establishes availability.

        configuration:
            Optional explicit configuration dictionary supplied by the shared
            printer/dialog. None reads this adapter's saved configuration.

        Status reports contain:
            toolName, label, status, available, version, location,
            diagnostic, details.

        available=None means availability was not established.
        No status operation may automatically install or update software.
        """
        configuration = cls.getConfiguration() if configuration is None else _copyProvenance(configuration)
        cls._settingsFromValues(configuration)

        return {
            "toolName": cls.toolName,
            "label": cls.toolLabel or cls.toolName,
            "status": "Check unsupported",
            "available": None,
            "version": None,
            "location": None,
            "diagnostic": "This adapter has not implemented installation-status checking.",
            "details": {},
        }
    
    @classmethod
    def getToolReports(cls, *, validate=False, timeout=10.0, cacheDirectory=None, configFile=None):
        """Return one report per installation of each registered tool.

        validate=False reads cached observations without probing.
        validate=True checks every profile and replaces its saved observation,
        including failed checks.

        Configuration selects installations; status records observations.
        Neither checking nor printing installs tools or changes the default.

        Status-cache v2:
            tools[toolName][installationName] = {
                context, checkedAt, report,
            }

        Legacy v1 status caches are not reused because they did not explicitly
        identify installation profiles. Reading never rewrites them; fresh
        validation writes v2.

        Unknown tool entries in a v2 cache are preserved.
        Concurrent configuration/status writers are unsupported.
        """
        import json
        import math
        import sys
        from copy import deepcopy
        from datetime import datetime, timezone

        if type(validate) is not bool:
            raise TypeError("validate must be Boolean.")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite.")

        adapters = cls.registeredTools()
        configPath = cls._toolSettingsPath(configFile)
        statusPath = configPath.with_name(f"{configPath.stem}.status.json")
        configurationDocument = cls._readToolSettings(configPath)
        statusDocument = {"schemaVersion": 2, "tools": {}}

        def uniqueKeys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate tool status-cache key: {key!r}")
                result[key] = value
            return result

        def rejectConstant(value):
            raise ValueError(f"Nonfinite tool status-cache value: {value}")

        if statusPath.exists():
            with statusPath.open("rb") as source:
                content = source.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError("Tool status cache exceeds 1 MiB.")

            previous = _copyProvenance(json.loads(
                content,
                object_pairs_hook=uniqueKeys,
                parse_constant=rejectConstant,
            ))

            if (
                type(previous) is not dict
                or set(previous) != {"schemaVersion", "tools"}
                or type(previous["schemaVersion"]) is not int
                or previous["schemaVersion"] not in (1, 2)
                or type(previous["tools"]) is not dict
            ):
                raise ValueError("Invalid tool status cache.")

            if previous["schemaVersion"] == 2:
                statusDocument = previous

        reportFields = {
            "toolName", "label", "status", "available",
            "version", "location", "diagnostic", "details",
        }

        def validateReport(report, toolName):
            """Reject malformed adapter reports before caching them."""
            report = _copyProvenance(report)

            if type(report) is not dict or set(report) != reportFields:
                raise ValueError("Invalid tool status-report fields.")
            if report["toolName"] != toolName:
                raise ValueError("Status report identifies a different tool.")
            if report["available"] is not None and type(report["available"]) is not bool:
                raise ValueError("available must be Boolean or None.")

            for field in ("label", "status"):
                if not isinstance(report[field], str) or not report[field].strip():
                    raise ValueError(f"{field} must be a nonblank string.")

            for field in ("version", "location", "diagnostic"):
                if report[field] is not None and not isinstance(report[field], str):
                    raise ValueError(f"{field} must be a string or None.")

            if type(report["details"]) is not dict:
                raise ValueError("Status details must be a dictionary.")

            return report

        reports = []

        for toolName, adapter in adapters.items():
            entry = adapter._getInstallationProfiles(configurationDocument)
            cachedProfiles = statusDocument["tools"].get(toolName, {})

            if type(cachedProfiles) is not dict:
                raise ValueError(f"Invalid status-cache profile mapping for {toolName!r}.")

            # Work on a copy; publication occurs only after all checks finish.
            cachedProfiles = deepcopy(cachedProfiles)

            for profileName, values in entry["installations"].items():
                # Validate THIS profile's values, not its enclosing container.
                settings = adapter._settingsFromValues(values)
                configuration = _copyProvenance({
                    field: getattr(settings, field)
                    for field in settings.configNames
                })

                context = {
                    "configuration": configuration,
                    "adapter": f"{adapter.__module__}.{adapter.__qualname__}",
                    "pythonExecutable": sys.executable,
                }
                emptyReport = {
                    "toolName": toolName,
                    "label": adapter.toolLabel or toolName,
                    "status": "Not checked",
                    "available": None,
                    "version": None,
                    "location": None,
                    "diagnostic": None,
                    "details": {},
                }
                checkedAt = None
                reportSource = "none"

                if validate:
                    try:
                        checked = adapter.getStatus(
                            configuration=deepcopy(configuration),
                            timeout=timeout,
                            cacheDirectory=cacheDirectory,
                        )
                        report = validateReport(checked, toolName)

                    except Exception as error:
                        # A failed fresh check replaces an older success.
                        report = {
                            **emptyReport,
                            "status": "Check failed",
                            "diagnostic": f"{type(error).__name__}: {error}",
                        }

                    checkedAt = datetime.now(timezone.utc).isoformat()
                    reportSource = "fresh"
                    cachedProfiles[profileName] = {
                        "context": context,
                        "checkedAt": checkedAt,
                        # Keep the cache small. Fresh returned reports retain
                        # details; command provenance remains in existing logs.
                        "report": {**report, "details": {}},
                    }

                else:
                    cached = cachedProfiles.get(profileName)

                    if cached is None:
                        report = emptyReport
                    else:
                        if (
                            type(cached) is not dict
                            or set(cached) != {"context", "checkedAt", "report"}
                            or type(cached["context"]) is not dict
                            or not isinstance(cached["checkedAt"], str)
                        ):
                            raise ValueError(
                                f"Invalid cached status for {toolName!r}/{profileName!r}."
                            )

                        previousReport = validateReport(cached["report"], toolName)

                        if cached["context"] == context:
                            report = previousReport
                            checkedAt = cached["checkedAt"]
                            reportSource = "cached"
                        else:
                            report = {
                                **emptyReport,
                                "status": "Needs recheck",
                                "diagnostic": "Configuration, adapter, or Python environment changed.",
                            }
                            reportSource = "stale"

                reports.append({
                    **report,
                    "installation": profileName,
                    "isDefault": profileName == entry["defaultInstallation"],
                    "configuration": configuration,
                    "checkedAt": checkedAt,
                    "reportSource": reportSource,
                })

            if validate:
                statusDocument["tools"][toolName] = cachedProfiles

        if validate:
            if cls._readToolSettings(configPath) != configurationDocument:
                raise ValueError("Configuration changed during validation; status was not saved.")
            cls._writeToolSettings(statusPath, statusDocument)

        return _copyProvenance(reports)

    @classmethod
    def print(cls, *, validate=False, verbose=False, timeout=10.0, cacheDirectory=None, configFile=None):
        """Print all registered tools and their installation profiles.

        Cached checks are labeled explicitly. validate=True refreshes each
        displayed installation and saves its latest result. No installation
        or automatic version switching occurs.
        """
        import builtins

        if type(verbose) is not bool:
            raise TypeError("verbose must be Boolean.")

        reports = cls.getToolReports(
            validate=validate,
            timeout=timeout,
            cacheDirectory=cacheDirectory,
            configFile=configFile,
        )
        path = cls._toolSettingsPath(configFile)

        def line(value, limit=80):
            text = " ".join(str(value).split())
            return text if len(text) <= limit else text[:limit - 1] + "…"

        headings = ("Tool", "Installation", "Status", "Check", "Version", "Location")
        rows = [
            (
                line(report["label"], 20),
                line(report["installation"] + (" *" if report["isDefault"] else ""), 24),
                line(report["status"], 24),
                report["reportSource"],
                line(report["version"] or "—", 40),
                line(report["location"] or "—", 100),
            )
            for report in reports
        ]

        builtins.print(f"\nTool configuration: {path}")
        builtins.print("* = default installation; cached = previous check, not fresh validation.")

        if rows:
            widths = [max(len(headings[index]), *(len(row[index]) for row in rows)) for index in range(5)]
            builtins.print()
            builtins.print("  ".join(headings[index].ljust(widths[index]) for index in range(5)) + "  " + headings[5])
            for row in rows:
                builtins.print("  ".join(row[index].ljust(widths[index]) for index in range(5)) + "  " + row[5])
        else:
            builtins.print("No registered tool adapters.")

        for report in reports:
            label = f"{report['label']} / {report['installation']}"
            if report["diagnostic"]:
                builtins.print(f"\n{label}: {line(report['diagnostic'], 1000)}")

            if verbose:
                builtins.print(f"\n{label}")
                builtins.print(f"  Checked: {report['checkedAt'] or 'Not checked'}")
                for field, value in report["configuration"].items():
                    automatic = value is None or isinstance(value, str) and not value.strip()
                    builtins.print(f"  {field}: {'Automatic/default' if automatic else repr(value)}")
                if report["version"]:
                    builtins.print(f"  Full reported version: {line(report['version'], 2000)}")
                if report["location"]:
                    builtins.print(f"  Location: {line(report['location'], 2000)}")

        if cls is pglExternalTool:
            document = cls._readToolSettings(path)
            unknown = sorted(set(document["tools"]) - set(cls.registeredTools()))
            if unknown:
                builtins.print("\nSaved configuration without a loaded adapter:")
                for name in unknown: builtins.print(f"  {name}")

        return reports
    
    @classmethod
    def printInstallationNotice(cls):
        """Display post-installation instructions before installation starts.

        Shared by tool installers. Does not validate, install, or modify settings.
        The reminder applies after a successful installation, not cancellation.
        """
        from .pglMessages import pglMessages

        className = cls.__name__
        command = (
            f"from {cls.__module__} import {className}\n"
            f"{className}.print(validate=True, timeout=120)"
        )

        pglMessages.print(
            f"\nIMPORTANT — AFTER INSTALLING {cls.toolLabel or className}\n\n"
            "After installation completes successfully:\n"
            "  1. Restart your notebook kernel.\n"
            "  2. Run the following command to validate the installation "
            "and refresh its saved status:\n\n"
            f"{command}\n",
            emphasize=True,
        )
    
    
################################################################
# AFNI tools
################################################################
class pglAFNI(pglExternalTool):
    """AFNI command access and installation diagnostics.

    directory:
        Optional directory containing AFNI executables.
        When omitted, the initial AFNI executable is discovered through PATH.

    Availability checks:
        Launch afni -ver to identify the installation.
        Optionally launch a requested command with -help.
        Require the requested command to come from the same executable
        directory as the identified AFNI installation.

    Checks establish executable availability, not scientific correctness or
    completeness of every optional AFNI dependency.

    Installation and typed scientific command adapters are not implemented yet.
    """
    toolName = "afni"
    toolLabel = "AFNI"
    multipleInstallations = False
    installationURL = "https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html"
    resolvedLocationField = "directory"
    

    def __init__(self, *, directory=None, installation=None):
        """Select an AFNI installation without probing or advancing setup."""
        super().__init__(
            executable="afni",
            directory=directory,
            installation=installation,
        )
        self.lastResult = None
        self.lastAvailability = None
    ################################################################
    # Check availability, including managed installation progress
    ################################################################
    def checkAvailable(self, commandName=None, *, timeout=10.0, cacheDirectory=None, installPath="~/proj/afni"):
        """Check the selected installation without advancing setup.

        Incomplete managed setup warns with continuation instructions and
        returns False. Finding another AFNI on PATH cannot hide that state.

        Verified managed setup is probed at installPath/abin.
        When no state exists, an existing installPath/abin directory is checked
        before falling back to ordinary PATH discovery.

        An explicitly configured directory outside installPath/abin checks
        that separate installation without consulting this root's state.

        The check does not modify the instance's executable directory,
        install files, mark verification complete, or run installation stages.
        """
        from pathlib import Path

        from .pglMessages import pglMessages

        self.lastResult = None
        self.lastAvailability = None

        try:
            installation = pglAFNIInstallation(installPath)
            managedDirectory = installation.root / "abin"
            explicitDirectory = None if self.directory is None else Path(self.directory).expanduser().resolve()

            selectedManaged = explicitDirectory is None or explicitDirectory == managedDirectory.resolve()

            if not selectedManaged:
                return self._checkAvailable(
                    commandName,
                    timeout=timeout,
                    cacheDirectory=cacheDirectory,
                )

            state = installation.readState()

            if state is not None and state["status"] != "verified":
                message = installation.continuationMessage(state)
                self.lastAvailability = {
                    "schemaVersion": 1,
                    "tool": "AFNI",
                    "requestedCommand": commandName,
                    "available": False,
                    "installationStatus": state["status"],
                    "installPath": str(installation.root),
                    "error": {"message": message},
                }
                pglMessages.warning(message)
                return False

            # Existing managed content is not silently bypassed in favor of
            # another installation on PATH, even if its executable is missing.
            if state is not None or managedDirectory.exists():
                probe = pglAFNI(directory=str(managedDirectory))
                available = probe._checkAvailable(
                    commandName,
                    timeout=timeout,
                    cacheDirectory=cacheDirectory,
                )
                self.lastResult = probe.lastResult
                self.lastAvailability = deepcopy(probe.lastAvailability)

                if self.lastAvailability is not None:
                    self.lastAvailability["installPath"] = str(installation.root)
                    self.lastAvailability["installationStatus"] = None if state is None else state["status"]

                return available

            return self._checkAvailable(
                commandName,
                timeout=timeout,
                cacheDirectory=cacheDirectory,
            )

        except Exception as error:
            self.lastAvailability = {
                "schemaVersion": 1,
                "tool": "AFNI",
                "requestedCommand": commandName,
                "available": False,
                "error": {
                    "errorType": type(error).__name__,
                    "message": str(error),
                },
            }
            pglMessages.warning(f"AFNI availability could not be established: {type(error).__name__}: {error}")
            return False
    
    ################################################################
    # Check AFNI installation and optional command availability
    ################################################################
    def _checkAvailable(self, commandName=None, *, timeout=10.0, cacheDirectory=None):
        """Check AFNI without opening images or modifying its installation.

        None checks the installation using afni -ver.
        A command name, for example "3dvolreg", additionally runs -help.

        Returns True when the probes succeed.
        Ordinary failures warn at the default level and return False.
        KeyboardInterrupt and SystemExit propagate.

        lastAvailability records the outcome, reported version, executable
        directory, and probe provenance. lastResult holds the latest probe
        result, including a failed probe when available.

        Probe logs are retained through the shared runner. Inside an action,
        probes also enter its invocation history automatically.

        This is an explicit diagnostic method: failure returns False even
        inside an action. Scientific execution must not ignore that result.
        Automatic AFNI preflight for scientific commands is a later step.

        No automatic installation or PATH/shell-profile modification occurs.

        Examples:
            afni = pglAFNI()
            available = afni.checkAvailable()
            available = afni.checkAvailable("3dvolreg")
        """
        import math
        import os
        import sys
        from pathlib import Path

        from .pglMessages import pglMessages

        self.lastResult = None
        self.lastAvailability = {
            "schemaVersion": 1,
            "tool": "AFNI",
            "requestedCommand": None,
            "available": False,
            "reportedVersion": None,
            "executableDirectory": None,
            "probes": [],
            "error": None,
        }

        def runProbe(executable, arguments):
            """Use the recording engine without duplicate public warnings."""
            try:
                result = self._runCommand(
                    executable,
                    arguments=arguments,
                    timeout=timeout,
                    cacheDirectory=cacheDirectory,
                )
            except pglToolExecutionError as error:
                self.lastResult = error.result
                self.lastAvailability["probes"].append(error.result.provenance)
                raise

            self.lastResult = result
            self.lastAvailability["probes"].append(result.provenance)
            return result

        def readProbeText(result, limit=16384):
            """Read bounded diagnostic text without loading arbitrary log sizes."""
            sections = []

            for path in (result.stdoutPath, result.stderrPath):
                if path is None: continue

                with path.open("rb") as source:
                    content = source.read(limit + 1)

                text = content[:limit].decode("utf-8", errors="replace").strip()
                if len(content) > limit: text += "\n[Diagnostic text truncated]"
                if text: sections.append(text)

            return "\n".join(sections)

        try:
            if commandName is not None:
                if not isinstance(commandName, str) or not commandName.strip():
                    raise ValueError("commandName must be a nonblank executable name or None.")

                # Package selection belongs to directory, not a second path
                # silently pointing to another installation.
                if (
                    commandName != commandName.strip()
                    or commandName.startswith("-")
                    or commandName in (".", "..")
                    or any(character in commandName for character in ("/", "\\", "\x00", "\n", "\r"))
                ):
                    raise ValueError("Supply an executable name such as '3dvolreg'; use directory to select an installation.")

                self.lastAvailability["requestedCommand"] = commandName

            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite.")

            # Resolve once, then invoke the explicitly resolved executable.
            # Keep its invocation directory for sibling command discovery;
            # the base runner separately records the resolved target identity.
            effectiveEnvironment = dict(os.environ)

            if self.directory is not None:
                invocationDirectory = Path(self.directory)
                afniExecutable = self._resolveExecutable("afni", effectiveEnvironment)
            else:
                import shutil

                discovered = shutil.which("afni", path=effectiveEnvironment.get("PATH", os.defpath))
                if discovered is None:
                    # Route missing executable attempts through the recorder.
                    runProbe("afni", ["-ver"])
                    raise RuntimeError("AFNI discovery returned inconsistent results.")

                invocationDirectory = Path(discovered).absolute().parent
                afniExecutable = self._resolveExecutable(str(Path(discovered).absolute()), effectiveEnvironment)

            self.lastAvailability["executableDirectory"] = str(invocationDirectory)

            versionResult = runProbe(str(afniExecutable), ["-ver"])
            versionText = readProbeText(versionResult)

            if not versionText:
                raise RuntimeError("AFNI exited successfully but returned no version information.")

            self.lastAvailability["reportedVersion"] = versionText

            # Do not identify one AFNI installation and then quietly check a
            # command from a different installation farther down PATH.
            if commandName is not None and commandName != "afni":
                commandPath = invocationDirectory / commandName
                helpResult = runProbe(str(commandPath), ["-help"])

                if not readProbeText(helpResult):
                    raise RuntimeError(f"{commandName!r} exited successfully but returned no help information.")

            self.lastAvailability["available"] = True
            pglMessages.message(
                f"AFNI is available: {invocationDirectory}"
                + (f"\nCommand checked: {commandName}" if commandName is not None else "")
                + f"\n{versionText}"
            )
            return True

        except Exception as error:
            self.lastAvailability["error"] = {
                "errorType": type(error).__name__,
                "message": str(error),
            }

            details = str(error)

            # Include actual loader/library diagnostics, not just the exit
            # code. Preserve the complete logs for further inspection.
            if self.lastResult is not None and self.lastResult.status != "success":
                try:
                    probeText = readProbeText(self.lastResult, limit=4000)
                    if probeText: details += f"\n\nProbe output:\n{probeText}"
                except OSError:
                    details += "\nProbe logs could not be read."

            platformNote = ""
            if sys.platform == "win32":
                platformNote = (
                    "\n\nNative Windows execution is not an established AFNI "
                    "backend here. Use AFNI's supported Linux/WSL guidance. "
                    "This adapter does not translate Windows paths into WSL "
                    "paths or automatically launch commands through WSL."
                )

            pglMessages.warning(
                f"AFNI availability check failed.\n\n{details}\n\n"
                f"Official installation instructions:\n{self.installationURL}\n\n"
                "If AFNI is already installed, supply its executable directory:\n"
                "    afni = pglAFNI(directory='/path/to/afni/bin')\n"
                "    afni.checkAvailable()\n\n"
                "A launch failure can indicate missing system libraries or "
                "an incompatible executable architecture, even when the file exists."
                f"{platformNote}"
            )
            return False
        
    ################################################################
    # Describe the supported installation procedure
    ################################################################
    def installPlan(self, installPath="~/proj/afni"):
        """Print and return the AFNI installation plan without making changes.

        installPath is the installation root, not the executable directory.
        Defaults to ~/proj/afni.

        Returns a JSON-compatible plan dictionary, or None with a warning
        for unsupported platforms or invalid paths.

        This initial planner supports macOS 12+ on Apple Silicon.
        Other platform adapters can be added without changing the public API.

        Does not download scripts, create directories, alter configuration,
        probe external commands, or request confirmation.

        Describes the complete procedure, not the current installation stage.
        The installation coordinator will resolve saved progress separately.

        The plan is descriptive, not an executable command specification.
        Exact staged scripts and their identities must be reviewed before
        execution is offered.

        Usage:
            plan = afni.installPlan()
            plan = afni.installPlan(installPath="~/tools/afni")
        """
        import os
        import platform
        from pathlib import Path

        from .pglMessages import pglMessages

        try:
            pathText = os.fspath(installPath)
            if not isinstance(pathText, str) or not pathText.strip():
                raise ValueError("installPath must be a nonblank local path.")

            if (
                "://" in pathText
                or "::" in pathText
                or any(character in pathText for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("installPath must be an ordinary local path without NUL or newlines.")

            # Expand and normalize without creating the destination.
            root = Path(pathText).expanduser().resolve()
            home = Path.home().resolve()

            if root == Path(root.anchor) or root == home:
                raise ValueError("Choose a dedicated installation directory, not the filesystem root or your home directory.")

            if root.exists() and not root.is_dir():
                raise ValueError(f"installPath exists but is not a directory: {root}")

            system = platform.system()
            machine = platform.machine()
            macOSVersion = platform.mac_ver()[0] if system == "Darwin" else None

            if system != "Darwin" or machine.lower() != "arm64":
                raise ValueError(
                    f"Automated installation planning is not implemented for "
                    f"{system} / {machine}. No changes were made.\n"
                    f"Use the official platform instructions:\n{self.installationURL}"
                )

            try:
                majorVersion = int(macOSVersion.split(".", 1)[0])
            except (AttributeError, ValueError):
                raise ValueError("Could not establish the macOS version.") from None

            if majorVersion < 12:
                raise ValueError("This installation procedure requires macOS 12 or newer.")

            executableDirectory = root / "abin"
            buildDirectory = root / "build"
            installationDirectory = root / "install"

            for directory in (executableDirectory, buildDirectory, installationDirectory):
                if directory.is_symlink():
                    raise ValueError(f"Installation subdirectories must not be symlinks: {directory}")
                if directory.exists() and not directory.is_dir():
                    raise ValueError(f"Installation subdirectory is occupied by a file: {directory}")

            # These are the official procedure sources supplied and reviewed.
            # Their mutable URLs do not establish a pinned installer identity.
            sourceRoot = "https://raw.githubusercontent.com/afni/afni/master/src/other_builds"
            scriptNames = (
                "OS_notes.macos_12_ARM_a_admin_pt1.zsh",
                "OS_notes.macos_12_ARM_a_admin_pt2.zsh",
                "OS_notes.macos_12_ARM_b_user.tcsh",
                "OS_notes.macos_12_b_user.tcsh",
            )

            sources = [
                {"filename": name, "url": f"{sourceRoot}/{name}"}
                for name in scriptNames
            ]

            sourceText = "\n".join(f"  {item['filename']}\n    {item['url']}" for item in sources)
            existingText = (
                "The executable directory already exists. Its contents must be "
                "inspected before installation is offered; this plan does not "
                "authorize overwriting them."
                if executableDirectory.exists()
                else "The executable directory does not currently exist."
            )

            details = f"""AFNI INSTALLATION PLAN
======================

Platform
--------
System: macOS {macOSVersion}
Architecture: {machine}
Installation procedure: official macOS Apple Silicon setup, with explicit
user-level destination adaptations.

Locations
---------
Installation root: {root}
AFNI executables and resources: {executableDirectory}
Build workspace and AFNI-managed backups: {buildDirectory}
Installation scripts, records, and logs: {installationDirectory}

{existingText}

STAGE 1 — ADMINISTRATOR SETUP
----------------------------
Run the official administrator scripts in Terminal, in this order:

  zsh OS_notes.macos_12_ARM_a_admin_pt1.zsh
  zsh OS_notes.macos_12_ARM_a_admin_pt2.zsh

The reviewed procedure:
- Runs the official Homebrew installer.
- Adds Homebrew initialization to shell startup files.
- Disables Homebrew analytics.
- Installs Python, compilers, and AFNI build dependencies through Homebrew.
- Installs XQuartz.
- Adds Homebrew Python paths to shell startup files.
- Downloads and installs R 4.5.2 system-wide.

Package managers and downloaded installers can make additional dependency
changes. Administrator passwords are entered directly in Terminal.
PGL does not collect passwords.

STAGE 2 — RESTART THE MAC
------------------------
The official procedure requires a reboot after administrator setup.
PGL will not restart the machine automatically.

After reboot, reopen Python or your notebook and call:

  afni = pglAFNI()
  afni.install()

For a nondefault installation root, pass the same installPath when resuming.

The installation coordinator will use its saved record to offer the next
stage. It must not silently repeat administrator setup or mistake a restarted
Python process for a rebooted computer.

STAGE 3 — USER INSTALLATION
--------------------------
Follow the official user-level setup, adapting destination references to
the locations above.

The reviewed procedure:
- Sets Terminal/XQuartz focus-following preferences.
- Bootstraps AFNI supporting scripts and atlases.
- Updates bash, zsh, and tcsh startup files and command completion.
- Builds package macos_13_ARM using /usr/bin/gcc.
- Installs the build into the selected executable directory.
- Configures R_LIBS when unset and installs AFNI R packages.
- Adds /opt/X11/bin to shell paths when needed.
- Creates missing AFNI/SUMA user configuration.
- Generates AFNI command help information.

Destination adaptations:
- Bootstrap with -bindir pointing to the selected executable directory.
- Configure shell paths with init_user_dotfiles.py -dir_bin for that directory.
- Build with explicit -abin and -build_root destinations.
- Replace other hard-coded ~/abin references consistently.
- Resolve subsequent AFNI commands from this installation.

The exact adapted scripts must be shown before approval. They are derived
from the official procedure, not byte-for-byte official scripts.

Homebrew, XQuartz, system R, shell startup files, user configuration, and
R libraries retain their normal locations outside the installation root.
HOME is not changed and no hidden ~/abin symlink is created.

EXISTING INSTALLATIONS AND BACKUPS
---------------------------------
AFNI's builder can back up and replace an existing executable installation.
It can also remove older AFNI-managed backups.
Build workspaces may be updated or replaced on subsequent builds.
Repeating an installation stage requires renewed approval.

VERIFICATION
------------
Check AFNI's reported version and required command availability.
Run afni_system_check.py -check_all and retain its diagnostics.
Review its recommendations and separately test AFNI/SUMA GUI functionality.

A script exiting zero or printing "setup is complete" does not establish
that all dependencies and scientific commands work.

The running notebook does not inherit shell-profile changes automatically.
A new terminal/kernel may be required.

CONSENT AND RECOVERY
-------------------
This method only describes the procedure. Nothing has been installed.
Before execution, PGL will print and display the actual stage plan and ask
for confirmation.

Cancelling before launch makes no installation changes.
Cancelling after launch does not undo completed changes.
Installation is not a transaction.

Official scripts download additional scripts, packages, and source code.
Recording the initial script hashes does not pin all downstream downloads.

Optional Bootcamp data are not included.

OFFICIAL PROCEDURE SOURCES
--------------------------
{sourceText}

Installation documentation:
  {self.installationURL}
"""

            plan = {
                "schemaVersion": 1,
                "kind": "installationOverview",
                "tool": "AFNI",
                "platform": {
                    "system": system,
                    "machine": machine,
                    "version": macOSVersion,
                },
                "installPath": str(root),
                "executableDirectory": str(executableDirectory),
                "buildDirectory": str(buildDirectory),
                "installationDirectory": str(installationDirectory),
                "sources": sources,
                "details": details,
            }

        except (TypeError, ValueError, OSError, RuntimeError) as error:
            pglMessages.warning(f"AFNI installation plan was not created: {error}")
            return None

        pglMessages.print(details)
        return plan
    
    ################################################################
    # Start or resume a staged AFNI installation
    ################################################################
    def install(self, installPath="~/proj/afni", *, stage="auto"):
        """Start, resume, review, or check a managed AFNI installation.

        installPath:
            Installation root; defaults to ~/proj/afni.

        stage:
            auto: select from saved progress.
            admin: request initial administrator setup.
            user: request user setup, allowing separately installed
                prerequisites after explicit acknowledgment.

        Normal progression:
            administrator setup → reboot → user setup
            → diagnostic review → verified installation

        Returns:
            A stage/state dictionary when launch was requested or installation
            verification succeeded.
            None on cancellation, refusal, or ordinary failure.

        Terminal launch is not installation success.
        Failed/interrupted/uncertain stages require explicit recovery review;
        they are not silently retried.

        After successful verification, this instance uses installPath/abin.
        Existing scientific session data and action history are not modified.

        Ordinary failures warn without escaping to the caller.
        KeyboardInterrupt and SystemExit propagate.
        """
        from .pglMessages import pglMessages
        self.printInstallationNotice()
        try:
            installation = pglAFNIInstallation(installPath)
            decision = installation.getResumeStatus(stage=stage)
            nextStage = decision["nextStage"]

            if nextStage is None:
                state = installation.readState()
                message = installation.continuationMessage(state) if state is not None else decision["message"]
                pglMessages.warning(message)
                return None

            if nextStage == "admin":
                return installation.launchAdmin()

            if nextStage == "user":
                return installation.launchUser(stage=stage)

            if nextStage == "review":
                state = installation.reviewInstallation()
                if state is not None:
                    self.directory = str(installation.root / "abin")
                return state

            if nextStage == "check":
                if not self.checkAvailable(
                    "3dvolreg",
                    installPath=installation.root,
                ):
                    return None

                # Do not let an explicitly selected unrelated installation
                # stand in for verification of the requested installPath.
                target = str(installation.root / "abin")
                if self.directory is not None:
                    from pathlib import Path

                    if Path(self.directory).expanduser().resolve() != installation.root / "abin":
                        probe = pglAFNI(directory=target)
                        if not probe._checkAvailable("3dvolreg"): return None
                        self.lastResult = probe.lastResult
                        self.lastAvailability = deepcopy(probe.lastAvailability)

                self.directory = target
                pglMessages.message(f"AFNI is ready at {target}. No installation commands were run.")
                return installation.readState()

            raise ValueError(f"Unsupported installation continuation: {nextStage!r}")

        except Exception as error:
            pglMessages.warning(f"AFNI installation did not advance: {type(error).__name__}: {error}")
            return None
        
    ################################################################
    # Estimate and interpret within-run rigid motion
    ################################################################
    def volreg(self, inputImage, *, referenceVolume=0, twoPass=True, cacheDirectory=None, timeout=None, sourceStageID=None, physicalSpaceID=None):
        """Estimate within-run motion and interpret its native AFNI matrices.

        Uses the existing _volreg() execution engine, then reads AFNI geometry
        attributes and converts matrices to explicit voxel sampling maps.

        referenceVolume is a zero-based index in the supplied image.
        This wrapper processes all supplied volumes in their existing order.

        Initial interpretation support:
            4D NIfTI, millimeter spatial units, coded qform.
            This wrapper does not require qform and sform to agree.
            Reference and input share the same image grid.

        sourceStageID:
            Explicit image-stage identity supplied by a scientific action.
            None uses a content-based identity for standalone execution.

        physicalSpaceID:
            Explicit physical reference identity. None uses a source-specific
            identity rather than assuming unrelated scans share a frame.

        AFNI-native coordinates are interpreted through pglMRITransforms using
        AFNI's queried geometry attributes, not treated as qform coordinates.

        Always estimates with -prefix NULL. No corrected image is generated.

        Returns pglToolResult with native outputs and interpreted transforms
        under provenance["invocation"]["result"]["interpretation"].

        Freeform failures warn and return None. Within an action, failures
        propagate internally so scientific execution stops immediately.
        """
        from contextlib import nullcontext
        from pathlib import Path

        import nibabel as nib
        import numpy as np

        from .pglMessages import pglMessages
        from .pglMRI import pglMRITransforms

        self.lastResult = None
        recorder = getExecutionRecorder()
        standalone = recorder is None
        if standalone: recorder = pglExecutionRecorder()

        context = recorder.activate() if standalone else nullcontext()

        try:
            with context:
                source = Path(inputImage).expanduser().resolve()
                parameters = {
                    "referenceVolume": referenceVolume,
                    "twoPass": twoPass,
                    "timeout": timeout,
                    "sourceStageID": sourceStageID,
                    "physicalSpaceID": physicalSpaceID,
                }
                invocationID = recorder.beginInvocation({
                    "operation": "interpretedWithinRunMotion",
                    "adapter": "pgl.AFNI.3dvolreg.sharedGrid.v1",
                    "inputPath": str(source),
                    "parameters": parameters,
                })
                result = pglToolResult(invocationID)
                details = {"outputs": {}, "interpretation": None}
                errorRaised = None
                errorRecord = None
                previousIDs = {
                    record["invocationID"] for record in recorder.exportInvocations()
                }

                try:
                    identity = self._fileIdentity(source)
                    image = nib.load(source)

                    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
                        raise TypeError("Expected a single-file NIfTI image.")
                    if len(image.shape) != 4 or image.shape[3] < 2:
                        raise ValueError("Within-run motion requires at least two volumes.")
                    if image.header.get_xyzt_units()[0] != "mm":
                        raise ValueError("This initial AFNI interpretation adapter requires millimeter spatial units.")

                    qform, _ = image.get_qform(coded=True)
                    if qform is None:
                        raise ValueError("A coded acquisition qform is required by the current frame interpretation.")

                    # Do not compare qform with sform. They may describe
                    # different coordinate systems for the same voxel array.
                    # Preserve AFNI-native-frame interpretation below.
                    stageID = sourceStageID or f"image:{identity['sha256']}"
                    spaceID = physicalSpaceID or f"sourcePhysical:{identity['sha256']}"
                    details["sourceIdentity"] = identity
                    details["sourceStageID"] = stageID

                    native = self._volreg(
                        source,
                        referenceVolume=referenceVolume,
                        twoPass=twoPass,
                        cacheDirectory=cacheDirectory,
                        timeout=timeout,
                    )
                    result.returnCode = native.returnCode
                    result.logDirectory = native.logDirectory
                    result.stdoutPath = native.stdoutPath
                    result.stderrPath = native.stderrPath

                    nativeDetails = native.provenance["invocation"]["result"]
                    executableDirectory = Path(nativeDetails["executableDirectory"])
                    attributes = {}

                    # Query the same installation used for estimation.
                    for name, count in (
                        ("ORIENT_SPECIFIC", 3),
                        ("DELTA", 3),
                        ("ORIGIN", 3),
                        ("IJK_TO_DICOM_REAL", 12),
                    ):
                        query = self._runCommand(
                            executableDirectory / "3dAttribute",
                            arguments=[name, str(source)],
                            timeout=15,
                            cacheDirectory=result.logDirectory,
                        )
                        text = query.stdoutPath.read_text(encoding="utf-8", errors="replace")
                        values = np.array([float(value) for value in text.split()])
                        if values.shape != (count,) or not np.isfinite(values).all():
                            raise ValueError(f"Unexpected AFNI attribute {name}.")
                        attributes[name] = values.tolist()

                    rows = np.loadtxt(
                        native.outputs["matrices"]["localPath"],
                        comments="#",
                        ndmin=2,
                    )

                    # AFNI's native millimeter frame is not the voxel-index
                    # frame or simply the qform world frame. Keep the explicit
                    # frame conversion in pglMRITransforms.
                    interpretation = pglMRITransforms.interpretAFNIVolreg(
                        rows,
                        attributes=attributes,
                        geometryID=stageID,
                        physicalSpaceID=spaceID,
                        spatialShape=list(image.shape[:3]),
                        voxelToPhysicalMM=qform,
                        volumeIndices=list(range(image.shape[3])),
                        referenceVolumeIndex=referenceVolume,
                        afniVersion=nativeDetails["reportedVersion"],
                    )

                    if self._fileIdentity(source) != identity:
                        raise ValueError("Source bytes changed during motion estimation/interpretation.")

                    details.update({
                        "outputs": native.outputs,
                        "interpretation": interpretation,
                        "reportedVersion": nativeDetails["reportedVersion"],
                        "executableDirectory": str(executableDirectory),
                        "sourceUnchanged": True,
                    })
                    result.status = "success"

                except BaseException as error:
                    errorRaised = error
                    result.status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                    errorRecord = {"errorType": type(error).__name__, "message": str(error)}

                    if isinstance(error, pglToolExecutionError):
                        failed = error.result
                        result.returnCode = failed.returnCode
                        result.stdoutPath = failed.stdoutPath
                        result.stderrPath = failed.stderrPath
                        if result.logDirectory is None: result.logDirectory = failed.logDirectory

                finally:
                    details["childInvocations"] = [
                        record for record in recorder.exportInvocations()
                        if record["invocationID"] not in previousIDs
                    ]
                    recorder.finishInvocation(
                        invocationID,
                        status=result.status,
                        result=details,
                        error=errorRecord,
                    )
                    if standalone: recorder.close()

                    invocation = next(
                        record for record in recorder.exportInvocations()
                        if record["invocationID"] == invocationID
                    )
                    result._provenance = {
                        "schemaVersion": 1,
                        "environment": recorder.exportEnvironment(),
                        "invocation": invocation,
                    }
                    self.lastResult = result

                if errorRaised is not None:
                    if not isinstance(errorRaised, Exception): raise errorRaised
                    raise pglToolExecutionError(
                        f"AFNI motion estimation/interpretation failed: {errorRecord['message']}",
                        result,
                    ) from errorRaised

                return result

        except Exception as error:
            if isinstance(error, pglToolExecutionError):
                self.lastResult = error.result

            active = getExecutionRecorder()
            if active is not None and active.actionID is not None:
                raise

            pglMessages.warning(f"AFNI motion estimation stopped: {type(error).__name__}: {error}")
            return None

    ################################################################
    # Recorded motion-estimation implementation
    ################################################################
    def _volreg(self, inputImage, *, referenceVolume=0, twoPass=True, cacheDirectory=None, timeout=None):
        """Execute estimation and validate outputs before reporting success.

        Creates one enclosing scientific-tool invocation plus the ordinary
        version and subprocess invocation records.

        Displays AFNI's verbose estimation output while retaining full logs.
        This is native tool output, not an estimated percentage or ETA.

        Input/output hashes identify file bytes, not scientific correctness.
        Source mutation is checked before/after execution, but concurrent
        source/tool modification is unsupported.

        Header inspection remains lazy with respect to voxel arrays.
        The AFNI subprocess necessarily reads image data for estimation.
        """
        import math
        import os
        import shutil
        import tempfile
        from contextlib import nullcontext
        from pathlib import Path

        import nibabel as nib
        import numpy as np

        if type(referenceVolume) is not int or referenceVolume < 0:
            raise ValueError("referenceVolume must be a nonnegative integer.")
        if type(twoPass) is not bool:
            raise ValueError("twoPass must be True or False.")
        if timeout is not None:
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout must be positive and finite, or None.")

        source = Path(inputImage).expanduser().resolve()
        if not source.is_file() or not source.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError("inputImage must be an existing local .nii or .nii.gz file.")

        # AFNI interprets selectors in dataset arguments. Reject ambiguous
        # literal filenames rather than silently treating them as selectors.
        if any(character in str(source) for character in ("[", "]", "<", ">", "\x00", "\n", "\r")):
            raise ValueError("The image path contains characters ambiguous in an AFNI dataset argument.")

        cacheRoot = None
        if cacheDirectory is not None:
            cacheText = os.fspath(cacheDirectory)
            if (
                not isinstance(cacheText, str) or not cacheText.strip()
                or "://" in cacheText or "::" in cacheText
                or any(character in cacheText for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("cacheDirectory must be an ordinary local path.")
            cacheRoot = Path(cacheText).expanduser().resolve()

        recorder = getExecutionRecorder()
        standalone = recorder is None
        if standalone: recorder = pglExecutionRecorder()

        # Standalone version/subprocess calls share this private context.
        # Within an action, retain the existing action recorder.
        context = recorder.activate() if standalone else nullcontext()

        with context:
            invocationID = recorder.beginInvocation({
                "adapter": f"{type(self).__module__}.{type(self).__qualname__}",
                "operation": "volreg",
                "parameters": {
                    "referenceVolume": referenceVolume,
                    "twoPass": twoPass,
                    "interpolation": "Fourier",
                    "writeCorrectedImage": False,
                    "verbose": True,
                },
                "inputPath": str(source),
                "timeoutSeconds": timeout,
                "dependencyCapture": {
                    "status": "declaredInputsAndOutputs",
                    "limitations": [
                        "Shared-library and implicit AFNI configuration contents are not captured.",
                        "Matrix interpretation/application is not implemented by this wrapper.",
                    ],
                },
            })

            result = pglToolResult(invocationID)
            details = {
                "inputs": {},
                "outputs": {},
                "childInvocationIDs": [],
                "reportedVersion": None,
                "workingDirectory": None,
                "sourceUnchanged": None,
            }
            executionError = None
            failureRecord = None

            try:
                # Hash before header inspection so metadata and estimation
                # are tied to an identified source, then check again afterward.
                sourceIdentity = self._fileIdentity(source)
                image = nib.load(str(source), mmap=True)

                if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
                    raise TypeError("Expected a single-file NIfTI image.")
                if len(image.shape) != 4 or image.shape[3] < 2:
                    raise ValueError("Motion estimation requires a 4D image with at least two volumes.")
                if referenceVolume >= image.shape[3]:
                    raise ValueError(f"referenceVolume is outside the supplied image's {image.shape[3]} volumes.")
                if image.get_data_dtype().kind not in "iuf":
                    raise ValueError("Motion estimation requires real-valued numerical image data.")

                volumeCount = int(image.shape[3])
                details["inputs"]["image"] = {
                    "localPath": str(source),
                    **sourceIdentity,
                    "shape": list(image.shape),
                    "referenceVolume": referenceVolume,
                }

                # No header normalization or assignment. AFNI-native frame
                # interpretation remains the responsibility of volreg().
                del image

                # Match availability selection without probing one installation
                # and subsequently executing another found through PATH.
                managed = pglAFNIInstallation()
                managedDirectory = managed.root / "abin"
                selectedDirectory = None if self.directory is None else Path(self.directory).expanduser().resolve()

                if selectedDirectory is None or selectedDirectory == managedDirectory.resolve():
                    state = managed.readState()
                    if state is not None and state["status"] != "verified":
                        raise RuntimeError(managed.continuationMessage(state))
                    if state is not None or managedDirectory.exists():
                        selectedDirectory = managedDirectory

                if selectedDirectory is None:
                    discovered = shutil.which("afni")
                    if discovered is None:
                        raise FileNotFoundError("AFNI is unavailable. Run afni.install() or supply its executable directory.")
                    selectedDirectory = Path(discovered).absolute().parent

                # Use absolute command paths throughout this execution.
                runtimeEnvironment = dict(os.environ)
                afniExecutable = self._resolveExecutable(str(selectedDirectory / "afni"), runtimeEnvironment)
                motionExecutable = self._resolveExecutable(str(selectedDirectory / "3dvolreg"), runtimeEnvironment)

                if cacheRoot is not None: cacheRoot.mkdir(parents=True, exist_ok=True)
                outputDirectory = Path(tempfile.mkdtemp(
                    prefix="pgl-volreg-",
                    dir=None if cacheRoot is None else str(cacheRoot),
                ))
                details["workingDirectory"] = str(outputDirectory)
                result.logDirectory = outputDirectory

                versionResult = self._runCommand(
                    str(afniExecutable),
                    arguments=["-ver"],
                    timeout=10.0,
                    cacheDirectory=outputDirectory,
                )
                details["childInvocationIDs"].append(versionResult.invocationID)

                versionSections = []
                for path in (versionResult.stdoutPath, versionResult.stderrPath):
                    with path.open("rb") as handle:
                        content = handle.read(65537)
                    if len(content) > 65536:
                        raise ValueError("AFNI version output exceeds the supported size.")
                    text = content.decode("utf-8", errors="replace").strip()
                    if text: versionSections.append(text)

                versionText = "\n".join(versionSections)
                if not versionText: raise ValueError("AFNI returned no version information.")
                details["reportedVersion"] = versionText
                details["executableDirectory"] = str(selectedDirectory)

                matrixPath = outputDirectory / "motion.aff12.1D"
                parameterPath = outputDirectory / "motion.1D"

                # Verbosity affects reporting, not the requested registration.
                arguments = ["-Fourier", "-verbose"]
                if twoPass: arguments.append("-twopass")
                arguments.extend([
                    "-base", f"{source}[{referenceVolume}]",
                    "-1Dfile", str(parameterPath),
                    "-1Dmatrix_save", str(matrixPath),
                    "-prefix", "NULL",
                    str(source),
                ])

                commandResult = self._runCommand(
                    str(motionExecutable),
                    arguments=arguments,
                    cwd=outputDirectory,
                    timeout=timeout,
                    cacheDirectory=outputDirectory,
                    streamOutput=True,
                )
                details["childInvocationIDs"].append(commandResult.invocationID)
                result.returnCode = commandResult.returnCode
                result.stdoutPath = commandResult.stdoutPath
                result.stderrPath = commandResult.stderrPath

                # A successful process is not enough: require one finite row
                # per supplied volume and the declared AFNI output shapes.
                def validateOutput(path, columns):
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f"Expected a regular motion output: {path}")

                    values = np.loadtxt(path, comments="#", ndmin=2)
                    if values.shape != (volumeCount, columns) or not np.isfinite(values).all():
                        raise ValueError(
                            f"{path.name}: expected {volumeCount} finite rows with "
                            f"{columns} columns; found shape {values.shape}."
                        )

                    return {
                        "localPath": str(path),
                        **self._fileIdentity(path),
                        "rows": volumeCount,
                        "columns": columns,
                    }

                matrices = validateOutput(matrixPath, 12)
                parameters = validateOutput(parameterPath, 6)

                actualSourceIdentity = self._fileIdentity(source)
                details["sourceUnchanged"] = actualSourceIdentity == sourceIdentity
                if not details["sourceUnchanged"]:
                    raise ValueError("The source image changed during motion estimation.")

                # Publish declared outputs only after all validation succeeds.
                details["outputs"] = {
                    "matrices": {
                        **matrices,
                        "format": "AFNI aff12.1D",
                        "semantics": "Native 3dvolreg -1Dmatrix_save output; not converted.",
                    },
                    "motionParameters": {
                        **parameters,
                        "format": "AFNI 3dvolreg 1Dfile",
                        "columnsDescription": ["roll", "pitch", "yaw", "dS", "dL", "dP"],
                        "columnUnits": ["degree", "degree", "degree", "mm", "mm", "mm"],
                    },
                }
                result.status = "success"

            except BaseException as error:
                executionError = error
                result.status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"

                if isinstance(error, pglToolExecutionError):
                    child = error.result
                    if child.invocationID not in details["childInvocationIDs"]:
                        details["childInvocationIDs"].append(child.invocationID)
                    result.returnCode = child.returnCode
                    result.stdoutPath = child.stdoutPath
                    result.stderrPath = child.stderrPath

                failureRecord = {
                    "errorType": type(error).__name__,
                    "message": str(error),
                }

            finally:
                # Include child command identities/logs in standalone results
                # as well as the full action recorder. Do not duplicate the
                # enclosing invocation inside itself.
                children = set(details["childInvocationIDs"])
                details["childInvocations"] = [
                    record for record in recorder.exportInvocations()
                    if record["invocationID"] in children
                ]

                recorder.finishInvocation(
                    invocationID,
                    status=result.status,
                    result=details,
                    error=failureRecord,
                )
                if standalone: recorder.close()

                invocation = next(
                    record for record in recorder.exportInvocations()
                    if record["invocationID"] == invocationID
                )
                result._provenance = {
                    "schemaVersion": 1,
                    "environment": recorder.exportEnvironment(),
                    "invocation": invocation,
                }

            if executionError is not None:
                if not isinstance(executionError, Exception): raise executionError

                location = f" Retained working files: {result.logDirectory}" if result.logDirectory is not None else ""
                raise pglToolExecutionError(
                    f"Motion estimation failed: {failureRecord['message']}{location}",
                    result,
                ) from executionError

            return result
        
    ################################################################
    # Estimate one fixed run-reference alignment
    ################################################################
    def alignRunReference(self, inputImage, referenceImage, *, inputVolume=0, referenceVolume=0, inputStageID=None, referenceStageID=None, physicalSpaceID=None, cacheDirectory=None, timeout=None):
        """Estimate one rigid mapping from a common reference to a run reference.

        inputImage/referenceImage:
            Explicit local NIfTI sources on matching acquisition grids.
            Acquisition-grid compatibility uses qform and spatial shape.
            This wrapper does not require qform and sform to agree.

        inputVolume/referenceVolume:
            Zero-based volume indices in the supplied files.
            A 3D image supports index zero only.

        Stage IDs:
            Explicit source-stage identities supplied by the owning action.
            Standalone calls default to content-based identities.

        physicalSpaceID:
            Shared coordinate reference established by the calling action.
            A standalone call uses the selected reference source as its anchor.

        Returns:
            pglToolResult with native matrix/motion-parameter outputs.
            provenance["invocation"]["result"]["alignment"] contains the
            native and voxel-coordinate reference-to-input sampling mappings.

        Runs 3dvolreg on one selected input volume with -prefix NULL.
        Does not resample images, choose scans, group runs, or register products.

        AFNI-native coordinate conversion remains delegated to
        pglMRITransforms using AFNI's queried geometry attributes.

        Freeform failures warn and return None.
        Inside an action, failures propagate internally to stop execution.
        KeyboardInterrupt/SystemExit propagate after recording.

        Initial support is the matching-grid, millimeter NIfTI case.
        Numerical grid agreement does not establish anatomical alignment.
        """
        import math
        import os
        import shutil
        import tempfile
        from contextlib import nullcontext
        from pathlib import Path

        import nibabel as nib
        import numpy as np

        from .pglMessages import pglMessages
        from .pglMRI import pglMRITransforms

        self.lastResult = None
        activeRecorder = getExecutionRecorder()
        recorder = activeRecorder
        standalone = recorder is None

        try:
            for name, index in (("inputVolume", inputVolume), ("referenceVolume", referenceVolume)):
                if type(index) is not int or index < 0:
                    raise ValueError(f"{name} must be a nonnegative integer.")

            if timeout is not None:
                if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError("timeout must be positive and finite.")

            for name, value in (
                ("inputStageID", inputStageID),
                ("referenceStageID", referenceStageID),
                ("physicalSpaceID", physicalSpaceID),
            ):
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                    or any(character in value for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError(f"{name} must be a nonblank string or None.")

            def sourcePath(value):
                path = Path(value).expanduser().resolve()
                if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
                    raise ValueError("Alignment inputs must be existing local NIfTI files.")
                if any(character in str(path) for character in ("[", "]", "<", ">", "\x00", "\n", "\r")):
                    raise ValueError("An image path contains AFNI dataset-selector characters.")
                return path

            movingPath = sourcePath(inputImage)
            basePath = sourcePath(referenceImage)

            cacheRoot = None
            if cacheDirectory is not None:
                text = os.fspath(cacheDirectory)
                if (
                    not isinstance(text, str) or not text.strip()
                    or "://" in text or "::" in text
                    or any(character in text for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError("cacheDirectory must be an ordinary local path.")
                cacheRoot = Path(text).expanduser().resolve()

            if standalone: recorder = pglExecutionRecorder()

            with recorder.activate() if standalone else nullcontext():
                invocationID = recorder.beginInvocation({
                    "operation": "runReferenceAlignment",
                    "adapter": "pgl.AFNI.3dvolreg.crossRunSharedGrid.v1",
                    "inputPath": str(movingPath),
                    "referencePath": str(basePath),
                    "inputVolume": inputVolume,
                    "referenceVolume": referenceVolume,
                    "parameters": {
                        "twoPass": True,
                        "interpolation": "Fourier",
                        "writeCorrectedImage": False,
                    },
                    "timeoutSeconds": timeout,
                })
                previousIDs = {
                    record["invocationID"] for record in recorder.exportInvocations()
                }
                result = pglToolResult(invocationID)
                details = {"inputs": {}, "outputs": {}, "alignment": None}
                executionError = None
                failureRecord = None

                try:
                    def inspectSource(path, volume):
                        identity = self._fileIdentity(path)
                        image = nib.load(path)

                        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
                            raise ValueError("Expected a single-file NIfTI image.")
                        if len(image.shape) not in (3, 4):
                            raise ValueError("Alignment sources must be 3D or 4D.")
                        count = image.shape[3] if len(image.shape) == 4 else 1
                        if volume >= count: raise ValueError(f"Reference volume is outside {path.name}.")
                        if image.header.get_xyzt_units()[0] != "mm":
                            raise ValueError("This adapter requires millimeter spatial units.")

                        qform, _ = image.get_qform(coded=True)
                        if qform is None: raise ValueError("Alignment requires a coded acquisition qform.")
                        qform = pglMRITransforms._matrix(qform, "acquisition qform")

                        shape = list(image.shape[:3])
                        corners = np.array([
                            [x, y, z, 1.0]
                            for x in (0, shape[0] - 1)
                            for y in (0, shape[1] - 1)
                            for z in (0, shape[2] - 1)
                        ]).T

                        # Compare acquisition prescriptions using qform.
                        # A separate sform mapping does not invalidate them.
                        spacing = np.asarray(image.header.get_zooms()[:3])
                        if not np.allclose(
                            spacing, np.linalg.norm(qform[:3, :3], axis=0),
                            rtol=0, atol=0.001,
                        ):
                            raise ValueError("Header voxel sizes disagree with qform.")

                        return identity, shape, qform, corners

                    movingIdentity, movingShape, movingAffine, movingCorners = inspectSource(movingPath, inputVolume)
                    baseIdentity, baseShape, baseAffine, baseCorners = inspectSource(basePath, referenceVolume)

                    gridError = float(np.linalg.norm(
                        ((movingAffine - baseAffine) @ baseCorners)[:3], axis=0,
                    ).max())
                    if movingShape != baseShape or gridError > 0.001:
                        raise ValueError(
                            "Cross-run estimation currently requires matching acquisition grids. "
                            "The action must group compatible runs before invoking this adapter."
                        )

                    movingStage = inputStageID or f"image:{movingIdentity['sha256']}"
                    baseStage = referenceStageID or f"image:{baseIdentity['sha256']}"
                    spaceID = physicalSpaceID or f"alignmentReference:{baseStage}"

                    details["inputs"] = {
                        "moving": {
                            "localPath": str(movingPath),
                            "sourceStageID": movingStage,
                            "volumeIndex": inputVolume,
                            **movingIdentity,
                        },
                        "reference": {
                            "localPath": str(basePath),
                            "sourceStageID": baseStage,
                            "volumeIndex": referenceVolume,
                            **baseIdentity,
                        },
                    }
                    details["gridComparisonErrorMM"] = gridError

                    # Select one installation and use explicit paths afterward.
                    selectedDirectory = None if self.directory is None else Path(self.directory).expanduser().resolve()
                    managed = pglAFNIInstallation()
                    managedDirectory = managed.root / "abin"

                    if selectedDirectory is None or selectedDirectory == managedDirectory.resolve():
                        state = managed.readState()
                        if state is not None and state["status"] != "verified":
                            raise RuntimeError(managed.continuationMessage(state))
                        if state is not None or managedDirectory.exists():
                            selectedDirectory = managedDirectory

                    if selectedDirectory is None:
                        discovered = shutil.which("afni")
                        if discovered is None:
                            raise FileNotFoundError("AFNI is unavailable. Run afni.install() or specify its executable directory.")
                        selectedDirectory = Path(discovered).absolute().parent

                    if cacheRoot is not None: cacheRoot.mkdir(parents=True, exist_ok=True)
                    outputDirectory = Path(tempfile.mkdtemp(
                        prefix="pgl-run-alignment-",
                        dir=None if cacheRoot is None else str(cacheRoot),
                    ))
                    result.logDirectory = outputDirectory
                    details["workingDirectory"] = str(outputDirectory)

                    versionResult = self._runCommand(
                        selectedDirectory / "afni",
                        arguments=["-ver"],
                        timeout=10,
                        cacheDirectory=outputDirectory,
                    )
                    versionParts = []
                    for path in (versionResult.stdoutPath, versionResult.stderrPath):
                        with path.open("rb") as handle: content = handle.read(65537)
                        if len(content) > 65536: raise ValueError("AFNI version output is unexpectedly large.")
                        text = content.decode("utf-8", errors="replace").strip()
                        if text: versionParts.append(text)
                    version = "\n".join(versionParts)
                    if not version: raise ValueError("AFNI returned no version information.")
                    details["reportedVersion"] = version

                    def readAttributes(path):
                        attributes = {}
                        for name, count in (
                            ("ORIENT_SPECIFIC", 3), ("DELTA", 3),
                            ("ORIGIN", 3), ("IJK_TO_DICOM_REAL", 12),
                        ):
                            query = self._runCommand(
                                selectedDirectory / "3dAttribute",
                                arguments=[name, str(path)],
                                timeout=15,
                                cacheDirectory=outputDirectory,
                            )
                            text = query.stdoutPath.read_text(encoding="utf-8", errors="replace")
                            values = np.array([float(value) for value in text.split()])
                            if values.shape != (count,) or not np.isfinite(values).all():
                                raise ValueError(f"Invalid AFNI attribute {name}.")
                            attributes[name] = values.tolist()
                        return attributes

                    # Reuse the tested frame construction through an identity
                    # mapping. These temporary identity transforms are not
                    # registration estimates or published products.
                    def makeFrames(path, shape, affine, stageID):
                        attributes = readAttributes(path)
                        identityRow = np.eye(4)[:3].reshape(1, 12)
                        interpreted = pglMRITransforms.interpretAFNIVolreg(
                            identityRow,
                            attributes=attributes,
                            geometryID=stageID,
                            physicalSpaceID=spaceID,
                            spatialShape=shape,
                            voxelToPhysicalMM=affine,
                            volumeIndices=[0],
                            referenceVolumeIndex=0,
                            afniVersion=version,
                        )
                        return {
                            "attributes": attributes,
                            "native": interpreted["nativeFrame"],
                            "voxel": interpreted["voxelFrame"],
                        }

                    movingFrames = makeFrames(movingPath, movingShape, movingAffine, movingStage)
                    baseFrames = makeFrames(basePath, baseShape, baseAffine, baseStage)

                    matrixPath = outputDirectory / "run-alignment.aff12.1D"
                    parameterPath = outputDirectory / "run-alignment.1D"

                    command = self._runCommand(
                        selectedDirectory / "3dvolreg",
                        arguments=[
                            "-Fourier", "-twopass",
                            "-base", f"{basePath}[{referenceVolume}]",
                            "-1Dfile", str(parameterPath),
                            "-1Dmatrix_save", str(matrixPath),
                            "-prefix", "NULL",
                            f"{movingPath}[{inputVolume}]",
                        ],
                        cwd=outputDirectory,
                        timeout=timeout,
                        cacheDirectory=outputDirectory,
                    )
                    result.returnCode = command.returnCode
                    result.stdoutPath = command.stdoutPath
                    result.stderrPath = command.stderrPath

                    def readOutput(path, columns):
                        if path.is_symlink() or not path.is_file():
                            raise ValueError(f"Expected regular output file: {path}")
                        values = np.loadtxt(path, comments="#", ndmin=2)
                        if values.shape != (1, columns) or not np.isfinite(values).all():
                            raise ValueError(f"{path.name} must contain one finite row of {columns} values.")
                        return values, {
                            "localPath": str(path),
                            **self._fileIdentity(path),
                            "rows": 1,
                            "columns": columns,
                        }

                    rows, matrixOutput = readOutput(matrixPath, 12)
                    _, parameterOutput = readOutput(parameterPath, 6)
                    matrix = np.eye(4)
                    matrix[:3] = rows.reshape(3, 4)
                    linear = matrix[:3, :3]

                    if (
                        not np.allclose(linear.T @ linear, np.eye(3), rtol=0, atol=0.0001)
                        or abs(np.linalg.det(linear) - 1) > 0.0001
                    ):
                        raise ValueError("Cross-run output is not a rigid transformation.")

                    # Preserve the explicit conversion from AFNI-native
                    # reference-to-input coordinates to voxel sampling maps.
                    native = pglMRITransforms.makeTransform(
                        matrix,
                        domainFrame=baseFrames["native"],
                        codomainFrame=movingFrames["native"],
                        purpose="sampling",
                        interpretationStatus="validated",
                        adapter="pgl.AFNI.3dvolreg.crossRunSharedGrid.v1",
                    )
                    voxel = pglMRITransforms.convert(
                        native,
                        domainFrame=baseFrames["voxel"],
                        codomainFrame=movingFrames["voxel"],
                    )

                    for path, identity in ((movingPath, movingIdentity), (basePath, baseIdentity)):
                        if self._fileIdentity(path) != identity:
                            raise ValueError("An alignment source changed during execution.")

                    details["outputs"] = {
                        "matrices": {**matrixOutput, "format": "AFNI aff12.1D"},
                        "motionParameters": {**parameterOutput, "format": "AFNI 3dvolreg 1Dfile"},
                    }
                    details["alignment"] = {
                        "schemaVersion": 1,
                        "kind": "AFNIRunReferenceAlignment",
                        "adapter": "pgl.AFNI.3dvolreg.crossRunSharedGrid.v1",
                        "afniVersion": version,
                        "inputStageID": movingStage,
                        "referenceStageID": baseStage,
                        "inputVolume": inputVolume,
                        "referenceVolume": referenceVolume,
                        "inputAttributes": movingFrames["attributes"],
                        "referenceAttributes": baseFrames["attributes"],
                        "nativeTransform": native,
                        "voxelTransform": voxel,
                        "validationScope": (
                            "Matching acquisition grids. The existing adapter records "
                            "synthetic cross-run rotation/translation and composition "
                            "tests with AFNI 26.3.11 on macOS ARM. Those tests do not "
                            "establish coverage of differing qform/sform mappings. "
                            "Registration quality must be evaluated separately."
                        ),
                        "description": (
                            "One fixed alignment between run reference images. "
                            "For a voxel location in the common reference, the "
                            "converted matrix gives the sampling location in "
                            "this run's reference image. No images were corrected "
                            "or moved by generating this estimate."
                        ),
                    }
                    result.status = "success"

                except BaseException as error:
                    executionError = error
                    result.status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                    failureRecord = {"errorType": type(error).__name__, "message": str(error)}
                    if isinstance(error, pglToolExecutionError):
                        child = error.result
                        result.returnCode = child.returnCode
                        result.stderrPath = child.stderrPath
                        result.stdoutPath = child.stdoutPath
                        if result.logDirectory is None: result.logDirectory = child.logDirectory

                finally:
                    details["childInvocations"] = [
                        record for record in recorder.exportInvocations()
                        if record["invocationID"] not in previousIDs
                    ]
                    recorder.finishInvocation(
                        invocationID,
                        status=result.status,
                        result=details,
                        error=failureRecord,
                    )
                    if standalone: recorder.close()

                    result._provenance = {
                        "schemaVersion": 1,
                        "environment": recorder.exportEnvironment(),
                        "invocation": next(
                            record for record in recorder.exportInvocations()
                            if record["invocationID"] == invocationID
                        ),
                    }
                    self.lastResult = result

                if executionError is not None:
                    if not isinstance(executionError, Exception): raise executionError
                    raise pglToolExecutionError(
                        f"Run-reference alignment failed: {failureRecord['message']}",
                        result,
                    ) from executionError

                return result

        except Exception as error:
            if isinstance(error, pglToolExecutionError): self.lastResult = error.result
            if activeRecorder is not None and activeRecorder.actionID is not None:
                raise
            pglMessages.warning(f"AFNI run-reference alignment stopped: {type(error).__name__}: {error}")
            return None
        
    ################################################################
    # Display an acquisition in an external viewer
    ################################################################
    def viewer(self, scanNum=0, *, backend="afni", afniDirectory=None, matlabExecutable=None, mlrDirectory=None, cacheDirectory=None):
        """Open a selected original acquisition in AFNI or MATLAB mlrVol.

        scanNum:
            Zero-based index in this MRI collection's scans.

        backend:
            "afni" (default) or "mlrVol", case-insensitive.

        afniDirectory:
            Optional AFNI executable directory.

        matlabExecutable:
            Optional full path to MATLAB's bin/matlab executable.

        mlrDirectory:
            Optional directory containing mlrVol.m.
            Other mrTools dependencies must be available through MATLAB setup.

        cacheDirectory:
            Optional cache override for acquisition retrieval and viewer files.

        Returns a launcher process, or None after a warning on failure.

        Does not access scan.data or implicitly select corrected-image products.
        mlrVol receives an independent copy because it supports editing/saving.
        Each MATLAB call starts a new desktop; existing sessions are not reused.
        """
        from pathlib import Path

        from .pglExternalTools import pglAFNI, pglMATLAB
        from .pglMessages import pglMessages

        try:
            if not isinstance(backend, str) or backend.lower() not in ("afni", "mlrvol"):
                raise ValueError("backend must be 'afni' or 'mlrVol'.")
            backendName = backend.lower()

            scans = list(self.scans)
            if not scans: raise ValueError("No MRI scans are attached.")
            if type(scanNum) is not int or not 0 <= scanNum < len(scans):
                raise ValueError(f"scanNum must be a zero-based integer between 0 and {len(scans) - 1}.")

            scan = scans[scanNum]

            # Prefer acquisition backing. Do not accidentally display an edited
            # working array in place of the original acquisition.
            if scan._backing is not None:
                cache = scan._cacheDirectory if cacheDirectory is None else cacheDirectory
                if cache is None or not str(cache).strip():
                    raise ValueError("A local cache directory is required to retrieve this acquisition.")

                source = Path(scan._backing.materialize(cache))

            elif scan.localPath:
                source = Path(scan.localPath).expanduser().absolute()

            else:
                raise ValueError(f"MRI scan {scanNum} has no acquisition source.")

            if not source.is_file():
                raise FileNotFoundError(f"MRI scan {scanNum}'s acquisition file is unavailable: {source}")

            scanName = scan.name.replace("\n", " ").replace("\r", " ")
            pglMessages.message(f"Opening MRI scan {scanNum} in {backend}: {scanName}")

            if backendName == "afni":
                afni = pglAFNI(directory=afniDirectory)
                return afni.viewer(source, cacheDirectory=cacheDirectory)

            return pglMATLAB.viewer(
                source,
                matlabExecutable=matlabExecutable,
                mlrDirectory=mlrDirectory,
                cacheDirectory=cacheDirectory,
            )

        except Exception as error:
            pglMessages.warning(f"MRI viewer could not be opened: {type(error).__name__}: {error}")
            return None
        
    @classmethod
    def getStatus(cls, *, configuration=None, timeout=10.0, cacheDirectory=None):
        """Adapt the existing AFNI availability check to the shared report.

        Preserves managed-installation checks, version probes, diagnostic logs,
        and failure handling. Does not install software or advance setup.

        A temporary adapter instance is used; existing scientific adapters are
        not reconfigured. Probe provenance remains available in report details.

        The existing AFNI checker may print diagnostics. Its output is not
        suppressed or redirected merely to make the summary table quieter.
        """
        import math
        from pathlib import Path

        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite.")

        configuration = cls.getConfiguration() if configuration is None else _copyProvenance(configuration)
        settings = cls._settingsFromValues(configuration)
        directory = settings.directory
        selectedDirectory = str(Path(directory).expanduser().absolute()) if directory.strip() else None

        probe = cls(directory=selectedDirectory)

        # A supplied configuration may intentionally request automatic discovery
        # even when the machine's default file contains another directory.
        # Override only this new probe instance, never saved settings.
        probe.directory = selectedDirectory
        probe.configurationSource = "saved" if selectedDirectory is not None else "automatic"

        available = bool(probe.checkAvailable(
            timeout=timeout,
            cacheDirectory=cacheDirectory,
        ))
        details = _copyProvenance(probe.lastAvailability or {})

        error = details.get("error")
        if isinstance(error, dict):
            diagnostic = error.get("message")
        elif error is None:
            diagnostic = None
        else:
            diagnostic = str(error)

        installationStatus = details.get("installationStatus")
        if available:
            status = "Available"
        elif installationStatus is not None and installationStatus != "verified":
            status = "Setup incomplete"
        else:
            status = "Unavailable"

        # Prefer the resolved directory reported by the actual probe.
        # A requested/configured path is not proof of resolution.
        location = details.get("executableDirectory")

        return {
            "toolName": cls.toolName,
            "label": cls.toolLabel or cls.toolName,
            "status": status,
            "available": available,
            "version": details.get("reportedVersion"),
            "location": location,
            "diagnostic": diagnostic,
            "details": details,
        }
    
################################################################
# AFNI installation state and administrator-stage launcher
################################################################
class pglAFNIInstallation:
    """Coordinate the supported macOS ARM administrator installation stage.

    Persistent records live under installPath/install.

    This initial coordinator:
        Reviews and launches the two official administrator scripts.
        Preserves the exact downloaded script bytes and their identities.
        Records progress and the boot identifier.
        Provides explicit instructions for rebooting and resuming.

    It does not:
        Run the user installation stage yet.
        Collect administrator passwords.
        Reboot the computer.
        Treat script completion as verified dependency installation.
        Automatically retry an interrupted or failed installation.

    State files are local installation bookkeeping, not scientific provenance.
    The installation directory must be trusted and user-owned.
    """

    scriptNames = (
        "OS_notes.macos_12_ARM_a_admin_pt1.zsh",
        "OS_notes.macos_12_ARM_a_admin_pt2.zsh",
    )
    sourceRoot = "https://raw.githubusercontent.com/afni/afni/master/src/other_builds"

    def __init__(self, installPath="~/proj/afni"):
        import os
        from pathlib import Path

        value = os.fspath(installPath)
        if (
            not isinstance(value, str)
            or not value.strip()
            or "://" in value
            or "::" in value
            or any(character in value for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("installPath must be an ordinary nonblank local path.")

        self.root = Path(value).expanduser().resolve()
        if self.root in (Path(self.root.anchor), Path.home().resolve()):
            raise ValueError("Use a dedicated installation directory.")

        self.directory = self.root / "install"
        self.statePath = self.directory / "state.json"

        for path in (self.root, self.directory):
            if path.is_symlink(): raise ValueError(f"Installation directory must not be a symlink: {path}")
            if path.exists() and not path.is_dir(): raise ValueError(f"Installation path is not a directory: {path}")

    ################################################################
    # Identify the current boot without changing system state
    ################################################################
    @staticmethod
    def getBootID():
        """Return macOS's boot-session UUID, or None when unavailable."""
        import platform
        import subprocess

        if platform.system() != "Darwin": return None

        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            value = result.stdout.strip()
            if result.returncode != 0 or not value: return None

            from uuid import UUID
            return str(UUID(value))

        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    ################################################################
    # Read existing state without creating installation files
    ################################################################
    def readState(self):
        """Read installation progress without accessing scientific payloads.

        Missing state returns None. Malformed or unreadable state raises
        rather than silently treating an existing installation as new.

        Recognizes administrator, user-installation, diagnostic-review, and
        verified states. A verified record is historical installation status,
        not proof that executables remain available today.
        """
        import json
        from pathlib import Path
        from uuid import UUID

        if self.statePath.is_symlink():
            raise ValueError("Installation state must not be a symlink.")
        if not self.statePath.exists(): return None

        def uniqueKeys(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError(f"Duplicate installation-state key: {key!r}")
                result[key] = value
            return result

        with self.statePath.open("rb") as source:
            content = source.read(1024 * 1024 + 1)

        if len(content) > 1024 * 1024:
            raise ValueError("Installation state exceeds its size limit.")

        state = json.loads(content, object_pairs_hook=uniqueKeys)

        if not isinstance(state, dict) or type(state.get("schemaVersion")) is not int or state["schemaVersion"] != 1:
            raise ValueError("Unsupported installation-state record.")

        if state.get("tool") != "AFNI" or state.get("installPath") != str(self.root):
            raise ValueError("Installation state does not belong to this installation.")

        allowed = {
            "launchPending", "adminRunning", "rebootPending",
            "failed", "interrupted",
            "userLaunchPending", "userRunning", "userFailed", "userInterrupted",
            "verificationPending", "verified",
        }
        if state.get("status") not in allowed:
            raise ValueError(f"Unknown installation status: {state.get('status')!r}.")

        identifier = state.get("installationID")
        if not isinstance(identifier, str):
            raise ValueError("Installation state has no execution identity.")

        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError("Installation identity must use canonical UUID formatting.")
        except ValueError:
            raise ValueError("Invalid installation execution identity.") from None

        # The worker records this only after completing its probes.
        # Do not permit a restored record to redirect verification elsewhere.
        if state["status"] in ("verificationPending", "verified"):
            expectedDirectory = str(self.root / "abin")
            if state.get("executableDirectory") != expectedDirectory:
                raise ValueError("Installation executable directory differs from installPath/abin.")

            logValue = state.get("systemCheckLog")
            if not isinstance(logValue, str) or not logValue:
                raise ValueError("Completed user setup has no system-check log reference.")

            logPath = Path(logValue)
            if not logPath.is_absolute() or self.directory not in logPath.resolve().parents:
                raise ValueError("System-check log must be beneath the installation records directory.")

        return state

    ################################################################
    # Explain recorded installation progress
    ################################################################
    def continuationMessage(self, state):
        """Describe recorded status and the explicit continuation command.

        Does not advance installation or infer successful completion.
        Uses the default short command for the default installation root.
        """
        from pathlib import Path

        defaultRoot = (Path.home() / "proj" / "afni").resolve()
        resume = "afni.install()" if self.root == defaultRoot else f"afni.install(installPath={str(self.root)!r})"
        status = state["status"]

        if status in ("launchPending", "adminRunning", "userLaunchPending", "userRunning"):
            stageName = "user" if status.startswith("user") else "administrator"
            message = (
                f"AFNI {stageName} installation is pending or running in Terminal.\n"
                "Check that window before starting another installation. "
                "If it closed unexpectedly, completion is uncertain."
            )

        elif status == "rebootPending":
            previousBoot = state.get("completedBootID")
            currentBoot = self.getBootID()

            if previousBoot is not None and currentBoot is not None:
                rebootMessage = (
                    "Restart your Mac before continuing."
                    if previousBoot == currentBoot
                    else "A new boot session was detected; user setup is next."
                )
            else:
                rebootMessage = (
                    "PGL cannot verify the reboot. Confirm that you restarted "
                    "after administrator setup before continuing."
                )

            message = (
                "AFNI installation is incomplete: administrator scripts finished, "
                "but user setup has not run.\n"
                f"{rebootMessage}"
            )

        elif status in ("failed", "interrupted", "userFailed", "userInterrupted"):
            stageName = "user" if status.startswith("user") else "administrator"
            message = (
                f"AFNI {stageName} setup did not finish successfully "
                f"(status={status}).\n"
                f"Diagnostic: {state.get('error') or 'See the Terminal output.'}\n"
                "Completed changes were not rolled back. Automatic retry is "
                "disabled until the partial installation is reviewed."
            )

        elif status == "verificationPending":
            message = (
                "AFNI setup and executable probes completed, but diagnostic "
                "review is still pending.\n"
                f"System check: {state['systemCheckLog']}\n"
                "Reviewing diagnostics must not rerun the installation."
            )

        elif status == "verified":
            message = (
                "AFNI installation was previously verified and reviewed.\n"
                "Current executable availability still requires a fresh check."
            )

        else:
            raise ValueError(f"Unknown installation status: {status!r}.")

        return (
            f"{message}\n\n"
            f"Continue or inspect installation status using:\n    {resume}\n\n"
            f"Installation records: {self.directory}"
        )

    ################################################################
    # Launch the reviewed administrator scripts
    ################################################################
    def launchAdmin(self):
        """Download for review, request approval, then launch Terminal.

        Downloads are read-only network requests before confirmation.
        No installation directory is created until approval.

        Returns a launch-state dictionary, or None if the user cancels.
        Operational errors propagate to the public install() warning boundary.

        Script stdout/stderr and password prompts remain in Terminal.
        This version stores script identities and stage outcomes, but does not
        record a complete Terminal transcript or password input.

        Exact downloaded administrator scripts are retained and executed.
        Their downstream downloads are not pinned by these hashes.
        """
        import hashlib
        import json
        import os
        import platform
        import shlex
        import subprocess
        import sys
        import urllib.request
        from pathlib import Path
        from uuid import uuid4

        from .pglDialog import pglDialogs
        from .pglMessages import pglMessages

        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            raise ValueError("This installer stage currently supports native macOS Apple Silicon only.")

        version = platform.mac_ver()[0]
        if not version or int(version.split(".", 1)[0]) < 12:
            raise ValueError("This installer stage requires macOS 12 or newer.")

        if self.readState() is not None:
            raise ValueError("An installation record already exists. Review it before launching another installation.")

        scripts = []

        for name in self.scriptNames:
            url = f"{self.sourceRoot}/{name}"

            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read(2 * 1024 * 1024 + 1)

            if not content or len(content) > 2 * 1024 * 1024:
                raise ValueError(f"Unexpected installer script size: {name}")

            text = content.decode("utf-8")
            if "\x00" in text: raise ValueError(f"Installer contains NUL: {name}")

            scripts.append({
                "name": name,
                "url": url,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "content": content,
                "text": text,
            })

        installationID = str(uuid4())
        stageDirectory = self.directory / installationID
        resume = f"afni.install(installPath={str(self.root)!r})"

        details = f"""AFNI ADMINISTRATOR INSTALLATION
================================

Platform: macOS {version}, Apple Silicon
Installation root: {self.root}
This stage's scripts and records: {stageDirectory}
Installer working directory: {Path.home()}

The following official scripts will run unchanged, in order, in Terminal:
  zsh OS_notes.macos_12_ARM_a_admin_pt1.zsh
  zsh OS_notes.macos_12_ARM_a_admin_pt2.zsh

Known effects from the reviewed procedure:
- Install/configure Homebrew and disable Homebrew analytics.
- Install Python, compilers, build libraries, and XQuartz.
- Modify shell startup files.
- Download and install system-wide R.
- Prompt for administrator authorization in Terminal.

The script bodies below are authoritative for this request.
They can download and run additional installers and change dependencies.
Their downstream downloads are not pinned by the displayed script hashes.

PGL will not collect passwords or reboot your Mac.
Terminal output is not captured as a transcript in this implementation.
Installation is not transactional; stopping does not undo completed changes.
A zero script exit status does not establish that every internal step worked.

AFTER BOTH SCRIPTS FINISH:
Review their Terminal output, restart your Mac, then reopen Python and run:
    from pgl.pglExternalTools import pglAFNI
    afni = pglAFNI()
    {resume}

No existing installation stage will be silently overwritten or retried.
"""

        for script in scripts:
            details += (
                f"\n\n{'=' * 72}\n"
                f"FILE: {script['name']}\n"
                f"SOURCE: {script['url']}\n"
                f"SHA-256: {script['sha256']}\n"
                f"{'=' * 72}\n\n{script['text']}"
            )

        pglMessages.print(details)

        if not pglDialogs.confirm(
            "Review the official scripts and system changes. Administrator setup requires a reboot afterward.",
            title="Install AFNI: administrator setup",
            details=details,
            confirmButtonText="Run in Terminal",
            copyText="Copy instructions",
        ):
            return None

        # An exclusive claim prevents two cooperating PGL callers from starting
        # installation after simultaneously reviewing the same empty state.
        self.directory.mkdir(parents=True, exist_ok=True)
        claimPath = self.directory / ".installation-claim"

        try:
            claimPath.mkdir()
        except FileExistsError:
            raise RuntimeError("Another installation has claimed this directory. Review its state before proceeding.") from None

        state = {
            "schemaVersion": 1,
            "tool": "AFNI",
            "installationID": installationID,
            "installPath": str(self.root),
            "status": "launchPending",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "completedBootID": None,
            "scripts": [
                {key: script[key] for key in ("name", "url", "sha256", "size_bytes")}
                for script in scripts
            ],
            "steps": [],
            "error": None,
        }

        def writeState():
            """Publish current state atomically within the installation directory."""
            temporary = self.directory / f".state-{uuid4().hex}.json"
            try:
                with temporary.open("x", encoding="utf-8") as target:
                    json.dump(state, target, indent=2, sort_keys=True, allow_nan=False)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, self.statePath)
            finally:
                if temporary.exists(): temporary.unlink()

        try:
            # Recheck after obtaining the claim. Existing records are not
            # discarded merely because another caller started meanwhile.
            if self.readState() is not None:
                raise RuntimeError("Installation state changed while confirmation was open.")

            stageDirectory.mkdir()
            (stageDirectory / "approved-plan.txt").write_text(details, encoding="utf-8")

            for script in scripts:
                with (stageDirectory / script["name"]).open("xb") as target:
                    target.write(script["content"])

            # The launcher uses only the standard library and runs outside the
            # notebook. It verifies scripts before execution and writes state
            # even when the parent Python session is no longer present.
            launcher = r'''
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

statePath = Path(sys.argv[1])
stageDirectory = Path(sys.argv[2])
expectedID = sys.argv[3]
state = json.loads(statePath.read_text(encoding="utf-8"))

if state["installationID"] != expectedID or state["status"] != "launchPending":
    raise RuntimeError("Installation launch identity/status mismatch.")

def saveState():
    temporary = statePath.parent / (".state-" + uuid4().hex + ".json")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(state, target, sort_keys=True, indent=2, allow_nan=False)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, statePath)
    finally:
        if temporary.exists(): temporary.unlink()

def bootID():
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return str(UUID(result.stdout.strip()))
    except Exception:
        return None

exitCode = 1
try:
    state["status"] = "adminRunning"
    state["startedAt"] = datetime.now(timezone.utc).isoformat()
    saveState()

    for script in state["scripts"]:
        path = stageDirectory / script["name"]
        content = path.read_bytes()
        if len(content) != script["size_bytes"] or hashlib.sha256(content).hexdigest() != script["sha256"]:
            raise RuntimeError("Installer script changed after approval: " + script["name"])

        print("\n=== Running " + script["name"] + " ===\n", flush=True)
        completed = subprocess.run(
            ["/bin/zsh", str(path)],
            cwd=str(Path.home()),
            check=False,
        )
        state["steps"].append({
            "script": script["name"],
            "returnCode": completed.returncode,
            "endedAt": datetime.now(timezone.utc).isoformat(),
        })
        saveState()

        if completed.returncode != 0:
            raise RuntimeError(script["name"] + " exited with code " + str(completed.returncode))

    state["status"] = "rebootPending"
    state["completedBootID"] = bootID()
    state["endedAt"] = datetime.now(timezone.utc).isoformat()
    saveState()
    exitCode = 0

    print("\nAFNI administrator scripts have finished.")
    print("Review the output above: script completion does not verify every dependency.")
    print("\nRestart your Mac before continuing.")
    print("After restarting, reopen Python or your notebook and run:\n")
    print("    from pgl.pglExternalTools import pglAFNI")
    print("    afni = pglAFNI()")
    print("    afni.install(installPath=" + repr(state["installPath"]) + ")")

except BaseException as error:
    state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
    state["error"] = type(error).__name__ + ": " + str(error)
    state["endedAt"] = datetime.now(timezone.utc).isoformat()
    saveState()
    print("\nAFNI setup did not finish: " + state["error"])
    print("Completed system changes have NOT been rolled back.")

print("\nInstallation state: " + str(statePath), flush=True)
try:
    input("\nPress Return to close this launcher...")
except (EOFError, KeyboardInterrupt):
    pass

sys.exit(exitCode)
'''
            launcherPath = stageDirectory / "run-admin.py"
            launcherPath.write_text(launcher, encoding="utf-8")

            # A .command file lets Terminal own the interactive terminal and
            # password prompts. Quoting protects paths containing spaces.
            commandPath = stageDirectory / "run-admin.command"
            command = [
                sys.executable,
                str(launcherPath),
                str(self.statePath),
                str(stageDirectory),
                installationID,
            ]
            commandPath.write_text(
                "#!/bin/zsh\nexec " + " ".join(shlex.quote(value) for value in command) + "\n",
                encoding="utf-8",
            )
            commandPath.chmod(0o700)
            writeState()

            result = subprocess.run(
                ["/usr/bin/open", "-a", "Terminal", str(commandPath)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Terminal launch was refused.")

        except BaseException as error:
            # Leave the claim in place: partial files or a launched Terminal
            # must not permit an accidental second installation.
            # Never overwrite progress written by a launcher that has started.
            existing = self.readState()
            if existing is None or (
                existing.get("installationID") == installationID
                and existing.get("status") == "launchPending"
            ):
                state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                state["error"] = f"{type(error).__name__}: {error}"
                writeState()
            raise

        pglMessages.message(
            "AFNI administrator installation was requested in Terminal.\n"
            "This is not confirmation that installation succeeded.\n"
            "Follow the Terminal prompts and reboot instructions."
        )
        return state
    
    ################################################################
    # Determine the next installation stage without changing state
    ################################################################
    def getResumeStatus(self, *, stage="auto"):
        """Return a read-only installation decision.

        stage:
            auto: select from recorded progress.
            admin: request initial administrator setup.
            user: request user setup, including separately installed prerequisites.

        nextStage:
            admin, user, review, check, or None.

        canOffer indicates that the coordinator may offer the next step.
        It is not permission to execute or approval of the installation.

        review displays diagnostics without reinstalling.
        check performs fresh executable checks without reinstalling.

        Failed, interrupted, and uncertain running stages require reviewed
        recovery. They are never silently retried or bypassed.
        """
        from uuid import UUID

        if stage not in ("auto", "admin", "user"):
            raise ValueError("stage must be 'auto', 'admin', or 'user'.")

        state = self.readState()
        decision = {
            "schemaVersion": 1,
            "installPath": str(self.root),
            "installationID": None if state is None else state["installationID"],
            "recordedStatus": None if state is None else state["status"],
            "status": None,
            "nextStage": None,
            "canOffer": False,
            "requiresPrerequisiteConfirmation": False,
            "requiresRebootConfirmation": False,
            "message": "",
        }

        def setDecision(status, message, *, nextStage=None, prerequisites=False, reboot=False):
            decision.update({
                "status": status,
                "nextStage": nextStage,
                "canOffer": nextStage is not None,
                "requiresPrerequisiteConfirmation": prerequisites,
                "requiresRebootConfirmation": reboot,
                "message": message,
            })
            return decision

        if state is None:
            if stage == "user":
                return setDecision(
                    "externalPrerequisites",
                    "No PGL administrator-stage record exists. Before user setup, "
                    "confirm that the official administrator prerequisites are "
                    "installed and any required restart has been completed.",
                    nextStage="user",
                    prerequisites=True,
                    reboot=True,
                )

            return setDecision(
                "newInstallation",
                "No installation record exists. Review and approve administrator "
                "setup before it is launched in Terminal.",
                nextStage="admin",
            )

        recordedStatus = state["status"]

        # A running record is not proof that its Terminal/process is still
        # alive. Uncertain completion must not trigger another installation.
        if recordedStatus in ("launchPending", "adminRunning", "userLaunchPending", "userRunning"):
            stageName = "user" if recordedStatus.startswith("user") else "administrator"
            return setDecision(
                "runningOrUncertain",
                f"AFNI {stageName} setup is recorded as pending or running. "
                "Check its Terminal window. If Terminal closed or the machine "
                "stopped unexpectedly, completion is uncertain. PGL will not "
                "launch another installer or skip ahead.",
            )

        if recordedStatus in ("failed", "interrupted", "userFailed", "userInterrupted"):
            stageName = "user" if recordedStatus.startswith("user") else "administrator"
            return setDecision(
                "recoveryRequired",
                f"AFNI {stageName} setup did not finish successfully "
                f"(status={recordedStatus}). Completed changes were not rolled "
                "back. Review the installation records and Terminal output "
                "before selecting a recovery procedure. Automatic retry is disabled.",
            )

        if recordedStatus == "verificationPending":
            return setDecision(
                "reviewRequired",
                "AFNI user setup and executable probes completed. Installation "
                "still requires review of the system-check diagnostics. "
                "Reviewing these results does not rerun installation.",
                nextStage="review",
            )

        if recordedStatus == "verified":
            return setDecision(
                "previouslyVerified",
                "This installation was previously verified and reviewed. "
                "Check current executable availability without reinstalling.",
                nextStage="check",
            )

        if recordedStatus != "rebootPending":
            raise ValueError(f"Unsupported resume status: {recordedStatus!r}.")

        if stage == "admin":
            return setDecision(
                "adminAlreadyCompleted",
                "The administrator scripts have finished. PGL will not repeat "
                "them implicitly. Restart if required, then use afni.install() "
                "to continue toward user setup.",
            )

        def normalizeBootID(value):
            if value is None: return None
            if not isinstance(value, str):
                raise ValueError("Boot identity must be a UUID string or None.")
            try:
                return str(UUID(value))
            except ValueError:
                raise ValueError("Invalid installation boot identity.") from None

        previousBoot = normalizeBootID(state.get("completedBootID"))
        currentBoot = normalizeBootID(self.getBootID())

        if previousBoot is not None and currentBoot is not None:
            if previousBoot == currentBoot:
                return setDecision(
                    "restartRequired",
                    "The Mac is still in the boot session in which administrator "
                    "setup finished. Restart your Mac, reopen Python or your "
                    "notebook, and run afni.install() again. Restarting Python "
                    "alone does not satisfy this requirement.",
                )

            return setDecision(
                "userStageReady",
                "A restart after administrator-script completion was detected. "
                "Confirm that the administrator prerequisites installed "
                "successfully before approving user setup. Reboot detection "
                "does not verify dependencies.",
                nextStage="user",
                prerequisites=True,
            )

        return setDecision(
            "restartUnverified",
            "PGL cannot verify the restart. Before approving user setup, "
            "confirm that administrator prerequisites were installed and "
            "the Mac was restarted afterward.",
            nextStage="user",
            prerequisites=True,
            reboot=True,
        )

    ################################################################
    # Prepare the official user script for this installation root
    ################################################################
    def prepareUserScript(self, sourceText):
        """Adapt AFNI installation destinations without executing the script.

        Identifies the bootstrap, shell-configuration, and build commands.
        Checks only the destination options needed for safe relocation.
        Comments, whitespace, and unrelated command options may change.

        Returns original/adapted identities, script text, and a reviewable diff.

        Does not download files, create directories, or execute commands.
        Preparation errors propagate to the installation coordinator, which
        warns and stops before offering execution.

        Installation paths remain restricted because the upstream procedure
        contains unquoted shell-path expressions. This is not a shell parser
        or a security audit of the official installer.
        """
        import difflib
        import hashlib
        import re
        import shlex

        if not isinstance(sourceText, str) or not sourceText.strip():
            raise ValueError("sourceText must contain the official user script.")
        if "\x00" in sourceText:
            raise ValueError("Installer text must not contain NUL.")

        executableDirectory = str(self.root / "abin")
        buildDirectory = str(self.root / "build")
        installationDirectory = str(self.directory)

        # Quoting our launcher alone cannot fix unquoted expressions inside
        # the upstream scripts and builder.
        if re.fullmatch(r"/[A-Za-z0-9_./-]+", str(self.root)) is None:
            raise ValueError(
                "This AFNI installation procedure currently requires a path "
                "without spaces or shell metacharacters."
            )

        # Normalize line endings for the generated Unix shell script.
        # sourceIdentity below still describes the exact supplied text.
        adapted = sourceText.replace("\r\n", "\n")
        if "\r" in adapted: raise ValueError("Unsupported carriage return in installer text.")

        # Relocate known home-relative destinations, including references in
        # conditions, commands, messages, and comments. Match whole path
        # components so names such as ~/abin_backup are not changed.
        for name, destination in (("abin", executableDirectory), ("afni_build", buildDirectory)):
            pattern = rf"(?:\$HOME/|\$\{{HOME\}}/|~/){name}(?![A-Za-z0-9_.-])"
            adapted = re.sub(pattern, lambda match: destination, adapted)

        # Group continued command lines while preserving their formatting.
        blocks = []
        pending = ""

        for line in adapted.splitlines(keepends=True):
            pending += line
            if line.rstrip("\n").endswith("\\"): continue
            blocks.append(pending)
            pending = ""

        if pending: blocks.append(pending)

        found = {"bootstrap": 0, "shellConfiguration": 0, "build": 0}
        preparedBlocks = []

        def requireOption(tokens, option, expected):
            """Require one unambiguous destination option on a relevant command."""
            positions = [index for index, token in enumerate(tokens) if token == option]
            if len(positions) != 1:
                raise ValueError(f"Cannot relocate installer: expected one {option} option on a relevant command.")

            index = positions[0]
            if index + 1 >= len(tokens) or tokens[index + 1] != expected:
                raise ValueError(f"Cannot relocate installer: {option} does not identify {expected!r}.")

        for block in blocks:
            stripped = block.lstrip()
            if not stripped or stripped.startswith("#"):
                preparedBlocks.append(block)
                continue

            # Only parse relevant simple commands. Leave the rest of the
            # official tcsh procedure unchanged.
            firstWord = stripped.split(None, 1)[0].strip("\"'")
            commandLeaf = firstWord.rsplit("/", 1)[-1]

            if commandLeaf not in ("tcsh", "init_user_dotfiles.py", "build_afni.py"):
                preparedBlocks.append(block)
                continue

            tokens = shlex.split(block.replace("\\\n", " "), comments=True, posix=True)
            if not tokens:
                preparedBlocks.append(block)
                continue

            if commandLeaf == "tcsh":
                isBootstrap = any(
                    token.rsplit("/", 1)[-1] == "@update.afni.binaries"
                    for token in tokens[1:]
                )
                if not isBootstrap:
                    preparedBlocks.append(block)
                    continue

                found["bootstrap"] += 1
                requireOption(tokens, "-bindir", executableDirectory)

            elif commandLeaf == "init_user_dotfiles.py":
                found["shellConfiguration"] += 1
                requireOption(tokens, "-dir_bin", executableDirectory)

                # Select this installation before invoking its helpers.
                # The official dotfile updates remain intact.
                indent = block[:len(block) - len(stripped)]
                setup = (
                    f"{indent}# Use this AFNI installation in the current installer shell.\n"
                    f"{indent}set path = ( {executableDirectory} $path )\n"
                    f"{indent}rehash\n"
                )

                block = re.sub(
                    r"^([ \t]*)\S+",
                    lambda match: f"{match.group(1)}{executableDirectory}/init_user_dotfiles.py",
                    block,
                    count=1,
                )
                block = setup + block

            else:
                found["build"] += 1
                requireOption(tokens, "-build_root", buildDirectory)

                # Preserve compiler/package/options supplied upstream.
                # Explicit -abin prevents PATH from selecting another target.
                if "-abin" in tokens:
                    requireOption(tokens, "-abin", executableDirectory)
                    installOption = ""
                else:
                    installOption = f" -abin {executableDirectory}"

                block = re.sub(
                    r"^([ \t]*)\S+",
                    lambda match: f"{match.group(1)}{executableDirectory}/build_afni.py{installOption}",
                    block,
                    count=1,
                )

            preparedBlocks.append(block)

        # Missing or duplicated essential commands make relocation ambiguous.
        # Do not impose conditions on comments or unrelated setup operations.
        for name, count in found.items():
            if count != 1:
                raise ValueError(
                    f"Cannot prepare AFNI user setup: found {count} {name} "
                    "commands; expected one. Review the installer structure."
                )

        adapted = "".join(preparedBlocks)

        def identity(text):
            content = text.encode("utf-8")
            return {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}

        difference = "".join(difflib.unified_diff(
            sourceText.splitlines(keepends=True),
            adapted.splitlines(keepends=True),
            fromfile="official/OS_notes.macos_12_b_user.tcsh",
            tofile="prepared/OS_notes.macos_12_b_user.tcsh",
        ))

        return {
            "schemaVersion": 1,
            "tool": "AFNI",
            "kind": "preparedUserScript",
            "installPath": str(self.root),
            "executableDirectory": executableDirectory,
            "buildDirectory": buildDirectory,
            "installationDirectory": installationDirectory,
            "sourceFilename": "OS_notes.macos_12_b_user.tcsh",
            "sourceIdentity": identity(sourceText),
            "preparedIdentity": identity(adapted),
            "changes": [
                {
                    "kind": "destination",
                    "description": "Relocate official abin and build-root references.",
                    "executableDirectory": executableDirectory,
                    "buildDirectory": buildDirectory,
                },
                {
                    "kind": "executableSelection",
                    "description": "Select this installation for helper commands and explicitly invoke its builder.",
                    "destination": executableDirectory,
                },
                {
                    "kind": "destination",
                    "description": "Ensure the builder has an explicit -abin destination.",
                    "destination": executableDirectory,
                },
            ],
            "scriptText": adapted,
            "diff": difference,
            "executionRequirements": [
                "Display and approve the prepared script before execution.",
                "Verify staged bytes against preparedIdentity before launch.",
                "Use a separate stage working directory for downloads and logs.",
                "Preserve the official ARM wrapper's Homebrew initialization.",
                "Use an explicitly prepared build environment.",
                "Independently verify the selected installation after completion.",
            ],
        }
        
    ################################################################
    # Execute an approved user stage in the Terminal worker
    ################################################################
    @staticmethod
    def runUserStage(requestFilename):
        """Execute one staged, approved user installation request.

        Internal worker entry point, not a user-facing installation API.
        The coordinator must obtain approval and stage the request first.

        The request identifies:
            The installation state and execution UUID.
            The prepared script and its expected identity.
            The working and executable directories.
            Explicit environment removals and overrides.
            Whether prerequisite/reboot acknowledgment was obtained.

        Script output is inherited by Terminal. Verification output is saved
        in the stage directory. Administrator passwords are not collected.

        Returns:
            0 when the script and executable probes succeed. State becomes
              verificationPending, not verified.
            1 on ordinary failure, with diagnostics and preserved state.

        KeyboardInterrupt/SystemExit record interruption and propagate.
        No rollback or automatic retry is attempted.

        The request and installation directory are trusted local inputs.
        Hashes detect changed script bytes; they do not authenticate a request
        or protect against a malicious concurrent writer.
        """
        import hashlib
        import json
        import os
        import platform
        import subprocess
        from datetime import datetime, timezone
        from pathlib import Path
        from uuid import uuid4

        state = None
        statePath = None
        ownsState = False

        def readJSON(path):
            with path.open("r", encoding="utf-8") as source:
                value = json.load(source)
            if not isinstance(value, dict): raise ValueError("Expected a JSON object.")
            return value

        def saveState():
            """Publish only while the worker still owns this installation."""
            current = readJSON(statePath)
            if current.get("installationID") != state["installationID"]:
                raise RuntimeError("Installation ownership changed; state was not overwritten.")

            temporary = statePath.parent / f".state-{uuid4().hex}.json"

            try:
                with temporary.open("x", encoding="utf-8") as target:
                    json.dump(state, target, sort_keys=True, indent=2, allow_nan=False)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, statePath)
            finally:
                if temporary.exists(): temporary.unlink()

        def saveFailure(error):
            """Preserve the original failure even if state publication fails."""
            if not ownsState: return

            state["status"] = "userInterrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "userFailed"
            state["error"] = f"{type(error).__name__}: {error}"
            state["endedAt"] = datetime.now(timezone.utc).isoformat()

            try:
                saveState()
            except Exception as stateError:
                print(f"Could not save installation failure state: {stateError}", flush=True)

        try:
            requestPath = Path(requestFilename).expanduser().absolute()
            request = readJSON(requestPath)

            if request.get("schemaVersion") != 1:
                raise ValueError("Unsupported user-stage request version.")
            if request.get("approved") is not True:
                raise ValueError("User-stage request has not been approved.")
            if request.get("prerequisitesConfirmed") is not True:
                raise ValueError("Administrator prerequisites/reboot were not acknowledged.")

            if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
                raise ValueError("This user-stage worker supports native macOS Apple Silicon only.")

            statePath = Path(request["statePath"])
            state = readJSON(statePath)

            if (
                state.get("installationID") != request.get("installationID")
                or state.get("status") != "userLaunchPending"
                or state.get("tool") != "AFNI"
            ):
                raise ValueError("User-stage request does not match pending installation state.")

            ownsState = True

            root = Path(state["installPath"]).resolve()
            stageDirectory = Path(request["stageDirectory"]).resolve()
            executableDirectory = root / "abin"
            scriptPath = stageDirectory / "user-setup.tcsh"

            if root / "install" not in stageDirectory.parents:
                raise ValueError("User-stage directory must be beneath installPath/install.")
            if not stageDirectory.is_dir():
                raise ValueError("The staged working directory is unavailable.")
            if scriptPath.is_symlink():
                raise ValueError("The prepared installer must not be a symlink.")

            content = scriptPath.read_bytes()
            actualIdentity = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            if actualIdentity != request.get("scriptIdentity"):
                raise ValueError("User installer bytes changed after approval.")

            # These choices must appear in the approved plan. Do not silently
            # inherit a notebook's Conda/Python environment for compilation.
            removals = request.get("environmentRemove")
            overrides = request.get("environmentOverrides")

            if not isinstance(removals, list) or not isinstance(overrides, dict):
                raise ValueError("Missing explicit user-stage environment configuration.")
            if any(not isinstance(name, str) or not name or "\x00" in name for name in removals):
                raise ValueError("Invalid environment-removal name.")

            environment = dict(os.environ)
            for name in removals: environment.pop(name, None)

            for name, value in overrides.items():
                if (
                    not isinstance(name, str) or not name or "=" in name or "\x00" in name
                    or not isinstance(value, str) or "\x00" in value
                ):
                    raise ValueError("Invalid user-stage environment override.")
                environment[name] = value

            if not environment.get("PATH"):
                raise ValueError("The user-stage environment requires PATH.")

            state.update({
                "status": "userRunning",
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "error": None,
                "userScriptIdentity": actualIdentity,
                "userStageDirectory": str(stageDirectory),
                "userSteps": [],
            })
            saveState()

            print("\n=== Running approved AFNI user setup ===", flush=True)
            print(f"Working directory: {stageDirectory}", flush=True)
            print(f"AFNI executable directory: {executableDirectory}", flush=True)

            # Matches the reviewed user wrapper's tcsh flags. Password/input
            # handling remains in Terminal, not the scientific command runner.
            completed = subprocess.run(
                ["/bin/tcsh", "-xef", str(scriptPath)],
                cwd=str(stageDirectory),
                env=environment,
                check=False,
            )

            state["userSteps"].append({
                "name": "userSetup",
                "returnCode": completed.returncode,
                "endedAt": datetime.now(timezone.utc).isoformat(),
            })
            saveState()

            if completed.returncode != 0:
                raise RuntimeError(f"AFNI user setup exited with code {completed.returncode}.")

            # Environment edits inside tcsh do not propagate back into Python.
            # Explicitly select the newly installed programs for verification.
            verificationEnvironment = dict(environment)
            verificationEnvironment["PATH"] = str(executableDirectory) + os.pathsep + environment["PATH"]

            probes = (
                ("afniVersion", "afni", ["-ver"], 30),
                ("motionCommand", "3dvolreg", ["-help"], 30),
                ("systemCheck", "afni_system_check.py", ["-check_all"], 300),
            )

            for name, command, arguments, timeout in probes:
                executable = executableDirectory / command
                logPath = stageDirectory / f"{name}.log"

                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise RuntimeError(f"Expected installed executable is unavailable: {executable}")

                print(f"\n=== Checking {command} ===", flush=True)

                with logPath.open("xb") as output:
                    completed = subprocess.run(
                        [str(executable), *arguments],
                        cwd=str(stageDirectory),
                        env=verificationEnvironment,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=False,
                    )

                state["userSteps"].append({
                    "name": name,
                    "returnCode": completed.returncode,
                    "logPath": str(logPath),
                    "endedAt": datetime.now(timezone.utc).isoformat(),
                })
                saveState()

                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{command} verification exited with code {completed.returncode}. "
                        f"Read {logPath}"
                    )
                if logPath.stat().st_size == 0:
                    raise RuntimeError(f"{command} produced no verification output. Read {logPath}")

            state.update({
                "status": "verificationPending",
                "endedAt": datetime.now(timezone.utc).isoformat(),
                "executableDirectory": str(executableDirectory),
                "systemCheckLog": str(stageDirectory / "systemCheck.log"),
            })
            saveState()

            print("\nAFNI user setup and executable probes completed.", flush=True)
            print("Installation is awaiting diagnostic review—not yet marked verified.", flush=True)
            print(f"System check: {state['systemCheckLog']}", flush=True)
            print("Review its recommendations and test AFNI/SUMA GUI operation.", flush=True)
            print("\nThen reopen Python and run:", flush=True)
            print(f"    afni.install(installPath={str(root)!r})", flush=True)
            return 0

        except (KeyboardInterrupt, SystemExit) as error:
            saveFailure(error)
            raise

        except Exception as error:
            saveFailure(error)
            print(f"\nAFNI user setup stopped: {type(error).__name__}: {error}", flush=True)
            print("Completed changes have not been rolled back.", flush=True)
            return 1
        
    ################################################################
    # Prepare, confirm, and launch the user installation stage
    ################################################################
    def launchUser(self, *, stage="auto"):
        """Offer the user stage and launch its approved worker in Terminal.

        Downloads the official general user script for review, adapts its
        destination paths, and includes the reviewed ARM Homebrew setup.

        No installation directories are created until approval.
        Requests are recorded before Terminal launch.
        Terminal launch is not treated as installation completion.

        Returns the pending launch record, or None after cancellation/refusal.
        Operational errors propagate to the public install() warning boundary.

        This method does not retry failed or uncertain stages automatically.
        """
        import hashlib
        import inspect
        import json
        import os
        import platform
        import shlex
        import subprocess
        import sys
        import textwrap
        import urllib.request
        from pathlib import Path
        from uuid import uuid4

        from .pglDialog import pglDialogs
        from .pglMessages import pglMessages

        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            raise ValueError("This installation stage supports native macOS Apple Silicon only.")

        decision = self.getResumeStatus(stage=stage)
        if decision["nextStage"] != "user":
            pglMessages.warning(decision["message"])
            return None

        originalState = self.readState()

        # Acknowledgment is explicit rather than hidden in a long script.
        acknowledgment = decision["message"]
        acknowledgment += "\n\nDid the required administrator setup finish successfully?"
        if decision["requiresRebootConfirmation"]:
            acknowledgment += "\nHave you also completed the required restart?"

        if not pglDialogs.confirm(
            acknowledgment,
            title="AFNI installation prerequisites",
            confirmButtonText="Prerequisites completed",
        ):
            return None

        sourceURL = f"{self.sourceRoot}/OS_notes.macos_12_b_user.tcsh"

        with urllib.request.urlopen(sourceURL, timeout=30) as response:
            sourceBytes = response.read(2 * 1024 * 1024 + 1)

        if not sourceBytes or len(sourceBytes) > 2 * 1024 * 1024:
            raise ValueError("Unexpected AFNI user-script download size.")

        sourceText = sourceBytes.decode("utf-8")
        prepared = self.prepareUserScript(sourceText)

        # Preserve the ARM wrapper's Homebrew startup-file setup. The general
        # script is fetched and reviewed above instead of downloaded again by
        # the official wrapper after confirmation.
        #
        # Always initialize this process's Homebrew environment: an existing
        # shellenv line does not prove this Terminal worker has evaluated it.
        brewSetup = r'''
# PGL launch preparation: retain the reviewed ARM Homebrew setup.
if ( ! -x /opt/homebrew/bin/brew ) then
    echo "** Homebrew is unavailable. Complete administrator setup first."
    exit 1
endif

set FOUND = 0
if ( -f "$HOME/.zshrc" ) then
    set FOUND = `grep shellenv "$HOME/.zshrc" | grep brew | wc -l`
endif

if ( ! $FOUND ) then
    echo "++ setup login shells with brew path"
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"')      >> $HOME/.zprofile
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"')      >> $HOME/.zshrc
    (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv bash)"') >> $HOME/.bashrc
    (echo; echo 'eval `/opt/homebrew/bin/brew shellenv tcsh`')    >> $HOME/.login
endif

eval `/opt/homebrew/bin/brew shellenv`
if ( $status ) then
    echo "** Could not initialize the Homebrew environment."
    exit 1
endif

'''
        scriptText = "#!/bin/tcsh\n" + brewSetup + prepared["scriptText"]
        scriptBytes = scriptText.encode("utf-8")
        scriptIdentity = {
            "sha256": hashlib.sha256(scriptBytes).hexdigest(),
            "size_bytes": len(scriptBytes),
        }

        # The worker itself needs only the standard library. Snapshot its
        # implementation so an installation never depends on later edits to
        # the loaded PGL package while its Terminal is running.
        workerSource = textwrap.dedent(inspect.getsource(type(self).runUserStage))
        workerLines = workerSource.splitlines(keepends=True)
        if workerLines and workerLines[0].strip() == "@staticmethod":
            workerLines.pop(0)
        workerSource = "".join(workerLines)

        if not workerSource.startswith("def runUserStage("):
            raise RuntimeError("Could not prepare the standalone user-stage worker.")

        workerSource += r'''

if __name__ == "__main__":
    import sys

    exitCode = runUserStage(sys.argv[1])
    try:
        input("\nPress Return to close this launcher...")
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(exitCode)
'''
        workerIdentity = hashlib.sha256(workerSource.encode("utf-8")).hexdigest()

        # Exclude notebook environment selection and common compiler overrides.
        # This is controlled preparation, not a claim of hermetic compilation.
        environmentRemove = sorted({
            name for name in os.environ
            if name.startswith(("CONDA_", "_CONDA_", "MAMBA_", "PYTHON"))
        } | {
            "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT",
            "CC", "CXX", "FC", "F77", "CFLAGS", "CXXFLAGS", "FFLAGS",
            "CPPFLAGS", "LDFLAGS", "CPATH", "LIBRARY_PATH",
            "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH", "SDKROOT",
        })

        environmentOverrides = {
            "PATH": ":".join((
                "/opt/homebrew/bin",
                "/opt/homebrew/sbin",
                "/opt/homebrew/opt/python/libexec/bin",
                "/Library/Frameworks/R.framework/Resources/bin",
                "/opt/X11/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            )),
        }

        stageID = str(uuid4())
        installationID = str(uuid4()) if originalState is None else originalState["installationID"]
        stageDirectory = self.directory / stageID
        defaultRoot = (Path.home() / "proj" / "afni").resolve()
        resume = "afni.install()" if self.root == defaultRoot else f"afni.install(installPath={str(self.root)!r})"

        details = f"""AFNI USER INSTALLATION
======================

Installation root: {self.root}
Executables/resources: {self.root / 'abin'}
Build workspace/backups: {self.root / 'build'}
This stage's records/logs: {stageDirectory}

Official general user script:
{sourceURL}

Downloaded source SHA-256:
{hashlib.sha256(sourceBytes).hexdigest()}

Prepared executable script SHA-256:
{scriptIdentity['sha256']}

Standalone worker SHA-256:
{workerIdentity}

EXECUTION
---------
Run the prepared script with:
    /bin/tcsh -xef user-setup.tcsh

Terminal displays the installation output.
PGL does not collect passwords or reboot the machine.
The installer can download additional scripts, packages, and source code.
Those downstream downloads are not pinned by the hashes above.

The ARM wrapper's Homebrew startup setup is included below.
The downloaded general user script is staged now, not downloaded again
after approval.

The worker runs using:
    {sys.executable} -I

That Python runs bookkeeping only. AFNI commands and build scripts use:
    PATH={environmentOverrides['PATH']}

The following environment variables are removed from the worker's child
process environment:
    {', '.join(environmentRemove)}

The user's HOME and existing R_LIBS are not redirected.
Homebrew shellenv is evaluated in the installer shell.
This is not a fully isolated or hermetic build.

KNOWN EFFECTS
-------------
- Configure Homebrew startup initialization if absent.
- Change Terminal/XQuartz focus-following preferences.
- Bootstrap AFNI scripts/resources into the selected abin directory.
- Update shell startup files and command completion.
- Build AFNI using the official procedure and explicit destination options.
- Configure/install AFNI R packages.
- Create missing AFNI/SUMA configuration and command help.
- Potentially replace existing AFNI files and prune AFNI-managed backups.

The exact prepared script below is authoritative for this request.
Cancelling before launch makes no installation changes.
Stopping after launch does not roll back completed changes.

VERIFICATION
------------
The worker probes this installation's afni and 3dvolreg executables and runs
afni_system_check.py -check_all. Logs remain in this stage directory.

The official script also writes out.ASC.txt from its own shell environment.
That environment can differ from the worker's separate verification process.
Both reports will be shown during review.

Successful probes leave status verificationPending—not verified.
After reviewing the Terminal output, continue using:
    {resume}

No complete Terminal transcript is captured by this implementation.

DESTINATION DIFF
----------------
{prepared['diff']}

PREPARED SCRIPT
---------------
{scriptText}
"""

        pglMessages.print(details)

        if not pglDialogs.confirm(
            "Review the installation script, destinations, and environment before launching Terminal.",
            title="Install AFNI: user setup",
            details=details,
            confirmButtonText="Run in Terminal",
            copyText="Copy instructions",
        ):
            return None

        # A reboot or another installation process may have changed the
        # situation while the user was reviewing the plan.
        if self.readState() != originalState:
            raise RuntimeError("Installation state changed during review. Run afni.install() again.")

        refreshed = self.getResumeStatus(stage=stage)
        if refreshed["nextStage"] != "user":
            raise RuntimeError(refreshed["message"])

        self.directory.mkdir(parents=True, exist_ok=True)

        # Persistent stage claim prevents two cooperating launchers from
        # publishing/starting the same user stage. Recovery is explicit.
        claimPath = self.directory / ".user-stage-claim"
        try:
            claimPath.mkdir()
        except FileExistsError:
            raise RuntimeError("The user stage has already been claimed. Review its state before launching again.") from None

        state = deepcopy(originalState) if originalState is not None else {
            "schemaVersion": 1,
            "tool": "AFNI",
            "installationID": installationID,
            "installPath": str(self.root),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

        state.update({
            "status": "userLaunchPending",
            "userStageID": stageID,
            "userStageDirectory": str(stageDirectory),
            "userScriptIdentity": scriptIdentity,
            "userSource": {
                "url": sourceURL,
                "sha256": hashlib.sha256(sourceBytes).hexdigest(),
                "size_bytes": len(sourceBytes),
            },
            "userWorkerSHA256": workerIdentity,
            "prerequisitesConfirmedAt": datetime.now(timezone.utc).isoformat(),
            "rebootConfirmationRequired": decision["requiresRebootConfirmation"],
            "error": None,
        })

        request = {
            "schemaVersion": 1,
            "approved": True,
            "prerequisitesConfirmed": True,
            "installationID": installationID,
            "statePath": str(self.statePath),
            "stageDirectory": str(stageDirectory),
            "scriptIdentity": scriptIdentity,
            "environmentRemove": environmentRemove,
            "environmentOverrides": environmentOverrides,
        }

        statePublished = False

        try:
            if self.readState() != originalState:
                raise RuntimeError("Installation state changed before staging.")

            stageDirectory.mkdir()

            with (stageDirectory / "official-user.tcsh").open("xb") as target:
                target.write(sourceBytes)
            with (stageDirectory / "user-setup.tcsh").open("xb") as target:
                target.write(scriptBytes)

            (stageDirectory / "approved-plan.txt").write_text(details, encoding="utf-8")
            (stageDirectory / "run-user.py").write_text(workerSource, encoding="utf-8")
            (stageDirectory / "request.json").write_text(
                json.dumps(request, sort_keys=True, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            # Use -I so notebook PYTHONPATH/site customization does not affect
            # the standalone bookkeeping process.
            command = [
                sys.executable,
                "-I",
                str(stageDirectory / "run-user.py"),
                str(stageDirectory / "request.json"),
            ]
            commandPath = stageDirectory / "run-user.command"
            commandPath.write_text(
                "#!/bin/zsh\nexec " + " ".join(shlex.quote(value) for value in command) + "\n",
                encoding="utf-8",
            )
            commandPath.chmod(0o700)

            temporary = self.directory / f".state-{uuid4().hex}.json"
            try:
                with temporary.open("x", encoding="utf-8") as target:
                    json.dump(state, target, sort_keys=True, indent=2, allow_nan=False)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, self.statePath)
                statePublished = True
            finally:
                if temporary.exists(): temporary.unlink()

            result = subprocess.run(
                ["/usr/bin/open", "-a", "Terminal", str(commandPath)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Terminal launch was refused.")

        except BaseException:
            if statePublished:
                # Launch failure/timeout can be ambiguous. Do not race a
                # worker by overwriting its state or allowing a second launch.
                pglMessages.warning(
                    "The AFNI user-stage launch could not be confirmed. "
                    "Check Terminal and the saved installation state before "
                    "retrying. The pending record and staged files were retained."
                )
            else:
                # No launcher was invoked. Remove only our empty claim;
                # keep staged files for diagnosis without changing old state.
                claimPath.rmdir()
            raise

        pglMessages.message(
            "AFNI user setup was requested in Terminal.\n"
            "Follow its output. Launching Terminal does not mean installation succeeded.\n"
            f"When it finishes, run {resume} to review diagnostics."
        )
        return state
    
    ################################################################
    # Review completed installation diagnostics
    ################################################################
    def reviewInstallation(self):
        """Review reports and freshly probe the selected installation.

        Returns the updated installation record after explicit acceptance and
        successful probes. Cancellation/refusal returns None.

        Does not rerun the installer, modify shell configuration, or regenerate
        system-check reports.

        'verified' means executable probes passed and the user explicitly
        accepted the displayed diagnostics. It does not certify every optional
        dependency, GUI operation, or scientific result.
        """
        import hashlib
        import json
        import os
        from pathlib import Path
        from uuid import uuid4

        from .pglDialog import pglDialogs
        from .pglMessages import pglMessages

        state = self.readState()
        if state is None or state["status"] != "verificationPending":
            raise ValueError("Installation is not awaiting diagnostic review.")

        systemCheckPath = Path(state["systemCheckLog"])
        stageDirectory = systemCheckPath.parent
        reports = []
        sections = []

        for label, path, required in (
            ("Worker system check", systemCheckPath, True),
            ("Official script system check", stageDirectory / "out.ASC.txt", False),
        ):
            if not path.exists():
                if required: raise FileNotFoundError(f"Required verification report is missing: {path}")
                sections.append(f"{label}\nNot available: {path}\n")
                continue

            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Verification report must be a regular nonsymlink file: {path}")

            # Never accept a report after showing only an undisclosed prefix.
            with path.open("rb") as source:
                content = source.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise ValueError(f"Report is too large for this review dialog. Inspect it separately: {path}")
            if not content:
                raise ValueError(f"Verification report is empty: {path}")

            reports.append({
                "label": label,
                "localPath": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            })
            sections.append(
                f"{label}\nFile: {path}\n{'-' * 72}\n"
                + content.decode("utf-8", errors="replace")
            )

        details = (
            "AFNI INSTALLATION REVIEW\n"
            "========================\n\n"
            f"Installation: {self.root}\n\n"
            "Read the recommendations in both reports, including 'Please Fix'.\n"
            "The reports may differ because the official script and worker\n"
            "verification run in different process environments.\n\n"
            "Resolve problems relevant to your intended use before accepting.\n"
            "Test AFNI/SUMA GUI functionality separately if you need those tools.\n\n"
            "Accepting records your diagnostic review. PGL will then freshly\n"
            "probe this installation's executables before marking it verified.\n"
            "No installer will run.\n\n"
            + "\n\n".join(sections)
        )

        pglMessages.print(details)

        if not pglDialogs.confirm(
            "Have you reviewed these diagnostics and resolved issues required for your intended use?",
            title="Review AFNI installation",
            details=details,
            confirmButtonText="Accept diagnostics",
            copyText="Copy reports",
        ):
            return None

        if self.readState() != state:
            raise RuntimeError("Installation state changed during diagnostic review.")

        # The user accepted these exact report bytes, not a report that changed
        # while the dialog was open.
        for record in reports:
            actual = pglExternalTool._fileIdentity(record["localPath"])
            expected = {key: record[key] for key in ("sha256", "size_bytes")}
            if actual != expected:
                raise ValueError("A verification report changed during review. Review the new report before accepting.")

        probe = pglAFNI(directory=str(self.root / "abin"))

        # Bypass the public pending-state guard: this is the coordinator's
        # explicit verification step, not an ordinary availability query.
        if not probe._checkAvailable("3dvolreg", cacheDirectory=stageDirectory):
            return None

        if self.readState() != state:
            raise RuntimeError("Installation state changed during executable verification.")

        updated = deepcopy(state)
        updated.update({
            "status": "verified",
            "verifiedAt": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "verification": {
                "meaning": "Executable probes passed and the user accepted the diagnostic reports.",
                "reports": reports,
                "availability": deepcopy(probe.lastAvailability),
                "guiTests": "Not automatically verified by PGL.",
            },
        })

        temporary = self.directory / f".state-{uuid4().hex}.json"

        try:
            with temporary.open("x", encoding="utf-8") as target:
                json.dump(updated, target, sort_keys=True, indent=2, allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.statePath)
        finally:
            if temporary.exists(): temporary.unlink()

        pglMessages.message(
            f"AFNI executable checks passed and diagnostic review was recorded.\n"
            f"Executable directory: {self.root / 'abin'}"
        )
        return updated
    
################################################################
# MATLAB viewer access
################################################################
class pglMATLAB(pglExternalTool):
    """MATLAB installations, explicit validation, and the existing mlrVol viewer.

    Configuration supports named installation profiles with one saved default.

    Construction resolves the executable but does not launch MATLAB.
    getStatus() explicitly launches a short MATLAB batch command to establish
    runtime availability and report the actual version.

    The viewer remains a separate interactive operation, not a scientific
    batch invocation. It opens an independent copy of the supplied NIfTI.

    No automatic installation, license changes, PATH edits, or MATLAB startup
    edits are performed.
    """

    toolName = "matlab"
    toolLabel = "MATLAB"
    multipleInstallations = True
    resolvedLocationField = "executable"

    configurationFields = {
        "executable": {
            "type": "path",
            "default": "",
            "label": "MATLAB executable",
            "help": (
                "Full path to MATLAB's executable, such as "
                "/Applications/MATLAB_R2026b.app/bin/matlab. "
                "Blank searches PATH, then /Applications on macOS. "
                "Multiple discovered installations require an explicit choice."
            ),
        },
    }

    @staticmethod
    def _findExecutable(executable=None):
        """Resolve MATLAB without launching it or changing configuration.

        An explicit path is authoritative: an invalid path never falls back.

        Automatic discovery:
            PATH first.
            If absent on macOS, /Applications/MATLAB*.app/bin/matlab.
            Multiple macOS candidates are rejected rather than version-sorted
            into an implicit selection.
        """
        import os
        import shutil
        import sys
        from pathlib import Path

        if executable is not None:
            text = os.fspath(executable)
            if (
                not isinstance(text, str) or not text.strip()
                or any(character in text for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("MATLAB executable must be a nonblank local path.")

            expanded = Path(text).expanduser()
            if not expanded.is_absolute():
                raise ValueError("Specify the full absolute path to MATLAB's executable.")

            candidate = expanded

        else:
            discovered = shutil.which("matlab")
            candidate = Path(discovered).absolute() if discovered is not None else None

            if candidate is None and sys.platform == "darwin":
                candidates = [
                    path for path in sorted(Path("/Applications").glob("MATLAB*.app/bin/matlab"))
                    if path.is_file() and os.access(path, os.X_OK)
                ]

                if len(candidates) == 1:
                    candidate = candidates[0]
                elif len(candidates) > 1:
                    choices = "\n".join(str(path) for path in candidates)
                    raise ValueError(
                        "Multiple MATLAB installations were found. Configure a "
                        "profile with an explicit executable:\n" + choices
                    )

            if candidate is None:
                raise FileNotFoundError(
                    "MATLAB was not found. Configure its full executable path."
                )

        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"MATLAB executable is unavailable: {candidate}")

        # Keep the invocation path, including a caller-selected symlink.
        # The command runner independently resolves and identifies its target.
        return candidate.absolute()

    def __init__(self, *, matlabExecutable=None, installation=None):
        """Select an explicit executable, named profile, or saved default.

        Explicit executable and named installation are mutually exclusive.
        Does not launch MATLAB or establish license/runtime availability.
        """
        if matlabExecutable is not None and installation is not None:
            raise ValueError("Supply matlabExecutable or installation, not both.")

        if matlabExecutable is not None:
            selectedExecutable = matlabExecutable
            selectionSource = "explicit"
            selectedProfile = None

        else:
            configuration = type(self).getConfiguration(installation=installation)
            configured = configuration["executable"]
            selectedExecutable = configured if configured.strip() else None
            selectionSource = "profile" if installation is not None else ("saved" if selectedExecutable else "automatic")

            document = type(self)._readToolSettings(type(self)._toolSettingsPath())
            profiles = type(self)._getInstallationProfiles(document)
            selectedProfile = profiles["defaultInstallation"] if installation is None else installation

        executable = self._findExecutable(selectedExecutable)

        # Explicit executable path prevents the generic runner from applying
        # executable-directory defaults or resolving a different installation.
        super().__init__(executable=str(executable))
        self.configurationSource = selectionSource
        self.installation = selectedProfile
        self.lastResult = None

    @classmethod
    def getStatus(cls, *, configuration=None, timeout=10.0, cacheDirectory=None):
        """Probe the selected MATLAB runtime and return a common tool report.

        Launches MATLAB with -batch and a fixed version-printing expression.
        This can initialize MATLAB, run startup.m, and check out a license.

        timeout is a subprocess deadline. MATLAB startup may need substantially
        longer than the generic 10-second tool default; explicitly request a
        suitable timeout when validating MATLAB installations.

        Probe stdout/stderr and executable identity use the existing command
        runner. No source paths are interpolated into MATLAB code.

        An executable that exists but cannot run is not reported as available.
        Missing configuration/runtime failures are returned as status records.
        KeyboardInterrupt and SystemExit propagate.
        """
        import math

        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite.")

        configuration = cls.getConfiguration() if configuration is None else _copyProvenance(configuration)
        settings = cls._settingsFromValues(configuration)

        report = {
            "toolName": cls.toolName,
            "label": cls.toolLabel,
            "status": "Unavailable",
            "available": False,
            "version": None,
            "location": None,
            "diagnostic": None,
            "details": {},
        }
        result = None

        try:
            configured = settings.executable
            executable = cls._findExecutable(configured if configured.strip() else None)
            report["location"] = str(executable)

            # Generic runner with an explicit path: do not consult another
            # profile or silently resolve another MATLAB during this probe.
            runner = pglExternalTool(executable=str(executable))
            result = runner._runCommand(
                arguments=[
                    "-batch",
                    "fprintf('PGL_MATLAB_VERSION:%s\\n',version); "
                    "fprintf('PGL_MATLAB_RELEASE:%s\\n',version('-release'));",
                ],
                timeout=timeout,
                cacheDirectory=cacheDirectory,
            )

            # Scan incrementally: startup messages may precede the markers.
            # Bound diagnostic processing rather than loading arbitrary logs.
            found = {}
            with result.stdoutPath.open("rb") as source:
                remaining = 4 * 1024 * 1024
                while remaining > 0:
                    line = source.readline(min(65536, remaining))
                    if not line: break
                    remaining -= len(line)
                    text = line.decode("utf-8", errors="replace").strip()

                    for marker, field in (
                        ("PGL_MATLAB_VERSION:", "version"),
                        ("PGL_MATLAB_RELEASE:", "release"),
                    ):
                        if text.startswith(marker):
                            value = text[len(marker):].strip()
                            if not value or field in found:
                                raise ValueError("MATLAB returned empty or repeated version markers.")
                            found[field] = value

            if set(found) != {"version", "release"}:
                raise ValueError("MATLAB exited successfully without the expected version report.")

            report["status"] = "Available"
            report["available"] = True
            report["version"] = f"{found['release']} — {found['version']}"
            report["details"] = {
                "release": found["release"],
                "version": found["version"],
                "probe": result.provenance,
            }

        except Exception as error:
            if isinstance(error, pglToolExecutionError):
                result = error.result

            report["diagnostic"] = f"{type(error).__name__}: {error}"
            if result is not None:
                report["details"] = {"probe": result.provenance}

        return _copyProvenance(report)

    @classmethod
    def viewer(cls, inputImage, *, matlabExecutable=None, installation=None, mlrDirectory=None, cacheDirectory=None):
        """Open a copy of an image in a new MATLAB desktop using mlrVol.

        MATLAB selection:
            matlabExecutable: explicit full executable path.
            installation: explicit saved profile.
            Neither: saved default profile, then its discovery policy.

        Explicit path and installation cannot both be supplied.

        mlrDirectory:
            Optional directory containing mlrVol.m.
            None retains MATLAB's normal startup/path configuration.
            Other mrTools dependencies must also be on MATLAB's path.

        Returns a launcher process, or None after a warning on failure.
        Launch does not establish that MATLAB successfully loaded the image.

        Viewer copy and startup logs are retained. Closing mlrVol does not
        close the MATLAB desktop. No acquisition files are modified.
        """
        import os
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        from .pglMessages import pglMessages

        try:
            source = Path(inputImage).expanduser().resolve()
            if not source.is_file() or not source.name.lower().endswith((".nii", ".nii.gz")):
                raise ValueError("inputImage must be an existing local .nii or .nii.gz file.")

            tool = cls(
                matlabExecutable=matlabExecutable,
                installation=installation,
            )

            mlrPath = None
            if mlrDirectory is not None:
                mlrPath = Path(mlrDirectory).expanduser().resolve()
                if not (mlrPath / "mlrVol.m").is_file():
                    raise ValueError("mlrDirectory must contain mlrVol.m.")

            cacheRoot = None
            if cacheDirectory is not None:
                text = os.fspath(cacheDirectory)
                if (
                    not isinstance(text, str) or not text.strip()
                    or "://" in text or "::" in text
                    or any(character in text for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError("cacheDirectory must be an ordinary local path.")
                cacheRoot = Path(text).expanduser().resolve()
                cacheRoot.mkdir(parents=True, exist_ok=True)

            workingDirectory = Path(tempfile.mkdtemp(
                prefix="pgl-mlrvol-",
                dir=None if cacheRoot is None else str(cacheRoot),
            ))

            # Real copy: mlrVol can edit/save, so never pass an acquisition
            # hard link or a shared source path as its writable image.
            viewerImage = workingDirectory / source.name
            pglMessages.message(f"Creating an independent viewer copy of {source.name}...")
            shutil.copyfile(source, viewerImage)

            environment = dict(os.environ)
            environment["PGL_MLRVOL_IMAGE"] = str(viewerImage)
            environment["PGL_MLRVOL_DIRECTORY"] = "" if mlrPath is None else str(mlrPath)

            matlabCommand = (
                "try; "
                "pglMlrDirectory=getenv('PGL_MLRVOL_DIRECTORY'); "
                "if ~isempty(pglMlrDirectory); addpath(pglMlrDirectory); end; "
                "assert(exist('mlrVol','file')~=0,"
                "'mlrVol is not on the MATLAB path. Configure mrTools or supply mlrDirectory.'); "
                "fprintf('Using mlrVol: %s\\n',which('mlrVol')); "
                "mlrVol(getenv('PGL_MLRVOL_IMAGE')); "
                "drawnow; "
                "catch pglViewerError; "
                "disp(getReport(pglViewerError,'extended','hyperlinks','off')); "
                "end;"
            )

            stdoutPath = workingDirectory / "stdout.log"
            stderrPath = workingDirectory / "stderr.log"
            matlabLog = workingDirectory / "matlab.log"

            # Interactive GUI launch remains separate from scientific batch
            # execution. -r leaves the desktop open; -batch would terminate it.
            with stdoutPath.open("xb") as stdout, stderrPath.open("xb") as stderr:
                process = subprocess.Popen(
                    [
                        tool.executable,
                        "-desktop",
                        "-logfile", str(matlabLog),
                        "-r", matlabCommand,
                    ],
                    cwd=str(workingDirectory),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                )

            pglMessages.message(
                f"MATLAB viewer process started for {source.name}.\n"
                f"Executable: {tool.executable}\n"
                f"Installation profile: {tool.installation or '(explicit executable)'}\n"
                f"Launcher process: {process.pid}\n"
                f"Viewer copy and startup logs: {workingDirectory}"
            )
            return process

        except Exception as error:
            pglMessages.warning(f"mlrVol viewer could not be launched: {type(error).__name__}: {error}")
            return None
        
class pglGLMsingle(pglExternalTool):
    """GLMsingle in the active Python environment.

    Configuration, construction, and cached printing never install software.
    Explicit validation probes a fresh process using sys.executable.
    Explicit installation requires confirmation and uses the same interpreter.

    A source checkout is an installation input, never an implicit sys.path edit.
    Multiple Python environments or imported versions are not managed here.
    """

    toolName = "glmsingle"
    toolLabel = "GLMsingle"
    multipleInstallations = False
    resolvedLocationField = None

    # An installed package directory is not necessarily an installable checkout,
    # so do not prefill sourceDirectory from a discovered package location.
    configurationFields = {
        "sourceDirectory": {
            "type": "path",
            "default": "~/proj",
            "label": "Installation parent directory",
            "help": (
                "Choose the parent folder for installation. PGL downloads the "
                "official GitHub repository into a GLMsingle subfolder: selecting "
                "~/proj creates ~/proj/GLMsingle. Do not select the GLMsingle "
                "subfolder itself. Blank uses ~/proj. Other folders inside the "
                "parent are untouched; an existing nonempty GLMsingle subfolder "
                "will not be overwritten."
            ),
        },
        "editableInstall": {
            "type": "bool",
            "default": False,
            "label": "Editable installation",
            "help": (
                "Install against the downloaded GLMsingle checkout so subsequent "
                "source edits affect execution. Otherwise use a normal pip installation."
            ),
        },
    }

    officialSource = "git+https://github.com/cvnlab/GLMsingle.git"

    def __init__(self):
        """Create an adapter without importing GLMsingle or launching commands."""
        import sys

        super().__init__(executable=sys.executable)
        self.lastResult = None
        self.lastAvailability = None

    @classmethod
    def _settingsFromValues(cls, values):
        """Validate installation preferences without requiring an existing checkout."""
        return super()._settingsFromValues(values)
    
    @classmethod
    def getStatus(cls, *, configuration=None, timeout=10.0, cacheDirectory=None):
        """Check GLMsingle in a fresh process using the active interpreter.

        Imports the fitting API but does not instantiate a model or fit data.
        Reports package location, distribution version when available, and the
        actual fit() signature.

        A fresh subprocess avoids importing or replacing GLMsingle in the
        notebook merely to inspect availability. The report describes a fresh
        process in this environment, not a previously cached notebook import.

        Importing package code is part of explicit validation. No installation,
        sys.path modification, or automatic dependency repair occurs.
        """
        import json
        import math
        import sys
        from uuid import uuid4

        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite.")

        configuration = cls.getConfiguration() if configuration is None else _copyProvenance(configuration)
        cls._settingsFromValues(configuration)

        report = {
            "toolName": cls.toolName,
            "label": cls.toolLabel,
            "status": "Unavailable",
            "available": False,
            "version": None,
            "location": None,
            "diagnostic": None,
            "details": {"pythonExecutable": sys.executable},
        }

        marker = f"PGL_GLMSINGLE_{uuid4().hex}:"

        # Fixed Python code; marker is supplied as an argument, not interpolated.
        probeCode = r'''
import importlib
import inspect
import json
import sys
from importlib import metadata
from pathlib import Path

record = {
    "status": "Unavailable",
    "available": False,
    "version": None,
    "location": None,
    "diagnostic": None,
    "fitSignature": None,
}

try:
    package = importlib.import_module("glmsingle")
    implementation = importlib.import_module("glmsingle.glmsingle")
    model = getattr(implementation, "GLM_single", None)

    if not inspect.isclass(model) or not callable(getattr(model, "fit", None)):
        raise TypeError("Expected glmsingle.glmsingle.GLM_single with a callable fit().")

    filename = getattr(implementation, "__file__", None)
    if filename:
        record["location"] = str(Path(filename).resolve())

    try:
        record["version"] = metadata.version("glmsingle")
    except metadata.PackageNotFoundError:
        value = getattr(package, "__version__", None)
        if isinstance(value, str):
            record["version"] = value

    record["fitSignature"] = str(inspect.signature(model.fit))
    record["available"] = True
    record["status"] = "Available"

except ModuleNotFoundError as error:
    record["status"] = "Not installed" if error.name == "glmsingle" else "Broken import"
    record["diagnostic"] = f"{type(error).__name__}: {error}"

except Exception as error:
    record["status"] = "Broken import"
    record["diagnostic"] = f"{type(error).__name__}: {error}"

print(sys.argv[1] + json.dumps(record, allow_nan=False), flush=True)
'''

        result = None

        try:
            runner = pglExternalTool(executable=sys.executable)
            result = runner._runCommand(
                arguments=["-c", probeCode, marker],
                timeout=timeout,
                cacheDirectory=cacheDirectory,
            )

            with result.stdoutPath.open("rb") as source:
                content = source.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError("GLMsingle probe output exceeds 1 MiB.")

            matches = [
                line[len(marker):]
                for line in content.decode("utf-8", errors="replace").splitlines()
                if line.startswith(marker)
            ]
            if len(matches) != 1:
                raise ValueError("GLMsingle probe did not return one identifiable report.")

            checked = json.loads(matches[0])
            if (
                type(checked) is not dict
                or set(checked) != {
                    "status", "available", "version", "location",
                    "diagnostic", "fitSignature",
                }
                or type(checked["available"]) is not bool
            ):
                raise ValueError("Malformed GLMsingle probe report.")

            for field in ("status", "version", "location", "diagnostic", "fitSignature"):
                if checked[field] is not None and not isinstance(checked[field], str):
                    raise ValueError(f"Invalid GLMsingle probe field {field!r}.")

            report.update({
                key: checked[key]
                for key in ("status", "available", "version", "location", "diagnostic")
            })
            report["details"].update({
                "fitSignature": checked["fitSignature"],
                "probe": result.provenance,
            })

        except Exception as error:
            if isinstance(error, pglToolExecutionError):
                result = error.result

            report["status"] = "Check failed"
            report["available"] = False
            report["diagnostic"] = f"{type(error).__name__}: {error}"

            if result is not None:
                report["details"]["probe"] = result.provenance

        return _copyProvenance(report)

    def install(self, *, timeout=None, cacheDirectory=None):
        """Download GLMsingle from GitHub and install into active Python.

        Uses the configured parent directory, defaulting to ~/proj, and creates
        a GLMsingle subfolder. That subfolder must be absent or empty.
        Existing sibling files and directories are untouched.
        
        Steps:
            Confirm source, destination, interpreter, and commands.
            Clone the official repository into the selected directory.
            Record the checked-out commit.
            Install that checkout using sys.executable -m pip.

        Existing nonempty directories are never overwritten, pulled, reset,
        or deleted. Failed clones/installations may leave files for diagnosis.

        Git and pip output is printed live and retained by the existing command
        runner. No shell commands or sys.path modifications are used.

        Installation requires Git on PATH and network access. Package build
        code executes, and pip may download or change dependencies.

        Returns the pip command result, or None on cancellation or a handled
        command failure. Restart the kernel and validate afterward.
        """
        import math
        import os
        import shlex
        import sys
        from pathlib import Path

        from .pglDialog import pglDialogs
        from .pglMessages import pglMessages
        self.printInstallationNotice()

        recorder = getExecutionRecorder()
        if recorder is not None and recorder.actionID is not None:
            raise RuntimeError("Install GLMsingle before starting scientific actions.")

        if timeout is not None and (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive and finite, or None.")

        settings = type(self)._settingsFromValues(type(self).getConfiguration())
        destinationText = settings.sourceDirectory.strip() or "~/proj"
        destination = Path(destinationText).expanduser().absolute() / "GLMsingle"
        
        if "://" in destinationText or "::" in destinationText:
            raise ValueError("The installation directory must be an ordinary local path.")

        def requireUnusedDestination():
            """Warn and refuse installation without modifying existing contents."""
            if destination.is_symlink():
                message = f"Installation cancelled: the destination is a symlink: {destination}"
            elif destination.exists() and not destination.is_dir():
                message = f"Installation cancelled: the destination is not a directory: {destination}"
            elif destination.exists() and any(destination.iterdir()):
                message = (
                    f"Installation cancelled: the directory is not empty: {destination}\n"
                    "Existing contents were not changed. If GLMsingle is already "
                    "installed, validate it with:\n"
                    "pglGLMsingle.print(validate=True, timeout=120)"
                )
            else:
                return True

            pglMessages.warning(message)
            return False

        if not requireUnusedDestination(): return None

        repository = "https://github.com/cvnlab/GLMsingle.git"

        # Resolve Git before requesting approval or creating directories.
        gitRunner = pglExternalTool(executable="git")
        gitExecutable = gitRunner._resolveExecutable("git", dict(os.environ))
        gitRunner = pglExternalTool(executable=str(gitExecutable))
        pythonRunner = pglExternalTool(executable=sys.executable)

        cloneArguments = ["clone", "--", repository, str(destination)]
        pipArguments = ["-m", "pip", "install"]
        if settings.editableInstall:
            pipArguments.append("-e")
        pipArguments.append(str(destination))

        details = (
            f"Repository:\n{repository}\n\n"
            f"Download/install directory:\n{destination}\n\n"
            f"Python environment:\n{sys.executable}\n\n"
            "Commands:\n"
            f"{shlex.join([str(gitExecutable), *cloneArguments])}\n"
            f"{shlex.join([sys.executable, *pipArguments])}\n\n"
            "The repository's current default branch will be downloaded. Its "
            "exact commit will be reported before installation.\n\n"
            "Pip executes package build code and may download or change "
            "dependencies. Existing nonempty directories will not be replaced.\n\n"
            "Restart the notebook kernel after installation."
        )

        if not pglDialogs.confirm(
            "Download and install GLMsingle?",
            title="Install GLMsingle",
            details=details,
            confirmButtonText="Download and install",
        ):
            return None

        self.lastResult = None
        self.lastInstallationResults = []

        def runStep(runner, arguments, *, cwd=None):
            try:
                result = runner._runCommand(
                    arguments=arguments,
                    cwd=cwd,
                    timeout=timeout,
                    cacheDirectory=cacheDirectory,
                    streamOutput=True,
                )
            except pglToolExecutionError as error:
                self.lastResult = error.result
                self.lastInstallationResults.append(error.result)
                raise

            self.lastResult = result
            self.lastInstallationResults.append(result)
            return result

        try:
            # Recheck after the dialog, before modifying the filesystem.
            if not requireUnusedDestination(): return None
            destination.parent.mkdir(parents=True, exist_ok=True)

            pglMessages.message(f"Downloading GLMsingle into {destination}")
            runStep(gitRunner, cloneArguments)

            pglMessages.message("Downloaded GLMsingle commit:")
            commitResult = runStep(
                gitRunner,
                ["rev-parse", "HEAD"],
                cwd=destination,
            )
            commit = commitResult.stdoutPath.read_text(encoding="utf-8").strip()
            self.lastInstallationCommit = commit

            if not any((destination / name).is_file() for name in ("pyproject.toml", "setup.py")):
                raise ValueError(
                    "The downloaded repository has no supported Python package "
                    "definition at its root. The checkout and logs were retained; "
                    "pip was not run."
                )

            pglMessages.message(f"Installing GLMsingle into {sys.executable}")
            result = runStep(pythonRunner, pipArguments)

            pglMessages.message(
                f"GLMsingle download and pip installation completed.\n"
                f"Checkout: {destination}\n"
                f"Commit: {commit}\n"
                f"Pip logs: {result.logDirectory}\n"
                "Restart the kernel, then validate GLMsingle."
            )
            return result

        except Exception as error:
            logDirectories = [
                str(result.logDirectory)
                for result in self.lastInstallationResults
                if result.logDirectory is not None
            ]
            logText = "\n".join(logDirectories) or "(No command logs were created.)"

            pglMessages.warning(
                f"GLMsingle installation did not complete: {type(error).__name__}: {error}\n"
                f"Command logs:\n{logText}\n"
                "Downloaded files, if any, were retained. Nothing was automatically deleted.",
                level=1,
            )
            return None
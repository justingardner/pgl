################################################################
#   filename: pglPipeline.py
#    purpose: Pipeline 
#         by: JLG
#       date: Aug 1, 2026
################################################################

#############
# Import
#############
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

# Track the active execution, not the action instance. ContextVar keeps
# independent execution contexts separate and supports nested action calls.
_activeActionID = ContextVar("pglActiveActionID", default=None)

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
        """Update outcome and timing without changing captured provenance."""
        self.actionStatus = action.status
        self.actionError = action.error
        self.endTime = pglTimestamp.getSecs()
        self.runDuration = self.endTime - self.startTime

    def toJSONdict(self, type="all"):
        """Preserve absent optional fields in previously saved history."""
        data = super().toJSONdict(type=type)
        originalFields = getattr(self, "_originalHistoryFields", None)

        for name in ("actionID", "parentActionID", "actionCode", "actionInput"):
            value = data.get(name)
            empty = not value if name in ("actionCode", "actionInput") else value is None

            if empty and (originalFields is None or name not in originalFields):
                data.pop(name, None)

        return data
    
    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore history without creating new execution provenance."""
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

    def history(self, verbose=False):
        """Print the object and its numbered history, optionally with settings."""
        self.print()
        pglMessages.printHeader("action history")

        if self._failedActionIDs:
            pglMessages.warning("This working object has failed action attempts and is blocked from checkpoint saving. Reload a saved checkpoint before continuing with a saveable analysis.", level=2)

        if not self.actionHistory:
            pglMessages.message(f"{type(self).__name__} has no history")
            return

        for iAction, action in enumerate(self.actionHistory):
            pglMessages.print(f"{iAction}: {action.toString()}")

            if verbose:
                if action.actionSettings:
                    for name, value in sorted(action.actionSettings.items()):
                        pglMessages.print(f"    {name} = {value!r}")
                else:
                    pglMessages.print("    (No saved configuration parameters)")
                    
    def getPipeline(self, name=None):
        """Extract an independent recipe from this object's recorded history.

        Does not edit history, load checkpoints, or execute actions.
        Extraction does not establish a default replay input.
        """
        return pglPipeline.fromHistory(self, name=name)

 
########################
# class pglAction
########################
class pglAction(pglActionable):
    name = Unicode("", help="Name of action")
    status = Instance(pglActionStatus, help="action status")
    error = Instance(Exception, allow_none=True, default_value=None, help="error")
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
        """Route configuration and run arguments, warning about unsupported names."""
        action = cls()

        configureNames = cls._getKeywordNames(action.configure) | set(action.traits())
        runNames = cls._getKeywordNames(action.run) | cls._getKeywordNames(action._run)
        validNames = configureNames | runNames

        if args:
            pglMessages.warning(f"{cls.__name__}.execute() requires keyword arguments, but received {len(args)} positional argument(s). Pass arguments as name=value.", level=2)
            action.printConfiguration()
            pglMessages.print(f"Accepted execute arguments: {', '.join(sorted(validNames)) or '(none)'}")
            return None

        unknownNames = set(kwargs) - validNames
        if unknownNames:
            pglMessages.warning(f"{cls.__name__}.execute() got unknown argument(s): {', '.join(sorted(unknownNames))}. Argument names are case-sensitive.", level=2)
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
        
    def run(self, *args, **kwargs):
        """Execute an action with fresh identity, nested context, and history.

        Successful actions attach history to their returned actionable.
        Failed actions attach history to directly supplied input actionables
        and any actionable result returned before failure.

        Failure marks those working objects unsafe for checkpoint saving.
        Later successful executions do not clear that state.

        Explicit setError() failures are preserved.
        Ordinary exceptions warn and return None.
        KeyboardInterrupt and SystemExit are recorded and propagated.

        Scientific mutations are not rolled back.
        """
        if getattr(self, "_executionInProgress", False):
            raise RuntimeError(f"Action {self.name} is already running. Use a separate action instance for another execution.")

        if not self.isConfigured():
            pglMessages.warning(f"Action {self.name} is not yet configured.", level=1)
            return None

        inputActionables = []

        # Inspect only directly supplied objects. Discovering provenance
        # recipients must not open scientific fields or deferred payloads.
        for value in (*args, *kwargs.values()):
            if isinstance(value, pglActionable) and value is not self and all(value is not existing for existing in inputActionables):
                inputActionables.append(value)

        previousStatus = self.status
        previousError = self.error
        self._executionInProgress = True
        entry = None
        result = None
        contextToken = None

        def appendEntry(owner):
            # Settings equality compares UUIDs; use identity to detect whether
            # this exact execution entry has already been attached.
            if owner is not self and all(existing is not entry for existing in owner.actionHistory):
                owner.actionHistory.append(entry)

        try:
            self.error = None
            self.status = pglActionStatus.RUNNING

            try:
                entry = pglActionHistory(self)
                self.actionHistory.append(entry)

            except BaseException:
                # Snapshot failure occurs before scientific execution.
                self.status = previousStatus
                self.error = previousError
                raise

            contextToken = _activeActionID.set(entry.actionID)

            try:
                result = self._run(*args, **kwargs)

                if self.status == pglActionStatus.FAILED or self.error is not None:
                    self.status = pglActionStatus.FAILED

                    if self.error is None:
                        self.error = RuntimeError(f"Action {self.name} reported failure without an error description.")

                elif self.status == pglActionStatus.SKIPPED:
                    pass

                elif result is None:
                    self.error = RuntimeError(f"Action {self.name} returned None without reporting failure or an explicit skip.")
                    self.status = pglActionStatus.FAILED
                    pglMessages.warning(str(self.error), level=1)

                else:
                    self.status = pglActionStatus.SUCCESS

            except (KeyboardInterrupt, SystemExit) as error:
                self.error = RuntimeError(f"{type(error).__name__}: action execution was interrupted.")
                self.status = pglActionStatus.FAILED
                raise

            except Exception as error:
                self.error = error
                self.status = pglActionStatus.FAILED
                pglMessages.warning(f"Error running action {self.name}: {error}", level=1)

            finally:
                if self.status == pglActionStatus.FAILED:
                    # Even an action returning None may already have mutated
                    # its input. Mark every direct input conservatively.
                    for owner in inputActionables:
                        owner._recordFailedAction(entry.actionID)

                    # Ingestion may create and return a partial session without
                    # having received an input session.
                    if isinstance(result, pglActionable) and result is not self:
                        result._recordFailedAction(entry.actionID)

                entry.update(self)

                if self.status == pglActionStatus.SUCCESS and isinstance(result, pglActionable):
                    # A new returned object must not silently erase a direct
                    # input's unresolved failure state.
                    for owner in inputActionables:
                        for failedActionID in owner._failedActionIDs:
                            result._recordFailedAction(failedActionID)

                    appendEntry(result)
                else:
                    for owner in inputActionables:
                        appendEntry(owner)

                    if isinstance(result, pglActionable):
                        appendEntry(result)

            if self.status == pglActionStatus.FAILED:
                return None

            if self.status == pglActionStatus.SUCCESS and not isinstance(result, pglActionable):
                pglMessages.warning(f"Action {self.name} returned type {type(result).__name__}; execution history is retained on the action and directly supplied input actionables.", level=1)

            return result

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
        """Print configuration traits, explicit configure arguments, and run arguments."""
        traits = self.traits()
        baseTraitNames = set(pglAction.class_traits())
        configureParameters = inspect.signature(self.configure).parameters
        runNames = self._getKeywordNames(self.run) | self._getKeywordNames(self._run)

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

                for methodName in ("configure", "run", "_run"):
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
    
    @staticmethod
    def _resolveActionClass(moduleName, qualname):
        """Resolve an action from already-loaded application code.

        Never import a module named by serialized metadata or execute saved
        source. Only ordinary module/class namespaces are traversed.

        Constructors remain trusted application code, not sandboxed code.
        """
        import sys

        if not isinstance(moduleName, str) or not moduleName or not isinstance(qualname, str) or not qualname:
            raise ValueError("Action identity requires a module and qualified class name.")

        current = sys.modules.get(moduleName)

        if current is None:
            raise ValueError(f"Action module {moduleName!r} is not loaded. Import the trusted action package before loading this pipeline.")

        for index, part in enumerate(qualname.split(".")):
            if not part.isidentifier() or (index and not isinstance(current, type)):
                raise ValueError(f"Action class is not addressable through its module: {moduleName}.{qualname}")

            current = vars(current).get(part)

            if current is None:
                raise ValueError(f"Action class is unavailable: {moduleName}.{qualname}")

        if not isinstance(current, type) or not issubclass(current, pglAction):
            raise TypeError(f"{moduleName}.{qualname} is not a pglAction class.")

        if current.__module__ != moduleName or current.__qualname__ != qualname:
            raise ValueError(f"Installed action identity does not match {moduleName}.{qualname}.")

        return current

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

        Does not modify source history, copy scientific payloads, or execute
        actions. Invalid provenance, inconsistent history, and unsupported
        extraction structures raise.

        Current database correspondence is checked later when default replay
        loads its checkpoint.
        """
        import hashlib
        import json

        from .pglDatabase import pglDatabase

        if not isinstance(actionable, pglActionable):
            raise TypeError("Pipeline extraction requires a pglActionable object.")

        history = list(actionable.actionHistory)
        candidates = []

        for index, entry in enumerate(history):
            if entry.parentActionID is not None or entry.actionStatus != pglActionStatus.SUCCESS:
                continue

            if entry.actionInput.get("kind") == "checkpoint":
                candidates.append((index, entry))

        if not candidates:
            pipeline = cls._fromHistoryRecipe(actionable, name=name)
            pglMessages.message("Pipeline extracted without a verified default input; supply session or sessionID when running it.")
            return pipeline

        sourceIndex, sourceEntry = candidates[0]
        inputRecord = deepcopy(sourceEntry.actionInput)
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
    
    @classmethod
    def _fromHistoryRecipe(cls, actionable, *, name=None):
        """Extract successful execution trees into an independent recipe.

        Pipeline executions become nested recipe steps. Their recorded
        selection determines which steps are retained, and child history
        verifies what actually executed.

        Failed/skipped top-level attempts are omitted with their descendants.
        Missing parents, duplicate IDs, cycles, and inconsistent successful
        execution records are rejected rather than guessed.

        Does not modify history, load checkpoints, or establish a replay input.
        """
        import json
        from types import SimpleNamespace
        from .pglSerialize import pglSerialize

        if not isinstance(actionable, pglActionable):
            raise TypeError("Pipeline extraction requires a pglActionable object.")

        history = list(actionable.actionHistory)

        if any(entry.actionStatus == pglActionStatus.RUNNING for entry in history):
            raise ValueError("Cannot extract a pipeline while its history contains running actions.")

        name = "extracted-pipeline" if name is None else name
        cls._validatePipelineName(name)

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
            entry = history[index]
            code = entry.actionCode
            path = f"history[{index}]({entry.actionName})"

            if not isinstance(code, dict) or not code.get("module") or not code.get("qualname"):
                raise ValueError(f"{path}: this legacy entry has no recorded action class identity. It cannot be reconstructed safely from its display name alone.")

            return cls._resolveActionClass(code["module"], code["qualname"])

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
                raise ValueError(f"{path}: installed configuration schema differs; missing={missing}, unknown={unknown}.")

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
            recorded = pglPipeline.fromJSON(snapshot["recipeJSON"], strict=True)

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

        pipeline = cls(name=name)
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

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
    
########################
# action history
########################
class pglActionHistory(pglTraitSettings):
    actionName = Unicode(help="Name of action")
    actionVersion = Unicode(help="Version of action")
    actionStatus = Instance(pglActionStatus, help="Status of action")
    actionError = Instance(Exception, allow_none=True, default_value=None, help="Error raised while running this action")
    actionSettings = Dict(default_value={}, help="Snapshot of the action's configuration traits")
    runDateTime = Unicode(help="Date and time when the action was run")
    startTime = Float(help="Start time of the action in seconds")
    endTime = Float(help="End time of the action in seconds")
    runDuration = Float(help="Time taken to run the action in seconds")

    def __init__(self, action=None):
        """Create a history entry, or an empty instance for deserialization."""
        super().__init__()
        self.actionStatus = pglActionStatus.INITIALIZED

        # The serializer restores saved traits after empty construction.
        if action is None:
            return

        self.runDateTime = datetime.now().astimezone().isoformat()
        self.actionName = action.name
        self.actionStatus = action.status
        self.actionVersion = action.version

        # Capture subclass configuration, not framework state or history.
        frameworkTraits = set(pglAction.class_traits())
        configurationNames = [name for name, trait in action.traits().items() if name not in frameworkTraits and trait.metadata.get("serialize", True)]
        self.actionSettings = deepcopy({name: getattr(action, name) for name in sorted(configurationNames)})

        self.startTime = pglTimestamp.getSecs()
        self.endTime = self.startTime
        self.runDuration = 0.0

    def update(self, action):
        """Update execution status and timing without changing the configuration snapshot."""
        self.actionStatus = action.status
        self.actionError = action.error
        self.endTime = pglTimestamp.getSecs()
        self.runDuration = self.endTime - self.startTime

    def __repr__(self):
        """Return a short status description."""
        return f"{self.actionName} status: {self.actionStatus.name}"

    def toString(self):
        """Return a readable history description."""
        if self.actionStatus.value > pglActionStatus.CONFIGURED.value:
            runDateTime = datetime.fromisoformat(self.runDateTime).strftime("%H:%M:%S %d/%m/%Y")
            return f"{self.actionName} ran at {runDateTime} with status: {self.actionStatus.name} duration: {pglTimestamp.formatDuration(self.runDuration)}"
        return f"{self.actionName} status: {self.actionStatus.name}"

    def print(self):
        """Print action history."""
        print(self.toString())
           

########################
# class pglActionable
########################
class pglActionable(pglTraitSettings):
    '''
    An actionable is any data structure that accepts an action history
    '''
    actionHistory = List(Instance(pglActionHistory),help="History of actions that have been run")
    
    def history(self):
        '''
        display history
        '''
        # print status
        self.print()
        from pgl import pglBase
        pglMessages.printHeader("action history")
        
        if not self.actionHistory:
            pglMessages.message(f"{type(self).__name__} has no history")
            return
    
        for iAction, action in enumerate(self.actionHistory):
            print(f"{iAction}: {action.toString()}")
            
    def print(self, verbose=False):
        '''
        print should print the status of the class (which appears at the top of the history list)
        '''
        super().print() 
    
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
        super().__init__()

        # initialize basic action
        self.name = self.__class__.__name__
        self.status = pglActionStatus.INITIALIZED
        self.version = "0.0"

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
        
        # check to make sure that the action is configured
        if not self.isConfigured():
            pglMessages.warning(f"Action {self.name} is not yet configured.", level=1)
            return None
            
        # set status
        self.status = pglActionStatus.RUNNING

        # initalize a history
        self.actionHistory.append(pglActionHistory(self))
        
        # run the action
        try:
            # try to run the action
            result = self._run(*args, **kwargs)            
            self.status = pglActionStatus.SUCCESS
            
            # update the action history
            self.actionHistory[-1].update(self)
            
            # if the action return something that is "actionable", then it means
            # that we can save the action history
            if isinstance(result, pglActionable):
                result.actionHistory.append(self.actionHistory[-1])            
            else:
                pglMessages.warning(f"Action {self.name} returned type {type(result).__name__} which is not actionable, no action history will be saved", level=1)
            
            return result
        
        except Exception as e:
            # report error and set status to failed
            self.error = e
            self.status = pglActionStatus.FAILED            
            
            # update the action history
            self.actionHistory[-1].update(self)
            
            # print warning message
            pglMessages.warning(f"Error running action {self.name}: {e}")
            return None
    
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
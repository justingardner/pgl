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
    actionSettings = Instance(pglTraitSettings, allow_none=True, default_value=None, help="Settings for this action")
    runDateTime = Unicode(help="Date and time when the action was run")
    startTime = Float(help="Start time of the action in seconds")
    endTime = Float(help="End time of the action in seconds")
    runDuration = Float(help="Time taken to run the action in seconds")
    
    def __init__(self, action: "pglAction"):
        '''
        Init the action history
        '''
        self.runDateTime = datetime.now().astimezone().isoformat()
        self.actionName = action.name
        self.actionStatus = action.status
        self.actionVersion = action.version

        # get time
        self.startTime = pglTimestamp.getSecs()
        self.endTime = self.startTime
        self.runDuration = self.endTime-self.startTime

    
    def update(self, action: "pglAction"):
        '''
        update the action history
        '''
        # set status and any error
        self.actionStatus = action.status
        if action.error is not None:
            self.actionError = action.error

        # get time
        self.endTime = pglTimestamp.getSecs()
        self.runDuration = self.endTime-self.startTime
        
    def __repr__(self):
        '''
        simple string representation
        '''
        return f"{self.actionName} status: {self.actionStatus.name}"
    
    def toString(self):
        '''
        string representation
        '''
        if self.actionStatus.value > pglActionStatus.CONFIGURED.value:
            return f"{self.actionName} ran at {datetime.fromisoformat(self.runDateTime).strftime("%H:%M:%S %d/%m/%Y")} with status: {self.actionStatus.name} duration: {pglTimestamp.formatDuration(self.runDuration)}"
        else:
            return f"{self.actionName} status: {self.actionStatus.name}"

    def print(self):
        '''
        print action history
        '''
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
    settings = Instance(pglTraitSettings, allow_none=True, default_value=None, help='Settings for this action')
    version = Unicode("", help='Verion of pglAction')
    
    # init action
    #-----------------
    def __init__(self):
        '''
        initialize the action
        '''
        self.name = self.__class__.__name__
        self.status = pglActionStatus.INITIALIZED
        self.version = "0.0"

    def configure(self) -> None:
        '''
        Configure the action, this should be subclassed, and the subclass should call this super function
        when done configured to establish that the action was actually properly configured
        '''
        
        # set status
        self.status = pglActionStatus.CONFIGURED
        
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
        '''
        print the action
        '''
        print(f"Action: {self.name} status: {self.status.name}")
        if verbose:
            self.settings.print()
   

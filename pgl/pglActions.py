################################################################
#   filename: pglActions.py
#    purpose: actions
#         by: JLG
#       date: Sept 12, 2026
################################################################

from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from pathlib import Path
from .pglExperiment import pglEventSegment
from .pglSettings import pglSettings
from .pglDialog import pglDialogs
from typing import Annotated
from .pglChoose import pglChoose
from .pglPipeline import pglAction
from .pglSession import pglSession, pglMNE
import numpy as np
from numbers import Integral
import matplotlib.pyplot as plt
from fsspec import AbstractFileSystem
from traitlets import HasTraits, Enum, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
import pandas as pd
from .pglExperiment import pglEventTrial, pglEventSegment
import matplotlib
from tqdm.auto import tqdm

#################################
# Collection of predefined actions
#################################
class pglActions():
    
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # loadSession
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class loadSession(pglAction):
        # settings
        selectedPaths = List(Unicode(), help="Paths of runs selected for loading")
        filesystemPrefix = Unicode("", help="Filesystem prefix like ssh:// which can be set if the files are not local")
        
        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, experimentName: str=None, subjectID: str=None, sessionName: str=None, runName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None) -> None:
            """Choose runs and exclude directories missing required experiment files."""
            import posixpath
            from .pglPipeline import pglActionStatus

            # Clear previous configuration in case this action instance is reused.
            self.status = pglActionStatus.INITIALIZED
            self.selectedPaths = []

            filesystem, runList, filesystemPrefix = pglChoose.getSessionRuns(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath)
            if filesystem is None:
                return

            requiredFiles = ("experimentSettings.json", "settings.json", "data.json", "state.json", "pgl.json")
            validPaths = []

            for runPath in runList:
                try:
                    if not filesystem.isdir(runPath):
                        pglMessages.warning(f"Skipping invalid run {runPath!r}: directory does not exist or is not a directory.", level=1)
                        continue

                    missingFiles = [filename for filename in requiredFiles if not filesystem.isfile(posixpath.join(runPath, filename))]
                    if missingFiles:
                        pglMessages.warning(f"Skipping incomplete run {runPath!r}: missing required files: {', '.join(missingFiles)}.", level=1)
                        continue

                except OSError as error:
                    pglMessages.warning(f"Skipping inaccessible run {runPath!r}: {error}", level=1)
                    continue

                validPaths.append(runPath)

            if not validPaths:
                pglMessages.warning("No valid run directories remain; session was not loaded.", level=1)
                return

            self.selectedPaths = validPaths
            self.filesystemPrefix = filesystemPrefix or ""
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self):
            '''
            Run the action to load the session
            
            Returns:
                pglSession: The loaded session (or None if )
            '''
            # import session
            from .pglSession import pglSession
            
            # just create the session variable
            session = pglSession(filesystemPrefix=self.filesystemPrefix, runList = self.selectedPaths)
            
            # and return
            return session
        
     
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    # action stub
    #+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+
    class stub(pglAction):

        # parameters
                
        ################################
        # configure
        ################################
        def configure(self, **kwargs) -> None:

            # set parameters
            self.configureTraits(**kwargs)
                
            # we are now configured, so call super to set status
            super().configure()
            
        ################################
        # run
        ################################
        def _run(self, session: pglSession) -> pglSession:
            '''
            
            
            Returns:
                pglSession:  session
            '''
            # import mne
            import mne

            # check for mne session
            if session.mne is None or session.mne.raw is None:
                self.setError("session does not have raw mne loaded")
                return None

            # check for mne epochs
            if session.mne.epochs is None:
                self.setError("session does not have epochs created: run mneCreateEpochs")
                return None

            # and return
            return session

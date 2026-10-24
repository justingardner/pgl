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
        sortByAcquisitionTime = Bool(True, help="Load runs in acquisition-time order, oldest first. If False, preserve the chooser's order.")
        
        ################################
        # configure
        ################################
        def configure(self, fullDataPath: str=None, settings: pglSettings=None, settingsName: str=None, experimentName: str=None, subjectID: str=None, sessionName: str=None, runName: str=None, filesystem: AbstractFileSystem=None, filesystemPrefix: str=None, dataPath: str=None, sortByAcquisitionTime: bool=True) -> None:
            """Choose valid runs and optionally sort oldest first.

            Uses startDateTime when every selected run has a valid,
            timezone-aware timestamp. Otherwise uses startTime for the
            entire selection, assuming those values share a comparable clock.

            Equal timestamps preserve chooser order.
            sortByAcquisitionTime=False preserves chooser order without
            reading acquisition timestamps.
            """
            import json
            import math
            import posixpath
            from datetime import datetime, timezone
            from .pglPipeline import pglActionStatus

            # Clear previous configuration if this action instance is reused.
            self.status = pglActionStatus.INITIALIZED
            self.selectedPaths = []
            self.filesystemPrefix = ""

            if type(sortByAcquisitionTime) is not bool:
                raise ValueError("sortByAcquisitionTime must be True or False.")

            self.sortByAcquisitionTime = sortByAcquisitionTime

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

            if self.sortByAcquisitionTime:
                runTimes = []

                for runPath in validPaths:
                    dataFile = posixpath.join(runPath, "data.json")

                    try:
                        # Read JSON without reconstructing experiment, event, task, or display objects.
                        with filesystem.open(dataFile, "rt", encoding="utf-8") as source:
                            document = json.load(source)

                        if not isinstance(document, dict):
                            raise ValueError("data.json must contain a JSON object.")

                    except (OSError, ValueError) as error:
                        pglMessages.warning(f"Cannot read acquisition metadata for {runPath!r}: {error} Session loading was not configured.", level=1)
                        return

                    acquiredAt = None
                    value = document.get("startDateTime")

                    if isinstance(value, str) and value.strip():
                        try:
                            parsed = datetime.fromisoformat(value.strip())

                            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                                acquiredAt = parsed.astimezone(timezone.utc)

                        except (ValueError, OverflowError):
                            pass

                    runTimes.append({"path": runPath, "startDateTime": acquiredAt, "startTime": document.get("startTime")})

                if all(record["startDateTime"] is not None for record in runTimes):
                    sortField = "startDateTime"
                else:
                    # Never mix calendar timestamps and getSecs() values within the same sort.
                    sortField = "startTime"

                    for record in runTimes:
                        value = record["startTime"]

                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            pglMessages.warning(f"Cannot order run {record['path']!r}: fallback startTime is missing or invalid. Session loading was not configured. Use sortByAcquisitionTime=False to preserve chooser order.", level=1)
                            return

                    pglMessages.warning("Some runs lack usable startDateTime values; sorting all selected runs by startTime instead. This assumes their startTime values share a comparable clock.", level=0)

                # Stable sorting preserves chooser order for equal times.
                runTimes.sort(key=lambda record: record[sortField])
                validPaths = [record["path"] for record in runTimes]

            # Record the actual loading order.
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

################################################################
#   filename: pglChoose.py
#    purpose: Dialogs for choosing experiments, runs etc 
#         by: JLG
#       date: Sept 12, 2026
################################################################

#############
# Import
#############
from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings, pglSettingsManager
from .pglDialog import pglDialogs
from .pglBase import pglBase
from pathlib import Path
from .pglSession import pglRun
from fsspec import AbstractFileSystem
from traitlets import Unicode, List, Instance
import re
from .pglParameter import pglParameter
from traitlets import HasTraits, Any, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from .pglSettings import pglItem
import os
from datetime import datetime, timezone
from fsspec.implementations.local import LocalFileSystem


class pglChooseLevel(pglTraitSettings):
    """Filesystem chooser with eager hierarchy discovery and lazy leaf loading."""

    # Shared traits
    name = Unicode("", help="Name of this filesystem entry", visible=False)
    dataPath = Unicode("", allow_none=True, help="Path within the filesystem for this entry", visible=False)
    childList = List(Instance(pglTraitSettings), settingsListKey="name", help="Child filesystem entries")
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="Filesystem used to access this entry", visible=False)
    filesystemPrefix = Unicode("", allow_none=True, help="Filesystem prefix used to recreate this path", visible=False)
    fullDataPath = Unicode("", allow_none=True, help="Full path to data", visible=False)

    # Per-class filesystem schema
    entryType = "directory"
    namePattern = None
    requiredFiles = ()
    childClass = None

    def __init__(self, name="", dataPath="", filesystem=None, filesystemPrefix=None, entries=None):
        super().__init__()

        # Validate only the root; children reuse their parent's filesystem.
        if filesystem is None:
            filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix)

        self.name = name
        self.dataPath = str(dataPath)
        self.fullDataPath = str(dataPath)
        self.filesystem = filesystem
        self.filesystemPrefix = filesystemPrefix or ""
        self.childList = self._getChildren(entries=entries) if self.childClass is not None and self.filesystem is not None else []

    # ----------------------------------------------------------------
    # Factory
    # ----------------------------------------------------------------
    @classmethod
    def create(cls, name="", dataPath="", filesystem=None, filesystemPrefix=None, entry=None):
        """Create a valid node, omitting branches with no valid leaves."""

        if filesystem is None:
            filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix)

        if filesystem is None:
            return None

        entries = None

        # List directories only when needed for discovery or validation.
        if cls.childClass is not None or cls.requiredFiles:
            try:
                entries = filesystem.ls(dataPath, detail=True)
            except (FileNotFoundError, OSError):
                return None

        if not cls._isValid(name=name, dataPath=dataPath, filesystem=filesystem, entry=entry, entries=entries):
            return None

        instance = cls(name=name, dataPath=dataPath, filesystem=filesystem, filesystemPrefix=filesystemPrefix, entries=entries)

        if cls.childClass is not None and not instance.childList:
            return None

        return instance

    # ----------------------------------------------------------------
    # Generic validation
    # ----------------------------------------------------------------
    @classmethod
    def _isValid(cls, name=None, dataPath=None, filesystem=None, entry=None, entries=None):
        """Validate entry type, optional name pattern, and required files."""

        if entry is not None and entry.get("type") != cls.entryType:
            return False

        if cls.namePattern is not None and (name is None or re.match(cls.namePattern, name) is None):
            return False

        if cls.requiredFiles:
            if entries is None:
                return False

            fileNames = {item["name"].rstrip("/").rsplit("/", 1)[-1] for item in entries if item.get("type") == "file"}

            if not set(cls.requiredFiles).issubset(fileNames):
                return False

        return True

    # ----------------------------------------------------------------
    # Creation-time sorting
    # ----------------------------------------------------------------
    @staticmethod
    def _asTimestamp(value):
        """Normalize Unix seconds, datetime objects, or ISO timestamps."""

        if value is None:
            return None

        try:
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))

            if isinstance(value, datetime):
                # Treat timestamps without timezone information as UTC.
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.timestamp()

            return float(value)
        except (TypeError, ValueError, OverflowError, OSError):
            return None

    def _getCreationTime(self, entry):
        """Return creation time in Unix seconds, or None if unavailable."""

        if isinstance(self.filesystem, LocalFileSystem):
            try:
                statInfo = os.stat(self.filesystem._strip_protocol(entry["name"]))
            except OSError:
                return None

            birthTime = getattr(statInfo, "st_birthtime", None)

            # Older Python versions expose Windows creation time as st_ctime.
            if birthTime is None and os.name == "nt":
                birthTime = statInfo.st_ctime

            # Do not use local fsspec "created": it may actually be Unix ctime.
            return self._asTimestamp(birthTime)

        # Remote metadata names depend on the filesystem backend.
        for key in ("birthtime", "st_birthtime", "created", "creation_time", "creationTime", "timeCreated", "CreationTime"):
            creationTime = self._asTimestamp(entry.get(key))
            if creationTime is not None:
                return creationTime

        return None

    def _childSortKey(self, entry):
        """Sort leaves oldest first, branches newest first, unknown times last."""

        creationTime = self._getCreationTime(entry)
        entryName = entry["name"].rstrip("/").rsplit("/", 1)[-1]
        childrenAreLeaves = self.childClass.childClass is None
        sortTime = creationTime if creationTime is not None else 0
        sortTime = sortTime if childrenAreLeaves else -sortTime

        return (creationTime is None, sortTime, entryName.casefold(), entryName)

    # ----------------------------------------------------------------
    # Find child entries
    # ----------------------------------------------------------------
    def _getChildren(self, entries=None):
        """Create valid direct children in creation-time order."""

        if entries is None:
            try:
                entries = self.filesystem.ls(self.dataPath, detail=True)
            except (FileNotFoundError, OSError):
                return []

        children = []

        # Filter by the child class's declared type before sorting.
        candidateEntries = [entry for entry in entries if entry.get("type") == self.childClass.entryType]

        for entry in sorted(candidateEntries, key=self._childSortKey):
            entryPath = entry["name"]
            entryName = entryPath.rstrip("/").rsplit("/", 1)[-1]
            child = self.childClass.create(name=entryName, dataPath=entryPath, filesystem=self.filesystem, filesystemPrefix=self.filesystemPrefix, entry=entry)

            if child is not None:
                children.append(child)

        return children


################################################################################
# Standard experiment chooser hierarchy
#
# Expected structure:
#
#     dataPath/
#         experiment/
#             s00001/
#                 session/
#                     run/
#
# The dialog behavior remains exactly as before because each level still
# declares childList metadata that pglTraitsDialog already understands.
################################################################################

class pglChooseRun(pglChooseLevel):
    """
    Leaf representing one experiment run directory.

    pglRun construction remains lazy: merely discovering and displaying
    runs does not open their contents.
    """

    entryType = "directory"
    childClass = None

    _tasks = Unicode("",allow_none=True,help="Stimulus type used for this run",enabled=False,)

    _run = Instance(pglRun,allow_none=True,default_value=None,serialize=False,help="Class representing run data",visible=False,)

    @property
    def tasks(self):
        """Lazy-load task names only when requested."""
        if not self._tasks:
            self._tasks = self.run.getTaskNames()
        return self._tasks

    @property
    def run(self):
        """Lazy-load the expensive pglRun object only when needed."""
        if self._run is None:
            self._run = pglRun(
                fullDataPath=self.dataPath,
                filesystem=self.filesystem,
                filesystemPrefix=self.filesystemPrefix,
            )
        return self._run

    @run.setter
    def run(self, value):
        self._run = value

    def display(self, fig=None):
        """Called by the existing traits-dialog display button."""
        self.run.display(fig=fig)


class pglChooseSession(pglChooseLevel):
    """
    Directory containing run directories.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Select run(s)",
        multiSelect=True,
        maxRowsVisible=6,
        hasPlotButton=True,
        buttonFunction="display",
        help="Runs in session directory",
    )

    entryType = "directory"
    childClass = pglChooseRun


class pglChooseSubject(pglChooseLevel):
    """
    Subject directory, required to have the form s#####.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Choose session",
        help="Sessions in subject directory",
    )

    entryType = "directory"
    namePattern = r"^s\d+$"
    childClass = pglChooseSession


class pglChooseExperiment(pglChooseLevel):
    """
    Experiment directory containing subject directories.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Choose subject",
        help="Subjects in experiment directory",
    )

    entryType = "directory"
    childClass = pglChooseSubject


class pglChooseData(pglChooseLevel):
    """
    Top-level data directory containing experiment directories.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Choose experiment",
        help="Experiments in data path",
    )

    entryType = "directory"
    childClass = pglChooseExperiment
    
def makeRecordingChooser(
    formatName,
    entryType,
    namePattern,
    requiredFiles=(),
):
    """
    Build a recording chooser hierarchy for one format.

    Returns the top-level data chooser class.

    formatName should be suitable for use in a Python class name,
    e.g. "Fieldline" or "NetStation".
    """

    classPrefix = f"pglChoose{formatName}"

    # Recording leaf: no children, even when it represents a directory.
    recordingClass = type(
        classPrefix,
        (pglChooseLevel,),
        {
            "__module__": __name__,
            "entryType": entryType,
            "namePattern": namePattern,
            "requiredFiles": tuple(requiredFiles),
            "childClass": None,
        },
    )

    sessionClass = type(
        f"{classPrefix}Session",
        (pglChooseLevel,),
        {
            "__module__": __name__,
            "entryType": "directory",
            "childClass": recordingClass,
            "childList": List(
                Instance(pglTraitSettings),
                settingsListKey="name",
                traitDisplayName=f"Select {formatName} run(s)",
                multiSelect=True,
                maxRowsVisible=6,
                help=f"{formatName} recordings in session directory",
            ),
        },
    )

    subjectClass = type(
        f"{classPrefix}Subject",
        (pglChooseLevel,),
        {
            "__module__": __name__,
            "entryType": "directory",
            "namePattern": r"^s\d+$",
            "childClass": sessionClass,
            "childList": List(
                Instance(pglTraitSettings),
                settingsListKey="name",
                traitDisplayName="Choose session",
                help=f"Sessions in {formatName} subject directory",
            ),
        },
    )

    experimentClass = type(
        f"{classPrefix}Experiment",
        (pglChooseLevel,),
        {
            "__module__": __name__,
            "entryType": "directory",
            "childClass": subjectClass,
            "childList": List(
                Instance(pglTraitSettings),
                settingsListKey="name",
                traitDisplayName="Choose subject",
                help=f"Subjects in {formatName} experiment directory",
            ),
        },
    )

    dataClass = type(
        f"{classPrefix}Data",
        (pglChooseLevel,),
        {
            "__module__": __name__,
            "entryType": "directory",
            "childClass": experimentClass,
            "childList": List(
                Instance(pglTraitSettings),
                settingsListKey="name",
                traitDisplayName="Choose experiment",
                help=f"Experiments in {formatName} data path",
            ),
        },
    )

    return dataClass


# ------------------------------------------------------------------
# Format-specific declarations
# ------------------------------------------------------------------

pglChooseFieldlineData = makeRecordingChooser(
    formatName="Fieldline",
    entryType="file",
    namePattern=r"^.*\.[Ff][Ii][Ff]$",
)

pglChooseNetStationData = makeRecordingChooser(
    formatName="NetStation",
    entryType="directory",
    namePattern=r"^.*\.[Mm][Ff][Ff]$",
    # Optional lightweight package validation:
    # requiredFiles=("info.xml", "signal1.bin"),
)    
##############################
# pglChooseListItem
##############################
class pglChooseListItem(pglTraitSettings):
    value = Any(
        default_value=None,
        visible=False,
    )

    displayName = Unicode(
        "",
        help="Value displayed in list",
    )

##############################
# pglChooseList
##############################
class pglChooseList(pglTraitSettings):
    items = List(
        Instance(pglChooseListItem),
        settingsListKey="displayName",
        traitDisplayName="Choose",
        multiSelect=True,
        maxRowsVisible=10,
        help="Items to choose from",
    )

##############################
# pglChoose
##############################
class pglChoose():
    '''
    Class which provides ways to choose runs and experiment directories
    '''

    @classmethod
    def getSessionRuns(cls, fullDataPath=None, settings=None, settingsName=None, experimentName=None, subjectID=None, sessionName=None, runName=None, filesystem=None, filesystemPrefix=None, dataPath=None):
        # choose runs in a session, return a list of runs
        return cls.getExperimentPath(fullDataPath=fullDataPath, settings=settings, settingsName=settingsName, experimentName=experimentName, subjectID=subjectID, sessionName=sessionName, runName=runName, filesystem=filesystem, filesystemPrefix=filesystemPrefix, dataPath=dataPath, allowMultipleRuns=True)

    @classmethod
    def getExperimentPath(cls, fullDataPath=None, settings=None, settingsName=None, experimentName=None, subjectID=None, sessionName=None, runName=None, filesystem=None, filesystemPrefix=None, dataPath=None, allowMultipleRuns=False):
        '''
        get the directory of the experiment. Many ways to call this to make it easy to get the correct experiemnt dir
        
        If you want to browse the full experiments:
        
            # use default settings to find dataDir
            pglChoose.getExperimentPath()
            
            # use settings name to find dataDir:
            pglChoose.getExperimentPath(settingsName='windowed')

            # or, call directly with the setting:
            s = pglSettingsManager.getSettings(settingsName='windowed')
            pglChoose.getExperimentPath(settings=s)
            
            # or, pass in an explicit path
            pglChoose.getExperimentPath(dataPath='/path/to/experiments')

        If you know the exact path:
            pglChoose.getExperimentPath('/data/experimentDir/subjectDir/sessionDir/runDir')
            
        If you want to browse runs for a particular experiment:
            pglChoose.getExperimentPath(experimentName='experimentName')
            
        
        Returns:
            A tuple consisting of:
                (filesystem, fullDataPath, filesystemPrefix)
            where:
                filesystem: fsspec filesystem for the path
                fullDataPath: path within in filesystem
                filesystemPrefix: Any filesystem prefix (e.g. ssh://gru.stanford.edu/) this is NOT needed
                    to access the path, it is returned in case the calling function wants to save it
                    so that the same path can be accessed again
        
        '''
        from .pglBase import pglBase
        if fullDataPath:
            # validate and return
            filesystem, fullDataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=fullDataPath, filesystemPrefix=filesystemPrefix)
            return (filesystem, fullDataPath, filesystemPrefix)
            
        # if not fullDatadir passed in, construct it from arguments
        else:
            if not dataPath: 
                if not settings:
                    # get the default settings
                    settings = pglSettingsManager.getSettings(settingsName=settingsName)
                    if settings is None:
                        pglMessages.warning(f"Could not find settings {settingsName}")
                        return (None, None, None)
                if settings:
                    # set dataPath to where settings tells us it is
                    dataPath= settings.dataPath

            # expand user
            fullDataPath = Path(dataPath).expanduser()
            
            # now that we have the start of a path, validate the filesystem
            filesystem, fullDataPath, filesystemPrefix = pglBase.validateFilesystem(filesystem=filesystem, dataPath=fullDataPath, filesystemPrefix=filesystemPrefix)
            if filesystem is None:
                pglMessages.warning("Could not find dataPath: {fullDataPath}")
                return (None, None, None)
            
            # add on experiment name
            if experimentName:
                fullDataPath = Path(fullDataPath) / experimentName
                # check that experimentName exists
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Experiment directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on subject experiment names
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='experimentNames', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                if filesystem is None: 
                    return (None, None, None)
                else: 
                    return (filesystem, fullDataPath, filesystemPrefix)
           
            # add a subjectID
            if subjectID:
                fullDataPath = fullDataPath / subjectID
                # check the subjectID 
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Subject directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on subject IDs
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='subjectIDs', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                if filesystem is None: 
                    return (None, None, None)
                else: 
                    return (filesystem, fullDataPath, filesystemPrefix)
                
            # add a sessionName
            if sessionName:
                fullDataPath = fullDataPath / sessionName
                # check the sessionName 
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Session directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
            else:
                # choose based on session names
                (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='sessionNames', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                if filesystem is None: 
                    return (None, None, None)
                else: 
                    return (filesystem, fullDataPath, filesystemPrefix)
                
            # add a runName
            if runName:
                fullDataPath = fullDataPath / runName
                # check the runName
                if not filesystem.exists(fullDataPath):
                    pglMessages.warning(f"Run directory {fullDataPath} does not exist")
                    return (None, fullDataPath, filesystemPrefix)
                else:
                    # choose based on run names
                    (filesystem, fullDataPath) = cls._chooseDialog(fullDataPath=fullDataPath, chooseLevel='runNames', filesystem=filesystem, allowMultipleRuns=allowMultipleRuns)
                    if filesystem is None: 
                        return (None, None, None)
                    else: 
                        return (filesystem, fullDataPath, filesystemPrefix)
              
        return (filesystem, fullDataPath, filesystemPrefix)

    @classmethod
    def getFieldline(
        cls,
        fullDataPath=None,
        settings=None,
        settingsName=None,
        filesystem=None,
        filesystemPrefix=None,
        dataPath=None,
    ):
        """Choose FIF files from an experiment/subject/session hierarchy."""
        return cls._getRecordings(
            chooserClass=pglChooseFieldlineData,
            formatName="Fieldline",
            fullDataPath=fullDataPath,
            settings=settings,
            settingsName=settingsName,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
            dataPath=dataPath,
        )

    @classmethod
    def getNetStation(
        cls,
        fullDataPath=None,
        settings=None,
        settingsName=None,
        filesystem=None,
        filesystemPrefix=None,
        dataPath=None,
    ):
        """Choose MFF packages from an experiment/subject/session hierarchy."""
        return cls._getRecordings(
            chooserClass=pglChooseNetStationData,
            formatName="NetStation",
            fullDataPath=fullDataPath,
            settings=settings,
            settingsName=settingsName,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
            dataPath=dataPath,
        )

    @classmethod
    def _getRecordings(
        cls,
        chooserClass,
        formatName,
        fullDataPath=None,
        settings=None,
        settingsName=None,
        filesystem=None,
        filesystemPrefix=None,
        dataPath=None,
    ):
        """
        Display a recording chooser using the supplied hierarchy class.

        Expected structure:
            dataPath/experiment/s#####/session/recording

        fullDataPath is an alternate name for the hierarchy root,
        not a direct path to an individual recording.

        Returns
        -------
        tuple
            (filesystem, recordingPaths, filesystemPrefix)

            recordingPaths is:
                A list of selected recording paths on success.
                [] if no recordings are found or selected.
                None if canceled or filesystem setup fails.
        """

        # Explicit fullDataPath takes precedence over dataPath.
        if fullDataPath is not None:
            dataPath = fullDataPath

        # Fall back to settings when no explicit root was supplied.
        if not dataPath:
            if settings is None:
                settings = pglSettingsManager.getSettings(
                    settingsName=settingsName
                )

            if settings is None:
                pglMessages.warning(
                    f"Could not find settings {settingsName}"
                )
                return (None, None, None)

            dataPath = settings.dataPath

        # Validate once; all descendant nodes reuse this filesystem.
        filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
        )

        if filesystem is None:
            pglMessages.warning(
                f"Could not access {formatName} data path: {dataPath}"
            )
            return (None, None, None)

        # Discover recordings without loading their signal data.
        chooser = chooserClass(
            dataPath=dataPath,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
        )

        if not chooser.childList:
            pglMessages.message(
                f"No {formatName} recordings found below {dataPath}"
            )
            return (filesystem, [], filesystemPrefix)

        chooser = pglDialogs.traitsDialog(chooser)

        if chooser is None:
            pglMessages.message(
                f"No {formatName} recordings selected"
            )
            return (None, None, filesystemPrefix)

        recordingPaths = cls.walkInstances(chooser)

        if not recordingPaths:
            pglMessages.message(
                f"No {formatName} recordings selected"
            )
            return (filesystem, [], filesystemPrefix)

        return (filesystem, recordingPaths, filesystemPrefix)

    @classmethod
    def chooseList(cls,values,key=None,traitDisplayName="Choose",maxRowsVisible=10,help=None):
        """
        Display a dialog for choosing one or more items from a list.

        Parameters
        ----------
        values : list
            List of values/items to choose from.

        key : str or callable, optional
            Determines the value displayed in the chooser.

            If None:
                The item itself is displayed.

            If str:
                The named attribute/key is displayed. This works for both
                objects and dictionaries.

            If callable:
                The callable is passed each item and its return value is
                displayed.

        traitDisplayName : str
            Display name for the chooser.

        maxRowsVisible : int
            Maximum number of rows shown by the dialog.

        help : str, optional
            Help text for the chooser.

        Returns
        -------
        list or None
            List containing the originally supplied selected values.

            Returns None if the user cancels or nothing is selected.

        Notes
        -----
        The dialog always permits multiple selection and always returns a
        list. Thus a single selection is returned as a one-element list.

        The dynamically-created pglTraitSettings objects are only used as
        dialog models. The original objects supplied in `values` are
        returned.
        """

        if values is None:
            return None

        values = list(values)

        if not values:
            pglMessages.message("No items to choose from")
            return None

        # --------------------------------------------------------------
        # Determine how an item should be displayed.
        # --------------------------------------------------------------
        def getDisplayValue(item):
            if key is None:
                return item

            if callable(key):
                return key(item)

            if isinstance(item, dict):
                return item[key]

            return getattr(item, key)

        # --------------------------------------------------------------
        # Create dialog items.
        # --------------------------------------------------------------
        items = [
            pglChooseListItem(
                value=value,
                displayName=str(getDisplayValue(value)),
            )
            for value in values
        ]

        chooser = pglChooseList(items=items)

        # --------------------------------------------------------------
        # Show dialog.
        # --------------------------------------------------------------
        chooser = pglDialogs.traitsDialog(chooser)

        if chooser is None:
            return []

        # --------------------------------------------------------------
        # Return the original values corresponding to selected items.
        # --------------------------------------------------------------
        selectedValues = [
            item.value
            for item in chooser.items
            if item.isSelected
        ]

        if not selectedValues:
            return []

        return selectedValues

    @ classmethod
    def _chooseDialog(cls, fullDataPath, filesystem=None, chooseLevel=None, allowMultipleRuns=False):
        '''
        Function that will put up a dialog to choose an experiment for loading
        '''
        # put up dialog
        if chooseLevel == 'experimentNames':
            s = pglChooseData(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        elif chooseLevel == 'subjectIDs':
            s = pglChooseExperiment(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        elif chooseLevel == 'sessionNames':
            s = pglChooseSubject(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        elif chooseLevel == 'runNames':
            s = pglChooseSession(dataPath=fullDataPath, filesystem=filesystem)
            s = pglDialogs.traitsDialog(s)
            if s is None:
                pglMessages.message("No runs selected")
                return (None, None)
        else:
            pglMessages.warning(f"Unkown choose level: {chooseLevel}")
            return (None, None)
        
        # walk structure to get runs that are selected
        runNames = cls.walkInstances(s)
        if not runNames:
            pglMessages.message("No runs selected")
            return (None, None)
        elif len(runNames)>1:
            if not allowMultipleRuns:
                pglMessages.message(f"Multiple runs selecting, using {runNames[0]}")
            else:
                return (filesystem, runNames)

        if allowMultipleRuns is False:
            return (filesystem, runNames[0])
        else:
            return (filesystem, runNames)

    # walk the structure to get to the leaves (which have runs)        
    @classmethod
    def walkInstances(cls, node, depth=0):
        selectedPaths = []
        childClass = getattr(type(node), "childClass", None)
        if childClass is None:
            # Leaf instance — get the dataPath if it was selected
            if node.isSelected: 
                selectedPaths.append(node.dataPath)
            return selectedPaths

        # childList holds the child instances
        for child in node.childList:
            selectedPaths.extend(cls.walkInstances(child, depth + 1))

        return(selectedPaths)

    # ----------------------------------------------------------------
    # chooseItems
    # ----------------------------------------------------------------
    @classmethod
    def chooseItems(cls, itemList):
        '''
        choose from a list of items
        
        Args:
            itemList (list of str) list of itmes to choose from
            
        Returns:
            List of chosen items
        '''
        # validate
        if not isinstance(itemList, list) or not all(isinstance(item, str) for item in itemList):
            pglMessages.warning("itemList must be a list of strings")
            return []

        # put up dialong
        l = pglList(itemList=[pglItem(name=item) for item in itemList])
        l = pglDialogs.traitsDialog(l)
        
        # extract selected
        if l:
            return [item.name for item in l.itemList if item.isSelected]
        else:
            return []
        
    # ----------------------------------------------------------------
    # chooseItems
    # ----------------------------------------------------------------
    @classmethod
    def chooseItem(cls, itemList):
        '''
        choose from a list of items
        
        Args:
            itemList (list of str) list of itmes to choose from
            
        Returns:
            List of chosen items
        '''
        # validate
        if not isinstance(itemList, list) or not all(isinstance(item, str) for item in itemList):
            pglMessages.warning("itemList must be a list of strings")
            return []

        # put up dialong
        l = pglListSelectOne(itemList=[pglItem(name=item) for item in itemList])
        l = pglDialogs.traitsDialog(l)
        
        # extract selected
        if l:
            return l.itemList[0].name
        else:
            return None
        
##################################
# pglTrialsByParameter
##################################
class pglTrialsByParameter(pglTraitSettings):
    parameterName = Unicode(help="Name of parameter that was used to sort trials by")
    parameterValues = List(help="List of all values that the parameter can take")
    parameter = Instance(pglParameter, help="The pglParameter instance of the parameter")
    nTrialsTotal = Int(help="total number of trials")
    volumes = List(List(Int()),help="A list of lists of volumes, one list for each value of the parameter")
    startTimes = List(List(Float()),help="A list of lists of times, one list for each value of the parameter")
    trialNums = List(List(Int()),help="A list of lists of trial volumes, one list for each value of the parameter")
    nTrials = List(Int(),help="A list of number of trials, one list for each value of the parameter")
           

class pglList(pglTraitSettings):
    itemList = List(Instance(pglItem), settingsListKey="name", style="dropdown", multiSelect=True, traitDisplayName="Choose items", help="List of items")

class pglListSelectOne(pglTraitSettings):
    itemList = List(Instance(pglItem), settingsListKey="name", traitDisplayName="Choose item", help="List of items")

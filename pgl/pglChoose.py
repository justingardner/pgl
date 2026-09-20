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

##################################################################
# Generic chooser hierarchy
#
# This is eager about discovering the filesystem hierarchy, so the
# complete tree is present when pglTraitsDialog opens.  It is lazy
# only about opening/reading expensive leaf data objects, e.g. pglRun.
#
# Each chooser class describes one filesystem level:
#
#     entryType      What this level represents: "directory" or "file"
#     namePattern    Optional regex applied to the basename
#     requiredFiles  Optional direct filenames required in a directory
#     childClass     Class representing entries below this level
#
# pglTraitsDialog needs no changes: each class retains its own
# childList trait metadata, including display name and multiSelect.
##################################################################
class pglChooseLevel(pglTraitSettings):
    """
    Base class for one level of a filesystem chooser hierarchy.

    Subclasses normally only need to declare:

        entryType = "directory"     # or "file"
        namePattern = r"...",       # optional
        requiredFiles = (...)       # optional, directories only
        childClass = SomeClass      # None for leaves

    and, for dialog display, redeclare childList with the desired
    trait metadata.
    """

    # ----------------------------------------------------------------
    # Traits shared by every filesystem node
    # ----------------------------------------------------------------
    name = Unicode(
        "",
        help="Name of this filesystem entry",
        visible=False,
    )

    dataPath = Unicode(
        "",
        allow_none=True,
        help="Path within the filesystem for this entry",
        visible=False,
    )

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        help="Child filesystem entries",
    )

    filesystem = Instance(
        AbstractFileSystem,
        allow_none=True,
        serialize=False,
        help="Filesystem used to access this entry",
        visible=False,
    )

    filesystemPrefix = Unicode(
        "",
        allow_none=True,
        help="Filesystem prefix used to recreate this path",
        visible=False,
    )

    # Retained in case other code currently refers to fullDataPath.
    # dataPath is the path actually used by this chooser hierarchy.
    fullDataPath = Unicode(
        "",
        allow_none=True,
        help="Full path to data",
        visible=False,
    )

    # ----------------------------------------------------------------
    # Per-class filesystem schema
    #
    # These are normal class attributes rather than traits. They define
    # what the class represents; they are not user-editable settings.
    # ----------------------------------------------------------------
    entryType = "directory"
    namePattern = None
    requiredFiles = ()
    childClass = None

    def __init__(
            self,
            name="",
            dataPath="",
            filesystem=None,
            filesystemPrefix=None,
            entries=None,
        ):
            super().__init__()

            # The root node may be constructed directly, without a filesystem.
            # Child nodes are always passed the already-established filesystem
            # from their parent, and must NOT re-validate or re-infer it.
            if filesystem is None:
                filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(
                    filesystem=filesystem,
                    dataPath=dataPath,
                    filesystemPrefix=filesystemPrefix,
                )

            self.name = name
            self.dataPath = str(dataPath)
            self.fullDataPath = str(dataPath)
            self.filesystem = filesystem
            self.filesystemPrefix = filesystemPrefix or ""

            if self.childClass is not None and self.filesystem is not None:
                self.childList = self._getChildren(entries=entries)
            else:
                self.childList = []
    # ----------------------------------------------------------------
    # Factory
    # ----------------------------------------------------------------
    @classmethod
    def create(
        cls,
        name="",
        dataPath="",
        filesystem=None,
        filesystemPrefix=None,
        entry=None,
    ):
        """
        Create one chooser node.

        The root path is validated when the top-level chooser is created.
        Descendant paths come directly from filesystem.ls(), so reuse the
        parent's filesystem rather than repeatedly calling
        validateFilesystem().
        """

        # This should normally only occur if someone directly calls:
        #
        #     pglChooseExperiment.create(dataPath="...")
        #
        # Rather than building it below an existing chooser node.
        if filesystem is None:
            filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(
                filesystem=filesystem,
                dataPath=dataPath,
                filesystemPrefix=filesystemPrefix,
            )

        if filesystem is None:
            return None

        entries = None

        # We need a directory listing if:
        #   1. This node has children to discover, or
        #   2. This node validates itself based on contained files.
        #
        # A file leaf needs neither.
        if cls.childClass is not None or cls.requiredFiles:
            try:
                entries = filesystem.ls(dataPath, detail=True)
            except (FileNotFoundError, OSError):
                return None

        if not cls._isValid(
            name=name,
            dataPath=dataPath,
            filesystem=filesystem,
            entry=entry,
            entries=entries,
        ):
            return None

        instance = cls(
            name=name,
            dataPath=dataPath,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
            entries=entries,
        )

        # Preserve existing chooser behavior: do not show a parent branch
        # unless it eventually contains at least one valid leaf.
        if cls.childClass is not None and not instance.childList:
            return None

        return instance

    # ----------------------------------------------------------------
    # Generic validation
    # ----------------------------------------------------------------
    @classmethod
    def _isValid(
        cls,
        name=None,
        dataPath=None,
        filesystem=None,
        entry=None,
        entries=None,
    ):
        """
        Generic validation shared by all chooser levels.

        Subclasses may override this for special validation, but should
        normally start by calling super()._isValid(...).

        Returns True if this filesystem entry is valid for cls.
        """

        # Validate whether the entry is a file or directory.
        #
        # The root object is instantiated directly rather than through
        # create(), so entry can be None there. Every child created by
        # _getChildren() receives a real fsspec detail dictionary.
        if entry is not None:
            if entry.get("type") != cls.entryType:
                return False

        # Optional regex validation of the entry basename.
        if cls.namePattern is not None:
            if name is None or re.match(cls.namePattern, name) is None:
                return False

        # Optional direct-file validation for directory entries.
        #
        # Example:
        #
        #     requiredFiles = ("events.tsv", "params.json")
        #
        if cls.requiredFiles:
            if entries is None:
                return False

            fileNames = {
                item["name"].rstrip("/").rsplit("/", 1)[-1]
                for item in entries
                if item.get("type") == "file"
            }

            if not set(cls.requiredFiles).issubset(fileNames):
                return False

        return True

    # ----------------------------------------------------------------
    # Find child entries
    # ----------------------------------------------------------------
    def _getChildren(self, entries=None):
        """
        Create one childClass object for every valid direct child of
        self.dataPath.

        There is intentionally no hardcoded directory filtering here.
        The child class declares whether it accepts directories or files
        through childClass.entryType.
        """

        if entries is None:
            try:
                entries = self.filesystem.ls(self.dataPath, detail=True)
            except (FileNotFoundError, OSError):
                return []

        children = []

        for entry in entries:
            entryPath = entry["name"]
            entryName = entryPath.rstrip("/").rsplit("/", 1)[-1]

            child = self.childClass.create(
                name=entryName,
                dataPath=entryPath,
                filesystem=self.filesystem,
                filesystemPrefix=self.filesystemPrefix,
                entry=entry,
            )

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
    
################################################################################
# Fieldline chooser hierarchy
#
# Expected structure:
#
#     dataPath/
#         experiment/
#             s00001/
#                 session/
#                     someRecording.fif
#                     anotherRecording.fif
#
# The leaf is a FIF FILE rather than a run DIRECTORY.
#
# Like pglChooseRun, pglChooseFieldline is intentionally lightweight:
# discovering the tree does not read FIF contents.  Add a lazy Fieldline/MNE
# object here later if/when you want a display button or data preview.
################################################################################


class pglChooseFieldline(pglChooseLevel):
    """
    Leaf representing one Fieldline FIF file.

    dataPath is inherited from pglChooseLevel and contains the full path
    to the FIF file.  This is what pglChoose.walkInstances() returns for
    selected files.
    """

    entryType = "file"
    namePattern = r"^.*\.[Ff][Ii][Ff]$"
    childClass = None

    # ------------------------------------------------------------------------
    # Later, if desired, add lazy Fieldline loading here. For example:
    #
    # _fieldline = Instance(pglFieldline, allow_none=True,
    #                       default_value=None, serialize=False)
    #
    # @property
    # def fieldline(self):
    #     if self._fieldline is None:
    #         self._fieldline = pglFieldline(
    #             fullDataPath=self.dataPath,
    #             filesystem=self.filesystem,
    #             filesystemPrefix=self.filesystemPrefix,
    #         )
    #     return self._fieldline
    #
    # def display(self, fig=None):
    #     self.fieldline.display(fig=fig)
    #
    # Do not add hasPlotButton=True below until this class has a real
    # display() method.
    # ------------------------------------------------------------------------


class pglChooseFieldlineSession(pglChooseLevel):
    """
    Directory containing Fieldline FIF files.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Select Fieldline run(s)",
        multiSelect=True,
        maxRowsVisible=6,
        help="Fieldline FIF files in session directory",
    )

    entryType = "directory"
    childClass = pglChooseFieldline


class pglChooseFieldlineSubject(pglChooseLevel):
    """
    Subject directory. Only directories of the form s##### are included.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Choose session",
        help="Sessions in Fieldline subject directory",
    )

    entryType = "directory"
    namePattern = r"^s\d+$"
    childClass = pglChooseFieldlineSession


class pglChooseFieldlineExperiment(pglChooseLevel):
    """
    Experiment directory containing Fieldline subject directories.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Choose subject",
        help="Subjects in Fieldline experiment directory",
    )

    entryType = "directory"
    childClass = pglChooseFieldlineSubject


class pglChooseFieldlineData(pglChooseLevel):
    """
    Top-level Fieldline data directory containing experiment directories.
    """

    childList = List(
        Instance(pglTraitSettings),
        settingsListKey="name",
        traitDisplayName="Choose experiment",
        help="Experiments in Fieldline data path",
    )

    entryType = "directory"
    childClass = pglChooseFieldlineExperiment
    
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
    def getFieldline(cls,fullDataPath=None,settings=None,settingsName=None,filesystem=None,filesystemPrefix=None,dataPath=None):
        """
        Display a Fieldline chooser and return selected FIF file paths.

        The expected hierarchy is:

            dataPath/
                experiment/
                    s#####/
                        session/
                            recording.fif

        Parameters
        ----------
        fullDataPath : str or Path, optional
            Root of the Fieldline hierarchy. If supplied, it takes precedence
            over dataPath and settings.

        settings : pglSettings, optional
            Settings object whose dataPath will be used if neither fullDataPath
            nor dataPath is supplied.

        settingsName : str, optional
            Name passed to pglSettingsManager.getSettings() if settings must be
            loaded automatically.

        filesystem : fsspec.AbstractFileSystem, optional
            Filesystem used to access the data.

        filesystemPrefix : str, optional
            Prefix retained for later reconstruction of the filesystem, such as
            an ssh:// prefix.

        dataPath : str or Path, optional
            Root of the Fieldline hierarchy.

        Returns
        -------
        tuple
            (filesystem, fifPaths, filesystemPrefix)

            filesystem:
                The validated fsspec filesystem.

            fifPaths:
                A list of selected full FIF paths, or None if the user cancels
                or no files are selected.

            filesystemPrefix:
                Prefix returned by pglBase.validateFilesystem().

        Examples
        --------
        filesystem, fifPaths, filesystemPrefix = pglChoose.chooseFieldline(
            dataPath="/Users/justin/Desktop/digitalbrain"
        )

        if fifPaths:
            for fifPath in fifPaths:
                print(fifPath)
        """

        # `fullDataPath` is simply an alternate explicit name for the root
        # Fieldline data path.
        if fullDataPath is not None:
            dataPath = fullDataPath

        # Obtain dataPath from settings only if the caller did not provide one.
        if not dataPath:
            if settings is None:
                settings = pglSettingsManager.getSettings(settingsName=settingsName)

            if settings is None:
                pglMessages.warning(
                    f"Could not find settings {settingsName}"
                )
                return (None, None, None)

            dataPath = settings.dataPath

        # Validate the root filesystem/path once. Descendants reuse this same
        # filesystem object through pglChooseLevel; they are not revalidated.
        filesystem, dataPath, filesystemPrefix = pglBase.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
        )

        if filesystem is None:
            pglMessages.warning(
                f"Could not access Fieldline data path: {dataPath}"
            )
            return (None, None, None)

        # Construct the complete chooser tree. This performs filesystem
        # discovery only; it does not load/open the potentially large FIF files.
        chooser = pglChooseFieldlineData(
            dataPath=dataPath,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
        )

        # If no valid experiment -> subject -> session -> FIF path exists,
        # there is no useful chooser to show.
        if not chooser.childList:
            pglMessages.message(
                f"No Fieldline FIF files found below {dataPath}"
            )
            return (filesystem, [], filesystemPrefix)

        chooser = pglDialogs.traitsDialog(chooser)

        if chooser is None:
            pglMessages.message("No Fieldline files selected")
            return (None, None, filesystemPrefix)

        # pglChooseFieldline leaves inherit dataPath from pglChooseLevel, so the
        # existing generic tree walker returns full selected FIF paths.
        fifPaths = cls.walkInstances(chooser)

        if not fifPaths:
            pglMessages.message("No Fieldline files selected")
            return (filesystem, [], filesystemPrefix)

        return (filesystem, fifPaths, filesystemPrefix)

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

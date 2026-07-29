################################################################
#   filename: pglSettings.py
#    purpose: Provides settings management for pgl
#         by: JLG
#       date: Feb 6, 2026
################################################################

#############
# Import
#############
from asyncio import subprocess
from curses import wrapper
from http.client import responses
from http.client import responses
from pathlib import Path
from urllib import response
from IPython.display import display, HTML, clear_output
from fileinput import filename
from ipywidgets.widgets import widget
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from datetime import datetime   
import numpy as np
import subprocess
import platform
from .pglBase import pglDisplayMessage
from .pglParameter import pglParameter, pglParameterBlock
from .pglSerialize import pglSerialize
from .pglDialog import pglSettingsEditable, pglDialogs
import Quartz
import CoreFoundation
from AppKit import NSScreen
from .pglBase import pglBase
import re
from collections import OrderedDict
from .pglMessages import pglMessages

displayDuration = 5  # seconds
#######################################
# Mixin class for pgl to provide settings management
#######################################
class pglSettingsManager:
    """
    Mixin class for pgl to provide settings management.
    """
    def __init__(self):   
        pass
    
    @classmethod
    def settings(cls):
        """
        Edit pgl settings. Brings up widget interface to edit settings
        """
        # get settings dir
        settingsDir = cls.getSettingsDir()
        
        # load all the seettings in there
        settingsList = []
        for filename in Path(settingsDir).glob("*.json"):
            settings = pglSettings.load(filename=filename)
            settingsList.append(settings)
        settingsList = pglSettingsList(settingsList)    
        
        # bring up dialog
        oldSettingsList = settingsList
        settingsList = pglDialogs.traitsDialog(settingsList)
        if settingsList is not None:
            print("save")
        
    
    def displaySettings(self):
        """
        Edit pgl display settings. Brings up widget interface to edit display settings
        """
        # get the display infos
        original = pglDisplaySettingsList(self.getDisplaySettings())
        
        # display the settings
        modified = pglDialogs.traitsDialog(original)

        # save the settings if user clicked OK
        if modified is not None:
            # for each display in modified list
            for modifiedDisplay in modified.settingsList:
                # compare to original
                matchingOriginalDisplay = next((originalDisplay for originalDisplay in original.settingsList if originalDisplay == modifiedDisplay), None)
                if matchingOriginalDisplay is not None:
                    # and if it is not equal (field by field) then save it
                    if not matchingOriginalDisplay.equals(modifiedDisplay):
                        # if the displayName changed, we need to change the name of the directory
                        if modifiedDisplay.displayName != matchingOriginalDisplay.displayName:
                            oldPath = self.getDisplayDir(matchingOriginalDisplay)
                            newPath = self.getDisplayDir(modifiedDisplay)
                            try:
                                # rename to the new path
                                if oldPath.exists(): oldPath.rename(newPath)
                            except OSError as e:
                                # if it did not work, provide an error
                                pglMessages.warning("Could not change directory name from {oldPath.name} to {newPath.name}, keeping {oldPath.name}")
                                modifiedDisplay.displayName = matchingOriginalDisplay.displayName
                        # save the modified display
                        modifiedDisplay.save()
    
    @classmethod            
    def getDisplaySettings(cls, displayName=None):
        '''
        Get all of the displaySettings
        
        Args:
            displayName (str): If not (default=None) will return the matching setting or None if not found
        '''
        displays = []
        
        # Get CGDisplayCreateUUIDFromDisplayID
        try:
            from ColorSync import CGDisplayCreateUUIDFromDisplayID
        except ImportError:
            # fallback: some builds expose it under Quartz
            from Quartz import CGDisplayCreateUUIDFromDisplayID
            
        # first check saved displays
        displayDir = cls.getDisplayDir()
        for p in displayDir.rglob('display.json'):
            # load json settings
            displaySettings = pglDisplaySettings().load(p)
            # make sure displayName matches diectory
            if pglBase.makeValidFilename(displaySettings.displayName) != str(p.parent.name):
                displaySettings.displayName= str(p.parent.name)
            # get the calibrations
            displaySettings.getCalibrations()
            # and add to display list
            displays.append(displaySettings)
                
        maxDisplays = 16        
        (err, active, count) = Quartz.CGGetActiveDisplayList(maxDisplays, None, None)
        for display in active:
            # initialize the displaySettings
            displaySettings = pglDisplaySettings()
            
            # get all supported modes
            modes = Quartz.CGDisplayCopyAllDisplayModes(display, None)
            displayModes = []

            for mode in modes:
                # get info about the mode
                w = Quartz.CGDisplayModeGetWidth(mode)
                h = Quartz.CGDisplayModeGetHeight(mode)
                refreshRate = Quartz.CGDisplayModeGetRefreshRate(mode)

                # make into a tuple
                pixelDims = (w, h)

                # See if we already have this resolution
                existingMode = next(
                    (m for m in displayModes if m.pixelDims == pixelDims),
                    None
                )

                if existingMode is not None:
                    # Add refresh rate if it is not already there
                    if refreshRate not in existingMode.refreshRate:
                        existingMode.refreshRate.append(refreshRate)
                else:
                    displayModeSettings = pglDisplayModeSettings()
                    displayModeSettings.modeName = f"{w} x {h}"
                    displayModeSettings.pixelDims = pixelDims
                    displayModeSettings.refreshRate = [refreshRate]
                    displayModes.append(displayModeSettings)

            displaySettings.displayModes = displayModes     
                   
            # get the current mode settings
            #mode = Quartz.CGDisplayCopyDisplayMode(display)
            #displaySettings.displayWidth = Quartz.CGDisplayModeGetWidth(mode)  
            #displaySettings.displayHeight = Quartz.CGDisplayModeGetHeight(mode)
            #displaySettings.refreshRate = Quartz.CGDisplayModeGetRefreshRate(mode)

            # get UUID
            uuidRef = CGDisplayCreateUUIDFromDisplayID(display)
            displaySettings.uuid = str(CoreFoundation.CFUUIDCreateString(None, uuidRef))
            
            # get other infor from quartz
            displaySettings.vendor        = Quartz.CGDisplayVendorNumber(display)
            displaySettings.model         = Quartz.CGDisplayModelNumber(display)
            displaySettings.serialNumber  = Quartz.CGDisplaySerialNumber(display)
            displaySettings.isMain        = Quartz.CGDisplayIsMain(display)
            displaySettings.isBuiltin     = Quartz.CGDisplayIsBuiltin(display)
            
            # get display human readable name
            displaySettings.displayName = cls.getMatchingDisplayName(display)                    
            
            # get the luminance calibrations
            displaySettings.getCalibrations()
            
            # check if we already have it in our list
            matchingDisplays = [d for d in displays if displaySettings == d]
            if len(matchingDisplays) > 1:
                raise RuntimeError(
                    f"Found multiple displays with UUID {displaySettings.uuid}"
                )
            matchingDisplay = matchingDisplays[0] if matchingDisplays else None
            if matchingDisplay is not None:
                # if so, update a few fields to the settings found above
                matchingDisplay.isMain = displaySettings.isMain
                matchingDisplay.isBuiltin = displaySettings.isBuiltin
            else:
                # append to our list of all displays
                displays.append(displaySettings)
                
        if displayName is not None:
            # find the display with the matching displayName (compare using makeValidFilename to make case insenstive)
            return next(
                (d for d in displays if pglBase.makeValidFilename(d.displayName) == pglBase.makeValidFilename(displayName)),
                None
            )                
        return(displays)

    @classmethod
    def getMatchingDisplayName(cls, display):
        
        displayName = None
        # get the display name from Appkit
        for screen in NSScreen.screens():
            # map back to a CGDirectDisplayID:
            if screen.deviceDescription()["NSScreenNumber"] == display:
                # localizedName is available on macOS 10.15+
                displayName = screen.localizedName()

        if displayName is not None:
            return displayName
        
        # if Appkit fails, then get through the system profiler info we have
        displayNames = cls.getDisplayNames(displayIndex=display)

        if len(displayNames) >= 1:
            return displayNames[0]
        else:
            return "Unknown display name"
    
    @classmethod
    def getDisplayNames(cls, displayIndex=None):
        '''
        Get display names
        '''

        displayNames = ['Windowed']

        # get names from gpuInfo
        if not cls.gpuInfo:
            print(cls.gpuInfo)
            return displayNames

        for gpuData in cls.gpuInfo.values():
            displays = gpuData.get("Displays", [])
            for display in displays:
                displayType = display.get('Display Type', None)
                displayName = display.get('DisplayName', 'Unknown')
                if displayType is not None:
                    name = f"{displayName}: {displayType}"
                else:
                    name = f"{displayName}"
                if name:
                    displayNames.append(name)

        if displayIndex is not None:
            if displayIndex <= len(displayNames) and displayIndex > 0:
                # move the selected display to the top
                displayNames.insert(0, displayNames.pop(displayIndex))
            else:
                displayNames.insert(0, "Unknown Display")
        
        return displayNames    
    
    @staticmethod       
    def getPGLDir():
        """
        Get the directory where settings are stored.

        Returns:
            str: The directory path where settings are stored.
        """
        # get the pglDir
        pglDir = Path.home() / ".pgl" 
        
        # check if it exists, create if not
        if not pglDir.exists():
            try:
                pglDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglSettings:onSave)</b> Created directory: {pglDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglSettings:onSave)</b> Error creating directory {pglDir}: {e}"))
                return None

        return pglDir
    
    @classmethod
    def getSettingsDir(cls):
        """
        Get the directory where screen settings are stored.

        Returns:
            str: The directory path where settings are stored.
        """
        # get the settingsDir
        settingsDir = cls.getPGLDir() / "settings"
        
        # check if it exists, create if not
        if not settingsDir.exists():
            try:
                settingsDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglSettings:getSettingsDir)</b> Created directory: {settingsDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglSettings:getSettingsDir)</b> Error creating directory {settingsDir}: {e}"))
                return None

        return settingsDir
    
    @classmethod
    def getCalibrations(cls, calibrationDir, oldCalibrations=None):
        '''
        Get all the calibrations in the calibrationDir. These will be labeled as YYYYMMDD or YYYYMMDD_HHMMSS
        
        Args:
            calibrationDir: Directory to search for calibrations under
            oldCalibrtions: List of old calibrations - if not None, will make sure the selected one
                is on top of the returned list
        '''
        # find all YYMMDD* directories underneath the calibrationDir
        pattern = re.compile(r'^\d{8}(_.*)?$')
        matches = [p for p in calibrationDir.rglob('*') if p.is_dir() and pattern.match(p.name)]

        # check for valid calibrations in the directory
        validCalibrations= ['None']
        hasLatest = False
        for m in sorted(matches):
            calibrationFile = m / "calibration.json"
            if calibrationFile.is_file:
                validCalibrations.append(m.name)
        if len(validCalibrations) > 1:
            validCalibrations.append('Latest')
            hasLatest = True
            
        # check our existing calibrations list
        if oldCalibrations is not None:
            # get the top of the list (this is the user selected one)
            currentCalibration = oldCalibrations[0]

            # find it in the new list and put it on top
            if currentCalibration in validCalibrations:
                validCalibrations.remove(currentCalibration)
                validCalibrations.insert(0, currentCalibration)
            else:
                # not found, complain
                swapWith = "Latest" if hasLatest else "None"
                print(f"(pglSettingsManager:getCalibrations) Selected calibration {currentCalibration} not found anymore, defaulting to {swapWith}")
                validCalibrations.remove(swapWith)
                validCalibrations.insert(0,swapWith)
        
        return(validCalibrations)
    
    
    @classmethod
    def getDisplayTemporalCalibrationDir(cls, displaySettings=None, makeDir=False):
        '''
        Get the directory where temporal calibrations live
        
        Args:
            displaySettings (default=None): pglDisplaySettings from which displayName and uuid will be used
                to find the matching directory. If not specified, will just return the top level displayDir
            makeDir (default=False): Set to True to create the directory if it does not already exist
        
        Returns:
            Path: The directory path where display luminance calibrations are stored        
        '''
        temporalCalibrationDir = cls.getDisplayDir(displaySettings, makeDir) / "temporal"
        
        # check if it exists, create if not
        if makeDir and not temporalCalibrationDir.exists():
            try:
                temporalCalibrationDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> Created directory: {temporalCalibrationDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> Error creating directory {temporalCalibrationDir}: {e}"))
                return None
        return temporalCalibrationDir


    @classmethod
    def getDisplayLuminanceCalibrationDir(cls, displaySettings=None, makeDir=False):
        '''
        Get the directory where luminance calibrations live
        
        Args:
            displaySettings (default=None): pglDisplaySettings from which displayName and uuid will be used
                to find the matching directory. If not specified, will just return the top level displayDir
            makeDir (default=False): Set to True to create the directory if it does not already exist
        
        Returns:
            Path: The directory path where display luminance calibrations are stored        
        '''
        luminanceCalibrationDir = cls.getDisplayDir(displaySettings, makeDir) / "luminance"
        
        # check if it exists, create if not
        if makeDir and not luminanceCalibrationDir.exists():
            try:
                luminanceCalibrationDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> Created directory: {luminanceCalibrationDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> Error creating directory {luminanceCalibrationDir}: {e}"))
                return None
        return luminanceCalibrationDir

    @classmethod
    def getDisplayDir(cls, displaySettings=None, makeDir=False):
        """
        Get the directory where display settings are saved
        
        Args:
            displaySettings (default=None): pglDisplaySettings from which displayName and uuid will be used
                to find the matching directory. If not specified, will just return the top level displayDir
            makeDir (default=False): Set to True to create the directory if it does not already exist
        
        Returns:
            Path: The directory path where display settings are stored
        """
        # get the main directory for displays
        displayDir = cls.getPGLDir() / "displays"
        
        # append display specific directory if displaySettings is passed in
        if displaySettings is not None:
            # get a valid filename for displayName
            displayName = pglBase.makeValidFilename(displaySettings.displayName)
            # and append that if it is not empty
            if displayName != "":
                displayDir = displayDir / displayName
            else:
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> No valid displayName found in displaySettings"))
                
        # check if it exists, create if not
        if makeDir and not displayDir.exists():
            try:
                displaysDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> Created directory: {displayDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:getDisplayDir)</b> Error creating directory {displayDir}: {e}"))
                return None

        return displayDir

    @classmethod
    def getCalibrationsDir(cls):
        """
        Get the directory where screen calibrations are stored

        Returns:
            str: The directory path where calibrations are stored
        """
        # get the screenSetttingsDir
        calibrationsDir = cls.getPGLDir() / "calibrations"
        
        # check if it exists, create if not
        if not calibrationsDir.exists():
            try:
                calibrationsDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglScreenSettings:getCalibrationsDir)</b> Created directory: {calibrationsDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:getCalibrationsDir)</b> Error creating directory {calibrationsDir}: {e}"))
                return None

        return calibrationsDir
    
    @classmethod
    def getSettings(cls, settingsName=None, settings=None, displayName=None, displaySettings=None):
        """
        Load settings form directory returned by getSettingsDir()
        
        If you pass a settingsName, it will look for a file named {name}.json in that directory. 
        Note that the name will be converted by pglBase.makeValidFilename so will be lowercase with spaces replaced by _ etc
        
        If settings is set, it will return that settings structure, supersceding settingsName
        
        If displayName is set, will look for that display in the directory returned by getDisplayDir() and will set 
        the display field of the settings from above to that display. If there is no settings from above, it will
        create a default settings and add the display to that.
    
        Args:
            settingsName (str): The name of the settings to use. If not set (and settings not set), will use default settings
            settings (pglSettings): An instance of the pglSettings class. If set, will supersede settingsName.
            displayName (str): The name of the display to use. If set, will be incorporated into settings (and supersede any
                conflicting settings). If there is no settings/settingsName will use default settings
            displaySettings (pglDisplaySettings): The settings of the dispaly to use, will supersed the displayName if set and
                behave in a similar fashion

        Returns:
            An instance of pglSettings
        """
        if settings is not None:
            if not isinstance(settings, pglSettings):
                pglMessages.warning("Settings must be pglSettings", level=2)
                return
        elif settingsName is not None:
            # get the settings directory and create the full path to the settings file
            settingsDir = cls.getSettingsDir()
            settingsPath = Path(settingsDir) / pglBase.makeValidFilename(settingsName)
            settingsPath = settingsPath.with_suffix(".json")
        
            # see if the file exists
            if not settingsPath.exists():
                pglMessages.warning(f"Settings file '{settingsPath}' not found.", level=2)
                return None
            else:
                pglMessages.message(f"Loading settings from '{settingsPath}'.")
                settings = pglSettings.load(filename=settingsPath)
        else:
            settings = pglSettings()
            
        if displaySettings is not None:
            if not isinstance(settings, pglSettings):
                pglMessages.warning("Display settings must be pgDisplaylSettings", level=2)
                return
        elif displayName is not None:
            displaySettings = cls.getDisplaySettings(displayName)
        
        if displaySettings is not None:
            settings.reloadDisplays(selected=displaySettings)
            
        return(settings)

##################################################
# used for inheritence
##################################################
class pglTraitSettings(HasTraits, pglSerialize):
    
    def _getOrderedTraits(self):
        """Return traits in class definition order."""
        ordered = OrderedDict()

        # Walk MRO from base class to subclass
        for cls in reversed(type(self).__mro__):
            for name, obj in cls.__dict__.items():
                if isinstance(obj, TraitType):
                    ordered[name] = obj

        return ordered

    def print(self):
        '''
        print
        '''
        print(f"{self.__class__.__name__}:")
        print("-" * 40)

        for name, trait in self._getOrderedTraits().items():

            # skip private/internal traits
            if name.startswith("_"):
                continue

            label = trait.metadata.get("traitDisplayName", name)
            value = getattr(self, name)

            print(f"{label:<30} {value}")

    def equals(self, other):
        '''
        Check for all traitlet field match between two settings
        '''
        if not isinstance(other, self.__class__):
            return False

        for name in self._getOrderedTraits():
            if getattr(self, name) != getattr(other, name):
                return False

        return True

##################################################
# display Settings 
##################################################
class pglDisplayModeSettings(pglTraitSettings):
    modeName = Unicode("", help="Temp")
    pixelDims = Tuple(Int(), Int(), default_value=(0,0), visible=False, help="Pixel dimensions of screen")
    refreshRate = List(Float(), help="Refresh rates supported for this pixel dimension")

class pglDisplaySettings(pglTraitSettings):
    displayName = Unicode("default", help="Names of screen")
    uuid = Unicode("", help="UUID of display", enabled=False)
    vendor = Int(0, help="Vendor number", enabled=False)
    model = Int(0, help="Model number", enabled=False)
    serialNumber = Int(0, help="Serial number", enabled=False)
    isMain = Bool(False, help="Whether the display is the main display", enabled=False)
    isBuiltin = Bool(False, help="Whether the display is the built-in display of e.g. a laptop", enabled=False)
    flipLeftRight = Bool(False, help="Whether to flip the display left-right")
    flipUpDown = Bool(False, help="Whether to flip the display up-down")
    displayDistance = Float(57, min=0.0, help="Distance from subject eyes to display in cm, used to calculate degress of visual angle")
    displaySize = Tuple(Float, Float, labels=("width","height"), default_value=(30, 20), help="Width and height of display in cm, used to calculate degrees of visual angle")
    displayModes = List(Instance(pglDisplayModeSettings), settingsListKey="modeName", hideKey=True, highlightSelector=False, traitDisplayName="pixelDims", help="All supported display modes")
    luminanceCalibration = List(Unicode(), hasPlotButton=True, buttonFunction="plotLuminanceCalibration", default_value=['None'], help="Which luminance calibration to use")
    temporalCalibration = List(Unicode(), hasPlotButton=True, buttonFunction="plotTemporalCalibration", default_value=['None'], help="Which temporal calibration to use")
            
    def save(self, filename=None):
        '''
        save
        
        Args:
            filename: Filename to save to, if ommitted, will generate path and filename using pglSettingsManager.getDisplayDir
        '''
        if filename is None:
            filename = pglSettingsManager.getDisplayDir(self) / "display.json"
        super().save(filename=filename)
        
    def getCalibrations(self):
        '''
        Looks into calibrations direcotry of display to find luminance and temporal calibrations
        This will populate the fields luminanceCalibration and temporalCalibration with a list of
        directory names of the calibrations
        '''
        
        # get all the luminance calibrations
        luminanceCalibrationDir = pglSettingsManager.getDisplayLuminanceCalibrationDir(displaySettings=self)
        self.luminanceCalibration = pglSettingsManager.getCalibrations(luminanceCalibrationDir, self.luminanceCalibration)
        
        # get all the temporal calibrations
        temporalCalibrationDir = pglSettingsManager.getDisplayTemporalCalibrationDir(displaySettings=self)
        self.temporalCalibration = pglSettingsManager.getCalibrations(temporalCalibrationDir, self.temporalCalibration)


    def plotLuminanceCalibration(self, fig, selected):
        '''
        load and plot the luminance calibration on the passed in axis
        '''
        if selected == "None":
            return False
        
        # load the calibration
        luminanceCalibrationDir = pglSettingsManager.getDisplayLuminanceCalibrationDir(self) / selected 
        
        # load the calibrtion
        from .pglCalibration import pglDisplayLuminanceCalibrationData
        calibration = pglDisplayLuminanceCalibrationData.load(displayName=self.displayName, filepath=luminanceCalibrationDir)
        calibration.display(fig=fig)
        return True
        
    def plotTemporalCalibration(self, fig, selected):
        '''
        load and plot the temporal calibration on the passed in axis
        '''
        if selected == "None":
            return False

        # load the calibration
        temporalCalibrationDir = pglSettingsManager.getDisplayTemporalCalibrationDir(self) / selected 

        # load the calibrtion
        from .pglCalibration import pglDisplayTemporalCalibrationData
        calibration = pglDisplayTemporalCalibrationData.load(displayName=self.displayName, filepath=temporalCalibrationDir)
        calibration.display(fig=fig)
        
        return True
    
    def __eq__(self, other):
        '''
        Define equality as when the two displays share the same uuid
        '''
        
        if not isinstance(other, pglDisplaySettings):
            return NotImplemented

        return self.uuid == other.uuid

class pglDisplaySettingsList(pglTraitSettings):

    settingsList = List(Instance(pglDisplaySettings), settingsListKey="displayName", traitDisplayName="Choose display", help="List of display settings")
    buttons = [("Test", "testDisplay")]

    def testDisplay(self):
        print(f"Testing: {self.settingsList[0].displayName}")
        print(self.settingsList[0].print())
        print(self.settingsList[0].displayModes[0].print())
        
    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList


 
##################################################
# Settings 
##################################################
class pglSettings(pglTraitSettings):
    
    settingsName = Unicode("default", help="Display name for these settings")
    displays = List(Instance(pglDisplaySettings), settingsListKey="displayName", highlightSelector=False, traitDisplayName="choose display", hideAll=True, help="Display - to edit display settings run pgl.displaySettings")
    calibrateForGamma = List(Float, default_value=[2.2, 1.0, 0], help="What gamma to target calibration for 0.0 = No calibration, 1.0=linear, 2.2 typical for images/movies")
    dataPath = Unicode("~/data",help="Path to data directory").tag(isPath=True)
    startKey = Unicode("space", allow_none=True, help="Key to start experiment")
    endKey = Unicode("escape", allow_none=True, help="Key to end experiment")
    volumeTriggerKey = Unicode("`", allow_none=True, help="Key press that signals scanner volume acquisition trigger")
    responseKeys = Unicode("1234", help="Keys used for subject responses. Can be a string like \"1234\" or a comma-separated list like 'left,right,up,down' and will map to response 0,1,2,etc")
    ignoreInitialVolumes = Int(0, min=0, step=1, help="Number of initial volumes to ignore")
    eatKeys = Bool(True, help="Whether to eat keypresses so they don't propagate to the OS. Will only eat the keys specified above.")
    startOnVolumeTrigger = Bool(False, help="Whether to start the experiment on the volume trigger key")
    manualPreStart = Bool(False, help="Whether to manually start the experiment before the volume trigger")
    closeScreenOnEnd = Bool(True, help="Whether to close the screen when the experiment ends")
    backgroundColor = List(trait=Float(min=0.0, max=1.0), default_value=[0.5, 0.5, 0.5],minlen=3,maxlen=3,help="Background color as a list of RGB values").tag(isRGB=True)
    eyetracker =  List(Unicode(), default_value=['None', 'Eyelink'], help="Eyetracker")
    
    @classmethod
    def load(cls, filename):
        '''
        Load pglSettings. 
    
        Also loads displays list so that it has up to date display settings
        '''
        # call super function to load all fields
        cls = super().load(filename)
        
        # reload the displays
        cls.reloadDisplays()
        
        return cls

    def reloadDisplays(self, selected=None):
        '''
        reload the displays so we have the most up-to-date display settings to choose from
        '''

        # load all the display settings
        displays = pglSettingsManager.getDisplaySettings()
        
        if selected is None:
            if self.displays is None or len(self.displays) == 0:
                # no selection, no existing displays, just use default list
                self.displays = displays
                return
            else:
                # get selected from top of existing list
                selected = self.displays[0]
                
        # if displays is not empty, then see if the selected display (the one on top)
        # matches one of the newly loaded displays
        if displays:
            if selected in displays:
                displays.insert(0, displays.pop(displays.index(selected)))
            else:
                displays.insert(0, selected)
            self.displays = displays
        else:
            if selected in self.displays:
                self.displays.insert(0, self.displays.pop(self.displays.index(selected)))
            else:
                self.displays.insert(0, selected)
        
class pglSettingsList(pglTraitSettings):

    settingsList = List(Instance(pglSettings), settingsListKey="settingsName", traitDisplayName="Choose settings", help="List of settings")
    buttons = [("Test", "testDisplay")]

    def testDisplay(self):
        print(self.settingsList[0].print())
        
    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList

    
    
##### DELETE EVERYTHING BELOW THIS LINE WHEN DONE FIXING pglSettings
class pglOLDSettings(pglSettingsEditable):
    # link back to settings select class
    settingsSelect = None 

    # ----- Put up edit dialog ---- #
    def edit(self):
        # call parent method
        super().edit()
        # disable / enable dependent traits
        self.disableEnable(self.displayNumber)

    # ----- callback for onSave button ---- # 
    def onSave(self, saveButton):
    
        # confirmation panel
        def confirmSave():
            # get the settingsDir
            from .pglExperiment import pglExperiment
            e = pglExperiment(None)
            settingsDir = e.getSettingsDir()

            # get the screenSetttingsDir
            settingsFilename = settingsDir / self.settingsName
            settingsFilename = settingsFilename.with_suffix(".json")
    
            # save it
            self.save(settingsFilename)
            pglDisplayMessage(f"<b>Saved settings to:</b> {settingsFilename}", duration=displayDuration)
        
            if self.settingsSelect is not None:
                # Just update this instance in the select list
                self.settingsSelect.update(self)
    
        panel = confirmationPanel(confirmMessage="Are you sure you want to save?", onConfirm=confirmSave)
        panel.display()

    # ----- callback for onDelete button ---- # 
    def onDelete(self, deleteButton):
        # confirmation panel
        def confirmDelete():
            # get the settingsDir
            from .pglExperiment import pglExperiment
            e = pglExperiment(None, suppressInitScreen=True)
            settingsDir = e.getSettingsDir()

            # get the screenSetttingsDir
            settingsFilename = settingsDir / self.settingsName
            settingsFilename = settingsFilename.with_suffix(".json")
            
            # delete the file
            try:
                settingsFilename.unlink()
                self.hide()
                if self.settingsSelect is not None:
                    # remove from settingsSelect
                    self.settingsSelect.remove(self)
                pglDisplayMessage(f"<b>Deleted settings file:</b> {settingsFilename}", duration=displayDuration)
            except Exception as e:
                pglDisplayMessage(f"<b>Error deleting settings file {settingsFilename}:</b> {e}", duration=displayDuration)
        
        panel = confirmationPanel(confirmMessage="Are you sure you want to delete?", onConfirm=confirmDelete)
        panel.display()
        
    # ----- callback for onTest button ---- #   
    def onTest(self, testButton):
        # init experiment
        from pgl import pgl
        pgl = pgl()
        from .pglExperiment import pglExperiment, pglTestTask
        e = pglExperiment(pgl, settings=self)
                
        # initialize task
        t = pglTestTask(pgl)
        e.addTask(t)
        
        # open screen
        e.initScreen()
        
        # and run
        e.run()

    # ----- default for settingsName ---- #
    @default('settingsName')
    def _default_settingsName(self):
        try:
            result = subprocess.run(
                ['scutil', '--get', 'ComputerName'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            # Fallback
            return platform.node().split('.')[0]
        
    # ----- Callbacks for list change ---- #
    # when the displayNames is selected then find the correct
    # displayNumber and set accordingly
    def onListSelect(self, traitName, change):
        # call parent method
        super().onListSelect(traitName, change)
        if traitName == "displayName":
            from pgl import pgl
            displayNames = pgl().getDisplayNames()
            # get the selected display name
            selectedName = change['new']
            # find the index in displayNames
            if selectedName in displayNames:
                index = displayNames.index(selectedName)
                # and set displayNumber accordingly
                self.displayNumber = index
    
     # ----- Callbacks for int change ---- #
    # when the displayNumber is changed, switch the displayNames accordingly
    def onIntSelect(self, traitName, change):
        if traitName == "displayNumber":
            # get the changed number
            displayNumber = change['new']
            # look it up in displays
            from pgl import pgl
            displayNames = pgl().getDisplayNames(displayNumber)
            # and update displayName
            self.displayName = displayNames
            # disable / enable dependent traits
            self.disableEnable(displayNumber)

    # ----- Disable / enable dependent traits ---- #
    def disableEnable(self, displayNumber):
        # Disable / enable traits dependent on displayNumber
        if hasattr(self, 'widgetMap'):
            for trait in ["windowWidth", "windowHeight"]:
                widget = self.widgetMap.get(trait)
                if widget:
                    widget.disabled = displayNumber!=0
                    



# Screen settings select
class pglSettingsSelect(pglSettingsEditable):
    
    # traits that can be edited
    settingsNames = List(Unicode(), help="Settings names")
    #default = Bool(False, help="Whether this is the default settings")
    
    # Variable containing all the settings, this is set by calling class
    settings = []

    def __init__(self, pgl=None):
        self.pgl = pgl
        super().__init__()
        
    def load(self, settingsName=None):
        # initialize experiment so that we can get settings from it
        from .pglExperiment import pglExperiment
        e = pglExperiment(self)
        
        # get the screen settings directory
        settingsDir = e.getSettingsDir()
        
        # cycle through all files in settingsDir with .json extension
        # and load as a pglScreenSettings instance
        settings = []
        for jsonFile in Path(settingsDir).glob("*.json"):
            # load settings from file
            s = pglSettings(jsonFile)
            # put in displayNames, putting the matching number on top
            s.displayName = self.pgl.getDisplayNames(s.displayNumber)
            # add a link to this settingsSelect
            s.settingsSelect = self
            # append to list
            settings.append(s)
            
        # if settings is empty, then create a default settings
        if len(settings) == 0:
            # create default settings
            settings.append(pglSettings())
            # and save
            settings[0].onSave(None)
            
        if settingsName is not None:
            # find the settings with this name and put it on top
            for i, s in enumerate(settings):
                if s.settingsName == settingsName:
                    # move to top
                    settings.insert(0, settings.pop(i))
                    break
                
        # Now set our settingsNames trait and settings
        self.settingsNames = [s.settingsName for s in settings]
        self.settings = settings
        
    # ----- Callbacks for list change ---- #
    # when the displayName is selected, edit those settings
    def onListSelect(self, traitName, change):
        # call parent method
        super().onListSelect(traitName, change)
        # load the selected settings
        selectedName = change['new']
        # go through settings, to see which one it matches to
        for s in self.settings:
            if s.settingsName == selectedName:
                # display the settings
                s.edit()
            else:
                s.hide()
    
    def remove(self, settingsInstance):
        # remove the settingsInstance from our list
        self.settings = [s for s in self.settings if s != settingsInstance]
        # update settingsNames
        self.settingsNames = [s.settingsName for s in self.settings]

    def update(self, settingsInstance):
        """
        Update or add a settings instance to the list.
        If the settingsName already exists, replace it.
        If it's new, add it to the list.
        """
        # Check if this settings name already exists
        existingIndex = None
        for i, s in enumerate(self.settings):
            if s.settingsName == settingsInstance.settingsName:
                existingIndex = i
                break
    
        if existingIndex is not None:
            # Replace existing - hide the old one first
            self.settings[existingIndex].hide()
            if hasattr(self.settings[existingIndex], 'wrapper'):
                self.settings[existingIndex].wrapper.close()
            self.settings[existingIndex] = settingsInstance
        else:
            # Add new to the list
            self.settings.append(settingsInstance)
    
        # Update settingsNames and move this one to top
        allNames = [s.settingsName for s in self.settings]
        self.settingsNames = [settingsInstance.settingsName] + [n for n in allNames if n != settingsInstance.settingsName]

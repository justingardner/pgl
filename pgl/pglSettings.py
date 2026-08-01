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
from .pglParameter import pglParameter, pglParameterBlock
from .pglSerialize import pglSerialize
from .pglDialog import pglDialogs
import Quartz
import CoreFoundation
from AppKit import NSScreen
from .pglBase import pglBase
import re
from collections import OrderedDict
from .pglMessages import pglMessages
import uuid

#######################################
# Mixin class for pgl to provide settings management
#######################################
class pglSettingsManager:
    """
    Mixin class for pgl to provide settings management.
    """
    def __init__(self):   
        pass
    
    def settings(self):
        """
        Edit pgl settings. Brings up widget interface to edit settings
        """
        # get settings dir
        settingsDir = self.getSettingsDir()
        
        # load all the seettings in there
        settingsList = []
        for filename in Path(settingsDir).glob("*.json"):
            settings = pglSettings.load(filename=filename)
            settingsList.append(settings)
        # if no saved settings, make a default one
        if not settingsList:
            settingsList.append(pglSettings())
        original = pglSettingsList(settingsList)    
        
        # bring up dialog
        modified = pglDialogs.traitsDialog(original)
        
        # and save
        self._saveModifiedSettings(modified, original)
                
    def displaySettings(self):
        """
        Edit pgl display settings. Brings up widget interface to edit display settings
        """
        # get the display infos
        original = pglDisplaySettingsList(self.getDisplaySettings())
        
        # display the settings
        modified = pglDialogs.traitsDialog(original)
        
        # and save
        self._saveModifiedSettings(modified, original)

    def _saveModifiedSettings(self, modifiedSettingsList, originalSettingsList):
        '''
        Save only modified settings from a settings list (could be either  pglDispalySettingsList or pglSettingsLIst)
        '''
        # save the settings if user clicked OK
        if modifiedSettingsList is not None:
            # for each display in modified list
            for modifiedSettings in modifiedSettingsList.settingsList:
                # compare to original
                matchingOriginalSettings = next((originalSettings for originalSettings in originalSettingsList.settingsList if originalSettings == modifiedSettings), None)
                if matchingOriginalSettings is not None:
                    # and if it is not equal (field by field) then save it
                    if not matchingOriginalSettings.equals(modifiedSettings):
                        # if the name changed, we need to change the name of the directory
                        if modifiedSettings.name != matchingOriginalSettings.name:
                            oldPath = matchingOriginalSettings.saveDir().parent
                            newPath = modifiedSettings.saveDir().parent
                            try:
                                # rename to the new path
                                if oldPath.exists(): oldPath.rename(newPath)
                            except OSError as e:
                                # if it did not work, provide an error
                                pglMessages.warning("Could not change directory name from {oldPath.name} to {newPath.name}, keeping {oldPath.name}")
                                modifiedSettings.name = matchingOriginalSettings.name
                        # save the modified display
                        modifiedSettings.save()
    
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
            if pglBase.makeValidFilename(displaySettings.name) != str(p.parent.name):
                displaySettings.name = str(p.parent.name)
            # get the calibrations
            displaySettings.getCalibrations()
            # set the displayNum to -1, so that the next piece of code can find it
            displaySettings.currentDisplayNum = -1
            # and add to display list
            displays.append(displaySettings)
                
        maxDisplays = 16        
        (err, active, count) = Quartz.CGGetActiveDisplayList(maxDisplays, None, None)

        for iDisplay, displayID in enumerate(active):
            # initialize the displaySettings
            displaySettings = pglDisplaySettings()
            
            # get all supported modes
            modes = Quartz.CGDisplayCopyAllDisplayModes(displayID, None)
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
            uuidRef = CGDisplayCreateUUIDFromDisplayID(displayID)
            displaySettings.uuid = str(CoreFoundation.CFUUIDCreateString(None, uuidRef))
            
            # get other infor from quartz
            displaySettings.vendor        = Quartz.CGDisplayVendorNumber(displayID)
            displaySettings.model         = Quartz.CGDisplayModelNumber(displayID)
            displaySettings.serialNumber  = Quartz.CGDisplaySerialNumber(displayID)
            displaySettings.isMain        = Quartz.CGDisplayIsMain(displayID)
            displaySettings.isBuiltin     = Quartz.CGDisplayIsBuiltin(displayID)
            displaySettings.gammaTableSize = Quartz.CGDisplayGammaTableCapacity(displayID)
            
            # get display human readable name
            displaySettings.name = cls.getMatchingDisplayName(displayID)                    
            
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
                # and set its current display num
                matchingDisplay.currentDisplayNum = iDisplay
                # and gamma table size
                matchingDisplay.gammaTableSize = displaySettings.gammaTableSize
            else:
                displaySettings.currentDisplayNum = iDisplay
                # append to our list of all displays
                displays.append(displaySettings)
                
        if displayName is not None:
            # find the display with the matching displayName (compare using makeValidFilename to make case insenstive)
            return next(
                (d for d in displays if pglBase.makeValidFilename(d.name) == pglBase.makeValidFilename(displayName)),
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
    def getDisplayTemporalCalibrationDir(cls, displaySettings=None, newCalibration=False):
        '''
        Get the directory where temporal calibrations live
        
        Args:
            displaySettings (default=None): pglDisplaySettings from which displayName and uuid will be used
                to find the matching directory. If not specified, will just return the top level displayDir
            makeDir (default=False): Set to True to create the directory if it does not already exist
        
        Returns:
            Path: The directory path where display luminance calibrations are stored        
        '''
        temporalCalibrationDir = cls.getDisplayDir(displaySettings) / "temporal"
        if newCalibration:
            temporalCalibrationDir = temporalCalibrationDir / datetime.now().strftime("%Y%m%d_%H%M%S")
       
        # check if it exists, create if not
        if newCalibration and not temporalCalibrationDir.exists():
            try:
                temporalCalibrationDir.mkdir(parents=True, exist_ok=True)
                pglMessages.message(f"Created directory: {temporalCalibrationDir}")
            except Exception as e:
                pglMessages.message(f"Error creating directory {temporalCalibrationDir}: {e}")
                return None
        return temporalCalibrationDir


    @classmethod
    def getDisplayLuminanceCalibrationDir(cls, displaySettings=None, newCalibration=False):
        '''
        Get the directory where luminance calibrations live
        
        Args:
            displaySettings (default=None): pglDisplaySettings from which displayName and uuid will be used
                to find the matching directory. If not specified, will just return the top level displayDir
            newCalibration (default=False): Set to True to also make a directory underneath with the data and time

        
        Returns:
            Path: The directory path where display luminance calibrations are stored        
        '''
        luminanceCalibrationDir = cls.getDisplayDir(displaySettings) / "luminance"
        if newCalibration:
            luminanceCalibrationDir = luminanceCalibrationDir / datetime.now().strftime("%Y%m%d_%H%M%S")
        # check if it exists, create if not
        if newCalibration and not luminanceCalibrationDir.exists():
            try:
                luminanceCalibrationDir.mkdir(parents=True, exist_ok=True)
                pglMessages.message(f"Created directory: {luminanceCalibrationDir}")
            except Exception as e:
                pglMessages.message(f"Error creating directory {luminanceCalibrationDir}: {e}")
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
            displayName = pglBase.makeValidFilename(displaySettings.name)
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
                pglMessages.warning("Settings must be pglSettings")
                return
        elif settingsName is not None:
            # get the settings directory and create the full path to the settings file
            settingsDir = cls.getSettingsDir()
            settingsPath = Path(settingsDir) / pglBase.makeValidFilename(settingsName)
            settingsPath = settingsPath.with_suffix(".json")
        
            # see if the file exists
            if not settingsPath.exists():
                pglMessages.warning(f"Settings file '{settingsPath}' not found.")
                return None
            else:
                pglMessages.message(f"Loading settings from '{settingsPath}'.")
                settings = pglSettings.load(filename=settingsPath)
        else:
            settings = pglSettings()

        # validate displaySettings / displayName
        if displaySettings is not None:
            if not isinstance(displaySettings, pglDisplaySettings):
                pglMessages.warning("Display settings must be pgDisplaylSettings")
                return
        elif displayName is not None:
            displaySettings = cls.getDisplaySettings(displayName)
        
        # update displays
        if displaySettings is not None:
            settings.reloadDisplays(selected=displaySettings, overwrite=True)

        return(settings)

##################################################
# used for inheritence
##################################################
class pglTraitSettings(HasTraits, pglSerialize):
    
    # all trait setttings should have a name field and a uuid
    name = Unicode("default", help="", visible=False, enabled=False)
    uuid = Unicode("", help="Universal unique identifier for this setting", visible=False, enabled=False)

    # default uuid
    @default("uuid")
    def _default_uuid(self):
        return str(uuid.uuid4())
    
        
    def __eq__(self, other):
        '''
        Define equality as when the two traitlet settings share the same uuid
        '''
        
        if not isinstance(other, pglTraitSettings):
            return NotImplemented

        return self.uuid == other.uuid
    
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
            if not self._valueEquals(getattr(self, name), getattr(other, name)):
                return False

        return True


    def _valueEquals(self, a, b):
        # same object or simple equality
        if a is b:
            return True

        # nested settings objects
        if hasattr(a, "equals") and hasattr(b, "equals"):
            return a.equals(b)

        # lists / tuples
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False

            return all(self._valueEquals(x, y) for x, y in zip(a, b))

        # dictionaries if you use them
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False

            return all(self._valueEquals(a[k], b[k]) for k in a)

        # numpy arrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)

        # fallback
        return a == b
##################################################
# display Settings 
##################################################
class pglDisplayModeSettings(pglTraitSettings):
    modeName = Unicode("", help="Temp")
    pixelDims = Tuple(Int(), Int(), default_value=(0,0), visible=False, help="Pixel dimensions of screen")
    refreshRate = List(Float(), help="Refresh rates supported for this pixel dimension")

    def __eq__(self, other):
        '''
        compare to other displayMode or to a tuple which has (pixelWidth, pixelHeight, refreshRate, ...)
        '''
        
        if isinstance(other, pglDisplayModeSettings):
            return (self.screenWidth, self.screenHeight, self.refresh) == \
                   (other.screenWidth, other.screenHeight, other.refresh)
        elif isinstance(other, tuple):
            if len(other) < 3:
                return False
            else:
                return (self.pixelDims[0], self.pixelDims[1], self.refreshRate) == other[0:3]
        return NotImplemented

class pglDisplaySettings(pglTraitSettings):
    name = Unicode("default", help="Names of screen")
    uuid = Unicode("", help="UUID of display", enabled=False)
    vendor = Int(0, help="Vendor number", enabled=False)
    model = Int(0, help="Model number", enabled=False)
    serialNumber = Int(0, help="Serial number", enabled=False)
    isMain = Bool(False, help="Whether the display is the main display", enabled=False)
    isBuiltin = Bool(False, help="Whether the display is the built-in display of e.g. a laptop", enabled=False)
    flipLeftRight = Bool(False, help="Whether to flip the display left-right")
    flipUpDown = Bool(False, help="Whether to flip the display up-down")
    currentDisplayNum = Int(-1, help="Which display number this corresponds to. If not currently connected will be -1", enabled=False)
    gammaTableSize = Int(-1, help="Size of gamma table", enabled=False)
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
        # get the filename
        if not filename: filename = self.saveDir()

        # Ensure directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)
 
        super().save(filename=filename)
        
    def saveDir(self):
        return pglSettingsManager.getDisplayDir(self) / "display.json"
    
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
        calibration = pglDisplayLuminanceCalibrationData.load(displayName=self.name, filepath=luminanceCalibrationDir)
        calibration.display(fig=fig)
        return True
    
    def getLuminanceCalibration(self):
       '''
       get the luminance calibration
       '''
       if not self.luminanceCalibration or self.luminanceCalibration[0] == 'None':
           pglMessages.warning("No luminance calibration found for {self.name}")
           return
       
       # load the calibration
       from .pglCalibration import pglDisplayLuminanceCalibrationData
       luminanceCalibrationDir = pglSettingsManager.getDisplayLuminanceCalibrationDir(self) / self.luminanceCalibration[0]
       calibration = pglDisplayLuminanceCalibrationData.load(displayName=self.name, filepath=luminanceCalibrationDir)
       if calibration is None:
           pglMessages.warning("Could not load luminance calibration from {luminanceCalibrationDir}") 
           return
       
       return calibration
    
    def setGamma(self, pgl, gamma):
        '''
        Set the gamma using the calibration
        '''
        
        # get the current luminance calibration
        calibration = self.getLuminanceCalibration()
        
        # and set the gamma
        calibration.setDisplayToGamma(pgl, self, gamma)
        
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
        calibration = pglDisplayTemporalCalibrationData.load(displayName=self.name, filepath=temporalCalibrationDir)
        calibration.display(fig=fig)
        
        return True
    

class pglDisplaySettingsList(pglTraitSettings):

    settingsList = List(Instance(pglDisplaySettings), settingsListKey="name", traitDisplayName="Choose display", help="List of display settings")
    buttons = [("Test", "testDisplay")]

    ##########################
    # test display settings
    ##########################
    def testDisplay(self):
        try:
            from pgl import pgl
            pgl = pgl()
            from .pglExperiment import pglExperiment
            from .pglTasks import pglTestTask

            e = pglExperiment(pgl, displaySettings=self.settingsList[0])

            # initialize task
            t = pglTestTask(pgl)
            e.addTask(t)
            
            # open screen
            e.initScreen()
            
            # and run
            e.run()
            
            # print settings
            self.settingsList[0].print()
            self.settingsList[0].displayModes[0].print()

        except Exception as e:
            pglMessages.warning(f"Could not run test. Error {type(e).__name__}: {e}")    
            return

        
    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList


 
##################################################
# Settings 
##################################################
class pglSettings(pglTraitSettings):
    
    name = Unicode("default", help="Display name for these settings")
    displays = List(Instance(pglDisplaySettings), settingsListKey="name", highlightSelector=False, traitDisplayName="choose display", hideAll=True, help="Display - to edit display settings run pgl.displaySettings")
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
    
    def __init__(self):
        super().__init__()
        self.reloadDisplays()

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
    
    def save(self, filename=None):
        '''
        save
        
        Args:
            filename: Filename to save to, if ommitted, will generate path and filename using pglSettingsManager.getSettingsDir
        '''
        # get the filename
        if not filename: filename = self.saveDir()

        # Ensure directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        super().save(filename=filename)
        
    def saveDir(self):
        return pglSettingsManager.getSettingsDir() / f"{pglBase.makeValidFilename(self.name)}.json"

    
    def reloadDisplays(self, selected=None, overwrite=False):
        '''
        reload the displays so we have the most up-to-date display settings to choose from
        
        Args:
            Selected (pglDisplaySettings): If set, puts the selected display on top of the dispaly list
            overwrite (Bool): If True, uses the selected and updates only the currentDispalyNum from loaded
                used for pgl.displaySettings button callback (which doesn't update from saved)
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
                # get the currentDisplayNum
                selectedMatch = displays.pop(displays.index(selected))
                # if overwrite then just copy currentDisplayNum
                if overwrite:
                    selected.currentDisplayNum = selectedMatch.currentDisplayNum
                else:
                    # if not overwrite, then get the latest version of the display
                    # which loads from disk and updates selectf ields like currentDisplayNum
                    # from current settings
                    selected = selectedMatch
                displays.insert(0, selected)
            else:
                displays.insert(0, selected)
            self.displays = displays
        else:
            if selected in self.displays:
                self.displays.pop(self.displays.index(selected))
                self.displays.insert(0, selected)
            else:
                self.displays.insert(0, selected)
        
class pglSettingsList(pglTraitSettings):

    settingsList = List(Instance(pglSettings), buttons = True, settingsListKey="name", traitDisplayName="Choose settings", help="List of settings")
    
    buttons = [("Test", "testDisplay")]

    ##########################
    # test display settings
    ##########################
    def testDisplay(self):
        try:
            from pgl import pgl
            pgl = pgl()
            from .pglExperiment import pglExperiment
            from .pglTasks import pglTestTask
            e = pglExperiment(pgl, settings=self.settingsList[0])
                    
            # initialize task
            t = pglTestTask(pgl)
            e.addTask(t)
            
            # open screen
            e.initScreen()
            
            # and run
            e.run()
            
            # print settings
            self.settingsList[0].print()
            self.settingsList[0].displays[0].print()
            self.settingsList[0].displays[0].displayModes[0].print()
        except Exception as e:
            pglMessages.warning(f"Could not run test. Error {type(e).__name__}: {e}")    
            return
        
    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList

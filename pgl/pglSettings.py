################################################################
#   filename: pglSettings.py
#    purpose: Provides settings management for pgl
#         by: JLG
#       date: Feb 6, 2026
################################################################

#############
# Import
#############
from curses import wrapper
from pathlib import Path
from IPython.display import display, HTML, clear_output
from fileinput import filename
from ipywidgets.widgets import widget
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from datetime import datetime   
import numpy as np
from .pglSerialize import pglSerialize
from .pglDialog import pglDialogs
from .pglStorage import pglStorage
from .pglInfo import pglInfo
import re
from collections import OrderedDict
from .pglMessages import pglMessages
import uuid
import posixpath
from os.path import exists, join
from copy import deepcopy
import getpass
from traitlets import default
import ast


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
        Edit pgl settings without requiring an instance.
        """
        # get settings list
        settingsList = cls.getSettings(returnSettingsList=True)

        # keep original
        original = pglSettingsList(settingsList)

        # bring up dialog
        modified = pglDialogs.traitsDialog(original)

        # and save
        cls._saveModifiedSettings(modified, original)

    @classmethod
    def displaySettings(cls):
        """
        Edit display settings without requiring an instance.
        """
        # get the display infos
        original = pglDisplaySettingsList(cls.getDisplaySettings())

        # display the settings
        modified = pglDialogs.traitsDialog(original)

        # and save
        cls._saveModifiedSettings(modified, original)

    @classmethod
    def importSettings(cls):
        """
        Edit import settings without requiring an instance.
        """
        # get settings list
        settingsList = cls.getImportSettings(returnSettingsList=True)

        # keep original
        original = pglImportSettingsList(settingsList)

        # bring up dialog
        modified = pglDialogs.traitsDialog(original)

        # and save
        cls._saveModifiedSettings(modified, original)

    def eyeTrackerSettings(self, eyeTracker=None, settings=None, settingsName=None):
        '''
        brings up dialog of settings for eye tracker
        
        Args:
            eyetracker (string): If set, brings up settings for particularl eye tracker (e.g. eyelink, trackPixx)
            settings (pglSettings): If set, brings up eyetracker settings for the eyetracker in settings
            settingsName (string): Name of settings, if set, gets corresponding settings and brings up eyetracker settings based on that one
        '''
        eyeTrackerSettings = self.getEyeTrackerSettings(eyeTracker=eyeTracker, settings=settings, settingsName=settingsName)
        if eyeTrackerSettings is None: 
            pglMessages.message("No eye tracker for settings")
            return
        eyeTrackerSettings = pglDialogs.traitsDialog(eyeTrackerSettings)
        if eyeTrackerSettings is not None:
            dataPath = self.getEyeTrackersDir()
            eyeTrackerSettings.save(filename=join(dataPath, 'eyelink.json'))
        
    def getEyeTrackerSettings(self, eyeTracker=None, settings=None, settingsName=None):
        '''
        get settings for eye tracker
        
        Args:
            eyetracker (string): If set, brings up settings for particularl eye tracker (e.g. eyelink, trackPixx)
            settings (pglSettings): If set, brings up eyetracker settings for the eyetracker in settings
            settingsName (string): Name of settings, if set, gets corresponding settings and brings up eyetracker settings based on that one
        
        Returns:
            pglEyeTrackerSettings
        '''
        from .pglEyelink import pglEyelinkSettings

        # get eye tracker 
        if eyeTracker is None:
            settings = self.getSettings(settingsName=settingsName, settings=settings)
            if settings is not None:
                eyeTracker = settings.eyetracker[0]
        
        if eyeTracker is None:
            pglMessages.message("No eye tracker to calibrte")
        elif eyeTracker.lower() == 'eyelink':            
            dataPath = join(self.getEyeTrackersDir(), 'eyelink.json')
            if exists(dataPath):
                return pglEyelinkSettings.load(filename=dataPath)
            else:
                return pglEyelinkSettings()
        
    @classmethod
    def isShiftPressed(cls):
        """Check Shift on macOS or Windows; otherwise use normal save behavior."""
        import sys

        # macOS: read the current keyboard modifier flags.
        if sys.platform == "darwin":
            import Quartz
            flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState)
            return bool(flags & Quartz.kCGEventFlagMaskShift)

        # Windows: the high bit indicates whether Shift is currently pressed.
        if sys.platform == "win32":
            import ctypes
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)

        # Other platforms: save only new or modified settings.
        return False    
    
    @classmethod
    def getImportSettingsDir(cls):
        """
        Get the directory where import settings are stored.

        Returns:
            Path: The directory path where import settings are stored.
        """
        # get the import settings directory
        importSettingsDir = cls.getPGLSettingsDir() / "importSettings"

        # check if it exists, create if not
        importSettingsDir.mkdir(parents=True, exist_ok=True)

        return importSettingsDir

    @classmethod
    def getImportSettings(cls, settingsName=None, settings=None, returnSettingsList=False):
        """
        Get import settings.

        If settings is supplied, return it unchanged.
        Otherwise, load the named settings or return the default settings.

        Args:
            settingsName (str): Name of the saved import settings to load.
            settings (pglImportSettings): Settings instance to use instead.
            returnSettingsList (bool): If True, ignore other arguments and
                return all settings with the default first.

        Returns:
            pglImportSettings, list[pglImportSettings], or None.
        """
        if not returnSettingsList:
            # explicitly supplied settings take precedence
            if settings is not None:
                if not isinstance(settings, pglImportSettings):
                    pglMessages.warning("Import settings must be pglImportSettings")
                    return None
                return settings

            # load a particular named configuration
            if settingsName is not None:
                settingsPath = cls.getImportSettingsDir() / f"{pglStorage.makeValidFilename(settingsName)}.json"

                if not settingsPath.is_file():
                    pglMessages.warning(f"Import settings file '{settingsPath}' not found.")
                    return None

                return pglImportSettings.load(filename=settingsPath)

        # load all saved import settings
        settingsList = []
        for settingsPath in sorted(cls.getImportSettingsDir().glob("*.json")):
            loadedSettings = pglImportSettings.load(filename=settingsPath)

            if loadedSettings is None:
                pglMessages.warning(f"Could not load import settings from '{settingsPath}'")
                continue

            settingsList.append(loadedSettings)

        # if no saved settings, make a default one
        if not settingsList:
            pglMessages.message("No saved import settings found, creating default settings.")
            defaultSettings = pglImportSettings()
            defaultSettings.isDefault = True
            defaultSettings.save()
            settingsList = [defaultSettings]
        else:
            # find the default settings
            defaultSettings = next((s for s in settingsList if s.isDefault), None)

            if defaultSettings is None:
                defaultSettings = settingsList[0]
                defaultSettings.isDefault = True
                defaultSettings.save()
                pglMessages.message(f"No default import settings found, using: {defaultSettings.name}")

            # put the default first
            settingsList.insert(0, settingsList.pop(settingsList.index(defaultSettings)))

            # keep only one default
            extraDefaults = [s for s in settingsList[1:] if s.isDefault]
            if extraDefaults:
                pglMessages.warning("Multiple default import settings found, keeping only the first one")
                for otherSettings in extraDefaults:
                    otherSettings.isDefault = False
                    otherSettings.save()

        return settingsList if returnSettingsList else settingsList[0]
    
    @classmethod
    def _saveModifiedSettings(cls, modifiedSettingsList, originalSettingsList):
        """
        Save new or modified settings and move deleted settings to trash.

        Display settings rename their containing directory so calibrations
        move with them. Other settings rename their individual JSON file.
        """
        # nothing to save if the user cancelled
        if modifiedSettingsList is None: return

        # save all settings if shift is pressed
        saveAll = cls.isShiftPressed()
        if saveAll:
            pglMessages.message("Shift key pressed, saving all settings")

        # save new or modified settings
        for modifiedSettings in modifiedSettingsList.settingsList:
            # match by settings identity
            matchingOriginalSettings = next(
                (
                    originalSettings
                    for originalSettings in originalSettingsList.settingsList
                    if originalSettings == modifiedSettings
                ),
                None,
            )

            # save new settings
            if matchingOriginalSettings is None:
                pglMessages.message(f"Saving new settings {modifiedSettings.name}")
                modifiedSettings.save()
                continue

            # skip unchanged settings unless saving all
            if not saveAll and matchingOriginalSettings.equals(modifiedSettings):
                continue

            # rename the saved file or display directory if the name changed
            if modifiedSettings.name != matchingOriginalSettings.name:
                oldPath = matchingOriginalSettings.saveDir()
                newPath = modifiedSettings.saveDir()

                # display directories also contain their calibrations
                if isinstance(modifiedSettings, pglDisplaySettings):
                    oldPath = oldPath.parent
                    newPath = newPath.parent

                if oldPath != newPath:
                    try:
                        # do not overwrite an existing destination
                        if newPath.exists():
                            raise FileExistsError(f"Destination already exists: {newPath}")

                        if oldPath.exists(): oldPath.rename(newPath)
                    except OSError as e:
                        pglMessages.warning(
                            f"Could not rename {oldPath.name} to {newPath.name}: {e}. "
                            f"Keeping {matchingOriginalSettings.name}"
                        )
                        modifiedSettings.name = matchingOriginalSettings.name

            # save modified settings
            modifiedSettings.save()

        # move deleted settings to trash
        for originalSettings in originalSettingsList.settingsList:
            if not any(
                modifiedSettings == originalSettings
                for modifiedSettings in modifiedSettingsList.settingsList
            ):
                pglMessages.message(f"Deleting settings {originalSettings.name}")
                cls.moveToTrash(originalSettings.saveDir())
    
    @classmethod
    def getDisplaySettings(cls, displayName=None):
        '''
        Get all of the displaySettings
        
        Args:
            displayName (str): If not (default=None) will return the matching setting or None if not found
        '''
        import Quartz
        import CoreFoundation
        displays = []
        
        # Get CGDisplayCreateUUIDFromDisplayID
        try:
            from ColorSync import CGDisplayCreateUUIDFromDisplayID
        except ImportError as e:
            pglMessages.message(f"CGDisplayCreateUUIDFromDisplayID not found in ColorSync, trying Quartz: {e}")
            # fallback: some builds expose it under Quartz
            from Quartz import CGDisplayCreateUUIDFromDisplayID
            
        # first check saved displays
        displayDir = cls.getDisplayDir()
        for p in displayDir.rglob('display.json'):
            # load json settings
            displaySettings = pglDisplaySettings().load(p)
            # make sure displayName matches diectory
            if pglStorage.makeValidFilename(displaySettings.name) != str(p.parent.name):
                displaySettings.name = str(p.parent.name)
            # get the calibrations
            displaySettings.getCalibrations()
            # set the displayNum to -1, so that the next piece of code can find it
            displaySettings.currentDisplayNum = -1
            # and add to display list
            displays.append(displaySettings)
                
        maxDisplays = 16        
        (err, active, count) = Quartz.CGGetActiveDisplayList(maxDisplays, None, None)
        currentRefreshRate = 0.0
        
        for iDisplay, displayID in enumerate(active):
            # initialize the displaySettings
            displaySettings = pglDisplaySettings()
            
            # get the current mode settings
            currentMode = Quartz.CGDisplayCopyDisplayMode(displayID)
            currentWidth = Quartz.CGDisplayModeGetWidth(currentMode)
            currentHeight = Quartz.CGDisplayModeGetHeight(currentMode)
            currentRefreshRate = Quartz.CGDisplayModeGetRefreshRate(currentMode)
            displaySettings.currentDisplayMode = (currentWidth, currentHeight, currentRefreshRate)

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
                    
                    # see if this mode matches current
                    if (w, h) == (currentWidth, currentHeight):
                        # put the current refresh rate at the top of the list
                        # making sure not to dupliacte it if it is already there
                        if currentRefreshRate in displayModeSettings.refreshRate:
                            displayModeSettings.refreshRate.remove(currentRefreshRate)
                        displayModeSettings.refreshRate.insert(0, currentRefreshRate)
                        # insert this mode at the top of the displayModes list
                        displayModes.insert(0, displayModeSettings)
                    else:
                        displayModes.append(displayModeSettings)

            displaySettings.displayModes = displayModes     
                   
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
                # and display mode
                matchingDisplay.currentDisplayMode = displaySettings.currentDisplayMode
                # and gamma table size
                matchingDisplay.gammaTableSize = displaySettings.gammaTableSize
            else:
                displaySettings.currentDisplayNum = iDisplay
                # append to our list of all displays
                displays.append(displaySettings)
                
        # see if we have a windowed display
        displaySettingsWindowed = next((d for d in displays if d.uuid == "windowed"), None)
        if displaySettingsWindowed is None:
            displaySettingsWindowed = pglDisplaySettingsWindowed()
            #displaySettingsWindowed = pglDisplaySettings()
            displaySettingsWindowed.name = "Windowed"
            displaySettingsWindowed.uuid = "windowed"
            displaySettingsWindowed.currentDisplayNum = 0
            displaySettingsWindowed.displayModes = [pglDisplayModeSettings(modeName="800 x 600", pixelDims=(800, 600), refreshRate=[60.0])]
            displays.append(displaySettingsWindowed)
        displaySettingsWindowed.currentDisplayNum = 0
        displaySettingsWindowed.currentDisplayMode = (currentWidth, currentHeight, currentRefreshRate)
        displaySettingsWindowed.luminanceCalibration = ['None']
        displaySettingsWindowed.temporalCalibration = ['None']
               
        if displayName is not None:
            # find the display with the matching displayName (compare using makeValidFilename to make case insenstive)
            return next(
                (d for d in displays if pglStorage.makeValidFilename(d.name) == pglStorage.makeValidFilename(displayName)),
                None
            )                
        return(displays)

    @classmethod
    def getMatchingDisplayName(cls, display):
        
        from AppKit import NSScreen
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
        if not getattr(cls, "gpuInfo", None):
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
    
    @classmethod
    def moveToTrash(cls, filepath):
        '''
        Move a file to the trash.
        '''
        try:
            # get trash directory
            trashDir = cls.getPGLSettingsDir() / "trash"
            if not trashDir.exists():
                trashDir.mkdir(parents=True, exist_ok=True)
            # create path to move to trash
            trashPath = trashDir / filepath.name
            # check if it exists, if so, add a timestamp to the name
            if trashPath.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                trashPath = trashDir / f"{filepath.stem}_{timestamp}{filepath.suffix}"  
            # move to trash
            filepath.rename(trashPath)
            pglMessages.message(f"Moved {filepath} to trash: {trashPath}")
        except Exception as e:
            pglMessages.warning(f"Error moving {filepath} to trash: {e}")
            return None
        
    @staticmethod       
    def getPGLSettingsDir():
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
        settingsDir = cls.getPGLSettingsDir() / "settings"
        
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
        displayDir = cls.getPGLSettingsDir() / "displays"
        
        # append display specific directory if displaySettings is passed in
        if displaySettings is not None:
            # get a valid filename for displayName
            displayName = pglStorage.makeValidFilename(displaySettings.name)
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
    def getEyeTrackersDir(cls):
        """
        Get the directory where eye tracker settings are stored

        Returns:
            str: The directory path where eye tracker settingss are stored
        """
        # get the eyeTrackersDir
        eyeTrackersDir = cls.getPGLSettingsDir() / "eyetrackers"
        
        # check if it exists, create if not
        if not eyeTrackersDir.exists():
            try:
                eyeTrackersDir.mkdir(parents=True, exist_ok=True)
                pglMessages.message(f"Created directory: {eyeTrackersDir}")
            except Exception as e:
                pglMessages.warning(f"Error creating directory {eyeTrackersDir}: {e}")
                return None

        return eyeTrackersDir

    @classmethod
    def getCalibrationsDir(cls):
        """
        Get the directory where screen calibrations are stored

        Returns:
            str: The directory path where calibrations are stored
        """
        # get the screenSetttingsDir
        calibrationsDir = cls.getPGLSettingsDir() / "calibrations"
        
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
    def getSettings(cls, settingsName=None, settings=None, displayName=None, displaySettings=None, returnSettingsList=False):
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
            returnSettingsList (bool): If true, ignores other arguments and returns a list of all settings

        Returns:
            An instance of pglSettings
        """
        if returnSettingsList:
            settings=None
            settingsName=None
            displaySettings=None
            displayName=None
            
        if settings is not None:
            if not isinstance(settings, pglSettings):
                pglMessages.warning("Settings must be pglSettings")
                return
        elif settingsName is not None:
            # get the settings directory and create the full path to the settings file
            settingsDir = cls.getSettingsDir()
            settingsPath = Path(settingsDir) / pglStorage.makeValidFilename(settingsName)
            settingsPath = settingsPath.with_suffix(".json")
        
            # see if the file exists
            if not settingsPath.exists():
                pglMessages.warning(f"Settings file '{settingsPath}' not found.")
                return None
            else:
                pglMessages.message(f"Loading settings from '{settingsPath}'.")
                settings = pglSettings.load(filename=settingsPath)
        else:
            # use a default settings if we are trying to just open a display
            if displaySettings is not None or displayName is not None:
                settings = pglSettings()
            else:
                # get settings dir
                settingsDir = cls.getSettingsDir()
                
                # load all the seettings in there
                settingsList = []
                for filename in Path(settingsDir).glob("*.json"):
                    settings = pglSettings.load(filename=filename)
                    settingsList.append(settings)
                # if no saved settings, make a default one
                if not settingsList:
                    pglMessages.message("No saved settings found, creating default settings.")
                    settings = pglSettings()
                    settings.isDefault = True
                    settings.save()
                    settingsList = [settings]
                else:
                    # count the number of settings that have isDefault set
                    nDefaultSettings = sum(1 for s in settingsList if s.isDefault)
                    # find the default settings
                    defaultSettings = next((s for s in settingsList if s.isDefault), None)
                    if defaultSettings is not None:
                        settings = defaultSettings
                        # put it on top of list
                        settingsList.insert(0, settingsList.pop(settingsList.index(defaultSettings)))
                        # if there is more than one, then warn user and reset them
                        if nDefaultSettings > 1:
                            pglMessages.warning(f"Found {nDefaultSettings} set to default, keeping only first one")
                            for s in settingsList[1:]:
                                if s.isDefault:
                                    s.isDefault = False
                                    # save changes
                                    s.save()
                    else:
                        # no default, set to first vailable
                        settings = settingsList[0]
                        settings.isDefault = True
                        settings.save()
                        pglMessages.message(f"No default settings found, using first available settings: {settings.name}.")
                if returnSettingsList: return settingsList

                            
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
    
    ################################################################
    # getStencilSettingsDir
    ################################################################
    @classmethod
    def getStencilSettingsDir(cls):
        """Get the directory containing saved stencil presets."""
        stencilDir = cls.getPGLSettingsDir() / "stencils"
        stencilDir.mkdir(parents=True, exist_ok=True)
        return stencilDir

    ################################################################
    # getStencilSettings
    ################################################################
    @classmethod
    def getStencilSettings(cls, stencilName=None):
        """
        Load saved stencil presets.

        Args:
            stencilName (str or None): Load one named preset when supplied;
                otherwise return all presets.

        Returns:
            list, pglStencilSettings, or None: All presets, the named preset,
                or None when a requested preset cannot be loaded.
        """
        stencilDir = cls.getStencilSettingsDir()

        # Load one requested preset.
        if stencilName is not None:
            filename = stencilDir / f"{pglStorage.makeValidFilename(stencilName)}.json"

            if not filename.is_file():
                pglMessages.warning(f"Stencil preset '{stencilName}' was not found.")
                return None

            return pglStencilSettings.load(filename=filename)

        # Load all saved presets. An empty directory returns an empty list.
        settingsList = []
        for filename in sorted(stencilDir.glob("*.json")):
            settings = pglStencilSettings.load(filename=filename)

            if not isinstance(settings, pglStencilSettings):
                pglMessages.warning(f"Could not load stencil preset from '{filename}'.")
                continue

            settingsList.append(settings)

        return settingsList

    ################################################################
    # stencilSettings
    ################################################################
    @classmethod
    def stencilSettings(cls):
        """Edit named stencil presets using New, Copy, and Delete."""
        savedSettings = cls.getStencilSettings()
        original = pglStencilSettingsList(savedSettings)

        # The list editor requires one entry. When no presets exist, its
        # starter entry is only a draft and should not be saved on Cancel.
        modified = pglDialogs.traitsDialog(original)
        if modified is None: return

        # Preset names become filenames. Prevent duplicate destinations
        # from silently overwriting another preset.
        filenames = set()
        for settings in modified.settingsList:
            filename = pglStorage.makeValidFilename(settings.name)

            if not filename or filename.casefold() == "none":
                pglMessages.warning(
                    "Stencil presets need a nonempty filename-compatible name "
                    "other than 'None'. No presets were saved."
                )
                return

            if filename.casefold() in filenames:
                pglMessages.warning(
                    f"Stencil preset name '{settings.name}' conflicts with another "
                    "preset filename. Give each preset a distinct name. "
                    "No presets were saved."
                )
                return

            filenames.add(filename.casefold())

        # If the editor started with a draft, there were no original saved
        # entries. Mark it as new even if the user accepted it unchanged.
        if not savedSettings:
            original.settingsList = []

        cls._saveModifiedSettings(modified, original)


##################################################
# used for inheritence
##################################################
class pglTraitSettings(HasTraits, pglSerialize):
    
    # all trait setttings should have a name field and a uuid
    name = Unicode("default", help="", visible=False, enabled=False)
    uuid = Unicode("", help="Universal unique identifier for this setting", visible=False, enabled=False)
    isDefault = Bool(False, help="Whether this is the default settings", visible=False, enabled=False)
    isSelected = Bool(False, help="Whether this settings is selected when used for multi-selection", visible=False, enabled=False)
    pglVersion = Unicode("", help="version number of pgl", visible=False, enabled=False)
    github = Unicode("", help="github revision number", visible=False, enabled=False)
    version = Unicode("1.0", help="version number of this tratilet", visible=False, enabled=False)
    
    # old class names that are no longer being used, which if they were serialized
    # will now load back into this type
    _oldSerializationNames = ["pglSettingsEditable"]
    
    # default uuid
    @default("uuid")
    def _default_uuid(self):
        return str(uuid.uuid4())

    @default("pglVersion")
    def _default_pglVersion(self):
        return pglInfo.version()
    
    @default("github")
    def _default_github(self):
        return pglInfo.getRepoRevision()
    
    
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

    def print(self, all=False):
        """
        Print traitlet settings.

        Parameters
        ----------
        all : bool, default=False
            If True, print all traits, including traits inherited from
            pglTraitSettings. If False, only print traits defined by the
            child class.
        """
        print(f"{self.__class__.__name__}:")
        print("-" * 40)

        for name, trait in self._getOrderedTraits().items():

            # skip private/internal traits
            if name.startswith("_"):
                continue

            # Unless all=True, skip traits defined by pglTraitSettings
            if not all and name in pglTraitSettings.__dict__:
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
    
    def extractAsDataFrameColumn(self):
        """Return named scalar components for behavioral extraction.

        Subclasses may return a dictionary whose keys are camelCaps component
        names and whose values are numeric, Boolean, string, or None.

        The default explicitly marks this object as unsupported.
        This is an analysis projection, not a serialization method.
        """
        return "unsupported"
    
    ################################################################
    # validateSettings
    ################################################################
    def validateSettings(self):
        """
        Validate traits using member-function callbacks named in metadata.

        Example:
            polygonCoordinates = Unicode(
                "",
                validate="validatePolygonCoordinates"
            )

        The callback receives the current value and returns True when valid,
        False otherwise. Invalid values are reset to their trait defaults.

        Visits all nested settings, including unselected list entries.

        Returns:
            bool: True if no values needed resetting, False otherwise.
        """
        visited = set()
        allValid = True

        def validateValue(value, path):
            nonlocal allValid

            # Traverse settings and containers, not arbitrary runtime attributes.
            if not isinstance(value, (HasTraits, list, tuple, dict)): return

            objectID = id(value)
            if objectID in visited: return
            visited.add(objectID)

            if isinstance(value, HasTraits):
                settingsName = getattr(value, "name", "")
                objectPath = f"{path} ({settingsName})" if settingsName else path

                # Preserve definition order where available.
                getOrderedTraits = getattr(value, "_getOrderedTraits", None)
                traits = getOrderedTraits() if callable(getOrderedTraits) else value.traits()

                for traitName, trait in traits.items():
                    fieldPath = f"{objectPath}.{traitName}"
                    callbackName = trait.metadata.get("validate")
                    traitValue = getattr(value, traitName)

                    if callbackName:
                        # A missing or failing callback is also a validation
                        # failure, but must not interrupt the dialog.
                        try:
                            callback = getattr(value, callbackName, None)
                            valid = callable(callback) and bool(callback(traitValue))
                        except Exception:
                            valid = False

                        if not valid:
                            allValid = False

                            try:
                                # Copy mutable defaults so settings do not share
                                # lists or nested default objects.
                                defaultValue = deepcopy(value.trait_defaults(traitName))
                                setattr(value, traitName, defaultValue)
                                pglMessages.warning(
                                    f"{fieldPath}: invalid value; reset to its default."
                                )
                            except Exception:
                                pglMessages.warning(
                                    f"{fieldPath}: invalid value; could not restore its default."
                                )

                    # Read again because validation may have reset the trait.
                    validateValue(getattr(value, traitName), fieldPath)

            elif isinstance(value, dict):
                for key, child in value.items():
                    validateValue(child, f"{path}[{key!r}]")

            else:
                for index, child in enumerate(value):
                    validateValue(child, f"{path}[{index}]")

        validateValue(self, type(self).__name__)
        return allValid

##################################################
# Stencil settings
##################################################
class pglStencilSettings(pglTraitSettings):

    name = Unicode("New stencil", help="Name of this stencil preset")

    useCircle = Bool(False, help="Include a filled circle in this stencil")
    centerX = Float(0.0, help="Circle center x coordinate", validate="validateFiniteNumber")
    centerY = Float(0.0, help="Circle center y coordinate", validate="validateFiniteNumber")
    radius = Unicode(
        "y",
        help=(
            "Positive numeric radius, or x for half the screen width, "
            "or y for half the screen height."
        ),
        validate="validateRadius",
    )

    usePolygon = Bool(False, help="Include a filled polygon in this stencil")
    polygonCoordinates = Unicode(
        "[-x,-y], [x,-y], [x,y], [-x,y]",
        help=(
            "At least three coordinate pairs, for example: "
            "[0,1], [2,2], [5,10]. "
            "Use -x/x for the left/right edges and -y/y for the bottom/top edges. "
            "An enclosing list is also accepted. "
            "The polygon closes automatically."
        ),
        validate="validatePolygonCoordinates",
    )

    ################################################################
    # validateFiniteNumber
    ################################################################
    def validateFiniteNumber(self, value): return bool(np.isfinite(value))

    ################################################################
    # parseRadius
    ################################################################
    @staticmethod
    def parseRadius(value):
        """
        Parse a positive numeric radius or the exact tokens x and y.

        Returns:
            float, str, or None: Positive radius, symbolic token, or None
                when invalid. Symbols represent screen half-dimensions.
        """
        if not isinstance(value, str): return None
        value = value.strip()

        if value in ("x", "y"): return value

        try:
            radius = float(value)
            return radius if np.isfinite(radius) and radius > 0 else None
        except (ValueError, OverflowError):
            return None

    ################################################################
    # validateRadius
    ################################################################
    def validateRadius(self, value): return self.parseRadius(value) is not None

    ################################################################
    # parsePolygonCoordinates
    ################################################################
    @staticmethod
    def parsePolygonCoordinates(value):
        """
        Parse polygon text without executing expressions.

        Accepted examples:
            [0,1], [2,2], [5,10]
            [[0,1], [2,2], [5,10]]
            [-x,-y], [x,-y], [x,y], [-x,y]

        Each coordinate must be a finite number or one of:
            -x, x, -y, y

        Returns:
            list or None: Pairs containing floats or symbolic strings.
                Returns None when invalid.

        Checks coordinate structure, not self-intersections or other
        geometric properties. Symbolic values are resolved at drawing time.
        """
        if not isinstance(value, str) or not value.strip(): return None

        def parseCoordinate(node):
            # Accept numeric literals, but not booleans or quoted strings.
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                number = float(node.value)
                return number if np.isfinite(number) else None

            # Accept only the exact symbolic names x and y.
            if isinstance(node, ast.Name) and node.id in ("x", "y"):
                return node.id

            # Accept signed numeric literals and the edge tokens -x/-y.
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    if isinstance(node.operand, ast.Name) and node.operand.id in ("x", "y"):
                        return "-" + node.operand.id

                if isinstance(node.op, (ast.USub, ast.UAdd)):
                    operand = node.operand
                    if isinstance(operand, ast.Constant) and type(operand.value) in (int, float):
                        number = float(operand.value)
                        if isinstance(node.op, ast.USub): number = -number
                        return number if np.isfinite(number) else None

            # Reject arithmetic, calls, attributes, and all other expressions.
            return None

        try:
            # Parse syntax only. Never compile or evaluate this tree.
            vertices = ast.parse(value.strip(), mode="eval").body

            if not isinstance(vertices, (ast.List, ast.Tuple)) or len(vertices.elts) < 3:
                return None

            coordinates = []
            for vertex in vertices.elts:
                if not isinstance(vertex, (ast.List, ast.Tuple)) or len(vertex.elts) != 2:
                    return None

                pair = [parseCoordinate(node) for node in vertex.elts]
                if any(coordinate is None for coordinate in pair): return None
                coordinates.append(pair)

            return coordinates

        except Exception:
            return None

    ################################################################
    # validatePolygonCoordinates
    ################################################################
    def validatePolygonCoordinates(self, value):
        """
        Allow empty text only when the polygon is disabled.

        Nonempty text is validated even when the polygon is disabled.
        The general validator warns and restores defaults on failure.
        """
        if not isinstance(value, str): return False
        if not value.strip(): return not self.usePolygon
        return self.parsePolygonCoordinates(value) is not None
    
    ################################################################
    # save
    ################################################################
    def save(self, filename=None, filesystem=None, filesystemPrefix=None):
        """Save this stencil preset, using its name when no path is supplied."""
        if filename is None: filename = self.saveDir()
        return super().save(
            filename=filename,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
        )

    ################################################################
    # saveDir
    ################################################################
    def saveDir(self):
        """Return the JSON path for this named stencil preset."""
        return pglSettingsManager.getStencilSettingsDir() / f"{pglStorage.makeValidFilename(self.name)}.json"


##################################################
# Stencil settings dialog list
##################################################
class pglStencilSettingsList(pglTraitSettings):

    settingsList = List(
        Instance(pglStencilSettings),
        buttons=True,
        settingsListKey="name",
        traitDisplayName="Choose stencil",
        help="Add, copy, delete, and edit named stencil presets",
    )

    def __init__(self, settingsList=None):
        super().__init__()

        # The existing list editor requires at least one entry.
        self.settingsList = list(settingsList) if settingsList else [pglStencilSettings()]

##################################################
# Import settings
##################################################
class pglImportSettings(pglTraitSettings):

    name = Unicode("default", help="Display name for these import settings")
    experimentPath = Unicode("~/data", help="Root directory containing experiment data", isPath=True)
    cacheDir = Unicode("~/data/pgl/cache", isPath=True, help="Local directory for materialized import data, including deferred Flywheel MRI files. Cached files are not automatically deleted.")
    # The dialog sends transient input directly to storeFlywheelApiKey().
    # The actual key must never be assigned to this settings trait.
    flywheelApiKey = Unicode("", hasSetButton=True, buttonFunction="storeFlywheelApiKey", buttonLabel="Store in keychain", isPassword=True, transientInput=True,
        help=(
            "Enter the complete Flywheel API key, including its server hostname, "
            "then click Store in keychain to verify and save it securely. "
            "Unsubmitted input is discarded. "
            "Cancel does not undo a successful keychain write."
        ),
    )

    def save(self, filename=None, filesystem=None, filesystemPrefix=None):
        """Save import settings, using saveDir() when filename is omitted."""
        # Resolve the destination, then use the existing settings serializer.
        if filename is None: filename = self.saveDir()
        return super().save(filename=filename, filesystem=filesystem, filesystemPrefix=filesystemPrefix)

    def saveDir(self):
        return pglSettingsManager.getImportSettingsDir() / f"{pglStorage.makeValidFilename(self.name)}.json"

    @staticmethod
    def _flywheelCredentialStore():
        """Return a supported OS credential store, or None.

        Follows the PostgreSQL credential-store policy without depending
        on PostgreSQL. Rejects plaintext, encrypted-file, and unknown backends.
        """
        try:
            import keyring

            # Accept only the explicitly supported OS credential stores.
            backend = keyring.get_keyring()
            backendType = (type(backend).__module__, type(backend).__name__)
            supported = {
                ("keyring.backends.macOS", "Keyring"),
                ("keyring.backends.Windows", "WinVaultKeyring"),
                ("keyring.backends.SecretService", "Keyring"),
            }

            if backendType not in supported:
                raise RuntimeError("Unsupported credential backend.")

            return backend

        except Exception:
            # Do not expose backend exception details or use a fallback store.
            pglMessages.oneTimeWarning(
                "Secure Flywheel credential storage is unavailable. "
                "Install keyring and configure macOS Keychain, Windows "
                "Credential Manager, or Linux Secret Service. "
                "No fallback credential source will be used.",
                level=0,
            )
            return None

    def _flywheelCredentialIdentity(self):
        """Use the profile UUID so renaming settings preserves access."""
        # Validate and normalize the UUID before using it as a keychain account.
        try:
            profileID = str(uuid.UUID(self.uuid))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("Import settings require a valid UUID.") from None

        return "pgl.flywheel.v1", profileID

    def storeFlywheelApiKey(self, apiKey):
        """Verify and securely store transient dialog input.

        Never assigns the key to a settings trait.
        Failed verification leaves any previously saved key unchanged.
        Returns "" after successful storage so the dialog clears its input.
        Cancel does not undo a completed keychain write.
        """
        try:
            # Validate input without including the supplied value in errors.
            if not isinstance(apiKey, str) or not apiKey.strip():
                raise ValueError("Enter a nonempty Flywheel API key.")

            if any(character in apiKey for character in ("\n", "\r", "\x00")):
                raise ValueError("Flywheel API keys must contain no newline or NUL characters.")

            if apiKey != apiKey.strip():
                raise ValueError("Remove leading or trailing whitespace from the Flywheel API key.")

            # Resolve this profile's keychain entry and require secure storage.
            service, account = self._flywheelCredentialIdentity()
            store = self._flywheelCredentialStore()
            if store is None:
                raise RuntimeError("Secure Flywheel credential storage is unavailable.")

            # Load the SDK only when a Flywheel key is being configured.
            try:
                import flywheel
            except ImportError:
                raise RuntimeError("Install the Flywheel SDK using: pip install flywheel-sdk") from None

            # Verify only the supplied key. SDK exception details may contain
            # credentials or authenticated URLs, so replace them with safe text.
            client = None
            try:
                client = flywheel.Client(apiKey)
                user = client.get_current_user()
                if user is None:
                    raise RuntimeError("No authenticated Flywheel user returned.")
            except Exception:
                raise RuntimeError(
                    "Could not verify the Flywheel API key. "
                    "Check the complete key and your network connection."
                ) from None
            finally:
                # Release this reference; this is not secure memory erasure.
                client = None

            # Write to the keychain only after verification succeeds.
            try:
                store.set_password(service, account, apiKey)
            except Exception:
                raise RuntimeError("The Flywheel API key was verified but could not be saved securely.") from None

            # Returning an empty string instructs the dialog to clear its input.
            pglMessages.message("Flywheel API key verified and saved in the OS credential store.")
            return ""

        finally:
            # Clearing a local reference does not guarantee secure memory erasure.
            apiKey = None

    def _readSavedFlywheelApiKey(self):
        """Retrieve this profile's key without prompting for an API key.

        The OS may request permission to access its credential store.
        There is no environment-variable or file fallback.
        """
        # Resolve the same identity used when storing the key.
        service, account = self._flywheelCredentialIdentity()
        store = self._flywheelCredentialStore()
        if store is None:
            raise RuntimeError("Secure Flywheel credential storage is unavailable.")

        # Read only from the approved OS credential store.
        try:
            apiKey = store.get_password(service, account)
        except Exception:
            raise RuntimeError("Could not read the saved Flywheel API key from the OS credential store.") from None

        # Missing credentials must be configured through importSettings().
        if not isinstance(apiKey, str) or not apiKey:
            raise RuntimeError(
                "No Flywheel API key is saved for these import settings. "
                "Open importSettings(), select this profile, and use 'Store in keychain'."
            )

        return apiKey


##################################################
# Import settings dialog list
##################################################
class pglImportSettingsList(pglTraitSettings):

    settingsList = List(
        Instance(pglImportSettings),
        buttons=True,
        setDefault=True,
        settingsListKey="name",
        traitDisplayName="Choose import settings",
        help="List of import settings",
    )

    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None: self.settingsList = settingsList
        
class pglStateDataSettings(pglTraitSettings):
    '''
    Base class for classes like experiments, parameters, staircases etc which
    have settings, state and data
    '''
    
    @classmethod
    def load(cls, dataPath, filesystem=None, filesystemPrefix=None, loadAsClass=None):
        """Load pglStateDataSettings

        Args:
            filename (str or Path): Path to the JSON file. May include a protocol
                qualifier (e.g. 'ssh://...') when `filesystem` is None.
            filesystem (fsspec.AbstractFileSystem, optional): Filesystem to read
                through. If None, it is inferred from `filename` (falling back to
                a local filesystem). Defaults to None.
            filesystemPrefix (str): 'ssh://' like strings to qualify filename

        Returns:
            The loaded object, or None if it could not be loaded.
        """
        # Validate/resolve filesystem and normalise the path
        from .pglStorage import pglStorage
        filesystem, dataPath, _ = pglStorage.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix)
        if filesystem is None:
            pglMessages.warning(f"(pglSerialize) Could not resolve a filesystem for '{dataPath}'")
            return None
    
        # if there is a traits file then    
        if filesystem.exists(posixpath.join(dataPath, "traits.json")):
            # call parent class load (i.e. pglSerialize all the traits)
            obj = super().load(filename=posixpath.join(dataPath, "traits.json"), filesystem=filesystem, loadAsClass=loadAsClass)
            if obj is None:
                pglMessages.warning(f"Could not load traits for: {posixpath.join(dataPath, 'traits.json')}")
                return None
        else:
            try:
                obj = cls()
            except Exception as e:
                pglMessages.warning(f"Could not initialize {cls.__name__} using __init__ function")
                return
        
        # load settings
        try:
            obj.settings = pglTraitSettings.load(filename=posixpath.join(dataPath, "settings.json"), filesystem=filesystem)
        except Exception as e:
            pglMessages.warning(f"Could not load settings {posixpath.join(dataPath, 'settings.json')}: {e}")
        try:
            obj.state = pglTraitSettings.load(filename=posixpath.join(dataPath, "state.json"), filesystem=filesystem)
        except Exception as e:
            pglMessages.warning(f"Could not load state {posixpath.join(dataPath, 'state.json')}: {e}")
        try:    
            obj.data = pglTraitSettings.load(filename=posixpath.join(dataPath, "data.json"), filesystem=filesystem)
        except Exception as e:
            pglMessages.warning(f"Could not load data {posixpath.join(dataPath, 'data.json')}: {e}")
        return obj

    def save(self, dataPath, filesystem=None, filesystemPrefix=None):
        '''
        Save pglStateDataSettings

        Args:
            filename (str or Path): Path to the JSON file. May include a protocol
                qualifier (e.g. 'ssh://...') when `filesystem` is None.
            filesystem (fsspec.AbstractFileSystem, optional): Filesystem to read
                through. If None, it is inferred from `filename` (falling back to
                a local filesystem). Defaults to None.
            filesystemPrefix (str): 'ssh://' like strings to qualify filename
        '''
        # Validate/resolve filesystem and normalise the path
        from .pglStorage import pglStorage
        filesystem, dataPath, _ = pglStorage.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix, create=True)
        if filesystem is None:
            pglMessages.warning(f"(pglSerialize) Could not resolve a filesystem for '{dataPath}'")
            return None
        
        # call parent class save (i.e. pglSerialize all the traits), but only if it has traits
        # that are not just the default ones in pglTraitSettings 
        if set(self.traits()) != set(pglTraitSettings().traits()):
            super().save(filename=posixpath.join(dataPath, "traits.json"), filesystem=filesystem)
        
        # save settings
        settings = getattr(self, 'settings', None)
        if settings is None:
            pglMessages.warning(f"Instance of {type(self).__name__} does not have settings, nothing to save",level=1)
        else:
            settings.save(filename=posixpath.join(dataPath, "settings.json"), filesystem=filesystem)
            
        # save state
        state = getattr(self, 'state', None)
        if state is None:
            pglMessages.warning(f"Instance of {type(self).__name__} does not have state, nothing to save",level=1)
        else:
            state.save(filename=posixpath.join(dataPath, "state.json"), filesystem=filesystem)
            
        # save data
        data = getattr(self, 'data', None)
        if data is None:
            pglMessages.warning(f"Instance of {type(self).__name__} does not have data, nothing to save",level=1)
        else:
            data.save(filename=posixpath.join(dataPath, "data.json"), filesystem=filesystem)
            
    def saveBundleStrict(self, dataPath, filesystem=None, filesystemPrefix=None):
        """Save required components to a directory, propagating failures.

        Preserves the existing traits/settings/state/data layout.
        Does not provide atomic writes or remove partial output.
        """
        from .pglStorage import pglStorage

        # Resolve required values before creating the destination directory.
        components = {}
        for name in ("settings", "state", "data"):
            component = getattr(self, name, None)
            if component is None: raise ValueError(f"Cannot save {type(self).__name__}: missing {name}")
            if not callable(getattr(component, "saveStrict", None)):
                raise TypeError(f"Cannot save {type(self).__name__}.{name}: no saveStrict() method")
            components[name] = component

        filesystem, dataPath, _ = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            create=True,
        )
        if filesystem is None: raise ValueError(f"Could not resolve bundle destination: {dataPath!r}")

        # Match the existing bundle layout, including optional traits.json.
        if set(self.traits()) != set(pglTraitSettings().traits()):
            super().saveStrict(filename=posixpath.join(dataPath, "traits.json"), filesystem=filesystem)

        for name, component in components.items():
            component.saveStrict(filename=posixpath.join(dataPath, f"{name}.json"), filesystem=filesystem)

        return dataPath
            
##################################################
# pglItem
##################################################
class pglItem(pglTraitSettings):
    name = Unicode("",visible=False)    

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
    currentDisplayNum = Int(-1, help="Which display number this corresponds to. If not currently connected will be -1", enabled=False)
    gammaTableSize = Int(-1, help="Size of gamma table", enabled=False)
    currentDisplayMode = Tuple(Int(), Int(), Float(), labels=("width","height", "refreshRate"), default_value=(0,0,0), help="Current display mode (width, height, refreshRate)", enabled=False)
    isMain = Bool(False, help="Whether the display is the main display", enabled=False)
    isBuiltin = Bool(False, help="Whether the display is the built-in display of e.g. a laptop", enabled=False)
    displayDistance = Float(57, min=0.0, help="Distance from subject eyes to display in cm, used to calculate degress of visual angle")
    displaySize = Tuple(Float, Float, labels=("width","height"), default_value=(30, 20), help="Width and height of display in cm, used to calculate degrees of visual angle")
    flipLeftRight = Bool(False, help="Whether to flip the display left-right")
    flipUpDown = Bool(False, help="Whether to flip the display up-down")
    windowPosition = Tuple(Int(), Int(), labels=("x","y"), default_value=(100, 100), help="Position of window in pixels", visible=False)
    windowSize = Tuple(Int(), Int(), labels=("width","height"), default_value=(800, 600), help="Size of window in pixels", visible=False)
    displayModes = List(Instance(pglDisplayModeSettings), settingsListKey="modeName", hideKey=True, highlightSelector=False, traitDisplayName="pixelDims", help="All supported display modes")
    luminanceCalibration = List(Unicode(), hasPlotButton=True, buttonFunction="plotLuminanceCalibration", default_value=['None'], help="Which luminance calibration to use")
    temporalCalibration = List(Unicode(), hasPlotButton=True, buttonFunction="plotTemporalCalibration", default_value=['None'], help="Which temporal calibration to use")
    
    @classmethod
    def load(cls, filename, filesystem=None, filesystemPrefix=None):
        '''
        Load pglDisplaySettings. 
    
        Also loads displays list so that it has up to date display settings
        '''
        filesystem, filename, _ = pglStorage.validateFilesystem(filesystem=filesystem, dataPath=filename, filesystemPrefix=filesystemPrefix)
        
        # check if filename exists
        if not filesystem.exists(str(filename)):
            # look for it in the display directory
            filename = str(Path(pglSettingsManager.getDisplayDir()) / pglStorage.makeValidFilename(filename) / "display.json")
            if not filesystem.exists(filename):
                pglMessages.warning(f"File {filename} does not exist")
                return None
            
        # call super function to load all fields
        cls = super().load(filename=filename, filesystem=filesystem)
        
        return cls
    
    def save(self, filename=None, filesystem=None, filesystemPrefix=None):
        '''
        save
        
        Args:
            filename: Filename to save to, if ommitted, will generate path and filename using pglSettingsManager.getDisplayDir
        '''
        # get the filename
        if not filename: filename = self.saveDir()

        super().save(filename=filename, filesystem=filesystem, filesystemPrefix=filesystemPrefix)
        pglMessages.message(f"Saved display settings to {filename}")
                
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
        
        if self.luminanceCalibration == 'None':
            pglMessages.message(f"LuminanceCalibration set to None: No luminance calibrations found for {self.name}")
            return False
        else:
            calibration = self.getLuminanceCalibration()
            calibration.display(fig=fig)
        
        return True
    
    def getLuminanceCalibration(self, calibrationName=None):
        '''
        get the luminance calibration
        '''
        if not calibrationName:
            if not self.luminanceCalibration or self.luminanceCalibration[0] == 'None':
                return
            elif self.luminanceCalibration[0] == 'Latest':
                # get the latest calibration
                luminanceCalibrationDir = pglSettingsManager.getDisplayLuminanceCalibrationDir(self)
                pattern = re.compile(r'^\d{8}(_.*)?$')
                matches = [p for p in luminanceCalibrationDir.rglob('*') if p.is_dir() and pattern.match(p.name)]
                if not matches:
                    pglMessages.warning(f"Luminance calibration set to Latest: No luminance calibrations found for {self.name}")
                    return
                latestCalibrationDir = max(matches, key=lambda p: p.name)
                calibrationName = latestCalibrationDir.name
                pglMessages.message(f"Luminance calibration set to Latest: Using {calibrationName} for {self.name}")
            else:
                calibrationName = self.luminanceCalibration[0]
        
        # load the calibration
        from .pglCalibration import pglDisplayLuminanceCalibrationData
        luminanceCalibrationDir = pglSettingsManager.getDisplayLuminanceCalibrationDir(self) / calibrationName
        calibration = pglDisplayLuminanceCalibrationData.load(displayName=self.name, filepath=luminanceCalibrationDir)
        if calibration is None:
            pglMessages.warning(f"Could not load luminance calibration from {luminanceCalibrationDir}") 
            return
       
        return calibration
    
    def setGamma(self, pgl, gamma):
        '''
        Set the gamma using the calibration
        '''
        
        # get the current luminance calibration
        calibration = self.getLuminanceCalibration()
        
        # and set the gamma
        if calibration:
            calibration.setDisplayToGamma(pgl, self, gamma)
        
    def plotTemporalCalibration(self, fig, selected):
        '''
        load and plot the temporal calibration on the passed in axis
        '''
        if selected == "None":
            return False

        # load the calibration
        calibration = self.getTemporalCalibration()
        if calibration is None:
            pglMessages.warning(f"Could not load temporal calibration for {self.name}") 
            return False

        # display
        calibration.display(fig=fig)
        
        return True
    
    def getTemporalCalibration(self, calibrationName=None):
        '''
        get the temporal calibration
        '''
        if not calibrationName:
            if not self.temporalCalibration or self.temporalCalibration[0] == 'None':
                return
            elif self.temporalCalibration[0] == 'Latest':
                # get the latest calibration
                temporalCalibrationDir = pglSettingsManager.getDisplayTemporalCalibrationDir(self)
                pattern = re.compile(r'^\d{8}(_.*)?$')
                matches = [p for p in temporalCalibrationDir.rglob('*') if p.is_dir() and pattern.match(p.name)]
                if not matches:
                    pglMessages.warning(f"Temporal calibration set to Latest: No temporal calibrations found for {self.name}")
                    return
                latestCalibrationDir = max(matches, key=lambda p: p.name)
                calibrationName = latestCalibrationDir.name
                pglMessages.message(f"Temporal calibration set to Latest: Using {calibrationName} for {self.name}")
            else:
                calibrationName = self.temporalCalibration[0]
                print(f"Using temporal calibration: {calibrationName}")
           
        # load the calibration
        from .pglCalibration import pglDisplayTemporalCalibrationData
        filepath = pglSettingsManager.getDisplayTemporalCalibrationDir(self) / calibrationName
        calibration = pglDisplayTemporalCalibrationData.load(displayName=self.name, filepath=filepath)
        if calibration is None:
            pglMessages.warning(f"Could not load temporal calibration {filepath} from {temporalCalibrationDir}") 
            return
       
        return calibration
    

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
            pgl.cleanUp()
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
            if not e.state.runFinishedWithError:
                self.settingsList[0].print()
                self.settingsList[0].displayModes[0].print()
            else:
                # clean up windows just in case the window did no close properly
                pgl.cleanUp()


        except Exception as e:
            pglMessages.warning(f"Could not run test. Error {type(e).__name__}: {e}")    
            return

        
    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList

class pglDisplaySettingsWindowed(pglDisplaySettings):
    windowPosition = Tuple(Int(), Int(), labels=("x","y"), default_value=(0, 0), help="Position of window in pixels", visible=True)
    windowSize = Tuple(Int(), Int(), labels=("width","height"), default_value=(800, 600), help="Size of window in pixels", visible=True)
    displayModes = List(Instance(pglDisplayModeSettings), settingsListKey="modeName", hideKey=True, highlightSelector=False, traitDisplayName="pixelDims", help="All supported display modes", visible=False)

##################################################
# Settings 
##################################################
class pglSettings(pglTraitSettings):
    
    name = Unicode("default", help="Display name for these settings")
    displays = List(Instance(pglDisplaySettings), settingsListKey="name", highlightSelector=False, traitDisplayName="choose display", hideAll=True, help="Display - to edit display settings run pgl.displaySettings")
    calibrateForGamma = List(Float, default_value=[0, 1.0, 2.2], help="What gamma to target calibration for 0.0 = No calibration, 1.0=linear, 2.2 typical for images/movies")
    dataPath = Unicode("~/data",help="Path to data directory").tag(isPath=True)
    startKey = Unicode("space", allow_none=True, help="Key to start experiment")
    endKey = Unicode("escape", allow_none=True, help="Key to end experiment")
    volumeTriggerKey = Unicode("`", allow_none=True, help="Key press that signals scanner volume acquisition trigger")
    responseKeys = Unicode("1234", help="Keys used for subject responses. Can be a string like \"1234\" or a comma-separated list like 'left,right,up,down' and will map to response 0,1,2,etc")
    ignoreInitialVolumes = Int(0, min=0, step=1, help="Number of initial volumes to ignore")
    eatKeys = Bool(True, help="Whether to eat keypresses so they don't propagate to the OS. Will only eat the keys specified above.")
    startOnVolumeTrigger = Bool(False, help="Whether to start the experiment on the volume trigger key")
    startScreen = List(Unicode(), default_value=["fixation cross", "blank", "waiting message"],help="What to show on the screen before the experiment starts.")
    manualPreStart = Bool(False, help="Whether to manually start the experiment before the volume trigger")
    closeScreenOnEnd = Bool(True, help="Whether to close the screen when the experiment ends")
    verbose = Bool(False, help="Typically set to False so that only essential messages and warnings are printed during experiment run")
    saveAbortedRunsToTrash = Bool(True, help="When a run is detected as aborted, saves that run into trash of the session directory")
    backgroundColor = List(trait=Float(min=0.0, max=1.0), default_value=[0.5, 0.5, 0.5],minlen=3,maxlen=3,help="Background color as a list of RGB values").tag(isRGB=True)
    stencil = List(Instance(pglStencilSettings),settingsListKey="name",highlightSelector=False,hideAll=True,traitDisplayName="stencil",help="Stencil preset to use. Edit presets with pgl.stencilSettings(). None disables masking.")
    _digitalIO = List(Instance(pglItem), default_value=[pglItem(name='DATAPixx'),pglItem(name='LabJack')], settingsListKey="name", multiSelect=True, style="dropdown", traitDisplayName="digitalIO", help='Select which devices to use for digital IO',visible=True)
    _devices = List(Instance(pglItem), default_value=[pglItem(name='RESPONSEPixx')], settingsListKey="name", multiSelect=True, style="dropdown", traitDisplayName="devices", help='Select which devices to use for digital IO', visible=True)
    eyetracker = List(Unicode(), default_value=['None', 'Eyelink', 'TRACKPixx'], help="Eyetracker")
    
    def __init__(self, refreshDisplays=True):
        super().__init__()

        # Keep refreshDisplays=False free of environment discovery.
        # fromJSONdict() uses this when restoring recorded settings.
        if refreshDisplays:
            self.reloadDisplays()
            self.reloadStencils()
            
    @property
    def digitalIO(self):
        return [d.name for d in self._digitalIO if d.isSelected]

    @property
    def devices(self):
        return [d.name for d in self._devices if d.isSelected]

    @classmethod
    def load(cls, filename, filesystem=None, refresh=True):
        """Load settings; use refresh=False to preserve recorded values."""
        obj = super().load(filename=filename, filesystem=filesystem)
        if obj is None:
            pglMessages.warning(f"Could not load settings from {filename}")
            return None

        if refresh:
            obj.reloadDisplays()
            obj.reloadStencils()
            obj.reconcileDefaults('_digitalIO')
            obj.reconcileDefaults('_devices')
            obj.reconcileDefaults('eyetracker')

        return obj
    
    @default("username")
    def _defaultUsername(self):
        try:
            return getpass.getuser()
        except (ImportError, KeyError, OSError):
            pglMessages.warning("Could not determine the OS username. Set username in pglSettings before saving to the data store.", level=0)
            return ""
    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore saved values without discovering current displays."""
        obj = cls(refreshDisplays=False)
        obj.updateTraitsFromDict(data, filename=filename)
        return obj
    
    #classmethod
    def reconcileDefaults(self, traitName):
        """
        Append missing defaults and normalize capitalization.

        Supports string lists and object lists with settingsListKey metadata.
        Preserves loaded order, custom entries, and other object settings.
        """
        trait = self.traits().get(traitName)
        if not isinstance(trait, List):
            raise TypeError(f"{traitName!r} must be a List trait")

        # trait_defaults also supports defaults defined with @default.
        defaultValues = self.trait_defaults(traitName)
        loadedValues = list(getattr(self, traitName))
        keyName = trait.metadata.get("settingsListKey")

        def getName(item):
            name = getattr(item, keyName) if keyName else item
            if not isinstance(name, str):
                raise TypeError(
                    f"{traitName!r} must contain strings or objects "
                    "with a string settingsListKey"
                )
            return name

        defaultsByName = {
            getName(item).casefold(): item
            for item in defaultValues
        }

        # Correct capitalization without replacing loaded objects.
        loadedNames = set()

        for index, item in enumerate(loadedValues):
            foldedName = getName(item).casefold()
            loadedNames.add(foldedName)

            if foldedName in defaultsByName:
                defaultItem = defaultsByName[foldedName]

                if keyName:
                    defaultName = getName(defaultItem)
                    if getName(item) != defaultName:
                        setattr(item, keyName, defaultName)
                else:
                    loadedValues[index] = defaultItem

        # Copy missing defaults so mutable objects are not shared.
        for defaultItem in defaultValues:
            foldedName = getName(defaultItem).casefold()

            if foldedName not in loadedNames:
                loadedValues.append(deepcopy(defaultItem))
                loadedNames.add(foldedName)

        # Reassign rather than mutate the trait list in place.
        setattr(self, traitName, loadedValues)
        
    def save(self, filename=None, filesystem=None, filesystemPrefix=None):
        '''
        save
        
        Args:
            filename: Filename to save to, if ommitted, will generate path and filename using pglSettingsManager.getSettingsDir
        '''
        # get the filename
        if not filename: filename = self.saveDir()

        super().save(filename=filename, filesystem=filesystem, filesystemPrefix=filesystemPrefix)
        pglMessages.message(f"Saved settings to {filename}")
        
    def saveDir(self):
        return pglSettingsManager.getSettingsDir() / f"{pglStorage.makeValidFilename(self.name)}.json"

    
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
                
    ################################################################
    # Default stencil selection
    ################################################################
    @default("stencil")
    def _defaultStencil(self):
        # A fixed identity distinguishes the no-stencil option from presets.
        # Construct a fresh object for each settings instance.
        return [
            pglStencilSettings(
                name="None",
                uuid="pgl-stencil-none",
                useCircle=False,
                usePolygon=False,
            )
        ]

    ################################################################
    # reloadStencils
    ################################################################
    def reloadStencils(self):
        """
        Refresh available stencil presets while preserving the selection.

        Match by UUID so renaming a preset preserves its selection.
        If a selected preset was deleted, retain its saved definition rather
        than silently removing the experiment's mask.
        """
        # Preserve the currently selected preset.
        selected = self.stencil[0] if self.stencil else None

        # Build a fresh no-stencil option and load current saved presets.
        noneStencil = self._defaultStencil()[0]
        available = [noneStencil] + pglSettingsManager.getStencilSettings()

        if selected is None:
            self.stencil = available
            return

        # Refresh the selected definition from disk when it still exists.
        matchingIndex = next(
            (
                index
                for index, preset in enumerate(available)
                if preset.uuid == selected.uuid
            ),
            None,
        )

        if matchingIndex is not None:
            available.insert(0, available.pop(matchingIndex))
        else:
            # Keep the recorded definition available if its source was removed.
            available.insert(0, selected)
            pglMessages.warning(
                f"Stencil preset '{selected.name}' is no longer in the preset "
                "directory. Keeping its saved definition in these settings."
            )

        # Selected-first order is the dialog's existing selection convention.
        self.stencil = available
                        
class pglSettingsList(pglTraitSettings):

    settingsList = List(Instance(pglSettings), buttons=True, setDefault=True, settingsListKey="name", traitDisplayName="Choose settings", help="List of settings")
    buttons = [("Test", "testDisplay")]

    ##########################
    # test display settings
    ##########################
    def testDisplay(self):
        try:
            # imports (inside function to avoid circular imports)
            from .pglExperiment import pglExperiment
            from .pglTasks import pglTestTask
            from pgl import pgl
            
            # init pgl
            pgl = pgl()

            # init experiment
            e = pglExperiment(pgl, settings=self.settingsList[0])
                    
            # initialize task
            e.addTask(pglTestTask(pgl))
            
            # open screen
            e.initScreen()
            
            # and run
            e.run()
            
            if not e.state.runFinishedWithError:
                # print settings
                self.settingsList[0].print()
                self.settingsList[0].displays[0].print()
                self.settingsList[0].displays[0].displayModes[0].print()
            else:
                # clean up windows just in case the window did no close properly
                pgl.cleanUp()

        except Exception as e:
            pglMessages.warning(f"Could not run test. Error {type(e).__name__}: {e}")    
            return
        
    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList



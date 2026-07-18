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
from collections import OrderedDict
from curses import wrapper
from http.client import responses
from http.client import responses
from pathlib import Path
from urllib import response
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from fileinput import filename
from ipywidgets.widgets import widget
from traitlets import HasTraits, Float, Int, List, TraitError, Unicode, Dict, default, link, Bool, TraitType
import json
from functools import partial
from datetime import datetime
import numpy as np
import subprocess
import platform
import copy
from .pglBase import pglDisplayMessage
from .pglParameter import pglParameter, pglParameterBlock
from .pglSerialize import pglSerialize
from .pglDialog import pglSettingsEditable, pglTraitsDialog

displayDuration = 5  # seconds
#######################################
# Mixin class for pgl to provide settings management
#######################################
class pglMainSettingsManager:
    """
    Mixin class for pgl to provide settings management.
    """
    def __init__(self):   
        pass
    
    def settings(self):
        """
        Edit pgl settings. Brings up widget interface to edit settings
        """
        # initialize settings select class
        settingsSelect = pglSettingsSelect(self)
        settingsSelect.load()
        
        # display the settings
        settingsSelect.edit() 
        
        # display the selected settings
        settingsSelect.settings[0].edit() 
    
    def displaySettings(self):
        """
        Edit pgl display settings. Brings up widget interface to edit display settings
        """
        # initialize settings select class
        displaySettingsSelect = pglDisplaySettingsSelect(self)
        #displaySettingsSelect.load()
        
        # display the settings
        displaySettingsSelect.edit() 
        
        # display the selected settings
        pglTraitsDialog(displaySettingsSelect.displaySettings[0])
    
    def getDisplayNames(self, displayIndex=None):
        
                
        displayNames.append('Windowed')

        # get names from gpuInfo
        if not self.gpuInfo:
            return displayNames

        for gpuData in self.gpuInfo.values():
            displays = gpuData.get("Displays", [])
            for display in displays:
                name = f"{display.get('DisplayName', f'Display {len(displayNames)}')}: {display.get('Display Type', 'Unknown')}"
                if name:
                    displayNames.append(name)

        if displayIndex is not None:
            if displayIndex < len(displayNames) and displayIndex >= 0:
                # move the selected display to the top
                displayNames.insert(0, displayNames.pop(displayIndex))
            else:
                displayNames.insert(0, "Unknown Display")
        
        return displayNames
#######################################
# Mixin class for pglExperiment which manages pgl settings 
#######################################
class pglSettingsManager:
    """
    pglSettingsManager class for managing settings of pgl.
    """
    def __init__(self):   
        pass
    def getPGLDir(self):
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
    def getSettingsDir(self):
        """
        Get the directory where screen settings are stored.

        Returns:
            str: The directory path where settings are stored.
        """
        # get the settingsDir
        settingsDir = self.getPGLDir() / "settings"
        
        # check if it exists, create if not
        if not settingsDir.exists():
            try:
                settingsDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglSettings:getSettingsDir)</b> Created directory: {settingsDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglSettings:getSettingsDir)</b> Error creating directory {settingsDir}: {e}"))
                return None

        return settingsDir
    
    def getCalibrationsDir(self):
        """
        Get the directory where screen calibrations are stored

        Returns:
            str: The directory path where calibrations are stored
        """
        # get the screenSetttingsDir
        calibrationsDir = self.getPGLDir() / "calibrations"
        
        # check if it exists, create if not
        if not calibrationsDir.exists():
            try:
                calibrationsDir.mkdir(parents=True, exist_ok=True)
                display(HTML(f"<b>(pglScreenSettings:getCalibrationsDir)</b> Created directory: {calibrationsDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:getCalibrationsDir)</b> Error creating directory {calibrationsDir}: {e}"))
                return None

        return calibrationsDir

    def getSettings(self, settingsName=None):
        """
        Load settings from a JSON file and return an instance of the settings class.
        This will look in the directory returned by getSettingsDir().
        If you pass a settingsName, it will look for a file named {name}.json in that directory. 
        If you do not pass a name, it will search through all files to see which one has the field default set
        to true

        Args:
            settingsName (type): The name of the file to load

        Returns:
            An instance of the pglSettings with values loaded from the specified JSON file.
        """
        if settingsName is None:
            settingsName = "Default"
            
        # get the settings directory and create the full path to the settings file
        settingsDir = self.getSettingsDir()
        settingsPath = Path(settingsDir) / settingsName
        settingsPath = settingsPath.with_suffix(".json")
        
        # see if the file exists
        if not settingsPath.exists():
            print(f"(pglSettingsManager:loadSettings) Settings file '{settingsPath}' not found.")
            return None
        else:
            print(f"(pglSettingsManager:loadSettings) Loading settings from '{settingsPath}'.")
            return pglSettings(filename=settingsPath)

##################################################
# display Settings select
##################################################
class pglDisplaySettingsSelect(pglSettingsEditable):
    
    # traits that can be edited
    displayNames = List(Unicode(), help="Settings names")
    
    # init
    def __init__(self, pgl=None):
        # keep pgl
        self.pgl = pgl
        
        # close and clean up any open pgl screen
        self.pgl.close()
        self.pgl.cleanUp()
        
        # open pgl
        pgl.open(0)
        
        # get info
        info = pgl.info()
        
        # for each display gets its name
        self.displayNames = []
        self.displaySettings = []
        numDisplays = int(info.get('display.numDisplays',None))
        for iDisplay in range(numDisplays):
            # get the displayName
            displayName = info.get(f"display{iDisplay}.name","unknown")
            self.displayNames.append(displayName)
            # create a pglDisplaySettings
            self.displaySettings.append(pglDisplaySettings())
            self.displaySettings[-1].displayName = displayName
            self.displaySettings[-1].displayWidth = int(info.get(f"display{iDisplay}.widthPixels",0))
            self.displaySettings[-1].displayHeight = int(info.get(f"display{iDisplay}.heightPixels",0))
            self.displaySettings[-1].refreshRate = int(info.get(f"display{iDisplay}.refreshRate",0))
            self.displaySettings[-1].uuid = info.get(f"display{iDisplay}.uuid","")
            self.displaySettings[-1].vendor = int(info.get(f"display{iDisplay}.vendor",""))
            self.displaySettings[-1].model = int(info.get(f"display{iDisplay}.model",""))
            self.displaySettings[-1].serialNumber = int(info.get(f"display{iDisplay}.serialNumber",""))

        # close and clear the open/close output
        pgl.close()
        from IPython.display import clear_output
        clear_output(wait=True)
        
        super().__init__()
        #self.displaySettings[0].edit()
        pglTraitsDialog(self.displaySettings[0])
        
class pglDisplaySettings(pglSettingsEditable):
    
    displayName = List(Unicode(), help="Names of screen")
    displayWidth = Int(0, help="Display widh in pixels")
    displayHeight = Int(0, help="Display height in pixels")
    refreshRate = Int(0, help="Refresh rate")
    uuid = Unicode("", "UUID of display")
    vendor = Int(0, "Vendor number")
    model = Int(0, "Model number")
    serialNumber = Int(0, "Serial number")
    luminanceCalibration = List(Unicode(), default_value=['Latest','None'], help="Which calibration to use")

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

# Settings
class pglSettings(pglSettingsEditable):
    
    settingsName = Unicode(help="Display name for these settings")
    displayName = List(Unicode(), help="Names of available screens")
    displayNumber = Int(0, min=0, step=1, help="Screen number, 0 for window, 1 for main, 2 for secondary etc")
    windowWidth = Int(800, min=100, step=10, help="Window width in pixels")
    windowHeight = Int(600, min=100, step=10, help="Window height in pixels")
    displayDistance = Float(57.0, min = 0.0, step=0.1, max=None, help="Distance in cm from subject to screen")
    displayWidth = Float(32.0, min = 0.0, step=0.1, max=None, help="Display width in cm")
    displayHeight = Float(18.0, min = 0.0, step=0.1, max=None, help="Display height in cm")
    flipLeftRight = Bool(False, help="Whether to flip the display left-right")
    flipUpDown = Bool(False, help="Whether to flip the display up-down")
    calibration = List(Unicode(), default_value=['Latest','None'], help="Which calibration to use")
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
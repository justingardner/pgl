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
        Edit pgl settings. Brings up widgt interface to edit settings
        """
        # initialize settings select class
        settingsSelect = pglSettingsSelect(self)
        settingsSelect.load()
        
        # display the settings
        settingsSelect.edit() 
        
        # display the selected settings
        settingsSelect.settings[0].edit() 
    
    def getDisplayNames(self, displayIndex=None):
        displayNames = ['Windowed']

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

#############
# Main class which should be subclassed for specific settings,
# provides methods for loading/saving from JSON and displaying widgets
# to edit the settings
#############
class pglSettingsEditable(HasTraits, pglSerialize):
    def __init__(self, filename=None):
        # Initialize HasTraits
        super().__init__()
        # Load from file if provided
        if filename:
            print(f"(pglSettingsEditable:init) Loading settings from '{filename}'.")
            self.updateFromFile(filename)
    
   # display parameters
    def __repr__(self):
        traitValues = ", ".join(f"{key}={getattr(self, key)!r}" for key in self.trait_names())
        return f"{self.__class__.__name__}({traitValues})"
    
    # setup CSS
    def setupDisplayStyle(self):
        # --- Dark widget CSS ---
        display(HTML("""
        <style>
        .dark-widget-card {
            background-color: #000 !important;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 16px;
        }

        .dark-widget-card .widget-label {
            color: #eaeaea !important;
        }

        .dark-widget-card input,
        .dark-widget-card textarea,
        .dark-widget-card select {
            background-color: #111 !important;
            color: #eaeaea !important;
            border: 1px solid #444 !important;
        }

        .dark-widget-card .slider {
            background-color: #222 !important;
        }

        .dark-widget-card select option {
            background-color: #111 !important;
            color: #eaeaea !important;
        }

        .help-text, .options-panel {
            background-color: #111;
            color: #eaeaea;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 8px;
        }
        </style>
        """))
        
    # gets the traits in the order that they are defined
    @classmethod
    def getOrderedTraits(cls):
        """Return traits defined in this class (not inherited), in definition order."""
        ordered = OrderedDict()
        for name, obj in cls.__dict__.items():
            if isinstance(obj, TraitType):
                ordered[name] = obj
        return ordered

    def makeWidgets(self):
        """Automatically create widgets for all traits in self."""
        style = {'description_width': '120px'}
        widgetRows = []
        allHelpText = ""

        # Initialize widget map
        self.widgetMap = {}

        for traitName, trait in self.getOrderedTraits().items():
            if traitName.startswith('_'):
                continue  # skip private traits
            
            helpText = getattr(trait, 'help', f"{traitName}: float value")
            if helpText:
                allHelpText += f"<b>{traitName}:</b> {helpText}<br>"
            
            # Float with min/max
            if isinstance(trait, Float) and trait.min is not None and trait.max is not None:
                traitStep = getattr(trait, 'step', (trait.max - trait.min) / 100)
                slider = widgets.FloatSlider(
                    min=trait.min, max=trait.max, step=traitStep,
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='calc(100% - 100px)'),
                    tooltip=helpText
                )
                text = widgets.BoundedFloatText(
                    min=trait.min, max=trait.max, step=traitStep,
                    layout=widgets.Layout(width='100px'),
                    tooltip=helpText
                )
                link((self, traitName), (slider, 'value'))
                link((self, traitName), (text, 'value'))
                row = widgets.HBox([slider, text])
                widgetRows.append(row)
                self.widgetMap[traitName] = row

            # Float with min only
            elif isinstance(trait, Float) and trait.min is not None:
                wFloat = widgets.BoundedFloatText(
                    description=traitName,
                    min=trait.min,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wFloat, 'value'))
                widgetRows.append(wFloat)
                self.widgetMap[traitName] = wFloat

            # Float without min/max
            elif isinstance(trait, Float):
                wFloat = widgets.FloatText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wFloat, 'value'))
                widgetRows.append(wFloat)
                self.widgetMap[traitName] = wFloat

            # Int
            elif isinstance(trait, Int):
                wInt = widgets.IntText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wInt, 'value'))
                wInt.observe(partial(self.onIntSelect, traitName), names='value')
                widgetRows.append(wInt)
                self.widgetMap[traitName] = wInt
            # Bool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    value=getattr(self, traitName, False),
                    tooltip=helpText,
                    indent=False
                )
                wLabel = widgets.Label(value=traitName, style=style)
                wLabel.layout.width = '125px' 
                wBox = widgets.HBox([wLabel, wBool], layout=widgets.Layout(width='100%'))
                
                link((self, traitName), (wBool, 'value'))
                wBool.observe(partial(self.onBoolSelect, traitName), names='value')
                widgetRows.append(wBox)
                self.widgetMap[traitName] = wBool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText,
                    indent=False
                )
                link((self, traitName), (wBool, 'value'))
                wBool.observe(partial(self.onBoolSelect, traitName), names='value')
                widgetRows.append(wBool)
                self.widgetMap[traitName] = wBool
            # Path
            elif isinstance(trait, Unicode) and trait.metadata.get("isPath", False):
                wPath = widgets.Text(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wPath, 'value'))
                wPath.on_submit(partial(self.onPathSubmit, traitName=traitName))
                widgetRows.append(wPath)
                self.widgetMap[traitName] = wPath

            # Unicode
            elif isinstance(trait, Unicode):
                wText = widgets.Text(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wText, 'value'))
                widgetRows.append(wText)
                self.widgetMap[traitName] = wText

            # Bool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    description=traitName,
                    value=getattr(self, traitName),
                    tooltip=helpText
                )
                link((self, traitName), (wBool, 'value'))
                widgetRows.append(wBool)
                self.widgetMap[traitName] = wBool
            elif (isinstance(trait, List) and trait.metadata.get("isRGB", False)):
                # Create three float inputs for R, G, B
                r_input = widgets.BoundedFloatText(
                    value=getattr(self, traitName)[0] if getattr(self, traitName) else 0.0,
                    min=0.0, max=1.0, step=0.01,
                    description='R:',
                    style={'description_width': '20px'},
                    layout=widgets.Layout(width='120px'),
                    tooltip=f"{helpText} - Red channel"
                )
                g_input = widgets.BoundedFloatText(
                    value=getattr(self, traitName)[1] if getattr(self, traitName) else 0.0,
                    min=0.0, max=1.0, step=0.01,
                    description='G:',
                    style={'description_width': '20px'},
                    layout=widgets.Layout(width='120px'),
                    tooltip=f"{helpText} - Green channel"
                )
                b_input = widgets.BoundedFloatText(
                    value=getattr(self, traitName)[2] if getattr(self, traitName) else 0.0,
                    min=0.0, max=1.0, step=0.01,
                    description='B:',
                    style={'description_width': '20px'},
                    layout=widgets.Layout(width='120px'),
                    tooltip=f"{helpText} - Blue channel"
                )
                
                # Label for the RGB group
                label = widgets.Label(value=traitName, style=style)
                label.layout.width = '125px'
                
                # Combine in HBox
                rgb_box = widgets.HBox([label, r_input, g_input, b_input])
                
                # Update the trait when any input changes
                def update_rgb(change, name=traitName, inputs=(r_input, g_input, b_input)):
                    setattr(self, name, [inputs[0].value, inputs[1].value, inputs[2].value])
                
                r_input.observe(update_rgb, names='value')
                g_input.observe(update_rgb, names='value')
                b_input.observe(update_rgb, names='value')
                
                # Optional: Update inputs when trait changes externally
                def update_inputs(change, inputs=(r_input, g_input, b_input)):
                    if change['new'] and len(change['new']) == 3:
                        inputs[0].value = change['new'][0]
                        inputs[1].value = change['new'][1]
                        inputs[2].value = change['new'][2]
                
                self.observe(update_inputs, names=traitName)
                
                widgetRows.append(rgb_box)
                self.widgetMap[traitName] = rgb_box
    
            # List
            elif isinstance(trait, List):
                currentList = getattr(self, traitName)
                wDropdown = widgets.Dropdown(
                    options=currentList,
                    value=currentList[0] if currentList else None,
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wDropdown, 'options'))
                wDropdown.observe(partial(self.onListSelect, traitName), names='value')
                widgetRows.append(wDropdown)
                self.widgetMap[traitName] = wDropdown

        # Help widget
        helpWidget = widgets.HTML(allHelpText)
        helpWidget.layout.display = 'none'
        helpWidget.add_class("help-text")

        helpButton = widgets.Button(
            description="Show Help",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        helpButton.on_click(partial(self.toggleHelp, helpWidget=helpWidget))

        saveButton = widgets.Button(
            description="Save settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onSave'):
            saveButton.layout.display = 'none'
        else:
            saveButton.on_click(partial(self.onSave))

        testButton = widgets.Button(
            description="Test settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onTest'):
            testButton.layout.display = 'none'
        else:
            testButton.on_click(partial(self.onTest))

        deleteButton = widgets.Button(
            description="Delete settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onDelete'):
            deleteButton.layout.display = 'none'
        else:
            deleteButton.on_click(partial(self.onDelete))
        
        spacer = widgets.Box(layout=widgets.Layout(width="120px"))
        
        cancelButton = widgets.Button(
            description="Cancel",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        cancelButton.on_click(partial(self.onCancel))

        # Pack all widgets
        widgetDisplay = widgetRows + [
            widgets.HBox([
                cancelButton,
                deleteButton,
                saveButton,
                spacer,
                testButton,
                widgets.Box(layout=widgets.Layout(flex='1')),
                helpButton
            ]),
            helpWidget
        ]

        return widgetDisplay
    def onCancel(self, cancelButton):
        self.hide()
        pass
    
    def hide(self):
        """
        Hide the settings widget.
        """
        if hasattr(self, 'wrapper'):
            self.wrapper.layout.display = 'none'
        
    def toggleHelp(self, helpButton, helpWidget):
        helpWidget.layout.display = 'block' if helpWidget.layout.display == 'none' else 'none'

    def onIntSelect(self, traitName, change):
        pass
        
    def onBoolSelect(self, traitName, change):
        pass
        
    def onListSelect(self, traitName, change):
        # get the selected and currentList
        selected = change['new']
        currentList = list(getattr(self, traitName))

        # not in current list
        if selected not in currentList:
            return

        # Move selected item to top of list
        newList = [selected] + [x for x in currentList if x != selected]

        # now set to this newList
        setattr(self, traitName, newList)

    def onPathSubmit(self, textWidget, traitName):
        raw = textWidget.value
        try:
            path = Path(raw).expanduser()

            if not path.exists():
                # set the border to indicate no change
                textWidget.layout.border = '2px solid red'
            else:
                #self.dataPath = str(path)            
                textWidget.layout.border = '2px solid green'
        except Exception:
            textWidget.layout.border = "2px solid red"
            
            
    # ----- Put up edit dialog ---- #
    def edit(self):
        # setup css styles
        self.setupDisplayStyle()
        
        # make widgets for each parameter
        widgetDisplay = self.makeWidgets()
        
        # --- Container for widgets display ---
        widgetsBox = widgets.Box(
            widgetDisplay,
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                gap='10px',
                width='100%'
            )
        )
        widgetsBox.add_class("dark-widget-card")

        # --- Centering wrapper ---
        self.wrapper = widgets.Box(
            [widgetsBox],
            layout=widgets.Layout(
                display='flex',
                justify_content='center',
                width='95%',
                margin='0 auto'
            )   
        )

        # display
        display(self.wrapper)

class confirmationPanel:
    def __init__(self, confirmMessage="Confirm?", onConfirm=None, onCancel=None):
        """
        onConfirm: function called if user clicks Yes
        onCancel: function called if user clicks No
        """
        # Message
        self.label = widgets.HTML(f"<b>{confirmMessage}</b>")

        # Yes button (green)
        self.yesButton = widgets.Button(
            description="Yes",
            button_style="success",
            layout=widgets.Layout(width="80px")
        )
        self.yesButton.on_click(self._yes_clicked)

        # No button (red)
        self.noButton = widgets.Button(
            description="No",
            button_style="danger",
            layout=widgets.Layout(width="80px")
        )
        self.noButton.on_click(self._no_clicked)

        # Store callbacks
        self.onConfirm = onConfirm
        self.onCancel = onCancel

        # Pack the panel
        self.panel = widgets.VBox([
            self.label,
            widgets.HBox([self.yesButton, self.noButton])
        ])
        
        # Output to display result
        self.output = widgets.Output()

    def _yes_clicked(self, b):
        with self.output:
            self.output.clear_output()
        if self.onConfirm:
            self.onConfirm()
        self._hide_panel()

    def _no_clicked(self, b):
        with self.output:
            self.output.clear_output()
        if self.onCancel:
            self.onCancel()
        self._hide_panel()

    def _hide_panel(self):
        # Hide the panel widgets
        self.panel.layout.display = 'none'

    def display(self):
        display(self.panel, self.output)
       
# Screen settings select
class pglSettingsSelect(pglSettingsEditable):
    
    # traits that can be edited
    settingsNames = List(Unicode(), help="Settings names")
    
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
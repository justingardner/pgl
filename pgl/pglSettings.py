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
from pathlib import Path
from IPython.display import display, HTML
import ipywidgets as widgets
from fileinput import filename
from ipywidgets.widgets import widget
from traitlets import HasTraits, Float, Int, List, TraitError, Unicode, Dict, default, link, Bool, TraitType
import json
from functools import partial
from datetime import datetime
import numpy as np
import pgl as pgl
import subprocess
import platform

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
        # initialize experiment so that we can get settings from it
        from .pglExperiment import pglExperiment
        e = pglExperiment(self, suppressInitScreen=True)
        
        # get the screen settings directory
        screenSettingsDir = e.getScreenSettingsDir()
        
        # cycle through all files in screenSettingsDir with .json extension
        # and load as a pglScreenSettings instance
        screenSettings = []
        for jsonFile in Path(screenSettingsDir).glob("*.json"):
            screenSettings.append(pglScreenSettings(jsonFile))

        # setup screenSettingsSelect class
        # to select a screen settings file
        screenSettingsSelect = pglScreenSettingsSelect()
        screenSettingsSelect.displayNames = [s.displayName for s in screenSettings]
        screenSettingsSelect.settings = screenSettings
        
        # display the settings
        screenSettingsSelect.edit() 
        
        # dispaly the selected settings
        screenSettings[0].edit() 
        
       

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
                display(HTML(f"<b>(pglScreenSettings:onSave)</b> Created directory: {settingsDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:onSave)</b> Error creating directory {settingsDir}: {e}"))
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
                display(HTML(f"<b>(pglScreenSettings:onSave)</b> Created directory: {calibrationsDir}"))
            except Exception as e:
                display(HTML(f"<b>(pglScreenSettings:onSave)</b> Error creating directory {calibrationsDir}: {e}"))
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

# Custom JSON encoder and decoder to handle special Instance
# traitlets of settings classs, so they can be serialized to JSON
class customJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            # Mark tuples specially so we can reconstruct them
            return {
                '__tuple__': True,
                'items': list(obj)
            }
        if isinstance(obj, HasTraits):
            # Store class name for reconstruction
            return {
                '__class__': obj.__class__.__name__,
                '__module__': obj.__class__.__module__,
                **{key: getattr(obj, key) for key in obj.trait_names() if not key.startswith('_')}
            }
        return super().default(obj)

class customJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
    
    def object_hook(self, obj):
        # Reconstruct tuples
        if isinstance(obj, dict) and obj.get('__tuple__'):
            items = obj['items']
            # Convert lists back to numpy arrays if needed
            converted_items = []
            for item in items:
                if isinstance(item, list):
                    try:
                        converted_items.append(np.array(item))
                    except:
                        converted_items.append(item)
                else:
                    converted_items.append(item)
            return tuple(converted_items)
        
        # Reconstruct HasTraits objects
        if '__class__' in obj and '__module__' in obj:
            try:
                import importlib
                module = importlib.import_module(obj['__module__'])
                cls = getattr(module, obj['__class__'])
                
                # Remove metadata keys
                data = {k: v for k, v in obj.items() if not k.startswith('__')}
                
                # Create instance and set attributes
                instance = cls()
                for key, value in data.items():
                    try:
                        setattr(instance, key, value)
                    except:
                        pass  # Skip if trait validation fails
                return instance
            except Exception as e:
                print(f"(pglSettings) Could not reconstruct {obj.get('__class__')}: {e}")
                return obj
        
        # Convert datetime strings and lists to numpy arrays
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    obj[key] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    pass
            # Don't auto-convert lists here since tuples handle it
        return obj

#############
# Main class which should be subclassed for specific settings,
# provides methods for loading/saving from JSON and displaying widgets
# to edit the settings
#############
class _pglSettings(HasTraits):
    def __init__(self, filename=None):
        # Initialize settings from a file
        if filename:
            self.load(filename)
        
    # Serialize to JSON file
    def save(self, filename):
        try:
            filename = Path(filename).with_suffix(".json")
            data = {key: getattr(self, key) for key in self.getOrderedTraits().keys()}
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4, cls=customJSONEncoder)
            print(f"(pglSettings) Saved settings to '{filename}'")
        except PermissionError:
            print(f"(pglSettings) No permission to write to '{filename}'")
        except IsADirectoryError:
            print(f"(pglSettings) '{filename}' is a directory, cannot write file")
        except OSError as e:
            print(f"(pglSettings) OS error while saving '{filename}': {e}")
        except Exception as e:
            print(f"(pglSettings) Unknown error ({type(e).__name__}) while saving '{filename}': {e}")    
      
    # load settings from JSON file      
    def load(self, filename):
        # --- Open JSON file ---
        try:
            filename = Path(filename).with_suffix(".json")
            with open(filename, 'r') as f:
                data = json.load(f, cls=customJSONDecoder)
        except FileNotFoundError:
            print(f"(pglSettings) File '{filename}' not found.")
        except PermissionError:
            print(f"(pglSettings) No permission to read '{filename}'.")
        except IsADirectoryError:
            print(f"(pglSettings) '{filename}' is a directory, not a file.")
        except OSError as e:
            print(f"(pglSettings) Error accessing '{filename}': {e}")
        # --- JSON formatting errors ---
        except json.JSONDecodeError as e:
            print(f"(pglSettings) Error decoding JSON in '{filename}': {e}")
        # --- Catch any other unknown errors ---
        except Exception as e:
            print(f"(pglSettings) Unknown error ({type(e).__name__}) while reading '{filename}': {e}")
        
        # --- Update traits from JSON data ---
        for key in self.trait_names():
            # if key is in the json file
            if key in data:
                try:
                    # set the attribute value
                    setattr(self, key, data[key])
                except TraitError:
                    # Handle trait type mismatch
                    trait = self.traits()[key]
                    expectedType = trait.__class__
                    gotType = type(data[key])
                    print(f"(pglSettings) '{key}' has wrong type in JSON (expected {expectedType.__name__}, got {gotType.__name__}), using default {getattr(self, key)}")
                except Exception as e:
                    # Handle any other errors
                    print(f"(pglSettings) Error: {e} Using default value for {key}: {getattr(self, key)}. ")
            # if not in the json file, use default value
            else:
                print(f"(pglSettings) '{key}' not found in JSON, using default {getattr(self, key)}")
        # keys in JSON that are not traits
        extraKeys = set(data.keys()) - set(self.trait_names())
        if extraKeys:
            print(f"(pglSettings) Did not load unknown keys from JSON: {list(extraKeys)}")
    
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

        for traitName, trait in self.getOrderedTraits().items():
            if traitName.startswith('_'):
                continue  # skip private traits
            
            # get helpText and add to allHelpText
            helpText = getattr(trait, 'help', f"{traitName}: float value")
            if helpText:
                allHelpText += f"<b>{traitName}:</b> {helpText}<br>"
            
            # Float
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
                # link slider and text to trait
                link((self, traitName), (slider, 'value'))
                link((self, traitName), (text, 'value'))
                # add as a row to the widgets
                widgetRows.append(widgets.HBox([slider, text]))

            elif isinstance(trait, Float) and trait.min is not None:
                wFloat = widgets.BoundedFloatText(
                    description=traitName,
                    min=trait.min,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                # traitlets will enforce min when the trait is updated
                link((self, traitName), (wFloat, 'value'))

                # add it to widgets
                widgetRows.append(wFloat)

            elif isinstance(trait, Float):
                wFloat = widgets.FloatText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                # traitlets will enforce min when the trait is updated
                link((self, traitName), (wFloat, 'value'))

                # add it to widgets
                widgetRows.append(wFloat)
            
            # Int
            elif isinstance(trait, Int):
                wInt = widgets.IntText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wInt, 'value'))
                widgetRows.append(wInt)

            elif isinstance(trait, Unicode) and trait.metadata.get("isPath", False):
                wPath = widgets.Text(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                # link the widget with the trait
                link((self, traitName), (wPath, 'value'))
                
                # set on_submit for path changes
                wPath.on_submit(partial(self.onPathSubmit, traitName=traitName))
                
                #append the path widget
                widgetRows.append(wPath)

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
            
            # Bool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    description=traitName,
                    value=getattr(self, traitName),
                    tooltip=helpText
                )
                link((self, traitName), (wBool, 'value'))
                widgetRows.append(wBool)
            # List
            elif isinstance(trait, List):
                currentList = getattr(self, traitName)
                wDropdown = widgets.Dropdown(
                    options=currentList,
                    value=currentList[0],
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                #link((self, traitName), (wDropdown, 'value'))
                wDropdown.observe(partial(self.onListSelect, traitName), names='value')
                widgetRows.append(wDropdown)
        # make helpWidget
        helpWidget = widgets.HTML(allHelpText)
        helpWidget.layout.display = 'none'  # hidden by default
        helpWidget.add_class("help-text")

        # --- Button to toggle help display ---
        helpButton = widgets.Button(
            description="Show Help",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        helpButton.on_click(partial(self.toggleHelp, helpWidget=helpWidget))
        
        # --- Button for save ---
        saveButton = widgets.Button(
            description="Save settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onSave'):
            saveButton.layout.display = 'none'
        else:
            saveButton.on_click(partial(self.onSave))
        
        
        # --- Button for test ---
        testButton = widgets.Button(
            description="Test settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        
        # if there is no onTest method defined by the subclass, hide the test button, otherwise link it to the onTest method
        if not hasattr(self, 'onTest'):
            testButton.layout.display = 'none'
        else:
            testButton.on_click(partial(self.onTest))
            

        # --- Button for cancel ---
        cancelButton = widgets.Button(
            description="Cancel",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        cancelButton.on_click(partial(self.onCancel))

        # pack up widgetRows, buttons and help
        widgetDisplay = widgetRows + [widgets.HBox([cancelButton, saveButton, testButton, widgets.Box(layout=widgets.Layout(flex='1')), helpButton]), helpWidget]

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

# Screen settings select
class pglScreenSettingsSelect(_pglSettings):
    
    # traits that can be edited
    displayNames = List(Unicode(), help="Display names")
    
    # Variable containing all the settings, this is set by calling class
    settings = []

    # when the displayName is selected, edit those settings
    def onListSelect(self, traitName, change):
        # call parent method
        super().onListSelect(traitName, change)
        # load the selected settings
        selectedName = change['new']
        # go through settings, to see which one it matches to
        for s in self.settings:
            if s.displayName == selectedName:
                # display the settings
                s.edit()
            else:
                s.hide()

# Screen settings
class pglSettings(_pglSettings):
    
    settingsName = Unicode(help="Display name for these settings")
    screenNumber = Int(0, min=0, max=2, step=1, help="Screen number, 0 for window, 1 for main, 2 for secondary etc")
    displayDistance = Float(57.0, min = 0.0, step=0.1, max=None, help="Distance in cm from subject to screen")
    displayWidth = Float(32.0, min = 0.0, step=0.1, max=None, help="Display width in cm")
    displayHeight = Float(18.0, min = 0.0, step=0.1, max=None, help="Display height in cm")
    dataPath = Unicode("~/data",help="Path to data directory").tag(isPath=True)
    
    def onSave(self, saveButton):
        # get the screenSetttingsDir
        screenSettingsDir = Path.home() / ".pgl" / "screenSettings"
        
        # get the name 
        settingsFilename = screenSettingsDir / self.displayName
        settingsFilename = settingsFilename.with_suffix(".json")
        
        # save it
        self.save(settingsFilename)
        display(HTML(f"<b>Saved settings to:</b> {settingsFilename}"))

    def onTest(self, testButton):
        from pgl import pgl, pglExperiment
        pgl = pgl()
        e = pglExperiment(pgl, suppressInitScreen=True)
        # print HTML
        display(HTML(f"<b>(pglSettings:onTest)</b> Testing settings: {self.settingsName}"))
        e.initScreen(settings = self)
        e.pgl.bullseye()
        e.pgl.flush()
        e.pgl.waitSecs(10)
        e.pgl.close()

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
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
from IPython.display import display, HTML
import ipywidgets as widgets
from fileinput import filename
from traitlets import HasTraits, Float, Int, TraitError, Unicode, Dict, link
import json


#############
# Main class
#############
class pglSettings(HasTraits):
    def __init__(self, filename=None):
        # Initialize settings from a file
        if filename:
            self.load(filename)
        
    # Serialize to JSON file
    def save(self, filename):
        try:
            data = {key: getattr(self, key) for key in self.trait_names()}
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
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
            with open(filename, 'r') as f:
                data = json.load(f)
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
    
    def makeWidgets(self):
        """Automatically create widgets for all traits in self."""
        style = {'description_width': '120px'}
        widgetRows = []

        for traitName, trait in self.traits().items():
            if traitName.startswith('_'):
                continue  # skip private traits

            # Float
            if isinstance(trait, Float):
                slider = widgets.FloatSlider(
                    min=0, max=1, step=0.01,
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='calc(100% - 100px)'),
                    tooltip=f"{traitName}: float value"
                )
                text = widgets.BoundedFloatText(
                    min=0, max=1, step=0.01,
                    layout=widgets.Layout(width='100px'),
                    tooltip=f"{traitName}: float value"
                )
                link((self, traitName), (slider, 'value'))
                link((self, traitName), (text, 'value'))
                widgetRows.append(widgets.HBox([slider, text]))

            # Int
            elif isinstance(trait, Int):
                wInt = widgets.IntText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=f"{traitName}: integer value"
                )
                link((self, traitName), (wInt, 'value'))
                widgetRows.append(wInt)

            # Unicode
            elif isinstance(trait, Unicode):
                wText = widgets.Text(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=f"{traitName}: text value"
                )
                link((self, traitName), (wText, 'value'))
                widgetRows.append(wText)

            # Bool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    description=traitName,
                    value=getattr(self, traitName),
                    tooltip=f"{traitName}: boolean value"
                )
                link((self, traitName), (wBool, 'value'))
                widgetRows.append(wBool)
        return widgetRows



    def edit(self):
        # setup css styles
        self.setupDisplayStyle()
        # make widgets for each parameter
        widgetRows = self.makeWidgets()
        # --- Container for widgets display ---
        widgetsBox = widgets.Box(
            widgetRows,
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                gap='10px',
                width='100%'
            )
        )
        widgetsBox.add_class("dark-widget-card")

        # --- Centering wrapper ---
        wrapper = widgets.Box(
            [widgetsBox],
            layout=widgets.Layout(
                display='flex',
                justify_content='center',
                width='95%',
                margin='0 auto'
            )   
        )

        display(wrapper)




# Screen settings
class pglScreenSettings(pglSettings):
    alpha = Float(0.5)
    n_iter = Int(100)
    name = Unicode("test")
    options = Unicode("Option 1")
    distance = Float(1.0)
    width = Float(10.0)
    height = Float(5.0)
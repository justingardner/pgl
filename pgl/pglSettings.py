################################################################
#   filename: pglSettings.py
#    purpose: Provides settings management for pgl
#         by: JLG
#       date: Feb 6, 2026
################################################################

#############
# Import
#############
from fileinput import filename
from traitlets import HasTraits, Float, Int, TraitError, Unicode, Dict
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
    
    def __repr__(self):
        traitValues = ", ".join(f"{key}={getattr(self, key)!r}" for key in self.trait_names())
        return f"{self.__class__.__name__}({traitValues})"


# Screen settings
class pglScreenSettings(pglSettings):
    alpha = Float(0.5)
    n_iter = Int(100)
    name = Unicode("test")
    options = Unicode("Option 1")
    distance = Float(1.0)
    width = Float(10.0)
    height = Float(5.0)
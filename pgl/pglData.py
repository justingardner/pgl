################################################################
#   filename: pglData.py
#    purpose: Classes that handle serializing and deserializing data
#         by: JLG
#       date: Feb 18, 2026
################################################################

##############
# Imports
##############
from pathlib import Path
from dataclasses import fields, dataclass
import json
import numpy as np
from typing import Any

##############
# pglData base class
##############
class pglData:
    '''
    Base class for pglData objects.
    Provides JSON save/load functionality with support for numpy arrays and Paths.
    '''

    #################
    # JSON encoder
    #################
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj: Any):
            # Convert numpy arrays to dict for exact round-trip
            if isinstance(obj, np.ndarray):
                return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype), "shape": obj.shape}
            # Convert Path objects to strings
            if isinstance(obj, Path):
                return str(obj)
            # Fallback
            return super().default(obj)

    #################
    # JSON decoder
    #################
    @staticmethod
    def JSONDecoder(dct):
        # Detect serialized numpy arrays
        if "__ndarray__" in dct:
            return np.array(dct["__ndarray__"], dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    #################
    # Save to JSON
    #################
    def save(self, filename):
        try:
            filename = Path(filename).with_suffix(".json")
            data = {f.name: getattr(self, f.name) for f in fields(self)}
            with open(filename, "w") as f:
                json.dump(data, f, indent=4, cls=self.JSONEncoder)
        except PermissionError:
            print(f"(pglData) No permission to write to '{filename}'")
        except IsADirectoryError:
            print(f"(pglData) '{filename}' is a directory, cannot write file")
        except OSError as e:
            print(f"(pglData) OS error while saving '{filename}': {e}")
        except Exception as e:
            print(f"(pglData) Unknown error ({type(e).__name__}) while saving '{filename}': {e}")

    #################
    # Load from JSON
    #################
    def load(self, filename):
        try:
            filename = Path(filename).with_suffix(".json")
            with open(filename, "r") as f:
                data = json.load(f, object_hook=self.JSONDecoder)

            # restore attributes in field order
            for f in fields(self):
                if f.name in data:
                    value = data[f.name]
                    # convert back to Path if field type is Path and value is str
                    if f.type == Path and isinstance(value, str):
                        value = Path(value)
                    setattr(self, f.name, value)

        except FileNotFoundError:
            print(f"(pglData) File '{filename}' not found.")
        except PermissionError:
            print(f"(pglData) No permission to read '{filename}'.")
        except IsADirectoryError:
            print(f"(pglData) '{filename}' is a directory, not a file.")
        except OSError as e:
            print(f"(pglData) Error accessing '{filename}': {e}")
        except json.JSONDecodeError as e:
            print(f"(pglData) Error decoding JSON in '{filename}': {e}")
        except Exception as e:
            print(f"(pglData) Unknown error ({type(e).__name__}) while reading '{filename}': {e}")

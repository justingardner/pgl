################################################################
#   filename: pglSerialize.py
#    purpose: Provides parent class to serialize and deserialize
#             classes. This can be mixed in, and custom methods
#             can be provided to handle JSON encoding and decoding
#         by: JLG
#       date: Feb 22, 2026
################################################################

##########################
# Imports for pglSerialize
##########################
from dataclasses import fields, is_dataclass
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from traitlets import HasTraits, TraitError

##########################
# pglSerialize class
##########################
class pglSerialize:
    ##########################
    # Save to JSON file
    ##########################
    def save(self, filename):
        """Save object to JSON file"""
        print(f"(pglSerialize) Saving {self.__class__.__name__} to '{filename}'")
        try:
            filename = Path(filename).with_suffix(".json")
            with open(filename, 'w') as f:
                f.write(self.toJSON())
        except PermissionError:
            print(f"(pglSerialize) No permission to write to '{filename}'")
        except IsADirectoryError:
            print(f"(pglSerialize) '{filename}' is a directory, cannot write file")
        except OSError as e:
            print(f"(pglSerialize) OS error while saving '{filename}': {e}")
        except Exception as e:
            print(f"(pglSerialize) Unknown error ({type(e).__name__}) while saving '{filename}': {e}")
    
    ##########################
    # Load from JSON file
    ##########################
    def load(self, filename):
        """Load object from JSON file"""
        try:
            filename = Path(filename).with_suffix(".json")
            with open(filename, 'r') as f:
                jsonString = f.read()
            
            # Deserialize using pglSerialize
            data = pglSerialize.fromJSON(jsonString)
            
            # If it's the same class as self, copy its attributes
            if isinstance(data, self.__class__):
                self.copyTraitsFrom(data)
                return
            
            # Otherwise treat as dict
            if isinstance(data, dict):
                self.updateTraitsFromDict(data, str(filename))
            else:
                print(f"(pglSerialize) Invalid data format in '{filename}'")
                
        except FileNotFoundError:
            print(f"(pglSerialize) File '{filename}' not found.")
        except PermissionError:
            print(f"(pglSerialize) No permission to read '{filename}'.")
        except IsADirectoryError:
            print(f"(pglSerialize) '{filename}' is a directory, not a file.")
        except OSError as e:
            print(f"(pglSerialize) Error accessing '{filename}': {e}")
        except Exception as e:
            print(f"(pglSerialize) Error reading '{filename}': {e}")

    ##########################
    # toJSON
    ##########################
    def toJSON(self, type="all"):
        def encodeObject(o):
            # Custom encoding for pglSerialize objects
            if isinstance(o, pglSerialize):
                return {
                    '__class__': o.__class__.__name__,
                    **o.toJSONdict(type)
                }
            
            # Handle datetime objects
            elif isinstance(o, datetime):
                return {
                    '__datetime__': True,
                    'value': o.isoformat()
                }
            
            # Handle numpy arrays
            elif isinstance(o, np.ndarray):
                return {
                    '__numpy__': True,
                    'dtype': str(o.dtype),
                    'shape': o.shape,
                    'data': o.tolist()
                }
            
            # Handle numpy scalar types
            elif isinstance(o, (np.integer, np.floating)):
                return o.item()
            
            # Handle tuples
            elif isinstance(o, tuple):
                return {
                    '__tuple__': True,
                    'items': [encodeObject(item) for item in o]
                }
            
            # Handle HasTraits objects (non-pglSerialize)
            elif isinstance(o, HasTraits) and not isinstance(o, pglSerialize):
                return {
                    '__hastraits__': True,
                    '__class__': o.__class__.__name__,
                    '__module__': o.__class__.__module__,
                    **{key: encodeObject(getattr(o, key)) 
                       for key in o.trait_names() if not key.startswith('_')}
                }
            
            # Recursively handle lists
            elif isinstance(o, list):
                return [encodeObject(item) for item in o]
            
            # Recursively handle dicts
            elif isinstance(o, dict):
                return {k: encodeObject(v) for k, v in o.items()}
            
            # Default handling
            return o.__dict__ if hasattr(o, '__dict__') else str(o)
        
        # dump to JSON string using custom encoder defined above
        return json.dumps(self, default=encodeObject, sort_keys=True, indent=4)
    
    ##########################
    # toJSONdict
    ##########################
    def toJSONdict(self, type="all"):
        # Convert object to dict for JSON serialization
        # Override in subclasses to control which attributes are included
        
        # If this is a HasTraits object, use trait_names()
        if isinstance(self, HasTraits):
            return {key: getattr(self, key) 
                    for key in self.trait_names() 
                    if not key.startswith('_')}
        
        # If this is a dataclass, manually build dict (preserves nested objects)
        if is_dataclass(self):
            return {
                f.name: getattr(self, f.name)
                for f in fields(self)
                if not f.name.startswith('_')
            }
    
       # Otherwise use __dict__
        return self.__dict__.copy()

    ##########################
    # fromJSON
    ##########################
    @classmethod
    def fromJSON(cls, jsonString):
        
        # Recursively collect all subclasses of pglSerialize
        def getAllSubclasses(baseClass):
            allSubclasses = {}
            for subclass in baseClass.__subclasses__():
                allSubclasses[subclass.__name__] = subclass
                allSubclasses.update(getAllSubclasses(subclass))
            return allSubclasses
        
        # Build registry of all known pglSerialize subclasses
        CLASS_REGISTRY = getAllSubclasses(pglSerialize)
        
        # Decode each dict, restoring objects by type markers
        def decodeObject(dct):
            # Restore datetime objects
            if '__datetime__' in dct:
                return datetime.fromisoformat(dct['value'])
            
            # Restore numpy arrays
            if '__numpy__' in dct:
                return np.array(dct['data'], dtype=dct['dtype']).reshape(dct['shape'])
            
            # Restore tuples
            if '__tuple__' in dct:
                return tuple(dct['items'])
            
            # Restore HasTraits objects
            if '__hastraits__' in dct:
                module_name = dct.pop('__module__')
                class_name = dct.pop('__class__')
                dct.pop('__hastraits__')
                
                try:
                    import importlib
                    module = importlib.import_module(module_name)
                    trait_class = getattr(module, class_name)
                    
                    obj = trait_class()
                    for key, value in dct.items():
                        setattr(obj, key, value)
                    return obj
                except (ImportError, AttributeError) as e:
                    print(f"(pglSerialize) Could not restore HasTraits object {class_name}: {e}")
                    return dct
            
            # Restore pglSerialize objects
            if '__class__' in dct:
                className = dct.pop('__class__')
                if className in CLASS_REGISTRY:
                    objectClass = CLASS_REGISTRY[className]
                    obj = objectClass.fromJSONdict(dct)
                    return obj
            
            return dct
        
        return json.loads(jsonString, object_hook=decodeObject)

    ##########################
    # fromJSONdict
    ##########################
    @classmethod
    def fromJSONdict(cls, data, type="all"):
        """Create instance from dict. Override for custom initialization/validation"""
        obj = cls.__new__(cls)
        
        # If this is a HasTraits class, use updateTraitsFromDict
        if issubclass(cls, HasTraits):
            obj.__init__()  # Initialize HasTraits properly
            obj.updateTraitsFromDict(data)
            # If this is a dataclass, use __init__ to get defaults and validation
        elif is_dataclass(cls):
            # Get all field names
            fieldNames = {f.name for f in fields(cls)}
            # Filter data to only include valid fields
            init_data = {k: v for k, v in data.items() if k in fieldNames}
            # Use normal __init__ (applies defaults, runs __post_init__, etc.)
            return cls(**init_data)
        else:
            obj.__dict__.update(data)
        
        return obj
    
    ##########################
    # updateTraitsFromDict - For HasTraits objects
    ##########################
    def updateTraitsFromDict(self, data, filename="<dict>", typeConverter=None):
        """
        Update traits from a dictionary with validation and error handling.
        
        Args:
            data: Dictionary of trait values
            filename: Source filename for error messages
            typeConverter: Optional callable(key, value) -> converted_value for custom type conversions
        """
        if not isinstance(self, HasTraits):
            # Fall back to simple dict update for non-HasTraits objects
            self.__dict__.update(data)
            return
        
        for key in self.trait_names():
            if key in data:
                try:
                    value = data[key]
                    
                    # Apply custom type converter if provided
                    if typeConverter and callable(typeConverter):
                        value = typeConverter(key, value)
                    
                    # Set the attribute
                    setattr(self, key, value)
                    
                except TraitError:
                    trait = self.traits()[key]
                    expectedType = trait.__class__
                    gotType = type(data[key])
                    print(f"(pglSerialize) '{key}' has wrong type in '{filename}' "
                          f"(expected {expectedType.__name__}, got {gotType.__name__}), "
                          f"using default {getattr(self, key)}")
                except Exception as e:
                    print(f"(pglSerialize) Error setting '{key}': {e}. "
                          f"Using default value: {getattr(self, key)}")
            else:
                print(f"(pglSerialize) '{key}' not found in '{filename}', "
                      f"using default {getattr(self, key)}")
        
        # Warn about unknown keys
        extraKeys = set(data.keys()) - set(self.trait_names())
        if extraKeys:
            print(f"(pglSerialize) Unknown keys in '{filename}' (ignored): {list(extraKeys)}")
            
    ##########################
    # copyTraitsFrom - Copy traits from another object
    ##########################
    def copyTraitsFrom(self, other, verbose=True):
        """
        Copy trait/attribute values from another object.
        
        Args:
            other: Source object to copy from
            verbose: Whether to print warnings on errors
        """
        # If both are HasTraits objects, use trait_names()
        if isinstance(self, HasTraits) and isinstance(other, HasTraits):
            keys = self.trait_names()
        # If only self is HasTraits
        elif isinstance(self, HasTraits):
            keys = self.trait_names()
        # If neither is HasTraits, use __dict__
        else:
            keys = self.__dict__.keys()
        
        # Copy each attribute
        for key in keys:
            if hasattr(other, key):
                try:
                    setattr(self, key, getattr(other, key))
                except TraitError as e:
                    if verbose:
                        print(f"(pglSerialize) Could not copy trait '{key}': {e}")
                except Exception as e:
                    if verbose:
                        print(f"(pglSerialize) Error copying '{key}': {e}")
            elif verbose:
                print(f"(pglSerialize) Source object missing attribute '{key}'")

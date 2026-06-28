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
# Recursively collect all subclasses
##########################
def pglGetAllSubclasses(baseClass):
    allSubclasses = {}
    for subclass in baseClass.__subclasses__():
        allSubclasses[subclass.__name__] = subclass
        allSubclasses.update(pglGetAllSubclasses(subclass))
    return allSubclasses

##########################
# pglSerialize class
##########################
class pglSerialize:
    '''
    Base class for serializing and deserializing objects to/from JSON
    If you inherit from this, then you can save and load to JSON files.
    If you have a standard class then it will use __dict__ to save and load attributes.
    If you have a HasTraits class, then it will use trait_names() to save and load attributes.
    If you have a dataclass, then it will use fields() to save and load attributes
    '''
    
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
    @classmethod
    def load(cls, filename):
        """Load a pglSerialize (or subclass) object from a JSON file and return it"""
        filename = Path(filename).with_suffix(".json")

        if not filename.exists():
            print(f"(pglSerialize) File '{filename}' not found.")
            return None

        if not filename.is_file():
            print(f"(pglSerialize) '{filename}' is not a file.")
            return None

        try:
            jsonString = filename.read_text()
            obj = cls.fromJSON(jsonString)  # uses your existing fromJSON
            return obj
        except PermissionError:
            print(f"(pglSerialize) No permission to read '{filename}'.")
        except OSError as e:
            print(f"(pglSerialize) OS error reading '{filename}': {e}")
        except json.JSONDecodeError as e:
            print(f"(pglSerialize) JSON decode error in '{filename}': {e}")
        except Exception as e:
            print(f"(pglSerialize) Unknown error loading '{filename}': {type(e).__name__}: {e}")

        return None
    ##########################
    # update instance function - updates this instance in place from a JSON file
    # (useful for updating an existing object rather than creating a new one)
    ##########################
    def updateFromFile(self, filename):
        """Update this object in place from a JSON file"""
        obj = self.__class__.load(filename)  # call the class method
        if obj is not None:
            self.copyTraitsFrom(obj)  # reuse existing copy logic
    
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
    
        # Otherwise use __dict__, skipping private attributes
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
    ##########################
    # fromJSON
    ##########################
    @classmethod
    def fromJSON(cls, jsonString):
                
        # Build registry of all known pglSerialize subclasses
        CLASS_REGISTRY = pglGetAllSubclasses(pglSerialize)
        
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
        import traceback
        import inspect
        
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
                    
                except TraitError as e:
                    trait = self.traits()[key]
                    expectedType = trait.__class__
                    gotValue = data[key]
                    gotType = type(gotValue)
                    
                    # Get more detailed type information
                    expectedTypeInfo = self._getDetailedTypeInfo(trait)
                    gotTypeInfo = self._getDetailedTypeInfo(gotValue)
                    
                    print(f"\n{'='*80}")
                    print(f"(pglSerialize) TRAIT TYPE MISMATCH ERROR")
                    print(f"{'='*80}")
                    print(f"Trait name:     '{key}'")
                    print(f"Source:         '{filename}'")
                    print(f"Object class:   {self.__class__.__name__}")
                    print(f"Object:         {self}")
                    print(f"-" * 80)
                    print(f"Expected type:  {expectedTypeInfo}")
                    print(f"Got type:       {gotTypeInfo}")
                    print(f"Got value:      {repr(gotValue)[:200]}")  # Truncate long values
                    print(f"Using default:  {repr(getattr(self, key))[:200]}")
                    print(f"-" * 80)
                    print(f"Original error: {e}")
                    print(f"-" * 80)
                    print("Call stack:")
                    # Print abbreviated stack trace (skip first 2 frames - this function)
                    for frame_info in traceback.extract_stack()[:-2]:
                        print(f"  File '{frame_info.filename}', line {frame_info.lineno}, in {frame_info.name}")
                        if frame_info.line:
                            print(f"    {frame_info.line}")
                    print(f"{'='*80}\n")
                    
                except Exception as e:
                    print(f"\n{'='*80}")
                    print(f"(pglSerialize) UNEXPECTED ERROR")
                    print(f"{'='*80}")
                    print(f"Trait name:     '{key}'")
                    print(f"Source:         '{filename}'")
                    print(f"Object class:   {self.__class__.__name__}")
                    print(f"Value:          {repr(data[key])[:200]}")
                    print(f"Error type:     {type(e).__name__}")
                    print(f"Error message:  {e}")
                    print(f"Using default:  {repr(getattr(self, key))[:200]}")
                    print(f"-" * 80)
                    print("Full traceback:")
                    traceback.print_exc()
                    print(f"{'='*80}\n")
            else:
                print(f"(pglSerialize) '{key}' not found in '{filename}', "
                    f"using default {getattr(self, key)}")
        
        # Warn about unknown keys
        extraKeys = set(data.keys()) - set(self.trait_names())
        if extraKeys:
            print(f"(pglSerialize) Unknown keys in '{filename}' (ignored): {list(extraKeys)}")


    def _getDetailedTypeInfo(self, obj):
        """
        Helper to get detailed type information for better error messages.
        """
        if hasattr(obj, '__class__'):
            typeName = obj.__class__.__name__
            
            # For trait types, get additional info
            if hasattr(obj, 'info'):
                return f"{typeName} ({obj.info()})"
            
            # For list/dict, show element types if possible
            if isinstance(obj, list):
                if obj:
                    elementTypes = set(type(x).__name__ for x in obj[:5])  # Sample first 5
                    return f"list (contains: {', '.join(elementTypes)})"
                return "list (empty)"
            
            if isinstance(obj, dict):
                if obj:
                    keyTypes = set(type(k).__name__ for k in list(obj.keys())[:5])
                    valueTypes = set(type(v).__name__ for v in list(obj.values())[:5])
                    return f"dict (keys: {', '.join(keyTypes)}, values: {', '.join(valueTypes)})"
                return "dict (empty)"
            
            return typeName
        
        return str(type(obj).__name__)
          
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

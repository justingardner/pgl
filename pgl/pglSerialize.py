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
from .pglMessages import pglMessages
import fsspec
from fsspec.core import url_to_fs
from types import SimpleNamespace

##########################
# Recursively collect all subclasses
##########################
def pglGetAllSubclasses(baseClass):
    allSubclasses = {}

    def addName(name, cls):
        # check for duplicate class
        if name in allSubclasses:
            # if it is in the same module, then its most likely just a stale jupyter class,
            # so overwrite the old one and move on
            existing = allSubclasses[name]
            if cls.__module__ == existing.__module__:
                allSubclasses[name] = cls
                return

            # different modules, more likely to be real name clash, so warn and ignore
            pglMessages.warning(
                f"Duplicate serialization name '{name}' "
                f"for {cls.__name__} and {allSubclasses[name].__name__}", level=0
            )
        allSubclasses[name] = cls

    def collect(cls):
        for subclass in cls.__subclasses__():

            # Canonical name
            addName(subclass.__name__, subclass)

            # Aliases for backwards compatibility.
            for alias in subclass.__dict__.get("_oldSerializationNames", []):
                #print(f"alias: {alias} subclass: {subclass}")
                addName(alias, subclass)

            # Recurse
            collect(subclass)

    collect(baseClass)
    return allSubclasses

##########################
# Should we read this trait's value right now, or would doing so
# force an unfired dynamic default (e.g. a lazy/network-backed field)?
##########################
def pglShouldReadTraitForSerialize(obj, traitName):
    '''Decide whether a trait should be serialized.

    Default: serialize all declared traits. A trait can explicitly opt out
    by setting serialize=False in its metadata. 
    '''
    trait = obj.traits().get(traitName)
    if trait is None:
        return False

    # Explicit opt-out wins
    if trait.metadata.get("serialize", True) is False:
        return False
        
    hasDynamicDefault = traitName in type(obj)._trait_default_generators
    if hasDynamicDefault and not obj.trait_has_value(traitName):
        # never been computed - reading now would fire the generator
        return False
    return True

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
    verbose = False
    _serializeUnregisteredFields = False
    ##########################
    # Save to JSON file
    ##########################
    def save(self, filename, filesystem=None, filesystemPrefix=None):
        """Save object to JSON file"""
        try:
            # Validate/resolve filesystem and normalise the path
            from .pglBase import pglBase
            dataPath = Path(filename).parent
            filesystem, dataPath, _ = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix, create=True)
            if filesystem is None:
                pglMessages.warning(f"(pglSerialize) Could not resolve a filesystem for '{filename}'.")
                return
            filename = Path(dataPath) / Path(filename).name

            # Make it json
            filename = str(Path(filename).with_suffix(".json"))
            
            #pglMessages.message(f"Saving {self.__class__.__name__} to '{filename}'")
            with filesystem.open(filename, 'w') as f:
                # call toJSON (filename is just for error displays)
                f.write(self.toJSON(filename=filename))
        except PermissionError:
            pglMessages.warning(f"(pglSerialize) No permission to write to '{filename}'")
        except IsADirectoryError:
            pglMessages.warning(f"(pglSerialize) '{filename}' is a directory, cannot write file")
        except OSError as e:
            pglMessages.warning(f"(pglSerialize) OS error while saving '{filename}': {e}")
        except Exception as e:
            pglMessages.warning(f"(pglSerialize) Unknown error ({type(e).__name__}) while saving '{filename}': {e}")
    
    ##########################
    # Load from JSON file
    ##########################
    @classmethod
    def load(cls, filename, filesystem=None, filesystemPrefix=None, loadAsClass=None):
        """Load a pglSerialize (or subclass) object from a JSON file and return it.

        Args:
            filename (str or Path): Path to the JSON file. May include a protocol
                qualifier (e.g. 'ssh://...') when `filesystem` is None.
            filesystem (fsspec.AbstractFileSystem, optional): Filesystem to read
                through. If None, it is inferred from `filename` (falling back to
                a local filesystem). Defaults to None.
            loadAsClass: IF set will load the class as the set class name

        Returns:
            The loaded object, or None if it could not be loaded.
        """
        # Ensure the filename has a .json suffix
        filename = str(Path(filename).with_suffix(".json"))

        # Validate/resolve filesystem and normalise the path
        from .pglBase import pglBase
        filesystem, filename, _ = pglBase.validateFilesystem(filesystem=filesystem, dataPath=filename, filesystemPrefix=filesystemPrefix)
        if filesystem is None:
            pglMessages.warning(f"(pglSerialize) Could not resolve a filesystem for '{filename}'.")
            return None

        if not filesystem.exists(filename):
            pglMessages.warning(f"File {filename} not found.", level=1)
            return None

        if not filesystem.isfile(filename):
            pglMessages.warning(f"{filename} is not a file.", level=1)
            return None

        try:
            with filesystem.open(filename, "r") as fileHandle:
                jsonString = fileHandle.read()
            
            obj = cls.fromJSON(jsonString, filename=filename, loadAsClass=loadAsClass)  # uses your existing fromJSON
            return obj
        except PermissionError:
            pglMessages.warning(f"(pglSerialize) No permission to read '{filename}'.", level=1)
        except OSError as e:
            pglMessages.warning(f"(pglSerialize) OS error reading '{filename}': {e}", level=1)
        except json.JSONDecodeError as e:
            pglMessages.warning(f"(pglSerialize) JSON decode error in '{filename}': {e}", level=1)
        except Exception as e:
            pglMessages.warning(f"(pglSerialize) Unknown error loading '{filename}': {type(e).__name__}: {e}", level=1)

        return None
    ##########################
    # update instance function - updates this instance in place from a JSON file
    # (useful for updating an existing object rather than creating a new one)
    ##########################
    def updateFromFile(self, filename, filesystem=None):
        obj = self.__class__.load(filename, filesystem=filesystem)
        if obj is not None:
            self.copyTraitsFrom(obj)    
    ##########################
    # toJSON: determines how to serialize to JSON different data types
    ##########################
    def toJSON(self, type="all", filename=None):
        """
        Serialize this object to JSON.

        Circular references are detected while traversing the object graph.
        When a circular reference is encountered, that field is omitted and
        a short warning is issued. Serialization then continues normally.
        """

        # Objects currently being traversed.
        #
        # This is intentionally an "active" set rather than a global "seen"
        # set. The same object may legitimately appear in multiple places;
        # only a reference back into the current traversal is a cycle.
        activeObjects = {}

        def encodeObject(o, path="root"):

            # ----------------------------------------------------------
            # Determine whether this object can participate in a cycle.
            # ----------------------------------------------------------
            trackObject = isinstance(
                o,
                (pglSerialize, HasTraits, SimpleNamespace, list, dict, tuple)
            )

            objectId = id(o) if trackObject else None

            # ----------------------------------------------------------
            # Circular reference detected.
            #
            # Return a private sentinel so the caller can omit the field
            # rather than writing "null" into the JSON.
            # ----------------------------------------------------------
            if trackObject and objectId in activeObjects:

                pglMessages.warning(
                    f"(pglSerialize) Circular reference skipped at "
                    f"'{path}' ({o.__class__.__name__})",
                    level=1
                )

                return _SERIALIZATION_SKIP

            # Mark object as active.
            if trackObject:
                activeObjects[objectId] = o

            try:

                # ------------------------------------------------------
                # Custom encoding for pglSerialize objects
                # ------------------------------------------------------
                if isinstance(o, pglSerialize):

                    data = o.toJSONdict(type)

                    result = {
                        '__class__': o.__class__.__name__
                    }

                    for key, value in data.items():

                        encoded = encodeObject(
                            value,
                            f"{path}.{key}"
                        )

                        # Omit fields that would create a cycle.
                        if encoded is not _SERIALIZATION_SKIP:
                            result[key] = encoded

                    return result

                # ------------------------------------------------------
                # datetime
                # ------------------------------------------------------
                elif isinstance(o, datetime):

                    return {
                        '__datetime__': True,
                        'value': o.isoformat()
                    }

                # ------------------------------------------------------
                # pathlib.Path
                # ------------------------------------------------------
                elif isinstance(o, Path):

                    return {
                        '__path__': True,
                        'value': str(o)
                    }                # ------------------------------------------------------
                # numpy arrays
                # ------------------------------------------------------
                elif isinstance(o, np.ndarray):

                    return {
                        '__numpy__': True,
                        'dtype': str(o.dtype),
                        'shape': o.shape,
                        'data': o.tolist()
                    }

                # ------------------------------------------------------
                # numpy scalar types
                # ------------------------------------------------------
                elif isinstance(o, (np.integer, np.floating)):

                    return o.item()

                # ------------------------------------------------------
                # tuples
                # ------------------------------------------------------
                elif isinstance(o, tuple):

                    items = []

                    for i, item in enumerate(o):

                        encoded = encodeObject(
                            item,
                            f"{path}[{i}]"
                        )

                        # A tuple cannot have an omitted element without
                        # changing its positional meaning, so preserve
                        # the existing behavior by using None here.
                        if encoded is _SERIALIZATION_SKIP:
                            encoded = None

                        items.append(encoded)

                    return {
                        '__tuple__': True,
                        'items': items
                    }

                # ------------------------------------------------------
                # HasTraits objects
                # ------------------------------------------------------
                elif isinstance(o, HasTraits) and not isinstance(o, pglSerialize):

                    result = {
                        '__hastraits__': True,
                        '__class__': o.__class__.__name__,
                        '__module__': o.__class__.__module__
                    }

                    for key in o.trait_names():

                        if not pglShouldReadTraitForSerialize(o, key):
                            continue

                        encoded = encodeObject(
                            getattr(o, key),
                            f"{path}.{key}"
                        )

                        # Omit cyclic fields.
                        if encoded is not _SERIALIZATION_SKIP:
                            result[key] = encoded

                    return result

                # ------------------------------------------------------
                # lists
                # ------------------------------------------------------
                elif isinstance(o, list):

                    result = []

                    for i, item in enumerate(o):

                        encoded = encodeObject(
                            item,
                            f"{path}[{i}]"
                        )

                        # Lists are positional, so preserve the existing
                        # list behavior by replacing a cyclic element
                        # with None rather than shifting the remaining
                        # elements.
                        if encoded is _SERIALIZATION_SKIP:
                            encoded = None

                        result.append(encoded)

                    return result

                # ------------------------------------------------------
                # dictionaries
                # ------------------------------------------------------
                elif isinstance(o, dict):

                    result = {}

                    for key, value in o.items():

                        encoded = encodeObject(
                            value,
                            f"{path}.{key}"
                        )

                        # Omit cyclic dictionary values entirely.
                        if encoded is not _SERIALIZATION_SKIP:
                            result[key] = encoded

                    return result

                # ------------------------------------------------------
                # ------------------------------------------------------
                # JSON primitive
                # ------------------------------------------------------
                elif o is None or isinstance(o, (str, int, float, bool)):
                    return o

                # ------------------------------------------------------
                # SimpleNamespace
                # ------------------------------------------------------
                elif isinstance(o, SimpleNamespace):

                    result = {
                        '__simplenamespace__': True
                    }

                    for key, value in vars(o).items():

                        encoded = encodeObject(
                            value,
                            f"{path}.{key}"
                        )

                        if encoded is not _SERIALIZATION_SKIP:
                            result[key] = encoded

                    return result
                
                # ------------------------------------------------------
                # Unknown object
                # ------------------------------------------------------
                else:
                    pglMessages.warning(
                        f"(pglSerialize) Unsupported object skipped at "
                        f"'{path}' ({o.__class__.__name__})"
                        + (f" while saving '{filename}'" if filename else ""),
                        level=1
                    )
                    return _SERIALIZATION_SKIP

            finally:

                if trackObject:
                    activeObjects.pop(objectId, None)

        # Private sentinel used to distinguish "omit this field" from
        # an actual serialized None.
        _SERIALIZATION_SKIP = object()

        # IMPORTANT:
        # Fully encode the object graph before handing anything to
        # json.dumps(). This prevents json.dumps() from doing its own
        # traversal and detecting a cycle that we failed to catch.
        encoded = encodeObject(self, "root")

        return json.dumps(
            encoded,
            sort_keys=True,
            indent=4
        )

   
    ##########################
    # toJSONdict: Decide what needs to be serialized.
    ##########################
    def toJSONdict(self, type="all"):
        isHasTraits = isinstance(self, HasTraits)
        isDataClass = is_dataclass(self)

        # Warn if both serialization mechanisms are present.
        if isHasTraits and isDataClass:
            pglMessages.warning(
                f"{self.__class__.__name__} is both a HasTraits object \n"
                f"and a dataclass. HasTraits attributes will be used for \n"
                f"serialization; dataclass fields will be ignored.",
            )

        result = {}

        # HasTraits takes precedence if both are present.
        if isHasTraits:
            result.update({
                key: getattr(self, key)
                for key in self.trait_names()
                # logic in function above for what keys to save/ignore
                # this function handles some traits that are lazy-loaded 
                if pglShouldReadTraitForSerialize(self, key)
            })

        # Dataclass serialization.
        elif isDataClass:
            result.update({
                f.name: getattr(self, f.name)
                for f in fields(self)
                # do not save private variables starting with _
                if not f.name.startswith("_")
            })

        # Find public __dict__ fields which weren't handled by
        # HasTraits or dataclass serialization.
        if hasattr(self, "__dict__"):
            # unregistered fields
            unregisteredFields = {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("_") and key not in result
            }

            if unregisteredFields:
                # if the class has _serialzeUnregisteredFields set to True
                # then add those to the fields we will save
                if self._serializeUnregisteredFields:
                    result.update(unregisteredFields)
                # if not, then warn that the fields will be dropped epending on verbose level
                elif pglSerialize.verbose:
                    pglMessages.warning(
                        f"{self.__class__.__name__} has unregistered "
                        f"attributes that will not be serialized: "
                        f"{list(unregisteredFields.keys())}",
                        level=0
                    )

        # If this is neither HasTraits nor a dataclass, preserve the
        # normal behavior: serialize public __dict__ fields.
        if not isHasTraits and not isDataClass:
            return {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            }

        return result
    ##########################
    # fromJSON
    ##########################
    @classmethod
    def fromJSON(cls, jsonString, filename=None, loadAsClass=None):
                
        # Build registry of all known pglSerialize subclasses
        CLASS_REGISTRY = pglGetAllSubclasses(pglSerialize)
        
        # Decode each dict, restoring objects by type markers
        def decodeObject(dct):
            # Restore datetime objects
            if '__datetime__' in dct:
                return datetime.fromisoformat(dct['value'])
            
            # Restore pathlib.Path objects
            if '__path__' in dct:
                return Path(dct['value'])

            # Restore numpy arrays
            if '__numpy__' in dct:
                return np.array(dct['data'], dtype=dct['dtype']).reshape(dct['shape'])
            
            # Restore tuples
            if '__tuple__' in dct:
                return tuple(dct['items'])

            # Restore SimpleNamespace objects
            if '__simplenamespace__' in dct:
                dct.pop('__simplenamespace__')
                return SimpleNamespace(**dct)            
            
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
                    pglMessages.warning(f"(pglSerialize) Could not restore HasTraits object {class_name}: {e}", level=1)
                    return dct
            
            # Restore pglSerialize objects
            if '__class__' in dct:
                className = dct.pop('__class__')
                if loadAsClass is not None:
                    # override setting in json
                    objectClass = loadAsClass
                elif className in CLASS_REGISTRY:
                    objectClass = CLASS_REGISTRY[className]
                else:
                    availableClasses = sorted(CLASS_REGISTRY.keys())
                    pglMessages.warning(f"(pglSerialize) Could not restore object of class '{className}' from '{filename}'.\nKnown classes: {availableClasses}")
                    dct["__class__"] = className
                    return dct
                obj = objectClass.fromJSONdict(dct, filename=filename)
                return obj
            return dct
        return json.loads(jsonString, object_hook=decodeObject)

    ##########################
    # fromJSONdict
    ##########################
    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Create instance from dict. Override for custom initialization/validation"""

        # If this is a HasTraits class, use updateTraitsFromDict
        if issubclass(cls, HasTraits):
            obj = cls.__new__(cls)
            obj.__init__()
            obj.updateTraitsFromDict(data, filename=filename)
            return obj

        # If this is a dataclass, use normal __init__ for declared fields,
        # then restore any registered/unregistered instance attributes.
        elif is_dataclass(cls):
            fieldNames = {f.name for f in fields(cls)}

            # Normal dataclass fields go through __init__, which preserves
            # defaults, validation, and __post_init__ behavior.
            initData = {
                key: value
                for key, value in data.items()
                if key in fieldNames
            }

            obj = cls(**initData)

            # Restore unregistered fields if this class allows them.
            if obj._serializeUnregisteredFields:
                unregisteredFields = {
                    key: value
                    for key, value in data.items()
                    if key not in fieldNames
                }

                for key, value in unregisteredFields.items():
                    setattr(obj, key, value)

            return obj

        # Otherwise use __dict__
        else:
            obj = cls.__new__(cls)
            obj.__dict__.update(data)
            return obj
    
    ##########################
    # updateTraitsFromDict - For HasTraits objects
    ##########################
    def updateTraitsFromDict(self, data, filename=None, typeConverter=None):
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
                    #print("Call stack:")
                    # Print abbreviated stack trace (skip first 2 frames - this function)
                    #for frame_info in traceback.extract_stack()[:-2]:
                    #    print(f"  File '{frame_info.filename}', line {frame_info.lineno}, in {frame_info.name}")
                    #    if frame_info.line:
                    #        print(f"    {frame_info.line}")
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
                if pglShouldReadTraitForSerialize(self, key):
                    pglMessages.message(f"'{key}' not found in {filename} using default {getattr(self, key)}", verbose=pglSerialize.verbose)

        # Handle unknown keys
        traitNames = set(self.trait_names())
        extraKeys = set(data.keys()) - traitNames

        if extraKeys:
            if self._serializeUnregisteredFields:
                for key in extraKeys:
                    try:
                        setattr(self, key, data[key])
                    except Exception as e:
                        pglMessages.warning(
                            f"Could not restore unregistered "
                            f"attribute '{key}' from '{filename}': {e}",
                            level=1
                        )
            else:
                pglMessages.message(
                    f"unknown keys in {filename} (ignored): "
                    f"{list(extraKeys)}",
                    verbose=pglSerialize.verbose
                )

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
    def copyTraitsFrom(self, other):
        """
        Copy trait/attribute values from another object.
        
        Args:
            other: Source object to copy from
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
                    pglMessages.warning(f"Could not copy trait '{key}': {e}", verbose=pglSerialize.verbose)
                except Exception as e:
                    pglMessages.warning(f"Error copying '{key}': {e}", verbose=pglSerialize.verbose)
            else:
                pglMessages.message(f"Source object missing attribute '{key}'",verbose=pglSerialize.verbose)

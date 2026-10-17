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
import posixpath
from enum import Enum
import sys
from contextvars import ContextVar

# Carry strict restoration policy through existing custom fromJSONdict()
# implementations without adding new keywords to every subclass.
_pglStrictRestorePath = ContextVar("pglStrictRestorePath", default=None)

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
#  pglSerializedError
##########################
class pglSerializedError(Exception):
    """Recorded error information, not a reconstructed runtime exception."""

    def __init__(self, originalModule, originalClass, message):
        super().__init__(message)
        self.originalModule = originalModule
        self.originalClass = originalClass
        
##########################
# Should we read this trait's value right now, or would doing so
# force an unfired dynamic default (e.g. a lazy/network-backed field)?
##########################
def pglShouldReadTraitForSerialize(obj, traitName):
    """Select persistent traits for serialization without reading their values."""
    trait = obj.traits().get(traitName)
    if trait is None or trait.metadata.get("serialize", True) is False: return False

    # Explicit pending backing must be resolved when serializing current contents.
    if trait.metadata.get("lazyload") is True:
        lazyIsPending = getattr(obj, "lazyIsPending", None)
        if callable(lazyIsPending) and lazyIsPending(traitName): return True

    # Preserve existing behavior for uncomputed, unbound dynamic defaults.
    hasDynamicDefault = traitName in type(obj)._trait_default_generators
    if hasDynamicDefault and not obj.trait_has_value(traitName): return False
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
            """Save object to JSON through an fsspec filesystem."""
            import posixpath

            try:
                from .pglBase import pglBase

                filename = str(filename)
                dataPath = posixpath.dirname(filename) or "."
                fileName = posixpath.basename(filename)

                filesystem, dataPath, _ = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix, create=True)
                if filesystem is None:
                    pglMessages.warning(f"(pglSerialize) Could not resolve a filesystem for '{filename}'.")
                    return

                filename = posixpath.join(dataPath, posixpath.splitext(fileName)[0] + ".json")

                with filesystem.open(filename, "w", encoding="utf-8") as file:
                    file.write(self.toJSON(filename=filename))

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
        import posixpath
        filename = posixpath.splitext(str(filename))[0] + ".json"

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
    def toJSON(self, type="all", filename=None, *, strict=False):
        """Serialize metadata using the existing PGL JSON representation.

        Default behavior remains permissive: unsupported values and circular
        references warn and are skipped, or become None inside sequences.

        strict=True rejects unsupported values, cycles, non-string mapping
        keys, reserved marker collisions, and nonfinite floating-point values.

        Existing class-specific toJSONdict() methods still select fields.
        This method does not change lazy-field selection or export payloads.
        Strict decoding is a separate operation.
        """
        import math

        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")

        activeObjects = {}
        serializationSkip = object()
        reservedMarkers = {"__class__", "__hastraits__", "__exception__", "__enum__", "__datetime__", "__path__", "__numpy__", "__range__", "__tuple__", "__simplenamespace__"}

        def unsupported(value, path, reason):
            """Raise in strict mode; otherwise retain warning-and-skip behavior."""
            message = f"(pglSerialize) {reason} at '{path}' ({value.__class__.__name__})"

            if filename:
                message += f" while saving '{filename}'"

            if strict:
                raise ValueError(message)

            pglMessages.warning(message, level=1)
            return serializationSkip

        def encodeFields(values, result, path):
            """Encode fields while protecting type markers in strict mode."""
            if strict:
                for key in values:
                    if not isinstance(key, str):
                        raise TypeError(f"{path}: metadata keys must be strings; received {key!r}.")

                    if key in reservedMarkers:
                        raise ValueError(f"{path}.{key}: field conflicts with a reserved serialization marker.")

            for key, value in values.items():
                encoded = encodeObject(value, f"{path}.{key}")

                if encoded is not serializationSkip:
                    result[key] = encoded

            return result

        def encodeSequence(values, path):
            """Preserve positional meaning when permissive encoding skips an item."""
            result = []

            for index, value in enumerate(values):
                encoded = encodeObject(value, f"{path}[{index}]")
                result.append(None if encoded is serializationSkip else encoded)

            return result

        def encodeObject(value, path="root"):
            """Encode one value, tracking only objects on the current traversal."""
            trackObject = isinstance(value, (pglSerialize, HasTraits, SimpleNamespace, list, dict, tuple))
            objectID = id(value) if trackObject else None

            # A shared object appearing twice is allowed. Only a reference back
            # into the current traversal is a circular reference.
            if trackObject and objectID in activeObjects:
                return unsupported(value, path, "Circular reference skipped")

            if trackObject:
                activeObjects[objectID] = value

            try:
                if isinstance(value, Exception):
                    if isinstance(value, pglSerializedError):
                        moduleName = value.originalModule
                        className = value.originalClass
                    else:
                        moduleName = value.__class__.__module__
                        className = value.__class__.__qualname__

                    if strict:
                        if not isinstance(moduleName, str) or not moduleName or not isinstance(className, str) or not className:
                            raise ValueError(f"{path}: invalid serialized exception identity.")

                    return {"__exception__": True, "__module__": moduleName, "__qualname__": className, "message": str(value)}

                # IntEnum and StrEnum must be handled before ordinary primitives.
                if isinstance(value, Enum):
                    if strict and (not isinstance(value.name, str) or not value.name):
                        raise ValueError(f"{path}: enum value has no restorable member name.")

                    return {"__enum__": True, "__module__": value.__class__.__module__, "__qualname__": value.__class__.__qualname__, "name": value.name}

                if isinstance(value, pglSerialize):
                    # Keep each class's field-selection implementation.
                    # Do not pass a new keyword to existing overrides.
                    data = value.toJSONdict(type)

                    if not isinstance(data, dict):
                        raise TypeError(f"{path}: {value.__class__.__name__}.toJSONdict() must return a dictionary.")

                    return encodeFields(data, {"__class__": value.__class__.__name__}, path)

                if isinstance(value, range):
                    return {"__range__": True, "start": value.start, "stop": value.stop, "step": value.step}

                if isinstance(value, datetime):
                    return {"__datetime__": True, "value": value.isoformat()}

                if isinstance(value, Path):
                    return {"__path__": True, "value": str(value)}

                if isinstance(value, np.ndarray):
                    if strict:
                        # The existing dtype/shape/tolist representation supports
                        # plain numeric and Unicode metadata arrays. Structured,
                        # object, complex, and other arrays need explicit codecs.
                        if value.dtype.fields is not None or value.dtype.kind not in "biufU":
                            raise TypeError(f"{path}: NumPy dtype {value.dtype} is not supported by strict metadata encoding.")

                        if value.dtype.kind == "f" and not np.isfinite(value).all():
                            raise ValueError(f"{path}: array contains nonfinite floating-point values.")

                    data = value.tolist()

                    if strict:
                        # Validate every converted item as metadata. Extended
                        # precision types may remain NumPy scalars after tolist().
                        data = encodeObject(data, f"{path}.data")

                    return {"__numpy__": True, "dtype": str(value.dtype), "shape": value.shape, "data": data}

                if isinstance(value, (np.integer, np.floating)):
                    scalar = value.item()

                    if strict:
                        if isinstance(scalar, np.generic):
                            raise TypeError(f"{path}: NumPy scalar {value.dtype} cannot be represented by the existing JSON scalar encoding.")

                        return encodeObject(scalar, path)

                    return scalar

                if isinstance(value, tuple):
                    return {"__tuple__": True, "items": encodeSequence(value, path)}

                if isinstance(value, HasTraits):
                    result = {"__hastraits__": True, "__class__": value.__class__.__name__, "__module__": value.__class__.__module__}

                    # Retain the supplied file's exact trait-selection policy.
                    # This can resolve pending lazyload traits; strict mode does
                    # not implement the proposed external-reference protocol.
                    for key in value.trait_names():
                        if not pglShouldReadTraitForSerialize(value, key):
                            continue

                        if strict and key in reservedMarkers:
                            raise ValueError(f"{path}.{key}: trait conflicts with a reserved serialization marker.")

                        encoded = encodeObject(getattr(value, key), f"{path}.{key}")

                        if encoded is not serializationSkip:
                            result[key] = encoded

                    return result

                if isinstance(value, list):
                    return encodeSequence(value, path)

                if isinstance(value, dict):
                    return encodeFields(value, {}, path)

                if value is None or isinstance(value, (str, int, float, bool)):
                    if strict and isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(f"{path}: nonfinite floating-point values are not supported by strict JSON encoding.")

                    return value

                if isinstance(value, SimpleNamespace):
                    return encodeFields(vars(value), {"__simplenamespace__": True}, path)

                return unsupported(value, path, "Unsupported object skipped")

            finally:
                if trackObject:
                    activeObjects.pop(objectID, None)

        # Complete our traversal before handing the result to json.dumps().
        encoded = encodeObject(self)
        return json.dumps(encoded, sort_keys=True, indent=4, allow_nan=not strict)

   
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
    # fromJSONStrict
    ##########################
    @classmethod
    def _fromJSONStrict(cls, jsonString, filename=None, loadAsClass=None, rootPath="root"):
        """Validate and restore metadata without silent fallback.

        Each reconstructed tagged object must re-encode to the same metadata,
        allowing canonical class names in place of registered legacy aliases.

        Explicit migrations that intentionally change document contents are
        separate from this exact-restoration mode.
        """
        import math

        if loadAsClass is not None and (not isinstance(loadAsClass, type) or not issubclass(loadAsClass, pglSerialize)):
            raise TypeError("loadAsClass must be a pglSerialize subclass.")

        classRegistry = pglGetAllSubclasses(pglSerialize)
        markers = {"__class__", "__hastraits__", "__exception__", "__enum__", "__datetime__", "__path__", "__numpy__", "__range__", "__tuple__", "__simplenamespace__"}

        def rejectConstant(value):
            raise ValueError(f"{rootPath}: nonstandard JSON constant {value!r} is not permitted.")

        def uniqueObject(pairs):
            result = {}

            for key, value in pairs:
                if key in result:
                    raise ValueError(f"{rootPath}: duplicate JSON key {key!r}.")

                result[key] = value

            return result

        document = json.loads(jsonString, object_pairs_hook=uniqueObject, parse_constant=rejectConstant)

        def resolveLoadedClass(moduleName, qualname, baseClass, path):
            """Resolve ordinary module/class namespaces without dynamic imports."""
            if not isinstance(moduleName, str) or not moduleName or not isinstance(qualname, str) or not qualname:
                raise ValueError(f"{path}: invalid class identity.")

            current = sys.modules.get(moduleName)

            if current is None:
                raise ValueError(f"{path}: module {moduleName!r} is not loaded.")

            for index, part in enumerate(qualname.split(".")):
                if not part.isidentifier() or (index and not isinstance(current, type)):
                    raise ValueError(f"{path}: invalid class name {qualname!r}.")

                current = vars(current).get(part)

                if current is None:
                    raise ValueError(f"{path}: unknown class {moduleName}.{qualname}.")

            if not isinstance(current, type) or not issubclass(current, baseClass):
                raise TypeError(f"{path}: {moduleName}.{qualname} is not a {baseClass.__name__} class.")

            return current

        def encodedDocument(value):
            # Invoke the shared encoder, not a second type-encoding system.
            return json.loads(pglSerialize.toJSON(value, strict=True))

        def sameDocument(left, right):
            # Comparing encoded JSON avoids Python treating True, 1, and 1.0
            # as interchangeable when checking configuration preservation.
            leftText = json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False)
            rightText = json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)
            return leftText == rightText

        def decode(value, path, isRoot=False):
            if isinstance(value, list):
                decoded = []
                normalized = []

                for index, item in enumerate(value):
                    result, expected = decode(item, f"{path}[{index}]")
                    decoded.append(result)
                    normalized.append(expected)

                return decoded, normalized

            if not isinstance(value, dict):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{path}: nonfinite JSON number.")

                return value, value

            present = set(value).intersection(markers)

            if not present:
                decoded = {}
                normalized = {}

                for key, item in value.items():
                    decoded[key], normalized[key] = decode(item, f"{path}.{key}")

                return decoded, normalized

            # A plain HasTraits record intentionally contains both markers.
            if present == {"__hastraits__", "__class__"}:
                marker = "__hastraits__"
            elif len(present) == 1:
                marker = next(iter(present))
            else:
                raise ValueError(f"{path}: conflicting serialization markers: {sorted(present)}.")

            def requireFields(names):
                if set(value) != set(names):
                    raise ValueError(f"{path}: invalid fields for {marker}.")

            if marker != "__class__" and value[marker] is not True:
                raise ValueError(f"{path}: {marker} must be true.")

            expected = dict(value)

            if marker == "__class__":
                className = value["__class__"]

                if not isinstance(className, str):
                    raise ValueError(f"{path}: class name must be a string.")

                objectClass = loadAsClass if isRoot and loadAsClass is not None else classRegistry.get(className)

                if objectClass is None:
                    raise ValueError(f"{path}: unknown serialization class {className!r}.")

                data = {}
                expected = {"__class__": objectClass.__name__}

                for key, item in value.items():
                    if key != "__class__":
                        data[key], expected[key] = decode(item, f"{path}.{key}")

                token = _pglStrictRestorePath.set(path)

                try:
                    obj = objectClass.fromJSONdict(data, filename=filename)
                except Exception as error:
                    raise ValueError(f"{path}: could not restore {className}: {error}") from error
                finally:
                    _pglStrictRestorePath.reset(token)

                if not isinstance(obj, objectClass):
                    raise TypeError(f"{path}: {className}.fromJSONdict() returned an unexpected object.")

            elif marker == "__hastraits__":
                if not {"__module__", "__class__"}.issubset(value):
                    raise ValueError(f"{path}: incomplete HasTraits identity.")

                objectClass = resolveLoadedClass(value["__module__"], value["__class__"], HasTraits, path)
                obj = objectClass()
                traits = obj.traits()
                expected = {key: value[key] for key in ("__hastraits__", "__class__", "__module__")}

                for key, item in value.items():
                    if key in expected:
                        continue

                    if key not in traits or traits[key].metadata.get("serialize", True) is False:
                        raise ValueError(f"{path}.{key}: unknown or nonpersistent trait.")

                    restored, expected[key] = decode(item, f"{path}.{key}")

                    try:
                        setattr(obj, key, restored)
                    except Exception as error:
                        raise ValueError(f"{path}.{key}: trait restoration failed: {error}") from error

            elif marker == "__exception__":
                requireFields(("__exception__", "__module__", "__qualname__", "message"))

                if any(not isinstance(value[key], str) for key in ("__module__", "__qualname__", "message")) or not value["__module__"] or not value["__qualname__"]:
                    raise ValueError(f"{path}: invalid exception record.")

                obj = pglSerializedError(value["__module__"], value["__qualname__"], value["message"])

            elif marker == "__enum__":
                requireFields(("__enum__", "__module__", "__qualname__", "name"))
                enumClass = resolveLoadedClass(value["__module__"], value["__qualname__"], Enum, path)

                if not isinstance(value["name"], str):
                    raise ValueError(f"{path}: enum member must be a string.")

                try:
                    obj = enumClass[value["name"]]
                except KeyError as error:
                    raise ValueError(f"{path}: unknown enum member {value['name']!r}.") from error

            elif marker in ("__datetime__", "__path__"):
                requireFields((marker, "value"))

                if not isinstance(value["value"], str):
                    raise ValueError(f"{path}: expected a string value.")

                obj = datetime.fromisoformat(value["value"]) if marker == "__datetime__" else Path(value["value"])

            elif marker == "__range__":
                requireFields(("__range__", "start", "stop", "step"))

                if any(item.__class__ is not int for item in (value["start"], value["stop"], value["step"])) or value["step"] == 0:
                    raise ValueError(f"{path}: invalid range.")

                obj = range(value["start"], value["stop"], value["step"])

            elif marker == "__tuple__":
                requireFields(("__tuple__", "items"))

                if not isinstance(value["items"], list):
                    raise ValueError(f"{path}: tuple items must be a list.")

                items, normalized = decode(value["items"], f"{path}.items")
                obj = tuple(items)
                expected = {"__tuple__": True, "items": normalized}

            elif marker == "__simplenamespace__":
                data = {}
                expected = {"__simplenamespace__": True}

                for key, item in value.items():
                    if key != "__simplenamespace__":
                        data[key], expected[key] = decode(item, f"{path}.{key}")

                obj = SimpleNamespace(**data)

            elif marker == "__numpy__":
                requireFields(("__numpy__", "dtype", "shape", "data"))

                if not isinstance(value["dtype"], str):
                    raise ValueError(f"{path}: array dtype must be a string.")

                dtype = np.dtype(value["dtype"])
                shape = value["shape"]

                if dtype.fields is not None or dtype.kind not in "biufU":
                    raise ValueError(f"{path}: dtype {dtype} is unsupported in strict metadata.")

                if not isinstance(shape, list) or any(size.__class__ is not int or size < 0 for size in shape):
                    raise ValueError(f"{path}: invalid array shape.")

                data, normalized = decode(value["data"], f"{path}.data")
                obj = np.array(data, dtype=dtype).reshape(shape)
                expected["data"] = normalized

            else:
                raise ValueError(f"{path}: unsupported serialization marker.")

            # Custom restorers may swallow their own errors or drop fields.
            # Re-encoding detects a changed persistent representation.
            try:
                actual = encodedDocument(obj)
            except Exception as error:
                raise ValueError(f"{path}: restored object cannot be strictly encoded: {error}") from error

            if not sameDocument(actual, expected):
                raise ValueError(f"{path}: restoration changed or discarded serialized metadata.")

            return obj, expected

        result, normalized = decode(document, rootPath, isRoot=True)
        return result
    
    ##########################
    # fromJSON
    ##########################
    @classmethod
    def fromJSON(cls, jsonString, filename=None, loadAsClass=None, *, strict=False):
        """Restore metadata using the existing PGL JSON representation.

        strict=False preserves legacy decoding behavior.

        strict=True rejects malformed records, unknown classes, duplicate
        JSON keys, rejected trait values, and changed/dropped metadata.
        Strictness propagates through nested custom restoration methods.

        In strict mode, loadAsClass overrides only the root PGL object.
        Constructors and restoration hooks are trusted application code;
        strict decoding is validation, not a sandbox.
        """
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")

        # Nested calls made by custom restoration hooks inherit strictness.
        inheritedPath = _pglStrictRestorePath.get()

        if strict or inheritedPath is not None:
            return cls._fromJSONStrict(jsonString, filename=filename, loadAsClass=loadAsClass, rootPath=inheritedPath or "root")

        classRegistry = pglGetAllSubclasses(pglSerialize)

        def decodeObject(dct):
            if "__exception__" in dct:
                expectedFields = {"__exception__", "__module__", "__qualname__", "message"}

                if set(dct) != expectedFields or dct["__exception__"] is not True or any(not isinstance(dct[key], str) for key in ("__module__", "__qualname__", "message")) or not dct["__module__"] or not dct["__qualname__"]:
                    raise ValueError("Invalid serialized exception record.")

                return pglSerializedError(originalModule=dct["__module__"], originalClass=dct["__qualname__"], message=dct["message"])

            if "__enum__" in dct:
                expectedFields = {"__enum__", "__module__", "__qualname__", "name"}

                if set(dct) != expectedFields or dct["__enum__"] is not True or any(not isinstance(dct[key], str) or not dct[key] for key in ("__module__", "__qualname__", "name")):
                    raise ValueError("Invalid serialized enum record.")

                moduleName = dct["__module__"]
                className = dct["__qualname__"]
                memberName = dct["name"]
                enumClass = sys.modules.get(moduleName)

                if enumClass is None:
                    raise ValueError(f"Enum module is not loaded: {moduleName!r}")

                for part in className.split("."):
                    if not part.isidentifier():
                        raise ValueError(f"Invalid enum class name: {className!r}")

                    enumClass = getattr(enumClass, part, None)

                if not isinstance(enumClass, type) or not issubclass(enumClass, Enum):
                    raise ValueError(f"Unknown enum class: {moduleName}.{className}")

                try:
                    return enumClass[memberName]
                except KeyError as error:
                    raise ValueError(f"Unknown enum member: {moduleName}.{className}.{memberName}") from error

            if "__datetime__" in dct:
                return datetime.fromisoformat(dct["value"])

            if "__path__" in dct:
                return Path(dct["value"])

            if "__numpy__" in dct:
                return np.array(dct["data"], dtype=dct["dtype"]).reshape(dct["shape"])

            if "__range__" in dct:
                expectedFields = {"__range__", "start", "stop", "step"}

                if set(dct) != expectedFields or dct["__range__"] is not True:
                    raise ValueError("Invalid serialized range record.")

                if any(type(dct[key]) is not int for key in ("start", "stop", "step")):
                    raise ValueError("Serialized range start, stop, and step must be integers.")

                if dct["step"] == 0:
                    raise ValueError("Serialized range step must not be zero.")

                return range(dct["start"], dct["stop"], dct["step"])

            if "__tuple__" in dct:
                return tuple(dct["items"])

            if "__simplenamespace__" in dct:
                dct.pop("__simplenamespace__")
                return SimpleNamespace(**dct)

            if "__hastraits__" in dct:
                moduleName = dct.pop("__module__")
                className = dct.pop("__class__")
                dct.pop("__hastraits__")

                try:
                    import importlib
                    module = importlib.import_module(moduleName)
                    traitClass = getattr(module, className)
                    obj = traitClass()

                    for key, value in dct.items():
                        setattr(obj, key, value)

                    return obj

                except (ImportError, AttributeError) as error:
                    pglMessages.warning(f"(pglSerialize) Could not restore HasTraits object {className}: {error}", level=1)
                    return dct

            if "__class__" in dct:
                className = dct.pop("__class__")

                if loadAsClass is not None:
                    objectClass = loadAsClass
                elif className in classRegistry:
                    objectClass = classRegistry[className]
                else:
                    availableClasses = sorted(classRegistry)
                    pglMessages.warning(f"(pglSerialize) Could not restore object of class '{className}' from '{filename}'.\nKnown classes: {availableClasses}")
                    dct["__class__"] = className
                    return dct

                return objectClass.fromJSONdict(dct, filename=filename)

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
        """Restore metadata with context-aware strict validation.

        Strict restoration rejects unknown/nonpersistent fields and propagates
        assignment errors. Permissive restoration warns and continues.

        Diagnostics never read the destination field or depend on an optional
        type-description helper.
        """
        strictPath = _pglStrictRestorePath.get()

        if not isinstance(data, dict):
            raise TypeError("Restoration data must be a dictionary.")

        if not isinstance(self, HasTraits):
            self.__dict__.update(data)
            return

        traits = self.traits()
        extraKeys = set(data) - set(traits)

        if strictPath is not None:
            if extraKeys and not self._serializeUnregisteredFields:
                raise ValueError(f"{strictPath}: unknown fields: {sorted(extraKeys)}.")

            for key in set(data).intersection(traits):
                if traits[key].metadata.get("serialize", True) is False:
                    raise ValueError(f"{strictPath}.{key}: cannot restore a nonpersistent trait.")

        for key, trait in traits.items():
            if key not in data:
                if pglSerialize.verbose and trait.metadata.get("serialize", True) is not False:
                    pglMessages.message(f"'{key}' not found in {filename}; current/default value left uninspected.", verbose=pglSerialize.verbose)

                continue

            try:
                value = data[key]

                if callable(typeConverter):
                    value = typeConverter(key, value)

                setattr(self, key, value)

            except TraitError as error:
                if strictPath is not None:
                    raise ValueError(f"{strictPath}.{key}: trait restoration failed: {error}") from error

                # Describe the trait definition and supplied metadata, not the
                # destination value, whose getter could trigger lazy loading.
                expectedType = trait.__class__.__name__
                suppliedType = data[key].__class__.__name__
                pglMessages.warning(f"(pglSerialize) Trait mismatch restoring {self.__class__.__name__}.{key} from '{filename}': expected trait {expectedType}, received {suppliedType}. Field not read for diagnostics. Error: {error}", level=1)

            except Exception as error:
                if strictPath is not None:
                    raise ValueError(f"{strictPath}.{key}: restoration failed: {error}") from error

                pglMessages.warning(f"(pglSerialize) Error restoring {self.__class__.__name__}.{key} from '{filename}': {error.__class__.__name__}: {error}. Field not read for diagnostics.", level=1)

        if not extraKeys:
            return

        if self._serializeUnregisteredFields:
            for key in extraKeys:
                try:
                    setattr(self, key, data[key])
                except Exception as error:
                    if strictPath is not None:
                        raise ValueError(f"{strictPath}.{key}: unregistered-field restoration failed: {error}") from error

                    pglMessages.warning(f"Could not restore unregistered attribute '{key}': {error}", level=1)

        elif pglSerialize.verbose:
            pglMessages.message(f"Unknown keys in {filename} (ignored): {sorted(extraKeys)}", verbose=pglSerialize.verbose)
          
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
                
                
    def saveStrict(self, filename, filesystem=None, filesystemPrefix=None):
        """Serialize before opening the destination; propagate failures."""
        from .pglBase import pglBase

        filename = str(filename)
        dataPath = posixpath.dirname(filename) or "."
        fileName = posixpath.splitext(posixpath.basename(filename))[0] + ".json"

        filesystem, dataPath, _ = pglBase.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath,
            filesystemPrefix=filesystemPrefix,
            create=True,
        )
        if filesystem is None: raise OSError(f"(pglSerialize) Could not resolve filesystem for '{filename}'")

        filename = posixpath.join(dataPath, fileName)

        # Resolve and encode everything before opening/truncating the destination.
        jsonString = self.toJSON(filename=filename)

        with filesystem.open(filename, "w", encoding="utf-8") as fileHandle:
            written = fileHandle.write(jsonString)
            if written != len(jsonString): raise OSError(f"(pglSerialize) Incomplete write to '{filename}'")

        return filename

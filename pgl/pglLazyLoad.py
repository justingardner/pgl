################################################################
#   filename: pglLazyLoad.py
#    purpose: lazy load mixin for handling fields that need
#             to be lazy loaded
#         by: JLG
#       date: Oct 12, 2026
################################################################
from dataclasses import dataclass, field


_pglMissing = object()


@dataclass(frozen=True)
class pglFsspecBacking:
    """An unopened resource reference, kept separate from serialized values."""

    fs: object
    path: str

    def open(self, mode="rb", **kwargs): return self.fs.open(self.path, mode=mode, **kwargs)


@dataclass
class pglLazyRuntime:
    pending: dict = field(default_factory=dict)
    retained: dict = field(default_factory=dict)
    resolving: set = field(default_factory=set)


class pglLazyLoad:
    """
    Place before HasTraits or a HasTraits-derived base in the inheritance list.

    Eligible traits:
        .tag(lazyload=True)
        .tag(lazyload=True, loader="_loadData")

    Loaders accept a backing reference and return the field value.
    They must not assign the field they are loading.

    Runtime state is not trait data and must be excluded from serialization.
    """

    def __getattribute__(self, name):
        runtime = object.__getattribute__(self, "__dict__").get("_lazyRuntimeState")
        if runtime is not None and name in runtime.pending:
            return object.__getattribute__(self, "_lazyResolve")(name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        runtime = object.__getattribute__(self, "__dict__").get("_lazyRuntimeState")
        if runtime is not None and (name in runtime.pending or name in runtime.retained or name in runtime.resolving):
            return self._lazyAssign(name, lambda: super(pglLazyLoad, self).__setattr__(name, value))
        return super().__setattr__(name, value)

    def set_trait(self, name, value):
        return self._lazyAssign(name, lambda: super(pglLazyLoad, self).set_trait(name, value))

    def _lazyRuntime(self):
        namespace = object.__getattribute__(self, "__dict__")
        runtime = namespace.get("_lazyRuntimeState")
        if runtime is None:
            runtime = pglLazyRuntime()
            namespace["_lazyRuntimeState"] = runtime
        return runtime

    def _lazyTrait(self, name):
        trait = self.traits().get(name)
        if trait is None: raise AttributeError(f"Unknown trait: {name!r}")
        if trait.metadata.get("lazyload") is not True: raise ValueError(f"Trait {name!r} is not tagged lazyload=True")
        return trait

    def lazyBind(self, name, backing, *, replace=False):
        """
        Attach backing without opening it or evaluating a trait default.

        Existing materialized values, including None, and existing backing
        require replace=True.
        """
        trait = self._lazyTrait(name)
        loaderName = trait.metadata.get("loader", "lazyLoadSerialized")
        if not isinstance(loaderName, str): raise TypeError(f"Loader metadata for {name!r} must name a method")
        if not callable(getattr(self, loaderName, None)): raise TypeError(f"Loader {loaderName!r} is not a callable member")

        runtime = self._lazyRuntime()
        if name in runtime.resolving: raise RuntimeError(f"Cannot bind {name!r} while it is loading")

        occupied = name in self._trait_values or name in runtime.pending or name in runtime.retained
        if occupied and not replace:
            raise ValueError(f"Trait {name!r} already has a value or backing; use replace=True to replace it explicitly")

        runtime.retained.pop(name, None)
        runtime.pending[name] = backing

    def lazyIsPending(self, name):
        """Inspect status without reading the field or evaluating its default."""
        self._lazyTrait(name)
        runtime = object.__getattribute__(self, "__dict__").get("_lazyRuntimeState")
        return runtime is not None and name in runtime.pending

    def _lazyAssign(self, name, operation):
        runtime = object.__getattribute__(self, "__dict__").get("_lazyRuntimeState")
        if runtime is None: return operation()
        if name in runtime.resolving: raise RuntimeError(f"Cannot assign {name!r} inside its own loader")

        pending = runtime.pending.pop(name, _pglMissing)
        retained = runtime.retained.pop(name, _pglMissing)
        if pending is _pglMissing and retained is _pglMissing: return operation()

        previous = self._trait_values.get(name, _pglMissing)
        try:
            # Notification callbacks must see the assignment, not the old backing.
            return operation()
        except BaseException:
            current = self._trait_values.get(name, _pglMissing)

            # Restore backing if assignment failed before committing a value.
            # An observer may instead raise after commitment; keep that assignment.
            if current is previous:
                if pending is not _pglMissing: runtime.pending[name] = pending
                if retained is not _pglMissing: runtime.retained[name] = retained
            raise

    def _lazyResolve(self, name):
        runtime = self._lazyRuntime()
        if name in runtime.resolving: raise RuntimeError(f"Recursive lazy loading of {name!r}")

        backing = runtime.pending[name]
        trait = self._lazyTrait(name)
        loaderName = trait.metadata.get("loader", "lazyLoadSerialized")

        runtime.resolving.add(name)
        try:
            value = getattr(self, loaderName)(backing)
        finally:
            runtime.resolving.remove(name)

        # Loader failures leave backing pending. During assignment, remove it
        # so validation and notification callbacks cannot restart the loader.
        runtime.pending.pop(name)
        try:
            super().set_trait(name, value)
        except BaseException:
            runtime.pending[name] = backing
            raise

        # Retain the reference without materializing the returned value.
        # Managed resource cleanup/local materialization is a later step.
        runtime.retained[name] = backing
        return super().__getattribute__(name)

    def lazyLoadSerialized(self, backing):
        """Read through the backing reference and decode with pglSerialize."""
        from .pglSerialize import pglSerialize

        with backing.open("rb") as fileHandle: jsonString = fileHandle.read()
        return pglSerialize.fromJSON(jsonString, filename=backing.path)
    
    def lazySaveField(self, name, destination):
        """Save a persistent lazy-capable field through its selected writer."""
        trait = self._lazyTrait(name)
        if trait.metadata.get("serialize", True) is False:
            raise ValueError(f"Trait {name!r} has serialize=False")

        writerName = trait.metadata.get("writer", "lazyWriteSerialized")
        if not isinstance(writerName, str): raise TypeError(f"Writer metadata for {name!r} must name a method")

        writer = getattr(self, writerName, None)
        if not callable(writer): raise TypeError(f"Writer {writerName!r} is not a callable member")

        # Resolve only after checking persistence and writer configuration.
        value = getattr(self, name)
        return writer(value, destination)

    def lazyWriteSerialized(self, value, destination):
        """Encode with pglSerialize before opening the exact destination path."""
        from .pglSerialize import pglSerialize

        # Use object overrides when present; otherwise use the existing encoder
        # for supported standalone values, including None, lists, and arrays.
        if isinstance(value, pglSerialize): jsonString = value.toJSON(filename=destination.path)
        else: jsonString = pglSerialize.toJSON(value, filename=destination.path)

        with destination.open("w", encoding="utf-8") as fileHandle:
            written = fileHandle.write(jsonString)
            if written != len(jsonString): raise OSError(f"Incomplete write to '{destination.path}'")

        return destination.path
################################################################
#   filename: pglStorage.py
#    purpose: Standalone scientific file storage, integrity
#             verification, and figure export
#         by: JLG
#       date: Oct 5, 2026
################################################################

import hashlib
import json
import os
import posixpath
import re
from pathlib import Path

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from .pglMessages import pglMessages


class pglStorage:
    """Provide persistence operations independent of PostgreSQL.

    Public capabilities:
        Save/load standalone scientific matrix bundles.
        Export individual figures or vertically stacked figure rows.
        Hash files and copy files with byte-identity verification.
        Encode deterministic JSON and validate portable matrix contents.

    Database checkpoint publication, manifests, registered locations,
    lineage, and lifecycle operations belong to pglDatabase.

    Filesystem writes are not database transactions. Failed publication can
    leave incomplete files or directories. Individual methods document their
    completion and overwrite behavior.

    Runtime filesystem credentials are supplied by callers and are not
    written into matrix metadata by this class.
    """

    MATRIX_CONTRACT = "pgl.matrix.zarr.v1"
    MAX_METADATA_BYTES = 16 * 1024 * 1024
    COPY_CHUNK_BYTES = 1024 * 1024

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    # ---------------------------------------------------------------------
    # Scientific matrices: standalone publication and loading
    #
    # Called directly for filesystem-only persistence, and by pglDatabase
    # when publishing or opening separately registered matrix products.
    # These methods do not query or register database records.
    # ---------------------------------------------------------------------

    # +++++++++++++++++++++++++ saveDataMatrix +++++++++++++++++++++++++++++
    @classmethod
    def saveDataMatrix(cls, matrix, dataPath, *, filesystem=None, storageOptions=None, chunks=None):
        """Publish a new standalone Zarr matrix bundle.

        Creates:
            <dataPath>/matrix-<new UUID>/data.zarr/
            <dataPath>/matrix-<new UUID>/matrix.json

        matrix.json is written last and serves as the publication descriptor.
        A missing or invalid descriptor prevents normal loading.

        Usage:
            savedPath = pglStorage.saveDataMatrix(matrix, "~/data/matrices")
            savedPath = pglStorage.saveDataMatrix(matrix, remoteRoot, filesystem=filesystem)

        Called by:
            Users saving standalone matrices.
            pglDatabase when publishing session or dataset products.

        Parameters:
            matrix:
                A pglDataMatrix with xarray scientific contents.
            dataPath:
                Parent directory for a new UUID-named bundle.
            filesystem:
                Optional resolved fsspec filesystem.
            storageOptions:
                Options for resolving dataPath when filesystem is omitted.
            chunks:
                Optional xarray chunk specification applied before writing.

        Returns:
            Filesystem-qualified path of the newly published bundle.

        Reads scientific arrays to write Zarr. Validates metadata and encoding
        first, but does not perform a full read-back verification of Zarr data.

        Does not change the working matrix's revision ID or register SQL rows.
        Lightweight identity defaults may be initialized during serialization.

        Existing bundles are not intentionally overwritten. Failure may leave
        an incomplete bundle. Concurrent mutation of the matrix is unsupported.
        Exceptions propagate.
        """
        from uuid import uuid4

        import xarray as xr

        from .pglData import pglDataMatrix
        from .pglSerialize import pglSerialize

        if not isinstance(matrix, pglDataMatrix):
            raise TypeError("matrix must be a pglDataMatrix.")

        cls.requireSafeSave(matrix)
        data = matrix.requireData()
        cls.validateDataMatrixEncoding(data)

        filesystem, root = cls._resolveStoragePath(dataPath, filesystem=filesystem, storageOptions=storageOptions, label="dataPath")
        revisionID = str(uuid4())
        bundlePath = posixpath.join(root, f"matrix-{revisionID}")
        zarrPath = posixpath.join(bundlePath, "data.zarr")

        # Materialize lightweight identity defaults before strict encoding.
        # Do not manufacture new construction provenance during saving.
        _ = matrix.uuid
        _ = matrix.pglVersion
        _ = matrix.github

        sourceMetadataText = matrix.toJSON(strict=True)
        metadataObject = pglSerialize.fromJSON(sourceMetadataText, strict=True)

        if type(metadataObject) is not type(matrix):
            raise TypeError("Matrix metadata restored as a different class.")

        if metadataObject.data is not None:
            raise ValueError("Matrix restoration must not construct scientific data.")

        metadataObject.matrixRevisionID = revisionID

        # A fresh artifact must not inherit a possibly stale content hash.
        metadataObject.contentHash = None
        metadataObject.hashAlgorithm = None
        metadataObject.hashSchemaVersion = None

        publishedMetadataText = metadataObject.toJSON(strict=True)
        restoredMetadata = pglSerialize.fromJSON(publishedMetadataText, strict=True)

        if restoredMetadata.toJSON(strict=True) != publishedMetadataText:
            raise ValueError("Published matrix metadata did not round-trip exactly.")

        if isinstance(data, xr.DataArray):
            valueName = "__pgl_matrix_values__"

            while valueName in data.coords or valueName in data.sizes:
                valueName += "_"

            persisted = data.to_dataset(name=valueName)
            kind = "DataArray"
            arrayName = data.name
        else:
            persisted = data
            kind = "Dataset"
            valueName = None
            arrayName = None

        # Remove backend-specific encoding without changing the source object.
        persisted = persisted.drop_encoding()

        if chunks is not None:
            persisted = persisted.chunk(chunks)

        descriptor = {
            "contractVersion": cls.MATRIX_CONTRACT,
            "zarrFormat": 2,
            "matrixRevisionID": revisionID,
            "kind": kind,
            "valueName": valueName,
            "arrayName": arrayName,
            "metadata": json.loads(publishedMetadataText),
        }

        # Preserve the existing matrix descriptor representation. Database
        # comparisons may use encodeJSON() for canonical structural comparison.
        descriptorBytes = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")

        if len(descriptorBytes) > cls.MAX_METADATA_BYTES:
            raise ValueError("Matrix descriptor exceeds the metadata size limit.")

        if filesystem.exists(bundlePath):
            raise FileExistsError(bundlePath)

        filesystem.makedirs(bundlePath, exist_ok=False)
        mapper = filesystem.get_mapper(zarrPath)

        # Preserve zeros and Boolean False as values, not missing sentinels.
        # Floating-point NaNs remain scientific array values.
        encoding = {name: {"_FillValue": None} for name in persisted.variables}
        persisted.to_zarr(mapper, mode="w-", consolidated=True, zarr_format=2, encoding=encoding)

        # Detect bookkeeping and attachment changes before publishing.
        # This cannot detect every in-place mutation of scientific arrays.
        cls.requireSafeSave(matrix)

        if matrix.data is not data or matrix.toJSON(strict=True) != sourceMetadataText:
            raise ValueError("Matrix attachment or metadata changed during publication.")

        descriptorPath = posixpath.join(bundlePath, "matrix.json")

        with filesystem.open(descriptorPath, "xb") as target:
            if target.write(descriptorBytes) != len(descriptorBytes):
                raise OSError("Incomplete matrix publication descriptor write.")

        return filesystem.unstrip_protocol(bundlePath)

    # +++++++++++++++++++++++++ loadDataMatrix +++++++++++++++++++++++++++++
    @classmethod
    def loadDataMatrix(cls, dataPath, *, filesystem=None, storageOptions=None):
        """Open a standalone matrix with lazy Zarr-backed scientific variables.

        Usage:
            matrix = pglStorage.loadDataMatrix(savedBundlePath)

        Called by:
            Users opening standalone matrix bundles.
            pglDatabase after resolving and checking a registered revision.

        dataPath identifies the bundle directory containing matrix.json and
        data.zarr, not the parent directory supplied to saveDataMatrix().

        Reads the descriptor and consolidated Zarr metadata. Does not scan
        scientific values or calculate payload hashes.

        Dimension indexes are not created automatically, avoiding eager
        coordinate reads. Label-based selection may require set_xindex().

        Returns:
            A restored pglDataMatrix with lazy scientific contents.

        The wrapper class must already be imported for strict restoration.
        Call matrix.close() when its resources are no longer needed.

        Does not connect to PostgreSQL or confirm catalog correspondence.
        Runtime credentials are not added to serialized metadata.
        Exceptions propagate; opened xarray resources are closed on failure.
        """
        import xarray as xr

        from .pglData import pglDataMatrix
        from .pglSerialize import pglSerialize

        filesystem, root = cls._resolveStoragePath(dataPath, filesystem=filesystem, storageOptions=storageOptions, label="dataPath")
        descriptor = cls._readMatrixDescriptor(filesystem, root)
        revisionID = descriptor["matrixRevisionID"]

        matrix = pglSerialize.fromJSON(json.dumps(descriptor["metadata"], allow_nan=False), strict=True)

        if not isinstance(matrix, pglDataMatrix):
            raise TypeError("Descriptor metadata did not restore a pglDataMatrix.")

        if matrix.data is not None:
            raise ValueError("Metadata restoration unexpectedly attached scientific data.")

        if matrix.matrixRevisionID != revisionID:
            raise ValueError("Matrix revision disagrees with its descriptor.")

        mapper = filesystem.get_mapper(posixpath.join(root, "data.zarr"))
        opened = xr.open_zarr(mapper, consolidated=True, chunks={}, mask_and_scale=False, decode_times=False, decode_timedelta=False, create_default_indexes=False)

        try:
            if descriptor["kind"] == "DataArray":
                valueName = descriptor["valueName"]

                if set(opened.data_vars) != {valueName}:
                    raise ValueError("DataArray payload has unexpected scientific variables.")

                data = opened[valueName].rename(descriptor["arrayName"])
                data.set_close(opened.close)
            else:
                data = opened

            cls.validateDataMatrixEncoding(data)
            matrix.data = data

            # A runtime origin reference, not a claim that exposed arrays
            # remain unchanged after loading.
            matrix._payloadReference = {
                "contractVersion": descriptor["contractVersion"],
                "matrixRevisionID": revisionID,
                "path": filesystem.unstrip_protocol(root),
            }

            return matrix

        except BaseException:
            opened.close()
            raise
        
    @classmethod
    def inventoryDataMatrixBundle(cls, dataPath, *, filesystem=None, storageOptions=None, expectedDescriptor=None, expectedFiles=None, progress=None):
        """Inventory current matrix-bundle bytes without opening scientific arrays.

        Reads bounded JSON metadata and streams every stored file for SHA-256.
        Requires individual Zarr metadata to match consolidated metadata.

        expectedDescriptor:
            Optional catalog descriptor to compare structurally.

        expectedFiles:
            Optional prior inventory: sorted records containing exactly
            name, sha256, and size_bytes. Requires exact file-set/byte equality.

        Returns:
            descriptor: Validated standalone publication descriptor.
            files: Sorted bundle-relative byte identities.
            bundleBytes: Sum of actual hashed file sizes.

        This establishes an observed stored-byte inventory, NOT complete Zarr
        semantic validity, required-chunk coverage, or integrity since original
        publication. Those require additional validation or trusted prior hashes.

        Detectable symlinks and special entries are rejected. Generic backend
        safety depends on accurate ls() metadata; local paths receive explicit
        symlink checks. Listings must be authoritative and complete.

        Concurrent mutation is unsupported. Repeated listing/size checks detect
        some changes but do not provide an atomic filesystem snapshot.

        Does not query SQL, decode chunks, construct matrices, write files, or
        close a caller-supplied filesystem. Errors propagate.
        Optional progress:
            A pglProgress-compatible reporter supporting call() and section().
            Reports this helper's work only; never finishes the parent display.
            Caller callback errors propagate like the existing hash/copy helpers.

        """
        from copy import deepcopy
        from .pglProgress import pglProgress

        if progress is not None: progress(0, "Listing and validating matrix metadata")

        from fsspec.implementations.local import LocalFileSystem

        filesystem, root = cls._resolveStoragePath(
            dataPath,
            filesystem=filesystem,
            storageOptions=storageOptions,
            label="matrix bundle",
        )
        root = root.rstrip("/")
        if not root: raise ValueError("A matrix bundle cannot be the filesystem root.")

        def validateName(name):
            """Require a portable bundle-relative POSIX filename."""
            if (
                not isinstance(name, str) or not name
                or name.startswith("/") or "\\" in name
                or any(character in name for character in ("\x00", "\n", "\r"))
                or any(part in ("", ".", "..") for part in name.split("/"))
            ):
                raise ValueError(f"Invalid matrix inventory filename: {name!r}")

        def requireLocalPathSafe(path):
            """Reject local symlinks, including symlinked ancestor directories."""
            if isinstance(filesystem, LocalFileSystem):
                localPath = Path(path).absolute()
                for candidate in (localPath, *localPath.parents):
                    if candidate.is_symlink():
                        raise ValueError(f"Symlinked matrix inventory path: {candidate}")

        def listFiles():
            """Enumerate through immediate children without silently skipping entries."""
            requireLocalPathSafe(root)
            if not filesystem.isdir(root): raise NotADirectoryError(root)

            pending = [root]
            visited = set()
            files = {}

            while pending:
                directory = pending.pop()
                if directory in visited: raise ValueError("Repeated matrix bundle directory.")
                visited.add(directory)

                for entry in filesystem.ls(directory, detail=True):
                    path = filesystem._strip_protocol(entry["name"])
                    relativeName = posixpath.relpath(path, root)
                    validateName(relativeName)

                    # A backend must return immediate children of this directory.
                    if posixpath.dirname(path.rstrip("/")) != directory:
                        raise ValueError("Backend listing returned a non-child bundle entry.")

                    requireLocalPathSafe(path)
                    if entry.get("islink") or entry.get("type") in ("link", "symlink"):
                        raise ValueError(f"Symlink inside matrix bundle: {relativeName}")

                    if relativeName != "matrix.json" and relativeName != "data.zarr" and not relativeName.startswith("data.zarr/"):
                        raise ValueError(f"Unexpected matrix bundle entry: {relativeName}")

                    kind = entry.get("type")
                    if kind == "directory":
                        if relativeName == "matrix.json":
                            raise ValueError("matrix.json must be a file.")
                        pending.append(path)
                    elif kind == "file":
                        if relativeName == "data.zarr":
                            raise ValueError("data.zarr must be a directory.")
                        if relativeName in files:
                            raise ValueError("Duplicate matrix bundle filename.")
                        size = entry.get("size")
                        if type(size) is not int or size < 0:
                            raise ValueError("Backend returned an invalid file size.")
                        files[relativeName] = size
                    else:
                        raise ValueError(f"Unsupported bundle entry type: {relativeName}")

            required = {
                "matrix.json",
                "data.zarr/.zgroup",
                "data.zarr/.zattrs",
                "data.zarr/.zmetadata",
            }
            if not required.issubset(files):
                raise ValueError("Matrix bundle lacks required descriptor/Zarr metadata.")

            return files

        def readJSON(name):
            """Read bounded metadata, rejecting duplicate keys and nonfinite numbers."""
            def uniqueKeys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result: raise ValueError(f"Duplicate JSON key in {name}: {key!r}")
                    result[key] = value
                return result

            def rejectConstant(value):
                raise ValueError(f"Invalid JSON constant in {name}: {value}")

            with filesystem.open(posixpath.join(root, name), "rb") as source:
                content = source.read(cls.MAX_METADATA_BYTES + 1)
            if len(content) > cls.MAX_METADATA_BYTES:
                raise ValueError(f"Matrix metadata exceeds the size limit: {name}")

            document = json.loads(content, object_pairs_hook=uniqueKeys, parse_constant=rejectConstant)
            # Also reject nonfinite numbers produced by oversized JSON exponents.
            cls.encodeJSON(document)
            return document, {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }

        # Validate prior verification evidence before any payload hashing.
        priorFiles = None
        if expectedFiles is not None:
            if type(expectedFiles) is not list:
                raise TypeError("expectedFiles must be a list or None.")

            priorFiles = {}
            names = []
            for record in expectedFiles:
                if type(record) is not dict or set(record) != {"name", "sha256", "size_bytes"}:
                    raise ValueError("Invalid expected matrix file record.")
                name = record["name"]
                validateName(name)
                if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
                    raise ValueError("Invalid expected matrix SHA-256.")
                if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
                    raise ValueError("Invalid expected matrix file size.")
                names.append(name)
                priorFiles[name] = dict(record)

            if names != sorted(set(names)):
                raise ValueError("Expected matrix filenames must be unique and sorted.")

        if expectedDescriptor is not None:
            cls._validateMatrixDescriptor(expectedDescriptor)
            expectedDescriptor = deepcopy(expectedDescriptor)

        before = listFiles()
        descriptor, descriptorIdentity = readJSON("matrix.json")
        cls._validateMatrixDescriptor(descriptor)

        if expectedDescriptor is not None and cls.encodeJSON(descriptor) != cls.encodeJSON(expectedDescriptor):
            raise ValueError("Stored matrix descriptor differs from catalog metadata.")

        consolidated, consolidatedIdentity = readJSON("data.zarr/.zmetadata")
        if (
            type(consolidated) is not dict
            or set(consolidated) != {"metadata", "zarr_consolidated_format"}
            or type(consolidated["zarr_consolidated_format"]) is not int
            or consolidated["zarr_consolidated_format"] != 1
            or type(consolidated["metadata"]) is not dict
        ):
            raise ValueError("Unsupported consolidated Zarr metadata.")

        metadata = consolidated["metadata"]
        metadataNames = set()
        metadataIdentities = {
            "matrix.json": descriptorIdentity,
            "data.zarr/.zmetadata": consolidatedIdentity,
        }

        for relativeName, expectedDocument in metadata.items():
            validateName(relativeName)
            if posixpath.basename(relativeName) not in {".zgroup", ".zattrs", ".zarray"}:
                raise ValueError("Unexpected consolidated Zarr metadata key.")

            name = f"data.zarr/{relativeName}"
            if name not in before:
                raise ValueError(f"Consolidated metadata has no stored counterpart: {name}")

            document, identity = readJSON(name)
            if type(document) is not dict:
                raise ValueError(f"Zarr metadata must contain an object: {name}")
            if cls.encodeJSON(document) != cls.encodeJSON(expectedDocument):
                raise ValueError(f"Consolidated and individual metadata disagree: {name}")

            metadataNames.add(name)
            metadataIdentities[name] = identity

        actualMetadataNames = {
            name for name in before
            if name.startswith("data.zarr/")
            and posixpath.basename(name) in {".zgroup", ".zattrs", ".zarray"}
        }
        if metadataNames != actualMetadataNames:
            raise ValueError("Consolidated metadata does not cover all stored Zarr metadata.")

        group = metadata.get(".zgroup")
        if type(group) is not dict or type(group.get("zarr_format")) is not int or group["zarr_format"] != 2:
            raise ValueError("Expected a Zarr format 2 root group.")

        records = []
        work = [
            {"name": name, "size_bytes": before[name]}
            for name in sorted(before)
        ]
        hashProgress = progress.section(0.05, 0.95) if progress is not None else None
        for item, fileProgress in pglProgress.fileSections(work, hashProgress):
            name = item["name"]
            identity = cls.hashFile(
                filesystem, posixpath.join(root, name),
                progress=fileProgress, expectedSize=before[name],
                description="Hashing stored matrix file",
            )
            if identity["size_bytes"] != before[name]:
                raise ValueError(f"Matrix file size changed during inventory: {name}")
            if name in metadataIdentities and identity != metadataIdentities[name]:
                raise ValueError(f"Matrix metadata changed during inventory: {name}")

            record = {"name": name, **identity}
            if priorFiles is not None and priorFiles.get(name) != record:
                raise ValueError(f"Matrix bytes differ from the expected inventory: {name}")
            records.append(record)

        if priorFiles is not None and set(priorFiles) != set(before):
            raise ValueError("Matrix file set differs from the expected inventory.")
        if progress is not None: progress(0.95, "Rechecking matrix file inventory")
        if listFiles() != before:
            raise ValueError("Matrix file inventory changed while hashing.")
        if progress is not None: progress(1, "Matrix byte inventory verified")

        return {
            "descriptor": descriptor,
            "files": records,
            "bundleBytes": sum(record["size_bytes"] for record in records),
        }
        
    @classmethod
    def inventoryDataMatrixBundleLayout(
        cls, dataPath, *, requireAllChunks, filesystem=None,
        storageOptions=None, expectedDescriptor=None, expectedFiles=None, progress=None,
    ):
        """Inventory stored bytes and validate the supported matrix chunk layout.

        Reuses inventoryDataMatrixBundle(); does not change its byte-inventory
        contract. Supports a flat Zarr-v2 group containing portable arrays.

        requireAllChunks:
            Required explicit Boolean policy.
            True: reject absent grid chunks, including an absent scalar chunk.
            False: report absent grid chunks without interpreting their values.

            Strict coverage is an eligibility policy, not proof that a sparse
            Zarr store is corrupt. No missing scientific values are inferred.

        Returns the existing descriptor/files/bundleBytes fields plus:
            layout:
                policy and aggregate chunk counts.
                arrays: sorted per-array schema and chunk counts.

        Validates:
            Flat group/array hierarchy and consolidated metadata coverage
            through the existing byte-inventory helper.
            Array shape, chunks, portable stored dtype, order, and separator.
            xarray dimension metadata and consistent dimension lengths.
            DataArray valueName correspondence.
            Canonical chunk keys, dimensionality, bounds, and optional coverage.

        Codec configurations receive structural checks only. Fill-value
        semantics, codec availability, encoded chunk contents, scientific
        values, and complete xarray reconstruction are NOT validated.

        Metadata read here is bound to the returned byte inventory by SHA-256.
        Concurrent mutation remains unsupported; this is not an atomic snapshot.

        Does not query SQL, decode chunks, construct matrices, write files,
        close caller-owned filesystems, or set bundleInventoryComplete.
        Optional progress:
            A pglProgress-compatible reporter supporting call() and section().
            Reports this helper's work only; never finishes the parent display.
            Caller callback errors propagate like the existing hash/copy helpers.

        """
        import hashlib
        import json
        import math
        import posixpath
        import re

        import numpy as np

        if type(requireAllChunks) is not bool:
            raise TypeError("requireAllChunks must be an explicit Boolean.")

        filesystem, root = cls._resolveStoragePath(
            dataPath,
            filesystem=filesystem,
            storageOptions=storageOptions,
            label="matrix bundle",
        )
        root = root.rstrip("/")

        inventory = cls.inventoryDataMatrixBundle(
            root,
            filesystem=filesystem,
            expectedDescriptor=expectedDescriptor,
            expectedFiles=expectedFiles,
            progress=progress.section(0, 0.90) if progress is not None else None,
        )
        if progress is not None: progress(0.90, "Validating matrix array and chunk layout")
        records = {record["name"]: record for record in inventory["files"]}

        # The existing helper parsed this exact metadata with bounded reads,
        # duplicate-key rejection, and strict JSON validation. Bind this second
        # read to those inventoried bytes before using its contents.
        metadataName = "data.zarr/.zmetadata"
        with filesystem.open(posixpath.join(root, metadataName), "rb") as source:
            content = source.read(cls.MAX_METADATA_BYTES + 1)

        if len(content) > cls.MAX_METADATA_BYTES:
            raise ValueError("Consolidated Zarr metadata exceeds the size limit.")

        metadataRecord = records[metadataName]
        if (
            len(content) != metadataRecord["size_bytes"]
            or hashlib.sha256(content).hexdigest() != metadataRecord["sha256"]
        ):
            raise ValueError("Consolidated metadata changed after byte inventory.")

        metadata = json.loads(content)["metadata"]

        def validateSchemaName(value, label):
            """Use the writer's portable variable/dimension name restrictions."""
            if (
                not isinstance(value, str) or not value or value in (".", "..")
                or any(character in value for character in ("/", "\\", "\x00"))
            ):
                raise ValueError(f"Invalid {label}: {value!r}")

        def validateCodecConfig(config, label):
            """Check configuration structure without importing/constructing codecs."""
            if (
                type(config) is not dict
                or not isinstance(config.get("id"), str)
                or not config["id"]
            ):
                raise ValueError(f"Invalid {label} codec configuration.")

        # saveDataMatrix writes an xarray Dataset at the root, not nested groups.
        arrayNames = set()
        for name in metadata:
            if name in (".zgroup", ".zattrs"): continue

            parts = name.split("/")
            if len(parts) != 2 or parts[1] not in (".zarray", ".zattrs"):
                raise ValueError(f"Unsupported matrix metadata hierarchy: {name}")

            validateSchemaName(parts[0], "array name")
            arrayNames.add(parts[0])

        arrays = {}
        dimensionSizes = {}
        requiredFields = {
            "zarr_format", "shape", "chunks", "dtype", "compressor",
            "fill_value", "order", "filters",
        }
        allowedFields = requiredFields | {"dimension_separator"}

        for arrayName in sorted(arrayNames):
            arrayKey = f"{arrayName}/.zarray"
            attrsKey = f"{arrayName}/.zattrs"

            if arrayKey not in metadata or attrsKey not in metadata:
                raise ValueError(f"Array lacks .zarray or .zattrs metadata: {arrayName}")

            array = metadata[arrayKey]
            attrs = metadata[attrsKey]

            if (
                not requiredFields.issubset(array)
                or not set(array).issubset(allowedFields)
                or type(array["zarr_format"]) is not int
                or array["zarr_format"] != 2
            ):
                raise ValueError(f"Unsupported Zarr array metadata: {arrayName}")

            shape = array["shape"]
            chunks = array["chunks"]
            if (
                type(shape) is not list or type(chunks) is not list
                or len(shape) != len(chunks)
                or any(type(size) is not int or size < 0 for size in shape)
                or any(type(size) is not int or size <= 0 for size in chunks)
            ):
                raise ValueError(f"Invalid array shape/chunks: {arrayName}")

            # Validate dtype metadata only; do not allocate an array.
            dtypeText = array["dtype"]
            if not isinstance(dtypeText, str):
                raise ValueError(f"Unsupported stored dtype: {arrayName}")

            try:
                dtype = np.dtype(dtypeText)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid stored dtype: {arrayName}") from error

            if (
                dtype.fields is not None or dtype.subdtype is not None
                or dtype.kind not in "biufU" or dtype.itemsize <= 0
                or (dtype.kind == "f" and dtype.itemsize not in (2, 4, 8))
            ):
                raise ValueError(f"Unsupported stored dtype for {arrayName}: {dtypeText}")

            order = array["order"]
            separator = array.get("dimension_separator", ".")
            if order not in ("C", "F") or separator not in (".", "/"):
                raise ValueError(f"Invalid array order or dimension separator: {arrayName}")

            if array["compressor"] is not None:
                validateCodecConfig(array["compressor"], f"{arrayName} compressor")

            filters = array["filters"]
            if filters is not None:
                if type(filters) is not list:
                    raise ValueError(f"Invalid filters for array: {arrayName}")
                for config in filters:
                    validateCodecConfig(config, f"{arrayName} filter")

            dimensions = attrs.get("_ARRAY_DIMENSIONS")
            if type(dimensions) is not list or len(dimensions) != len(shape):
                raise ValueError(f"Invalid _ARRAY_DIMENSIONS for array: {arrayName}")

            for dimension, size in zip(dimensions, shape):
                validateSchemaName(dimension, "dimension name")
                if dimension in dimensionSizes and dimensionSizes[dimension] != size:
                    raise ValueError(f"Inconsistent stored dimension length: {dimension}")
                dimensionSizes[dimension] = size

            chunkGrid = [
                (size + chunk - 1) // chunk
                for size, chunk in zip(shape, chunks)
            ]
            arrays[arrayName] = {
                "name": arrayName,
                "shape": list(shape),
                "chunks": list(chunks),
                "dimensions": list(dimensions),
                "dtype": dtypeText,
                "order": order,
                "dimensionSeparator": separator,
                "chunkGrid": chunkGrid,
                # Scalars have one grid chunk; zero-length axes yield zero.
                "expectedChunkCount": math.prod(chunkGrid),
                "observedChunkCount": 0,
            }

        descriptor = inventory["descriptor"]
        if descriptor["kind"] == "DataArray" and descriptor["valueName"] not in arrays:
            raise ValueError("DataArray descriptor valueName has no stored array.")

        metadataFiles = {
            "matrix.json", "data.zarr/.zmetadata",
            *(f"data.zarr/{name}" for name in metadata),
        }

        for name in records:
            if name in metadataFiles: continue

            if not name.startswith("data.zarr/"):
                raise ValueError(f"Unexpected matrix file: {name}")

            relativeName = name[len("data.zarr/"):]
            parts = relativeName.split("/", 1)
            if len(parts) != 2 or parts[0] not in arrays:
                raise ValueError(f"Stored file has no owning array: {name}")

            array = arrays[parts[0]]
            chunkKey = parts[1]

            if not array["shape"]:
                if chunkKey != "0":
                    raise ValueError(f"Invalid scalar chunk key: {name}")
            else:
                coordinates = chunkKey.split(array["dimensionSeparator"])
                if (
                    len(coordinates) != len(array["shape"])
                    or any(
                        re.fullmatch(r"0|[1-9][0-9]*", coordinate) is None
                        for coordinate in coordinates
                    )
                ):
                    raise ValueError(f"Invalid chunk key: {name}")

                if any(
                    int(coordinate) >= bound
                    for coordinate, bound in zip(coordinates, array["chunkGrid"])
                ):
                    raise ValueError(f"Chunk coordinate outside array grid: {name}")

            array["observedChunkCount"] += 1

        arrayReports = []
        for arrayName in sorted(arrays):
            array = arrays[arrayName]
            array["missingChunkCount"] = (
                array["expectedChunkCount"] - array["observedChunkCount"]
            )

            # Canonical keys and unique filenames guarantee unique coordinates.
            # Count equality therefore establishes grid coverage without building
            # or iterating over every theoretically possible chunk coordinate.
            if array["missingChunkCount"] < 0:
                raise ValueError(f"Too many stored chunks for array: {arrayName}")

            if requireAllChunks and array["missingChunkCount"]:
                raise ValueError(
                    f"Strict chunk coverage failed for {arrayName}: "
                    f"{array['missingChunkCount']} grid chunk(s) absent."
                )

            arrayReports.append(array)

        expectedCount = sum(array["expectedChunkCount"] for array in arrayReports)
        observedCount = sum(array["observedChunkCount"] for array in arrayReports)
        if progress is not None: progress(1, "Matrix byte inventory and layout verified")

        return {
            **inventory,
            "layout": {
                "requireAllChunks": requireAllChunks,
                "allGridChunksPresent": expectedCount == observedCount,
                "expectedChunkCount": expectedCount,
                "observedChunkCount": observedCount,
                "missingChunkCount": expectedCount - observedCount,
                "arrays": arrayReports,
            },
        }
        
    @classmethod
    def copyDataMatrixBundle(
        cls, dataPath, destinationPath, *, expectedDescriptor, expectedFiles,
        requireAllChunks, sourceFilesystem=None, destinationFilesystem=None,
        sourceStorageOptions=None, destinationStorageOptions=None, progress=None,
    ):
        """Copy a matrix bundle unchanged against previously captured identities.

        destinationPath:
            Exact new bundle path, not a parent directory. Its parent must
            already exist. Existing destinations are rejected, never reused.

        expectedDescriptor / expectedFiles:
            Required catalog/publication descriptor and captured byte inventory.
            Source preflight, streamed copying, read-back verification, and
            final destination inventory must agree with this evidence.

        requireAllChunks:
            Required explicit Boolean coverage policy. Passed unchanged to
            source and destination layout validation.

        Returns:
            destinationPath: Filesystem-qualified copied bundle path.
            inventory: Verified destination descriptor/files/bytes/layout.

        Copies stored bytes without generating a revision UUID, rewriting
        provenance, decoding chunks, or constructing scientific arrays.

        Copies matrix.json last, but this is NOT atomic publication. A failure
        during or after its copy may leave a descriptor at the destination.
        Successful return is required before later catalog registration.
        No SQL registration or bundleInventoryComplete promotion occurs here.

        Failure retains all created directories and partial files. Nothing is
        deleted, rolled back, retried, or silently substituted.

        Local destination paths receive ancestor-symlink checks. Generic
        backend safety depends on authoritative metadata and exclusive-create
        semantics. Concurrent writers/mutation and aliased remote namespaces
        are unsupported. Overlap checks cover local paths and matching
        filesystem instances; they do not establish arbitrary remote identity.

        Does not close caller-supplied or fsspec-resolved filesystems.
        Errors propagate.
        Optional progress:
            A pglProgress-compatible reporter supporting call() and section().
            Reports this helper's work only; never finishes the parent display.
            Caller callback errors propagate like the existing hash/copy helpers.

        """
        from copy import deepcopy
        from .pglProgress import pglProgress
        from pathlib import Path
        import posixpath

        from fsspec.implementations.local import LocalFileSystem

        if type(requireAllChunks) is not bool:
            raise TypeError("requireAllChunks must be an explicit Boolean.")
        if expectedDescriptor is None or expectedFiles is None:
            raise ValueError("A captured descriptor and file inventory are required.")

        # Snapshot caller evidence before filesystem work. The inventory helper
        # validates its full structure, ordering, hashes, and catalog contract.
        expectedDescriptor = deepcopy(expectedDescriptor)
        expectedFiles = deepcopy(expectedFiles)

        sourceFilesystem, sourceRoot = cls._resolveStoragePath(
            dataPath,
            filesystem=sourceFilesystem,
            storageOptions=sourceStorageOptions,
            label="source matrix bundle",
        )
        destinationFilesystem, destinationRoot = cls._resolveStoragePath(
            destinationPath,
            filesystem=destinationFilesystem,
            storageOptions=destinationStorageOptions,
            label="destination matrix bundle",
        )
        sourceRoot = sourceRoot.rstrip("/")
        destinationRoot = destinationRoot.rstrip("/")

        if not sourceRoot or not destinationRoot:
            raise ValueError("A matrix bundle cannot be the filesystem root.")

        def requireDestinationSafe(path):
            """Reject local symlinked ancestors without resolving them away."""
            if isinstance(destinationFilesystem, LocalFileSystem):
                localPath = Path(path).absolute()
                for candidate in (localPath, *localPath.parents):
                    if candidate.is_symlink():
                        raise ValueError(f"Symlinked matrix destination path: {candidate}")

        def pathsOverlap(first, second):
            """Compare normalized paths within an already identified namespace."""
            first = posixpath.normpath(first)
            second = posixpath.normpath(second)
            return (
                first == second
                or first.startswith(second.rstrip("/") + "/")
                or second.startswith(first.rstrip("/") + "/")
            )

        requireDestinationSafe(destinationRoot)

        bothLocal = (
            isinstance(sourceFilesystem, LocalFileSystem)
            and isinstance(destinationFilesystem, LocalFileSystem)
        )
        if bothLocal:
            first = str(Path(sourceRoot).absolute())
            second = str(Path(destinationRoot).absolute())
            if pathsOverlap(first, second):
                raise ValueError("Source and destination matrix paths must not overlap.")
        elif sourceFilesystem is destinationFilesystem:
            if pathsOverlap(sourceRoot, destinationRoot):
                raise ValueError("Source and destination matrix paths must not overlap.")

        if destinationFilesystem.exists(destinationRoot):
            raise FileExistsError(destinationRoot)

        parent = posixpath.dirname(destinationRoot) or "."
        requireDestinationSafe(parent)
        if not destinationFilesystem.isdir(parent):
            raise NotADirectoryError(
                f"Matrix destination parent must already exist: {parent}"
            )

        # Fail before creating a destination if the source no longer matches
        # the evidence captured by export inspection.
        sourceInventory = cls.inventoryDataMatrixBundleLayout(
            sourceRoot,
            filesystem=sourceFilesystem,
            expectedDescriptor=expectedDescriptor,
            expectedFiles=expectedFiles,
            requireAllChunks=requireAllChunks,
            progress=progress.section(0, 0.20) if progress is not None else None,
        )

        # These checks are defensive, not a concurrency/atomicity guarantee.
        requireDestinationSafe(destinationRoot)
        if destinationFilesystem.exists(destinationRoot):
            raise FileExistsError(destinationRoot)
        destinationFilesystem.makedirs(destinationRoot, exist_ok=False)
        requireDestinationSafe(destinationRoot)

        records = sourceInventory["files"]
        descriptorRecord = next(
            record for record in records if record["name"] == "matrix.json"
        )
        copyOrder = [
            record for record in records if record["name"] != "matrix.json"
        ] + [descriptorRecord]

        copyProgress = progress.section(0.20, 0.80) if progress is not None else None
        for record, fileProgress in pglProgress.fileSections(copyOrder, copyProgress):
            name = record["name"]
            target = posixpath.join(destinationRoot, name)
            requireDestinationSafe(target)

            cls.copyVerifiedFile(
                sourceFilesystem,
                posixpath.join(sourceRoot, name),
                destinationFilesystem,
                destinationRoot,
                {
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                },
                objectKey=name,
                progress=fileProgress,
            )

        # Check the complete observed destination file set, metadata, hashes,
        # and layout rather than relying solely on per-file copy success.
        destinationInventory = cls.inventoryDataMatrixBundleLayout(
            destinationRoot,
            filesystem=destinationFilesystem,
            expectedDescriptor=expectedDescriptor,
            expectedFiles=expectedFiles,
            requireAllChunks=requireAllChunks,
            progress=progress.section(0.80, 0.99) if progress is not None else None,
        )
        if destinationInventory != sourceInventory:
            raise ValueError("Copied matrix inventory differs from source preflight.")
        if progress is not None: progress(1, "Matrix bundle copy verified")

        return {
            "destinationPath": destinationFilesystem.unstrip_protocol(destinationRoot),
            "inventory": destinationInventory,
        }

    # ---------------------------------------------------------------------
    # Figures: individual export and row collection
    #
    # Optional presentation output. Failures warn instead of failing the
    # scientific analysis. No database configuration is involved.
    # ---------------------------------------------------------------------

    # ++++++++++++++++++++++++++++ exportFig ++++++++++++++++++++++++++++++++
    @classmethod
    def exportFig(cls, fig, filename=None, *, filesystem=None, storageOptions=None, dpi=100, overwrite=True):
        """Export a Matplotlib figure or request browser display.

        Usage:
            pglStorage.exportFig(fig, "~/reports/result.png")
            pglStorage.exportFig(fig)

        Without filename:
            Render a temporary PNG and request a browser tab.
            Do not produce inline Jupyter display.
            Retain the temporary file for asynchronous browser access.

        With filename:
            Write through fsspec.
            Support PNG, SVG, PDF, JPEG, and WebP.
            Replace missing or unsupported extensions with .png.

        Plain local paths expand ~. Remote paths use backend semantics.
        Rendering completes before the destination is opened.

        Returns:
            Resolved output path, or temporary browser-image path.
            None on export/render failure.

        Browser-launch failure warns but returns the rendered image path.
        Writes are not transactional and may leave partial output.
        Exclusive creation depends on backend support when overwrite=False.

        Does not close, clear, or display the figure inline.
        """
        import tempfile
        import webbrowser
        from io import BytesIO

        from matplotlib.figure import Figure

        try:
            if not isinstance(fig, Figure):
                raise TypeError("fig must be a Matplotlib Figure.")

            if type(dpi) is not int or dpi <= 0:
                raise ValueError("dpi must be a positive integer.")

            if type(overwrite) is not bool:
                raise ValueError("overwrite must be a boolean.")

            displayOnly = filename is None
            outputFormat = "png"
            destination = None
            destinationFS = None

            if displayOnly:
                if filesystem is not None or storageOptions:
                    raise ValueError("Filesystem options require an export filename.")
            else:
                destinationFS, destination, outputFormat = cls._resolveFigureDestination(filename, filesystem=filesystem, storageOptions=storageOptions)
                cls._checkOutputDestination(destinationFS, destination, overwrite=overwrite)

            # Finish rendering before creating or truncating output.
            with BytesIO() as buffer:
                fig.savefig(buffer, format=outputFormat, dpi=dpi, facecolor="white", transparent=False)
                content = buffer.getvalue()

            if displayOnly:
                with tempfile.NamedTemporaryFile(prefix="pgl-figure-", suffix=".png", delete=False) as temporary:
                    if temporary.write(content) != len(content):
                        raise OSError("Incomplete temporary figure write.")

                    displayPath = Path(temporary.name).resolve()

                try:
                    opened = webbrowser.open(displayPath.as_uri(), new=2)
                except Exception:
                    opened = False

                if opened:
                    pglMessages.message(f"Requested browser display: {displayPath}")
                else:
                    pglMessages.warning(f"Could not open a browser. The figure is available at {displayPath}.", level=1)

                return str(displayPath)

            parent = posixpath.dirname(destination)

            if parent:
                destinationFS.makedirs(parent, exist_ok=True)

            mode = "wb" if overwrite else "xb"

            with destinationFS.open(destination, mode) as target:
                if target.write(content) != len(content):
                    raise OSError("Incomplete figure write.")

            pglMessages.message(f"Figure exported as {outputFormat.upper()}: {posixpath.basename(destination)}")
            return destination

        except FileExistsError:
            pglMessages.warning("Figure export skipped: the output file already exists. Use overwrite=True to replace it.", level=1)
            return None

        except IsADirectoryError:
            pglMessages.warning("Figure export skipped: the destination is a directory. Supply a filename such as evoked.png.", level=1)
            return None

        except Exception as error:
            # Backend exception text can expose credentials or signed URLs.
            # KeyboardInterrupt and SystemExit still propagate.
            pglMessages.warning(f"Could not export/display figure ({type(error).__name__}). Check the figure, filename, format support, storage options, and permissions. Processing will continue.", level=1)
            return None

    # ++++++++++++++++++++++++++ exportFigRows ++++++++++++++++++++++++++++++
    @staticmethod
    def exportFigRows(filename=None, *, filesystem=None, storageOptions=None, dpi=100, overwrite=True, maxPixels=100_000_000):
        """Create a collector that exports figure rows as one PNG.

        Usage:
            with pglStorage.exportFigRows("report.png") as output:
                output.add(firstFigure)
                output.add(secondFigure)
                output.finish()

        add() renders each figure immediately to temporary storage.
        finish() explicitly assembles and publishes the combined image.
        Leaving the with block only cleans up; it does not auto-publish.

        filename=None creates a silent no-op collector.
        Does not open a browser or produce inline display.
        Does not close the supplied figures.

        Returns:
            A pglFigureRowExport collector.

        maxPixels limits final image area, not total process memory.
        """
        return pglFigureRowExport(filename, filesystem=filesystem, storageOptions=storageOptions, dpi=dpi, overwrite=overwrite, maxPixels=maxPixels)

    ###################################
    # make valid filename
    ###################################
    @staticmethod
    def makeValidFilename(nameStr):
        
        # replace anything that isn't a letter, number, dash or underscore with an underscore
        cleanStr = re.sub(r'[^A-Za-z0-9_-]', '_', nameStr)
        
        # collapse runs of underscores into one
        cleanStr = re.sub(r'_+', '_', cleanStr)
        
        # strip leading/trailing underscores and dots
        cleanStr = cleanStr.strip('_.')
        
        # guard against an empty result
        return cleanStr.lower() or 'unnamed'


    ################################################################
    # validate filesystem: used for fsspec 
    ################################################################
    @staticmethod
    def validateFilesystem(filesystem=None, dataPath=None, filesystemPrefix="", create=False, requireExists=True):
        '''
        Return a valid fsspec filesystem and data path.

        Filesystem selection priority:
            1. Explicitly supplied filesystem
            2. Explicitly supplied filesystemPrefix
            3. Prefix embedded in dataPath
            4. Local filesystem ("file")
        '''

        import fsspec
        from fsspec.core import url_to_fs
        from fsspec import AbstractFileSystem

        # ------------------------------------------------------------
        # 1. Normalize dataPath and separate any embedded prefix
        # ------------------------------------------------------------
        dataPath = str(dataPath) if dataPath is not None else None
        originalDataPath = dataPath
        if dataPath is None or dataPath == "":
            dataPath = ""
            dataPathPrefix = ""

        else:
            dataPath = str(dataPath)

            if "://" in dataPath:
                protocol, _, rest = dataPath.partition("://")
                authority = rest.split("/", 1)[0]
                dataPathPrefix = f"{protocol}://{authority}"
                dataPath = rest[len(authority):]
            else:
                dataPathPrefix = ""

        # ------------------------------------------------------------
        # 2. Determine filesystem
        # ------------------------------------------------------------

        if filesystem is not None:

            if not isinstance(filesystem, AbstractFileSystem):
                pglMessages.warning(f"Expected an fsspec AbstractFileSystem, got {type(filesystem).__name__}.")
                return None, dataPath, ""

            dataPath = filesystem._strip_protocol(originalDataPath) if originalDataPath else ""
            filesystemPrefix = filesystemPrefix or dataPathPrefix
                
        elif filesystemPrefix:

            filesystemPrefix = str(filesystemPrefix)
            try:
                filesystem, dataPath = url_to_fs(f"{filesystemPrefix.rstrip('/')}/{dataPath.lstrip('/')}")
            except Exception as e:
                pglMessages.warning(f"Error accessing filesystem {filesystemPrefix}: {e}")

        elif dataPathPrefix:

            filesystemPrefix = dataPathPrefix
            try:
                filesystem, dataPath = url_to_fs(originalDataPath)
            except Exception as e:
                pglMessages.warning(f"Error accessing filesystem {filesystemPrefix}: {e}")
            
        else:

            filesystemPrefix = ""
            filesystem = fsspec.filesystem("file")

        # ------------------------------------------------------------
        # 3. Validate filesystem and dataPath
        # ------------------------------------------------------------
        try:
            if not filesystem.exists(dataPath):
                if create:
                    filesystem.makedirs(dataPath, exist_ok=True)
                elif requireExists:
                    pglMessages.warning(f"{dataPath} does not exist")
                    filesystem = None
        except Exception as e:
                pglMessages.warning(f"Error accessing {dataPath}: {e}")
                filesystem = None
            
        return filesystem, dataPath, filesystemPrefix
    # ---------------------------------------------------------------------
    # Reusable file operations
    #
    # Used by standalone callers and pglDatabase publication code.
    # These functions know nothing about catalog IDs or checkpoint lineage.
    # ---------------------------------------------------------------------

    # +++++++++++++++++++++++++++++ hashFile ++++++++++++++++++++++++++++++++
    @staticmethod
    def hashFile(filesystem, path, chunkSize=1024 * 1024, *, progress=None, expectedSize=None, description="Hashing file"):
        """Compute SHA-256 and byte count with optional progress reporting.

        progress receives (fraction, description).
        expectedSize affects progress estimation only; actual bytes are read
        through EOF and returned for the caller's integrity comparison.

        Does not modify files or register database records.
        """
        if type(chunkSize) is not int or chunkSize <= 0:
            raise ValueError("chunkSize must be a positive integer.")

        if expectedSize is not None and (type(expectedSize) is not int or expectedSize < 0):
            raise ValueError("expectedSize must be a nonnegative integer or None.")

        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None.")

        digest = hashlib.sha256()
        sizeBytes = 0

        def reportProgress(finished=False):
            if progress is None:
                return

            if expectedSize is None:
                fraction = 1.0 if finished else None
                detail = f"{sizeBytes / 1048576:.1f} MiB read"
            else:
                fraction = 1.0 if finished else min(sizeBytes / max(expectedSize, 1), 1.0)
                detail = f"{sizeBytes / 1048576:.1f} / {expectedSize / 1048576:.1f} MiB"

            progress(fraction, f"{description} — {detail}")

        reportProgress()

        with filesystem.open(path, "rb") as source:
            while True:
                chunk = source.read(chunkSize)

                if not chunk:
                    break

                digest.update(chunk)
                sizeBytes += len(chunk)
                reportProgress()

        reportProgress(finished=True)
        return {"sha256": digest.hexdigest(), "size_bytes": sizeBytes}

    # +++++++++++++++++++++++++ copyVerifiedFile ++++++++++++++++++++++++++++
    @classmethod
    def copyVerifiedFile(cls, sourceFilesystem, sourcePath, destinationFilesystem, destinationRoot, expected, *, objectKey, progress=None):
        """Copy exclusively and verify both source and stored bytes.

        progress receives (fraction, description).
        Copying receives half the progress range; read-back verification
        receives the other half.

        Preserves exclusive creation and all content-identity checks.
        Errors propagate and may leave incomplete or unregistered files.
        Does not register database records or publish a checkpoint.
        """
        expectedHash = expected["sha256"]
        expectedSize = expected["size_bytes"]

        if not isinstance(expectedHash, str) or re.fullmatch(r"[0-9a-f]{64}", expectedHash) is None:
            raise ValueError("Expected SHA-256 must contain 64 lowercase hexadecimal characters.")

        if type(expectedSize) is not int or expectedSize < 0:
            raise ValueError("Expected size must be a nonnegative integer.")

        if not isinstance(objectKey, str) or objectKey.startswith("/") or "\\" in objectKey or "\x00" in objectKey or any(part in {"", ".", ".."} for part in objectKey.split("/")):
            raise ValueError("objectKey must be a valid relative POSIX path.")

        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None.")

        protocols = destinationFilesystem.protocol
        protocols = {protocols} if isinstance(protocols, str) else set(protocols)
        copyDescription = "Copying file" if protocols.intersection({"file", "local"}) else "Uploading file"

        if progress is not None:
            progress(0, "Preparing destination")

        destinationPath = posixpath.join(destinationRoot, objectKey)
        destinationFilesystem.makedirs(posixpath.dirname(destinationPath), exist_ok=True)

        digest = hashlib.sha256()
        sizeBytes = 0

        if progress is not None:
            progress(0, f"{copyDescription} — 0.0 / {expectedSize / 1048576:.1f} MiB")

        with sourceFilesystem.open(sourcePath, "rb") as source:
            with destinationFilesystem.open(destinationPath, "xb") as destination:
                while True:
                    chunk = source.read(cls.COPY_CHUNK_BYTES)

                    if not chunk:
                        break

                    sizeBytes += len(chunk)

                    if sizeBytes > expectedSize:
                        raise ValueError(f"Source payload exceeds its expected size: {objectKey}")

                    if destination.write(chunk) != len(chunk):
                        raise OSError(f"Incomplete payload write: {objectKey}")

                    digest.update(chunk)

                    if progress is not None:
                        progress(0.5 * sizeBytes / max(expectedSize, 1), f"{copyDescription} — {sizeBytes / 1048576:.1f} / {expectedSize / 1048576:.1f} MiB")

                if progress is not None:
                    progress(0.5, "Closing copied file")

        if digest.hexdigest() != expectedHash or sizeBytes != expectedSize:
            raise ValueError(f"Source bytes do not match the expected identity: {objectKey}")

        def reportVerification(fraction, description):
            if progress is not None:
                progress(None if fraction is None else 0.5 + 0.5 * fraction, description)

        stored = cls.hashFile(destinationFilesystem, destinationPath, progress=reportVerification if progress is not None else None, expectedSize=expectedSize, description="Verifying copied file")

        if stored != {"sha256": expectedHash, "size_bytes": expectedSize}:
            raise ValueError(f"Stored payload verification failed: {objectKey}")

        if progress is not None:
            progress(1, "Copy verified")

        return {"object_key": objectKey, **stored}

    # ++++++++++++++++++++++++++++ encodeJSON +++++++++++++++++++++++++++++++
    @staticmethod
    def encodeJSON(document):
        """Encode JSON deterministically as UTF-8 bytes.

        Used for stable metadata comparisons and content fingerprints by
        standalone persistence and database checkpoint code.

        Sorts dictionary keys, uses compact separators, preserves Unicode,
        and rejects NaN/infinity in metadata.

        Returns bytes without modifying the input.
        Unsupported values and circular structures raise serialization errors.

        This defines PGL's deterministic encoding; it is not a claim of
        compliance with an external canonical-JSON standard.
        """
        return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

    # ---------------------------------------------------------------------
    # Reusable persistence validation
    #
    # These checks protect standalone exports as well as database publication.
    # They do not establish catalog correspondence or scientific correctness.
    # ---------------------------------------------------------------------

    # ++++++++++++++++++++++++++ requireSafeSave ++++++++++++++++++++++++++++
    @staticmethod
    def requireSafeSave(actionable):
        """Reject an actionable object affected by failed action attempts.

        Called before standalone matrix saving and by pglDatabase before
        publishing scientific working objects.

        Checks runtime failure bookkeeping only. Does not inspect arrays,
        validate database history baselines, or clear failure restrictions.

        Returns None when the guard passes.
        Raises TypeError for a non-actionable object or ValueError when failed
        actions may have left partial changes.

        A passing guard is not proof of scientific validity.
        """
        from .pglPipeline import pglActionable

        if not isinstance(actionable, pglActionable):
            raise TypeError("Scientific saving requires a pglActionable object.")

        failedIDs = tuple(actionable._failedActionIDs)

        if not failedIDs:
            return

        failedIDSet = set(failedIDs)
        names = list(dict.fromkeys(entry.actionName for entry in actionable.actionHistory if entry.actionID in failedIDSet))
        details = ", ".join(names) if names else ", ".join(failedIDs)

        raise ValueError(f"Cannot save: this working object may contain partial changes from failed actions: {details}. Reload saved data and rerun the corrected analysis. A later successful action does not clear this restriction.")

    # ++++++++++++++++++++ validateDataMatrixEncoding +++++++++++++++++++++++
    @staticmethod
    def validateDataMatrixEncoding(data):
        """Validate the portable xarray schema supported by matrix storage.

        Called by saveDataMatrix(), loadDataMatrix(), and database publication
        preflight checks.

        Accepts:
            xarray DataArray or Dataset.
            Real numeric, Boolean, and fixed-width Unicode variables.
            Attributes composed of plain JSON types.

        Checks names, dtype metadata, attributes, and reserved encoding fields.
        Does not scan or compute scientific arrays.

        Returns None on success.
        Raises TypeError or ValueError for unsupported encoding.

        This is structural validation, not scientific-value validation or
        proof that every backend can successfully write the supplied data.
        """
        import math

        import xarray as xr

        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise TypeError("Expected an xarray DataArray or Dataset.")

        # +++++++++++++++++++++++++ validateName ++++++++++++++++++++++++++++
        def validateName(value, label):
            """Reject names incompatible with the portable storage layout."""
            if not isinstance(value, str) or not value or value in (".", "..") or any(character in value for character in ("/", "\\", "\x00")):
                raise ValueError(f"{label} must be a nonempty string without path separators or NUL.")

        # +++++++++++++++++++++++ validateAttribute +++++++++++++++++++++++++
        def validateAttribute(value, path, active):
            """Validate JSON attributes recursively, rejecting active cycles."""
            if value is None or type(value) in (str, bool, int):
                if isinstance(value, str) and "\x00" in value:
                    raise ValueError(f"{path}: attribute strings must not contain NUL.")
                return

            if type(value) is float:
                if not math.isfinite(value):
                    raise ValueError(f"{path}: attribute numbers must be finite.")
                return

            if type(value) not in (list, dict):
                raise TypeError(f"{path}: attributes must use plain JSON types, not {type(value).__name__}.")

            identity = id(value)

            if identity in active:
                raise ValueError(f"{path}: circular attributes are not supported.")

            active.add(identity)

            try:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if not isinstance(key, str) or "\x00" in key:
                            raise ValueError(f"{path}: attribute keys must be strings without NUL.")

                        validateAttribute(child, f"{path}.{key}", active)
                else:
                    for index, child in enumerate(value):
                        validateAttribute(child, f"{path}[{index}]", active)
            finally:
                active.remove(identity)

        for name in data.sizes:
            validateName(name, "Dimension name")

        validateAttribute(dict(data.attrs), "attrs", set())

        if isinstance(data, xr.Dataset):
            variables = [(name, variable, True) for name, variable in data.variables.items()]
        else:
            if data.name is not None:
                validateName(data.name, "DataArray name")

            variables = [(name, variable, True) for name, variable in data.coords.items()]
            variables.append(("<values>", data, False))

        for name, variable, namedVariable in variables:
            if namedVariable:
                validateName(name, "Variable name")

            if variable.dtype.fields is not None or variable.dtype.kind not in "biufU":
                raise TypeError(f"Variable {name!r} has unsupported dtype {variable.dtype}. Provide an explicit portable encoding before publishing.")

            if variable.dtype.kind == "f" and variable.dtype.itemsize not in (2, 4, 8):
                raise TypeError(f"Variable {name!r} uses unsupported floating-point precision.")

            reserved = {"_ARRAY_DIMENSIONS", "_FillValue", "missing_value", "scale_factor", "add_offset", "coordinates"}
            conflicts = reserved.intersection(variable.attrs)

            if conflicts:
                raise ValueError(f"Variable {name!r} has reserved encoding attributes: {sorted(conflicts)}.")

            validateAttribute(dict(variable.attrs), f"variable[{name!r}].attrs", set())

    # =====================================================================
    # IMPLEMENTATION SUPPORT
    # =====================================================================

    # ---------------------------------------------------------------------
    # Filesystem resolution
    #
    # Shared by standalone matrix operations and figure destination handling.
    # Resolution may establish backend connections, but does not write files.
    # ---------------------------------------------------------------------

    # ++++++++++++++++++++++++ _resolveStoragePath ++++++++++++++++++++++++++
    @staticmethod
    def _resolveStoragePath(location, *, filesystem=None, storageOptions=None, label="path"):
        """Resolve a local path or URL without creating its destination.

        Internal helper for matrix saving/loading and figure exports.

        Accepts strings or os.PathLike values. Expands ~ for ordinary local
        paths, but leaves remote home-directory conventions to the backend.

        Returns:
            (filesystem, backendPath)

        A supplied filesystem owns its authentication configuration, so
        nonempty storageOptions cannot also be supplied.
        Resolution and authentication errors propagate.
        """
        if filesystem is not None and storageOptions:
            raise ValueError("Supply filesystem or storageOptions, not both.")

        if location is None:
            raise ValueError(f"{label} is required.")

        try:
            text = os.fspath(location)
        except TypeError as error:
            raise ValueError(f"{label} must be a path or URL.") from error

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} must be a nonempty path or URL.")

        if any(character in text for character in ("\x00", "\n", "\r")):
            raise ValueError(f"{label} must not contain NUL or newline characters.")

        if filesystem is None:
            if "://" not in text and "::" not in text:
                text = str(Path(text).expanduser())

            filesystem, path = url_to_fs(text, **(storageOptions or {}))
        else:
            if not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec AbstractFileSystem.")

            path = filesystem._strip_protocol(text)

        return filesystem, path

    # ---------------------------------------------------------------------
    # Matrix descriptor loading and validation
    #
    # loadDataMatrix() owns the sequence:
    # read descriptor -> validate descriptor -> restore wrapper -> open Zarr.
    # ---------------------------------------------------------------------

    # ++++++++++++++++++++++++ _readMatrixDescriptor ++++++++++++++++++++++++
    @classmethod
    def _readMatrixDescriptor(cls, filesystem, root):
        """Read and validate a bounded matrix.json document.

        Used by loadDataMatrix(). Reads metadata only, not scientific arrays.
        Rejects duplicate JSON keys and nonstandard numeric constants.

        Returns the validated descriptor dictionary.
        File, decoding, and validation failures propagate.
        """
        descriptorPath = posixpath.join(root, "matrix.json")

        with filesystem.open(descriptorPath, "rb") as source:
            content = source.read(cls.MAX_METADATA_BYTES + 1)

        if len(content) > cls.MAX_METADATA_BYTES:
            raise ValueError("Matrix descriptor exceeds the metadata size limit.")

        # +++++++++++++++++++++++++ uniqueObject ++++++++++++++++++++++++++++
        def uniqueObject(pairs):
            """Reject duplicate object keys instead of silently keeping one."""
            result = {}

            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate matrix descriptor key: {key!r}")

                result[key] = value

            return result

        # ++++++++++++++++++++++++ rejectConstant +++++++++++++++++++++++++++
        def rejectConstant(value):
            """Reject NaN and infinity tokens in publication metadata."""
            raise ValueError(f"Invalid JSON constant in matrix descriptor: {value}")

        descriptor = json.loads(content, object_pairs_hook=uniqueObject, parse_constant=rejectConstant)
        cls._validateMatrixDescriptor(descriptor)
        return descriptor

    # ++++++++++++++++++++++ _validateMatrixDescriptor ++++++++++++++++++++++
    @classmethod
    def _validateMatrixDescriptor(cls, descriptor):
        """Validate standalone descriptor structure without filesystem access.

        Used by _readMatrixDescriptor(). Does not reconstruct metadata,
        inspect Zarr, or compare a database revision.

        Returns None on success; malformed descriptors raise ValueError.
        """
        from uuid import UUID

        expectedFields = {"contractVersion", "zarrFormat", "matrixRevisionID", "kind", "valueName", "arrayName", "metadata"}

        if not isinstance(descriptor, dict) or set(descriptor) != expectedFields:
            raise ValueError("Invalid matrix descriptor fields.")

        if descriptor["contractVersion"] != cls.MATRIX_CONTRACT:
            raise ValueError("Unsupported matrix persistence contract.")

        if type(descriptor["zarrFormat"]) is not int or descriptor["zarrFormat"] != 2:
            raise ValueError("Unsupported matrix Zarr format.")

        revisionID = descriptor["matrixRevisionID"]

        if not isinstance(revisionID, str):
            raise ValueError("Matrix revision UUID must be a string.")

        try:
            canonicalRevision = str(UUID(revisionID))
        except (ValueError, AttributeError) as error:
            raise ValueError("Invalid matrix revision UUID.") from error

        if revisionID != canonicalRevision:
            raise ValueError("Matrix revision must use canonical UUID formatting.")

        if descriptor["kind"] not in ("DataArray", "Dataset"):
            raise ValueError("Unsupported matrix container kind.")

        if not isinstance(descriptor["metadata"], dict):
            raise ValueError("Matrix metadata must contain a serialized object.")

        if descriptor["kind"] == "Dataset":
            if descriptor["valueName"] is not None or descriptor["arrayName"] is not None:
                raise ValueError("Dataset descriptor contains DataArray-only fields.")
        else:
            valueName = descriptor["valueName"]
            arrayName = descriptor["arrayName"]

            if not isinstance(valueName, str) or not valueName:
                raise ValueError("DataArray descriptor lacks a stored variable name.")

            if arrayName is not None and not isinstance(arrayName, str):
                raise ValueError("Invalid original DataArray name.")

    # ---------------------------------------------------------------------
    # Figure destination handling
    #
    # Shared by exportFig() and pglFigureRowExport.
    # Helpers raise errors; the optional-export callers translate them into
    # warnings and disable export without suppressing scientific errors.
    # ---------------------------------------------------------------------

    # ++++++++++++++++++++++ _resolveFigureDestination ++++++++++++++++++++++
    @classmethod
    def _resolveFigureDestination(cls, filename, *, filesystem=None, storageOptions=None, pngOnly=False):
        """Resolve and normalize an output filename without creating it.

        Returns (filesystem, backendPath, outputFormat).
        Checks directory misuse and normalizes supported extensions.

        pngOnly=True is used by the row collector.
        Existing-file overwrite policy is checked separately.
        """
        try:
            location = os.fspath(filename)
        except TypeError as error:
            raise ValueError("filename must be a path or URL.") from error

        if not isinstance(location, str) or not location.strip():
            raise ValueError("filename must be a nonempty path or URL.")

        if location.endswith(("/", "\\")):
            raise ValueError("Supply a figure filename, not only a directory.")

        filesystem, destination = cls._resolveStoragePath(location, filesystem=filesystem, storageOptions=storageOptions, label="filename")
        protocols = filesystem.protocol
        protocols = {protocols} if isinstance(protocols, str) else set(protocols)

        if protocols.intersection({"file", "local"}):
            destination = str(Path(destination).expanduser().resolve())

        if not destination or destination.endswith("/"):
            raise ValueError("Supply a figure filename.")

        if filesystem.isdir(destination):
            raise IsADirectoryError("The requested figure destination is a directory.")

        stem, extension = posixpath.splitext(destination)
        extension = extension.lower()

        if pngOnly:
            if extension and extension != ".png":
                pglMessages.warning("Combined figure export uses PNG; the filename extension was changed to .png.", level=0)

            return filesystem, stem + ".png", "png"

        formats = {".png": "png", ".svg": "svg", ".pdf": "pdf", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}

        if extension not in formats:
            if extension:
                pglMessages.warning(f"Unsupported figure extension {extension!r}; exporting as PNG instead.", level=0)

            return filesystem, stem + ".png", "png"

        return filesystem, stem + extension, formats[extension]

    # +++++++++++++++++++++++ _checkOutputDestination +++++++++++++++++++++++
    @staticmethod
    def _checkOutputDestination(filesystem, destination, *, overwrite):
        """Reject an existing directory or a protected existing output file.

        Used before rendering/assembly by figure exporters.
        Does not create files and is not an atomic reservation.

        Exporters also use exclusive creation when overwrite=False.
        """
        if not filesystem.exists(destination):
            return

        if filesystem.isdir(destination):
            raise IsADirectoryError("The normalized destination is a directory.")

        if not overwrite:
            raise FileExistsError("The output already exists.")


# =========================================================================
# FIGURE ROW EXPORT
# =========================================================================

class pglFigureRowExport:
    """Collect rendered figure rows and publish one vertically stacked PNG.

    Normally constructed through pglStorage.exportFigRows().

    Row figures are rendered immediately to temporary files; figure handles
    are not retained. Narrower rows are centered without resizing.

    Lifecycle:
        add(fig) renders and collects a row.
        finish() assembles and publishes output.
        close() discards temporary files without publishing.
        A with block guarantees close(), but does not call finish().

    Export errors warn and disable the collector rather than failing analysis.
    maxPixels limits final image area, not total process memory.

    Direct construction retains its existing conservative defaults.
    The pglStorage.exportFigRows() convenience factory has separate defaults.
    """

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    # +++++++++++++++++++++++++++++ __init__ ++++++++++++++++++++++++++++++++
    def __init__(self, filename, *, filesystem=None, storageOptions=None, dpi=200, overwrite=False, maxPixels=40_000_000):
        """Prepare an optional PNG destination without creating its output.

        filename=None creates a silent no-op collector.

        Creates temporary local storage for requested exports after validating
        the destination. Preparation failures warn and disable the collector.
        Does not open a browser or retain any figure.
        """
        from tempfile import TemporaryDirectory

        self._temporary = None
        self._rows = []
        self._width = 0
        self._height = 0
        self._filesystem = None
        self._destination = None
        self._result = None
        self._closed = False
        self._finished = False
        self._disabled = filename is None
        self._dpi = dpi
        self._overwrite = overwrite
        self._maxPixels = maxPixels
        self._exportRequested = filename is not None

        if self._disabled:
            return

        try:
            if type(dpi) is not int or dpi <= 0:
                raise ValueError("dpi must be a positive integer.")

            if type(overwrite) is not bool:
                raise ValueError("overwrite must be a boolean.")

            if type(maxPixels) is not int or maxPixels <= 0:
                raise ValueError("maxPixels must be a positive integer.")

            filesystem, destination, _ = pglStorage._resolveFigureDestination(filename, filesystem=filesystem, storageOptions=storageOptions, pngOnly=True)
            pglStorage._checkOutputDestination(filesystem, destination, overwrite=overwrite)

            self._filesystem = filesystem
            self._destination = destination
            self._temporary = TemporaryDirectory(prefix="pgl-figure-rows-")

        except Exception as error:
            self._fail("prepare figure export", error)

    # +++++++++++++++++++++++++++++++ add +++++++++++++++++++++++++++++++++++
    def add(self, fig):
        """Render one figure row immediately without retaining its handle.

        Called by plotting code after completing each row.

        Returns True when collected, otherwise False.
        Does not display, clear, or close the supplied figure.

        Checks approximate pixel area before rendering and actual area after.
        Rendering or resource errors disable this optional export.
        """
        if self._disabled or self._closed or self._finished:
            if self._exportRequested and not self._finished:
                pglMessages.warning("This figure row was NOT collected: combined PNG export was cancelled earlier. Notebook display and report generation are continuing, but no combined PNG will be saved.", level=1)

            return False

        try:
            import math

            import matplotlib
            from matplotlib.figure import Figure
            from PIL import Image

            if not isinstance(fig, Figure):
                raise TypeError("fig must be a Matplotlib Figure.")

            widthInches, heightInches = fig.get_size_inches()
            estimatedWidth = math.ceil(float(widthInches) * self._dpi)
            estimatedHeight = math.ceil(float(heightInches) * self._dpi)

            if estimatedWidth <= 0 or estimatedHeight <= 0:
                raise ValueError("Figure dimensions must be positive.")

            estimatedPixels = max(self._width, estimatedWidth) * (self._height + estimatedHeight)

            if estimatedPixels > self._maxPixels:
                pglMessages.warning(f"Combined figure export would exceed maxPixels={self._maxPixels:,}. Reduce dpi, reduce the number of rows, or explicitly increase maxPixels.", level=1)
                self._disabled = True
                self.close()
                return False

            rowPath = Path(self._temporary.name) / f"row{len(self._rows):06d}.png"

            # Preserve the designed canvas size even if global settings use
            # savefig.bbox='tight'.
            with matplotlib.rc_context({"savefig.bbox": None}):
                fig.savefig(rowPath, format="png", dpi=self._dpi, facecolor="white", transparent=False, bbox_inches=None)

            with Image.open(rowPath) as image:
                width, height = image.size

            combinedWidth = max(self._width, width)
            combinedHeight = self._height + height

            if combinedWidth * combinedHeight > self._maxPixels:
                pglMessages.warning(f"Rendered rows exceed maxPixels={self._maxPixels:,}; combined figure export was cancelled.", level=1)
                self._disabled = True
                self.close()
                return False

            self._rows.append((rowPath, width, height))
            self._width = combinedWidth
            self._height = combinedHeight
            return True

        except Exception as error:
            self._fail("render figure row", error)
            return False

    # +++++++++++++++++++++++++++++ finish ++++++++++++++++++++++++++++++++++
    def finish(self):
        """Assemble collected rows and explicitly publish the final PNG.

        Returns the resolved destination on success, otherwise None.
        Repeated calls after success return the original result without
        rewriting output.

        Builds the complete PNG locally before opening the destination.
        Remote writes are streamed, but image assembly requires memory for
        the final canvas and a decoded row.

        A failed write may leave partial output. Temporary files are cleaned
        up on both success and failure.
        """
        if self._finished:
            return self._result

        if self._disabled or self._closed:
            return None

        try:
            if not self._rows:
                pglMessages.warning("No figure rows were collected; no PNG was exported.", level=1)
                self.close()
                return None

            if self._width * self._height > self._maxPixels:
                raise ValueError("Combined image exceeds maxPixels.")

            pglStorage._checkOutputDestination(self._filesystem, self._destination, overwrite=self._overwrite)

            pglMessages.message(f"Assembling {len(self._rows)} figure rows into one PNG ({self._width} × {self._height} pixels)...")
            combinedPath = self._assembleRows()
            self._writeCombinedImage(combinedPath)

            self._result = self._destination
            self._finished = True
            pglMessages.message(f"Combined figure exported: {posixpath.basename(self._destination)}")
            return self._result

        except Exception as error:
            self._fail("save combined figure", error)
            return None

        finally:
            self.close()

    # ++++++++++++++++++++++++++++++ close ++++++++++++++++++++++++++++++++++
    def close(self):
        """Discard temporary files without publishing unfinished output.

        Safe to call repeatedly, including after finish() or export failure.
        Does not remove a completed or partially written destination file.
        Cleanup failures warn rather than replacing an earlier error.
        """
        self._closed = True
        temporary = self._temporary
        self._temporary = None
        self._rows = []

        if temporary is not None:
            try:
                temporary.cleanup()
            except Exception as error:
                pglMessages.warning(f"Could not remove temporary figure-export files ({type(error).__name__}).", level=1)

    # ++++++++++++++++++++++++++++ __enter__ ++++++++++++++++++++++++++++++++
    def __enter__(self):
        """Return this collector for deterministic cleanup in a with block."""
        return self

    # +++++++++++++++++++++++++++++ __exit__ ++++++++++++++++++++++++++++++++
    def __exit__(self, exceptionType, exception, traceback):
        """Clean up without auto-publishing or suppressing caller errors."""
        self.close()
        return False

    # =====================================================================
    # IMPLEMENTATION SUPPORT
    # =====================================================================

    # ---------------------------------------------------------------------
    # Assembly and publication
    #
    # finish() calls these in order. Neither changes the success bookkeeping;
    # finish() records success only after the destination write has completed.
    # ---------------------------------------------------------------------

    # +++++++++++++++++++++++++++ _assembleRows +++++++++++++++++++++++++++++
    def _assembleRows(self):
        """Assemble temporary row images into one complete local PNG.

        Called only by finish() after final dimension and destination checks.
        Centers rows on a white RGB canvas without resizing.

        Returns the temporary combined-image Path.
        Releases Pillow image resources before returning or propagating errors.
        Does not open the final destination.
        """
        from PIL import Image

        combinedPath = Path(self._temporary.name) / "combined.png"
        canvas = Image.new("RGB", (self._width, self._height), color="white")

        try:
            top = 0

            for rowPath, width, height in self._rows:
                with Image.open(rowPath) as source:
                    rowImage = source.convert("RGBA")

                    try:
                        left = (self._width - width) // 2
                        canvas.paste(rowImage, (left, top), rowImage)
                    finally:
                        rowImage.close()

                top += height

            canvas.save(combinedPath, format="PNG", dpi=(self._dpi, self._dpi))

        finally:
            canvas.close()

        return combinedPath

    # ++++++++++++++++++++++++ _writeCombinedImage ++++++++++++++++++++++++++
    def _writeCombinedImage(self, combinedPath):
        """Stream the assembled PNG to the final fsspec destination.

        Called by finish() after local assembly succeeds.
        Creates parent directories and respects the overwrite policy.

        Returns None on success. Write errors propagate to finish(), which
        disables export and warns. Partial destination files are not removed.
        """
        parent = posixpath.dirname(self._destination)

        if parent:
            self._filesystem.makedirs(parent, exist_ok=True)

        mode = "wb" if self._overwrite else "xb"

        with combinedPath.open("rb") as source:
            with self._filesystem.open(self._destination, mode) as target:
                while True:
                    chunk = source.read(pglStorage.COPY_CHUNK_BYTES)

                    if not chunk:
                        break

                    if target.write(chunk) != len(chunk):
                        raise OSError("Incomplete combined figure write.")

    # ---------------------------------------------------------------------
    # Optional-export failure handling
    # ---------------------------------------------------------------------

    # ++++++++++++++++++++++++++++++ _fail ++++++++++++++++++++++++++++++++++
    def _fail(self, operation, error):
        """Disable this optional export, report the category, and clean up.

        Called by construction, add(), and finish() after export failures.
        Does not print backend exception text that might expose credentials.
        Does not suppress exceptions raised by the surrounding analysis.
        """
        self._disabled = True

        if isinstance(error, FileExistsError):
            pglMessages.warning("Combined figure export skipped: the output already exists. Use overwrite=True to replace it.", level=2)
        elif isinstance(error, MemoryError):
            pglMessages.warning("Combined figure export stopped: insufficient memory. Reduce dpi or export fewer rows. Analysis and notebook display will continue.", level=2)
        else:
            pglMessages.warning(f"Could not {operation} ({type(error).__name__}); combined export was disabled. Analysis and notebook display will continue.", level=2)

        self.close()
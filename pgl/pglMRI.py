################################################################
#   filename: pglMRI.py
#    purpose: Class for holding mri data used by pglSession
#         by: JLG
#       date: Oct 31, 2026
################################################################

#############
# Import modules
#############
from copy import deepcopy
from pathlib import Path

from traitlets import Any, Dict, Instance, List, Unicode

from .pglPipeline import pglActionable
from .pglSettings import pglTraitSettings

################################################################
# MRI filesystem backing
################################################################
class pglMRIFileBacking:
    """Deferred MRI source on a caller-owned fsspec filesystem.

    Construction stores references only: no stat, connection, or payload read.

    Supports ordinary local/remote fsspec filesystems and the database's
    pglCheckpointFilesystem. The filesystem must remain available while
    deferred readers depend on it.

    expectedIdentity optionally records the expected SHA-256 and byte count.
    This class validates that record but does not independently verify bytes.
    Materialization/export will enforce it while streaming the source.

    pglCheckpointFilesystem additionally performs its existing verification
    when a file is opened.

    This runtime object must not be included in serialized scan metadata.
    """

    def __init__(self, filesystem, path, *, expectedIdentity=None):
        from fsspec import AbstractFileSystem

        if not isinstance(filesystem, AbstractFileSystem):
            raise TypeError("MRI backing requires an fsspec filesystem.")

        if not isinstance(path, str) or not path.strip():
            raise ValueError("MRI backing path must be a nonblank string.")

        if "\x00" in path:
            raise ValueError("MRI backing path must not contain NUL.")

        # Store immutable identity components rather than retaining a mutable
        # caller dictionary. No filesystem access occurs during validation.
        self._sha256 = None
        self._sizeBytes = None

        if expectedIdentity is not None:
            import re

            if (
                not isinstance(expectedIdentity, dict)
                or set(expectedIdentity) != {"sha256", "size_bytes"}
            ):
                raise ValueError(
                    "MRI identity must contain exactly sha256 and size_bytes."
                )

            digest = expectedIdentity["sha256"]
            size = expectedIdentity["size_bytes"]

            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError("MRI SHA-256 must be 64 lowercase hex characters.")

            if type(size) is not int or not 0 <= size <= 9223372036854775807:
                raise ValueError("MRI byte count must be a nonnegative bigint.")

            self._sha256 = digest
            self._sizeBytes = size

        self.fs = filesystem
        self.path = path

    @property
    def expectedIdentity(self):
        """Return an independent identity record, or None if not established."""
        if self._sha256 is None: return None
        return {"sha256": self._sha256, "size_bytes": self._sizeBytes}

    def open(self):
        """Open a binary source stream; the caller must close the stream.

        No SQL connection is needed. Database sources use the checkpoint
        filesystem's retained payload-access context.

        Does not close the underlying filesystem, which belongs to its owner.
        Read, authentication, and integrity failures propagate to the caller.
        """
        return self.fs.open(self.path, "rb")
    
    def materialize(self, cacheDirectory):
        """Return a verified local NIfTI path, streaming into cache if needed.

        Does not load image headers or voxel arrays.

        With an expected identity:
            Verify and reuse an existing cached copy, or download and check
            source bytes against that identity.

        Without an expected identity:
            Read the source and establish its identity for this call.
            Does not mutate the backing's expectedIdentity.

        Cache files must remain unchanged and available while image readers
        use them. The cache directory must be trusted and caller-owned.

        Progress is synchronous. Failed transfers clean up private staging.
        Published cache files are never silently overwritten or repaired.

        The underlying filesystem remains caller-owned and is not closed.
        """
        import hashlib
        import os
        import tempfile
        from pathlib import Path

        from .pglProgress import pglProgress

        cacheText = os.fspath(cacheDirectory)
        if (
            not isinstance(cacheText, str)
            or not cacheText.strip()
            or "://" in cacheText
            or "::" in cacheText
            or any(character in cacheText for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("cacheDirectory must be an ordinary local path.")

        # Cache filenames retain the format extension required by NiBabel.
        # Source labels and remote paths never become local directory paths.
        lowerPath = self.path.lower()
        if lowerPath.endswith(".nii.gz"):
            filename = "image.nii.gz"
        elif lowerPath.endswith(".nii"):
            filename = "image.nii"
        else:
            raise ValueError("MRI materialization supports .nii and .nii.gz.")

        cacheRoot = Path(cacheText).expanduser().resolve() / "mri"
        expected = self.expectedIdentity

        def cachePath(identity):
            return cacheRoot / f"{identity['sha256']}-{identity['size_bytes']}" / filename

        def verifyFile(path, identity, progress):
            """Hash actual cached bytes, without accepting leaf symlinks."""
            import stat

            if path.parent.is_symlink() or path.is_symlink():
                raise ValueError("MRI cache files/directories must not be symlinks.")

            digest = hashlib.sha256()
            size = 0
            progress(0, "Verifying cached NIfTI")

            with path.open("rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("MRI cache payload must be a regular file.")

                while True:
                    block = source.read(1024 * 1024)
                    if not block: break
                    size += len(block)

                    if size > identity["size_bytes"]:
                        raise ValueError("MRI cache payload exceeds its recorded size.")

                    digest.update(block)
                    progress(
                        size / max(identity["size_bytes"], 1),
                        f"Verifying cached NIfTI — {size:,} / "
                        f"{identity['size_bytes']:,} bytes",
                    )

            actual = {"sha256": digest.hexdigest(), "size_bytes": size}
            if actual != identity:
                raise ValueError("MRI cache integrity mismatch.")

            return path

        with pglProgress("Preparing local MRI file") as progress:
            if expected is not None:
                target = cachePath(expected)

                if os.path.lexists(target):
                    verifyFile(target, expected, progress)
                    progress.finish("Verified cached MRI file")
                    return target

            cacheRoot.mkdir(parents=True, exist_ok=True)

            # TemporaryDirectory removes partial bytes on failure or interrupt.
            # Publish only after source hashing and local read-back agree.
            with tempfile.TemporaryDirectory(
                prefix=".building-",
                dir=str(cacheRoot),
            ) as temporary:
                stagedFile = Path(temporary) / filename
                digest = hashlib.sha256()
                size = 0
                transferProgress = progress.section(0, 0.8)
                transferProgress(0, "Opening MRI source")

                # A checkpoint filesystem may verify/spool its payload before
                # returning this stream. That work retains its existing policy.
                with self.open() as source, stagedFile.open("xb") as targetFile:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block: break

                        size += len(block)
                        if expected is not None and size > expected["size_bytes"]:
                            raise ValueError("MRI source exceeds its expected size.")

                        digest.update(block)
                        if targetFile.write(block) != len(block):
                            raise OSError("Incomplete MRI cache write.")

                        if expected is None:
                            transferProgress(
                                None,
                                f"Reading MRI source — {size:,} bytes",
                            )
                        else:
                            transferProgress(
                                size / max(expected["size_bytes"], 1),
                                f"Reading MRI source — {size:,} / "
                                f"{expected['size_bytes']:,} bytes",
                            )

                    targetFile.flush()
                    os.fsync(targetFile.fileno())

                actual = {"sha256": digest.hexdigest(), "size_bytes": size}
                if expected is not None and actual != expected:
                    raise ValueError("MRI source integrity mismatch.")

                transferProgress(1, "MRI source read")
                verifyFile(stagedFile, actual, progress.section(0.8, 0.98))
                target = cachePath(actual)

                if target.parent.is_symlink():
                    raise ValueError("MRI cache directory must not be a symlink.")

                target.parent.mkdir(parents=True, exist_ok=True)

                # An atomic hard link publishes the completed staging file
                # without overwriting another process's cached copy.
                # Both paths are on the same cache filesystem.
                try:
                    os.link(stagedFile, target)
                except FileExistsError:
                    verifyFile(target, actual, progress.section(0.98, 0.99))

            # Staging has been cleaned up. The published link retains the data.
            progress.finish("Verified local MRI file ready")
            return target

################################################################
# NIfTI header snapshots
################################################################
class pglMRINiftiMetadata:
    """Extract original on-disk NIfTI metadata without reading voxel arrays.

    Supports single-file NIfTI-1 and NIfTI-2, compressed or uncompressed.

    Human-readable fields accompany exact encoded header/extension bytes.
    Nonfinite numbers and byte strings use explicit JSON-safe representations.

    This helper does not modify an image, attach metadata, or publish files.
    """

    @staticmethod
    def _encodeValue(value):
        """Convert header values to JSON-safe representations without NaNs."""
        import base64
        import math
        import numpy as np

        if isinstance(value, np.ndarray):
            return pglMRINiftiMetadata._encodeValue(value.tolist())

        if isinstance(value, np.generic):
            return pglMRINiftiMetadata._encodeValue(value.item())

        if isinstance(value, bytes):
            return {
                "encoding": "base64",
                "data": base64.b64encode(value).decode("ascii"),
            }

        if isinstance(value, (list, tuple)):
            return [pglMRINiftiMetadata._encodeValue(item) for item in value]

        if isinstance(value, float) and not math.isfinite(value):
            label = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
            return {"encoding": "nonfinite", "value": label}

        if value is None or isinstance(value, (str, bool, int, float)):
            return value

        raise TypeError(f"Unsupported NIfTI header value: {type(value).__name__}")

    @classmethod
    def read(cls, filename, *, maxHeaderBytes=16 * 1024 * 1024):
        """Read a complete original header snapshot from a local NIfTI.

        Reads only the header region preceding voxel data. Does not construct
        or load a voxel array.

        Raw header bytes preserve exact stored fields, including string
        padding and original scaling. Extension payloads include their padding.

        maxHeaderBytes bounds the entire uncompressed pre-voxel region.
        Larger headers are rejected rather than partially captured.

        Invalid files and unsupported encodings raise: these are source-data
        errors, not expected user-interface refusals.
        """
        import base64
        import gzip
        import json
        import struct
        from pathlib import Path

        import nibabel as nib
        import numpy as np

        if type(maxHeaderBytes) is not int or maxHeaderBytes < 544:
            raise ValueError("maxHeaderBytes must be an integer of at least 544.")

        path = Path(filename).expanduser().absolute()
        lowerName = path.name.lower()

        if not lowerName.endswith((".nii", ".nii.gz")):
            raise ValueError("Header snapshots require a .nii or .nii.gz file.")

        opener = gzip.open if lowerName.endswith(".gz") else open

        def encodeBytes(content):
            return base64.b64encode(content).decode("ascii")

        with opener(path, "rb") as source:
            firstBytes = source.read(4)
            if len(firstBytes) != 4:
                raise ValueError("Truncated NIfTI header.")

            # sizeof_hdr identifies both the header version and byte order.
            headerSize = None
            endian = None

            for candidate in ("<", ">"):
                size = struct.unpack(candidate + "i", firstBytes)[0]
                if size in (348, 540):
                    headerSize, endian = size, candidate
                    break

            if headerSize is None:
                raise ValueError("Unrecognized NIfTI header size or byte order.")

            headerBytes = firstBytes + source.read(headerSize - 4)
            if len(headerBytes) != headerSize:
                raise ValueError("Truncated NIfTI header.")

            headerClass = nib.Nifti1Header if headerSize == 348 else nib.Nifti2Header

            # Do not let header checking normalize the original stored values.
            # Structural validation below rejects unsupported files explicitly.
            header = headerClass(
                binaryblock=headerBytes,
                endianness=endian,
                check=False,
            )

            # NIfTI-1 stores magic at bytes 344–347.
            # NIfTI-2 stores magic plus its separate EOL check at bytes 4–11.
            # Inspect original bytes, not NumPy string-field conversions.
            signature = headerBytes[344:348] if headerSize == 348 else headerBytes[4:12]
            expectedSignature = b"n+1\x00" if headerSize == 348 else b"n+2\x00\r\n\x1a\n"
            if signature != expectedSignature:
                raise ValueError("Expected a single-file NIfTI image.")
            
            offsetValue = header["vox_offset"].item()
            if (
                not np.isfinite(offsetValue)
                or offsetValue != int(offsetValue)
                or not headerSize + 4 <= offsetValue <= maxHeaderBytes
            ):
                raise ValueError(
                    "Invalid NIfTI voxel offset or header region exceeds "
                    "maxHeaderBytes."
                )

            voxelOffset = int(offsetValue)
            remaining = source.read(voxelOffset - headerSize)
            if len(remaining) != voxelOffset - headerSize:
                raise ValueError("Truncated NIfTI extension/header region.")

        extensionFlag = remaining[:4]
        extensionArea = remaining[4:]
        extensions = []

        if extensionFlag[0] != 0:
            position = 0

            while position < len(extensionArea):
                tail = extensionArea[position:]

                # Preserve otherwise unused alignment bytes separately below.
                if not any(tail): break

                if len(tail) < 8:
                    raise ValueError("Truncated NIfTI extension record.")

                size, code = struct.unpack(endian + "ii", tail[:8])
                if size < 16 or size % 16 or size > len(tail):
                    raise ValueError("Invalid NIfTI extension size.")

                extensions.append({
                    "size": size,
                    "code": code,
                    "payloadEncoding": "base64",
                    "payload": encodeBytes(tail[8:size]),
                })
                position += size

        # These matrices are interpretations of stored fields. The exact
        # original fields and binary bytes remain separately available.
        # A zero transform code means unspecified, not a valid coded transform.
        qformCode = int(header["qform_code"])
        sformCode = int(header["sform_code"])

        snapshot = {
            "version": 1,
            "niftiVersion": 1 if headerSize == 348 else 2,
            "byteOrder": "little" if endian == "<" else "big",
            "fields": {
                name: cls._encodeValue(header[name])
                for name in header.keys()
            },
            "transforms": {
                "qform": cls._encodeValue(header.get_qform()),
                "qform_code": qformCode,
                "sform": cls._encodeValue(header.get_sform()),
                "sform_code": sformCode,
                "selectedAffine": cls._encodeValue(header.get_best_affine()),
                "selectedAffinePolicy": "NiBabel get_best_affine",
            },
            "extensions": extensions,
            "raw": {
                "encoding": "base64",
                "header": encodeBytes(headerBytes),
                "extensionFlag": encodeBytes(extensionFlag),
                "extensionArea": encodeBytes(extensionArea),
            },
        }

        # Explicitly verify compatibility with our strict metadata JSON policy.
        json.dumps(snapshot, allow_nan=False)
        return snapshot
    
    @classmethod
    def fromData(cls, data):
        """Snapshot a working NiBabel image without reading voxel values.

        Unlike read(), this describes in-memory state, not original file bytes.

        Records:
            Every current header field.
            Header-derived qform, sform, and preferred affine.
            The image object's actual affine separately.
            Data-proxy scaling, when a proxy is present.
            Current extensions serialized using NiBabel's extension writer.

        The image affine and header-derived affine can differ after direct
        edits. Record both rather than silently reconciling them.

        Does not call update_header(), get_fdata(), or convert the data proxy
        to an array. Does not mutate the supplied image.
        """
        import base64
        import json
        import struct
        import sys
        from io import BytesIO

        import nibabel as nib

        if not isinstance(data, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("Expected a single-file NiBabel NIfTI image.")

        header = data.header
        endian = header.endianness
        nativeEndian = "<" if sys.byteorder == "little" else ">"
        extensions = []

        # Extensions may contain edited structured contents. Serialize their
        # current representation, not stale original on-disk payload bytes.
        for extension in header.extensions:
            buffer = BytesIO()
            extension.write_to(buffer, byteswap=endian != nativeEndian)
            content = buffer.getvalue()

            if len(content) < 8:
                raise ValueError("Invalid serialized NIfTI extension.")

            size, code = struct.unpack(endian + "ii", content[:8])
            if size != len(content) or size < 16 or size % 16:
                raise ValueError("Invalid serialized NIfTI extension size.")

            extensions.append({
                "size": size,
                "code": code,
                "payloadEncoding": "base64",
                "payload": base64.b64encode(content[8:]).decode("ascii"),
            })

        # Inspect proxy metadata only. Never access its numerical contents.
        dataObject = data.dataobj
        isProxy = bool(nib.is_proxy(dataObject))
        proxyScaling = None

        if isProxy:
            proxyScaling = {
                "slope": cls._encodeValue(dataObject.slope),
                "intercept": cls._encodeValue(dataObject.inter),
            }

        snapshot = {
            "version": 1,
            "snapshotKind": "workingImage",
            "niftiVersion": 2 if isinstance(header, nib.Nifti2Header) else 1,
            "byteOrder": "little" if endian == "<" else "big",
            "fields": {
                name: cls._encodeValue(header[name])
                for name in header.keys()
            },
            "transforms": {
                "qform": cls._encodeValue(header.get_qform()),
                "qform_code": int(header["qform_code"]),
                "sform": cls._encodeValue(header.get_sform()),
                "sform_code": int(header["sform_code"]),
                "selectedAffine": cls._encodeValue(header.get_best_affine()),
                "selectedAffinePolicy": "NiBabel header.get_best_affine",
                "imageAffine": cls._encodeValue(data.affine),
            },
            "dataRepresentation": {
                "shape": cls._encodeValue(data.shape),
                "isProxy": isProxy,
                "proxyScaling": proxyScaling,
            },
            "extensions": extensions,
            "workingHeader": {
                "encoding": "base64",
                "header": base64.b64encode(header.binaryblock).decode("ascii"),
                "description": (
                    "Current in-memory header bytes, not an original file "
                    "header or a prediction of bytes produced by saving."
                ),
            },
        }

        # Do not permit accidental NaN literals or unsupported objects.
        json.dumps(snapshot, allow_nan=False)
        return snapshot
    
    @classmethod
    def readBacking(cls, backing, *, maxHeaderBytes=16 * 1024 * 1024):
        """Extract original header metadata from a deferred filesystem source.

        Reads the header region without materializing the complete NIfTI or
        constructing a NiBabel image/data proxy.

        A small temporary uncompressed header file lets us reuse the tested
        read() parser. It is deleted before returning.

        For compressed sources, gzip may read ahead beyond the header.
        A checkpoint filesystem may independently verify/spool the entire file
        when opened; this helper does not bypass that filesystem policy.

        This is not full-payload verification. Acquisition export must still
        hash and verify the complete source independently. Callers must keep
        sources unchanged across header extraction and payload export.

        Does not mutate scan metadata or close the caller-owned filesystem.
        Malformed headers and source read failures propagate.
        """
        import gzip
        import math
        import struct
        from contextlib import ExitStack
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import nibabel as nib

        if not isinstance(backing, pglMRIFileBacking):
            raise TypeError("Header extraction requires a pglMRIFileBacking.")

        if type(maxHeaderBytes) is not int or maxHeaderBytes < 544:
            raise ValueError("maxHeaderBytes must be an integer of at least 544.")

        lowerPath = backing.path.lower()
        if not lowerPath.endswith((".nii", ".nii.gz")):
            raise ValueError("Header extraction requires a .nii or .nii.gz source.")

        def readExactly(source, count):
            """Handle streams that return fewer bytes than requested."""
            chunks = []
            remaining = count

            while remaining:
                block = source.read(remaining)
                if not block:
                    raise ValueError("Truncated NIfTI header region.")

                chunks.append(block)
                remaining -= len(block)

            return b"".join(chunks)

        with ExitStack() as resources:
            source = resources.enter_context(backing.open())

            if lowerPath.endswith(".nii.gz"):
                source = resources.enter_context(
                    gzip.GzipFile(fileobj=source, mode="rb")
                )

            firstBytes = readExactly(source, 4)
            headerSize = None
            endian = None

            for candidate in ("<", ">"):
                size = struct.unpack(candidate + "i", firstBytes)[0]
                if size in (348, 540):
                    headerSize, endian = size, candidate
                    break

            if headerSize is None:
                raise ValueError("Unrecognized NIfTI header size or byte order.")

            headerBytes = firstBytes + readExactly(source, headerSize - 4)
            headerClass = nib.Nifti1Header if headerSize == 348 else nib.Nifti2Header
            header = headerClass(
                binaryblock=headerBytes,
                endianness=endian,
                check=False,
            )

            # Reject paired-file NIfTI before interpreting its voxel offset.
            # NIfTI-2's signature spans both magic and the EOL-check field.
            signature = headerBytes[344:348] if headerSize == 348 else headerBytes[4:12]
            expectedSignature = b"n+1\x00" if headerSize == 348 else b"n+2\x00\r\n\x1a\n"
            if signature != expectedSignature:
                raise ValueError("Expected a single-file NIfTI image.")

            offset = header["vox_offset"].item()
            if (
                not math.isfinite(offset)
                or offset != int(offset)
                or not headerSize + 4 <= offset <= maxHeaderBytes
            ):
                raise ValueError(
                    "Invalid NIfTI voxel offset or header region exceeds "
                    "maxHeaderBytes."
                )

            # Never request the voxel region. The bounded temporary file below
            # contains exactly the uncompressed bytes preceding voxel data.
            remainingBytes = readExactly(source, int(offset) - headerSize)

        # The existing parser reads only the pre-voxel region, so this header-
        # only file is sufficient. Do not attempt nib.load() on this file.
        with TemporaryDirectory(prefix="pgl-mri-header-") as temporary:
            headerPath = Path(temporary) / "header.nii"

            with headerPath.open("xb") as target:
                for block in (headerBytes, remainingBytes):
                    if target.write(block) != len(block):
                        raise OSError("Incomplete temporary NIfTI header write.")

            return cls.read(headerPath, maxHeaderBytes=maxHeaderBytes)

################################################################
# MRI scan
################################################################
class pglMRIScan(pglTraitSettings):
    """A NIfTI scan with local or deferred filesystem backing.

    sourceMetadata:
        Original source provenance, including the original on-disk header
        snapshot under sourceMetadata["nifti"].

    metadata:
        Expandable annotations and current NIfTI metadata.
        Use updateMetadata() for validated edits and trait notifications.

    Accessing data resolves the source and returns a cached NiBabel image.
    Header information is read, but the complete voxel array remains lazy.

    Filesystem backing is implemented. Deferred Flywheel references,
    current-header synchronization, and bundle/database persistence
    are not implemented yet.
    """
    name = Unicode("", help="Display name of scan")
    
    # Local paths are runtime access information, not portable checkpoint
    # locations. Persistence will handle payload references explicitly.
    localPath = Unicode(
        "",
        help="Local NIfTI filename",
        visible=False,
    ).tag(serialize=False)

    # Preserve source provenance separately from editable annotations.
    # This dictionary is not technically immutable; callers should treat it
    # as a snapshot and put their own corrections/notes in metadata instead.
    sourceMetadata = Dict(
        default_value={},
        help="Original source identifiers and acquisition metadata",
    )

    metadata = Dict(
        default_value={},
        help="Editable scan annotations and application-specific metadata",
    )

    # NiBabel readers must never enter JSON metadata.
    _data = Any(default_value=None, allow_none=True, serialize=False)
    
    # Runtime source access and cache configuration are not portable metadata.
    _backing = Any(default_value=None, allow_none=True, serialize=False)
    _cacheDirectory = Unicode("").tag(serialize=False)


    @staticmethod
    def _copyMetadata(value):
        """Validate JSON-compatible contents and return an independent copy.

        No implicit conversion of arrays, tuples, timestamps, or arbitrary
        objects. Timestamps should be supplied as ISO strings.

        Reject cycles, non-string dictionary keys, and nonfinite numbers.
        This is an internal validator; public update methods report expected
        validation failures through pglMessages.warning().
        """
        import math
        from copy import deepcopy

        active = set()

        def validate(item):
            if item is None or type(item) in (str, bool, int): return

            if type(item) is float:
                if not math.isfinite(item):
                    raise ValueError("Metadata numbers must be finite.")
                return

            if type(item) not in (dict, list):
                raise TypeError(
                    f"Unsupported metadata type: {type(item).__name__}. "
                    "Use JSON-compatible values; encode timestamps as ISO strings."
                )

            identity = id(item)
            if identity in active:
                raise ValueError("Metadata must not contain circular references.")

            active.add(identity)
            try:
                if type(item) is dict:
                    for key, child in item.items():
                        if not isinstance(key, str):
                            raise TypeError("Metadata dictionary keys must be strings.")
                        validate(child)
                else:
                    for child in item: validate(child)
            finally:
                active.remove(identity)

        if type(value) is not dict:
            raise TypeError("Scan metadata must be a dictionary.")

        validate(value)
        return deepcopy(value)

    def updateMetadata(self, sectionName, **values):
        """Merge fields into one metadata section without changing provenance.

        Creates the section if absent. Existing sections must be dictionaries.
        Unspecified fields and other sections are preserved.

        A supplied field replaces its previous value in full; nested
        dictionaries are not recursively merged. None is a stored value,
        not an instruction to delete a field.

        Returns an independent copy of the updated section, or None after
        warning about invalid input. Validation failure leaves metadata intact.

        Examples:
            scan.updateMetadata("annotations", notes="Motion", exclude=True)
            scan.updateMetadata("qc", reviewed=True, reviewer="justin")
        """
        from .pglMessages import pglMessages

        try:
            if not isinstance(sectionName, str) or not sectionName.strip():
                raise ValueError("Metadata section name must be a nonblank string.")

            if any(character in sectionName for character in ("\x00", "\n", "\r")):
                raise ValueError("Metadata section name must not contain NUL or newlines.")

            # Validate both existing contents and proposed values before
            # assigning anything. Copy nested values to avoid caller aliases.
            updated = self._copyMetadata(self.metadata)
            changes = self._copyMetadata(values)
            section = updated.get(sectionName, {})

            if type(section) is not dict:
                raise ValueError(
                    f"Metadata section {sectionName!r} already exists "
                    "but is not a dictionary."
                )

            updated[sectionName] = {**section, **changes}
            result = self._copyMetadata(updated[sectionName])

        except (TypeError, ValueError) as error:
            pglMessages.warning(
                f"Scan metadata was not updated: {error}",
                level=0,
            )
            return None

        # Assignment, rather than in-place mutation, notifies trait observers.
        # Observer/programming errors are not hidden as input-validation errors.
        self.metadata = updated
        return result

    @property
    def data(self):
        """Resolve a source and return a cached, lazy NiBabel image.

        Deferred filesystem sources are materialized into a verified cache.
        Existing localPath-only scans retain direct local access.

        Captures the original on-disk header in sourceMetadata["nifti"].
        An existing original snapshot must agree with the source being opened.
        Initializes metadata["nifti"] only if that section is absent.

        Returns a NiBabel image object, not a materialized voxel array.
        Source/cache files must remain unchanged and available while used.

        Missing configuration/dependencies warn and return None.
        Storage, format, and integrity failures propagate.

        Current-header synchronization after data edits is not implemented
        yet. Editing metadata does not modify the NIfTI header.
        """
        from pathlib import Path

        from .pglMessages import pglMessages

        if self._data is not None: return self._data

        try:
            import nibabel as nib
        except ImportError:
            pglMessages.warning(
                "MRI data access requires nibabel. Install using: pip install nibabel",
                level=0,
            )
            return None

        if self._backing is None and not self.localPath:
            pglMessages.warning("This MRI scan has no attached source.", level=0)
            return None

        if self._backing is not None and not self._cacheDirectory:
            pglMessages.warning(
                "A local cacheDirectory is required for this MRI source.",
                level=0,
            )
            return None

        # Validate/copy metadata before potentially expensive source reads.
        sourceMetadata = self._copyMetadata(self.sourceMetadata)
        currentMetadata = self._copyMetadata(self.metadata)

        if self._backing is not None:
            if not isinstance(self._backing, pglMRIFileBacking):
                raise TypeError("Unsupported MRI backing type.")

            path = self._backing.materialize(self._cacheDirectory)
        else:
            path = Path(self.localPath).expanduser().absolute()

        # Read stored fields explicitly: NiBabel may consume scaling fields
        # when opening the image, so data.header is not the original header.
        originalSnapshot = pglMRINiftiMetadata.read(path)

        if "nifti" in sourceMetadata:
            if sourceMetadata["nifti"] != originalSnapshot:
                raise ValueError(
                    "MRI source header differs from its preserved original "
                    "snapshot. Refusing to replace source provenance."
                )
        else:
            sourceMetadata["nifti"] = self._copyMetadata(originalSnapshot)

        if "nifti" not in currentMetadata:
            currentMetadata["nifti"] = self._copyMetadata(originalSnapshot)
        elif type(currentMetadata["nifti"]) is not dict:
            raise ValueError("metadata['nifti'] must be a dictionary.")

        # Opening creates a lazy data proxy; do not call get_fdata() here.
        data = nib.load(str(path), mmap=True)

        if not isinstance(data, (nib.Nifti1Image, nib.Nifti2Image)):
            raise TypeError("MRI scans currently support single-file NIfTI images.")

        self.sourceMetadata = sourceMetadata
        self.metadata = currentMetadata

        # Retain the backing as authoritative. localPath records its resolved
        # cache location rather than replacing the deferred source reference.
        self.localPath = str(path)
        self._data = data
        return data
    
    def attachBacking(self, backing, *, cacheDirectory):
        """Attach a deferred filesystem source without reading its contents.

        Returns this scan on success, or None with a warning for invalid input.

        Only unattached scans are accepted. Source replacement must explicitly
        handle cached data and provenance rather than carrying them across.

        The caller owns the backing filesystem and must keep it available.
        cacheDirectory is local to the machine running the data reader.
        """
        import os
        from pathlib import Path

        from .pglMessages import pglMessages

        try:
            if not isinstance(backing, pglMRIFileBacking):
                raise TypeError("backing must be a pglMRIFileBacking.")

            if self._backing is not None or self.localPath or self._data is not None:
                raise ValueError("This scan already has an attached source or data.")

            cacheText = os.fspath(cacheDirectory)
            if (
                not isinstance(cacheText, str)
                or not cacheText.strip()
                or "://" in cacheText
                or "::" in cacheText
                or any(character in cacheText for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("cacheDirectory must be an ordinary local path.")

            # Does not inspect the source or create the cache.
            cachePath = str(Path(cacheText).expanduser().absolute())

        except (TypeError, ValueError) as error:
            pglMessages.warning(f"MRI source was not attached: {error}", level=0)
            return None

        self._cacheDirectory = cachePath
        self._backing = backing
        return self
    
    def syncNiftiMetadata(self):
        """Replace current NIfTI metadata from already-loaded scan data.

        Does not resolve a deferred source, read voxel arrays, or write files.
        Original sourceMetadata and other metadata sections are preserved.

        Returns an independent copy of the new snapshot.
        If data has not been opened, warns and returns None.

        Invalid working image/header contents raise rather than silently
        preserving stale metadata and reporting success.
        """
        from .pglMessages import pglMessages

        # Inspect the cache directly. Accessing self.data here would download
        # a deferred source merely to synchronize metadata.
        if self._data is None:
            pglMessages.warning(
                "NIfTI metadata was not synchronized: scan data has not been "
                "opened. Access scan.data first.",
                level=0,
            )
            return None

        updated = self._copyMetadata(self.metadata)
        snapshot = self._copyMetadata(pglMRINiftiMetadata.fromData(self._data))
        result = self._copyMetadata(snapshot)

        # Replace the complete section, not a shallow merge. Otherwise raw
        # fields from the original snapshot could survive as stale metadata.
        updated["nifti"] = snapshot
        self.metadata = updated
        return result
    
    def exportSource(self, filename, *, filesystem=None):
        """Stream original source bytes to a new destination file.

        Acquisition export only:
            Uses the attached backing, or the original localPath.
            Does not serialize edited self._data or current header metadata.
            Does not materialize a deferred source into a local cache.

        filesystem:
            Explicit destination fsspec filesystem. Without one, filename
            is resolved by fsspec, supporting local paths and supported URLs.

        Returns:
            {"sha256": digest, "size_bytes": size}

        Expected configuration refusals warn and return None.
        Read/write failures and integrity mismatches propagate.

        Destination bytes are read back and verified before returning.
        Existing files are never deliberately overwritten.
        Failed exports may leave a partial destination; the enclosing bundle
        publisher is responsible for staging and cleanup.

        Does not modify scan metadata, publish a manifest, or register SQL rows.
        Source/destination filesystems remain caller-owned.
        """
        import hashlib
        import os
        import posixpath

        from fsspec import AbstractFileSystem
        from fsspec.core import url_to_fs
        from fsspec.implementations.local import LocalFileSystem

        from .pglMessages import pglMessages
        from .pglProgress import pglProgress

        if self._backing is None and not self.localPath:
            pglMessages.warning("MRI source was not exported: no source is attached.", level=0)
            return None

        if self._backing is not None:
            if not isinstance(self._backing, pglMRIFileBacking):
                raise TypeError("Unsupported MRI backing type.")

            sourceBacking = self._backing
        else:
            # Local-only scans still have a source independent of cached data.
            sourceBacking = pglMRIFileBacking(
                LocalFileSystem(),
                str(Path(self.localPath).expanduser().absolute()),
            )

        try:
            destinationText = os.fspath(filename)
            if not isinstance(destinationText, str) or not destinationText.strip():
                raise ValueError("filename must be a nonempty path.")

            if any(character in destinationText for character in ("\x00", "\n", "\r")):
                raise ValueError("filename must not contain NUL or newlines.")

            if filesystem is not None and not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec filesystem.")

            # Byte-preserving export must also preserve compression semantics.
            sourceSuffix = ".nii.gz" if sourceBacking.path.lower().endswith(".nii.gz") else ".nii"
            if not sourceBacking.path.lower().endswith((".nii", ".nii.gz")):
                raise ValueError("The MRI source must be .nii or .nii.gz.")

            if not destinationText.lower().endswith(sourceSuffix):
                raise ValueError(f"Destination must retain the source extension {sourceSuffix}.")

            if sourceSuffix == ".nii" and destinationText.lower().endswith(".nii.gz"):
                raise ValueError("An uncompressed source cannot be exported as .nii.gz without conversion.")

        except (TypeError, ValueError) as error:
            pglMessages.warning(f"MRI source was not exported: {error}", level=0)
            return None

        # Filesystem resolution may involve authentication. Do not conceal
        # connection/security failures as ordinary argument refusals.
        if filesystem is None:
            filesystem, destination = url_to_fs(destinationText)
        else:
            destination = filesystem._strip_protocol(destinationText)

        if filesystem.exists(destination):
            pglMessages.warning(
                "MRI source was not exported: the destination already exists. "
                "Nothing was overwritten.",
                level=0,
            )
            return None

        expected = sourceBacking.expectedIdentity
        identity = None

        with pglProgress("Exporting MRI acquisition source") as progress:
            parent = posixpath.dirname(destination)
            if parent: filesystem.makedirs(parent, exist_ok=True)

            copyProgress = progress.section(0, 0.65)
            copyProgress(0, "Opening acquisition source")

            digest = hashlib.sha256()
            size = 0

            # Open the source before creating destination output.
            # A database checkpoint filesystem retains its own verification
            # policy; this exporter does not bypass it.
            with sourceBacking.open() as source:
                with filesystem.open(destination, "xb") as target:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block: break

                        size += len(block)
                        if expected is not None and size > expected["size_bytes"]:
                            raise ValueError("MRI acquisition source exceeds its expected size.")

                        digest.update(block)
                        if target.write(block) != len(block):
                            raise OSError("Incomplete MRI acquisition export.")

                        fraction = None if expected is None else size / max(expected["size_bytes"], 1)
                        copyProgress(fraction, f"Streaming original NIfTI — {size:,} bytes")

            identity = {"sha256": digest.hexdigest(), "size_bytes": size}
            if expected is not None and identity != expected:
                raise ValueError("MRI acquisition source integrity mismatch.")

            copyProgress(1, "Original NIfTI transferred")
            verifyProgress = progress.section(0.65, 1)
            verifyProgress(0, "Verifying destination")

            storedDigest = hashlib.sha256()
            storedSize = 0

            with filesystem.open(destination, "rb") as stored:
                while True:
                    block = stored.read(1024 * 1024)
                    if not block: break

                    storedSize += len(block)
                    if storedSize > size:
                        raise ValueError("Exported MRI exceeds its expected size.")

                    storedDigest.update(block)
                    verifyProgress(
                        storedSize / max(size, 1),
                        f"Verifying destination — {storedSize:,} / {size:,} bytes",
                    )

            if {"sha256": storedDigest.hexdigest(), "size_bytes": storedSize} != identity:
                raise ValueError("Exported MRI integrity mismatch.")

            progress.finish("Original MRI source exported and verified")

        return identity


################################################################
# MRI collection
################################################################
class pglMRI(pglActionable):
    """Ordered MRI scans, independent of behavioral runs.

    This initial implementation attaches local NIfTI references.
    It does not copy files, publish checkpoints, or infer behavioral matching.
    """

    scans = List(
        Instance(pglMRIScan),
        help="Ordered MRI scans",
    )

    def add(self, filename, *, name=None, sourceMetadata=None):
        """Attach a local NIfTI without opening its header or voxel array.

        Returns the new pglMRIScan.

        Source metadata is copied so subsequent caller edits do not modify
        the attached record. Metadata may contain identifying subject details.

        File existence and extension are checked here. NIfTI format validation
        occurs on first image access. Recorded source hashes are not verified
        by this attachment operation.
        """
        path = Path(filename).expanduser().absolute()

        if not path.is_file():
            raise FileNotFoundError(f"MRI source file is unavailable: {path}")

        if not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError("MRI scans currently require .nii or .nii.gz files.")

        if name is None:
            name = path.name

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scan name must be a nonblank string.")

        if sourceMetadata is None:
            sourceMetadata = {}

        if not isinstance(sourceMetadata, dict):
            raise TypeError("sourceMetadata must be a dictionary or None.")

        scan = pglMRIScan(
            name=name,
            localPath=str(path),
            sourceMetadata=deepcopy(sourceMetadata),
        )

        # Assign a new list so trait observers see the attachment.
        self.scans = [*self.scans, scan]
        return scan
    
    def saveAcquisitionBundle(self, dataPath, *, filesystem=None):
        """Save original MRI acquisition files and portable scan metadata."""
        return pglMRIBundle.saveAcquisition(self, dataPath, filesystem=filesystem)
    
    @classmethod
    def loadAcquisitionBundle(cls, dataPath, *, filesystem=None, cacheDirectory):
        """Restore acquisition metadata and deferred MRI file references."""
        return pglMRIBundle.loadAcquisition(dataPath, filesystem=filesystem, cacheDirectory=cacheDirectory)
    
################################################################
# MRI acquisition bundles
################################################################
class pglMRIBundle:
    """Portable acquisition payloads and ordered scan metadata.

    This contract stores scan UUIDs, names, provenance, and annotations.
    It does not persist collection action history or database/session state.

    Payloads are original source bytes, not edited working images.
    Database publication and lazy bundle restoration are separate steps.
    """

    contract = "pgl.mri.acquisition.v1"

    @classmethod
    def saveAcquisition(cls, mri, dataPath, *, filesystem=None):
        """Stream original scan files into a fresh acquisition bundle.

        Writes mri.json last, after payload and metadata verification.
        No complete local staging copy or voxel-array loading is required.

        Original header snapshots are extracted from verified exported bytes.
        Existing sourceMetadata["nifti"] must agree with those bytes.

        Annotations are retained. The saved metadata["nifti"] is reset to an
        independent original-header snapshot because this is an acquisition
        export, not a save of current image/header edits.

        Working scans, metadata, readers, and sources are not modified.
        Concurrent mutation of scans, metadata, or sources is unsupported.

        Returns the destination path, or None for expected input refusals.
        Storage, serialization, and integrity failures propagate.
        Failed saves may leave partial output; no automatic deletion occurs.
        """
        import hashlib
        import json
        import os
        import posixpath

        from fsspec import AbstractFileSystem
        from fsspec.core import url_to_fs
        from fsspec.implementations.local import LocalFileSystem

        from .pglMessages import pglMessages

        if type(mri) is not pglMRI:
            pglMessages.warning("MRI export requires a pglMRI instance.", level=0)
            return None

        scans = list(mri.scans)
        if not scans:
            pglMessages.warning("MRI bundle was not saved: no scans are attached.", level=0)
            return None

        try:
            pathText = os.fspath(dataPath)
            if not isinstance(pathText, str) or not pathText.strip():
                raise ValueError("dataPath must be a nonempty path.")

            if any(character in pathText for character in ("\x00", "\n", "\r")):
                raise ValueError("dataPath must not contain NUL or newlines.")

            if filesystem is not None and not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec filesystem.")

        except (TypeError, ValueError) as error:
            pglMessages.warning(f"MRI bundle was not saved: {error}", level=0)
            return None

        # Resolve outside argument handling: authentication and storage errors
        # must not be disguised as ordinary input refusals.
        if filesystem is None:
            filesystem, destination = url_to_fs(pathText)
        else:
            destination = filesystem._strip_protocol(pathText)

        if filesystem.exists(destination):
            pglMessages.warning(
                "MRI bundle was not saved: the destination already exists. "
                "Nothing was overwritten.",
                level=0,
            )
            return None

        prepared = []

        # Validate all records before creating output. No header or voxel
        # access occurs here. Copies prevent later edits from changing records.
        for index, scan in enumerate(scans):
            if scan._backing is not None:
                if not isinstance(scan._backing, pglMRIFileBacking):
                    raise TypeError("Unsupported MRI source backing.")

                sourcePath = scan._backing.path
            else:
                sourcePath = scan.localPath

            if not sourcePath:
                pglMessages.warning(
                    f"MRI bundle was not saved: scan {index} has no source.",
                    level=0,
                )
                return None

            if not sourcePath.lower().endswith((".nii", ".nii.gz")):
                pglMessages.warning(
                    f"MRI bundle was not saved: scan {index} is not a .nii or .nii.gz source.",
                    level=0,
                )
                return None

            # Avoid creating a bundle beneath a source file's path.
            # Ordinary existing destinations were already rejected above.
            suffix = ".nii.gz" if sourcePath.lower().endswith(".nii.gz") else ".nii"
            prepared.append({
                "scan": scan,
                "uuid": scan.uuid,
                "name": scan.name,
                "sourceMetadata": scan._copyMetadata(scan.sourceMetadata),
                "metadata": scan._copyMetadata(scan.metadata),
                "path": f"scans/{index:06d}/source{suffix}",
            })

        # Validate metadata encoding before starting payload writes.
        def encode(document):
            return json.dumps(
                document,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")

        for item in prepared:
            encode({key: value for key, value in item.items() if key != "scan"})

        # For local storage, reject a dangling destination symlink too.
        if isinstance(filesystem, LocalFileSystem):
            if os.path.lexists(filesystem._strip_protocol(destination)):
                pglMessages.warning("MRI bundle destination already exists.", level=0)
                return None

        filesystem.makedirs(destination, exist_ok=False)
        records = []

        for index, item in enumerate(prepared):
            scan = item["scan"]
            relativePath = item["path"]
            target = posixpath.join(destination, relativePath)

            pglMessages.message(f"Exporting MRI scan {index + 1}/{len(scans)}: {item['name']}")
            identity = scan.exportSource(target, filesystem=filesystem)

            if identity is None:
                raise RuntimeError("MRI payload export was refused; the bundle is incomplete.")

            # Re-read the destination header through the tested source parser.
            # This snapshot describes the exported bytes, not a separate read
            # from a potentially changing original source.
            exportedBacking = pglMRIFileBacking(
                filesystem,
                target,
                expectedIdentity=identity,
            )
            originalHeader = pglMRINiftiMetadata.readBacking(exportedBacking)

            sourceMetadata = scan._copyMetadata(item["sourceMetadata"])
            metadata = scan._copyMetadata(item["metadata"])

            if "nifti" in sourceMetadata and sourceMetadata["nifti"] != originalHeader:
                raise ValueError(
                    f"Scan {index}: exported header differs from preserved "
                    "original provenance."
                )

            sourceMetadata["nifti"] = scan._copyMetadata(originalHeader)

            # This archive contains original acquisition bytes. Do not pair
            # them with current-header edits that were not actually exported.
            metadata["nifti"] = scan._copyMetadata(originalHeader)

            records.append({
                "index": index,
                "uuid": item["uuid"],
                "name": item["name"],
                "path": relativePath,
                **identity,
                "sourceMetadata": sourceMetadata,
                "metadata": metadata,
            })

        # Detect collection replacement/reordering and metadata edits before
        # publishing the descriptor. In-place image edits are irrelevant here:
        # this operation intentionally exports the original source bytes.
        if len(mri.scans) != len(scans) or any(
            current is not original for current, original in zip(mri.scans, scans)
        ):
            raise ValueError("MRI scan collection changed during export.")

        for item in prepared:
            scan = item["scan"]
            if (
                scan.uuid != item["uuid"]
                or scan.name != item["name"]
                or scan.sourceMetadata != item["sourceMetadata"]
                or scan.metadata != item["metadata"]
            ):
                raise ValueError("MRI scan metadata changed during export.")

        descriptor = {
            "contract": cls.contract,
            "scans": records,
        }
        cls.validateDescriptor(descriptor)
        content = encode(descriptor)
        descriptorPath = posixpath.join(destination, "mri.json")

        # The descriptor is the completion record. Never write it before all
        # scan exports and metadata checks succeed.
        with filesystem.open(descriptorPath, "xb") as target:
            if target.write(content) != len(content):
                raise OSError("Incomplete MRI descriptor write.")

        with filesystem.open(descriptorPath, "rb") as source:
            stored = source.read(len(content) + 1)

        if len(stored) != len(content) or hashlib.sha256(stored).digest() != hashlib.sha256(content).digest():
            raise ValueError("MRI descriptor verification failed.")

        pglMessages.message(f"Saved acquisition bundle containing {len(records)} MRI scans.")
        return destination
    
    @classmethod
    def validateDescriptor(cls, descriptor):
        """Validate acquisition metadata without reading any payload files.

        Requires ordered scans, deterministic safe paths, valid identities,
        and independent JSON-compatible metadata.

        Repeated scan UUIDs are permitted: the same scan may appear more than
        once in an ordered collection. Payload paths remain unique.

        Structural validation does not establish that header metadata agrees
        with actual NIfTI bytes. First data access checks that correspondence.
        """
        import re
        from uuid import UUID

        if not isinstance(descriptor, dict) or set(descriptor) != {"contract", "scans"}:
            raise ValueError("Invalid MRI bundle descriptor fields.")

        if descriptor["contract"] != cls.contract:
            raise ValueError("Unsupported MRI acquisition bundle contract.")

        records = descriptor["scans"]
        if not isinstance(records, list) or not records:
            raise ValueError("MRI acquisition bundle must contain scans.")

        expectedFields = {
            "index", "uuid", "name", "path", "sha256", "size_bytes",
            "sourceMetadata", "metadata",
        }

        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != expectedFields:
                raise ValueError(f"MRI scan {index}: invalid record fields.")

            if type(record["index"]) is not int or record["index"] != index:
                raise ValueError("MRI scan indices must be consecutive and ordered.")

            identifier = record["uuid"]
            if not isinstance(identifier, str):
                raise ValueError(f"MRI scan {index}: uuid must be a string.")

            try:
                canonicalUUID = str(UUID(identifier))
            except ValueError:
                raise ValueError(f"MRI scan {index}: invalid UUID.") from None

            if identifier != canonicalUUID:
                raise ValueError(f"MRI scan {index}: UUID must use canonical formatting.")

            if not isinstance(record["name"], str) or not record["name"].strip():
                raise ValueError(f"MRI scan {index}: name must be nonblank.")

            # Exact paths exclude traversal, absolute paths, aliases, and
            # undeclared filenames while retaining compression semantics.
            allowedPaths = {
                f"scans/{index:06d}/source.nii",
                f"scans/{index:06d}/source.nii.gz",
            }
            if not isinstance(record["path"], str) or record["path"] not in allowedPaths:
                raise ValueError(f"MRI scan {index}: invalid payload path.")

            digest = record["sha256"]
            size = record["size_bytes"]

            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"MRI scan {index}: invalid SHA-256.")

            if type(size) is not int or not 0 <= size <= 9223372036854775807:
                raise ValueError(f"MRI scan {index}: invalid byte count.")

            sourceMetadata = pglMRIScan._copyMetadata(record["sourceMetadata"])
            metadata = pglMRIScan._copyMetadata(record["metadata"])
            original = sourceMetadata.get("nifti")

            expectedHeaderFields = {
                "version", "niftiVersion", "byteOrder", "fields",
                "transforms", "extensions", "raw",
            }
            if not isinstance(original, dict) or set(original) != expectedHeaderFields:
                raise ValueError(f"MRI scan {index}: missing or invalid original NIfTI snapshot.")

            if type(original["version"]) is not int or original["version"] != 1:
                raise ValueError("Unsupported NIfTI snapshot version.")

            if type(original["niftiVersion"]) is not int or original["niftiVersion"] not in (1, 2):
                raise ValueError("Unsupported NIfTI version.")

            if original["byteOrder"] not in ("little", "big"):
                raise ValueError("Invalid NIfTI byte order.")

            for field in ("fields", "transforms", "raw"):
                if not isinstance(original[field], dict):
                    raise ValueError(f"NIfTI snapshot {field} must be a dictionary.")

            if not isinstance(original["extensions"], list):
                raise ValueError("NIfTI extensions must be a list.")

            # Validate all decoded header fields/extensions by replaying the
            # existing header parser against its exact preserved raw bytes.
            import base64
            import binascii
            from pathlib import Path
            from tempfile import TemporaryDirectory

            raw = original["raw"]
            if set(raw) != {"encoding", "header", "extensionFlag", "extensionArea"} or raw["encoding"] != "base64":
                raise ValueError("Invalid original NIfTI raw-header encoding.")

            try:
                parts = []
                for field in ("header", "extensionFlag", "extensionArea"):
                    if not isinstance(raw[field], str):
                        raise ValueError("Raw header components must be base64 strings.")
                    parts.append(base64.b64decode(raw[field], validate=True))
            except (ValueError, binascii.Error):
                raise ValueError("Invalid original NIfTI base64 content.") from None

            headerSize = 348 if original["niftiVersion"] == 1 else 540
            if len(parts[0]) != headerSize or len(parts[1]) != 4:
                raise ValueError("Invalid original NIfTI header/extension-flag size.")

            headerBytes = b"".join(parts)
            if len(headerBytes) > 16 * 1024 * 1024:
                raise ValueError("NIfTI header snapshot exceeds the supported size limit.")

            # Only descriptor metadata is written here, never voxel payloads.
            with TemporaryDirectory(prefix="pgl-mri-validate-header-") as temporary:
                path = Path(temporary) / "header.nii"
                path.write_bytes(headerBytes)
                reconstructed = pglMRINiftiMetadata.read(path)

            if reconstructed != original:
                raise ValueError("NIfTI snapshot fields disagree with its preserved raw header.")

            # Acquisition files contain original bytes, so their initial current
            # header snapshot must not claim unapplied working-image edits.
            if metadata.get("nifti") != original:
                raise ValueError("Acquisition current-header metadata differs from its original header.")

        return descriptor

    @classmethod
    def loadAcquisition(cls, dataPath, *, filesystem=None, cacheDirectory, maxDescriptorBytes=64 * 1024 * 1024):
        """Restore ordered scan metadata and deferred filesystem backings.

        Reads mri.json only. Header-snapshot validation uses metadata already
        contained in that descriptor; no NIfTI payload is opened or stat'ed.

        Requires a caller-owned local cache for eventual scan.data access.
        Does not create that cache during restoration.

        File hashes are enforced during materialization. Existing valid cached
        payloads may be reused without reopening their original source.

        This standalone descriptor is trusted input: hashes establish payload
        correspondence, not authenticity. Database integration will verify the
        descriptor against the checkpoint's registered identity.

        Returns pglMRI, or None for expected argument refusals.
        Storage, decoding, and integrity failures propagate.
        """
        import json
        import os
        import posixpath
        from pathlib import Path

        from fsspec import AbstractFileSystem
        from fsspec.core import url_to_fs

        from .pglMessages import pglMessages

        try:
            pathText = os.fspath(dataPath)
            cacheText = os.fspath(cacheDirectory)

            if not isinstance(pathText, str) or not pathText.strip():
                raise ValueError("dataPath must be a nonempty path.")

            if any(character in pathText for character in ("\x00", "\n", "\r")):
                raise ValueError("dataPath must not contain NUL or newlines.")

            if (
                not isinstance(cacheText, str)
                or not cacheText.strip()
                or "://" in cacheText
                or "::" in cacheText
                or any(character in cacheText for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("cacheDirectory must be an ordinary local path.")

            if filesystem is not None and not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec filesystem.")

            if type(maxDescriptorBytes) is not int or maxDescriptorBytes <= 0:
                raise ValueError("maxDescriptorBytes must be a positive integer.")

        except (TypeError, ValueError) as error:
            pglMessages.warning(f"MRI bundle was not loaded: {error}", level=0)
            return None

        if filesystem is None:
            filesystem, root = url_to_fs(pathText)
        else:
            root = filesystem._strip_protocol(pathText)

        def uniqueKeys(pairs):
            """Reject duplicate JSON keys rather than silently losing fields."""
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError(f"Duplicate MRI descriptor key: {key!r}")
                result[key] = value
            return result

        with filesystem.open(posixpath.join(root, "mri.json"), "rb") as source:
            content = source.read(maxDescriptorBytes + 1)

        if len(content) > maxDescriptorBytes:
            raise ValueError("MRI descriptor exceeds the configured size limit.")

        descriptor = json.loads(content, object_pairs_hook=uniqueKeys)
        cls.validateDescriptor(descriptor)

        cachePath = str(Path(cacheText).expanduser().absolute())
        scans = []

        for record in descriptor["scans"]:
            scan = pglMRIScan(
                uuid=record["uuid"],
                name=record["name"],
                sourceMetadata=pglMRIScan._copyMetadata(record["sourceMetadata"]),
                metadata=pglMRIScan._copyMetadata(record["metadata"]),
            )

            # Construction is read-free. The filesystem is retained, not closed;
            # its owner must keep it available for future payload access.
            backing = pglMRIFileBacking(
                filesystem,
                posixpath.join(root, record["path"]),
                expectedIdentity={key: record[key] for key in ("sha256", "size_bytes")},
            )
            if scan.attachBacking(backing, cacheDirectory=cachePath) is None:
                raise RuntimeError("Could not attach a validated MRI bundle backing.")

            scans.append(scan)

        mri = pglMRI()
        mri.scans = scans
        return mri
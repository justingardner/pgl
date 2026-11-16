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
from uuid import uuid4
import numpy as np
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
    
    def getScanSummary(self, *, readHeaders=True):
        """Summarize NIfTI geometry and TR without loading voxel arrays.

        Uses an already-open working image, stored header metadata, or an
        optional source-header read. Does not access self.data or modify metadata.

        scanDims contains the first four dimensions, padded with ones.
        Higher dimensions are reported separately rather than collapsed.

        voxelDims and fov are converted to millimeters when spatial units
        are recognized. Affine matrices retain their original header units.

        trSeconds is the fourth-axis spacing converted to seconds.
        Requires a fourth dimension and recognized temporal units.
        Missing, invalid, unknown-unit, or frequency-domain spacing returns None.

        A transform code of zero means the corresponding matrix is uncoded
        and should not be treated as an authoritative spatial transform.
        """
        import math
        from copy import deepcopy

        # Prefer the working image's actual header when it is already open.
        # This snapshot does not read voxel values.
        snapshot = None
        if self._data is not None:
            snapshot = pglMRINiftiMetadata.fromData(self._data)
        else:
            snapshot = self.metadata.get("nifti")
            if snapshot is None: snapshot = self.sourceMetadata.get("nifti")

        # Read only a missing header, never materialize through self.data.
        # A backing filesystem may independently spool or verify its payload.
        if snapshot is None and readHeaders:
            if self._backing is not None:
                snapshot = pglMRINiftiMetadata.readBacking(self._backing)
            elif self.localPath:
                snapshot = pglMRINiftiMetadata.read(self.localPath)

        summary = {
            "scanDims": None,
            "extraDims": [],
            "voxelDims": None,
            "fov": None,
            "spatialUnits": "unknown",
            "trSeconds": None,
            "qform_code": None,
            "qform": None,
            "sform_code": None,
            "sform": None,
        }
        if snapshot is None: return summary
        if not isinstance(snapshot, dict): raise ValueError("Invalid NIfTI header snapshot.")

        fields = snapshot.get("fields", {})
        transforms = snapshot.get("transforms", {})
        if not isinstance(fields, dict) or not isinstance(transforms, dict):
            raise ValueError("Invalid NIfTI header fields or transforms.")

        # Extract active dimensions and pad absent axes with one.
        # Do not reinterpret higher axes as additional time volumes.
        hasTimeDimension = False
        dimensions = fields.get("dim")
        if isinstance(dimensions, list) and dimensions:
            count = dimensions[0]
            if type(count) is int and 1 <= count <= 7 and len(dimensions) > count:
                shape = dimensions[1:count + 1]
                if all(type(size) is int and size > 0 for size in shape):
                    summary["scanDims"] = (shape + [1, 1, 1, 1])[:4]
                    summary["extraDims"] = shape[4:]
                    hasTimeDimension = count >= 4

        # NIfTI's low three unit bits encode spatial units.
        # Do not assume millimeters when the header declares unknown units.
        units = fields.get("xyzt_units")
        millimetersPerUnit = None
        if type(units) is int:
            spatialCode = units & 0x07
            summary["spatialUnits"] = {1: "m", 2: "mm", 3: "um"}.get(spatialCode, "unknown")
            millimetersPerUnit = {1: 1000.0, 2: 1.0, 3: 0.001}.get(spatialCode)

        # Voxel spacing comes from pixdim, not affine translation or rotation.
        pixelDimensions = fields.get("pixdim")
        if millimetersPerUnit is not None and isinstance(pixelDimensions, list) and len(pixelDimensions) >= 4:
            spacing = pixelDimensions[1:4]
            if all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in spacing):
                summary["voxelDims"] = [float(value) * millimetersPerUnit for value in spacing]

        # Temporal units occupy bits 3–5, independently of spatial units.
        # Accept seconds, milliseconds, and microseconds only.
        # Unknown and frequency-domain units cannot establish a TR.
        secondsPerUnit = None
        if type(units) is int:
            temporalCode = units & 0x38
            secondsPerUnit = {8: 1.0, 16: 0.001, 24: 0.000001}.get(temporalCode)

        if (
            hasTimeDimension
            and secondsPerUnit is not None
            and isinstance(pixelDimensions, list)
            and len(pixelDimensions) >= 5
        ):
            temporalSpacing = pixelDimensions[4]
            if (
                type(temporalSpacing) in (int, float)
                and math.isfinite(temporalSpacing)
                and temporalSpacing > 0
            ):
                summary["trSeconds"] = float(temporalSpacing) * secondsPerUnit

        # FOV spans complete voxels along each image axis.
        # This is not the distance between the first and last voxel centers.
        if summary["scanDims"] is not None and summary["voxelDims"] is not None:
            summary["fov"] = [
                count * spacing
                for count, spacing in zip(summary["scanDims"][:3], summary["voxelDims"])
            ]

        # Preserve both header transforms without selecting or reconciling them.
        # The matrices remain in the original header's coordinate units.
        for form in ("qform", "sform"):
            code = fields.get(f"{form}_code", transforms.get(f"{form}_code"))
            summary[f"{form}_code"] = code if type(code) is int else None
            summary[form] = deepcopy(transforms.get(form))

        return summary


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
    
    def print(self, *, match=None, verbose=False, readHeaders=True):
        """Print aligned geometry summaries for all or matching scans.

        match optionally selects scans whose acquisition label contains the
        supplied string, ignoring case. Filtering occurs before header reads.
        Scan numbers retain their original one-based collection positions.

        Prints the acquisition label first, followed by HH:MM and scan info.
        The acquisition date and timezone appear immediately before the filename.
        Scan numbers and information columns align across selected scans.

        Uses Flywheel's session timezone, falling back to computer-local time.
        Naive timestamps remain unchanged and are labeled explicitly.

        verbose=True adds qform/sform codes and matrices below each summary.
        readHeaders=False avoids opening source files.

        Voxel dimensions and FOV use millimeters.
        Transform matrices retain the original header's spatial units.
        """
        import math
        from pathlib import Path
        from .pglProgress import pglProgress

        if match is not None and not isinstance(match, str):
            raise TypeError("match must be a string or None.")

        scans = list(self.scans)
        if not scans:
            print("No MRI scans.")
            return

        def formatNumber(value):
            """Format finite numbers without unnecessary trailing zeros."""
            if type(value) not in (int, float) or not math.isfinite(value): return "unknown"
            return f"{value:.6g}"

        def formatVector(values):
            return "unknown" if values is None else "[" + ", ".join(formatNumber(value) for value in values) + "]"

        def getScanLabel(scan):
            """Resolve the acquisition label and original filename separately."""
            filename = None
            label = None

            # Prefer explicit Flywheel provenance over parsing the display name.
            selection = scan.sourceMetadata.get("selection", {})
            if isinstance(selection, dict):
                value = selection.get("fileName")
                if isinstance(value, str) and value.strip(): filename = value

            acquisition = scan.sourceMetadata.get("acquisition", {})
            if isinstance(acquisition, dict):
                value = acquisition.get("label")
                if isinstance(value, str) and value.strip(): label = value

            # File metadata can also preserve the original filename.
            fileMetadata = scan.sourceMetadata.get("file", {})
            if filename is None and isinstance(fileMetadata, dict):
                value = fileMetadata.get("name")
                if isinstance(value, str) and value.strip(): filename = value

            # Older display names use "acquisition label: filename".
            # Strip only an exact known filename suffix when one is available.
            if filename is not None and label is None:
                suffix = f": {filename}"
                if scan.name.endswith(suffix): label = scan.name[:-len(suffix)]

            if filename is None:
                prefix, separator, suffix = scan.name.rpartition(": ")
                if separator and suffix.lower().endswith((".nii", ".nii.gz")):
                    filename = suffix
                    if label is None: label = prefix

            # Local scans use their filename; other scans retain their name.
            if filename is None:
                if scan._backing is None and scan.localPath:
                    filename = Path(scan.localPath).name
                else:
                    filename = scan.name or "unnamed"

            # Do not repeat a filename as the leading acquisition label.
            if not label:
                label = scan.name if scan.name and scan.name != filename else "(unlabeled)"

            return label, filename

        def printMatrix(matrix):
            """Print a 4x4 matrix with aligned columns."""
            if not isinstance(matrix, list) or len(matrix) != 4:
                print("      unknown")
                return
            if any(not isinstance(row, list) or len(row) != 4 for row in matrix):
                print("      unknown")
                return

            rows = [[formatNumber(value) for value in row] for row in matrix]
            widths = [max(len(row[column]) for row in rows) for column in range(4)]

            for row in rows:
                columns = "  ".join(value.rjust(width) for value, width in zip(row, widths))
                print(f"      [{columns}]")

        def formatFileSize(scan):
            """Format stored file size in MB or GB with one significant digit."""
            size = None

            # Prefer the backing's recorded byte count when available.
            if scan._backing is not None:
                identity = scan._backing.expectedIdentity
                if identity is not None: size = identity["size_bytes"]

            # Flywheel preserves the source size in its file metadata.
            if size is None:
                fileMetadata = scan.sourceMetadata.get("file", {})
                if isinstance(fileMetadata, dict): size = fileMetadata.get("size")

            # Local files require only a filesystem stat, not a payload read.
            if size is None and scan.localPath:
                try:
                    size = Path(scan.localPath).expanduser().stat().st_size
                except OSError:
                    pass

            if type(size) is not int or size < 0: return "unknown"

            # Use decimal units and round to one significant digit.
            value, unit = (size / 1_000_000, "MB") if size < 1_000_000_000 else (size / 1_000_000_000, "GB")
            rounded = float(f"{value:.1g}")
            return f"{rounded:,.0f} {unit}" if rounded >= 1 or rounded == 0 else f"{rounded:g} {unit}"

        def formatAcquisitionTime(scan):
            """Return separate HH:MM and date/timezone strings.

            Missing or unusable session timezones fall back to computer-local
            time. Naive timestamps remain unchanged; their timezone is unknown.
            Original source metadata is never modified.
            """
            from datetime import datetime
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            # Use the scan acquisition timestamp, not database creation time.
            metadata = scan.sourceMetadata
            if not isinstance(metadata, dict): return "unknown", "unknown"

            acquisition = metadata.get("acquisition", {})
            timestamp = acquisition.get("timestamp") if isinstance(acquisition, dict) else None

            if isinstance(timestamp, datetime):
                acquired = timestamp
            elif isinstance(timestamp, str) and timestamp.strip():
                # Accept ISO timestamps, including UTC timestamps ending in Z.
                try:
                    acquired = datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
                except ValueError:
                    return "unknown", "unknown"
            else:
                return "unknown", "unknown"

            # Do not guess the meaning of a timestamp without an offset.
            if acquired.tzinfo is None or acquired.utcoffset() is None:
                return acquired.strftime("%H:%M"), f"{acquired:%m/%d/%Y} (timezone unspecified)"

            sessionMetadata = metadata.get("session", {})
            sessionTimezone = sessionMetadata.get("timezone") if isinstance(sessionMetadata, dict) else None
            displayTimestamp = None

            if isinstance(sessionTimezone, str) and sessionTimezone.strip():
                try:
                    displayTimestamp = acquired.astimezone(ZoneInfo(sessionTimezone.strip()))
                except (ZoneInfoNotFoundError, ValueError, OverflowError, OSError):
                    pglMessages.warning(f"Could not use scanner-session timezone {sessionTimezone!r} for {scan.name!r}; falling back to computer-local time.", level=1)

            # Absent or empty timezone metadata is an ordinary fallback.
            if displayTimestamp is None:
                try:
                    displayTimestamp = acquired.astimezone()
                except (ValueError, OverflowError, OSError):
                    displayTimestamp = acquired
                    pglMessages.warning(f"Could not convert acquisition time for {scan.name!r} to computer-local time; retaining the source timezone.", level=1)

            timezoneLabel = displayTimestamp.tzname() or displayTimestamp.strftime("%z")
            return displayTimestamp.strftime("%H:%M"), f"{displayTimestamp:%m/%d/%Y} ({timezoneLabel})"

        # Filter labels before reading headers or gathering file information.
        # Preserve original scan numbers rather than renumbering matches.
        selectedScans = []
        matchText = match.casefold() if match is not None else None

        for index, scan in enumerate(scans, start=1):
            label, filename = getScanLabel(scan)
            if matchText is not None and matchText not in label.casefold(): continue
            selectedScans.append((index, scan, label, filename))

        if not selectedScans:
            print(f"No MRI scan labels match {match!r}.")
            return

        if match is not None:
            print(f"{len(selectedScans)} of {len(scans)} MRI scans matched {match!r}.")

        if not selectedScans: return

        # Prepare summaries once so every column can use its longest value.
        rows = []
        with pglProgress("Gathering MRI scan information") as progress:
            for position, (index, scan, label, filename) in enumerate(selectedScans):
                # Progress measures completed scans, not downloaded bytes.
                # Header requests can take different amounts of time per scan.
                progress(position / len(selectedScans), f"Reading scan {position + 1}/{len(selectedScans)}: {label}")
                summary = scan.getScanSummary(readHeaders=readHeaders)
                acquisitionTime, acquisitionDate = formatAcquisitionTime(scan)

                rows.append({
                    "index": index,
                    "label": label,
                    "filename": filename,
                    "summary": summary,
                    "time": acquisitionTime,
                    "date": acquisitionDate,
                    "dims": f"dims={formatVector(summary['scanDims'])}",
                    "voxelDims": f"voxelDims={formatVector(summary['voxelDims'])}mm",
                    "fov": f"fov={formatVector(summary['fov'])}mm",
                    "size": f"size={formatFileSize(scan)}",
                    "extraDims": f"extraDims={formatVector(summary['extraDims'])}" if summary["extraDims"] else "",
                })

            progress.finish(f"Gathered information for {len(selectedScans)} MRI scans")

        numberWidth = len(str(len(scans)))
        labelWidth = max(len(row["label"]) for row in rows)
        columnNames = ["time", "dims", "voxelDims", "fov", "size"]
        if any(row["extraDims"] for row in rows): columnNames.append("extraDims")
        columnNames.append("date")
        columnWidths = {name: max(len(row[name]) for row in rows) for name in columnNames}

        # Preserve collection order and number scans starting at one.
        for row in rows:
            summary = row["summary"]
            information = "  ".join(row[name].ljust(columnWidths[name]) for name in columnNames)

            # Keep the ordinary summary on one line without hiding higher axes.
            print(f"{row['index']:>{numberWidth}}: {row['label']:<{labelWidth}}  {information} : {row['filename']}")

            # Show both header transforms only when verbose output is requested.
            if verbose:
                for form in ("qform", "sform"):
                    code = summary[f"{form}_code"]
                    codeText = "unknown" if code is None else str(code)
                    note = " (uncoded; not an authoritative transform)" if code == 0 else ""
                    print(f"   {form}_code={codeText}{note}")
                    print(f"   {form} (header spatial units: {summary['spatialUnits']}):")
                    printMatrix(summary[form])
    
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
    
################################################################
# Coordinate-frame and linear-transform metadata
################################################################
class pglMRITransforms:
    """Construct and manipulate explicit linear coordinate mappings.

    Records contain JSON-compatible metadata, not image arrays or readers.

    All matrices use homogeneous column vectors:
        destination = matrix @ source

    Physical geometry is represented internally as RAS millimeters:
        positive X = Right, positive Y = Anterior, positive Z = Superior.

    This module does not decide which NIfTI header is authoritative. Callers
    must explicitly supply the acquisition geometry selected and validated by
    the action/tool adapter.

    These are internal scientific helpers. Invalid definitions raise to the
    owning action runner, which warns and stops execution.

    No file writes, source modification, resampling, or external-tool calls
    occur here.
    """

    ################################################################
    # Validate a homogeneous matrix
    ################################################################
    @staticmethod
    def _matrix(value, label):
        """Return a finite, invertible 4x4 matrix without caller aliases."""
        try:
            matrix = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must contain numerical values.") from None

        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{label} must be a finite 4x4 matrix.")
        if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10):
            raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1].")

        try:
            np.linalg.inv(matrix[:3, :3])
        except np.linalg.LinAlgError:
            raise ValueError(f"{label} must be invertible.") from None

        return matrix.copy()

    @staticmethod
    def _text(value, label):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{label} must be a nonblank string without NUL.")
        return value

    ################################################################
    # Construct a coordinate-frame record
    ################################################################
    @classmethod
    def makeFrame(cls, *, geometryID, physicalSpaceID, spatialShape, voxelToPhysicalMM, representation, voxelToFrame=None, frameID=None):
        """Describe coordinates associated with an explicitly identified grid.

        geometryID:
            Stable identity of this exact image geometry/source stage.
            Do not generate a different identity each time it is inspected.

        physicalSpaceID:
            Identity of the physical reference used by voxelToPhysicalMM.
            A label such as RAS or scanner is not a reference-space identity.
            Do not assume different acquisitions share a reference merely
            because their headers contain the same transform code.

        spatialShape:
            Three spatial dimensions in voxel-index order.

        voxelToPhysicalMM:
            Full voxel-center-index to physical-coordinate matrix, including
            origin, orientation, and spacing. Coordinates must be RAS mm.

        representation:
            voxelIndices:
                Zero-based voxel-center indices; voxelToFrame is identity.

            scannerRAS:
                RAS physical coordinates in mm.

            scannerLPS:
                LPS physical coordinates in mm.

            toolNominal:
                Tool-defined nominal dataset coordinates in mm.
                Requires an explicit voxelToFrame supplied by its adapter.
                The original acquisition angle remains separately recorded in
                voxelToPhysicalMM.

        A representation label does not prove that a tool used those coordinates.
        Tool adapters must establish that correspondence through testing.
        """
        cls._text(geometryID, "geometryID")
        cls._text(physicalSpaceID, "physicalSpaceID")
        if frameID is not None: cls._text(frameID, "frameID")

        if (
            not isinstance(spatialShape, (list, tuple))
            or len(spatialShape) != 3
            or any(type(size) is not int or size <= 0 for size in spatialShape)
        ):
            raise ValueError("spatialShape must contain three positive integers.")

        physical = cls._matrix(voxelToPhysicalMM, "voxelToPhysicalMM")
        rasToLPS = np.diag([-1.0, -1.0, 1.0, 1.0])

        if representation == "voxelIndices":
            expected = np.eye(4)
            units = "voxel"
        elif representation == "scannerRAS":
            expected = physical
            units = "mm"
        elif representation == "scannerLPS":
            expected = rasToLPS @ physical
            units = "mm"
        elif representation == "toolNominal":
            if voxelToFrame is None:
                raise ValueError("toolNominal requires an explicit voxelToFrame matrix.")
            expected = None
            units = "mm"
        else:
            raise ValueError("Unsupported coordinate representation.")

        mapping = expected if voxelToFrame is None else cls._matrix(voxelToFrame, "voxelToFrame")

        if expected is not None and not np.allclose(mapping, expected, rtol=0, atol=1e-8):
            raise ValueError("voxelToFrame disagrees with the declared coordinate representation.")

        return {
            "schemaVersion": 1,
            "frameID": frameID or str(uuid4()),
            "geometryID": geometryID,
            "physicalSpaceID": physicalSpaceID,
            "spatialShape": list(spatialShape),
            "representation": representation,
            "units": units,
            "indexOrigin": 0,
            "indexLocation": "voxelCenter",
            "voxelToFrame": mapping.tolist(),
            "voxelToPhysicalMM": physical.tolist(),
        }

    ################################################################
    # Validate a restored frame
    ################################################################
    @classmethod
    def _validateFrame(cls, frame):
        """Validate recorded definitions without fabricating missing fields."""
        expectedFields = {
            "schemaVersion", "frameID", "geometryID", "physicalSpaceID",
            "spatialShape", "representation", "units", "indexOrigin",
            "indexLocation", "voxelToFrame", "voxelToPhysicalMM",
        }
        if not isinstance(frame, dict) or set(frame) != expectedFields:
            raise ValueError("Invalid coordinate-frame fields.")
        if type(frame["schemaVersion"]) is not int or frame["schemaVersion"] != 1:
            raise ValueError("Unsupported coordinate-frame version.")
        if type(frame["indexOrigin"]) is not int or frame["indexOrigin"] != 0:
            raise ValueError("Only zero-based voxel indices are supported.")
        if frame["indexLocation"] != "voxelCenter":
            raise ValueError("Only voxel-center indexing is supported.")

        validated = cls.makeFrame(
            geometryID=frame["geometryID"],
            physicalSpaceID=frame["physicalSpaceID"],
            spatialShape=frame["spatialShape"],
            voxelToPhysicalMM=frame["voxelToPhysicalMM"],
            representation=frame["representation"],
            voxelToFrame=frame["voxelToFrame"],
            frameID=frame["frameID"],
        )
        if frame["units"] != validated["units"]:
            raise ValueError("Coordinate units disagree with the representation.")

        return validated

    ################################################################
    # Construct a transform record
    ################################################################
    @classmethod
    def makeTransform(cls, matrix, *, domainFrame, codomainFrame, purpose="sampling", interpretationStatus="provisional", adapter="explicit", transformID=None, inputTransformIDs=None):
        """Record an affine mapping with explicit endpoint frames.

        domainFrame:
            Coordinates accepted by the matrix.

        codomainFrame:
            Coordinates produced by the matrix.

        purpose:
            sampling: output/reference location -> input sampling location.
            pointMapping: source point -> destination point.

        interpretationStatus:
            provisional: coordinate interpretation still needs validation.
            validated: the supplying adapter has established the convention.

        'validated' is a recorded assertion by the caller/adapter, not a test
        performed by this constructor. Coordinate conversion cannot upgrade
        a provisional interpretation.

        All matrices are recorded as affine. A rigid physical transform may
        have a nonorthogonal numerical matrix after conversion to differently
        scaled voxel coordinates; do not infer rigidity from that matrix alone.

        inputTransformIDs links algebraic conversions/composites to their
        inputs. Product registration supplies action/invocation provenance
        and references to original tool artifacts separately.
        """
        mapping = cls._matrix(matrix, "transform matrix")
        domain = cls._validateFrame(domainFrame)
        codomain = cls._validateFrame(codomainFrame)

        if purpose not in ("sampling", "pointMapping"):
            raise ValueError("purpose must be sampling or pointMapping.")
        if interpretationStatus not in ("provisional", "validated"):
            raise ValueError("interpretationStatus must be provisional or validated.")
        cls._text(adapter, "adapter")
        if transformID is not None: cls._text(transformID, "transformID")

        inputs = [] if inputTransformIDs is None else list(inputTransformIDs)
        for identifier in inputs: cls._text(identifier, "input transform identity")
        if len(set(inputs)) != len(inputs):
            raise ValueError("inputTransformIDs must not contain duplicates.")

        if domain["frameID"] == codomain["frameID"] and domain != codomain:
            raise ValueError("One frame ID cannot have conflicting definitions.")

        return {
            "schemaVersion": 1,
            "transformID": transformID or str(uuid4()),
            "transformType": "affine",
            "matrixConvention": "homogeneousColumnVectors",
            "matrix": mapping.tolist(),
            "domainFrame": domain,
            "codomainFrame": codomain,
            "purpose": purpose,
            "interpretationStatus": interpretationStatus,
            "adapter": adapter,
            "inputTransformIDs": inputs,
        }

    ################################################################
    # Validate a restored transform
    ################################################################
    @classmethod
    def _validateTransform(cls, transform):
        expectedFields = {
            "schemaVersion", "transformID", "transformType", "matrixConvention",
            "matrix", "domainFrame", "codomainFrame", "purpose",
            "interpretationStatus", "adapter", "inputTransformIDs",
        }
        if not isinstance(transform, dict) or set(transform) != expectedFields:
            raise ValueError("Invalid transform fields.")
        if type(transform["schemaVersion"]) is not int or transform["schemaVersion"] != 1:
            raise ValueError("Unsupported transform version.")
        if transform["transformType"] != "affine":
            raise ValueError("This implementation supports affine matrices only.")
        if transform["matrixConvention"] != "homogeneousColumnVectors":
            raise ValueError("Unsupported matrix convention.")
        if type(transform["inputTransformIDs"]) is not list:
            raise ValueError("inputTransformIDs must be a list.")

        return cls.makeTransform(
            transform["matrix"],
            domainFrame=transform["domainFrame"],
            codomainFrame=transform["codomainFrame"],
            purpose=transform["purpose"],
            interpretationStatus=transform["interpretationStatus"],
            adapter=transform["adapter"],
            transformID=transform["transformID"],
            inputTransformIDs=transform["inputTransformIDs"],
        )

    ################################################################
    # Convert coordinate descriptions without moving image data
    ################################################################
    @classmethod
    def convert(cls, transform, *, domainFrame=None, codomainFrame=None):
        """Re-express a mapping in other coordinates of the same endpoint grids.

        Does not infer registration between different images or grids.

        If B maps voxel indices into a frame, conversion from frame F to G is:
            coordinatesG = B_G @ inverse(B_F) @ coordinatesF

        Therefore:
            newMatrix =
                B_newCodomain @ inverse(B_oldCodomain)
                @ oldMatrix
                @ B_oldDomain @ inverse(B_newDomain)

        No voxel data are interpolated. Provisional status is preserved.
        """
        original = cls._validateTransform(transform)
        oldDomain = original["domainFrame"]
        oldCodomain = original["codomainFrame"]
        newDomain = oldDomain if domainFrame is None else cls._validateFrame(domainFrame)
        newCodomain = oldCodomain if codomainFrame is None else cls._validateFrame(codomainFrame)

        def requireSameGeometry(old, new):
            if (
                old["geometryID"] != new["geometryID"]
                or old["physicalSpaceID"] != new["physicalSpaceID"]
                or old["spatialShape"] != new["spatialShape"]
                or not np.allclose(old["voxelToPhysicalMM"], new["voxelToPhysicalMM"], rtol=0, atol=1e-8)
            ):
                raise ValueError("Coordinate conversion requires the same endpoint image geometry.")
            if old["frameID"] == new["frameID"] and old != new:
                raise ValueError("A changed frame definition requires a different frame ID.")

        requireSameGeometry(oldDomain, newDomain)
        requireSameGeometry(oldCodomain, newCodomain)

        oldDomainMatrix = np.asarray(oldDomain["voxelToFrame"])
        oldCodomainMatrix = np.asarray(oldCodomain["voxelToFrame"])
        newDomainMatrix = np.asarray(newDomain["voxelToFrame"])
        newCodomainMatrix = np.asarray(newCodomain["voxelToFrame"])

        matrix = (
            newCodomainMatrix
            @ np.linalg.inv(oldCodomainMatrix)
            @ np.asarray(original["matrix"])
            @ oldDomainMatrix
            @ np.linalg.inv(newDomainMatrix)
        )

        return cls.makeTransform(
            matrix,
            domainFrame=newDomain,
            codomainFrame=newCodomain,
            purpose=original["purpose"],
            interpretationStatus=original["interpretationStatus"],
            adapter="pgl.coordinateConversion.v1",
            inputTransformIDs=[original["transformID"]],
        )

    ################################################################
    # Compose connected mappings
    ################################################################
    @classmethod
    def compose(cls, first, second):
        """Return the mapping obtained by applying first, THEN second.

        With column vectors:
            result.matrix = second.matrix @ first.matrix

        The intermediate frame identity and complete definition must match.
        Convert coordinate descriptions explicitly before composition if needed.

        Does not sample images. Any provisional input keeps the result
        provisional.
        """
        first = cls._validateTransform(first)
        second = cls._validateTransform(second)

        if first["codomainFrame"] != second["domainFrame"]:
            raise ValueError("Transform endpoints do not connect. Resolve/convert the intermediate frame explicitly.")
        if first["purpose"] != second["purpose"]:
            raise ValueError("Sampling and point mappings must not be composed without explicit interpretation.")

        inputs = list(dict.fromkeys([first["transformID"], second["transformID"]]))
        status = (
            "validated"
            if first["interpretationStatus"] == second["interpretationStatus"] == "validated"
            else "provisional"
        )

        return cls.makeTransform(
            np.asarray(second["matrix"]) @ np.asarray(first["matrix"]),
            domainFrame=first["domainFrame"],
            codomainFrame=second["codomainFrame"],
            purpose=first["purpose"],
            interpretationStatus=status,
            adapter="pgl.affineComposition.v1",
            inputTransformIDs=inputs,
        )

    ################################################################
    # Explain coordinates in ordinary language
    ################################################################
    @classmethod
    def describeFrame(cls, frame):
        """Generate an explanation from validated structured fields."""
        frame = cls._validateFrame(frame)
        representation = frame["representation"]

        descriptions = {
            "voxelIndices": (
                "Positions are measured by voxel number along the image's first, "
                "second, and third axes, starting at zero at the center of the "
                "first voxel. Fractional positions lie between voxel centers."
            ),
            "scannerRAS": (
                "Positions are measured in millimeters in the recorded physical "
                "reference space. Positive axes point Right, Anterior, and "
                "Superior. The image's recorded position and acquisition angle "
                "are included in the voxel-to-coordinate mapping."
            ),
            "scannerLPS": (
                "Positions are measured in millimeters in the recorded physical "
                "reference space. Positive axes point Left, Posterior, and "
                "Superior. The image's recorded position and acquisition angle "
                "are included in the voxel-to-coordinate mapping."
            ),
            "toolNominal": (
                "Positions are measured in millimeters using the tool's nominal "
                "description of the image grid. The angle at which the scan was "
                "acquired is not included in these axes; the original physical "
                "geometry is stored separately. Axis order, signs, spacing, and "
                "origin are defined by voxelToFrame, not guessed from the tool name."
            ),
        }
        return descriptions[representation]

    ################################################################
    # Explain a complete transform
    ################################################################
    @classmethod
    def describe(cls, transform):
        """Return readable semantics without maintaining a separate text truth."""
        record = cls._validateTransform(transform)

        meaning = (
            "For each location in the output/reference, this matrix gives the "
            "location to sample in the input. It does not directly describe "
            "where to push an input voxel."
            if record["purpose"] == "sampling" else
            "For a point in the source frame, this matrix gives its position "
            "in the destination frame."
        )
        status = (
            "PROVISIONAL: coordinate interpretation is not approved for application."
            if record["interpretationStatus"] == "provisional" else
            "Coordinate interpretation is declared validated by the supplying adapter; "
            "this does not establish scientific registration quality."
        )

        return (
            f"Transform: {record['transformID']}\n"
            f"Mapping: {record['domainFrame']['frameID']} → {record['codomainFrame']['frameID']}\n\n"
            f"{meaning}\n\n"
            f"Starting coordinates:\n{cls.describeFrame(record['domainFrame'])}\n\n"
            f"Result coordinates:\n{cls.describeFrame(record['codomainFrame'])}\n\n"
            "Recording, converting, or composing this matrix does not change "
            "image data. Applying it to generate an image generally requires "
            "interpolation on an explicitly selected output grid.\n\n"
            f"{status}"
        )
        
    ################################################################
    # Interpret within-run AFNI motion matrices
    ################################################################
    @classmethod
    def interpretAFNIVolreg(cls, matrixRows, *, attributes, geometryID, physicalSpaceID, spatialShape, voxelToPhysicalMM, volumeIndices, referenceVolumeIndex, afniVersion):
        """Convert native within-run 3dvolreg matrices to voxel sampling maps.

        Pure metadata/matrix operation:
            No image reads, header edits, resampling, or external commands.

        matrixRows:
            Native -1Dmatrix_save rows, one 12-value row per estimated volume.

        attributes:
            Numerical values obtained from the SAME source image using
            3dAttribute:
                ORIENT_SPECIFIC
                DELTA
                ORIGIN
                IJK_TO_DICOM_REAL

        voxelToPhysicalMM:
            Original acquisition voxel-to-RAS-mm mapping, selected and
            validated by the calling action.

        volumeIndices:
            Original acquisition volume indices in matrix-row order.
            Must also describe the order of volumes supplied to 3dvolreg.

        referenceVolumeIndex:
            Original acquisition index of the reference in that same series.

        afniVersion:
            Reported version/build text from the installation used.

        Scope:
            Within-run, common-grid 3dvolreg -1Dmatrix_save interpretation.
            Reference and input share the supplied geometry.
            Does not handle an external reference image or another source grid.

        Convention:
            Native matrices map reference locations to input sampling
            locations in AFNI's nominal dataset coordinates.

            B maps voxel indices to those nominal coordinates.
            V = inverse(B) @ M @ B maps reference voxel indices to input
            voxel indices.

        Returns:
            A portable record containing frames, native interpreted transforms,
            converted sampling transforms, original-volume associations, and
            readable descriptions.

        Validation:
            Coordinate interpretation was tested using AFNI 26.3.11 on macOS
            ARM with synthetic translation, rotation, and permuted/flipped axes.
            This does not independently verify registration quality or certify
            every AFNI build. The actual build is retained separately.

        Raises internally to the owning action runner on invalid definitions.
        """
        if not isinstance(afniVersion, str) or not afniVersion.strip():
            raise ValueError("Record the actual AFNI version/build used.")

        if (
            type(volumeIndices) is not list or not volumeIndices
            or any(type(index) is not int or index < 0 for index in volumeIndices)
            or len(set(volumeIndices)) != len(volumeIndices)
        ):
            raise ValueError("volumeIndices must contain unique nonnegative original-volume indices.")

        if type(referenceVolumeIndex) is not int or referenceVolumeIndex not in volumeIndices:
            raise ValueError("The reference must be a volume represented in this motion estimate.")

        rows = np.asarray(matrixRows, dtype=float)
        if rows.shape != (len(volumeIndices), 12) or not np.isfinite(rows).all():
            raise ValueError("Expected one finite 12-value matrix row per declared volume.")

        if not isinstance(attributes, dict):
            raise TypeError("AFNI geometry attributes must be a dictionary.")

        values = {}
        for name, count in (
            ("ORIENT_SPECIFIC", 3),
            ("DELTA", 3),
            ("ORIGIN", 3),
            ("IJK_TO_DICOM_REAL", 12),
        ):
            try:
                value = np.asarray(attributes[name], dtype=float)
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Missing or invalid AFNI attribute {name}.") from None

            if value.size != count or not np.isfinite(value).all():
                raise ValueError(f"AFNI attribute {name} must contain {count} finite values.")
            values[name] = value.reshape(-1)

        orientation = values["ORIENT_SPECIFIC"]
        if not np.equal(orientation, np.floor(orientation)).all():
            raise ValueError("AFNI orientation codes must be integers.")

        orientation = orientation.astype(int)
        if np.any(orientation < 0) or np.any(orientation > 5):
            raise ValueError("Unsupported AFNI orientation codes.")

        # AFNI codes:
        # 0 R→L, 1 L→R, 2 P→A, 3 A→P, 4 I→S, 5 S→I.
        # DELTA already includes direction signs; do not apply them twice.
        physicalAxes = [int(code // 2) for code in orientation]
        if len(set(physicalAxes)) != 3:
            raise ValueError("AFNI orientation must describe three distinct axes.")

        delta = values["DELTA"]
        expectedSigns = np.array([1, -1, -1, 1, 1, -1])[orientation]
        if np.any(delta == 0) or not np.array_equal(np.sign(delta), expectedSigns):
            raise ValueError("AFNI DELTA signs disagree with ORIENT_SPECIFIC.")

        physical = cls._matrix(voxelToPhysicalMM, "voxelToPhysicalMM")
        spacing = np.linalg.norm(physical[:3, :3], axis=0)

        # 3dAttribute prints rounded numerical text. These checks establish
        # correspondence, not permission to alter the acquisition geometry.
        if not np.allclose(np.abs(delta), spacing, rtol=0, atol=0.001):
            raise ValueError("AFNI nominal voxel spacing differs from the declared acquisition spacing.")

        nominal = np.zeros((4, 4))
        nominal[3, 3] = 1
        for voxelAxis, physicalAxis in enumerate(physicalAxes):
            nominal[physicalAxis, voxelAxis] = delta[voxelAxis]
            nominal[physicalAxis, 3] = values["ORIGIN"][voxelAxis]

        geometry = {
            "geometryID": geometryID,
            "physicalSpaceID": physicalSpaceID,
            "spatialShape": spatialShape,
            "voxelToPhysicalMM": physical,
        }
        nativeFrame = cls.makeFrame(
            **geometry,
            representation="toolNominal",
            voxelToFrame=nominal,
        )
        voxelFrame = cls.makeFrame(**geometry, representation="voxelIndices")

        realDICOM = np.eye(4)
        realDICOM[:3] = values["IJK_TO_DICOM_REAL"].reshape(3, 4)
        realDICOM = cls._matrix(realDICOM, "AFNI IJK_TO_DICOM_REAL")
        rasToLPS = np.diag([-1.0, -1.0, 1.0, 1.0])

        corners = np.array([
            [x, y, z, 1.0]
            for x in (0, spatialShape[0] - 1)
            for y in (0, spatialShape[1] - 1)
            for z in (0, spatialShape[2] - 1)
        ]).T
        geometryError = float(np.linalg.norm(
            ((realDICOM - rasToLPS @ physical) @ corners)[:3],
            axis=0,
        ).max())

        if geometryError > 0.01:
            raise ValueError(
                f"AFNI's reported physical mapping differs from the supplied "
                f"acquisition geometry by up to {geometryError:.6g} mm."
            )

        nativeTransforms = []
        voxelTransforms = []
        associations = []
        adapter = "pgl.AFNI.3dvolreg.sharedGrid.v1"

        for rowIndex, (originalVolumeIndex, row) in enumerate(zip(volumeIndices, rows)):
            matrix = np.eye(4)
            matrix[:3] = row.reshape(3, 4)

            # Within a shared nominal millimeter frame, 3dvolreg's linear
            # component should be a proper rotation, not scale/shear/reflection.
            linear = matrix[:3, :3]
            if (
                not np.allclose(linear.T @ linear, np.eye(3), rtol=0, atol=0.0001)
                or abs(np.linalg.det(linear) - 1.0) > 0.0001
            ):
                raise ValueError(f"Motion matrix row {rowIndex} is not a rigid rotation/translation.")

            native = cls.makeTransform(
                matrix,
                domainFrame=nativeFrame,
                codomainFrame=nativeFrame,
                purpose="sampling",
                interpretationStatus="validated",
                adapter=adapter,
            )
            converted = cls.convert(
                native,
                domainFrame=voxelFrame,
                codomainFrame=voxelFrame,
            )

            nativeTransforms.append(native)
            voxelTransforms.append(converted)
            associations.append({
                "rowIndex": rowIndex,
                "originalVolumeIndex": originalVolumeIndex,
                "referenceVolumeIndex": referenceVolumeIndex,
                "nativeTransformID": native["transformID"],
                "voxelTransformID": converted["transformID"],
            })

        explanation = (
            "These motion estimates describe head rotation and movement relative "
            "to a reference volume. AFNI expresses positions in millimeters using "
            "its nominal image-grid axes, rather than including the recorded "
            "angle of the acquisition in those axes.\n\n"
            "The native matrix tells us where to sample the input for a location "
            "in the reference. PGL converts that description into zero-based voxel "
            "coordinates using the recorded axis order, signs, spacing, and origin.\n\n"
            "The converted matrix tells us which input voxel location to sample "
            "for each reference voxel. Fractional voxel positions require "
            "interpolation when correction is applied. Conversion alone changes "
            "neither voxel values nor image headers."
        )

        return {
            "schemaVersion": 1,
            "kind": "AFNIWithinRunMotionInterpretation",
            "adapter": adapter,
            "afniVersion": afniVersion,
            "validationScope": {
                "coordinateInterpretation": "validatedForSharedGridWithinRunMapping",
                "evidence": (
                    "Synthetic translation and rotation tests, including "
                    "permuted/flipped axes, passed with AFNI 26.3.11 on macOS ARM."
                ),
                "registrationQuality": "Not established by coordinate conversion.",
                "crossGridRegistration": "Not supported by this adapter.",
            },
            "geometryID": geometryID,
            "nativeFrame": nativeFrame,
            "voxelFrame": voxelFrame,
            "afniAttributes": {name: value.tolist() for name, value in values.items()},
            "attributeGeometryErrorMM": geometryError,
            "volumeAssociations": associations,
            "nativeTransforms": nativeTransforms,
            "voxelTransforms": voxelTransforms,
            "description": explanation,
        }
        
    ################################################################
    # Describe motion in patient-oriented units
    ################################################################
    @classmethod
    def motionSummary(cls, transforms, *, volumeIndices, referenceVolumeIndex, trSeconds=None, volumeTimesSeconds=None, measurementPointRASMM=None):
        """Calculate readable motion traces without reading or changing images.

        transforms:
            One validated reference-to-input sampling transform per volume.
            Endpoint definitions must describe a common physical reference
            space. All transforms must use the same reference image geometry.

            Matrices are converted to RAS millimeters before decomposition.
            Raw AFNI matrix rows are not accepted directly.

        volumeIndices:
            Original acquisition volume indices in transform order.

        referenceVolumeIndex:
            Original acquisition index of the reference volume.

        trSeconds:
            Optional positive acquisition TR. When explicit times are absent,
            time is originalVolumeIndex * TR. Dropped volumes do not reset time.

        volumeTimesSeconds:
            Optional explicit acquisition-relative volume times in seconds,
            one per transform. Takes precedence over TR-derived times.
            Times must increase in the supplied volume order.

        measurementPointRASMM:
            Fixed physical point at which displacement is measured.
            None uses the center of the reference image's voxel grid.
            This is an image-center measurement, not a tracked anatomical
            landmark or automatically estimated center of the head.

        Interpretation:
            For rigid registration of anatomy, a reference-to-input sampling
            map describes estimated head position relative to the reference.
            It is not the inverse correction point map.

            For general affine registration, scale/shear describe the fitted
            image mapping, not necessarily actual physical deformation.

        Rotation:
            Right polar decomposition separates rotation from symmetric
            stretch: linear = rotation @ stretch.
            Rotation-vector components use the principal angle (up to 180 deg).
            They are not sequential Euler rotations.

        Scale/shear:
            diagonal(stretch) - 1 is reported as percent stretch.
            Off-diagonal stretch entries are reported as percent symmetric
            cross-axis coupling after removing rotation.
            These are not engineering shear strains or shear angles.

        Returns JSON-compatible numbers and generated descriptions.
        Internal validation errors propagate to the action/display boundary.
        Does not plot, interpolate images, or mutate the supplied transforms.
        """
        import math
        from scipy.spatial.transform import Rotation

        if not isinstance(transforms, (list, tuple)) or not transforms:
            raise ValueError("Supply at least one interpreted transform.")

        if (
            type(volumeIndices) is not list
            or len(volumeIndices) != len(transforms)
            or any(type(index) is not int or index < 0 for index in volumeIndices)
            or len(set(volumeIndices)) != len(volumeIndices)
        ):
            raise ValueError("volumeIndices must provide one unique nonnegative original index per transform.")

        if type(referenceVolumeIndex) is not int or referenceVolumeIndex not in volumeIndices:
            raise ValueError("The reference volume must be represented in volumeIndices.")

        if trSeconds is not None:
            if type(trSeconds) not in (int, float) or not math.isfinite(trSeconds) or trSeconds <= 0:
                raise ValueError("trSeconds must be positive and finite.")

        if volumeTimesSeconds is not None:
            if not isinstance(volumeTimesSeconds, (list, tuple)) or len(volumeTimesSeconds) != len(transforms):
                raise ValueError("Supply one explicit time per transform.")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in volumeTimesSeconds):
                raise ValueError("Explicit volume times must be finite numbers.")

            timeValues = [float(value) for value in volumeTimesSeconds]
            if any(right <= left for left, right in zip(timeValues, timeValues[1:])):
                raise ValueError("Explicit volume times must increase in the supplied order.")

            timeSource = "explicitVolumeTimes"
            timeUnits = "seconds"
            timeLabel = "Time from acquisition start (s)"

        elif trSeconds is not None:
            timeValues = [float(index * trSeconds) for index in volumeIndices]
            timeSource = "originalVolumeIndexTimesTR"
            timeUnits = "seconds"
            timeLabel = "Time from acquisition start (s)"

        else:
            timeValues = list(volumeIndices)
            timeSource = "originalVolumeIndex"
            timeUnits = "volumeIndex"
            timeLabel = "Original volume index"

        records = [cls._validateTransform(record) for record in transforms]
        referenceFrame = records[0]["domainFrame"]

        for record in records:
            if record["purpose"] != "sampling":
                raise ValueError("Motion display requires declared reference-to-input sampling maps.")
            if record["interpretationStatus"] != "validated":
                raise ValueError("Motion display requires a validated coordinate interpretation.")

            domain = record["domainFrame"]
            codomain = record["codomainFrame"]

            if (
                domain["geometryID"] != referenceFrame["geometryID"]
                or domain["physicalSpaceID"] != referenceFrame["physicalSpaceID"]
                or domain["spatialShape"] != referenceFrame["spatialShape"]
                or not np.allclose(
                    domain["voxelToPhysicalMM"],
                    referenceFrame["voxelToPhysicalMM"],
                    rtol=0,
                    atol=1e-8,
                )
            ):
                raise ValueError("All motion estimates must share one reference image geometry.")

            if domain["physicalSpaceID"] != codomain["physicalSpaceID"]:
                raise ValueError(
                    "Motion endpoints do not share a declared physical reference. "
                    "Provide an explicit common-frame interpretation before display."
                )

        referencePhysical = np.asarray(referenceFrame["voxelToPhysicalMM"])

        if measurementPointRASMM is None:
            centerVoxel = (np.asarray(referenceFrame["spatialShape"], dtype=float) - 1) / 2
            point = (referencePhysical @ np.append(centerVoxel, 1.0))[:3]
            pointDescription = "Center of the reference image grid"
        else:
            point = np.asarray(measurementPointRASMM, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("measurementPointRASMM must contain three finite coordinates.")
            pointDescription = "Explicit fixed measurement point"

        def physicalFrame(frame):
            """Change coordinate representation, not the underlying image."""
            return cls.makeFrame(
                geometryID=frame["geometryID"],
                physicalSpaceID=frame["physicalSpaceID"],
                spatialShape=frame["spatialShape"],
                voxelToPhysicalMM=frame["voxelToPhysicalMM"],
                representation="scannerRAS",
            )

        displacements = []
        rotations = []
        stretches = []
        couplings = []

        for record in records:
            converted = cls.convert(
                record,
                domainFrame=physicalFrame(record["domainFrame"]),
                codomainFrame=physicalFrame(record["codomainFrame"]),
            )
            matrix = np.asarray(converted["matrix"])
            linear = matrix[:3, :3]

            # Reject reflected mappings rather than forcing them into a
            # plausible-looking positive rotation.
            if np.linalg.det(linear) <= 0:
                raise ValueError("Reflected mappings cannot be displayed as ordinary head motion.")

            left, singularValues, rightTranspose = np.linalg.svd(linear)
            rotation = left @ rightTranspose
            stretch = rightTranspose.T @ np.diag(singularValues) @ rightTranspose

            displacement = linear @ point + matrix[:3, 3] - point
            rotationVector = Rotation.from_matrix(rotation).as_rotvec(degrees=True)

            displacements.append(displacement.tolist())
            rotations.append(rotationVector.tolist())
            stretches.append((100 * (np.diag(stretch) - 1)).tolist())
            couplings.append((100 * np.array([
                stretch[0, 1], stretch[0, 2], stretch[1, 2],
            ])).tolist())

        referenceRow = volumeIndices.index(referenceVolumeIndex)

        return {
            "schemaVersion": 1,
            "kind": "motionSummary",
            "transformIDs": [record["transformID"] for record in records],
            "volumeIndices": list(volumeIndices),
            "referenceVolumeIndex": referenceVolumeIndex,
            "referenceRow": referenceRow,
            "time": {
                "values": timeValues,
                "units": timeUnits,
                "source": timeSource,
                "label": timeLabel,
                "referenceValue": timeValues[referenceRow],
            },
            "coordinateConvention": "patientRAS",
            "axisLabels": [
                "Left–right (+ right)",
                "Posterior–anterior (+ toward face)",
                "Inferior–superior (+ toward top of head)",
            ],
            "measurementPointRASMM": point.tolist(),
            "measurementPointDescription": pointDescription,
            "translationMM": displacements,
            "rotationVectorDegrees": rotations,
            "stretchPercent": stretches,
            "symmetricCouplingPercent": couplings,
            "couplingLabels": [
                "Left–right / posterior–anterior",
                "Left–right / inferior–superior",
                "Posterior–anterior / inferior–superior",
            ],
            "description": (
                "Estimated movement relative to the reference volume. Zero "
                "displacement means the fixed measurement point has not moved; "
                "zero rotation means no rotation relative to the reference. "
                "The reference estimates are not artificially reset to zero.\n\n"
                "Translations describe movement of the reference image center "
                "or explicitly supplied point, not movement of scanner coordinate "
                "zero. Rotations are rotation-vector components about fixed "
                "patient right, anterior, and superior axes, in degrees; they "
                "are not three successive Euler rotations.\n\n"
                "For affine estimates, stretch and symmetric cross-axis coupling "
                "describe the fitted image mapping after rotation is separated. "
                "A coupling of 1% corresponds to 1 mm of cross-axis displacement "
                "per 100 mm along the other axis in that symmetric stretch "
                "mapping. These values are not automatically physical head deformation."
            ),
        }
    
    ################################################################
    # Display motion in patient-oriented units
    ################################################################
    @classmethod
    def displayMotion(cls, transforms, *, volumeIndices, referenceVolumeIndex, trSeconds=None, volumeTimesSeconds=None, measurementPointRASMM=None, showAffineComponents=None, title="Head movement relative to reference", fig=None):
        """Plot interpreted motion without changing scientific data.

        Uses motionSummary() for coordinate conversion and numerical values.

        showAffineComponents:
            False: translation and rotation panels only.
            True: also show stretch and symmetric cross-axis coupling.
            None: show affine panels when either exceeds 0.001 percent.
            This display threshold does not modify or round recorded values.

        fig:
            Optional Matplotlib figure to replace after input validation.
            None creates a new figure.

        Returns:
            Matplotlib figure, or None after a warning.
            Does not call plt.show(), save a file, or modify session state.

        Validation failures use the default stopping-warning level.
        Rendering failures use level=1 because scientific results remain valid.

        Translation is displacement at a fixed point: the reference image
        center by default, not an automatically identified anatomical landmark.

        Rotations are principal rotation-vector components, not sequential
        Euler rotations. Positive angles follow the right-hand rule around
        the positive patient RAS axes.

        Affine stretch/coupling describe the fitted image mapping. They should
        not automatically be interpreted as physical deformation of the head.
        """
        from .pglMessages import pglMessages

        try:
            if showAffineComponents is not None and type(showAffineComponents) is not bool:
                raise ValueError("showAffineComponents must be True, False, or None.")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a nonblank string.")

            # Finish scientific validation before touching an existing figure.
            summary = cls.motionSummary(
                transforms,
                volumeIndices=volumeIndices,
                referenceVolumeIndex=referenceVolumeIndex,
                trSeconds=trSeconds,
                volumeTimesSeconds=volumeTimesSeconds,
                measurementPointRASMM=measurementPointRASMM,
            )

        except Exception as error:
            pglMessages.warning(f"Motion display stopped: {type(error).__name__}: {error}")
            return None

        createdFigure = False
        plt = None

        try:
            import matplotlib.pyplot as plt

            timeValues = np.asarray(summary["time"]["values"])
            translation = np.asarray(summary["translationMM"])
            rotation = np.asarray(summary["rotationVectorDegrees"])
            stretch = np.asarray(summary["stretchPercent"])
            coupling = np.asarray(summary["symmetricCouplingPercent"])

            includeAffine = showAffineComponents
            if includeAffine is None:
                includeAffine = bool(
                    np.max(np.abs(stretch)) > 0.001
                    or np.max(np.abs(coupling)) > 0.001
                )

            panelCount = 4 if includeAffine else 2

            if fig is None:
                fig = plt.figure(figsize=(11, 10 if includeAffine else 7), constrained_layout=True)
                createdFigure = True
            else:
                fig.clear()
                fig.set_layout_engine("constrained")

            axes = fig.subplots(panelCount, 1, sharex=True, squeeze=False)[:, 0]
            colors = ("#0072B2", "#D55E00", "#009E73")
            markers = ("o", "s", "^")

            # Fixed patient directions—not voxel-axis order or head-following axes.
            translationLabels = (
                "Left–right (+ right)",
                "Back–front (+ toward face)",
                "Feet–head (+ toward top of head)",
            )
            rotationLabels = (
                "About left–right axis",
                "About back–front axis",
                "About feet–head axis",
            )
            stretchLabels = (
                "Left–right",
                "Back–front",
                "Feet–head",
            )
            couplingLabels = (
                "Left–right / back–front",
                "Left–right / feet–head",
                "Back–front / feet–head",
            )

            panels = [
                (translation, translationLabels, "Displacement (mm)", "Movement at the fixed reference point"),
                (rotation, rotationLabels, "Rotation (degrees)", "Rotation-vector components in fixed patient axes"),
            ]

            if includeAffine:
                panels.extend([
                    (stretch, stretchLabels, "Stretch change (%)", "Stretch after separating rotation"),
                    (coupling, couplingLabels, "Cross-axis coupling (%)", "Symmetric cross-axis coupling after separating rotation"),
                ])

            # Sorting is for display only. Scientific records retain their
            # original transform/volume ordering.
            order = np.argsort(timeValues, kind="stable")
            plottedTimes = timeValues[order]
            referenceTime = summary["time"]["referenceValue"]

            for axis, (values, labels, ylabel, panelTitle) in zip(axes, panels):
                axis.axhline(0, color="0.55", linewidth=0.8, zorder=0)

                for component, (label, color, marker) in enumerate(zip(labels, colors, markers)):
                    axis.plot(
                        plottedTimes,
                        values[order, component],
                        label=label,
                        color=color,
                        linewidth=1.2,
                        marker=marker,
                        markersize=3,
                    )

                axis.axvline(
                    referenceTime,
                    color="0.25",
                    linestyle="--",
                    linewidth=1,
                    label=f"Reference volume {referenceVolumeIndex}",
                )
                axis.set_ylabel(ylabel)
                axis.set_title(panelTitle, fontsize=10, loc="left")
                axis.grid(True, alpha=0.2)
                axis.legend(loc="best", fontsize=8)

            axes[-1].set_xlabel(summary["time"]["label"])

            if len(plottedTimes) == 1:
                width = 0.5
                if summary["time"]["units"] == "seconds" and trSeconds is not None:
                    width = float(trSeconds) / 2
                axes[-1].set_xlim(float(plottedTimes[0]) - width, float(plottedTimes[0]) + width)

            point = summary["measurementPointRASMM"]
            pointText = ", ".join(f"{value:.2f}" for value in point)

            explanation = (
                f"Reference: original volume {referenceVolumeIndex} (zero-based)\n"
                f"{summary['measurementPointDescription']}: RAS [{pointText}] mm\n"
                "Positive rotation: right-hand rule about patient Right, Anterior, Superior axes.\n"
                "Rotation components describe one axis/angle vector—not successive Euler rotations."
            )
            if includeAffine:
                explanation += (
                    "\nAffine panels describe the fitted mapping, not necessarily head deformation."
                )

            fig.suptitle(f"{title}\n{explanation}", fontsize=10)
            return fig

        except Exception as error:
            # A rendering failure must not invalidate completed estimation.
            # Close only a figure created here, never the caller's figure.
            if createdFigure and plt is not None:
                plt.close(fig)

            pglMessages.warning(
                f"Motion results are available, but plotting failed: "
                f"{type(error).__name__}: {error}",
                level=1,
            )
            return None
        
    ################################################################
    # Validate a recorded within-run motion product
    ################################################################
    @classmethod
    def validateMotionProduct(cls, record):
        """Validate motion metadata without accessing payload files.

        Version 1:
            Native AFNI outputs with unverified coordinate interpretation.

        Version 2:
            Adds interpretation and timing to each scan record.
            Interpretation is produced by interpretAFNIVolreg().
            Timing contains trSeconds and volumeTimesSeconds, either of which
            may be None.

        Returns independent scan records indexed by acquisition UUID.

        Structural/numerical consistency is checked, not provenance
        authenticity or registration quality. Artifact bytes must be verified
        separately when consumed.

        Invalid definitions raise to the calling action or display boundary.
        """
        import re

        from .pglExternalTools import _copyProvenance

        record = _copyProvenance(record)
        if type(record) is not dict or record.get("kind") != "rigidMotion":
            raise ValueError("Expected a rigidMotion product.")

        metadata = record.get("metadata")
        artifacts = record.get("artifacts")
        if type(metadata) is not dict or type(artifacts) is not dict:
            raise ValueError("Motion product requires metadata and artifact references.")

        expectedFields = {"schemaVersion", "operation", "backend", "matrixConvention", "scans"}
        if set(metadata) != expectedFields:
            raise ValueError("Invalid motion metadata fields.")

        version = metadata["schemaVersion"]
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported motion metadata version.")
        if metadata["operation"] != "withinRunMotion" or metadata["backend"] != "AFNI":
            raise ValueError("Expected AFNI within-run motion metadata.")

        expectedConvention = {
            "format": "AFNI aff12.1D",
            "producerOption": "3dvolreg -1Dmatrix_save",
            "mappingStatus": "unverified" if version == 1 else "interpreted",
        }
        if metadata["matrixConvention"] != expectedConvention:
            raise ValueError("Motion matrix convention does not match its metadata version.")

        scans = metadata["scans"]
        if type(scans) is not list or not scans:
            raise ValueError("Motion metadata must contain scan records.")

        expectedScanFields = {
            "scanIndex", "scanUUID", "sourceStageID", "sourceIdentity",
            "volumeIndices", "referenceVolumeIndex",
            "matricesRole", "motionParametersRole",
        }
        if version == 2: expectedScanFields |= {"interpretation", "timing"}

        byScan = {}
        scanIndices = set()
        usedRoles = set()

        def requireText(value, label):
            if (
                not isinstance(value, str) or not value.strip()
                or any(character in value for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError(f"{label} must be a nonblank string without NUL or newlines.")

        def validateIdentity(identity):
            if type(identity) is not dict or set(identity) != {"sha256", "size_bytes"}:
                raise ValueError("Identity requires sha256 and size_bytes.")
            if not isinstance(identity["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None:
                raise ValueError("Invalid SHA-256.")
            if type(identity["size_bytes"]) is not int or identity["size_bytes"] <= 0:
                raise ValueError("Byte count must be a positive integer.")

        for scan in scans:
            if type(scan) is not dict or set(scan) != expectedScanFields:
                raise ValueError("Invalid motion scan record fields.")

            index = scan["scanIndex"]
            if type(index) is not int or index < 0 or index in scanIndices:
                raise ValueError("Motion scan indices must be unique nonnegative integers.")
            scanIndices.add(index)

            scanUUID = scan["scanUUID"]
            requireText(scanUUID, "scanUUID")
            if scanUUID in byScan: raise ValueError("Motion product contains a repeated scan UUID.")

            requireText(scan["sourceStageID"], "sourceStageID")
            validateIdentity(scan["sourceIdentity"])

            indices = scan["volumeIndices"]
            if (
                type(indices) is not list or not indices
                or any(type(value) is not int or value < 0 for value in indices)
                or len(set(indices)) != len(indices)
            ):
                raise ValueError("volumeIndices must contain unique nonnegative original-volume indices.")

            reference = scan["referenceVolumeIndex"]
            if type(reference) is not int or reference not in indices:
                raise ValueError("referenceVolumeIndex must identify a represented original volume.")

            for field in ("matricesRole", "motionParametersRole"):
                role = scan[field]
                requireText(role, field)
                if role in usedRoles: raise ValueError("Motion artifact roles must be distinct.")
                usedRoles.add(role)

                artifact = artifacts.get(role)
                if type(artifact) is not dict or set(artifact) != {"attachmentName", "sha256", "size_bytes"}:
                    raise ValueError(f"Missing or invalid artifact role {role!r}.")
                requireText(artifact["attachmentName"], "attachmentName")
                validateIdentity({key: artifact[key] for key in ("sha256", "size_bytes")})

            if version == 2:
                interpreted = scan["interpretation"]
                timing = scan["timing"]

                if type(timing) is not dict or set(timing) != {"trSeconds", "volumeTimesSeconds"}:
                    raise ValueError("Motion timing requires trSeconds and volumeTimesSeconds.")

                if (
                    type(interpreted) is not dict
                    or type(interpreted.get("schemaVersion")) is not int
                    or interpreted["schemaVersion"] != 1
                    or interpreted.get("kind") != "AFNIWithinRunMotionInterpretation"
                    or interpreted.get("adapter") != "pgl.AFNI.3dvolreg.sharedGrid.v1"
                ):
                    raise ValueError("Unsupported AFNI motion interpretation.")

                requireText(interpreted.get("afniVersion"), "AFNI version")
                if interpreted.get("geometryID") != scan["sourceStageID"]:
                    raise ValueError("Interpreted geometry must identify the motion source stage.")

                nativeFrame = cls._validateFrame(interpreted["nativeFrame"])
                voxelFrame = cls._validateFrame(interpreted["voxelFrame"])
                if nativeFrame["representation"] != "toolNominal":
                    raise ValueError("Native AFNI frame must use toolNominal coordinates.")
                if voxelFrame["representation"] != "voxelIndices":
                    raise ValueError("Converted frame must use voxel indices.")
                if nativeFrame["geometryID"] != scan["sourceStageID"]:
                    raise ValueError("Native frame does not identify the motion source stage.")

                native = interpreted.get("nativeTransforms")
                converted = interpreted.get("voxelTransforms")
                associations = interpreted.get("volumeAssociations")
                if any(type(values) is not list or len(values) != len(indices) for values in (native, converted, associations)):
                    raise ValueError("Interpretation must contain one transform and association per volume.")

                transformIDs = []

                for rowIndex, originalIndex in enumerate(indices):
                    nativeRecord = cls._validateTransform(native[rowIndex])
                    voxelRecord = cls._validateTransform(converted[rowIndex])

                    if nativeRecord["domainFrame"] != nativeFrame or nativeRecord["codomainFrame"] != nativeFrame:
                        raise ValueError("Native motion transform has inconsistent endpoint frames.")
                    if voxelRecord["domainFrame"] != voxelFrame or voxelRecord["codomainFrame"] != voxelFrame:
                        raise ValueError("Converted motion transform has inconsistent endpoint frames.")

                    if (
                        nativeRecord["purpose"] != "sampling"
                        or nativeRecord["interpretationStatus"] != "validated"
                        or nativeRecord["adapter"] != interpreted["adapter"]
                    ):
                        raise ValueError("Native motion transform lacks the supported interpreted sampling convention.")

                    linear = np.asarray(nativeRecord["matrix"])[:3, :3]
                    if (
                        not np.allclose(linear.T @ linear, np.eye(3), rtol=0, atol=0.0001)
                        or abs(np.linalg.det(linear) - 1) > 0.0001
                    ):
                        raise ValueError("Native rigid-motion transform contains scale, shear, or reflection.")

                    expected = cls.convert(
                        nativeRecord,
                        domainFrame=voxelFrame,
                        codomainFrame=voxelFrame,
                    )
                    if (
                        voxelRecord["purpose"] != expected["purpose"]
                        or voxelRecord["interpretationStatus"] != expected["interpretationStatus"]
                        or voxelRecord["adapter"] != expected["adapter"]
                        or voxelRecord["inputTransformIDs"] != expected["inputTransformIDs"]
                        or not np.allclose(voxelRecord["matrix"], expected["matrix"], rtol=0, atol=1e-8)
                    ):
                        raise ValueError("Stored voxel transform disagrees with its native-coordinate conversion.")

                    expectedAssociation = {
                        "rowIndex": rowIndex,
                        "originalVolumeIndex": originalIndex,
                        "referenceVolumeIndex": reference,
                        "nativeTransformID": nativeRecord["transformID"],
                        "voxelTransformID": voxelRecord["transformID"],
                    }
                    if associations[rowIndex] != expectedAssociation:
                        raise ValueError("Transform/volume association is inconsistent.")

                    transformIDs.extend([nativeRecord["transformID"], voxelRecord["transformID"]])

                if len(set(transformIDs)) != len(transformIDs):
                    raise ValueError("Motion transform identities must be unique within a scan record.")

                # Reuse the tested checks for time, physical frames, and
                # displayable sampling semantics. No image payload is read.
                cls.motionSummary(
                    converted,
                    volumeIndices=indices,
                    referenceVolumeIndex=reference,
                    trSeconds=timing["trSeconds"],
                    volumeTimesSeconds=timing["volumeTimesSeconds"],
                )

            byScan[scanUUID] = scan

        return byScan

    ################################################################
    # Display a bound, interpreted motion product
    ################################################################
    @classmethod
    def displayMotionProduct(cls, product, *, scanIndices=None, showAffineComponents=None, measurementPointRASMM=None, cacheDirectory=None):
        """Return one motion figure per selected scan from a bound product.

        scanIndices:
            Optional zero-based indices in the owner's CURRENT MRI collection.
            None displays all scans represented by the product.

        Verifies exact product ownership and matrix/parameter artifact bytes.
        Native matrix rows must agree with the interpreted transforms.

        Returns a list of figures. Rendering failures may omit a figure after
        a recoverable warning. Dependency/metadata/integrity failures raise
        before any figures are created.

        Does not estimate, apply, save, or automatically show images.
        Current scan indices are resolved by UUID, not historical position.
        """
        from pathlib import Path

        from .pglExternalTools import pglExternalTool

        owner = product._requireOwner()
        product._findProducingAction(owner)
        record = product.record
        byScan = cls.validateMotionProduct(record)

        if record["metadata"]["schemaVersion"] != 2:
            raise ValueError(
                "This motion product contains native matrices without interpreted "
                "coordinate frames. Patient-coordinate motion cannot be plotted yet."
            )
        if getattr(owner, "mri", None) is None:
            raise ValueError("The owning session has no MRI collection.")

        scans = list(owner.mri.scans)
        positions = {}

        for index, scan in enumerate(scans):
            positions.setdefault(scan.uuid, []).append(index)

        if scanIndices is None:
            selected = []
            for scanUUID in byScan:
                matches = positions.get(scanUUID, [])
                if len(matches) != 1:
                    raise ValueError("A motion scan is absent or ambiguous in the current MRI collection.")
                selected.append(matches[0])
        else:
            if not isinstance(scanIndices, (list, tuple)) or not scanIndices:
                raise ValueError("scanIndices must be a nonempty list/tuple or None.")
            selected = list(scanIndices)
            if any(type(index) is not int or not 0 <= index < len(scans) for index in selected):
                raise ValueError("Invalid zero-based session.mri.scans index.")
            if len(set(selected)) != len(selected):
                raise ValueError("scanIndices must not contain duplicates.")

        def artifactPath(role):
            """Resolve exact bytes through the existing attachment interface."""
            artifact = record["artifacts"][role]
            attachment = owner.derivedFiles.get(artifact["attachmentName"])
            if attachment is None:
                raise ValueError(f"Motion artifact {role!r} is unavailable or was discarded.")

            expected = {key: artifact[key] for key in ("sha256", "size_bytes")}
            if any(attachment.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Attachment identity differs from motion artifact {role!r}.")

            path = owner.getDerivedFile(artifact["attachmentName"], cacheDirectory=cacheDirectory)
            if path is None: raise ValueError(f"Motion artifact {role!r} could not be retrieved.")
            path = Path(path)

            if pglExternalTool._fileIdentity(path) != expected:
                raise ValueError(f"Motion artifact {role!r} failed byte verification.")
            return path

        prepared = []

        # Validate all requested data before creating any figures.
        for index in selected:
            scan = scans[index]
            if len(positions[scan.uuid]) != 1:
                raise ValueError("The selected acquisition UUID appears more than once.")
            if scan.uuid not in byScan:
                raise ValueError(f"Motion product does not cover MRI scan {index}.")

            entry = byScan[scan.uuid]
            count = len(entry["volumeIndices"])
            matrices = np.loadtxt(artifactPath(entry["matricesRole"]), comments="#", ndmin=2)
            parameters = np.loadtxt(artifactPath(entry["motionParametersRole"]), comments="#", ndmin=2)

            if matrices.shape != (count, 12) or not np.isfinite(matrices).all():
                raise ValueError("Motion matrix file has invalid dimensions or values.")
            if parameters.shape != (count, 6) or not np.isfinite(parameters).all():
                raise ValueError("Motion parameter file has invalid dimensions or values.")

            interpreted = entry["interpretation"]
            storedRows = np.array([
                np.asarray(transform["matrix"])[:3].reshape(-1)
                for transform in interpreted["nativeTransforms"]
            ])
            if not np.allclose(matrices, storedRows, rtol=0, atol=1e-10):
                raise ValueError("Native matrix file disagrees with the product's interpreted matrices.")

            timing = entry["timing"]
            cls.motionSummary(
                interpreted["voxelTransforms"],
                volumeIndices=entry["volumeIndices"],
                referenceVolumeIndex=entry["referenceVolumeIndex"],
                trSeconds=timing["trSeconds"],
                volumeTimesSeconds=timing["volumeTimesSeconds"],
                measurementPointRASMM=measurementPointRASMM,
            )
            prepared.append((index, scan.name, entry))

        figures = []
        for index, name, entry in prepared:
            timing = entry["timing"]
            figure = cls.displayMotion(
                entry["interpretation"]["voxelTransforms"],
                volumeIndices=entry["volumeIndices"],
                referenceVolumeIndex=entry["referenceVolumeIndex"],
                trSeconds=timing["trSeconds"],
                volumeTimesSeconds=timing["volumeTimesSeconds"],
                measurementPointRASMM=measurementPointRASMM,
                showAffineComponents=showAffineComponents,
                title=f"{product.name} — MRI scan {index}: {name}",
            )
            if figure is not None: figures.append(figure)

        return figures
    
    ################################################################
    # Describe fixed run-to-reference alignments
    ################################################################
    @classmethod
    def runAlignmentSummary(cls, transforms, *, scanIndices, referenceScanIndex, scanNames=None):
        """Calculate patient-oriented movement for one shared-grid run group.

        transforms:
            One validated common-reference -> run-reference sampling map per
            selected scan. The reference scan must also have a transform,
            normally identity.

        scanIndices:
            Zero-based session.mri.scans indices, in the requested display order.
            These are scan indices, not volume indices or timestamps.

        referenceScanIndex:
            Explicit common reference scan in scanIndices.

        scanNames:
            Optional display labels corresponding to scanIndices.

        Initial scope:
            Rigid run-to-reference mappings on matching acquisition grids.
            All endpoints must use the same declared physical reference space.
            Different grid geometries require a separately supported adapter.

        Measures displacement at the common reference image's center.
        Rotations are principal rotation-vector components in patient RAS axes.

        Returns portable summary metadata. Does not read images, display
        figures, resample data, or modify transforms.

        Errors propagate to the action/display validation boundary.
        """
        from scipy.spatial.transform import Rotation

        if not isinstance(transforms, (list, tuple)) or not transforms:
            raise ValueError("Supply one interpreted alignment transform per scan.")

        if (
            type(scanIndices) is not list
            or len(scanIndices) != len(transforms)
            or any(type(index) is not int or index < 0 for index in scanIndices)
            or len(set(scanIndices)) != len(scanIndices)
        ):
            raise ValueError("scanIndices must contain one unique nonnegative MRI index per transform.")

        if type(referenceScanIndex) is not int or referenceScanIndex not in scanIndices:
            raise ValueError("referenceScanIndex must identify one of the selected scans.")

        if scanNames is None:
            scanNames = [f"MRI scan {index}" for index in scanIndices]
        elif (
            not isinstance(scanNames, (list, tuple))
            or len(scanNames) != len(scanIndices)
            or any(not isinstance(name, str) for name in scanNames)
        ):
            raise ValueError("scanNames must contain one string per selected scan.")

        records = [cls._validateTransform(transform) for transform in transforms]
        referenceFrame = records[0]["domainFrame"]
        referencePhysical = np.asarray(referenceFrame["voxelToPhysicalMM"])
        centerVoxel = (np.asarray(referenceFrame["spatialShape"], dtype=float) - 1) / 2
        center = (referencePhysical @ np.append(centerVoxel, 1.0))[:3]

        def requireSharedGrid(frame):
            """Restrict this display contract to the tested shared-grid case."""
            if frame["physicalSpaceID"] != referenceFrame["physicalSpaceID"]:
                raise ValueError("Alignment endpoints do not share a declared patient reference space.")
            if frame["spatialShape"] != referenceFrame["spatialShape"]:
                raise ValueError("This run-alignment display supports matching acquisition grids only.")
            if not np.allclose(
                frame["voxelToPhysicalMM"],
                referencePhysical,
                rtol=0,
                atol=0.0001,
            ):
                raise ValueError("Alignment endpoint geometry differs from the common acquisition grid.")

        def physicalFrame(frame):
            return cls.makeFrame(
                geometryID=frame["geometryID"],
                physicalSpaceID=frame["physicalSpaceID"],
                spatialShape=frame["spatialShape"],
                voxelToPhysicalMM=frame["voxelToPhysicalMM"],
                representation="scannerRAS",
            )

        translations = []
        rotations = []

        for record in records:
            if record["purpose"] != "sampling":
                raise ValueError("Run alignment requires reference-to-input sampling mappings.")
            if record["interpretationStatus"] != "validated":
                raise ValueError("Run alignment requires validated coordinate interpretation.")

            domain = record["domainFrame"]
            codomain = record["codomainFrame"]

            if domain["geometryID"] != referenceFrame["geometryID"]:
                raise ValueError("All run alignments must start from the same reference image geometry.")

            requireSharedGrid(domain)
            requireSharedGrid(codomain)

            physical = cls.convert(
                record,
                domainFrame=physicalFrame(domain),
                codomainFrame=physicalFrame(codomain),
            )
            matrix = np.asarray(physical["matrix"])
            linear = matrix[:3, :3]

            # Do not silently turn an affine deformation into a head-rotation
            # plot. This first run-alignment contract is explicitly rigid.
            if (
                not np.allclose(linear.T @ linear, np.eye(3), rtol=0, atol=0.0001)
                or abs(np.linalg.det(linear) - 1.0) > 0.0001
            ):
                raise ValueError("Run alignment contains scale, shear, or reflection instead of rigid motion.")

            # Remove only numerical nonorthogonality when extracting angles.
            left, _, rightTranspose = np.linalg.svd(linear)
            rotation = left @ rightTranspose

            translations.append((linear @ center + matrix[:3, 3] - center).tolist())
            rotations.append(Rotation.from_matrix(rotation).as_rotvec(degrees=True).tolist())

        referencePosition = scanIndices.index(referenceScanIndex)

        # The reference-to-itself mapping should be identity. Report a bad
        # association instead of visually forcing its plotted values to zero.
        if (
            np.max(np.abs(translations[referencePosition])) > 0.001
            or np.max(np.abs(rotations[referencePosition])) > 0.001
        ):
            raise ValueError("The selected reference scan does not have an approximately identity alignment.")

        return {
            "schemaVersion": 1,
            "kind": "runAlignmentSummary",
            "scanIndices": list(scanIndices),
            "scanNames": list(scanNames),
            "referenceScanIndex": referenceScanIndex,
            "referencePosition": referencePosition,
            "transformIDs": [record["transformID"] for record in records],
            "coordinateConvention": "patientRAS",
            "measurementPointRASMM": center.tolist(),
            "translationMM": translations,
            "rotationVectorDegrees": rotations,
            "axisLabels": [
                "Left–right (+ right)",
                "Back–front (+ toward face)",
                "Feet–head (+ toward top of head)",
            ],
            "description": (
                "One fixed alignment per run, relative to the selected common "
                "reference run. These are between-run estimates, not the "
                "per-volume movement within each run.\n\n"
                "Translation measures displacement at the common reference "
                "image's center. Rotation-vector components describe rotation "
                "about fixed patient Right, Anterior, and Superior axes. "
                "Positive rotation follows the right-hand rule. The components "
                "are not three successive Euler rotations.\n\n"
                "These plots show what registration estimated. Anatomical "
                "overlays are still needed to evaluate whether it aligned "
                "the images correctly."
            ),
        }

    ################################################################
    # Display fixed run-to-reference alignments
    ################################################################
    @classmethod
    def displayRunAlignment(cls, transforms, *, scanIndices, referenceScanIndex, scanNames=None, groupLabel=None, applicationMessage=None, fig=None):
        """Plot one geometry group's between-run translations and rotations.

        Returns a Matplotlib figure, or None after a warning.
        Does not call plt.show(), save figures, or access image payloads.

        applicationMessage:
            Optional text derived by the caller from recorded application
            provenance. When absent, the figure makes no application claim.

            Do not infer "applied" from the mere existence of transform files.

        Numeric/coordinate validation occurs before clearing a supplied figure.
        Validation failures stop with the default warning level.
        Recoverable rendering failures warn with level=1.

        Preserves selected run order. Connecting lines are visual guides,
        not claims about continuous acquisition timing.
        """
        from .pglMessages import pglMessages

        try:
            for name, value in (("groupLabel", groupLabel), ("applicationMessage", applicationMessage)):
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{name} must be a nonblank string or None.")

            summary = cls.runAlignmentSummary(
                transforms,
                scanIndices=scanIndices,
                referenceScanIndex=referenceScanIndex,
                scanNames=scanNames,
            )

        except Exception as error:
            pglMessages.warning(f"Run-alignment display stopped: {type(error).__name__}: {error}")
            return None

        createdFigure = False
        plt = None

        try:
            import matplotlib.pyplot as plt

            if fig is None:
                fig = plt.figure(figsize=(11, 8), constrained_layout=True)
                createdFigure = True
            else:
                fig.clear()
                fig.set_layout_engine("constrained")

            axes = fig.subplots(2, 1, sharex=True)
            positions = np.arange(len(summary["scanIndices"]))
            referencePosition = summary["referencePosition"]
            colors = ("#0072B2", "#D55E00", "#009E73")
            markers = ("o", "s", "^")

            panels = (
                (
                    summary["translationMM"],
                    summary["axisLabels"],
                    "Displacement (mm)",
                    "Movement at the common reference image center",
                ),
                (
                    summary["rotationVectorDegrees"],
                    [
                        "About left–right axis",
                        "About back–front axis",
                        "About feet–head axis",
                    ],
                    "Rotation (degrees)",
                    "Rotation-vector components in fixed patient axes",
                ),
            )

            for axis, (values, labels, ylabel, heading) in zip(axes, panels):
                values = np.asarray(values)
                axis.axhline(0, color="0.55", linewidth=0.8, zorder=0)
                axis.axvspan(
                    referencePosition - 0.2,
                    referencePosition + 0.2,
                    color="0.5",
                    alpha=0.15,
                    label=f"Reference scan {referenceScanIndex}",
                )

                for component, (label, color, marker) in enumerate(zip(labels, colors, markers)):
                    axis.plot(
                        positions,
                        values[:, component],
                        color=color,
                        marker=marker,
                        linestyle="--",
                        linewidth=1,
                        markersize=5,
                        label=label,
                    )

                axis.set_ylabel(ylabel)
                axis.set_title(heading, fontsize=10, loc="left")
                axis.grid(True, alpha=0.2)
                axis.legend(loc="best", fontsize=8)

            labels = [
                f"{index}: {name}" + (" [reference]" if index == referenceScanIndex else "")
                for index, name in zip(summary["scanIndices"], summary["scanNames"])
            ]
            axes[-1].set_xticks(positions)
            axes[-1].set_xticklabels(labels, rotation=25, ha="right")
            axes[-1].set_xlabel("MRI scans in selected order (zero-based session.mri.scans indices)")
            axes[-1].set_xlim(-0.5, max(0.5, len(positions) - 0.5))

            heading = "Between-run alignment"
            if groupLabel is not None: heading += f" — {groupLabel}"
            pointText = ", ".join(f"{value:.2f}" for value in summary["measurementPointRASMM"])

            statusText = (
                applicationMessage
                if applicationMessage is not None
                else "Estimates shown; image-application status is not specified."
            )
            fig.suptitle(
                f"{heading}\n"
                f"Common reference: MRI scan {referenceScanIndex}\n"
                f"Measurement point: reference image center, RAS [{pointText}] mm\n"
                f"{statusText}",
                fontsize=10,
            )
            fig.supxlabel(
                "Positive rotations follow the right-hand rule. Lines connect selected runs, not continuous time.\n"
                "Estimated movement is not proof of anatomical alignment; inspect image overlays too.",
                fontsize=8,
            )
            return fig

        except Exception as error:
            if createdFigure and plt is not None: plt.close(fig)
            pglMessages.warning(
                f"Run-alignment results are available, but plotting failed: "
                f"{type(error).__name__}: {error}",
                level=1,
            )
            return None
        
    ################################################################
    # Validate a recorded run-alignment product
    ################################################################
    @classmethod
    def validateRunAlignmentProduct(cls, record):
        """Validate the shared-grid run-alignment product without reading files.

        Checks group membership, reference associations, endpoint geometry,
        native-to-voxel conversion, and native artifact references.

        Returns an independent product metadata copy.

        Cached summaries are not authoritative; summaries are recomputed from
        the recorded transforms when displayed.

        Does not establish payload availability or registration quality.
        Composition must additionally match the exact within-run motion
        product and verify its reference/source associations.
        """
        import re

        from .pglExternalTools import _copyProvenance

        record = _copyProvenance(record)
        if type(record) is not dict or record.get("kind") != "runAlignment":
            raise ValueError("Expected a runAlignment product.")

        metadata = record.get("metadata")
        artifacts = record.get("artifacts")
        if type(metadata) is not dict or type(artifacts) is not dict:
            raise ValueError("Run alignment requires metadata and artifact records.")
        if type(metadata.get("schemaVersion")) is not int or metadata["schemaVersion"] != 1:
            raise ValueError("Unsupported run-alignment metadata version.")
        if metadata.get("operation") != "runReferenceAlignment" or metadata.get("backend") != "AFNI":
            raise ValueError("Unsupported run-alignment operation/backend.")

        motionID = metadata.get("withinRunMotionProductID")
        producerID = metadata.get("withinRunMotionProducerActionID")
        if not isinstance(motionID, str) or not motionID:
            raise ValueError("Run alignment must identify its input motion product.")
        if not isinstance(producerID, str) or not producerID:
            raise ValueError("Run alignment must identify its input motion execution.")
        if record.get("inputProductIDs") != [motionID]:
            raise ValueError("Run-alignment input links disagree with its motion-product association.")

        plan = metadata.get("plan")
        entries = metadata.get("scans")
        if type(plan) is not dict or type(entries) is not list or not entries:
            raise ValueError("Run alignment requires a resolved plan and scan records.")
        if plan.get("requiresExplicitOutputGrid") is not False:
            raise ValueError("This run-alignment contract supports independent matching-grid groups only.")

        plannedMotion = plan.get("withinRunMotion", {})
        if (
            plannedMotion.get("productID") != motionID
            or plannedMotion.get("producerActionID") != producerID
        ):
            raise ValueError("The alignment plan names a different input motion product.")

        groups = plan.get("alignmentGroups")
        if type(groups) is not list or not groups:
            raise ValueError("Run alignment has no reference groups.")

        def requireText(value, label):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a nonblank string.")

        def validateIdentity(identity):
            if (
                type(identity) is not dict
                or not isinstance(identity.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
                or type(identity.get("size_bytes")) is not int
                or identity["size_bytes"] <= 0
            ):
                raise ValueError("Invalid recorded file identity.")

        byIndex = {}
        scanUUIDs = set()

        for entry in entries:
            if type(entry) is not dict:
                raise ValueError("Invalid run-alignment scan record.")

            index = entry.get("scanIndex")
            if type(index) is not int or index < 0 or index in byIndex:
                raise ValueError("Recorded scan indices must be unique nonnegative integers.")

            scanUUID = entry.get("scanUUID")
            requireText(scanUUID, "scanUUID")
            if scanUUID in scanUUIDs:
                raise ValueError("Run alignment contains repeated acquisition UUIDs.")
            scanUUIDs.add(scanUUID)

            requireText(entry.get("sourceStageID"), "sourceStageID")
            validateIdentity(entry.get("sourceIdentity"))

            volumes = entry.get("volumeIndices")
            referenceVolume = entry.get("referenceVolumeIndex")
            referenceRow = entry.get("referenceRow")
            if (
                type(volumes) is not list or not volumes
                or any(type(value) is not int or value < 0 for value in volumes)
                or len(set(volumes)) != len(volumes)
                or type(referenceVolume) is not int or referenceVolume not in volumes
                or type(referenceRow) is not int
                or referenceRow != volumes.index(referenceVolume)
            ):
                raise ValueError("Invalid run-reference volume correspondence.")

            if type(entry.get("isReference")) is not bool:
                raise ValueError("isReference must be Boolean.")

            byIndex[index] = entry

        usedScans = set()
        usedGroups = set()
        usedRoles = set()

        for group in groups:
            if type(group) is not dict:
                raise ValueError("Invalid alignment group.")

            groupIndex = group.get("groupIndex")
            members = group.get("scanIndices")
            referenceIndex = group.get("referenceScanIndex")

            if type(groupIndex) is not int or groupIndex < 0 or groupIndex in usedGroups:
                raise ValueError("Alignment group indices must be unique nonnegative integers.")
            usedGroups.add(groupIndex)

            if (
                type(members) is not list or not members
                or any(type(index) is not int for index in members)
                or len(set(members)) != len(members)
                or any(index not in byIndex for index in members)
                or usedScans.intersection(members)
                or type(referenceIndex) is not int or referenceIndex not in members
            ):
                raise ValueError("Alignment groups have missing, repeated, or invalid scan references.")

            usedScans.update(members)
            reference = byIndex[referenceIndex]
            if group.get("referenceScanUUID") != reference["scanUUID"]:
                raise ValueError("Group reference UUID disagrees with its scan record.")

            voxelTransforms = []

            for index in members:
                entry = byIndex[index]
                isReference = index == referenceIndex

                if (
                    entry.get("groupIndex") != groupIndex
                    or entry["isReference"] != isReference
                    or entry.get("commonReferenceScanIndex") != referenceIndex
                    or entry.get("commonReferenceScanUUID") != reference["scanUUID"]
                    or entry.get("commonReferenceSourceStageID") != reference["sourceStageID"]
                    or entry.get("commonReferenceVolumeIndex") != reference["referenceVolumeIndex"]
                ):
                    raise ValueError("Run alignment has inconsistent common-reference associations.")

                voxel = cls._validateTransform(entry["voxelTransform"])
                if (
                    voxel["purpose"] != "sampling"
                    or voxel["interpretationStatus"] != "validated"
                    or voxel["domainFrame"]["representation"] != "voxelIndices"
                    or voxel["codomainFrame"]["representation"] != "voxelIndices"
                    or voxel["domainFrame"]["geometryID"] != reference["sourceStageID"]
                    or voxel["codomainFrame"]["geometryID"] != entry["sourceStageID"]
                ):
                    raise ValueError("Run alignment has inconsistent voxel sampling frames.")

                if isReference:
                    if (
                        entry.get("matricesRole") is not None
                        or entry.get("motionParametersRole") is not None
                        or entry.get("interpretation") is not None
                        or not np.allclose(voxel["matrix"], np.eye(4), rtol=0, atol=1e-10)
                    ):
                        raise ValueError("The reference scan must have an explicit identity without fabricated native files.")

                else:
                    interpretation = entry.get("interpretation")
                    if (
                        type(interpretation) is not dict
                        or interpretation.get("kind") != "AFNIRunReferenceAlignment"
                        or interpretation.get("adapter") != "pgl.AFNI.3dvolreg.crossRunSharedGrid.v1"
                        or interpretation.get("inputStageID") != entry["sourceStageID"]
                        or interpretation.get("referenceStageID") != reference["sourceStageID"]
                        or interpretation.get("inputVolume") != entry["referenceRow"]
                        or interpretation.get("referenceVolume") != reference["referenceRow"]
                    ):
                        raise ValueError("Unsupported or inconsistent AFNI alignment interpretation.")

                    if interpretation.get("voxelTransform") != entry["voxelTransform"]:
                        raise ValueError("Duplicate voxel-transform records disagree.")

                    native = cls._validateTransform(interpretation["nativeTransform"])
                    if (
                        native["purpose"] != "sampling"
                        or native["interpretationStatus"] != "validated"
                        or native["domainFrame"]["representation"] != "toolNominal"
                        or native["codomainFrame"]["representation"] != "toolNominal"
                    ):
                        raise ValueError("Invalid native AFNI alignment convention.")

                    expected = cls.convert(
                        native,
                        domainFrame=voxel["domainFrame"],
                        codomainFrame=voxel["codomainFrame"],
                    )
                    if (
                        voxel["inputTransformIDs"] != expected["inputTransformIDs"]
                        or not np.allclose(voxel["matrix"], expected["matrix"], rtol=0, atol=1e-8)
                    ):
                        raise ValueError("Voxel alignment disagrees with the native matrix conversion.")

                    for field in ("matricesRole", "motionParametersRole"):
                        role = entry.get(field)
                        requireText(role, field)
                        if role in usedRoles:
                            raise ValueError("Run-alignment artifact roles must be distinct.")
                        usedRoles.add(role)

                        artifact = artifacts.get(role)
                        if type(artifact) is not dict:
                            raise ValueError(f"Missing alignment artifact {role!r}.")
                        requireText(artifact.get("attachmentName"), "attachmentName")
                        validateIdentity(artifact)

                voxelTransforms.append(voxel)

            # Recompute patient-coordinate values and validate rigidity/common
            # geometry. Never rely on a cached plotting summary.
            cls.runAlignmentSummary(
                voxelTransforms,
                scanIndices=members,
                referenceScanIndex=referenceIndex,
            )

        if usedScans != set(byIndex):
            raise ValueError("Some alignment scan records are not assigned to a group.")

        return metadata
    
    ################################################################
    # Display a recorded run-alignment product
    ################################################################
    @classmethod
    def displayRunAlignmentProduct(cls, product, *, scanIndices=None, cacheDirectory=None):
        """Return one alignment-summary figure per selected geometry group.

        Uses the product's bound session and resolves current scan positions
        by UUID, never by an obsolete recorded collection index.

        scanIndices:
            Optional current zero-based MRI indices selecting groups to show.
            Each selected group is displayed in full, including its reference.

        Verifies native matrix/parameter files before plotting.
        Identity-reference records require no fabricated file.

        Does not apply transforms, read voxel arrays, save figures, or call
        plt.show(). Does not infer application status from file existence.

        Invalid products/dependencies raise to product.display(), which warns.
        Recoverable plotting failures are handled by displayRunAlignment().
        """
        from pathlib import Path

        from .pglExternalTools import pglExternalTool

        owner = product._requireOwner()
        product._findProducingAction(owner)
        record = product.record
        metadata = cls.validateRunAlignmentProduct(record)

        if getattr(owner, "mri", None) is None:
            raise ValueError("The owning session has no MRI collection.")

        scans = list(owner.mri.scans)
        positions = {}
        for index, scan in enumerate(scans):
            positions.setdefault(scan.uuid, []).append(index)

        entries = {entry["scanIndex"]: entry for entry in metadata["scans"]}
        groups = metadata["plan"]["alignmentGroups"]

        if scanIndices is not None:
            if (
                not isinstance(scanIndices, (list, tuple)) or not scanIndices
                or any(type(index) is not int or not 0 <= index < len(scans) for index in scanIndices)
                or len(set(scanIndices)) != len(scanIndices)
            ):
                raise ValueError("scanIndices must be unique current MRI indices.")

            selectedUUIDs = {scans[index].uuid for index in scanIndices}
            availableUUIDs = {entry["scanUUID"] for entry in entries.values()}
            if selectedUUIDs - availableUUIDs:
                raise ValueError("The alignment product does not cover every selected scan.")

            groups = [
                group for group in groups
                if any(entries[index]["scanUUID"] in selectedUUIDs for index in group["scanIndices"])
            ]

        def artifactValues(role, columns):
            artifact = record["artifacts"][role]
            attachment = owner.derivedFiles.get(artifact["attachmentName"])
            expected = {key: artifact[key] for key in ("sha256", "size_bytes")}

            if attachment is None or any(attachment.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Alignment artifact {role!r} is unavailable or its identity changed.")

            path = owner.getDerivedFile(artifact["attachmentName"], cacheDirectory=cacheDirectory)
            if path is None: raise ValueError(f"Alignment artifact {role!r} could not be retrieved.")
            path = Path(path)

            if pglExternalTool._fileIdentity(path) != expected:
                raise ValueError(f"Alignment artifact {role!r} failed byte verification.")

            values = np.loadtxt(path, comments="#", ndmin=2)
            if values.shape != (1, columns) or not np.isfinite(values).all():
                raise ValueError(f"Invalid alignment artifact {role!r}.")
            return values

        prepared = []

        # Resolve and validate all selected groups before creating figures.
        for group in groups:
            groupEntries = [entries[index] for index in group["scanIndices"]]
            currentIndices = []

            for entry in groupEntries:
                matches = positions.get(entry["scanUUID"], [])
                if len(matches) != 1:
                    raise ValueError("An alignment scan is absent or ambiguous in the current session.")
                currentIndices.append(matches[0])

                if not entry["isReference"]:
                    rows = artifactValues(entry["matricesRole"], 12)
                    artifactValues(entry["motionParametersRole"], 6)
                    native = np.asarray(entry["interpretation"]["nativeTransform"]["matrix"])
                    if not np.allclose(rows.reshape(3, 4), native[:3], rtol=0, atol=1e-10):
                        raise ValueError("Native alignment file disagrees with the interpreted matrix.")

            referenceUUID = entries[group["referenceScanIndex"]]["scanUUID"]
            referenceIndex = positions[referenceUUID][0]
            transforms = [entry["voxelTransform"] for entry in groupEntries]
            names = [scans[index].name for index in currentIndices]

            cls.runAlignmentSummary(
                transforms,
                scanIndices=currentIndices,
                referenceScanIndex=referenceIndex,
                scanNames=names,
            )
            prepared.append((group, transforms, currentIndices, referenceIndex, names))

        figures = []
        for group, transforms, indices, referenceIndex, names in prepared:
            figure = cls.displayRunAlignment(
                transforms,
                scanIndices=indices,
                referenceScanIndex=referenceIndex,
                scanNames=names,
                groupLabel=f"Geometry group {group['groupIndex']}",
                applicationMessage=(
                    "Alignment estimates shown. Estimation did not modify images; "
                    "later application status is not yet resolved by this display."
                ),
            )
            if figure is not None: figures.append(figure)

        return figures
    
    ################################################################
    # Compose recorded within-run motion and between-run alignment
    ################################################################
    @classmethod
    def composeRunMotion(cls, motionProduct, alignmentProduct):
        """Prepare composite sampling maps from explicitly matched products.

        Both arguments are product-record dictionaries.

        Verifies:
            Exact input motion product and producing execution.
            Scan UUID, source stage, content identity, and volume correspondence.
            Recorded within-run and common-reference volumes.
            Compatible endpoint geometry and sampling conventions.

        Initial support:
            Within-run rigid motion plus fixed shared-grid run alignment.
            Each output retains its own acquisition grid.
            Grids may differ only by small numerical geometry roundoff.

        Sampling order:
            Original output-grid voxel
                -> common reference voxel
                -> this run's reference voxel
                -> this run's original input voxel.

        For exactly matching grids:
            composite = withinRun @ runAlignment

        Explicit coordinate bridges connect separately recorded descriptions
        of the same source grid. They do not estimate another registration.

        Returns a portable application plan. All intermediate transforms
        referenced by generated lineage are retained in derivedTransforms.
        Original transforms remain in the explicitly linked input products.

        Does not read files, resample images, publish products, or modify
        history. The application action must still verify native artifacts
        and source bytes before using this plan.

        Staged records are permitted for immediate application within their
        producing action. This helper does not commit an action's products.
        """
        from copy import deepcopy

        from .pglExternalTools import _copyProvenance

        motionProduct = _copyProvenance(motionProduct)
        alignmentProduct = _copyProvenance(alignmentProduct)

        motionByScan = cls.validateMotionProduct(motionProduct)
        if motionProduct["metadata"]["schemaVersion"] != 2:
            raise ValueError("Composition requires interpreted version-2 motion.")

        alignment = cls.validateRunAlignmentProduct(alignmentProduct)

        for record, label in (
            (motionProduct, "motion"),
            (alignmentProduct, "run alignment"),
        ):
            cls._text(record.get("productID"), f"{label} productID")
            cls._text(record.get("producerActionID"), f"{label} producerActionID")

        if (
            alignment["withinRunMotionProductID"] != motionProduct["productID"]
            or alignment["withinRunMotionProducerActionID"] != motionProduct["producerActionID"]
        ):
            raise ValueError(
                "Run alignment was not estimated from this exact motion product "
                "and producing execution. Supply the recorded input product."
            )

        entries = alignment["scans"]
        alignmentByScan = {entry["scanUUID"]: entry for entry in entries}
        geometryToleranceMM = 0.001

        def requireMatchingGrid(left, right):
            """Check grid locations without changing either recorded geometry."""
            if left["spatialShape"] != right["spatialShape"]:
                raise ValueError("Composite application requires matching spatial dimensions.")

            shape = left["spatialShape"]
            corners = np.array([
                [x, y, z, 1.0]
                for x in (0, shape[0] - 1)
                for y in (0, shape[1] - 1)
                for z in (0, shape[2] - 1)
            ]).T
            difference = (
                np.asarray(left["voxelToPhysicalMM"], dtype=float)
                - np.asarray(right["voxelToPhysicalMM"], dtype=float)
            )
            error = float(np.linalg.norm((difference @ corners)[:3], axis=0).max())

            if error > geometryToleranceMM:
                raise ValueError(
                    f"Acquisition grids differ by up to {error:.6g} mm. "
                    "Cross-grid application is not supported by this adapter."
                )

        def sameSourceBridge(sourceFrame, targetFrame):
            """Connect coordinate descriptions of the same identified source.

            The source-stage identity and physical geometry must agree.
            Different physicalSpaceID labels are permitted only here, where
            the product associations establish that both describe the same
            acquisition/source stage.
            """
            sourceFrame = cls._validateFrame(sourceFrame)
            targetFrame = cls._validateFrame(targetFrame)

            if sourceFrame["geometryID"] != targetFrame["geometryID"]:
                raise ValueError("A same-source bridge requires matching geometry IDs.")

            if sourceFrame["spatialShape"] != targetFrame["spatialShape"]:
                raise ValueError("Conflicting dimensions for the same source stage.")

            if not np.allclose(
                sourceFrame["voxelToPhysicalMM"],
                targetFrame["voxelToPhysicalMM"],
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError("Conflicting physical geometry for the same source stage.")

            matrix = (
                np.asarray(targetFrame["voxelToFrame"], dtype=float)
                @ np.linalg.inv(np.asarray(sourceFrame["voxelToFrame"], dtype=float))
            )
            return cls.makeTransform(
                matrix,
                domainFrame=sourceFrame,
                codomainFrame=targetFrame,
                purpose="sampling",
                interpretationStatus="validated",
                adapter="pgl.sameSourceCoordinateBridge.v1",
            )

        # Verify all scientific associations before constructing the plan.
        for entry in entries:
            scanUUID = entry["scanUUID"]
            motion = motionByScan.get(scanUUID)

            if motion is None:
                raise ValueError(f"Motion product does not cover alignment scan {scanUUID!r}.")

            for field in (
                "sourceStageID", "sourceIdentity",
                "volumeIndices", "referenceVolumeIndex",
            ):
                if entry[field] != motion[field]:
                    raise ValueError(f"Run alignment disagrees with within-run motion: {field}.")

            expectedRow = motion["volumeIndices"].index(motion["referenceVolumeIndex"])
            if entry["referenceRow"] != expectedRow:
                raise ValueError("Run-alignment reference row disagrees with motion volume ordering.")

            referenceUUID = entry["commonReferenceScanUUID"]
            referenceMotion = motionByScan.get(referenceUUID)
            referenceAlignment = alignmentByScan.get(referenceUUID)

            if referenceMotion is None or referenceAlignment is None:
                raise ValueError("The common reference is missing from the input products.")

            if (
                not referenceAlignment["isReference"]
                or referenceAlignment["groupIndex"] != entry["groupIndex"]
            ):
                raise ValueError("The common reference is not this group's identity-reference record.")

            if (
                entry["commonReferenceSourceStageID"] != referenceMotion["sourceStageID"]
                or entry["commonReferenceVolumeIndex"] != referenceMotion["referenceVolumeIndex"]
            ):
                raise ValueError("The common reference stage/volume disagrees with within-run motion.")

        results = []

        for entry in entries:
            scanUUID = entry["scanUUID"]
            motion = motionByScan[scanUUID]
            interpretation = motion["interpretation"]

            withinFrame = cls._validateFrame(interpretation["voxelFrame"])
            runTransform = cls._validateTransform(entry["voxelTransform"])
            commonFrame = runTransform["domainFrame"]
            runFrame = runTransform["codomainFrame"]
            referenceMotion = motionByScan[entry["commonReferenceScanUUID"]]
            referenceWithinFrame = cls._validateFrame(
                referenceMotion["interpretation"]["voxelFrame"]
            )

            if any(
                frame["representation"] != "voxelIndices"
                for frame in (withinFrame, runFrame, commonFrame, referenceWithinFrame)
            ):
                raise ValueError("Composite application requires interpreted voxel sampling maps.")

            if (
                runFrame["geometryID"] != motion["sourceStageID"]
                or commonFrame["geometryID"] != referenceMotion["sourceStageID"]
            ):
                raise ValueError("Run-alignment endpoint stages disagree with the motion product.")

            # Verify both representations of the common reference describe
            # the exact same source geometry, even though their space labels
            # were assigned by different producing executions.
            if (
                commonFrame["spatialShape"] != referenceWithinFrame["spatialShape"]
                or not np.allclose(
                    commonFrame["voxelToPhysicalMM"],
                    referenceWithinFrame["voxelToPhysicalMM"],
                    rtol=0,
                    atol=1e-8,
                )
            ):
                raise ValueError("Common-reference geometry differs between the input products.")

            requireMatchingGrid(commonFrame, runFrame)

            # The output remains this run's own acquisition grid. Give its
            # group-coordinate description an explicit frame definition.
            outputFrame = cls.makeFrame(
                geometryID=withinFrame["geometryID"],
                physicalSpaceID=commonFrame["physicalSpaceID"],
                spatialShape=withinFrame["spatialShape"],
                voxelToPhysicalMM=withinFrame["voxelToPhysicalMM"],
                representation="voxelIndices",
            )
            requireMatchingGrid(outputFrame, commonFrame)

            originalToOutput = sameSourceBridge(withinFrame, outputFrame)
            runToWithin = sameSourceBridge(runFrame, withinFrame)

            # Preserve tiny grid differences explicitly instead of replacing
            # an acquisition header or silently assuming identical coordinates.
            outputToCommon = cls.makeTransform(
                np.linalg.inv(np.asarray(commonFrame["voxelToPhysicalMM"], dtype=float))
                @ np.asarray(outputFrame["voxelToPhysicalMM"], dtype=float),
                domainFrame=outputFrame,
                codomainFrame=commonFrame,
                purpose="sampling",
                interpretationStatus="validated",
                adapter="pgl.sharedAcquisitionGridBridge.v1",
            )

            # compose(first, second) applies first THEN second.
            originalToCommon = cls.compose(originalToOutput, outputToCommon)
            originalToRun = cls.compose(originalToCommon, runTransform)
            originalToWithin = cls.compose(originalToRun, runToWithin)

            derivedTransforms = [
                originalToOutput,
                outputToCommon,
                runToWithin,
                originalToCommon,
                originalToRun,
                originalToWithin,
            ]
            composites = []

            for rowIndex, withinTransform in enumerate(interpretation["voxelTransforms"]):
                withinTransform = cls._validateTransform(withinTransform)
                final = cls.compose(originalToWithin, withinTransform)

                if (
                    final["domainFrame"] != withinFrame
                    or final["codomainFrame"] != withinFrame
                ):
                    raise ValueError("Composite endpoints do not describe the original acquisition grid.")

                derivedTransforms.append(final)
                composites.append({
                    "rowIndex": rowIndex,
                    "originalVolumeIndex": motion["volumeIndices"][rowIndex],
                    "withinRunTransformID": withinTransform["transformID"],
                    "runAlignmentTransformID": runTransform["transformID"],
                    "transform": final,
                })

            # Every generated lineage reference must resolve either to another
            # retained derived transform or to an explicit input transform.
            inputTransformIDs = {
                runTransform["transformID"],
                *(
                    transform["transformID"]
                    for transform in interpretation["voxelTransforms"]
                ),
            }
            generatedIDs = [transform["transformID"] for transform in derivedTransforms]

            if len(set(generatedIDs)) != len(generatedIDs):
                raise ValueError("Generated transform identities are not unique.")

            knownIDs = inputTransformIDs | set(generatedIDs)
            for transform in derivedTransforms:
                if set(transform["inputTransformIDs"]) - knownIDs:
                    raise ValueError("Composite lineage refers to a missing transform.")

            results.append({
                "scanUUID": scanUUID,
                "recordedScanIndex": entry["scanIndex"],
                "groupIndex": entry["groupIndex"],
                "sourceStageID": motion["sourceStageID"],
                "sourceIdentity": deepcopy(motion["sourceIdentity"]),
                "volumeIndices": list(motion["volumeIndices"]),
                "withinRunReferenceVolumeIndex": motion["referenceVolumeIndex"],
                "commonReferenceScanUUID": entry["commonReferenceScanUUID"],
                "commonReferenceVolumeIndex": entry["commonReferenceVolumeIndex"],
                "outputFrame": withinFrame,
                "inputTransformIDs": sorted(inputTransformIDs),
                "runAlignmentTransform": runTransform,
                "derivedTransforms": derivedTransforms,
                "composites": composites,
            })

        return {
            "schemaVersion": 1,
            "kind": "withinAndBetweenRunMotionApplicationPlan",
            "motionProductID": motionProduct["productID"],
            "motionProducerActionID": motionProduct["producerActionID"],
            "alignmentProductID": alignmentProduct["productID"],
            "alignmentProducerActionID": alignmentProduct["producerActionID"],
            "gridToleranceMM": geometryToleranceMM,
            "outputGrid": "eachOriginalAcquisition",
            "scans": results,
            "description": (
                "For each voxel on a run's original acquisition grid, locate "
                "the corresponding common-reference position, then the run's "
                "reference position, then the original input-volume sampling "
                "position. Apply this composite once to the original source. "
                "No already-corrected image is used as input."
            ),
        }
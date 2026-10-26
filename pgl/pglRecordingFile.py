################################################################
#   filename: pglRecordingFile.py
#    purpose: Save and load PGL command/presentation recordings
#             in a versioned, self-contained HDF5 file.
#         by: JLG
#       date: Oct 25, 2026
################################################################

#############
# Imports
#############
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from .pglSerialize import pglSerialize


class pglRecordingFile:
    """Local HDF5 storage for captured commands and presentation timing.

    This class stores data only. Loading does not open a renderer, execute
    commands, or establish that a recording is independently replayable.
    """

    formatName = "pglRecording"
    formatVersion = 1

    commandArrayNames = ("commandValues", "commandTimes")
    presentationArrayNames = (
        "drawablePresented",
        "processedTime",
        "drawableAcquired",
        "success",
        "commandCode",
        "commandIndices",
    )

    ################################################################
    # Validate the capture structures used by this format
    ################################################################
    @classmethod
    def validate(cls, commands=None, presentations=None):
        """Validate basic structure and associations without interpreting timing."""
        if commands is None and presentations is None:
            raise ValueError("No command or presentation recording was supplied.")

        for label, data, arrayNames in (
            ("commands", commands, cls.commandArrayNames),
            ("presentations", presentations, cls.presentationArrayNames),
        ):
            if data is None: continue
            if not isinstance(data, dict): raise TypeError(f"{label} must be a dictionary.")

            count = data.get("count")
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError(f"{label}.count must be a nonnegative integer.")

            for name in arrayNames:
                array = data.get(name)
                if not isinstance(array, np.ndarray) or array.ndim != 1 or len(array) != count:
                    raise ValueError(f"{label}.{name} must be a one-dimensional array of length {count}.")
                if array.dtype.kind not in "iuf":
                    raise TypeError(f"{label}.{name} must be a numeric array.")

        if commands is not None:
            if commands["commandValues"].dtype.kind not in "iu":
                raise TypeError("Command values must be integers.")

            payloads = commands.get("commandData")
            if not isinstance(payloads, list) or len(payloads) != commands["count"]:
                raise ValueError("commandData must contain one entry per command.")

            for chunks in payloads:
                if chunks is None: continue
                if not isinstance(chunks, list): raise TypeError("Each command payload must be a list or None.")
                if any(not isinstance(chunk, bytes) for chunk in chunks):
                    raise TypeError("Payload chunks must be immutable bytes.")

        if presentations is not None:
            indices = presentations["commandIndices"]
            if indices.dtype.kind != "i": raise TypeError("Presentation command indices must be signed integers.")
            if np.any(indices < -1): raise ValueError("Presentation command indices cannot be less than -1.")

            # Do not silently write a presentation-only file with dangling
            # references to command data that was omitted.
            if commands is None:
                if np.any(indices != -1): raise ValueError("Presentation command associations require the command recording.")
            elif np.any(indices >= commands["count"]):
                raise ValueError("Presentation association exceeds the command recording.")

    ################################################################
    # Metadata encoding
    ################################################################
    @staticmethod
    def _encodeMetadata(data, excludedNames):
        """Use the existing strict JSON codec for small metadata only."""
        metadata = {name: value for name, value in data.items() if name not in excludedNames}
        return pglSerialize.toJSON(metadata, strict=True)

    @staticmethod
    def _writeMetadata(group, text):
        """Store JSON as a UTF-8 dataset rather than a large HDF5 attribute."""
        group.create_dataset("metadata", data=text, dtype=h5py.string_dtype("utf-8"))

    @staticmethod
    def _readMetadata(group):
        """Restore metadata without permissive field skipping."""
        metadata = pglSerialize.fromJSON(group["metadata"].asstr()[()], strict=True)
        if not isinstance(metadata, dict): raise ValueError("Recording metadata must restore to a dictionary.")
        return metadata

    ################################################################
    # Save
    ################################################################
    @classmethod
    def save(cls, filename, commands=None, presentations=None):
        """Write a local recording, atomically replacing the destination.

        Capture must already be stopped. No recording objects are modified.
        Validation/metadata encoding occurs before creating the temporary file.
        """
        cls.validate(commands, presentations)

        commandMetadata = None
        presentationMetadata = None
        if commands is not None:
            commandMetadata = cls._encodeMetadata(commands, (*cls.commandArrayNames, "commandData"))
        if presentations is not None:
            presentationMetadata = cls._encodeMetadata(presentations, cls.presentationArrayNames)

        filename = Path(filename).expanduser()
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Same-directory temporary file allows local atomic replacement.
        descriptor, temporaryName = tempfile.mkstemp(prefix=f".{filename.name}.", suffix=".tmp", dir=filename.parent)
        os.close(descriptor)

        try:
            with h5py.File(temporaryName, "w") as file:
                file.attrs["formatName"] = cls.formatName
                file.attrs["formatVersion"] = cls.formatVersion

                if commands is not None:
                    group = file.create_group("commands")
                    cls._writeMetadata(group, commandMetadata)

                    for name in cls.commandArrayNames:
                        group.create_dataset(name, data=commands[name])

                    # Preserve both command boundaries and individual write()
                    # chunks. No concatenation of all payload bytes is required.
                    payloads = commands["commandData"]
                    commandOffsets = np.zeros(len(payloads) + 1, dtype=np.uint64)
                    payloadPresent = np.zeros(len(payloads), dtype=np.bool_)
                    chunkOffsets = [0]

                    for index, chunks in enumerate(payloads):
                        payloadPresent[index] = chunks is not None
                        if chunks is not None:
                            for chunk in chunks: chunkOffsets.append(chunkOffsets[-1] + len(chunk))
                        commandOffsets[index + 1] = len(chunkOffsets) - 1

                    group.create_dataset("commandChunkOffsets", data=commandOffsets)
                    group.create_dataset("chunkByteOffsets", data=np.asarray(chunkOffsets, dtype=np.uint64))
                    group.create_dataset("payloadPresent", data=payloadPresent)
                    payloadBytes = group.create_dataset("payloadBytes", shape=(chunkOffsets[-1],), dtype=np.uint8)

                    # Disk work happens after recording. Retain the original
                    # chunk buffers and avoid an additional giant bytes object.
                    byteIndex = 0
                    for chunks in payloads:
                        if chunks is None: continue
                        for chunk in chunks:
                            if chunk: payloadBytes[byteIndex:byteIndex + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                            byteIndex += len(chunk)

                if presentations is not None:
                    group = file.create_group("presentations")
                    cls._writeMetadata(group, presentationMetadata)

                    for name in cls.presentationArrayNames:
                        group.create_dataset(name, data=presentations[name])

            # Publish only after all datasets have been written and HDF5 closed.
            os.replace(temporaryName, filename)

        finally:
            # Leave an existing destination untouched if writing failed.
            if os.path.exists(temporaryName): os.unlink(temporaryName)

        return filename

    ################################################################
    # Validate packed payload offsets
    ################################################################
    @staticmethod
    def _validateOffsets(offsets, expectedLength, finalValue, label):
        """Reject malformed offsets before slicing payload storage."""
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != expectedLength:
            raise ValueError(f"Invalid {label} shape or dtype.")
        if offsets[0] != 0 or offsets[-1] != finalValue:
            raise ValueError(f"Invalid {label} endpoints.")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"{label} must be nondecreasing.")

    ################################################################
    # Load
    ################################################################
    @classmethod
    def load(cls, filename):
        """Load arrays and payload bytes without executing recorded commands."""
        commands = None
        presentations = None

        with h5py.File(Path(filename).expanduser(), "r") as file:
            if file.attrs.get("formatName") != cls.formatName:
                raise ValueError("Not a PGL recording file.")
            if file.attrs.get("formatVersion") != cls.formatVersion:
                raise ValueError(f"Unsupported recording format version: {file.attrs.get('formatVersion')}")

            if "commands" in file:
                group = file["commands"]
                commands = cls._readMetadata(group)
                for name in cls.commandArrayNames: commands[name] = group[name][...]

                count = len(commands["commandValues"])
                commandOffsets = group["commandChunkOffsets"][...]
                chunkOffsets = group["chunkByteOffsets"][...]
                payloadPresent = group["payloadPresent"][...]
                payloadBytes = group["payloadBytes"]

                if payloadBytes.ndim != 1 or payloadBytes.dtype != np.dtype("uint8"):
                    raise ValueError("Invalid payload byte dataset.")
                if payloadPresent.shape != (count,) or payloadPresent.dtype != np.dtype("bool"):
                    raise ValueError("Invalid payload-presence array.")
                if chunkOffsets.ndim != 1 or len(chunkOffsets) == 0:
                    raise ValueError("Invalid chunk offsets.")

                cls._validateOffsets(commandOffsets, count + 1, len(chunkOffsets) - 1, "command chunk offsets")
                cls._validateOffsets(chunkOffsets, len(chunkOffsets), len(payloadBytes), "chunk byte offsets")

                payloads = []
                for index in range(count):
                    firstChunk = int(commandOffsets[index])
                    lastChunk = int(commandOffsets[index + 1])

                    if not payloadPresent[index]:
                        if firstChunk != lastChunk: raise ValueError("Absent command payload has recorded chunks.")
                        payloads.append(None)
                        continue

                    chunks = []
                    for chunkIndex in range(firstChunk, lastChunk):
                        start = int(chunkOffsets[chunkIndex])
                        stop = int(chunkOffsets[chunkIndex + 1])
                        chunks.append(payloadBytes[start:stop].tobytes())
                    payloads.append(chunks)

                commands["commandData"] = payloads

            if "presentations" in file:
                group = file["presentations"]
                presentations = cls._readMetadata(group)
                for name in cls.presentationArrayNames: presentations[name] = group[name][...]

        cls.validate(commands, presentations)
        return {"commands": commands, "presentations": presentations}
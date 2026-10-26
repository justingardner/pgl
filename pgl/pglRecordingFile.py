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
    """Local HDF5 storage for commands, presentations, and movie timing.

    Loading does not open a renderer, execute commands, or establish that
    a recording is independently replayable.
    """

    formatName = "pglRecording"
    formatVersion = 3

    commandArrayNames = ("commandValues", "commandTimes")
    presentationArrayNames = (
        "drawablePresented",
        "processedTime",
        "drawableAcquired",
        "success",
        "commandCode",
        "commandIndices",
    )
    playbackArrayNames = (
        "presentedTimes",
        "movieTimes",
        "targetPresentationTimestamps",
        "drawFrameTimes",
    )
    movieDrawNumericFields = (
        "drawFrameTime",
        "targetPresentationTimestamp",
        "movieTime",
        "drawablePresented",
        "flushSuccess",
    )

    ################################################################
    # Validate command and presentation structures
    ################################################################
    @classmethod
    def validate(cls, commands=None, presentations=None):
        """Validate structure and associations without interpreting timing."""
        if commands is None and presentations is None:
            raise ValueError("No command or presentation recording was supplied.")

        for label, data, arrayNames in (
            ("commands", commands, cls.commandArrayNames),
            ("presentations", presentations, cls.presentationArrayNames),
        ):
            if data is None: continue
            if not isinstance(data, dict): raise TypeError(f"{label} must be a dictionary.")

            count = data.get("count")
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
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

            # Do not silently omit commands referenced by presentation records.
            if commands is None:
                if np.any(indices != -1): raise ValueError("Presentation command associations require the command recording.")
            elif np.any(indices >= commands["count"]):
                raise ValueError("Presentation association exceeds the command recording.")

    ################################################################
    # Metadata encoding
    ################################################################
    @staticmethod
    def _encodeMetadata(data, excludedNames):
        """Use strict JSON encoding for small metadata, not payload arrays."""
        metadata = {name: value for name, value in data.items() if name not in excludedNames}
        return pglSerialize.toJSON(metadata, strict=True)

    @staticmethod
    def _writeMetadata(group, text):
        """Store JSON as a UTF-8 dataset rather than a large attribute."""
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
    def save(cls, filename, commands=None, presentations=None, moviePlaybacks=None, movieDraws=None, metadata=None):
        """Write a local recording and atomically replace the destination.

        Capture must already be stopped. Validation and metadata encoding occur
        before creating the temporary file. Existing capture data is unchanged.

        Atomic publication does not guarantee durability after power loss.
        """
        if all(value is None for value in (commands, presentations, moviePlaybacks, movieDraws, metadata)):
            raise ValueError("No recording data was supplied.")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Session metadata must be a dictionary or None.")

        if commands is not None or presentations is not None:
            cls.validate(commands, presentations)

        # Encode metadata before any destination file can be replaced.
        sessionMetadata = None if metadata is None else cls._encodeMetadata(metadata, ())
        commandMetadata = None if commands is None else cls._encodeMetadata(commands, (*cls.commandArrayNames, "commandData"))
        presentationMetadata = None if presentations is None else cls._encodeMetadata(presentations, cls.presentationArrayNames)
        preparedPlaybacks, preparedDraws = cls._prepareMovies(moviePlaybacks, movieDraws)

        filename = Path(filename).expanduser()
        filename.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporaryName = tempfile.mkstemp(prefix=f".{filename.name}.", suffix=".tmp", dir=filename.parent)

        try:
            os.close(descriptor)

            with h5py.File(temporaryName, "w") as file:
                file.attrs["formatName"] = cls.formatName
                file.attrs["formatVersion"] = cls.formatVersion

                if sessionMetadata is not None: cls._writeMetadata(file, sessionMetadata)

                if commands is not None:
                    group = file.create_group("commands")
                    cls._writeMetadata(group, commandMetadata)
                    cls._writeCommands(group, commands)

                if presentations is not None:
                    group = file.create_group("presentations")
                    cls._writeMetadata(group, presentationMetadata)
                    for name in cls.presentationArrayNames: group.create_dataset(name, data=presentations[name])

                cls._writeMovies(file, preparedPlaybacks, preparedDraws)

            # Publish only after all writes and HDF5 closure succeed.
            os.replace(temporaryName, filename)

        finally:
            if os.path.exists(temporaryName): os.unlink(temporaryName)

        return filename

    ################################################################
    # Write command arrays and packed payloads
    ################################################################
    @classmethod
    def _writeCommands(cls, group, commands):
        """Preserve command/chunk boundaries without concatenating all bytes."""
        for name in cls.commandArrayNames: group.create_dataset(name, data=commands[name])

        # Preserve None separately from an empty list or an empty bytes chunk.
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

        # Avoid building one additional giant bytes object.
        byteIndex = 0
        for chunks in payloads:
            if chunks is None: continue
            for chunk in chunks:
                if chunk: payloadBytes[byteIndex:byteIndex + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
                byteIndex += len(chunk)

    ################################################################
    # Validate packed payload offsets
    ################################################################
    @staticmethod
    def _validateOffsets(offsets, expectedLength, finalValue, label):
        """Reject malformed offsets before slicing payload storage."""
        if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != expectedLength or expectedLength < 1:
            raise ValueError(f"Invalid {label} shape or dtype.")
        if offsets[0] != 0 or offsets[-1] != finalValue:
            raise ValueError(f"Invalid {label} endpoints.")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"{label} must be nondecreasing.")

    ################################################################
    # Read command arrays and packed payloads
    ################################################################
    @classmethod
    def _readCommands(cls, group):
        """Restore outgoing command bytes without executing them."""
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
        return commands

    ################################################################
    # Load
    ################################################################
    @classmethod
    def load(cls, filename):
        """Load supported recording versions without executing commands.

        Older files without session metadata return metadata=None.
        """
        metadata = None
        commands = None
        presentations = None
        moviePlaybacks = None
        movieDraws = None

        with h5py.File(Path(filename).expanduser(), "r") as file:
            if file.attrs.get("formatName") != cls.formatName:
                raise ValueError("Not a PGL recording file.")

            version = file.attrs.get("formatVersion")
            if version not in (1, 2, cls.formatVersion):
                raise ValueError(f"Unsupported recording format version: {version}")

            if version == cls.formatVersion and "metadata" in file:
                metadata = cls._readMetadata(file)

            if "commands" in file: commands = cls._readCommands(file["commands"])

            if "presentations" in file:
                group = file["presentations"]
                presentations = cls._readMetadata(group)
                for name in cls.presentationArrayNames: presentations[name] = group[name][...]

            if version >= 2:
                moviePlaybacks, movieDraws = cls._readMovies(file)

        if commands is not None or presentations is not None:
            cls.validate(commands, presentations)

        if all(value is None for value in (commands, presentations, moviePlaybacks, movieDraws, metadata)):
            raise ValueError("File contains no recording sections.")

        return {
            "metadata": metadata,
            "commands": commands,
            "presentations": presentations,
            "moviePlaybacks": moviePlaybacks,
            "movieDraws": movieDraws,
        }

    ################################################################
    # Prepare movie data before opening the destination
    ################################################################
    @classmethod
    def _prepareMovies(cls, moviePlaybacks, movieDraws):
        """Validate and prepare movie data outside the file-writing operation."""
        preparedPlaybacks = None

        if moviePlaybacks is not None:
            if not isinstance(moviePlaybacks, list): raise TypeError("moviePlaybacks must be a list.")
            preparedPlaybacks = []

            for record in moviePlaybacks:
                if not isinstance(record, dict): raise TypeError("Each playback must be a dictionary.")

                arrays = {}
                for name in cls.playbackArrayNames:
                    value = record[name]
                    if value is not None:
                        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind not in "iuf":
                            raise TypeError(f"Playback {name} must be a numeric one-dimensional array or None.")
                    arrays[name] = value

                # Incomplete timing remains savable for later diagnosis.
                metadata = cls._encodeMetadata(record, cls.playbackArrayNames)
                preparedPlaybacks.append((metadata, arrays))

        preparedDraws = None
        if movieDraws is not None:
            if not isinstance(movieDraws, dict) or set(movieDraws) != {"sources", "samples", "flushGroupCount"}:
                raise ValueError("Unexpected movie-draw fields.")

            sources = movieDraws["sources"]
            samples = movieDraws["samples"]
            flushGroupCount = movieDraws["flushGroupCount"]

            if not isinstance(sources, list) or not isinstance(samples, list):
                raise TypeError("Movie-draw sources and samples must be lists.")
            if isinstance(flushGroupCount, bool) or not isinstance(flushGroupCount, int) or flushGroupCount < 0:
                raise ValueError("flushGroupCount must be a nonnegative integer.")

            numericFields = cls.movieDrawNumericFields
            requiredFields = {
                "sourceIndex", "flushGroupIndex", "flushStatus",
                "flushTimingRecorded", *numericFields,
            }

            count = len(samples)
            values = np.zeros((count, len(numericFields)), dtype=np.float64)
            present = np.zeros((count, len(numericFields)), dtype=np.bool_)
            sourceIndices = np.empty(count, dtype=np.int64)
            flushGroupIndices = np.empty(count, dtype=np.int64)
            timingRecorded = np.empty(count, dtype=np.bool_)
            statuses = []

            for index, sample in enumerate(samples):
                if not isinstance(sample, dict) or set(sample) != requiredFields:
                    raise ValueError("Unexpected movie-draw sample fields.")

                sourceIndex = sample["sourceIndex"]
                groupIndex = sample["flushGroupIndex"]
                for name, value in (("sourceIndex", sourceIndex), ("flushGroupIndex", groupIndex)):
                    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                        raise TypeError(f"{name} must be an integer.")

                if not 0 <= sourceIndex < len(sources): raise ValueError("Invalid movie-draw source index.")
                if groupIndex < -1 or groupIndex >= flushGroupCount: raise ValueError("Invalid movie-draw flush group index.")
                if not isinstance(sample["flushStatus"], str): raise TypeError("flushStatus must be a string.")
                if not isinstance(sample["flushTimingRecorded"], (bool, np.bool_)):
                    raise TypeError("flushTimingRecorded must be Boolean.")

                sourceIndices[index] = sourceIndex
                flushGroupIndices[index] = groupIndex
                timingRecorded[index] = sample["flushTimingRecorded"]
                statuses.append(sample["flushStatus"])

                # Float64 times and uint32 success values fit these columns.
                # Presence is separate so None remains distinct from NaN.
                for column, name in enumerate(numericFields):
                    value = sample[name]
                    if value is None: continue
                    array = np.asarray(value)
                    if array.ndim != 0 or array.dtype.kind not in "iuf":
                        raise TypeError(f"Movie-draw {name} must be a numeric scalar or None.")
                    values[index, column] = array
                    present[index, column] = True

            metadata = pglSerialize.toJSON({
                "sources": sources,
                "flushGroupCount": flushGroupCount,
                "numericFields": list(numericFields),
                "statuses": statuses,
            }, strict=True)

            preparedDraws = {
                "metadata": metadata,
                "values": values,
                "present": present,
                "sourceIndices": sourceIndices,
                "flushGroupIndices": flushGroupIndices,
                "timingRecorded": timingRecorded,
            }

        return preparedPlaybacks, preparedDraws

    ################################################################
    # Write prepared movie sections
    ################################################################
    @classmethod
    def _writeMovies(cls, file, preparedPlaybacks, preparedDraws):
        """Store playback arrays and columnar incremental-draw samples."""
        if preparedPlaybacks is None and preparedDraws is None: return
        movies = file.create_group("movies")

        if preparedPlaybacks is not None:
            playbacks = movies.create_group("playbacks")
            playbacks.attrs["count"] = len(preparedPlaybacks)

            for index, (metadata, arrays) in enumerate(preparedPlaybacks):
                group = playbacks.create_group(str(index))
                cls._writeMetadata(group, metadata)

                # An absent timing dataset represents an original None value.
                for name, value in arrays.items():
                    if value is not None: group.create_dataset(name, data=value)

        if preparedDraws is not None:
            group = movies.create_group("draws")
            cls._writeMetadata(group, preparedDraws["metadata"])
            for name, value in preparedDraws.items():
                if name != "metadata": group.create_dataset(name, data=value)

    ################################################################
    # Read movie sections
    ################################################################
    @classmethod
    def _readMovies(cls, file):
        """Restore movie data without interpreting or repairing raw timing."""
        moviePlaybacks = None
        movieDraws = None

        if "movies" not in file: return moviePlaybacks, movieDraws
        movies = file["movies"]

        if "playbacks" in movies:
            playbacks = movies["playbacks"]
            count = int(playbacks.attrs["count"])
            if count < 0 or set(playbacks.keys()) != {str(index) for index in range(count)}:
                raise ValueError("Invalid movie playback groups.")

            moviePlaybacks = []
            for index in range(count):
                group = playbacks[str(index)]
                record = cls._readMetadata(group)
                for name in cls.playbackArrayNames:
                    record[name] = group[name][...] if name in group else None
                moviePlaybacks.append(record)

        if "draws" in movies:
            group = movies["draws"]
            metadata = cls._readMetadata(group)
            numericFields = cls.movieDrawNumericFields
            if metadata["numericFields"] != list(numericFields):
                raise ValueError("Unknown movie-draw numeric fields.")

            values = group["values"][...]
            present = group["present"][...]
            sourceIndices = group["sourceIndices"][...]
            flushGroupIndices = group["flushGroupIndices"][...]
            timingRecorded = group["timingRecorded"][...]
            statuses = metadata["statuses"]
            count = len(statuses)

            if values.shape != (count, len(numericFields)) or values.dtype != np.dtype("float64"):
                raise ValueError("Invalid movie-draw values.")
            if present.shape != values.shape or present.dtype != np.dtype("bool"):
                raise ValueError("Invalid movie-draw presence mask.")
            for array in (sourceIndices, flushGroupIndices):
                if array.shape != (count,) or array.dtype.kind != "i":
                    raise ValueError("Invalid movie-draw indices.")
            if timingRecorded.shape != (count,) or timingRecorded.dtype != np.dtype("bool"):
                raise ValueError("Invalid movie-draw timing flags.")

            samples = []
            for index in range(count):
                sample = {
                    "sourceIndex": int(sourceIndices[index]),
                    "flushGroupIndex": int(flushGroupIndices[index]),
                    "flushStatus": statuses[index],
                    "flushTimingRecorded": bool(timingRecorded[index]),
                }
                for column, name in enumerate(numericFields):
                    sample[name] = float(values[index, column]) if present[index, column] else None
                samples.append(sample)

            movieDraws = {
                "sources": metadata["sources"],
                "samples": samples,
                "flushGroupCount": metadata["flushGroupCount"],
            }

        # Reuse the structural checks used before saving.
        cls._prepareMovies(moviePlaybacks, movieDraws)
        return moviePlaybacks, movieDraws
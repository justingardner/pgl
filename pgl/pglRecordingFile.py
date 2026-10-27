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
    # Analyze a recording without replaying or modifying it
    ################################################################
    @classmethod
    def analyze(cls, recording, byteOrder="little"):
        """Return offline findings about captured commands and timing.

        This is diagnostic analysis, not proof of replay completeness.
        Movies are analyzed as source references and timing, not replayed.

        byteOrder describes the computer that packed the command payloads.
        Current Mac captures use little-endian encoding; older recordings
        do not explicitly store the capture computer's byte order.
        """
        if not isinstance(recording, dict): raise TypeError("recording must be a dictionary.")
        if byteOrder not in ("little", "big"): raise ValueError("byteOrder must be 'little' or 'big'.")

        commands = recording.get("commands")
        presentations = recording.get("presentations")
        if commands is not None or presentations is not None:
            cls.validate(commands, presentations)

        findings = {}

        def add(code, severity, message, location=None):
            # Aggregate repeated findings; avoid storing thousands of indices
            # in metadata for a long recording. Counts remain complete.
            if code not in findings:
                findings[code] = {
                    "code": code,
                    "severity": severity,
                    "message": message,
                    "count": 0,
                    "examples": [],
                }
            finding = findings[code]
            finding["count"] += 1
            if location is not None and len(finding["examples"]) < 10:
                finding["examples"].append(location)

        def usableTime(value):
            # This is a necessary timing check, not proof of presentation.
            if value is None: return False
            array = np.asarray(value)
            return bool(array.ndim == 0 and array.dtype.kind in "iuf"
                        and np.isfinite(array) and array > 0)

        commandInfo = cls._analyzeCommands(commands, byteOrder, add)

        if presentations is None:
            add("flushTimingNotRecorded", "info", "No explicit flush timing section is available.")
        else:
            associated = set()
            for index in range(presentations["count"]):
                location = {"section": "presentations", "index": index}
                commandIndex = int(presentations["commandIndices"][index])
                presented = presentations["drawablePresented"][index]
                success = presentations["success"][index]

                if not usableTime(presented):
                    add("flushPresentationTimeUnusable", "warning",
                        "Flush presentation time is missing, nonfinite, or nonpositive.", location)

                if not np.isfinite(success):
                    add("flushSuccessMissing", "warning", "Flush success result is missing or nonfinite.", location)
                elif success != 1:
                    add("flushNotSuccessful", "warning", "Flush result does not report success=1.", location)

                if commandIndex == -1:
                    add("flushWithoutCommandLink", "info",
                        "Flush timing has no recorded command association; this is allowed.", location)
                else:
                    if commandIndex in associated:
                        add("duplicateFlushCommandLink", "warning",
                            "Multiple presentation rows reference the same command.", location)
                    associated.add(commandIndex)

                    name = commandInfo["names"][commandIndex]
                    if name is not None and name != "mglFlush":
                        add("flushLinkedToNonFlush", "warning",
                            "A presentation row references a command other than mglFlush.", location)

                    # Compare raw numeric codes even if the command name is unknown.
                    code = presentations["commandCode"][index]
                    if np.isfinite(code) and code != commands["commandValues"][commandIndex]:
                        add("flushCommandCodeMismatch", "warning",
                            "The returned command code differs from the associated recorded command.", location)

            for commandIndex in commandInfo["flushIndices"]:
                if commandIndex not in associated:
                    add("flushWithoutTimingRow", "info",
                        "A recorded flush has no timing row; capture may have been disabled or interrupted.",
                        {"section": "commands", "index": commandIndex})

        cls._analyzeMovies(recording, commandInfo, usableTime, add)

        orderedFindings = [findings[code] for code in sorted(findings)]
        warningTypes = sum(finding["severity"] == "warning" for finding in orderedFindings)
        infoTypes = sum(finding["severity"] == "info" for finding in orderedFindings)

        return {
            "analysisVersion": 1,
            "replayCompleteness": "notEstablished",
            "assumptions": {
                "commandByteOrder": byteOrder,
                "hostClockBasis": "assumedSameAsGetSecs",
                "drawingReplaySetup": "matchingVisualAngleExtent",
                "movieAnalysis": "sourceReferencesAndTimingOnly",
            },
            "coverage": {
                "commandCount": 0 if commands is None else int(commands["count"]),
                "presentationCount": 0 if presentations is None else int(presentations["count"]),
                "reviewedCommandNames": sorted(commandInfo["reviewedNames"]),
                "unverifiedCommandNames": sorted(commandInfo["unverifiedNames"]),
            },
            "limitations": [
                "No findings does not establish replay completeness.",
                "Recorded command bytes describe attempted sends, not confirmed delivery or execution.",
                "Missing timing does not identify a communication failure or prove a dropped physical refresh.",
                "Positive presentation times are timing evidence, not independently verified visibility.",
                "Initial nonstandard renderer state and changes during capture gaps are not reconstructed.",
                "Movie source updates are not automatically treated as displayed-frame changes.",
                "External movie filenames are not resolved or checked for availability.",
                "Payload byte order is supplied to analysis, not verified from capture metadata.",
            ],
            "summary": f"{warningTypes} warning categories; {infoTypes} informational categories. Replay completeness not established.",
            "findings": orderedFindings,
        }

    ################################################################
    # Analyze reviewed atomic commands and resource lifetimes
    ################################################################
    @classmethod
    def _analyzeCommands(cls, commands, byteOrder, add):
        """Inspect known payload layouts without sending commands.

        Resource tracking describes recorded requests and returned IDs.
        Deletion success is not retained, so deletion requests end the
        analyzer's usable mapping conservatively.
        """
        info = {
            "names": [],
            "flushIndices": [],
            "reviewedNames": set(),
            "unverifiedNames": set(),
        }
        if commands is None:
            add("commandsNotRecorded", "info",
                "No command section is available; drawing reconstruction is outside this recording's coverage.")
            return info

        import struct

        prefix = "<" if byteOrder == "little" else ">"
        count = int(commands["count"])
        metadata = commands.get("metadata") or {}
        mapping = metadata.get("commandValues") or {}
        namesByValue = {}
        for name, value in mapping.items():
            namesByValue.setdefault(int(value), []).append(name)

        def commandName(value):
            names = namesByValue.get(int(value), [])
            return names[0] if len(names) == 1 else None

        info["names"] = [commandName(value) for value in commands["commandValues"]]

        def uint(chunk):
            if len(chunk) != 4: raise ValueError("Expected a uint32 chunk.")
            return struct.unpack(prefix + "I", chunk)[0]

        def string(chunk):
            if len(chunk) < 4: raise ValueError("Missing string length.")
            length = uint(chunk[:4])
            if len(chunk) != 4 + 2 * length: raise ValueError("Incorrect UTF-16 string length.")
            # _pglComm writes string content as UTF-16LE, independently
            # of the native numeric prefix encoding.
            return chunk[4:].decode("utf-16le")

        def sizes(chunks, expected):
            if len(chunks) != len(expected) or any(len(chunk) != size for chunk, size in zip(chunks, expected)):
                raise ValueError("Unexpected payload chunk lengths.")

        # Segment boundaries constrain resource certainty. Treat each resumed
        # segment conservatively, even if its command indices are adjacent.
        segments = commands.get("segments") or []
        boundaries = set()
        previousStop = 0
        segmentsValid = bool(segments)
        for segmentIndex, segment in enumerate(segments):
            start = segment.get("commandStartIndex")
            stop = segment.get("commandStopIndex")
            valid = (
                isinstance(start, (int, np.integer)) and not isinstance(start, (bool, np.bool_))
                and isinstance(stop, (int, np.integer)) and not isinstance(stop, (bool, np.bool_))
                and 0 <= start <= stop <= count and start == previousStop
            )
            if not valid:
                segmentsValid = False
                break
            if segmentIndex > 0:
                boundaries.add(int(start))
                add("commandCaptureGap", "warning",
                    "Capture resumed in another segment; intervening stimulus history and resource state are unknown.",
                    {"section": "segments", "index": segmentIndex})
            previousStop = int(stop)

        if previousStop != count: segmentsValid = False
        if not segmentsValid:
            add("commandSegmentsUnverified", "warning",
                "Segment ranges are absent or incomplete; resource continuity cannot be established.")

        resourcesByCommand = {}
        for resourceIndex, resource in enumerate(commands.get("resources") or []):
            index = resource.get("commandIndex")
            if (not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
                    or not 0 <= index < count):
                add("invalidResourceCommandLink", "warning",
                    "A resource record has an invalid creation-command index.",
                    {"section": "resources", "index": resourceIndex})
                continue
            resourcesByCommand.setdefault(int(index), []).append(resource)

        basicNames = {
            "mglFlush", "mglSetClearColor", "mglSetXform",
            "mglDots", "mglLine", "mglQuad", "mglArcs",
        }
        movieNames = {
            "mglMovieCreate", "mglMoviePlay", "mglMovieDrawFrame",
            "mglMovieSetDisplayPosition", "mglMovieStatus", "mglMovieDelete",
        }
        reviewed = basicNames | movieNames | {
            "mglSetTextRenderingParameters", "mglCreateTextTexture",
            "mglCreateTexture", "mglBltTexture", "mglDeleteTexture",
        }
        specialized = {
            "mglCreateTextTexture", "mglCreateTexture", "mglMovieCreate",
            "mglMoviePlay", "mglMovieDrawFrame", "mglMovieStatus", "mglMovieDelete",
        }
        activeResources = {}

        for index, value in enumerate(commands["commandValues"]):
            name = info["names"][index]
            location = {"section": "commands", "index": index, "command": name}
            if index in boundaries: activeResources.clear()
            if not segmentsValid: activeResources.clear()

            if not np.isfinite(commands["commandTimes"][index]):
                add("commandLoggingTimeMissing", "warning",
                    "A client command logging timestamp is nonfinite.", location)

            if name == "mglFlush": info["flushIndices"].append(index)

            if name not in reviewed:
                label = name if name is not None else f"code:{int(value)}"
                info["unverifiedNames"].add(label)
                add("commandProtocolUnverified", "warning",
                    "A command's payload, responses, and state effects have not been reviewed.", location)
                # An unknown operation could alter resource state.
                activeResources.clear()
                continue

            info["reviewedNames"].add(name)
            if name in specialized:
                add("specializedResponseHandling", "info",
                    "This command returns command-specific data in addition to ordinary completion results.", location)

            chunks = commands["commandData"][index]
            creationType = None
            source = None
            filename = None
            useType = None
            resourceID = None
            delete = False

            try:
                if not chunks or len(chunks[0]) != 2:
                    raise ValueError("Missing command-code chunk.")
                if struct.unpack(prefix + "H", chunks[0])[0] != int(value):
                    raise ValueError("Command-code chunk mismatch.")
                payload = chunks[1:]

                if name == "mglFlush":
                    sizes(payload, [])
                elif name == "mglSetClearColor":
                    sizes(payload, [12])
                elif name == "mglSetXform":
                    sizes(payload, [64])
                elif name in ("mglDots", "mglLine", "mglQuad", "mglArcs"):
                    if len(payload) != 2: raise ValueError("Expected count and vertex data.")
                    columns = {"mglDots": 11, "mglLine": 6, "mglQuad": 6, "mglArcs": 14}[name]
                    sizes(payload, [4, uint(payload[0]) * columns * 4])
                elif name == "mglSetTextRenderingParameters":
                    if len(payload) != 11: raise ValueError("Incomplete text settings.")
                    string(payload[0])
                    sizes(payload[1:], [4] * 10)
                elif name == "mglCreateTextTexture":
                    if len(payload) != 1: raise ValueError("Expected text string.")
                    string(payload[0])
                    creationType, source = "texture", "serverText"
                elif name == "mglCreateTexture":
                    if len(payload) != 3: raise ValueError("Expected image dimensions and pixels.")
                    width, height = uint(payload[0]), uint(payload[1])
                    sizes(payload, [4, 4, width * height * 4 * 4])
                    creationType, source = "texture", "uploadedPixels"
                elif name == "mglBltTexture":
                    if len(payload) != 7: raise ValueError("Incomplete texture draw.")
                    sizes(payload, [4, 4, 4, 4, uint(payload[3]) * 5 * 4, 4, 4])
                    useType, resourceID = "texture", uint(payload[6])
                elif name == "mglDeleteTexture":
                    sizes(payload, [4])
                    useType, resourceID, delete = "texture", uint(payload[0]), True
                elif name == "mglMovieCreate":
                    if len(payload) != 1: raise ValueError("Expected movie filename.")
                    filename = string(payload[0])
                    creationType, source = "movie", "externalMovie"
                    add("externalMovieReference", "info",
                        "Movie content is referenced by filename, not embedded or checked for availability.",
                        {**location, "filename": filename})
                elif name == "mglMovieSetDisplayPosition":
                    if len(payload) != 3: raise ValueError("Incomplete movie placement.")
                    sizes(payload, [4, 4, uint(payload[1]) * 5 * 4])
                    useType, resourceID = "movie", uint(payload[0])
                elif name in movieNames:
                    sizes(payload, [4])
                    useType, resourceID = "movie", uint(payload[0])
                    delete = name == "mglMovieDelete"

            except (ValueError, UnicodeError, struct.error) as error:
                add("commandPayloadIncomplete", "warning",
                    "A reviewed command has a missing, malformed, or unexpected payload.",
                    {**location, "detail": str(error)})
                activeResources.clear()
                continue

            if useType is not None:
                key = (useType, resourceID)
                if key not in activeResources:
                    add("resourceCreationUnresolved", "warning",
                        "A referenced ID has no established creation mapping in the reviewed continuous sequence; "
                        "it may predate capture, lack a returned-ID record, or follow a gap/deletion/unverified command.",
                        {**location, "resourceType": useType, "originalID": resourceID})
                if delete: activeResources.pop(key, None)

            records = resourcesByCommand.get(index, [])
            if creationType is None:
                if records:
                    add("resourceOnNonCreationCommand", "warning",
                        "Resource metadata is attached to a reviewed non-creation command.", location)
                continue

            matching = [
                record for record in records
                if record.get("resourceType") == creationType
                and isinstance(record.get("originalID"), (int, np.integer))
                and not isinstance(record.get("originalID"), (bool, np.bool_))
                and 0 <= record["originalID"] <= 0xFFFFFFFF
            ]
            if len(records) != 1 or len(matching) != 1:
                add("creationIdMissingOrAmbiguous", "warning",
                    "A creation command does not have exactly one valid returned-resource ID record.", location)
                continue

            record = matching[0]
            resourceMetadata = record.get("metadata") or {}
            if resourceMetadata.get("source") != source:
                add("resourceSourceMismatch", "warning",
                    "Resource source metadata differs from the creation-command type.", location)
            if creationType == "movie" and resourceMetadata.get("filename") != filename:
                add("movieFilenameMismatch", "warning",
                    "The resource filename differs from the filename sent in the creation command.", location)

            key = (creationType, int(record["originalID"]))
            if key in activeResources:
                add("resourceIdReusedWithoutDeletion", "warning",
                    "A creation returns an ID that already has a live recorded mapping.", location)
            activeResources[key] = index

        return info

    ################################################################
    # Analyze movie source references and timing, not movie replay
    ################################################################
    @classmethod
    def _analyzeMovies(cls, recording, commandInfo, usableTime, add):
        """Check retained movie timing without forward-filling source frames."""
        def sourceReference(metadata, location):
            filename = metadata.get("filename")
            if not isinstance(filename, str) or not filename:
                add("movieFilenameMissing", "warning", "Movie source filename is absent or empty.", location)
            else:
                add("externalMovieReference", "info",
                    "Movie content is referenced by filename, not embedded or checked for availability.",
                    {**location, "filename": filename})

        def sourceTime(value, location):
            if value is None:
                add("movieSourceTimeInvalid", "warning", "Movie source time is missing or invalid.", location)
                return False
            array = np.asarray(value)
            valid = bool(array.ndim == 0 and array.dtype.kind in "iuf"
                         and np.isfinite(array) and (array == -1 or array >= 0))
            if not valid:
                add("movieSourceTimeInvalid", "warning", "Movie source time is missing or invalid.", location)
                return False
            return bool(array >= 0)

        linkedPlaybacks = set()
        for index, playback in enumerate(recording.get("moviePlaybacks") or []):
            location = {"section": "moviePlaybacks", "index": index}
            metadata = playback.get("metadata") or {}
            sourceReference(metadata, location)
            commandIndex = metadata.get("commandIndex", -1)

            if commandIndex != -1:
                validLink = (
                    isinstance(commandIndex, (int, np.integer))
                    and not isinstance(commandIndex, (bool, np.bool_))
                    and 0 <= commandIndex < len(commandInfo["names"])
                    and commandInfo["names"][commandIndex] == "mglMoviePlay"
                )
                if not validLink:
                    add("moviePlaybackCommandLinkInvalid", "warning",
                        "Playback metadata does not reference a reviewed mglMoviePlay command.", location)
                else:
                    linkedPlaybacks.add(int(commandIndex))

            arrays = {}
            for name in cls.playbackArrayNames:
                array = playback.get(name)
                if not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype.kind not in "iuf":
                    add("moviePlaybackArrayMissing", "warning",
                        "A movie playback timing array is absent or structurally invalid.",
                        {**location, "field": name})
                    arrays[name] = None
                else:
                    arrays[name] = array

            lengths = [len(array) for array in arrays.values() if array is not None]
            if len(set(lengths)) > 1:
                add("moviePlaybackLengthsDiffer", "warning",
                    "Movie playback timing arrays have different lengths; they were not repaired.", location)
            if lengths and all(length == 0 for length in lengths):
                add("moviePlaybackEmpty", "info", "Movie playback contains no timing samples.", location)

            presented = arrays["presentedTimes"]
            if presented is not None:
                for sampleIndex, time in enumerate(presented):
                    if not usableTime(time):
                        add("moviePresentationTimeUnusable", "warning",
                            "Movie presentation time is missing, nonfinite, or nonpositive.",
                            {**location, "sampleIndex": sampleIndex})

            movieTimes = arrays["movieTimes"]
            if movieTimes is not None:
                for sampleIndex, time in enumerate(movieTimes):
                    sampleLocation = {**location, "sampleIndex": sampleIndex}
                    update = sourceTime(time, sampleLocation)
                    if update and (presented is None or sampleIndex >= len(presented)
                                   or not usableTime(presented[sampleIndex])):
                        add("movieUpdateWithoutPresentationTime", "warning",
                            "A new source-frame time lacks usable associated presentation timing; "
                            "do not automatically advance displayed content.", sampleLocation)

        for index, name in enumerate(commandInfo["names"]):
            if name == "mglMoviePlay" and index not in linkedPlaybacks:
                add("moviePlayWithoutTimingRecord", "warning",
                    "A captured movie-play command has no linked retained playback record.",
                    {"section": "commands", "index": index})

        draws = recording.get("movieDraws") or {}
        sources = draws.get("sources") or []
        for index, source in enumerate(sources):
            sourceReference(source, {"section": "movieDrawSources", "index": index})

        groupCount = draws.get("flushGroupCount", 0)
        for index, sample in enumerate(draws.get("samples") or []):
            location = {"section": "movieDraws", "index": index}
            sourceIndex = sample.get("sourceIndex")
            if (not isinstance(sourceIndex, (int, np.integer))
                    or isinstance(sourceIndex, (bool, np.bool_))
                    or not 0 <= sourceIndex < len(sources)):
                add("movieDrawSourceInvalid", "warning", "Movie draw source index is invalid.", location)

            status = sample.get("flushStatus")
            group = sample.get("flushGroupIndex")
            validGroup = (
                isinstance(group, (int, np.integer)) and not isinstance(group, (bool, np.bool_))
                and 0 <= group < groupCount
            )
            if not validGroup or status != "resultsReceived":
                add("movieDrawFlushUnresolved", "warning",
                    "Movie draw has no completed, valid flush association.", location)

            timingRecorded = bool(sample.get("flushTimingRecorded", False))
            presented = sample.get("drawablePresented")
            if not timingRecorded:
                add("movieDrawTimingNotRecorded", "info",
                    "Movie draw has no retained flush presentation timing.", location)
            elif not usableTime(presented):
                add("moviePresentationTimeUnusable", "warning",
                    "Movie presentation time is missing, nonfinite, or nonpositive.", location)

            success = sample.get("flushSuccess")
            if timingRecorded and (success is None or not np.isfinite(success) or success != 1):
                add("movieDrawFlushSuccessUnconfirmed", "warning",
                    "The associated flush does not have a retained success=1 result.", location)

            update = sourceTime(sample.get("movieTime"), location)
            if update and not (validGroup and status == "resultsReceived"
                               and timingRecorded and usableTime(presented)):
                add("movieUpdateWithoutPresentationTime", "warning",
                    "A new source-frame time lacks usable associated presentation timing; "
                    "do not automatically advance displayed content.", location)

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
    
    ################################################################
    # Select drawing frames using original reported presentation timing
    ################################################################
    @classmethod
    def selectFrames(cls, recording, time=None, startTime=None, stopTime=None):
        """Select candidate drawing frames for a time or half-open range.

        Specify either time, or both startTime and stopTime. Times use the
        original recording clock; no offsets or timestamp repairs are applied.

        Frame indices count recorded mglFlush commands, matching the drawing
        reconstruction API. This method does not execute or reconstruct them.

        Selection uses a reported-timing model:
        - A timestamp group supplies candidate content until the next group.
        - Equal timestamps do not establish which frame was displayed.
        - Untimed flushes are conservatively included between neighboring
          reported boundaries, without using command submission times.
        - Recording stop bounds the query, not the final image's visibility.

        Duplicate groups additionally retain the preceding timestamp group's
        frames as possible carry-over content. This is deliberately conservative,
        not a model of the renderer's actual duplicate-timestamp behavior.

        Returns candidate indices, selected intervals/groups, and ambiguities.
        Absence of detected ambiguity is not independent presentation validation.
        """
        if not isinstance(recording, dict):
            raise TypeError("recording must be a dictionary.")

        def finiteScalar(value, label):
            array = np.asarray(value)
            if (array.ndim != 0 or array.dtype.kind not in "iuf"
                    or not np.isfinite(array)):
                raise ValueError(f"{label} must be a finite numeric scalar.")
            return float(array)

        # Validate query form before inspecting any renderer data.
        if time is not None:
            if startTime is not None or stopTime is not None:
                raise ValueError("Specify time OR startTime and stopTime.")
            queryTime = finiteScalar(time, "time")
            query = {"type": "time", "time": queryTime}
        else:
            if startTime is None or stopTime is None:
                raise ValueError("Specify time OR both startTime and stopTime.")
            queryStart = finiteScalar(startTime, "startTime")
            queryStop = finiteScalar(stopTime, "stopTime")
            if queryStop <= queryStart:
                raise ValueError("stopTime must be greater than startTime.")
            query = {
                "type": "range",
                "startTime": queryStart,
                "stopTime": queryStop,
            }

        metadata = recording.get("metadata")
        commands = recording.get("commands")
        presentations = recording.get("presentations")

        if not isinstance(metadata, dict):
            raise ValueError("Session metadata is required for time selection.")
        if commands is None:
            raise ValueError("Drawing time selection requires recorded commands.")
        if metadata.get("outcome") == "active":
            raise ValueError("Stop recording before time selection.")

        sessionStart = finiteScalar(metadata.get("startTime"), "session startTime")
        sessionStop = finiteScalar(metadata.get("stopTime"), "session stopTime")
        if sessionStop <= sessionStart:
            raise ValueError("Session stopTime must follow startTime.")

        if query["type"] == "time":
            if not sessionStart <= queryTime < sessionStop:
                raise ValueError("Requested time is outside [session start, session stop).")
        elif queryStart < sessionStart or queryStop > sessionStop:
            raise ValueError("Requested range extends outside the recorded session.")

        cls.validate(commands, presentations)

        segments = commands.get("segments") or []
        if (len(segments) != 1
                or segments[0].get("commandStartIndex") != 0
                or segments[0].get("commandStopIndex") != commands["count"]):
            raise ValueError("Time selection currently requires one continuous command segment.")

        mapping = (commands.get("metadata") or {}).get("commandValues") or {}
        if "mglFlush" not in mapping:
            raise ValueError("The saved command mapping has no mglFlush entry.")

        # Movies have a separate source-frame/timing interpretation. Do not
        # accidentally equate a movie playback with an ordinary flush frame.
        movieCodes = {
            int(value) for name, value in mapping.items()
            if name.startswith("mglMovie")
        }
        draws = recording.get("movieDraws") or {}
        if (recording.get("moviePlaybacks") or draws.get("samples")
                or any(int(value) in movieCodes for value in commands["commandValues"])):
            raise ValueError("This drawing selector does not yet support movie-containing recordings.")

        flushCode = int(mapping["mglFlush"])
        flushCommands = [
            int(index) for index in
            np.flatnonzero(commands["commandValues"] == flushCode)
        ]
        if not flushCommands:
            raise ValueError("Recording contains no flush frames to select.")

        commandToFrame = {
            commandIndex: frameIndex
            for frameIndex, commandIndex in enumerate(flushCommands)
        }
        rowByFrame = {}

        if presentations is not None:
            for row in range(presentations["count"]):
                commandIndex = int(presentations["commandIndices"][row])

                # An unlinked presentation means the captured command stream
                # does not cover all available presentation evidence.
                if commandIndex == -1:
                    raise ValueError(
                        "Unlinked presentation rows prevent drawing-only time selection; "
                        "use frame-index reconstruction instead."
                    )
                if commandIndex not in commandToFrame:
                    raise ValueError("A presentation row references a non-flush command.")

                frameIndex = commandToFrame[commandIndex]
                if frameIndex in rowByFrame:
                    raise ValueError("Multiple presentation rows reference the same flush.")

                returnedCode = presentations["commandCode"][row]
                if np.isfinite(returnedCode) and returnedCode != flushCode:
                    raise ValueError("Presentation command code differs from mglFlush.")
                rowByFrame[frameIndex] = row

        # Read raw values in command order, not presentation-row order.
        frameRecords = []
        groups = []
        untimedFrames = []

        for frameIndex, commandIndex in enumerate(flushCommands):
            row = rowByFrame.get(frameIndex)
            presented = None if row is None else float(presentations["drawablePresented"][row])
            success = None if row is None else float(presentations["success"][row])
            usable = (
                presented is not None and np.isfinite(presented) and presented > 0
                and success is not None and np.isfinite(success) and success == 1
            )

            frameRecords.append({
                "frameIndex": frameIndex,
                "commandIndex": commandIndex,
                "presentationIndex": row,
                "drawablePresented": presented,
                "success": success,
                "timingUsable": bool(usable),
            })

            if not usable:
                untimedFrames.append(frameIndex)
                continue

            if not sessionStart <= presented < sessionStop:
                raise ValueError(
                    f"Frame {frameIndex} has a presentation time outside session bounds."
                )
            if groups and presented < groups[-1]["time"]:
                raise ValueError(
                    "Presentation times move backward in command order; "
                    "use frame-index reconstruction instead."
                )

            if groups and presented == groups[-1]["time"]:
                groups[-1]["frameIndices"].append(frameIndex)
            else:
                groups.append({
                    "groupIndex": len(groups),
                    "time": presented,
                    "frameIndices": [frameIndex],
                })

        # Describe intervals conservatively. Extra candidate indices do not
        # imply that those images actually became visible.
        intervals = []

        def addInterval(start, stop, frameIndices, groupIndex, issues):
            if stop <= start: return
            intervals.append({
                "startTime": start,
                "stopTime": stop,
                "frameIndices": set(frameIndices),
                "groupIndex": groupIndex,
                "issues": list(issues),
            })

        firstTime = groups[0]["time"] if groups else sessionStop
        addInterval(sessionStart, firstTime, [], None, ["initialContentUnknown"])

        for groupIndex, group in enumerate(groups):
            stop = groups[groupIndex + 1]["time"] if groupIndex + 1 < len(groups) else sessionStop
            candidates = list(group["frameIndices"])
            issues = []

            if len(group["frameIndices"]) > 1:
                issues.append("duplicatePresentationTimestamp")
                if groupIndex > 0:
                    candidates.extend(groups[groupIndex - 1]["frameIndices"])
                else:
                    issues.append("precedingContentUnknown")

            addInterval(group["time"], stop, candidates, groupIndex, issues)

        # Locate untimed commands relative to timed command groups. These
        # bounds use the reported-order model, not submission-time estimates.
        invalidRegions = []

        if not groups:
            invalidRegions.append({
                "startTime": sessionStart,
                "stopTime": sessionStop,
                "frameIndices": list(untimedFrames),
            })
        else:
            validFrameIndices = [
                index for index, frame in enumerate(frameRecords) if frame["timingUsable"]
            ]
            groupByFrame = {
                frameIndex: group["groupIndex"]
                for group in groups for frameIndex in group["frameIndices"]
            }

            for frameIndex in untimedFrames:
                position = int(np.searchsorted(validFrameIndices, frameIndex))
                previousFrame = validFrameIndices[position - 1] if position else None
                nextFrame = (
                    validFrameIndices[position]
                    if position < len(validFrameIndices) else None
                )

                start = (
                    sessionStart if previousFrame is None
                    else frameRecords[previousFrame]["drawablePresented"]
                )
                stop = (
                    sessionStop if nextFrame is None
                    else frameRecords[nextFrame]["drawablePresented"]
                )

                # If an untimed flush falls between equal reported times,
                # mark that group's following interval as ambiguous as well.
                if stop <= start and nextFrame is not None:
                    groupIndex = groupByFrame[nextFrame]
                    stop = (
                        groups[groupIndex + 1]["time"]
                        if groupIndex + 1 < len(groups) else sessionStop
                    )

                invalidRegions.append({
                    "startTime": start,
                    "stopTime": stop,
                    "frameIndices": [frameIndex],
                })

        for region in invalidRegions:
            for interval in intervals:
                if (interval["startTime"] < region["stopTime"]
                        and region["startTime"] < interval["stopTime"]):
                    interval["frameIndices"].update(region["frameIndices"])
                    if "untimedFlush" not in interval["issues"]:
                        interval["issues"].append("untimedFlush")

        def overlapsQuery(start, stop):
            if query["type"] == "time":
                return start <= queryTime < stop
            return start < queryStop and queryStart < stop

        selectedIntervals = []
        selectedIndices = set()
        selectedGroupIndices = set()
        ambiguityCodes = set()

        for interval in intervals:
            if not overlapsQuery(interval["startTime"], interval["stopTime"]): continue

            entry = {
                **interval,
                "frameIndices": sorted(interval["frameIndices"]),
            }
            selectedIntervals.append(entry)
            selectedIndices.update(entry["frameIndices"])
            ambiguityCodes.update(entry["issues"])
            if entry["groupIndex"] is not None:
                selectedGroupIndices.add(entry["groupIndex"])

        # Include timing groups for conservative carried-in candidates too.
        for group in groups:
            if any(index in selectedIndices for index in group["frameIndices"]):
                selectedGroupIndices.add(group["groupIndex"])

        return {
            "selectionVersion": 1,
            "recordingId": metadata.get("recordingId"),
            "query": query,
            "frameIndices": sorted(selectedIndices),
            "frames": [frameRecords[index] for index in sorted(selectedIndices)],
            "timingGroups": [
                group for group in groups
                if group["groupIndex"] in selectedGroupIndices
            ],
            "intervals": selectedIntervals,
            "ambiguous": bool(ambiguityCodes),
            "ambiguities": sorted(ambiguityCodes),
            "sessionBounds": {
                "startTime": sessionStart,
                "stopTime": sessionStop,
            },
            "selectionBasis": "reportedPresentationTimingWithConservativeCandidates",
            "physicalVisibilityVerified": False,
            "limitations": [
                "Candidate frames are not confirmed physical presentations.",
                "Reported timestamp order is assumed to describe presentation order.",
                "Repeated or stale renderer timestamps can invalidate that assumption.",
                "Untimed flush bounds are inferred from neighboring reported groups, not measured.",
                "The final interval is bounded by capture stop, not image disappearance.",
                "Pre-capture content is unknown.",
                "No command timestamps replace missing presentation times.",
            ],
        }
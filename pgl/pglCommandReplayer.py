################################################################
#   filename: pglCommandReplayer.py
#    purpose: Class that implements storing a copy of all pgl
#             commands so that they can be replayed later. THis
#             is useful for recreating the stimulus sequence so
#             that it can be used in model fitting such as pRF models.
#         by: JLG
#       date: August 20, 2025
################################################################

#############
# Import modules
#############
import numpy as np

#############
# pglCommandReplayer class
#############
class pglCommandReplayer:
    '''
    Class for recording and replaying pgl commands.

    '''
    def __init__(self):
        '''
        Initialize command recording and independent flush recording.
        '''
        # Command buffers are allocated when recording first starts.
        self.commandRecording = False
        self.logSize = 0
        self.logEntryIndex = 0
        self.commandLogGrowthCount = 0
        self.commandLog = np.empty(0, dtype=np.uint16)
        self.commandLogTimes = np.empty(0, dtype=np.float64)
        self.commandLogData = np.empty(0, dtype=object)

        # Segments use half-open command ranges: [start, stop).
        self.commandSegments = []
        self.commandResources = []
        self.commandRecordingMetadata = None
        self._activeCommandSegment = None

        # Flush recording is independent of command recording.
        self.flushRecording = False
        self._presentationLog = None

    ################################################################
    # Start or resume command recording
    ################################################################
    def commandRecord(self, estimatedDuration=600, commandsPerFrame=20):
        """Start a segment, allocating buffers on the first call.

        estimatedDuration is in seconds, not a recording limit.
        Later segments append to the same recording without reallocating
        from the estimate or discarding previously recorded commands.
        """
        import math
        from copy import deepcopy

        if self.commandRecording: return
        if not self.isOpen(): raise RuntimeError("Open the screen before recording commands.")
        if getattr(self, "_batchState", 0) != 0 or getattr(self, "_profileMode", 0) < 0:
            raise RuntimeError("Command recording during batch mode is not supported yet.")

        if self.commandRecordingMetadata is None:
            if isinstance(estimatedDuration, (bool, np.bool_)):
                raise ValueError("estimatedDuration must be a positive number of seconds.")

            estimatedDuration = float(estimatedDuration)
            frameRate = float(self.frameRate)

            if not math.isfinite(estimatedDuration) or estimatedDuration <= 0:
                raise ValueError("estimatedDuration must be finite and positive.")
            if not math.isfinite(frameRate) or frameRate <= 0:
                raise ValueError("The cached display frame rate must be finite and positive.")
            if isinstance(commandsPerFrame, (bool, np.bool_)) or not isinstance(commandsPerFrame, (int, np.integer)) or commandsPerFrame <= 0:
                raise ValueError("commandsPerFrame must be a positive integer.")

            estimatedFrames = estimatedDuration * frameRate
            if not math.isfinite(estimatedFrames): raise ValueError("Estimated capacity is too large.")
            capacity = max(1, math.ceil(estimatedFrames)) * int(commandsPerFrame)

            # Allocate locally before replacing recorder state.
            commandLog = np.empty(capacity, dtype=np.uint16)
            commandLogTimes = np.empty(capacity, dtype=np.float64)
            commandLogData = np.full(capacity, None, dtype=object)

            metadata = {
                "estimatedDuration": estimatedDuration,
                "allocationFrameRate": frameRate,
                "commandsPerFrame": int(commandsPerFrame),
                "timestampUnits": "seconds",
                "timestampMeaning": "clientCommandLoggingTime",
                "commandValues": {name: int(value) for name, value in self.s.commandValues.items()},
            }

            self.commandLog = commandLog
            self.commandLogTimes = commandLogTimes
            self.commandLogData = commandLogData
            self.logSize = capacity
            self.commandRecordingMetadata = metadata

        # This copies cached Python-side settings; it does not query or change
        # the renderer. Preserve their provenance rather than claiming that
        # the settings were independently verified on the server.
        segment = {
            "startTime": float(self.getSecs()),
            "stopTime": None,
            "commandStartIndex": self.logEntryIndex,
            "commandStopIndex": None,
            "textRenderingParameters": deepcopy(getattr(self, "textRenderingParameters", None)),
            "textSettingsSource": "clientCache",
        }

        self.commandSegments.append(segment)
        self._activeCommandSegment = segment
        self.commandRecording = True

    ################################################################
    # Record command value and client time
    ################################################################
    def logCommandValue(self, commandValue):
        """Called by _pglComm before sending the command-code bytes."""
        if self.logEntryIndex >= self.logSize:
            # Growth is a fallback. Do not print during presentation.
            capacity = max(1, self.logSize * 2)
            commandLog = np.empty(capacity, dtype=np.uint16)
            commandLogTimes = np.empty(capacity, dtype=np.float64)
            commandLogData = np.full(capacity, None, dtype=object)

            commandLog[:self.logEntryIndex] = self.commandLog[:self.logEntryIndex]
            commandLogTimes[:self.logEntryIndex] = self.commandLogTimes[:self.logEntryIndex]
            commandLogData[:self.logEntryIndex] = self.commandLogData[:self.logEntryIndex]

            self.commandLog = commandLog
            self.commandLogTimes = commandLogTimes
            self.commandLogData = commandLogData
            self.logSize = capacity
            self.commandLogGrowthCount += 1

        # This is client logging time, not presentation time.
        index = self.logEntryIndex
        self.commandLogTimes[index] = self.getSecs()
        self.commandLog[index] = commandValue
        self.logEntryIndex = index + 1

    ################################################################
    # Retain already-packed payload bytes
    ################################################################
    def logCommandData(self, commandData):
        """Retain immutable bytes already created by _pglComm.write()."""
        segment = self._activeCommandSegment
        if segment is None or self.logEntryIndex <= segment["commandStartIndex"]:
            raise RuntimeError("Command payload received without a command in the active segment.")

        index = self.logEntryIndex - 1
        if self.commandLogData[index] is None: self.commandLogData[index] = []

        # Keep the existing bytes object; do not duplicate its contents.
        self.commandLogData[index].append(commandData)

    ################################################################
    # Get the current recorded command index
    ################################################################
    def commandRecordIndex(self):
        """Return the latest command index in the active segment, or -1.

        Call immediately after writeCommand(), before issuing another command.
        """
        if not self.commandRecording: return -1
        segment = self._activeCommandSegment
        if segment is None or self.logEntryIndex <= segment["commandStartIndex"]: return -1
        return self.logEntryIndex - 1

    ################################################################
    # Retain a resource ID returned by a creation command
    ################################################################
    def commandRecordResource(self, commandIndex, resourceType, originalID, **metadata):
        """Associate a returned resource ID with its recorded creation command.

        This records information absent from the outgoing command stream.
        Resource classification and replay mapping happen later.
        """
        if commandIndex == -1: return
        if not 0 <= commandIndex < self.logEntryIndex: raise ValueError("Invalid resource creation command index.")

        self.commandResources.append({
            "commandIndex": int(commandIndex),
            "resourceType": resourceType,
            "originalID": int(originalID),
            "metadata": metadata,
        })

    ################################################################
    # Stop the current command segment
    ################################################################
    def commandRecordStop(self):
        """Stop command capture without discarding previous segments."""
        if not self.commandRecording: return

        # Disable first so an error obtaining the boundary time does not
        # leave capture running during recovery.
        self.commandRecording = False
        segment = self._activeCommandSegment
        self._activeCommandSegment = None
        segment["commandStopIndex"] = self.logEntryIndex
        segment["stopTime"] = float(self.getSecs())

    ################################################################
    # Retrieve recorded commands
    ################################################################
    def commandRecordData(self):
        """Return populated data after capture stops.

        Arrays and mutable containers are copied. Immutable payload bytes
        remain shared safely, avoiding another copy of all payload contents.
        """
        from copy import deepcopy

        if self.commandRecording: raise RuntimeError("Stop command recording before retrieving its data.")
        if self.commandRecordingMetadata is None: return None

        count = self.logEntryIndex
        return {
            "commandValues": self.commandLog[:count].copy(),
            "commandTimes": self.commandLogTimes[:count].copy(),
            "commandData": [None if chunks is None else list(chunks) for chunks in self.commandLogData[:count]],
            "segments": deepcopy(self.commandSegments),
            "resources": deepcopy(self.commandResources),
            "metadata": deepcopy(self.commandRecordingMetadata),
            "count": count,
            "capacity": self.logSize,
            "growthCount": self.commandLogGrowthCount,
        }

    ################################################################
    # Explicitly discard command data
    ################################################################
    def commandRecordClear(self):
        """Release command data without leaving dangling flush associations."""
        if self.commandRecording: raise RuntimeError("Stop command recording before clearing its data.")

        # Flush records may reference these command indices. Clear/export the
        # presentation recording first rather than silently invalidate links.
        if self._presentationLog is not None:
            raise RuntimeError("Retrieve/save and clear the flush log before clearing command data.")

        self.logSize = 0
        self.logEntryIndex = 0
        self.commandLogGrowthCount = 0
        self.commandLog = np.empty(0, dtype=np.uint16)
        self.commandLogTimes = np.empty(0, dtype=np.float64)
        self.commandLogData = np.empty(0, dtype=object)
        self.commandSegments = []
        self.commandResources = []
        self.commandRecordingMetadata = None
        self._activeCommandSegment = None

    def commandList(self):
        '''
        List all recorded pgl commands.
        '''
        for iCommand in range(self.logEntryIndex):
            commandValue = self.commandLog[iCommand]
            commandName = self.s.getCommandName(commandValue)
            print(f"Command {iCommand}: {commandName} (Value: {commandValue}, Data length: {len(self.commandLogData[iCommand]) if self.commandLogData[iCommand] is not None else 0})")

    def commandReplay(self, frameGrab=False):
        '''
        Replay all recorded pgl commands.
        '''
        # if we are going to frame grab
        mglFlushCommandValue = self.s.getCommandValue("mglFlush")
        if frameGrab:
            # init frame grab mode (which makes an offscreen context)
            self.frameGrabInit()
            # count the number of mglFlush commands
            mglFlushCount = np.sum(self.commandLog == mglFlushCommandValue)
            frames = np.zeros((mglFlushCount, self.screenHeight.pix, self.screenWidth.pix, 4), dtype=np.float32)
        iFrame = 0

        for iCommand in range(self.logEntryIndex):
            # display commands in verbose mode
            if self.verbose>1: print(f"(pglCommandReplayer) Replaying command {iCommand}: {self.s.getCommandName(self.commandLog[iCommand])} ")
            # replay the command (just need to pass command data, which includes
            # the command value and any associated data)
            self.s.replayCommand(self.commandLogData[iCommand])
            # if the command was flush, and we are frame grabbing, grab the frame
            if frameGrab and self.commandLog[iCommand] == mglFlushCommandValue:
                # grab the frame
                frames[iFrame,:,:,:] = self.frameGrab()
                iFrame += 1      

        # if we are frame grabbing, return the frames
        if frameGrab:
            # end the frame grab mode
            self.frameGrabEnd()
            return frames
        else:
            return None
        
    ################################################################
    # Start flush presentation recording
    ################################################################
    def flushRecord(self, estimatedDuration=600):
        """Start a new flush log using estimated duration in seconds.

        Uses the cached display frame rate; does not query the renderer.
        A stopped log must be explicitly cleared before starting another.
        """
        from .pglPresentationLog import pglPresentationLog

        if self.flushRecording: raise RuntimeError("Flush recording is already active.")
        if self._presentationLog is not None: raise RuntimeError("A flush log already exists. Retrieve/save it, then call flushRecordClear().")
        if getattr(self, "_batchState", 0) != 0 or getattr(self, "_profileMode", 0) < 0:
            raise RuntimeError("Flush recording during batch mode is not implemented yet.")
        if not self.isOpen(): raise RuntimeError("Open the screen before starting flush recording.")

        # Construct first so validation/allocation failure leaves recording off.
        presentationLog = pglPresentationLog(estimatedDuration, self.frameRate)
        self._presentationLog = presentationLog
        self.flushRecording = True

    ################################################################
    # Stop flush presentation recording
    ################################################################
    def flushRecordStop(self):
        """Stop capture while retaining all recorded results."""
        self.flushRecording = False

    ################################################################
    # Retrieve recorded flush data
    ################################################################
    def flushRecordData(self):
        """Return populated data after stopping capture, or None if no log exists."""
        if self.flushRecording: raise RuntimeError("Stop flush recording before retrieving its data.")
        if self._presentationLog is None: return None
        return self._presentationLog.getData()

    ################################################################
    # Explicitly discard a stopped flush recording
    ################################################################
    def flushRecordClear(self):
        """Release a stopped recording; never silently discard active data."""
        if self.flushRecording: raise RuntimeError("Stop flush recording before clearing its data.")
        self._presentationLog = None
        
    ################################################################
    # Retain one renderer-controlled movie playback
    ################################################################
    def movieRecordPlayback(self, metadata, presentedTimes, movieTimes, targetPresentationTimestamps, drawFrameTimes, commandResults):
        """Retain playback timing during an active recording session.

        Before the session API is used, preserve standalone capture behavior.
        Once a session is stopped/cleared, require recordingStart() to resume.
        """
        from copy import deepcopy

        state = getattr(self, "_recordingState", None)
        if state is not None and state != "active": return None

        # Retain on PGL, not the movie object, so another play() or movie
        # deletion cannot discard earlier playback timing.
        if not hasattr(self, "_moviePlaybacks"): self._moviePlaybacks = []

        record = {
            "playbackIndex": len(self._moviePlaybacks),
            "metadata": deepcopy(metadata),
            "presentedTimes": presentedTimes,
            "movieTimes": movieTimes,
            "targetPresentationTimestamps": targetPresentationTimestamps,
            "drawFrameTimes": drawFrameTimes,
            "commandResults": commandResults,
        }
        self._moviePlaybacks.append(record)
        return record["playbackIndex"]
    
    ################################################################
    # Retrieve movie playback records
    ################################################################
    def movieRecordData(self):
        """Return independent copies of retained playback records."""
        from copy import deepcopy
        return deepcopy(getattr(self, "_moviePlaybacks", []))

    ################################################################
    # Explicitly discard movie playback records
    ################################################################
    def movieRecordClear(self):
        """Release retained playback data after exporting or saving it."""
        self._moviePlaybacks = []
        
    ################################################################
    # Retain one task-controlled movie draw
    ################################################################
    def movieRecordDraw(self, movie, drawFrameTime, targetPresentationTimestamp, movieTime):
        """Retain a movie draw until its next explicit flush association.

        Metadata is copied when placement changes or a new collection begins.
        No movie object or image pixels are retained in the recording.
        """
        from copy import deepcopy

        state = getattr(self, "_recordingState", None)
        if state is not None and state != "active": return None

        if not hasattr(self, "_movieDrawSamples"):
            self._movieDrawSamples = []
            self._movieDrawSources = []
            self._movieDrawPending = []
            self._movieDrawFlushCount = 0

        positionParams = getattr(movie, "positionParams", None)
        context = getattr(movie, "_movieDrawRecordContext", None)

        # setDisplayPosition() replaces this dictionary. Use that identity to
        # detect placement updates without copying metadata on every frame.
        if context is None or context[0] is not self._movieDrawSources or context[2] is not positionParams:
            def scalar(value):
                return value.item() if isinstance(value, (np.ndarray, np.generic)) and np.size(value) == 1 else value

            transform = getattr(movie, "preferredTransform", None)
            metadata = {
                "filename": movie.filename,
                "originalMovieID": int(movie.movieNum),
                "widthPixels": scalar(getattr(movie, "width", None)),
                "heightPixels": scalar(getattr(movie, "height", None)),
                "sourceFrameRate": scalar(getattr(movie, "frameRate", None)),
                "duration": scalar(getattr(movie, "duration", None)),
                "positionParameters": deepcopy(positionParams),
                "preferredTransform": None if transform is None else {
                    name: scalar(getattr(transform, name))
                    for name in ("a", "b", "c", "d", "tx", "ty")
                },
                "hostTimestampUnits": "seconds",
                "hostClockBasis": "assumedSameAsGetSecs",
                "movieTimeUnits": "seconds",
                "movieTimeBasis": "sourceMovie",
                "movieTimeMinusOneMeaning": "noNewSourceFrame",
            }

            sourceIndex = len(self._movieDrawSources)
            self._movieDrawSources.append(metadata)
            movie._movieDrawRecordContext = (self._movieDrawSources, sourceIndex, positionParams)
        else:
            sourceIndex = context[1]

        # Retain values already returned by the renderer. No extra timing
        # queries, socket operations, or image copies are performed.
        sample = {
            "sourceIndex": sourceIndex,
            "drawFrameTime": drawFrameTime,
            "targetPresentationTimestamp": targetPresentationTimestamp,
            "movieTime": movieTime,
            "flushGroupIndex": -1,
            "flushStatus": "pending",
            "flushTimingRecorded": False,
            "drawablePresented": None,
            "flushSuccess": None,
        }

        sampleIndex = len(self._movieDrawSamples)
        self._movieDrawSamples.append(sample)
        self._movieDrawPending.append(sampleIndex)
        return sampleIndex
    
    ################################################################
    # Associate pending movie draws with a flush attempt
    ################################################################
    def movieRecordFlush(self, commandResults, recordTiming, interrupted=False):
        """Associate pending draws without interpreting presentation validity.

        flushGroupIndex is local to this movie-draw collection. It identifies
        a shared flush attempt, not an index into the presentation-log array.
        """
        pending = getattr(self, "_movieDrawPending", None)
        if not pending: return

        groupIndex = self._movieDrawFlushCount
        self._movieDrawFlushCount += 1

        if interrupted:
            status = "interrupted"
        elif commandResults is None:
            status = "missingResults"
        else:
            status = "resultsReceived"

        # Capture raw timing only when flush logging is enabled.
        presentedTime = None
        success = None
        if recordTiming and commandResults is not None:
            presentedTime = commandResults.get("drawablePresented")
            success = commandResults.get("success")

        for sampleIndex in pending:
            sample = self._movieDrawSamples[sampleIndex]
            sample["flushGroupIndex"] = groupIndex
            sample["flushStatus"] = status
            sample["flushTimingRecorded"] = bool(recordTiming)
            sample["drawablePresented"] = presentedTime
            sample["flushSuccess"] = success

        # Consume the pending group even when timing is missing. It must not
        # accidentally attach to some later, unrelated presentation.
        pending.clear()

    ################################################################
    # Retrieve task-controlled movie drawing records
    ################################################################
    def movieDrawRecordData(self):
        """Return independent copies, including draws without a subsequent flush."""
        from copy import deepcopy

        return deepcopy({
            "sources": getattr(self, "_movieDrawSources", []),
            "samples": getattr(self, "_movieDrawSamples", []),
            "flushGroupCount": getattr(self, "_movieDrawFlushCount", 0),
        })

    ################################################################
    # Explicitly discard task-controlled movie drawing records
    ################################################################
    def movieDrawRecordClear(self):
        """Release incremental movie records without changing playback records.

        This explicitly discards any pending draws as well. Source contexts
        on movie objects are invalidated by replacing the source-list object.
        """
        self._movieDrawSamples = []
        self._movieDrawSources = []
        self._movieDrawPending = []
        self._movieDrawFlushCount = 0
        
    ################################################################
    # Start a run-level recording session
    ################################################################
    def recordingStart(self, estimatedDuration=600, recordFlush=True, recordCommands=False):
        """Start a fresh recording session without discarding existing data.

        estimatedDuration controls initial capacity, not recording duration.
        Movie timing is retained during the session independently of the
        command and flush recording options.
        """
        import math
        import uuid
        from copy import deepcopy

        if getattr(self, "_recordingState", None) in ("active", "stopped"):
            raise RuntimeError("A recording session already exists. Save/export it, then call recordingClear().")
        if not isinstance(recordFlush, bool) or not isinstance(recordCommands, bool):
            raise TypeError("recordFlush and recordCommands must be Boolean.")
        if isinstance(estimatedDuration, (bool, np.bool_)):
            raise ValueError("estimatedDuration must be a positive number of seconds.")

        estimatedDuration = float(estimatedDuration)
        frameRate = float(self.frameRate)
        if not math.isfinite(estimatedDuration) or estimatedDuration <= 0:
            raise ValueError("estimatedDuration must be finite and positive.")
        if not math.isfinite(frameRate) or frameRate <= 0:
            raise ValueError("The cached display frame rate must be finite and positive.")
        if not self.isOpen(): raise RuntimeError("Open the screen before starting recording.")
        if getattr(self, "_batchState", 0) != 0 or getattr(self, "_profileMode", 0) < 0:
            raise RuntimeError("Run-level recording during batch mode is not supported yet.")

        # Do not silently mix standalone recordings with a new session.
        if self.commandRecording or self.flushRecording:
            raise RuntimeError("Stop and export existing capture before starting a session.")
        if self.commandRecordingMetadata is not None or self._presentationLog is not None:
            raise RuntimeError("Export and clear existing command/flush data before starting a session.")
        if getattr(self, "_moviePlaybacks", []) or getattr(self, "_movieDrawSamples", []):
            raise RuntimeError("Export and clear existing movie data before starting a session.")

        # Snapshot cached display information only. No extra renderer queries.
        def snapshotDimensions(name):
            value = getattr(self, name, None)
            return {
                unit: deepcopy(getattr(value, unit))
                for unit in ("pix", "cm", "deg")
                if value is not None and hasattr(value, unit)
            }

        metadata = {
            "recordingId": str(uuid.uuid4()),
            "estimatedDuration": estimatedDuration,
            "allocationFrameRate": frameRate,
            "recordFlush": recordFlush,
            "recordCommandsInitially": recordCommands,
            "recordMovies": True,
            "startTime": float(self.getSecs()),
            "stopTime": None,
            "outcome": "active",
            "timestampUnits": "seconds",
            "clockBasis": "assumedSameAsGetSecs",
            "initialDisplay": {
                "screenWidth": snapshotDimensions("screenWidth"),
                "screenHeight": snapshotDimensions("screenHeight"),
                "distanceToScreen": snapshotDimensions("distanceToScreen"),
                "coordinateFrame": deepcopy(getattr(self, "coordinateFrame", None)),
                "clearScreenColor": deepcopy(getattr(self, "clearScreenColor", None)),
                "textRenderingParameters": deepcopy(getattr(self, "textRenderingParameters", None)),
            },
        }

        # Allocate before marking the session active. If allocation fails,
        # release only the new, still-empty buffers created by this call.
        try:
            if recordFlush: self.flushRecord(estimatedDuration=estimatedDuration)
            if recordCommands: self.commandRecord(estimatedDuration=estimatedDuration)
        except BaseException:
            self.flushRecordStop()
            self.commandRecordStop()
            self.flushRecordClear()
            self.commandRecordClear()
            raise

        # A new source-list identity invalidates cached movie source contexts.
        self.movieRecordClear()
        self.movieDrawRecordClear()
        self._recordingMetadata = metadata
        self._recordingState = "active"

    ################################################################
    # Stop run-level capture
    ################################################################
    def recordingStop(self, outcome="completed"):
        """Stop capture without discarding data or performing disk/device work.

        outcome describes acquisition, not whether saving or cleanup succeeds.
        Repeated calls leave the original stop boundary unchanged.
        """
        if outcome not in ("completed", "aborted", "error", "interrupted"):
            raise ValueError("Unknown recording outcome.")
        if getattr(self, "_recordingState", None) != "active": return

        # Disable the session first so subsequent movie operations cannot
        # append data during finalization or device cleanup.
        self._recordingState = "stopped"
        self._recordingMetadata["outcome"] = outcome
        self.flushRecordStop()

        try:
            self.commandRecordStop()
        finally:
            # Unflushed samples remain explicitly unassociated. Do not allow
            # a later cleanup flush to claim they were presented.
            pending = getattr(self, "_movieDrawPending", [])
            for sampleIndex in pending:
                self._movieDrawSamples[sampleIndex]["flushStatus"] = "recordingStoppedBeforeFlush"
            pending.clear()

            self._recordingMetadata["stopTime"] = float(self.getSecs())

    ################################################################
    # Export a stopped session
    ################################################################
    def recordingData(self):
        """Return independent recording data for analysis or saving.

        Call only after stopping capture. Export can copy substantial data,
        so it does not belong in a frame-critical loop.
        """
        from copy import deepcopy

        state = getattr(self, "_recordingState", None)
        if state == "active": raise RuntimeError("Stop recording before exporting it.")
        if state != "stopped": return None

        return {
            "metadata": deepcopy(self._recordingMetadata),
            "commands": self.commandRecordData(),
            "presentations": self.flushRecordData(),
            "moviePlaybacks": self.movieRecordData(),
            "movieDraws": self.movieDrawRecordData(),
        }

    ################################################################
    # Explicitly release a stopped session
    ################################################################
    def recordingClear(self):
        """Discard stopped recording data after saving/exporting it."""
        if getattr(self, "_recordingState", None) == "active":
            raise RuntimeError("Stop recording before clearing it.")
        if self.commandRecording or self.flushRecording:
            raise RuntimeError("Stop command and flush capture before clearing recording data.")

        # Clear presentation associations before releasing their commands.
        self.flushRecordClear()
        self.commandRecordClear()
        self.movieRecordClear()
        self.movieDrawRecordClear()
        self._recordingMetadata = None
        self._recordingState = "idle"
        
    ################################################################
    # Finalize and save the current recording session
    ################################################################
    def recordingSave(self, filename, outcome="completed"):
        """Stop capture, analyze, and save while retaining buffers for retry.

        A previously stopped session keeps its original outcome and stop time.
        Analysis is diagnostic: failure must not prevent raw recording saving.
        Saving does not establish that every command is supported by replay.
        """
        from .pglRecordingFile import pglRecordingFile
        from .pglMessages import pglMessages

        state = getattr(self, "_recordingState", None)
        if state not in ("active", "stopped"):
            raise RuntimeError("No recording session is available to save.")
        if outcome not in ("completed", "aborted", "error", "interrupted"):
            raise ValueError("Unknown recording outcome.")

        # Low-level capture must not have been restarted outside the session.
        if state == "stopped" and (self.commandRecording or self.flushRecording):
            raise RuntimeError("Capture was restarted after the session stopped.")

        if state == "active": self.recordingStop(outcome=outcome)

        # Export outside presentation. Exported metadata is independent of
        # the live session; analysis does not modify the captured buffers.
        data = self.recordingData()
        warningMessage = None

        try:
            report = dict(pglRecordingFile.analyze(data))
            report["status"] = "completed"

            # Check report serialization inside the analysis failure boundary.
            # An unserializable report must not block saving the raw capture.
            pglRecordingFile._encodeMetadata({"analysis": report}, ())

            warningFindings = [
                finding for finding in report["findings"]
                if finding["severity"] == "warning"
            ]
            if warningFindings:
                details = "; ".join(
                    f"{finding['code']}={finding['count']}"
                    for finding in warningFindings[:5]
                )
                if len(warningFindings) > 5:
                    details += f"; {len(warningFindings) - 5} additional categories"
                warningMessage = f"Recording analysis: {details}. See metadata['analysis'] for details."

        except Exception as error:
            # Preserve a small, serializable diagnostic instead of allowing
            # optional analysis to prevent essential raw recording storage.
            report = {
                "status": "failed",
                "replayCompleteness": "notEstablished",
                "error": {
                    "module": type(error).__module__,
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            warningMessage = (
                f"Recording analysis failed ({type(error).__name__}: {error}). "
                "Raw recording was saved with analysis failure metadata."
            )

        data["metadata"]["analysis"] = report

        # HDF5 failures still propagate normally. Buffers remain available
        # for retry, and no successful publication is reported prematurely.
        savedFilename = pglRecordingFile.save(filename, **data)

        # Diagnostic output must not turn a successful publication into a
        # reported save failure, including when warning handling is customized.
        if warningMessage:
            try:
                pglMessages.warning(warningMessage)
            except Exception:
                pass

        return savedFilename
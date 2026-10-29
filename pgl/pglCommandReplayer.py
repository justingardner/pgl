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
        """Stop capture and save raw recording data without consistency analysis.

        A previously stopped session keeps its original outcome and stop time.
        Recording analysis is deferred until explicitly requested during analysis.

        Save failures propagate, and buffers remain available for retry.
        Successful saving does not establish timing quality or replay support.
        """
        from .pglRecordingFile import pglRecordingFile

        # Require an available session and a recognized acquisition outcome.
        state = getattr(self, "_recordingState", None)
        if state not in ("active", "stopped"): raise RuntimeError("No recording session is available to save.")
        if outcome not in ("completed", "aborted", "error", "interrupted"): raise ValueError("Unknown recording outcome.")

        # Prevent exporting a stopped session whose low-level capture was
        # restarted independently of the recording-session lifecycle.
        if state == "stopped" and (self.commandRecording or self.flushRecording):
            raise RuntimeError("Capture was restarted after the session stopped.")

        # Stop active capture; leave an already stopped session unchanged.
        if state == "active": self.recordingStop(outcome=outcome)

        # Export raw capture outside presentation. Do not run analyze() here:
        # timing diagnostics and replay-completeness checks belong in analysis.
        data = self.recordingData()

        # Publish through the existing file writer, preserving its validation
        # and write-error handling. Do not clear buffers or hide save failures.
        return pglRecordingFile.save(filename, **data)
    
    ################################################################
    # Prepare drawing reconstruction without touching the renderer
    ################################################################
    @staticmethod
    def _recordingReconstructionPlan(recording, frameIndices=None):
        """Validate a continuous drawing recording and select flush outputs.

        Frame indices identify mglFlush commands in command order, not rows
        in the optional presentation log. Presentation rows are linked through
        their recorded commandIndices.

        Earlier commands remain in the plan even when only later frames are
        requested. Selection controls image readback, not state reconstruction.

        This first implementation supports basic drawing, including arcs, and
        server-generated text. Movies, uploaded image creation, capture gaps,
        and unreviewed commands are rejected before renderer execution.

        Payload decoding assumes the current little-endian capture platform.
        """
        from copy import deepcopy
        from .pglRecordingFile import pglRecordingFile

        if not isinstance(recording, dict):
            raise TypeError("recording must be a dictionary.")

        commands = recording.get("commands")
        presentations = recording.get("presentations")
        metadata = recording.get("metadata")

        if commands is None:
            raise ValueError("Drawing reconstruction requires recorded commands.")
        if not isinstance(metadata, dict):
            raise ValueError("Drawing reconstruction requires session metadata.")
        if metadata.get("outcome") == "active":
            raise ValueError("Stop capture before preparing reconstruction.")

        pglRecordingFile.validate(commands, presentations)

        # Gaps cannot be repaired by treating a flush as a state checkpoint.
        segments = commands.get("segments") or []
        if len(segments) != 1:
            raise ValueError("This reconstruction version requires one continuous command segment.")

        segment = segments[0]
        if (segment.get("commandStartIndex") != 0
                or segment.get("commandStopIndex") != commands["count"]):
            raise ValueError("The command segment must cover the entire stopped command recording.")

        supportedNames = {
            "mglSetClearColor",
            "mglSetXform",
            "mglQuad",
            "mglLine",
            "mglDots",
            "mglArcs",
            "mglFlush",
            "mglSetTextRenderingParameters",
            "mglCreateTextTexture",
            "mglBltTexture",
            "mglDeleteTexture",
        }

        # Reuse reviewed payload and resource-lifetime checks. Timing findings
        # do not block pixel reconstruction, but malformed/unresolved command
        # state does. This helper performs no renderer or filesystem work.
        #
        # Arc payload validation is already handled by _analyzeCommands:
        # one uint32 count followed by 14 float32 values per arc.
        commandFindings = []

        def collectFinding(code, severity, message, location=None):
            commandFindings.append({
                "code": code,
                "severity": severity,
                "message": message,
                "location": location,
            })

        info = pglRecordingFile._analyzeCommands(commands, "little", collectFinding)

        for index, name in enumerate(info["names"]):
            if name not in supportedNames:
                raise ValueError(
                    f"Command {index} ({name!r}) is not supported by drawing reconstruction."
                )

        for finding in commandFindings:
            if finding["severity"] != "warning": continue

            # Client logging times are not used to execute drawing commands
            # or assign physical presentation times.
            if finding["code"] == "commandLoggingTimeMissing": continue

            raise ValueError(
                f"Cannot reconstruct: {finding['code']} at "
                f"{finding['location']}: {finding['message']}"
            )

        # Match the original pixel dimensions and angular extent. Recorded
        # dot sizes and text textures already contain pixel-valued quantities.
        display = metadata.get("initialDisplay") or {}

        def positiveDimension(dimension, unit, integer=False):
            value = (display.get(dimension) or {}).get(unit)
            array = np.asarray(value)
            if (array.ndim != 0 or array.dtype.kind not in "iuf"
                    or not np.isfinite(array) or array <= 0):
                raise ValueError(f"Missing or invalid initialDisplay.{dimension}.{unit}.")
            if integer:
                if float(array) != int(array):
                    raise ValueError(f"initialDisplay.{dimension}.{unit} must be an integer.")
                return int(array)
            return float(array)

        setup = {
            "widthPixels": positiveDimension("screenWidth", "pix", integer=True),
            "heightPixels": positiveDimension("screenHeight", "pix", integer=True),
            "widthDegrees": positiveDimension("screenWidth", "deg"),
            "heightDegrees": positiveDimension("screenHeight", "deg"),
            "clearScreenColor": deepcopy(display.get("clearScreenColor")),
        }

        # A fresh renderer must receive these settings explicitly rather
        # than relying on Python's cached font state.
        textSettings = deepcopy(segment.get("textRenderingParameters"))
        if "mglCreateTextTexture" in info["names"]:
            requiredSettings = {
                "fontName", "fontSize", "paddingPixels",
                "foregroundColor", "backgroundColor",
            }
            if not isinstance(textSettings, dict) or not requiredSettings.issubset(textSettings):
                raise ValueError("Text reconstruction requires complete segment-start text settings.")

        flushCommandIndices = info["flushIndices"]
        frameCount = len(flushCommandIndices)

        if frameIndices is None:
            selectedIndices = list(range(frameCount))
        else:
            try:
                selectedIndices = list(frameIndices)
            except TypeError as error:
                raise TypeError("frameIndices must be an iterable of integer indices or None.") from error

            for index in selectedIndices:
                if (not isinstance(index, (int, np.integer))
                        or isinstance(index, (bool, np.bool_))):
                    raise TypeError("Each frame index must be an integer.")
                if not 0 <= index < frameCount:
                    raise ValueError(f"Frame index {index} is outside [0, {frameCount}).")

            selectedIndices = [int(index) for index in selectedIndices]
            if len(set(selectedIndices)) != len(selectedIndices):
                raise ValueError("Duplicate frame indices are not supported.")

            # Execution and output follow original command order.
            selectedIndices.sort()

        # Link optional physical presentation records by command index.
        # Never assume presentation-row index equals reconstructed-frame index.
        presentationByCommand = {}
        unlinkedPresentationRows = []

        if presentations is not None:
            for row in range(presentations["count"]):
                commandIndex = int(presentations["commandIndices"][row])
                if commandIndex == -1:
                    unlinkedPresentationRows.append(row)
                    continue

                if info["names"][commandIndex] != "mglFlush":
                    raise ValueError(f"Presentation row {row} references a non-flush command.")
                if commandIndex in presentationByCommand:
                    raise ValueError(f"Multiple presentation rows reference command {commandIndex}.")

                returnedCode = presentations["commandCode"][row]
                if (np.isfinite(returnedCode)
                        and returnedCode != commands["commandValues"][commandIndex]):
                    raise ValueError(f"Presentation row {row} has a mismatched command code.")

                presentationByCommand[commandIndex] = row

        frames = []
        for frameIndex, commandIndex in enumerate(flushCommandIndices):
            row = presentationByCommand.get(commandIndex)
            timing = None

            if row is not None:
                # Copy raw scalars without offsets, sentinel repair, or
                # substitution of client/processing timestamps.
                timing = {
                    name: presentations[name][row].item()
                    for name in pglRecordingFile.presentationArrayNames
                    if name != "commandIndices"
                }

            frames.append({
                "frameIndex": frameIndex,
                "commandIndex": commandIndex,
                "presentationIndex": row,
                "timing": timing,
            })

        resources = {
            int(resource["commandIndex"]): deepcopy(resource)
            for resource in commands.get("resources", [])
        }

        # Snapshot containers but share immutable packed bytes. No extra
        # content copy of large vertex payloads is made.
        plannedCommands = [
            {
                "commandIndex": index,
                "name": name,
                "chunks": tuple(commands["commandData"][index]),
                "resource": resources.get(index),
            }
            for index, name in enumerate(info["names"])
        ]

        # Execute all preceding commands, including unselected flushes.
        # An empty selection needs no renderer work.
        executeStopIndex = (
            frames[selectedIndices[-1]]["commandIndex"] + 1
            if selectedIndices else 0
        )

        return {
            "planVersion": 1,
            "recordingId": metadata.get("recordingId"),
            "commandByteOrder": "little",
            "setup": setup,
            "textSettings": textSettings,
            "commands": plannedCommands,
            "frames": frames,
            "selectedFrameIndices": selectedIndices,
            "executeStopIndex": executeStopIndex,
            "unlinkedPresentationRows": unlinkedPresentationRows,
            "initialStateBasis": "matchingVisualAngleExtent",
        }
        
    ################################################################
    # Reconstruct selected recorded drawing frames
    ################################################################
    def recordingReconstruct(self, recording, frameIndices=None, onFrame=None,
                             maxMemoryBytes=256 * 1024 * 1024):
        """Reconstruct drawing frames using a temporary renderer connection.

        frameIndices selects recorded flushes, not presentation-log rows.
        Earlier commands and unselected flushes still execute to establish
        renderer state. Only selected frames are read back.

        With onFrame=None, return frames in result["frames"], subject to
        maxMemoryBytes. Otherwise call onFrame(frame) synchronously and do
        not retain images internally. Callback exceptions stop reconstruction.

        The callback receives:
            image: float32 (height, width, 4), unchanged orientation
            frameIndex: index among recorded flush commands
            commandIndex: original recorded command index
            presentationIndex: linked timing row, or None
            timing: raw recorded flush results, or None

        This method owns the renderer it opens and closes it afterward.
        It does not change or clear the source recording. Initial setup uses
        matching visual-angle extent, not arbitrary pre-capture renderer state.
        """
        import struct
        import sys
        from copy import deepcopy

        if sys.byteorder != "little":
            raise RuntimeError("Drawing reconstruction currently requires a little-endian platform.")
        if onFrame is not None and not callable(onFrame):
            raise TypeError("onFrame must be callable or None.")
        if (isinstance(maxMemoryBytes, (bool, np.bool_))
                or not isinstance(maxMemoryBytes, (int, np.integer))
                or maxMemoryBytes <= 0):
            raise ValueError("maxMemoryBytes must be a positive integer.")

        if self.isOpen(): raise RuntimeError("Close the current renderer before reconstruction.")
        if self.renderTarget is not None:
            raise RuntimeError("An old render-target reference remains; resolve it before reconstruction.")
        if getattr(self, "_recordingState", None) == "active":
            raise RuntimeError("Stop capture before reconstruction.")
        if self.commandRecording or self.flushRecording:
            raise RuntimeError("Stop command and flush capture before reconstruction.")
        if getattr(self, "_batchState", 0) != 0 or getattr(self, "_profileMode", 0) != 0:
            raise RuntimeError("Disable batching and profiling before reconstruction.")

        # Complete offline command/resource validation before renderer work.
        plan = self._recordingReconstructionPlan(recording, frameIndices)
        selected = set(plan["selectedFrameIndices"])
        setup = plan["setup"]
        bytesPerFrame = setup["widthPixels"] * setup["heightPixels"] * 4 * 4
        estimatedImageBytes = len(selected) * bytesPerFrame

        if onFrame is None and estimatedImageBytes > maxMemoryBytes:
            raise ValueError(
                f"Selected images require approximately {estimatedImageBytes / 1024**2:.1f} MiB. "
                "Use onFrame to stream images, select fewer frames, or explicitly increase maxMemoryBytes."
            )

        # Establish the recorded clear color without adding an unrecorded
        # full-screen quad. Reject invalid metadata before opening the renderer.
        clearColor = setup["clearScreenColor"]
        if clearColor is None:
            raise ValueError("Initial clearScreenColor is required for this reconstruction version.")
        clearColor = np.asarray(clearColor, dtype=np.float32)
        if clearColor.shape != (3,) or not np.all(np.isfinite(clearColor)):
            raise ValueError("Initial clearScreenColor must contain three finite RGB values.")

        result = {
            "recordingId": plan["recordingId"],
            "selectedFrameIndices": list(plan["selectedFrameIndices"]),
            "frameCount": 0,
            "frames": [] if onFrame is None else None,
            "estimatedImageBytes": estimatedImageBytes,
            "initialStateBasis": plan["initialStateBasis"],
            "timingBasis": "originalRecordedFlushResults",
        }
        if not selected: return result

        def checkCompletion(results, label):
            success = None if results is None else results.get("success")
            if success is None or np.size(success) != 1 or not np.all(np.asarray(success) == 1):
                raise RuntimeError(f"{label} did not report success=1.")

        def sendChunks(chunks):
            # Direct transmission of prevalidated bytes, without recording
            # replay traffic or swallowing sendall failures.
            for chunk in chunks: self.s.s.sendall(chunk)

        def readAck(label):
            ack = self.s.readAck()
            if ack is None: raise RuntimeError(f"Missing acknowledgment for {label}.")
            return ack

        def readUInt(label):
            value = self.s.read(np.uint32)
            array = np.asarray(value)
            if array.ndim != 0 or array.dtype.kind not in "iu":
                raise RuntimeError(f"Missing or invalid {label}.")
            return int(array)

        framesByCommand = {
            frame["commandIndex"]: frame for frame in plan["frames"]
        }
        textureMap = {}
        primaryError = None

        try:
            self.open(
                0,
                screenWidth=setup["widthPixels"],
                screenHeight=setup["heightPixels"],
                backgroundColor=clearColor.tolist(),
            )
            if not self.isOpen(): raise RuntimeError("Reconstruction renderer did not open.")

            self.setTransformScale(
                2.0 / setup["widthDegrees"],
                2.0 / setup["heightDegrees"],
            )
            checkCompletion(self.commandResults, "Coordinate setup")

            # Force actual server settings rather than trusting a cache left
            # over from a previously opened renderer.
            if any(
                command["name"] == "mglCreateTextTexture"
                for command in plan["commands"][:plan["executeStopIndex"]]
            ):
                if not self.textSetRenderingParameters(**plan["textSettings"]):
                    raise RuntimeError("Could not restore segment-start text settings.")
                checkCompletion(self.commandResults, "Initial text settings")

            self.frameGrabInit()
            if self.renderTarget is None:
                raise RuntimeError("Offscreen target creation failed.")
            checkCompletion(self.commandResults, "Target selection")

            # Explicit setup boundary before executing recorded commands.
            # It does not create an output frame or consume original timing.
            self.flush()
            checkCompletion(self.commandResults, "Target setup flush")

            for command in plan["commands"][:plan["executeStopIndex"]]:
                name = command["name"]
                chunks = command["chunks"]
                index = command["commandIndex"]

                if name == "mglCreateTextTexture":
                    sendChunks(chunks[:1])
                    ack = readAck(name)
                    sendChunks(chunks[1:])

                    creationResult = self.s.read(np.float64)
                    value = np.asarray(creationResult)
                    if (value.ndim != 0 or value.dtype.kind not in "iuf"
                            or not np.isfinite(value) or value < 0):
                        raise RuntimeError(f"Text creation failed at command {index}.")

                    width = readUInt("text width")
                    height = readUInt("text height")
                    replayID = readUInt("text texture ID")
                    readUInt("texture count")
                    checkCompletion(self.s.readCommandResults(ack), name)

                    resource = command["resource"]
                    originalID = int(resource["originalID"])
                    resourceMetadata = resource.get("metadata") or {}

                    # Font/environment differences can change text geometry.
                    for field, actual in (("widthPixels", width), ("heightPixels", height)):
                        expected = resourceMetadata.get(field)
                        if expected is not None and int(expected) != actual:
                            raise RuntimeError(
                                f"Text {field} differs at command {index}: "
                                f"recorded={expected}, reconstructed={actual}."
                            )
                    textureMap[originalID] = replayID

                elif name == "mglBltTexture":
                    replacement = list(chunks)
                    originalID = struct.unpack("<I", chunks[7])[0]
                    replacement[7] = struct.pack("<I", textureMap[originalID])

                    sendChunks(replacement[:1])
                    ack = readAck(name)
                    sendChunks(replacement[1:])
                    checkCompletion(self.s.readCommandResults(ack), name)

                elif name == "mglDeleteTexture":
                    originalID = struct.unpack("<I", chunks[1])[0]
                    sendChunks([
                        chunks[0],
                        struct.pack("<I", textureMap[originalID]),
                    ])
                    checkCompletion(self.s.readCommandResults(), name)
                    del textureMap[originalID]

                elif name in ("mglSetTextRenderingParameters", "mglArcs"):
                    # Match the original acknowledgment-before-payload sequence.
                    # Arc payloads contain the count and packed vertex data;
                    # replay those validated bytes without conversion.
                    sendChunks(chunks[:1])
                    ack = readAck(name)
                    sendChunks(chunks[1:])
                    checkCompletion(self.s.readCommandResults(ack), name)

                else:
                    # All remaining admitted commands use ordinary completion
                    # results in the reviewed Python paths.
                    sendChunks(chunks)
                    checkCompletion(self.s.readCommandResults(), f"{name} at {index}")

                if name == "mglFlush":
                    frameInfo = framesByCommand[index]
                    if frameInfo["frameIndex"] not in selected: continue

                    frame = deepcopy(frameInfo)
                    frame["image"] = self.frameGrab()

                    if onFrame is None:
                        result["frames"].append(frame)
                    else:
                        onFrame(frame)

                    result["frameCount"] += 1
                    # Do not retain the callback's image until the next readback.
                    del frame

        except BaseException as error:
            primaryError = error
            raise

        finally:
            # The renderer is exclusively owned by this operation. Closing it
            # releases all replay textures, including resources whose recorded
            # deletion lies beyond the selected prefix. Do not send more
            # protocol commands after a failed exchange.
            try:
                if self.close() is False:
                    raise RuntimeError("Reconstruction renderer shutdown was not confirmed.")

                # Renderer exit invalidates its target ID. Clear it before
                # releasing the Python object to avoid destructor deletion.
                if self.renderTarget is not None:
                    self.renderTarget.imageNum = None
                    self.renderTarget = None

            except BaseException as cleanupError:
                if primaryError is None: raise
                if hasattr(primaryError, "add_note"):
                    primaryError.add_note(
                        f"Secondary renderer cleanup failure: "
                        f"{type(cleanupError).__name__}: {cleanupError}"
                    )

        return result
    
    ################################################################
    # Reconstruct drawing frames selected by original experiment time
    ################################################################
    def recordingReconstructTime(self, recording, time=None, startTime=None,
                                 stopTime=None, onFrame=None,
                                 allowAmbiguous=False,
                                 maxMemoryBytes=256 * 1024 * 1024):
        """Reconstruct candidate frames at a time or within [startTime, stopTime).

        Times use the original recording/experiment clock, without offsets.

        By default, detected timing ambiguity raises before opening a renderer.
        With allowAmbiguous=True, reconstruct all selected candidates without
        claiming which candidate was physically displayed.

        onFrame receives the same frame dictionaries as recordingReconstruct().
        The returned selection retains interval/group information, assumptions,
        and ambiguities alongside the reconstruction result.

        An unambiguous selection is still conditional on reported renderer
        timing; it is not independent verification of physical visibility.
        """
        from .pglRecordingFile import pglRecordingFile

        if not isinstance(allowAmbiguous, bool):
            raise TypeError("allowAmbiguous must be Boolean.")

        # Selection is offline and happens before any renderer operation.
        selection = pglRecordingFile.selectFrames(
            recording,
            time=time,
            startTime=startTime,
            stopTime=stopTime,
        )

        if selection["ambiguous"] and not allowAmbiguous:
            reasons = ", ".join(selection["ambiguities"])
            raise ValueError(
                f"Time selection is ambiguous: {reasons}. "
                "Inspect pglRecordingFile.selectFrames() or explicitly set "
                "allowAmbiguous=True to reconstruct candidate images."
            )

        reconstruction = self.recordingReconstruct(
            recording,
            frameIndices=selection["frameIndices"],
            onFrame=onFrame,
            maxMemoryBytes=maxMemoryBytes,
        )

        return {
            "selection": selection,
            "reconstruction": reconstruction,
        }
        
    ################################################################
    # Reconstruct selected recorded drawing frames
    ################################################################
    def recordingReconstruct(self, recording, frameIndices=None, onFrame=None,
                             maxMemoryBytes=256 * 1024 * 1024, progress=None):
        """Reconstruct drawing frames using a temporary renderer connection.

        frameIndices selects recorded flushes, not presentation-log rows.
        Earlier commands and unselected flushes still execute to establish
        renderer state. Only selected frames are read back.

        With onFrame=None, return frames in result["frames"], subject to
        maxMemoryBytes. Otherwise call onFrame(frame) synchronously and do
        not retain images internally. Callback exceptions stop reconstruction.

        The callback receives:
            image: float32 (height, width, 4), unchanged orientation
            frameIndex: index among recorded flush commands
            commandIndex: original recorded command index
            presentationIndex: linked timing row, or None
            timing: raw recorded flush results, or None

        progress is an optional pglProgress-compatible callable or section.
        Its owner is responsible for finish() and close(). Progress advances
        during command replay, including commands without selected readback.

        This method owns the renderer it opens and closes it afterward.
        It does not change or clear the source recording. Initial setup uses
        matching visual-angle extent, not arbitrary pre-capture renderer state.
        """
        import struct
        import sys
        from copy import deepcopy

        if sys.byteorder != "little":
            raise RuntimeError("Drawing reconstruction currently requires a little-endian platform.")
        if onFrame is not None and not callable(onFrame):
            raise TypeError("onFrame must be callable or None.")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None.")
        if (isinstance(maxMemoryBytes, (bool, np.bool_))
                or not isinstance(maxMemoryBytes, (int, np.integer))
                or maxMemoryBytes <= 0):
            raise ValueError("maxMemoryBytes must be a positive integer.")

        if self.isOpen(): raise RuntimeError("Close the current renderer before reconstruction.")
        if self.renderTarget is not None:
            raise RuntimeError("An old render-target reference remains; resolve it before reconstruction.")
        if getattr(self, "_recordingState", None) == "active":
            raise RuntimeError("Stop capture before reconstruction.")
        if self.commandRecording or self.flushRecording:
            raise RuntimeError("Stop command and flush capture before reconstruction.")
        if getattr(self, "_batchState", 0) != 0 or getattr(self, "_profileMode", 0) != 0:
            raise RuntimeError("Disable batching and profiling before reconstruction.")

        # Complete offline command/resource validation before renderer work.
        if progress is not None: progress(0, "Validating reconstruction")
        plan = self._recordingReconstructionPlan(recording, frameIndices)
        selected = set(plan["selectedFrameIndices"])
        setup = plan["setup"]
        bytesPerFrame = setup["widthPixels"] * setup["heightPixels"] * 4 * 4
        estimatedImageBytes = len(selected) * bytesPerFrame

        if onFrame is None and estimatedImageBytes > maxMemoryBytes:
            raise ValueError(
                f"Selected images require approximately {estimatedImageBytes / 1024**2:.1f} MiB. "
                "Use onFrame to stream images, select fewer frames, or explicitly increase maxMemoryBytes."
            )

        # Establish the recorded clear color without adding an unrecorded
        # full-screen quad. Reject invalid metadata before opening the renderer.
        clearColor = setup["clearScreenColor"]
        if clearColor is None:
            raise ValueError("Initial clearScreenColor is required for this reconstruction version.")
        clearColor = np.asarray(clearColor, dtype=np.float32)
        if clearColor.shape != (3,) or not np.all(np.isfinite(clearColor)):
            raise ValueError("Initial clearScreenColor must contain three finite RGB values.")

        result = {
            "recordingId": plan["recordingId"],
            "selectedFrameIndices": list(plan["selectedFrameIndices"]),
            "frameCount": 0,
            "frames": [] if onFrame is None else None,
            "estimatedImageBytes": estimatedImageBytes,
            "initialStateBasis": plan["initialStateBasis"],
            "timingBasis": "originalRecordedFlushResults",
        }
        if not selected:
            if progress is not None: progress(1, "No frames selected")
            return result

        def checkCompletion(results, label):
            success = None if results is None else results.get("success")
            if success is None or np.size(success) != 1 or not np.all(np.asarray(success) == 1):
                raise RuntimeError(f"{label} did not report success=1.")

        def sendChunks(chunks):
            # Direct transmission of prevalidated bytes, without recording
            # replay traffic or swallowing sendall failures.
            for chunk in chunks: self.s.s.sendall(chunk)

        def readAck(label):
            ack = self.s.readAck()
            if ack is None: raise RuntimeError(f"Missing acknowledgment for {label}.")
            return ack

        def readUInt(label):
            value = self.s.read(np.uint32)
            array = np.asarray(value)
            if array.ndim != 0 or array.dtype.kind not in "iu":
                raise RuntimeError(f"Missing or invalid {label}.")
            return int(array)

        framesByCommand = {
            frame["commandIndex"]: frame for frame in plan["frames"]
        }
        textureMap = {}
        primaryError = None
        executeStopIndex = plan["executeStopIndex"]

        try:
            if progress is not None: progress(0.03, "Opening reconstruction renderer")
            self.open(
                0,
                screenWidth=setup["widthPixels"],
                screenHeight=setup["heightPixels"],
                backgroundColor=clearColor.tolist(),
            )
            if not self.isOpen(): raise RuntimeError("Reconstruction renderer did not open.")

            self.setTransformScale(
                2.0 / setup["widthDegrees"],
                2.0 / setup["heightDegrees"],
            )
            checkCompletion(self.commandResults, "Coordinate setup")

            # Force actual server settings rather than trusting a cache left
            # over from a previously opened renderer.
            if any(
                command["name"] == "mglCreateTextTexture"
                for command in plan["commands"][:executeStopIndex]
            ):
                if not self.textSetRenderingParameters(**plan["textSettings"]):
                    raise RuntimeError("Could not restore segment-start text settings.")
                checkCompletion(self.commandResults, "Initial text settings")

            self.frameGrabInit()
            if self.renderTarget is None:
                raise RuntimeError("Offscreen target creation failed.")
            checkCompletion(self.commandResults, "Target selection")

            # Explicit setup boundary before executing recorded commands.
            # It does not create an output frame or consume original timing.
            self.flush()
            checkCompletion(self.commandResults, "Target setup flush")

            if progress is not None: progress(0.08, "Reconstructing frames")

            for command in plan["commands"][:executeStopIndex]:
                name = command["name"]
                chunks = command["chunks"]
                index = command["commandIndex"]

                if name == "mglCreateTextTexture":
                    sendChunks(chunks[:1])
                    ack = readAck(name)
                    sendChunks(chunks[1:])

                    creationResult = self.s.read(np.float64)
                    value = np.asarray(creationResult)
                    if (value.ndim != 0 or value.dtype.kind not in "iuf"
                            or not np.isfinite(value) or value < 0):
                        raise RuntimeError(f"Text creation failed at command {index}.")

                    width = readUInt("text width")
                    height = readUInt("text height")
                    replayID = readUInt("text texture ID")
                    readUInt("texture count")
                    checkCompletion(self.s.readCommandResults(ack), name)

                    resource = command["resource"]
                    originalID = int(resource["originalID"])
                    resourceMetadata = resource.get("metadata") or {}

                    # Font/environment differences can change text geometry.
                    for field, actual in (("widthPixels", width), ("heightPixels", height)):
                        expected = resourceMetadata.get(field)
                        if expected is not None and int(expected) != actual:
                            raise RuntimeError(
                                f"Text {field} differs at command {index}: "
                                f"recorded={expected}, reconstructed={actual}."
                            )
                    textureMap[originalID] = replayID

                elif name == "mglBltTexture":
                    replacement = list(chunks)
                    originalID = struct.unpack("<I", chunks[7])[0]
                    replacement[7] = struct.pack("<I", textureMap[originalID])

                    sendChunks(replacement[:1])
                    ack = readAck(name)
                    sendChunks(replacement[1:])
                    checkCompletion(self.s.readCommandResults(ack), name)

                elif name == "mglDeleteTexture":
                    originalID = struct.unpack("<I", chunks[1])[0]
                    sendChunks([
                        chunks[0],
                        struct.pack("<I", textureMap[originalID]),
                    ])
                    checkCompletion(self.s.readCommandResults(), name)
                    del textureMap[originalID]

                elif name in ("mglSetTextRenderingParameters", "mglArcs"):
                    # Match the original acknowledgment-before-payload sequence.
                    # Arc count and vertex bytes are replayed without conversion.
                    sendChunks(chunks[:1])
                    ack = readAck(name)
                    sendChunks(chunks[1:])
                    checkCompletion(self.s.readCommandResults(ack), name)

                else:
                    # All remaining admitted commands use ordinary completion
                    # results in the reviewed Python paths.
                    sendChunks(chunks)
                    checkCompletion(self.s.readCommandResults(), f"{name} at {index}")

                if name == "mglFlush":
                    frameInfo = framesByCommand[index]

                    if frameInfo["frameIndex"] in selected:
                        frame = deepcopy(frameInfo)
                        frame["image"] = self.frameGrab()

                        if onFrame is None:
                            result["frames"].append(frame)
                        else:
                            onFrame(frame)

                        result["frameCount"] += 1

                        # Do not retain the callback's image until next readback.
                        del frame

                # Advance even when no selected image was read back.
                # pglProgress throttles display rendering internally.
                if progress is not None:
                    fraction = 0.08 + 0.87 * (index + 1) / executeStopIndex
                    progress(
                        fraction,
                        f"Reconstructing frames — commands {index + 1:,}/{executeStopIndex:,}; "
                        f"images {result['frameCount']:,}/{len(selected):,}",
                    )

            if progress is not None: progress(0.95, "Closing reconstruction renderer")

        except BaseException as error:
            primaryError = error
            raise

        finally:
            # The renderer is exclusively owned by this operation. Closing it
            # releases replay textures, including resources whose recorded
            # deletion lies beyond the selected prefix. Do not send more
            # protocol commands after a failed exchange.
            try:
                if self.close() is False:
                    raise RuntimeError("Reconstruction renderer shutdown was not confirmed.")

                # Renderer exit invalidates its target ID. Clear it before
                # releasing the Python object to avoid destructor deletion.
                if self.renderTarget is not None:
                    self.renderTarget.imageNum = None
                    self.renderTarget = None

            except BaseException as cleanupError:
                if primaryError is None: raise
                if hasattr(primaryError, "add_note"):
                    primaryError.add_note(
                        f"Secondary renderer cleanup failure: "
                        f"{type(cleanupError).__name__}: {cleanupError}"
                    )

        # This completes only the assigned section, not the owner's whole job.
        if progress is not None: progress(1, "Reconstruction complete")
        return result
    
    ################################################################
    # Export a timestamp-sampled reconstruction movie
    ################################################################
    def recordingMovie(self, recording, outputFilename, fps=None, sampling="nearest",
                       allowAmbiguous=False, overwrite=False):
        """Stream a reconstructed recording to a timestamp-sampled MP4.

        fps=None infers the rate from median positive presentation intervals.
        sampling is "nearest" by default, or "previous".

        allowAmbiguous=True explicitly accepts the timing helper's conventions:
            Exclude untimed images from timestamp selection.
            Choose the last flush at duplicate timestamps.

        All preceding drawing commands still execute to establish state.
        Only selected images are read back. Repeated output samples reuse
        the same encoded-input pixel bytes.

        Output is an 8-bit RGB-to-H.264 yuv420p preview, not a lossless archive.
        Alpha is discarded and RGB is clipped to [0, 1].
        The tested image orientation is preserved.

        The destination is published only after encoding and renderer cleanup.
        Existing files are not replaced unless overwrite=True.
        """
        import os
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        from .pglProgress import pglProgress
        from .pglRecordingFile import pglRecordingFile

        if not isinstance(overwrite, bool): raise TypeError("overwrite must be Boolean.")
        if self.isOpen(): raise RuntimeError("Close the current renderer before movie export.")
        if self.renderTarget is not None: raise RuntimeError("Resolve the stale render target before movie export.")
        if getattr(self, "_recordingState", None) == "active":
            raise RuntimeError("Stop capture before movie export.")
        if self.commandRecording or self.flushRecording:
            raise RuntimeError("Stop command and flush capture before movie export.")
        if getattr(self, "_batchState", 0) != 0 or getattr(self, "_profileMode", 0) != 0:
            raise RuntimeError("Disable batching and profiling before movie export.")

        ffmpegPath = shutil.which("ffmpeg")
        if ffmpegPath is None: raise RuntimeError("FFmpeg was not found on PATH.")

        outputPath = Path(outputFilename).expanduser().absolute()
        if outputPath.suffix.lower() != ".mp4": raise ValueError("outputFilename must end in .mp4.")
        if os.path.lexists(outputPath) and not overwrite: raise FileExistsError(outputPath)

        with pglProgress("Exporting recording movie") as progress:
            progress(0, "Validating recording")
            reconstructionPlan = self._recordingReconstructionPlan(recording)

            if reconstructionPlan["unlinkedPresentationRows"]:
                raise ValueError("Movie export requires presentation rows linked to recorded flushes.")

            timingPlan = pglRecordingFile._recordingMovieTimingPlan(
                reconstructionPlan["frames"],
                fps=fps,
                sampling=sampling,
                allowAmbiguous=allowAmbiguous,
                progress=progress.section(0.02, 0.08),
            )

            setup = reconstructionPlan["setup"]
            width = setup["widthPixels"]
            height = setup["heightPixels"]
            expectedShape = (height, width, 4)

            # H.264 yuv420p requires even dimensions. Do not silently resize.
            if width % 2 or height % 2:
                raise ValueError("This MP4 exporter requires even width and height.")

            sourceIndices = timingPlan["sourceFrameIndices"]
            selectedIndices = timingPlan["selectedFrameIndices"]
            outputCount = timingPlan["outputFrameCount"]

            # Monotonic source selection permits streaming without an image cache.
            if np.any(np.diff(sourceIndices) < 0):
                raise ValueError("Movie source selection must follow recorded frame order.")

            selectedArray, repeatCounts = np.unique(sourceIndices, return_counts=True)
            if selectedArray.tolist() != selectedIndices:
                raise ValueError("Movie source selection is inconsistent.")

            chosenFps = timingPlan["fps"]
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            replayProgress = progress.section(0.08, 0.94)

            # Release the redundant plan containers before reconstruction.
            # The source recording still owns its immutable command bytes.
            del reconstructionPlan

            writtenCount = 0
            receivedCount = 0

            # Temporary output resides beside the destination for publication
            # on the same filesystem. Partial output is removed on failure.
            with tempfile.TemporaryDirectory(prefix=".pglMovie_", dir=str(outputPath.parent)) as temporaryDirectory:
                temporaryPath = Path(temporaryDirectory) / "movie.mp4"

                # A file avoids deadlock from an unread stderr pipe.
                with tempfile.TemporaryFile(mode="w+b") as encoderLog:
                    command = [
                        ffmpegPath,
                        "-hide_banner",
                        "-loglevel", "error",
                        "-nostdin",
                        "-y",
                        "-f", "rawvideo",
                        "-pixel_format", "rgb24",
                        "-video_size", f"{width}x{height}",
                        "-framerate", str(chosenFps),
                        "-i", "pipe:0",
                        "-an",
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "18",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(temporaryPath),
                    ]

                    progress(0.08, "Starting movie encoder")
                    encoder = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=encoderLog,
                    )

                    def encoderDetails():
                        # Keep diagnostic messages bounded.
                        encoderLog.flush()
                        encoderLog.seek(0, os.SEEK_END)
                        encoderLog.seek(max(0, encoderLog.tell() - 8192))
                        return encoderLog.read().decode("utf-8", errors="replace").strip()

                    def encodeFrame(frame):
                        nonlocal writtenCount, receivedCount

                        if receivedCount >= len(selectedIndices):
                            raise RuntimeError("Reconstruction returned an unexpected extra image.")

                        expectedIndex = selectedIndices[receivedCount]
                        if frame["frameIndex"] != expectedIndex:
                            raise RuntimeError(
                                f"Expected source frame {expectedIndex}, received {frame['frameIndex']}."
                            )

                        image = frame["image"]
                        if image.shape != expectedShape:
                            raise ValueError(f"Unexpected image shape: {image.shape}.")
                        if not np.all(np.isfinite(image)):
                            raise ValueError(f"Frame {expectedIndex} contains nonfinite pixels.")

                        # Preserve orientation. Conversion affects only the movie
                        # preview, never the source recording or recorded bytes.
                        rgb = np.rint(np.clip(image[:, :, :3], 0, 1) * 255).astype(np.uint8)
                        payload = rgb.tobytes(order="C")

                        # One source image may supply several output sample times.
                        # Reuse its bytes rather than reading the renderer again.
                        for repeatIndex in range(int(repeatCounts[receivedCount])):
                            if encoder.poll() is not None:
                                raise RuntimeError("FFmpeg exited before reconstruction finished.")

                            byteCount = encoder.stdin.write(payload)
                            if byteCount != len(payload):
                                raise RuntimeError("Incomplete frame write to FFmpeg.")

                            writtenCount += 1

                            # Keep long repeated-image runs visibly active.
                            # Fractional progress is based on completed replay
                            # commands; this only updates the status description.
                            if repeatIndex % 30 == 0:
                                replayProgress(
                                    None,
                                    f"Reconstructing frames — encoding movie frame "
                                    f"{writtenCount:,}/{outputCount:,}",
                                )

                        receivedCount += 1

                    try:
                        reconstruction = self.recordingReconstruct(
                            recording,
                            frameIndices=selectedIndices,
                            onFrame=encodeFrame,
                            progress=replayProgress,
                        )

                        if receivedCount != len(selectedIndices):
                            raise RuntimeError("Reconstruction returned an incomplete source-image sequence.")
                        if writtenCount != outputCount:
                            raise RuntimeError("Movie output frame count mismatch.")
                        if reconstruction["frameCount"] != receivedCount:
                            raise RuntimeError("Reconstruction frame count mismatch.")
                        if self.isOpen(): raise RuntimeError("Reconstruction left the renderer open.")
                        if self.renderTarget is not None:
                            raise RuntimeError("Reconstruction left a stale render target.")

                        # EOF completes encoding and writes the MP4 index.
                        progress(0.95, "Finalizing movie encoder")
                        encoder.stdin.close()
                        returnCode = encoder.wait(timeout=120)
                        if returnCode != 0:
                            raise RuntimeError(f"FFmpeg exited with code {returnCode}.")
                        if not temporaryPath.is_file() or temporaryPath.stat().st_size == 0:
                            raise RuntimeError("FFmpeg did not produce a nonempty movie.")

                        progress(0.98, "Publishing movie")

                        if overwrite:
                            os.replace(temporaryPath, outputPath)
                        else:
                            # Atomic no-clobber publication on the same filesystem.
                            # Unlike an existence check followed by replace(), this
                            # cannot overwrite a file created by another process.
                            os.link(temporaryPath, outputPath)
                            temporaryPath.unlink()

                    except BaseException as error:
                        # Stop the encoder before closing its input, which might
                        # otherwise block if the encoder stopped consuming bytes.
                        try:
                            if encoder.poll() is None: encoder.kill()
                            encoder.wait(timeout=10)
                        except Exception as cleanupError:
                            if hasattr(error, "add_note"):
                                error.add_note(f"Encoder cleanup failed: {cleanupError}")

                        try:
                            details = encoderDetails()
                            if details and hasattr(error, "add_note"):
                                error.add_note(f"FFmpeg diagnostics:\n{details}")
                        except Exception:
                            pass

                        raise

                    finally:
                        try:
                            if encoder.stdin is not None and not encoder.stdin.closed: encoder.stdin.close()
                        except (BrokenPipeError, OSError):
                            pass

            # Finish only after publication and temporary-resource cleanup.
            result = {
                "outputFilename": str(outputPath),
                "recordingId": reconstruction["recordingId"],
                "frameCount": writtenCount,
                "sourceImageCount": receivedCount,
                "fps": chosenFps,
                "durationSeconds": timingPlan["outputDurationSeconds"],
                "widthPixels": width,
                "heightPixels": height,
                "sampling": sampling,
                "timingMode": "recordedPresentationSampling",
                "physicalVisibilityVerified": False,
                "timingPlan": timingPlan,
            }
            progress.finish(f"Movie saved — {writtenCount:,} frames at {chosenFps:g} fps")

        return result
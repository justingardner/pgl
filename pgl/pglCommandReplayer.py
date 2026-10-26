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
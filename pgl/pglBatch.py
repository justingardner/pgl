################################################################
#   filename: pglBatch.py
#    purpose: Module for providing pgl batch processing which
#             allows you to run a set of drawing commands
#             with multiiple flushes and put those in the
#             queue to display. They only display when you
#             run the batch. This allows for the most control
#             over frame timing as the commands are precomputed
#             and communicated from python and stored in the
#             queue rather than being sent every frame.
#         by: JLG
#       date: July 22, 2025
################################################################

#############
# Import modules
#############
import numpy as np

#############
# Batch class
#############
class pglBatch:
    '''
    Class for managing batch processing in pgl.

    This class allows you to queue up drawing commands and flush them
    to the display in a controlled manner, improving performance and
    timing accuracy.
    
    Args:
        None

    Returns:
        None
    '''
    _batchState = 0

    def batchStart(self):
        '''
        Start a new batch.

        Drawing commands and flushes are queued until batchRun().
        Batch recording requires separate response/timing handling.
        '''
        if self._batchState != 0:
            raise RuntimeError("Batch already started. End the current batch first.")
        if self._profileMode != 0:
            raise RuntimeError("Profiling is already active. Stop profiling before starting a batch.")
        if getattr(self, "_recordingState", None) == "active":
            raise RuntimeError("Batch mode during a recording session is not supported yet.")
        if getattr(self, "flushRecording", False) or getattr(self, "commandRecording", False):
            raise RuntimeError("Stop flush and command recording before starting a batch.")

        # Initialize profiling before requesting batch mode.
        self._profileMode = -1
        try:
            self._profileModeStart()
        except BaseException:
            self._profileMode = 0
            raise

        # An interrupted exchange must not falsely mark the batch idle.
        self._batchState = 1
        self.s.writeCommand("mglStartBatch")
        ack = self.s.readAck()

        if self.verbose > 0: print("(pglBatch:batchStart) Batch started.")
    
    def batchRun(self):
        '''
        Run the batch.

        This function processes all queued commands and sends them to
        the display for rendering. It ensures that all commands are
        executed in the order they were added to the batch.
        
        Returns:
            None
        '''
        if self._batchState != 1:
            print("(pglBatch:batchRun) ❌ Batch not started. Please start a batch before running it.")
            return
        self._batchState = 2
        self.s.writeCommand("mglProcessBatch")
        ack = self.s.readAck()
        if self.verbose>0: print("(pglBatch:batchRun) Batch run initiated.")
    
    def batchEnd(self):
        '''
        Finish a running batch and collect its command results.

        Returns:
            flushTimes: Array of times for the batch's flush commands.
        '''
        if self._batchState != 2:
            raise RuntimeError("Batch has not run. Call batchRun() before batchEnd().")

        # Read command count.
        nCommands = self.s.read(np.uint32)
        if nCommands is None: raise RuntimeError("Could not read batch command count.")
        nCommands = int(nCommands)
        if nCommands <= 0: raise RuntimeError("Batch returned no command results.")

        # Send command to finish the batch.
        self.s.writeCommand("mglFinishBatch")
        ack = self.s.readAck()

        # Read results before marking the batch complete.
        commandResults = self.s.readCommandResults(0, nCommands)
        if commandResults is None: raise RuntimeError("Could not read batch command results.")

        commandCode = commandResults.get("commandCode")
        flushTime = commandResults.get(self.profileCommandResultsField)
        if commandCode is None or flushTime is None:
            raise RuntimeError("Batch results are missing command codes or profile timestamps.")

        # np.atleast_1d also handles the single-command case.
        commandCode = np.atleast_1d(commandCode)
        flushTime = np.atleast_1d(flushTime)
        if commandCode.size != nCommands or flushTime.size != nCommands:
            raise RuntimeError("Batch result lengths do not match its command count.")

        flushMask = commandCode == self.s.getCommandValue("mglFlush")

        # Preserve every returned result separately. The existing profile
        # display expects commandResults to align with flushTimes.
        self.profileInfo["allCommandResults"] = commandResults
        self.profileInfo["flushTimes"] = flushTime[flushMask].copy()
        self.profileInfo["commandResults"] = {}

        for key, value in commandResults.items():
            if value is None:
                self.profileInfo["commandResults"][key] = np.full(int(flushMask.sum()), np.nan)
            else:
                values = np.atleast_1d(value)
                if values.size == nCommands:
                    self.profileInfo["commandResults"][key] = values[flushMask].copy()
                elif values.size == 1:
                    # The acknowledgment is shared rather than per-command.
                    self.profileInfo["commandResults"][key] = np.full(int(flushMask.sum()), values[0])
                else:
                    raise RuntimeError(f"Unexpected batch result length for {key!r}.")

        # Finalize only after the results have been collected and validated.
        self._profileModeEnd()
        self.profileList.append(self.profileInfo)
        self._batchState = 0
        self._profileMode = 0
        self.profileModeBufferIndex = 0

        if self.verbose > 0: print(f"(pglBatch:batchEnd) Ended batch with {nCommands} commands.")
        return self.profileInfo["flushTimes"]

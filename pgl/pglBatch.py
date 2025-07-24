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

        This function initializes the batch processing. Once this is run
        you can do drawaing commands and flush but they will not be immediately
        displayed. Instead they will be queued in the app and only displayed
        when you call batchRun()
        
        Returns:
            None
        '''
        if self._batchState != 0:
            print("(pglBatch:batchStart) ❌ Batch already started. Please end the current batch before starting a new one.")
            return
        self._batchState = 1

        # turn on profiling for batch mode
        if self._profileMode > 0:
            print("(pglBatch:pglBatchStart) Profile mode is already started")
            # Fix, Fix, Fix should handle this contingency by combining info
            return
        self._profileMode = -1
        self._profileModeStart()

        # send command for starting the batch
        self.s.writeCommand("mglStartBatch")
        ack = self.s.readAck()

        if self.verbose>0: print("(pglBatch:batchStart) Batch started.")
    
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
        End the current batch.

        This function finalizes the batch processing, ensuring that all
        commands have been executed and the display is updated accordingly.

        Returns:
            None
        '''
        if self._batchState != 2:
            print("(pglBatch:batchEnd) ❌ Batch not run. Please run a batch before ending it.")
            return
        
        # read command count
        nCommands = self.s.read(np.uint32)

        # send command to finish batch
        self.s.writeCommand("mglFinishBatch")
        ack = self.s.readAck()
        
        # end the profiling
        self._profileModeEnd()

        
        # read command results
        commandResults = self.s.readCommandResults(0, nCommands)
        
        self._batchState = 0
        if self.verbose > 0:
            print(f"(pglBatch:batchEnd) Ended batch with {nCommands} commands.")

        # now end profiling info
        self.profileInfo['commandResults'] = commandResults

        # get the flushTime for all flush events
        commandCode = commandResults.get('commandCode',np.array([]))
        flushTime= commandResults.get(self.profileCommandResultsField,np.array([]))
        self.profileInfo['flushTimes'] = flushTime[commandCode==self.s.getCommandValue("mglFlush")]

         # Save the profile information to the profileList
        self.profileList.append(self.profileInfo)

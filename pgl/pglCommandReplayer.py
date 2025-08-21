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
        Initialize the pglReplayer.
        '''
        # status of command recording
        self.commandRecording = False

        # inital size of the log
        self.logSize = 1000
        # log for command numbers
        self.commandLog = np.zeros((self.logSize,), dtype=np.uint16)
        # log for data that is associated with each command
        self.commandLogData = np.full((self.logSize,), None, dtype=object)
        # index for current log entry
        self.logEntryIndex = 0

    def commandRecord(self):
        '''
        Record all pgl commands that are called after this method is called.
        Until commandRecordStop() is called.
        '''
        self.commandRecording = True

    def logCommandValue(self, commandValue):
        '''
        Will log commands that are being sent to the socket.
        This gets called by the _pglComm class when a command is sent.
        '''
        self.commandLog[self.logEntryIndex] = commandValue
        self.logEntryIndex += 1
        # if we reach the end of the log, reset the index
        if self.logEntryIndex >= self.logSize:
            # if we are at the end of the log, we can double the size of the log
            self.logSize *= 2
            # create a new log with the new size
            newCommandLog = np.zeros((self.logSize,), dtype=np.uint16)
            newCommandLogData = np.full((self.logSize,), None, dtype=object)
            # copy the old log to the new log
            newCommandLog[:len(self.commandLog)] = self.commandLog
            newCommandLogData[:len(self.commandLogData)] = self.commandLogData
            self.commandLog = newCommandLog
            self.commandLogData = newCommandLogData
    def logCommandData(self, commandData):
        '''
        Will log command data that are being sent to the socket.
        This gets called by the _pglComm class when a command is sent.
        '''
        # this should always get called after logCommandValue, so
        # we can assume that the commandData corresponds to the last command
        if self.commandLogData[self.logEntryIndex-1] is None:
            self.commandLogData[self.logEntryIndex-1] = []
        self.commandLogData[self.logEntryIndex-1].append(commandData)

    def commandRecordStop(self):
        '''
        Stop recording pgl commands.
        '''
        self.commandRecording = False
        if self.verbose:
            print(f"(pglCommandReplayer) Recorded {self.logEntryIndex} pgl commands.")

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
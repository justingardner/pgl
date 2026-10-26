################################################################
#   filename: pglProfile.py
#    purpose: Module for providing pgl profiling information
#             such as dropped frames and performance metrics.
#         by: JLG
#       date: July 22, 2025
################################################################

#############
# Import modules
#############
import numpy as np
import matplotlib.pyplot as plt
import time

#############
# Profile class
#############
class pglProfile:
    '''
    Class for profiling pgl performance metrics.
    
    '''
    ################################################################
    # Variables
    ################################################################
    _profileMode = 0  # Default profile mode (0 = off, 1 = dropped frames, 2 = detailed)
    profileModeBufferSize = None
    profileModeFlushBuffer = None
    profileModeCommandResults = None
    profileModeBufferIndex = 0
    profileList = []
    profileInfo = {}
    #profileCommandResultsField = 'drawablePresented'  # Default field for command results to track
    profileCommandResultsField = 'processedTime'  # Default field for command results to track

    ################################################################
    # Initialize instance-owned profile state
    ################################################################
    def __init__(self):
        """Keep profile buffers and history separate for each PGL instance."""
        self._profileMode = 0
        self.profileModeBufferSize = None
        self.profileModeFlushBuffer = None
        self.profileModeCommandResults = None
        self.profileModeBufferIndex = 0
        self.profileList = []
        self.profileInfo = {}

    ################################################################
    # profileMode property
    ################################################################
    @property
    def profileMode(self):
        """Get the current profiling mode."""
        return self._profileMode

    @profileMode.setter
    def profileMode(self, level):
        """Start/stop profiling without discarding an active session."""
        if level not in (0, 1, 2):
            raise ValueError("profileMode must be 0 (off), 1 (timing), or 2 (detailed).")

        # Batch profiling owns its lifecycle until batchEnd().
        if self._profileMode < 0:
            raise RuntimeError("Batch profiling is active; finish the batch before changing profileMode.")

        # Reassigning the current mode must not reset its buffers.
        if level == self._profileMode: return

        # Finish the existing session before switching to another mode.
        if self._profileMode > 0:
            self._profileModeEnd()
            count = self.profileModeBufferIndex

            # Include every populated entry, and detach it from live buffers.
            self.profileInfo["flushTimes"] = self.profileModeFlushBuffer[:count].copy()
            self.profileInfo["commandResults"] = None

            if self._profileMode >= 2:
                results = self.profileModeCommandResults[:count]
                fieldNames = {key for result in results for key in result}
                self.profileInfo["commandResults"] = {
                    key: np.asarray([
                        np.nan if result.get(key) is None else result[key]
                        for result in results
                    ])
                    for key in fieldNames
                }

            self.profileList.append(self.profileInfo)
            self.profileModeBufferIndex = 0
            self._profileMode = 0

        if level > 0:
            self._profileMode = level
            try:
                self._profileModeStart()
            except BaseException:
                # Do not leave profiling enabled after failed initialization.
                self._profileMode = 0
                raise

        if self.verbose > 0: print(f"(pglProfile) profileMode set to {self._profileMode}")

    ################################################################
    # _profileModeStart
    ################################################################
    def _profileModeStart(self):
        """Initialize a new profile session, including batch profiling."""
        import copy

        # Get the display frame rate before allocating buffers.
        frameRate = float(self.getFrameRate())
        if not np.isfinite(frameRate) or frameRate <= 0:
            raise ValueError("Profiling requires a finite, positive display frame rate.")

        # Default allocation is 60 seconds of frames.
        if self.profileModeBufferSize is None:
            self.profileModeBufferSize = max(1, int(np.ceil(frameRate * 60)))
            if self.verbose > 0:
                print(f"(pglProfile) profileModeBufferSize set to {self.profileModeBufferSize} frames ({self.profileModeBufferSize / frameRate:.2f} seconds)")
                print("(pglProfile) Will reallocate if exceeded; set profileModeBufferSize before starting to choose another capacity.")

        capacity = self.profileModeBufferSize
        if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, (int, np.integer)) or capacity <= 0:
            raise ValueError("profileModeBufferSize must be a positive integer.")
        self.profileModeBufferSize = int(capacity)

        # Always start a fresh session, even when capacity was already set.
        # Batch mode collects its result arrays at batchEnd(), not in flush().
        self.profileModeBufferIndex = 0
        self.profileModeFlushBuffer = np.empty(self.profileModeBufferSize, dtype=np.float64)
        self.profileModeCommandResults = [None] * self.profileModeBufferSize if self._profileMode >= 2 else None
        self.profileInfo = {}

        # Record session start information.
        self.profileInfo["startTime"] = time.time()
        localTime = time.localtime(self.profileInfo["startTime"])
        self.profileInfo["startTimeStr"] = time.strftime("%Y-%m-%d %H:%M:%S", localTime)

        # Sample CPU/GPU clocks using the existing profiling behavior.
        cpuTime, gpuTime = self.getTimestamps()
        self.profileInfo["cpuTime"] = cpuTime
        self.profileInfo["gpuTime"] = gpuTime
        self.profileInfo["profileMode"] = self._profileMode
        self.profileInfo["profileModeBufferStartSize"] = self.profileModeBufferSize
        self.profileInfo["frameRate"] = frameRate

        # Snapshot display information so later display changes cannot modify
        # the metadata of an earlier profile.
        self.getWindowFrameInDisplay()
        self.profileInfo["whichScreen"] = self.whichScreen
        self.profileInfo["screenX"] = copy.deepcopy(self.screenX)
        self.profileInfo["screenY"] = copy.deepcopy(self.screenY)
        self.profileInfo["screenWidth"] = copy.deepcopy(self.screenWidth)
        self.profileInfo["screenHeight"] = copy.deepcopy(self.screenHeight)

    ################################################################
    # _profileModeEnd
    ################################################################
    def _profileModeEnd(self):
        """Record the end of the current profile session."""
        self.profileInfo["endTime"] = time.time()
        localTime = time.localtime(self.profileInfo["endTime"])
        self.profileInfo["endTimeStr"] = time.strftime("%H:%M:%S", localTime)

    ################################################################
    # profileModeDisplay
    ################################################################
    def profileModeDisplay(self):
        """
        Display the current profile mode and buffer size.
        """
        if len(self.profileList) <= 0:
            print("(pglProfile) No profile data available.")
        else:
            for iProfile, profileInfo in enumerate(self.profileList):
                # display the number of frames in the profile
                print(f"-------- pglProfile {iProfile+1} -------------")
                flushTimes = profileInfo['flushTimes']
                if self.verbose>1: print(flushTimes)
                nFrames = len(flushTimes)
                if (nFrames==0):
                    print("(pglProfile) No frames in profile.")
                    continue
                totalTime = flushTimes[-1] - flushTimes[0]
                expectedFrameTime = 1 / profileInfo['frameRate']*1000  # in ms
                # get the difference in times
                frameTimes = np.diff(flushTimes)
                if self.verbose>1: print(frameTimes*1000)
                # compute stats
                meanFrameTime = np.mean(frameTimes)
                medianFrameTime = np.median(frameTimes)
                stdFrameTime = np.std(frameTimes)
                dropCriteria = meanFrameTime + meanFrameTime/2
                droppedFrames = np.sum(frameTimes > dropCriteria)
                # add one frames worth of time to totalTime (since we calculated the difference from the end of first to last, but didnt include the first frame
                totalTime += expectedFrameTime / 1000
                # format info for display
                profileText = f"{nFrames} frames, {totalTime:0.3f} secs Screen: {profileInfo['whichScreen']} ({profileInfo['screenWidth'].pix}x{profileInfo['screenHeight'].pix})"
                timeText = f"Started: {profileInfo['startTimeStr']} Ended: {profileInfo['endTimeStr']}"
                frameText = f"Frame Rate: {profileInfo['frameRate']} Hz Expected Frame Time: {expectedFrameTime:.2f} ms"
                frameTimeText = f"Median frame time: {medianFrameTime*1000:.2f} ms, {meanFrameTime*1000:.2f} ± {stdFrameTime*1000:.2f} mean ± std ms"
                droppedFramesText = f"Dropped frames (longer than {dropCriteria*1000:0.2f} ms): {droppedFrames} ({droppedFrames/nFrames*100:.2f}%)"
                # print information
                print(profileText)
                print(timeText)
                print(frameText)
                print(frameTimeText)
                print(droppedFramesText)
                # print them for each of the dropped frames
                commandResults = profileInfo.get('commandResults', None)
                for iDroppedFrame, frameTime in enumerate(frameTimes):
                    if frameTime > dropCriteria:
                        print(f"  Dropped Frame {iDroppedFrame+1}: {frameTime*1000:.2f} ms")
                        if commandResults is not None:
                            if iDroppedFrame > 0:
                                self.printCommandResults(commandResults,relativeToTime=commandResults['processedTime'][iDroppedFrame-1],prefix=f"    ",index=iDroppedFrame)
                            else:
                                self.printCommandResults(commandResults,prefix=f"    ",index=iDroppedFrame)

                # plot a historgram
                counts, _, _ = plt.hist(frameTimes*1000, bins=50, alpha=0.75, edgecolor='black')
                plt.title(f'Frame Time Histogram\n{profileText}\n{timeText}\n{frameText}\n{frameTimeText}\n{droppedFramesText}')
                plt.xlabel('Frame Time (ms)')
                plt.ylabel('Count (n)')
                plt.grid(axis='y', alpha=0.75)

                # Annotate with an arrow and line
                plt.axvline(expectedFrameTime, color='red', linestyle='dashed', linewidth=1)
                plt.annotate('Expected Frame Time',
                    xy=(expectedFrameTime, max(counts)*0.9),  # The point (expectedFrameTime, y_value)
                    xytext=(expectedFrameTime + 1, max(counts)*0.9),  # The text position
                    horizontalalignment='left',
                    verticalalignment='center',
                    arrowprops=dict(facecolor='red', shrink=0.05))
                plt.show()
    ################################################################
    # profileModeClearAll
    ################################################################
    def profileModeClearAll(self):
        """
        Clear all profile data.
        """
        self.profileList = []
        print("(pglProfile) Cleared all profile data.")

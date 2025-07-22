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

    ################################################################
    # profileMode property
    ################################################################
    @property
    def profileMode(self):
        # Get the current verbosity level.
        return self._profileMode
    @profileMode.setter
    def profileMode(self, level):
        # Set the verbosity level.
        if level < 0 or level > 2:
            print("(pglProfile) profileMode must be either 0 (off), 1 (dropped frames), or 2 (detailed).")
        else:
            # set the verbosity level
            self._profileMode = level

        # If profileMode is set
        if self._profileMode > 0:
            # get frameRate of screen
            _,_,frameRate,_ = self.getResolution()
            # set default profileModeBufferSize if not set
            if self.profileModeBufferSize is None:
                # get frameRate
                # set buffer size to 60 seconds worth of frames
                self.profileModeBufferSize = int(frameRate * 60)
                # print the buffer size
                if self.verbose > 0:
                    print(f"(pglProfile) profileModeBufferSize set to {self.profileModeBufferSize} frames ({self.profileModeBufferSize / frameRate:.2f} seconds)")
                    print(f"(pglProfile) Will reallocate if this is exceeded, but you can change this with pgl.profileModeBufferSize = <new size>")
                # init buffer 
                self.profileModeFlushBuffer = np.zeros(self.profileModeBufferSize)
                self.profileModeBufferIndex = 0
                if self._profileMode >= 2:
                    # If profileMode is set to detailed, initialize the profileInfo dict
                    self.profileModeCommandResults =[{} for _ in range(self.profileModeBufferSize)]
                else:
                    self.profileModeCommandResults = None
            # initialize information about the profile
            self.profileInfo['startTime'] = time.time()
            localTime = time.localtime(self.profileInfo['startTime'])
            self.profileInfo['startTimeStr'] = time.strftime("%Y-%m-%d %H:%M:%S", localTime)

            self.profileInfo['profileMode'] = self._profileMode
            self.profileInfo['profileModeBufferStartSize'] = self.profileModeBufferSize
            self.profileInfo['frameRate'] = frameRate
            # save screen info
            self.getWindowFrameInDisplay()
            self.profileInfo['whichScreen'] = self.whichScreen
            self.profileInfo['screenX'] = self.screenX
            self.profileInfo['screenY'] = self.screenY
            self.profileInfo['screenWidth'] = self.screenWidth
            self.profileInfo['screenHeight'] = self.screenHeight
        else:
            # If profileMode is set to off, then save the profile information
            if self.profileModeBufferIndex > 0:
                print("(pglProfile) profileMode is off, saving profile data.")
                # set end time
                self.profileInfo['endTime'] = time.time()
                localTime = time.localtime(self.profileInfo['endTime'])
                self.profileInfo['endTimeStr'] = time.strftime("%H:%M:%S", localTime)

                # package into the dict
                self.profileInfo['flushTimes'] = self.profileModeFlushBuffer[:self.profileModeBufferIndex-1]
                # Save command results (can be none if profileMode is 1)
                self.profileInfo['commandResults'] = self.profileModeCommandResults[:self.profileModeBufferIndex-1]
                # Save the profile information to the profileList
                self.profileList.append(self.profileInfo)
                # Reset the buffer index
                self.profileModeBufferIndex = 0

        # Print the new verbosity level
        if self.verbose > 0: print(f"(pglProfile) profileMode set to {self._profileMode}")

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
                nFrames = len(flushTimes)
                totalTime = flushTimes[-1] - flushTimes[0]
                expectedFrameTime = 1 / profileInfo['frameRate']*1000  # in ms
                # get the difference in times
                frameTimes = np.diff(flushTimes)
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
                                self.printCommandResults(commandResults[iDroppedFrame],commandResults[iDroppedFrame-1]['processedTime'],prefix=f"    ")
                            else:
                                self.printCommandResults(commandResults[iDroppedFrame],prefix=f"    ")

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

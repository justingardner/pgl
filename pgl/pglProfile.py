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
    profileModeBufferIndex = 0
    profileList = []

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
            # set default profileModeBufferSize if not set
            if self.profileModeBufferSize is None:
                # get frameRate
                _,_,frameRate,_ = self.getResolution()
                # set buffer size to 60 seconds worth of frames
                self.profileModeBufferSize = int(frameRate * 60)
                # print the buffer size
                if self.verbose > 0:
                    print(f"(pglProfile) profileModeBufferSize set to {self.profileModeBufferSize} frames ({self.profileModeBufferSize / frameRate:.2f} seconds)")
                    print(f"(pglProfile) Will reallocate if this is exceeded, but you can change this with pgl.profileModeBufferSize = <new size>")
                # init buffer FIX, FIX, FIX deal with profile mode = 2 here
                self.profileModeFlushBuffer = np.zeros(self.profileModeBufferSize)
                self.profileModeBufferIndex = 0
        else:
            # If profileMode is set to off, then save the profile information
            if self.profileModeBufferIndex > 0:
                print("(pglProfile) profileMode is off, saving profile data.")
                # Save the profile information to the profileList
                self.profileList.append(self.profileModeFlushBuffer[:self.profileModeBufferIndex-1])
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
            for iProfile, profile in enumerate(self.profileList):
                # display the number of frames in the profile
                print(f"(pglProfile) Profile {iProfile+1}: {len(profile)} frames")
                # get the difference in times
                frameTimes = np.diff(profile)
                # compute stats
                meanFrameTime = np.mean(frameTimes)
                stdFrameTime = np.std(frameTimes)
                droppedFrames = np.sum(frameTimes > meanFrameTime + 5 * stdFrameTime)
                # display the mean and std of the frame times
                print(f"(pglProfile) Mean frame time: {meanFrameTime*1000:.2f} ms, std: {stdFrameTime*1000:.2f} ms")
                print(f"(pglProfile) Dropped frames: {droppedFrames}")  

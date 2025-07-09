################################################################
# # pgl.py: Main module for the pgl psychophysics and experiment library
################################################################

#############
# Import modules
#############
import platform, subprocess, pprint

from ._socket import _socket
from . import _displayInfo

#############
# Main class
#############
class pgl:
    ################################################################
    # Variables
    ################################################################
    _verbose = 1 # verbosity level, 0 = silent, 1 = normal, 2 = verbose
    macOSversion = None
    hardwareInfo = None

    ################################################################
    # Init Function
    ################################################################
    def __init__(self):
        # check os
        if not self.checkOS():
            raise Exception("(pgl) Unsupported OS")
                
        # start socket library
        self.s = _socket()

        # Init verbose level to normal
        self.verbose = 1

        # print what we are doing
        if self.verbose > 0: print("(pgl) Main library instance created")
    
    ################################################################
    # Verbose property
    ################################################################
    @property
    def verbose(self):
        # Get the current verbosity level.
        return self._verbose
    @verbose.setter
    def verbose(self, level):
        # Set the verbosity level.
        if level < 0 or level > 2:
            print("(pgl) Verbosity level must be between 0 and 2")
        else:
            # set the verbosity level
            self._verbose = level
            # tell the displayInfo c-code library to set the verbosity level
            _displayInfo.setVerbose(level)

        # Print the new verbosity level
        if self._verbose > 0: print(f"(pgl) Verbosity level set to {self._verbose}")

    #################################################################
    # Test function  
    #################################################################
    def helloworld(self):
        # print hello world
        if self.verbose > 0: print("(pgl) Hello World!")
        
    ################################################################
    # Open a screen
    ################################################################
    def open(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        # Print what we are doing
        if self.verbose > 0: print(f"(pgl:open) Opening screen {whichScreen} with dimensions {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")
        # Here you would add the code to actually open the screen using a graphics library
        return True
    
    ################################################################
    # Check OS compatibility
    ################################################################
    def checkOS(self):
        """
        Check if the current operating system is macOS.
        """

        if platform.system() == "Darwin":
            # get version
            self.macOSversion = platform.mac_ver()
            # get hardware info
            try:
                hardwareInfo = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # parse into a dict for easier access
                lines = hardwareInfo.stdout.splitlines()
                self.hardwareInfo = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        # add to dict
                        self.hardwareInfo[key] = value

            except subprocess.CalledProcessError as e:
                self.hardwareInfo = f"Error retrieving hardware info: {e}"

            # Print the macOS version and hardware info
            if self.verbose > 0: print("(pgl:checkOS) Running on macOS version:", self.macOSversion[0])
            if self.verbose > 1: 
                print("(pgl:checkOS) Hardware info")
                pprint.pprint(self.hardwareInfo)
            return True
        else:
            # not macOS
            print("(pgl:checkOS) PGL is only supported on macOS")
            return False
    
    ################################################################
    # Get the display resolution
    ################################################################
    def getResolution(self, whichScreen):
        """
        Get the resolution and display settings for a given screen.

        This function retrieves the width, height, refresh rate, and bit depth of the specified
        display using the underlying `_displayInfo` compiled extension. 

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays.

        Returns:
            tuple[int, int, int, int]: A 4-tuple containing:
                - width (int): Screen width in pixels.
                - height (int): Screen height in pixels.
                - refresh_rate (int): Refresh rate in Hz.
                - bit_depth (int): Color depth in bits per pixel.

        Raises:
            None. Errors are signaled by a return value of (-1, -1, -1, -1)

        Verbose Mode:
            Module-level 'verbose' (pgl.verbose) can be set to display:
                - 1 Screen resolution, refresh rate and bit depth
                - 2 all available modes for the display

        Author:
            JLG

        Date:
            July 9, 2025
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:getResolution) Getting resolution for screen {whichScreen}")

        # Call the C function to get the display info
        return _displayInfo.getResolution(whichScreen)
    
    ################################################################
    # Set the display resolution
    ################################################################
    def setResolution(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:setResolution) Setting resolution for screen {whichScreen} to {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")

        # Call the C function to set the display info
        return _displayInfo.setResolution(whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth)  
    ################################################################
    # Get the number of displays and the default display
    ################################################################
    def getNumDisplaysAndDefault(self):
        # Print what we are doing
        if self.verbose > 1: print("(pgl:getNumDisplaysAndDefault) Getting number of displays and default display")
        
        # Call the C function to get the number of displays and the default display
        return _displayInfo.getNumDisplaysAndDefault()     
    
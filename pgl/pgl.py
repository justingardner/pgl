################################################################
# # pgl.py: Main module for the pgl psychophysics and experiment library
################################################################

#############
# Import modules
#############
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
        print(f"(pgl) Verbosity level set to {self._verbose}")

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
    # TODO: Check OS compatibility
    ################################################################
    def checkOS(self):
        # Check here that the OS is supported
        # For now, we assume all OSes are supported
        print("(pgl:checkOS) TODO: Check OS compatibility")
        return True
    
    ################################################################
    # Get the display resolution
    ################################################################
    def getResolution(self, whichScreen):
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
    
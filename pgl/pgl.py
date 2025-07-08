################################################################
# # pgl.py: Main module for the pgl psychophysics and experiment library
################################################################

#############
# Import socket module for communicating with mgl standalone
#############
from ._socket import _socket

#############
# Main class
#############
class pgl:
    verbose = 1; # verbosity level, 0 = silent, 1 = normal, 2 = verbose

    # Init Function
    def __init__(self):
        # start socket library
        self.s = _socket();
        # check os
        if not self.checkOS():
            raise Exception("(pgl) Unsupported OS")
        
        # print what we are doing
        if self.verbose > 0: print("(pgl) Main library instance created")

    # Test function
    def helloworld(self):
        # print hello world
        if self.verbose > 0: print("(pgl) Hello World!")
        
    
    def open(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        # Print what we are doing
        if self.verbose > 0: print(f"(pgl:open) Opening screen {whichScreen} with dimensions {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")
        # Here you would add the code to actually open the screen using a graphics library
        return True
    
    # TODO
    def checkOS(self):
        # Check here that the OS is supported
        # For now, we assume all OSes are supported
        print("(pgl:checkOS) TODO: Check OS compatibility")
        return True
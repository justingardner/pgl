################################################################
#   filename: pglDraw.py
#    purpose: Functions for drawing to the screen
#             for the pgl psychophysics and experiment library
#         by: JLG
#       date: July 9, 2025
################################################################

#############
# Import modules
#############
import numpy as np

#############
# Drawing class
#############
class pglDraw:
    ################################################################
    # Init Function
    ################################################################
    def __init__(self):
        """
        Initialize the pglDraw class.

        This class provides methods for drawing operations on the screen.

        Args:
            None

        Returns:
            None
        """
        pass

    ################################################################
    # clearScreen
    ################################################################
    def clearScreen(self, color):
        """
        Clear the screen with a specified color.

        Args:
            color (list or tuple): RGB color values as a list or tuple of three floats in the range [0, 1].

        Returns:
            bool: True if the screen was cleared successfully, False otherwise.
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:clearScreen) Clearing screen with color {color}")

        # Check if the socket is connected
        if not self.s:
            print("(pgl:clearScreen) ❌ Not connected to socket")
            return False
        
        # Send the clear command
        self.s.writeCommand("mglSetClearColor")
        
        # send the color data
        self.s.write(np.array(color, dtype=np.float32))
        
        # Read the command results
        self.commandResults = self.s.readCommandResults()
        
 
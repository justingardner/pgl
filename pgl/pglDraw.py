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
    """
    pglDraw class for drawing operations.
    """

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
        
    ################################################################
    # dots
    ################################################################
    def dots(self, x, y, z=None, dotColor=None, dotSize=None, dotShape=None, dotAntialiasingBorder=None):
        """
        Draw dots

        Args:
            

        Returns:
            bool: True if the dots drew correctly
        """
        # set defaults
        if z is None: z = 0.0
        if dotColor is None: dotColor = np.ones(4)
        if dotSize is None: dotSize = np.ones(2)*10
        if dotShape is None: dotShape = 1
        if dotAntialiasingBorder is None: dotAntialiasingBorder = 0

        # make into an array
        dotData = np.array([1000,y,z,*dotColor,*dotSize,dotShape,dotAntialiasingBorder], dtype=np.float32)

        # send dots commanbd
        self.s.writeCommand("mglDots")
        # send the number of dots
        self.s.write(np.uint32(1))
        # send the data
        self.s.write(dotData)
        # read the command results
        self.s.readCommandResults()


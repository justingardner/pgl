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
            color (list or tuple): RGB color values as a list or tuple of three floats in the range [0, 1]. Scalar
                values will be converted to grayscale.

        Returns:
            bool: True if the screen was cleared successfully, False otherwise.
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:clearScreen) Clearing screen with color {color}")

        # Validate the color
        color = self.validateColor(color,withAlpha = False)

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
        dotData = np.array([x,y,z,*dotColor,*dotSize,dotShape,dotAntialiasingBorder], dtype=np.float32)

        # send dots commanbd
        self.s.writeCommand("mglDots")
        # send the number of dots
        self.s.write(np.uint32(1))
        # send the data
        self.s.write(dotData)
        # read the command results
        self.s.readCommandResults()

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
        dotData = np.array([x,y,z,*dotColor,*dotSize,dotShape,dotAntialiasingBorder], dtype=np.float32)

        # send dots commanbd
        self.s.writeCommand("mglDots")
        # send the number of dots
        self.s.write(np.uint32(1))
        # send the data
        self.s.write(dotData)
        # read the command results
        self.s.readCommandResults()

    ################################################################
    # line
    ################################################################
    def line(self, x1, y1, x2, y2, lineColor=None):
        """
        Draw a line

        Args:
            x1 (float): x coordinate of the start point
            y1 (float): y coordinate of the start point
            x2 (float): x coordinate of the end point
            y2 (float): y coordinate of the end point
            lineColor (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1]

        Returns:
            None
        """
        # set defaults
        if lineColor is None: lineColor = np.ones(3)

        # validate color
        lineColor = self.validateColor(lineColor, withAlpha=False)

        # make into vertex data
        if isinstance(x1, (list, tuple, np.ndarray)):
            # validate that x1, y1, x2, y2 are all the same length
            if not (len(x1) == len(y1) == len(x2) == len(y2)):
                print("(pglDraw:line) x1, y1, x2, y2 must all be the same length.")
                return
            nLines = len(x1)
            # create vertex data for multiple lines
            vertexData = np.array([], dtype=np.float32)
            for iLine in range(nLines):
                # if x1 is a list or array, use the corresponding y1, x2, y2
                vertexData = np.append(vertexData,np.array([x1[iLine], y1[iLine], 0, *lineColor, x2[iLine], y2[iLine], 0, *lineColor], dtype=np.float32))

        else:
            # if x1 is a single value, then draw a single line
            vertexData = np.array([x1, y1, 0, *lineColor, x2, y2, 0, *lineColor], dtype=np.float32)
            nLines = 1

        # send line command
        self.s.writeCommand("mglLine")
        #ack = self.s.readAck()
        # send the number of vertices
        self.s.write(np.uint32(nLines * 2))
        # send the data
        self.s.write(vertexData)
        # read the command results
        self.s.readCommandResults()

    ################################################################
    # fixationCross
    ################################################################
    def fixationCross(self, size=1, x=0, y=0, color=None):
        """
        Draw a fixation cross.

        Args:
            size (float): The size of the cross.
            x (float): The x coordinate of the center of the cross.
            y (float): The y coordinate of the center of the cross.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].

        Returns:
            None
        """
        # validate color
        color = self.validateColor(color, withAlpha=False)

        # draw the horizontal line
        self.line(x - size, y, x + size, y, color)

        # draw the vertical line
        self.line(x, y - size, x, y + size, color)

    ################################################################
    # fixationCross
    ################################################################
    def fixationCross(self, size=1, x=0, y=0, color=None):
        """
        Draw a fixation cross.

        Args:
            size (float): The size of the cross.
            x (float): The x coordinate of the center of the cross.
            y (float): The y coordinate of the center of the cross.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].

        Returns:
            None
        """
        # validate color
        color = self.validateColor(color, withAlpha=False)

        # draw the horizontal line
        self.line(x - size, y, x + size, y, color)

        # draw the vertical line
        self.line(x, y - size, x, y + size, color)

    ################################################################
    # circle
    ################################################################
    def circle(self, radius=1, x=0, y=0, color=None, numSegments=36):
        """
        Draw a circle.

        Args:
            radius (float): The radius of the circle.
            x (float): The x coordinate of the center of the circle.
            y (float): The y coordinate of the center of the circle.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].

        Returns:
            None
        """
        # validate color
        color = self.validateColor(color, withAlpha=False)

        # draw the circle using a series of line segments
        for i in range(numSegments):
            # calculate start and stop angle for segment
            angle1 = 2 * np.pi * i / numSegments
            angle2 = 2 * np.pi * (i + 1) / numSegments
            # draw that segment
            x1 = x + radius * np.cos(angle1)
            y1 = y + radius * np.sin(angle1)
            x2 = x + radius * np.cos(angle2)
            y2 = y + radius * np.sin(angle2)
            self.line(x1, y1, x2, y2, color)

    ####################################################
    # validate color
    ####################################################
    def validateColor(self, color, withAlpha=True):
        """
        Validate the color input.

        Args:
            color (list or tuple): RGB color values as a list or tuple of three floats in the range [0, 1]. IF a scalar
            color is provided, it will be converted to a grayscale color. 

        Returns:
            np.ndarray: A numpy array of the validated color.
        """
        if color is None:
            print("(pglDraw:validateColor) Color is None. Defaulting to white.")
            color = [1.0, 1.0, 1.0]

        # If a scalar is provided, convert it to grayscale
        if isinstance(color, (int, float)):
            color = [color, color, color]

        if not isinstance(color, (list, tuple, np.ndarray)):
            print("(pgldraw:validateColor) Color must be a list or tuple of three floats. Defaulting to white.")
            color = [1.0, 1.0, 1.0]

        if len(color) < 3:
            print("(pgldraw:validateColor) Color must be a list or tuple of three floats. Defaulting to white.")
            color = [1.0, 1.0, 1.0]

        if withAlpha and len(color) < 4:
            # If withAlpha is True, ensure the color has an alpha channel
            color = list(color) + [1.0]
        
        if not withAlpha and len(color) > 3:
            # say what we are doing
            if self.verbose>0: print("(pglDraw:validateColor) Ignoring alpha channel in color.")
            # If withAlpha is False, ignore the alpha channel
            color = color[:3]   

        # Convert to numpy array and ensure it's float32
        return np.array(color, dtype=np.float32)

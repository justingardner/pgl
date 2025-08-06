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
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

#############
# Drawing class
#############
class pglDraw:
    """
    pglDraw class for drawing operations.
    """
    def __init__(self):   
        # set current line starting at top of screen
        self.currentLine = 1
    

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
    def dots(self, x, y, z=None, color=None, dotSize=None, dotShape=None, dotAntialiasingBorder=None):
        """
        Draw dots

        Args:
            

        Returns:
            bool: True if the dots drew correctly
        """
        # set defaults
        if z is None: z = 0.0
        if color is None: color = np.ones(4)
        if dotSize is None: dotSize = np.ones(2)*10
        if dotShape is None: dotShape = 1
        if dotAntialiasingBorder is None: dotAntialiasingBorder = 0

        # make into an array
        dotData = np.array([x,y,z,*color,*dotSize,dotShape,dotAntialiasingBorder], dtype=np.float32)

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
    def line(self, x1, y1, x2, y2, color=None, units=None):
        """
        Draw a line

        Args:
            x1 (float): x coordinate of the start point
            y1 (float): y coordinate of the start point
            x2 (float): x coordinate of the end point
            y2 (float): y coordinate of the end point
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1]

        Returns:
            None
        """
        # set defaults
        if color is None: color = np.ones(3)

        # validate color
        color = self.validateColor(color, withAlpha=False)

        # Convert units if necessary
        if units is None:
            pass
        elif units.lower() in ("pixels","pix","pixel","px"):
            x1, y1 = self.pix2deg(x1, y1)
            x2, y2 = self.pix2deg(x2, y2)
        elif units != "device":
            print(f"(pglDraw:line) Invalid units '{units}'. Using deg units.")

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
                vertexData = np.append(vertexData,np.array([x1[iLine], y1[iLine], 0, *color, x2[iLine], y2[iLine], 0, *color], dtype=np.float32))

        else:
            # if x1 is a single value, then draw a single line
            vertexData = np.array([x1, y1, 0, *color, x2, y2, 0, *color], dtype=np.float32)
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
    ####################################################
    # arcs
    ####################################################
    def arc(self, x, y, innerRadius=0, outerRadius=1, startAngle=0, stopAngle=np.pi, borderSize=0.1, color=None):
        """
        Draw arcs.

        Args:
            x (float or list): x coordinate(s) of the center of the arc(s).
            y (float or list): y coordinate(s) of the center of the arc(s).
            innerRadius (float): The inner radius of the arc.
            outerRadius (float): The outer radius of the arc.
            startAngle (float): The starting angle of the arc in radians.           
            stopAngle (float): The stopping angle of the arc in radians.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].

        Returns:
            None
        """
        # validate color
        color = self.validateColor(color)

        # get vertex data
        vertexData = [x, y, 0, *color, innerRadius, outerRadius, innerRadius, outerRadius, startAngle, stopAngle, borderSize]
        print(np.array(vertexData, dtype=np.double))
        # send arc command
        self.s.writeCommand("mglArcs")
        ack = self.s.readAck()

        # send the number of arcs
        self.s.write(np.uint32(1))
        # send the vertex data
        self.s.write(np.array(vertexData, dtype=np.float32))

        # read the command results
        results = self.s.readCommandResults(ack)
    ######################################################
    # text
    ######################################################
    def text(self, str, x=0, y=None, color=None, fontSize=40, fontName="Helvetica", line=None):
        """
        Draw text on the screen.

        Args:
            str (str): The text to draw.
            x (float): The x coordinate of the text. 
            y (float): The y coordinate of the text. If omitted and line is not set
                        then the text will be drawn at the top of the screen and
                        each subsequent call will draw text one line below until
                        the bottom of the screen is reached, upon which it will
                        revert to the top of the screen.
            line (int): The line number to draw the text on. IF this
                        is set, y will be ignored and instead the text
                        will be drawn on the specified line from top of screen.
                        If line is negative, then from the bottom of the screen.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            fontSize (int): The size of the font.
            fontName (str): The name of the font.


        Returns:
            None
        """
        # validate color
        color = self.validateColor(color)

        # Load Font
        fontPath = "/System/Library/Fonts"
        p = Path(fontName)
        # add ttf suffix if not specified
        if not p.suffix: fontName += ".ttc"
        if p.is_absolute():
            # If the fontName is an absolute path, use it directly
            fontFullName = fontName
        else:
            # join with default font path
            fontFullName = os.path.join(fontPath, fontName)
        try:
            # Try to load the specified font
            font = ImageFont.truetype(fontFullName, fontSize)
        except:
            print(f"(pglDraw:text) Failed to load font '{fontName}' from '{fontPath}'. Using default font.")
            fontFullName = os.path.join(fontPath,"Helvetica.ttc")
            font = ImageFont.truetype(fontFullName, fontSize)
        
        # padding around text
        padding = 11

        # Create a dummy image to measure text size
        dummyImg = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummyImg)

        # Get bounding box for text sample of characters that are usually the tallest
        # to get font height, get text width from actual string
        bbox = draw.textbbox((0, 0), "HITLFEfhkl", font=font)
        textHeight = bbox[3] - bbox[1]
        bbox = draw.textbbox((0, 0), str, font=font)
        textWidth = bbox[2] - bbox[0]

        # Create an image with transparent background
        img = Image.new("RGBA", (textWidth + 2 * padding, textHeight + 2 * padding), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw anti-aliased text
        draw.text((padding, padding), str, font=font, fill=tuple(int(255*c) for c in color))

        # get text height in degrees
        textHeight = textHeight * self.yPix2Deg
        padding = padding * self.yPix2Deg
        lineHeight = textHeight + padding * 2

        if line is None and y is None:
            # get current line number
            line = self.currentLine
            # check to see if we will go over the bottom
            if ((line+1) * lineHeight) > self.screenHeight.deg:
                # reset next line number to 1
                self.currentLine = 1
            else:
                self.currentLine += 1


        # calculate line if necessary
        if line is not None:
            if line>0:
                # if line is specified, calculate y based on line number
                y =  self.screenHeight.deg / 2 - textHeight/2 - (line - 1) * (textHeight + padding * 2) - padding
            else:
                # if line is negative, calculate y based on line number from the bottom
                y = -self.screenHeight.deg / 2 + textHeight/2 + (-line - 1) * (textHeight + padding * 2) + padding
            
        # create the image and display
        img = self.imageCreate(np.array(img))
        img.display(displayLocation=(x,y))

    def test(self):
        try:
            import signal
            # pause the signal handler for Ctrl-C
            originalHandler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.waitSecs(5)
            print("Test complete")
        finally:
            # Restore original handler so Ctrl-C works again
            signal.signal(signal.SIGINT, originalHandler)


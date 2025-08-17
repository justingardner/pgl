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
    def dots(self, x, y, z=0, color=None, dotSize=np.ones(2)*0.2, dotShape=1, dotAntialiasingBorder=0):
        """
        Draw dots

        Args:
            x (float or array-like): The x coordinates of the dots.
            y (float or array-like): The y coordinates of the dots.
            z (float or array-like, optional): The z coordinates of the dots (default is 0.0).
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            dotSize: The size of the dots, if scalar, then both x and y dimension will be the same. 
                    If a list or array of length 2, then the first value is the x dimension and the second value is the y dimension.
                    Units are in degrees. Note that dots are meant to render less than 1 degree or so
                    each, so size is not guaranteed if they are larger than that. Default is 0.2 degrees
            dotShape (int, optional): The shape of the dots 0=rectangular, 1=circular (default is 1).
            dotAntialiasingBorder (float, optional): The antialiasing border size in pixels(default is 0).

        Returns:
            bool: True if the dots were drawn successfully, False otherwise.
        """
        # Convert inputs to 1D arrays (even if scalar)
        x = np.atleast_1d(np.ravel(x))
        y = np.atleast_1d(np.ravel(y))

        # Check that lengths match
        if not (x.shape == y.shape):
            print("(pglDraw:dots) x, y must all be the same length.")
            return False

        # Init dotVertexData with appropriate size matrix
        n = x.shape[0]
        dotVertexData = np.zeros((n, 11), dtype=np.float32)
        dotVertexData[:, 0] = x
        dotVertexData[:, 1] = y

        # If z is provided, ensure it matches the length of x and y
        z = np.atleast_1d(np.ravel(z)).astype(np.float32)
        # check if z is a scalar
        if z.shape[0] not in (1,n):
            # If z is an array, it must match the length of x and y
            print("(pglDraw:dots) z must be the same length as x and y or a scalar.")
            return False            

        # Validate color and put into matrix
        color = self.validateColor(color, withAlpha=True, n=n)
        dotVertexData[:, 3:7] = color 

        # Validate dotSize and put into matrix. 
        dotSize = np.atleast_1d(dotSize).astype(np.float32)
        if dotSize.shape[0] == 1: dotSize = [dotSize[0], dotSize[0]]
        dotSize[0] *= self.xDeg2Pix
        dotSize[1] *= self.yDeg2Pix
        dotVertexData[:, 7:9] = dotSize

        # Validate dotShape and put into matrix
        dotShape = np.atleast_1d(np.ravel(dotShape)).astype(np.float32)
        dotVertexData[:, 9] = dotShape

        # Validate dotAntialiasingBorder and put into matrix
        dotAntialiasingBorder = np.atleast_1d(np.ravel(dotAntialiasingBorder)).astype(np.float32)
        dotVertexData[:, 10] = dotAntialiasingBorder

        # send dots command
        self.s.writeCommand("mglDots")
        # send the number of dots
        self.s.write(np.uint32(n))
        # send the data
        self.s.write(dotVertexData)
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

    ################################################################
    # quad
    ################################################################
    def quad(self, vertices, color=None):
        '''
        Draw a quad

        Args:
            vertices(np.array): n x 4 x 2 matrix of vertices. You can omit
            the nth dimension and just pass a 4x2 matrix for one quad
            color:
        '''
        
        # get the vertices as an n x 4 x 2 matrix
        vertices = np.atleast_3d(np.array(vertices, dtype=np.float32))
        if vertices.shape[-1] == 1: vertices = np.moveaxis(vertices, -1, 0)

        # check dimensions
        if vertices.shape[1] != 4 or vertices.shape[2] != 2:
            print("(pglDraw:quad) Vertices must be an n x 4 x 2 matrix, but got shape", vertices.shape)
            return

        # get the number of quads
        nQuads = vertices.shape[0]

        # get the color
        color = self.validateColor(color, n=nQuads, withAlpha=False, forceN=True)

        # Each quad is actually two triangles put together
        # these are the indices for each of the triangles
        # ie 0,1,2 and 2,3,0
        quadTriangleIndices = np.array([0, 1, 2, 2, 3, 0])

        # Create an array of vertex data which is x y z r g b for each vertex
        vertexData = np.hstack([
            # reshape the vertices to be 6n x 2
            # where the 6 indexes are the two triangles vertices
            vertices[:,quadTriangleIndices,:].reshape(-1,2),
            # add a column of zeros for z
            np.zeros((nQuads*6,1)),
            # add the color for each vertex
            np.repeat(color[:, np.newaxis, :], 6, axis=1).reshape(-1, 3)
        ]).astype(np.float32)

        # send quad command
        self.s.writeCommand("mglQuad")
        # send the number of vertices
        self.s.write(np.uint32(6 * nQuads))
        # send the data
        self.s.write(vertexData)
        # read the command results
        self.s.readCommandResults()

    ####################################################
    # validate color
    ####################################################
    def validateColor(self, color, withAlpha=True, n=None, forceN=False):
        """
        Validate the color input.

        Args:
            color (list or tuple): RGB color values as a list or tuple of three floats in the range [0, 1]. IF a scalar
            color is provided, it will be converted to a grayscale color. 

        Returns:
            np.ndarray: A numpy array of the validated color.
        """
        if color is None:
            color = np.array([[1.0, 1.0, 1.0]]).astype(np.float32)  # Default to white if no color is provided

        # convert to numpy array
        color = np.atleast_2d(np.array(color, dtype=np.float32))

        if n is None:
            if color.shape[0]>1:
                # If n is not specified, just use the first color
                color = np.atleast_2d(color[0,:])
                print(f"(pgldraw:validateColor) {color.shape[0]} colors provided, using only first color.")
        else:
            if color.shape[0] not in (1,n):
                # If n is specified, ensure the color matches the length of n
                print(f"(pgldraw:validateColor) {color.shape[0]} colors provided, but expected {n}. Using only the first color.")
                color = np.atleast_2d(color[0,:])
            if forceN and color.shape[0] != n:
                color = np.repeat(color, n, axis=0)

        # If a scalar is provided, convert it to grayscale
        if color.shape[1] == 1:
            color[:,2] = color[:,1]
            color[:,3] = color[:,1]

        if color.shape[1] < 3:
            print("(pgldraw:validateColor) Color must be a list or tuple of three floats. Defaulting to white.")
            color = np.array([[1.0, 1.0, 1.0]]).astype(np.float32)    

        if withAlpha and color.shape[1] < 4:
            # If withAlpha is True, ensure the color has an alpha channel
            color =  np.column_stack((color, np.full((color.shape[0],), 1.0)))
        
        if not withAlpha and color.shape[1] > 3:
            # say what we are doing
            if self.verbose>0: print("(pglDraw:validateColor) Ignoring alpha channel in color.")
            # If withAlpha is False, ignore the alpha channel
            color = color[:,0:3]   

        return(color)
    
    
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
        linesPerScreen = int(self.screenHeight.deg / lineHeight)
        
        if line is None and y is None:
            # get current line number
            line = self.currentLine
            # and update
            self.currentLine += 1

        # negative line numbers
        if line < 0: line = linesPerScreen + line + 1

        # calculate line if necessary
        if line is not None:
            # if line is specified, calculate y based on line number
            y =  self.screenHeight.deg / 2 - textHeight/2 - (line - 1) * (textHeight + padding * 2) - padding
            self.currentLine = line+1

        # check to see if we will go over the bottom
        if self.currentLine > linesPerScreen: self.currentLine = 1
   
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


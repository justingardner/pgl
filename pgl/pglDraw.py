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
import math
import numpy as np
from .pglImage import pglImageInstance
from .pglMessages import pglMessages

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
        
        self.textRenderingParameters = {
            "fontName": "Helvetica",
            "fontSize": 40.0,
            "paddingPixels": 11,
            "foregroundColor": (1.0, 1.0, 1.0, 1.0),
            "backgroundColor": (0.0, 0.0, 0.0, 0.0)
        }
        
        # Default clearScreen behavior. Experiment apertures temporarily
        # enable quad clearing so the render-pass background stays black.
        self.clearScreenDrawQuad = False

    ################################################################
    # clearScreen
    ################################################################
    def clearScreen(self, color, drawQuad=None):
        """
        Set the render-pass clear color or draw a full-screen background quad.

        Args:
            color: RGB values in [0, 1], or a scalar grayscale value.
            drawQuad (bool or None): None uses self.clearScreenDrawQuad.
                False sets the renderer's clear color.
                True draws a full-screen quad through the current stencil
                without changing the renderer's clear color.
        Returns:
            bool: True after submitting the requested operation,
                False if disconnected or the clear-color command fails.

        Notes:
            Setting the clear color affects render passes created afterward;
            it does not retroactively clear a pass already in progress.

            The quad uses an identity transform to cover the screen, then
            restores the cached drawing transform. Stencil testing remains
            active throughout.
        """
        
        # Explicit arguments override the current instance default.
        if drawQuad is None: drawQuad = self.clearScreenDrawQuad
        
        # Print what we are doing.
        if self.verbose > 1:
            print(f"(pgl:clearScreen) Clearing screen with color {color}, drawQuad={drawQuad}")

        # Validate the color.
        color = self.validateColor(color, withAlpha=False)

        # Check if the socket is connected.
        if not self.s:
            print("(pgl:clearScreen) ❌ Not connected to socket")
            return False

        if drawQuad:
            # Cover the screen independently of visual-angle calibration,
            # display flips, or the current coordinate transform.
            fullScreenQuad = np.array([
                [-1, -1],
                [ 1, -1],
                [ 1,  1],
                [-1,  1]
            ], dtype=np.float32)

            savedTransform = np.array(self.xform, copy=True)

            try:
                # setTransform() sends the matrix without changing self.xform.
                self.setTransform(np.eye(4))
                self.quad(fullScreenQuad, color=color, units="device")
            finally:
                self.setTransform(savedTransform)

        else:
            # Set the background color used when a render pass begins.
            if not self.s.writeCommand("mglSetClearColor"): return False

            self.s.write(np.array(color, dtype=np.float32))
            self.commandResults = self.s.readCommandResults()

            if self.commandResults is None: return False
            success = self.commandResults.get("success")
            if success is None or not np.all(np.asarray(success) != 0):
                pglMessages.warning("Could not set the render background color.")
                return False

        # Remember the requested background color in either mode.
        # In quad mode, this is not necessarily the renderer's clear color.
        self.clearScreenColor = color.flatten().tolist()
        return True
         
    ################################################################
    # dots
    ################################################################
    def dots(self, x, y, z=0, color=None, dotSize=np.ones(2)*0.2, dotShape=1, dotAntialiasingBorder=0, units=None):
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
            units (string, optional): Defaults to deg, but can be set to "pix" for pixels

        Returns:
            bool: True if the dots were drawn successfully, False otherwise.
        """
        # Convert units if necessary
        if units is None:
            pass
        elif units.lower() in ("pixels","pix","pixel","px"):
            x, y = self.pix2deg(x, y)
            # try to convert dotSize - note that this is a bit inaccurate if
            # x and y are scaled differently (which is typical). Taking
            # the larger of the two to ensure as large as
            (x1,y1) = self.pix2deg(dotSize,dotSize)
            (x2,y2) = self.pix2deg(0,0)
            xDotSize = np.abs(x1-x2)
            yDotSize = np.abs(y1-y2)
            dotSize = np.maximum(xDotSize, yDotSize)
        elif units != "device":
            print(f"(pglDraw:line) Invalid units '{units}'. Using deg units.")

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
        color = self.validateColor(color, withAlpha=False, n = 1)
        color = color.flatten()
        
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
    def fixationCross(self, size=1, x=0, y=0, color=None, units=None):
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
        self.line(x - size/2, y, x + size/2, y, color, units=units)

        # draw the vertical line
        self.line(x, y - size/2, x, y + size/2, color, units=units)

    ################################################################
    # fixationABC
    ################################################################
    def fixationABC(self, size=1, x=0, y=0, color=1, backgroundColor =0, units=None):
        '''
        Draw ABC fixation cross from Thaler, Schütz, Goodale & Gegenfurtner (2013) Vision Research 76:31-42

        Args:
            size (float): size of fixation cross in degrees
            x (float): The x coordinate of the center of the cross.
            y (float): The y coordinate of the center of the cross.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            backgroundColor (list or tuple, optional): color for background disc
        
        Returns:
            None
        '''

        self.arc(x,y,0,size/2,stopAngle=2*np.pi,borderSize=0,color=backgroundColor)
        self.rect(x-size/2,y,width=size,height=size/4,color=color,hAlign='left',vAlign='center')
        self.rect(x,y-size/2,width=size/4,height=size,color=color,hAlign='center',vAlign='top')
        self.arc(x,y,0,size/10,stopAngle=2*np.pi,borderSize=0,color=backgroundColor)
        
    ################################################################
    # circle
    ################################################################
    def circle(self, radius=1, x=0, y=0, color=None, numSegments=36, fill=False, units=None):
        """
        Draw a circle. Note that arc is more general, and possibly faster (but does not work in pixel coordinates)

        Args:
            radius (float): The radius of the circle.
            x (float): The x coordinate of the center of the circle.
            y (float): The y coordinate of the center of the circle.
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            numSegments (int, optional): The number of segments to use to approximate the circle (default is 36).
            fill (bool, optional): Whether to fill the circle (default is False).
            units (str, optional): The units for the coordinates and radius. If 'pix' then pixels, otherwise degrees.

        Returns:
            None
        """
        # validate color
        color = self.validateColor(color, withAlpha=False)
        
        # if radius is a scalar, make it into a list
        if np.isscalar(radius): radius = [radius, radius]
        
        # Convert units if necessary
        if units is None:
            pass
        elif units.lower() in ("pixels","pix","pixel","px"):
            x, y = self.pix2deg(x, y)
            radius[0] = radius[0] * self.xPix2Deg
            radius[1] = radius[1] * self.yPix2Deg
        elif units != "device":
            print(f"(pglDraw:circle) Invalid units '{units}'. Using deg units.")

        if fill:
            quad = np.zeros((numSegments,4,2), dtype=np.float32)
            # draw the circle using a series of line segments
            for i in range(numSegments):
                # calculate start and stop angle for segment
                angle1 = 2 * np.pi * i / numSegments
                angle2 = 2 * np.pi * (i + 1) / numSegments
                # draw that segment
                x1 = x + radius[0] * np.cos(angle1)
                y1 = y + radius[1] * np.sin(angle1)
                x2 = x + radius[0] * np.cos(angle2)
                y2 = y + radius[1] * np.sin(angle2)
                # make it into a quad
                quad[i,:,:] = np.array([[x, y], [x1, y1], [x2, y2], [x, y]])
            self.quad(quad, color=color)
        else:
            # draw the circle using a series of line segments
            for i in range(numSegments):
                # calculate start and stop angle for segment
                angle1 = 2 * np.pi * i / numSegments
                angle2 = 2 * np.pi * (i + 1) / numSegments
                # draw that segment
                x1 = x + radius[0] * np.cos(angle1)
                y1 = y + radius[1] * np.sin(angle1)
                x2 = x + radius[0] * np.cos(angle2)
                y2 = y + radius[1] * np.sin(angle2)
                self.line(x1, y1, x2, y2, color)

    ####################################################
    # ellipse
    ####################################################
    def ellipse(
        self,
        width=1,
        height=1,
        x=0,
        y=0,
        color=None,
        numSegments=72,
        fill=False,
        rotation=0,
        units=None
    ):
        """
        Draw an ellipse.

        Args:
            width (float): Full width of the ellipse.
            height (float): Full height of the ellipse.
            x (float): X coordinate of the center.
            y (float): Y coordinate of the center.
            color (list or tuple, optional): RGB color values in [0, 1].
            numSegments (int): Number of line segments used to approximate
                the ellipse. Default is 72.
            fill (bool): Whether to fill the ellipse. Default is False.
            rotation (float): Rotation of the ellipse in degrees,
                counterclockwise. Default is 0.
            units (str, optional): Units of x, y, width and height.
                "pix" specifies pixels; otherwise degrees.

        Returns:
            None
        """

        # Validate color
        color = self.validateColor(color, withAlpha=False)

        # Validate number of segments
        if not isinstance(numSegments, (int, np.integer)) or numSegments < 3:
            print("(pglDraw:ellipse) numSegments must be an integer >= 3.")
            return

        # Convert dimensions to radii
        radiusX = width / 2
        radiusY = height / 2

        # Convert units if necessary
        if units is None:
            pass
        elif units.lower() in ("pixels", "pix", "pixel", "px"):
            x, y = self.pix2deg(x, y)
            radiusX *= self.xPix2Deg
            radiusY *= self.yPix2Deg
        elif units != "device":
            print(f"(pglDraw:ellipse) Invalid units '{units}'. Using deg units.")

        # Generate angles around the ellipse
        angles = np.linspace(
            0,
            2 * np.pi,
            numSegments + 1,
            dtype=np.float32
        )

        # Generate unrotated ellipse
        ellipseX = radiusX * np.cos(angles)
        ellipseY = radiusY * np.sin(angles)

        # Convert rotation from degrees to radians internally
        rotationRadians = np.deg2rad(rotation)

        # Apply rotation
        if rotationRadians != 0:
            cosRotation = np.cos(rotationRadians)
            sinRotation = np.sin(rotationRadians)

            rotatedX = (
                ellipseX * cosRotation -
                ellipseY * sinRotation
            )

            rotatedY = (
                ellipseX * sinRotation +
                ellipseY * cosRotation
            )

            ellipseX = rotatedX
            ellipseY = rotatedY

        # Translate to requested center
        ellipseX += x
        ellipseY += y

        if fill:
            # Create one quad for each segment.
            #
            # The four vertices form a triangle fan:
            #
            #       p1
            #      /  \
            #     /    \
            #    C------p2
            #
            # C = center

            vertices = np.zeros(
                (numSegments, 4, 2),
                dtype=np.float32
            )

            for i in range(numSegments):
                vertices[i] = np.array([
                    [x, y],
                    [ellipseX[i], ellipseY[i]],
                    [ellipseX[i + 1], ellipseY[i + 1]],
                    [x, y]
                ], dtype=np.float32)

            self.quad(
                vertices,
                color=color,
                units="device"
            )

        else:
            # Draw all segments in a single mglLine command.
            self.line(
                ellipseX[:-1],
                ellipseY[:-1],
                ellipseX[1:],
                ellipseY[1:],
                color=color,
                units="device"
            )

    ################################################################
    # quad
    ################################################################
    def quad(self, vertices, color=None, units=None):
        '''
        Draw a quad

        Args:
            vertices(np.array): n x 4 x 2 matrix of vertices. You can omit
            the nth dimension and just pass a 4x2 matrix for one quad
            color: color of quad
            units: units of vertices, either "deg" or "pix" (default is "deg")
        e.g.:
            # draw a quad half the dimensions of the screen
            w = self.screenWidth.deg/2
            h = self.screenHeight.deg/2
            vertices = np.array([
                [-w/2, -h/2],   # bottom-left
                [w/2, -h/2],   # bottom-right
                [w/2, h/2],   # top-right
                [-w/2, h/2],   # top-left
            ])
        self.quad(vertices, color=[0.2, 0.7, 0.8])

        '''
        
        # get the vertices as an n x 4 x 2 matrix
        vertices = np.atleast_3d(np.array(vertices, dtype=np.float32))
        if vertices.shape[-1] == 1: vertices = np.moveaxis(vertices, -1, 0)

        # Convert units if necessary
        if units is None:
            pass
        elif units.lower() in ("pixels","pix","pixel","px"):
            vertices[:,:,0], vertices[:,:,1] = self.pix2deg(vertices[:,:,0], vertices[:,:,1])
        elif units != "device":
            print(f"(pglDraw:quad) Invalid units '{units}'. Using deg units.")

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

    ################################################################
    # rect
    ################################################################
    def rect(self, x=0, y=0, width=1, height=1, color=[1, 1, 1], hAlign='center', vAlign='center', units=None):
        '''
        Draw a quad

        Args:
            x (float): x coordinate of the center of the rectangle
            y (float): y coordinate of the center of the rectangle
            width (float): width of the rectangle
            height (float): height of the rectangle
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            hAlign (str): horizontal alignment of the rectangle ('center', 'left', 'right'). Defaults to center
            vAlign (str): vertical alignment of the rectangle ('center', 'top', 'bottom'). Defaults to center
        '''
        # calculate the vertices
        if hAlign.lower() == 'center':
            x1 = x - width / 2
            x2 = x + width / 2
        elif hAlign.lower() == 'left':
            x1 = x
            x2 = x + width
        elif hAlign.lower() == 'right':
            x1 = x - width
            x2 = x
        else:
            print(f"(pglDraw:rect) Invalid hAlign '{hAlign}'. Using 'center'.")
            x1 = x - width / 2
            x2 = x + width / 2
            
        if vAlign.lower() == 'center':
            y1 = y - height / 2
            y2 = y + height / 2
        elif vAlign.lower() == 'top':
            y1 = y
            y2 = y + height
        elif vAlign.lower() == 'bottom':
            y1 = y - height
            y2 = y
        else:
            print(f"(pglDraw:rect) Invalid vAlign '{vAlign}'. Using 'center'.")
            y1 = y - height / 2
            y2 = y + height / 2
            
        vertices = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        self.quad(vertices, color=color, units=units)
            
    ################################################################
    # stencilCreateBegin
    ################################################################
    def stencilCreateBegin(self, stencilNumber=1, invert=False):
        """
        Initialize a stencil plane and begin drawing its mask.

        Call at a render-pass boundary, after flushing any previous drawing.
        Subsequent drawing defines the mask until stencilCreateEnd().

        Args:
            stencilNumber (int): Stencil plane number, from 1 through 7.
            invert (bool or int): False/0 allows drawing inside the mask;
                True/1 allows drawing outside it.

        Returns:
            dict or None: Final command results, or None on failure.

        Notes:
            Initialization includes a flush. Resizing the display invalidates
            existing masks when the renderer recreates its stencil texture.
        """
        # Make sure that a screen is open.
        if not self.isOpen():
            pglMessages.warning("No screen is open.")
            return None

        # Validate arguments before sending any commands.
        if not isinstance(stencilNumber, (int, np.integer)) or not 1 <= stencilNumber <= 7:
            pglMessages.warning("stencilNumber must be an integer from 1 through 7.")
            return None

        if not isinstance(invert, (bool, np.bool_, int, np.integer)) or invert not in (0, 1):
            pglMessages.warning("invert must be False/0 or True/1.")
            return None

        # Preserve the current transform for drawing the caller's mask.
        savedTransform = np.array(self.xform, copy=True)

        # Cover the screen in normalized coordinates, with an extra margin.
        fullScreenQuad = np.array([
            [-2, -2],
            [ 2, -2],
            [ 2,  2],
            [-2,  2]
        ], dtype=np.float32)

        # First initialize the plane with the opposite value, then begin
        # writing the requested mask value.
        for initializing in (True, False):
            passInvert = not bool(invert) if initializing else bool(invert)

            # Start creation before any drawing command opens the render pass.
            if not self.s.writeCommand("mglStartStencilCreation"): return None
            ackTime = self.s.readAck()
            if ackTime is None: return None

            self.s.write(np.uint32(stencilNumber))
            self.s.write(np.uint32(passInvert))
            self.commandResults = self.s.readCommandResults(ackTime)

            # Do not continue if the renderer did not accept the command.
            if self.commandResults is None: return None
            success = self.commandResults.get("success")
            if success is None or not np.all(np.asarray(success) != 0):
                pglMessages.warning("Could not start stencil creation.")
                return None

            if initializing:
                # Use identity coordinates only for the initialization quad.
                # setTransform() sends the matrix without changing self.xform.
                try:
                    self.setTransform(np.eye(4))
                    self.quad(fullScreenQuad, color=1, units="device")
                finally:
                    # Restore INSIDE the initialization pass, before flushing.
                    # Restoring after stencilCreateEnd() would open a new pass
                    # with stencil creation disabled.
                    self.setTransform(savedTransform)

                # Store the initialized stencil plane and end creation.
                if self.stencilCreateEnd() is None: return None

        return self.commandResults

    ################################################################
    # stencilCreateEnd
    ################################################################
    def stencilCreateEnd(self):
        """
        Finish drawing a stencil mask and disable stencil creation.

        Flushes the render pass so Metal stores the updated stencil buffer,
        then returns to ordinary drawing with no stencil selected.

        Returns:
            dict or None: Finish-command results, or None on failure.
        """
        # Make sure that a screen is open.
        if not self.isOpen():
            pglMessages.warning("No screen is open.")
            return None

        # Complete the render pass that writes the stencil buffer.
        # flush() returns a presentation timestamp, not a success boolean.
        self.flush()
        if self.commandResults is None: return None
        success = self.commandResults.get("success")
        if success is None or not np.all(np.asarray(success) != 0):
            pglMessages.warning("Could not flush stencil creation.")
            return None

        # Return to ordinary drawing with no stencil selected.
        if not self.s.writeCommand("mglFinishStencilCreation"): return None
        ackTime = self.s.readAck()
        if ackTime is None: return None

        self.commandResults = self.s.readCommandResults(ackTime)
        if self.commandResults is None: return None
        success = self.commandResults.get("success")
        if success is None or not np.all(np.asarray(success) != 0):
            pglMessages.warning("Could not finish stencil creation.")
            return None

        return self.commandResults

    ################################################################
    # stencilSelect
    ################################################################
    def stencilSelect(self, stencilNumber=0):
        """
        Select a stencil mask for subsequent drawing.

        Args:
            stencilNumber (int): Stencil plane number, or 0 to disable masking.
                The supported upper limit depends on the renderer.

        Returns:
            dict or None: Command results, or None on failure.
        """
        # Make sure that a screen is open.
        if not self.isOpen():
            pglMessages.warning("No screen is open.")
            return None

        # Zero disables masking; positive values select a stencil plane.
        if not isinstance(stencilNumber, (int, np.integer)) or not 0 <= stencilNumber <= np.iinfo(np.uint32).max:
            pglMessages.warning("stencilNumber must be a nonnegative uint32 integer.")
            return None

        # Select the requested stencil.
        if not self.s.writeCommand("mglSelectStencil"): return None
        ackTime = self.s.readAck()
        if ackTime is None: return None

        self.s.write(np.uint32(stencilNumber))
        self.commandResults = self.s.readCommandResults(ackTime)

        if self.commandResults is None: return None
        success = self.commandResults.get("success")
        if success is None or not np.all(np.asarray(success) != 0):
            pglMessages.warning("Could not select stencil.")
            return None

        return self.commandResults        
    
    ################################################################
    # triangulatePolygon
    ################################################################
    @staticmethod
    def triangulatePolygon(vertices):
        """
        Triangulate a simple polygon using ear clipping.

        Args:
            vertices: Numeric n x 2 coordinates, ordered around the perimeter.
                Clockwise and counterclockwise order are both accepted.
                Repeating the first vertex at the end is optional.

        Returns:
            np.ndarray or None: Triangle coordinates with shape (nTriangles, 3, 2),
                or None when the polygon is invalid.

        Supports concave polygons, but not self-intersections or holes.
        Collinear intermediate vertices are removed.
        """
        def invalidPolygon(message):
            pglMessages.warning(f"Could not triangulate polygon: {message}")
            return None

        # Convert and check coordinates before doing geometric operations.
        try:
            vertices = np.asarray(vertices, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return invalidPolygon("coordinates must be numeric x/y pairs.")

        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            return invalidPolygon("at least three x/y pairs are required.")

        if not np.all(np.isfinite(vertices)):
            return invalidPolygon("coordinates must be finite.")

        vertices = vertices.copy()

        # Normalize geometry for dimensionless tolerance checks.
        # Returned triangles still use the original coordinates.
        with np.errstate(over="ignore", invalid="ignore"):
            extent = np.ptp(vertices, axis=0)
            scale = np.max(extent)
            points = (vertices - vertices[0]) / scale if scale > 0 else None

        if points is None or not np.isfinite(scale) or not np.all(np.isfinite(points)):
            return invalidPolygon("coordinates have zero or unsupported extent.")

        tolerance = 1e-10

        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        # Accept an explicitly closed polygon.
        if np.max(np.abs(points[-1] - points[0])) <= tolerance:
            points = points[:-1]
            vertices = vertices[:-1]

        if len(points) < 3:
            return invalidPolygon("at least three distinct vertices are required.")

        # Repeated vertices create ambiguous or degenerate boundaries.
        for index in range(len(points)):
            if np.any(np.max(np.abs(points[index + 1:] - points[index]), axis=1) <= tolerance):
                return invalidPolygon("vertices must not repeat, except for an optional closing vertex.")

        # Remove straight-through intermediate points. Reject edges that
        # reverse direction and overlap along the same line.
        remaining = list(range(len(points)))

        while len(remaining) > 3:
            removed = False

            for position, current in enumerate(remaining):
                previous = remaining[position - 1]
                following = remaining[(position + 1) % len(remaining)]
                a, b, c = points[[previous, current, following]]

                if abs(cross(a, b, c)) <= tolerance:
                    if np.dot(b - a, c - b) < 0:
                        return invalidPolygon("adjacent edges overlap or reverse direction.")

                    remaining.pop(position)
                    removed = True
                    break

            if not removed: break

        def onSegment(a, b, point):
            return (
                abs(cross(a, b, point)) <= tolerance
                and np.all(point >= np.minimum(a, b) - tolerance)
                and np.all(point <= np.maximum(a, b) + tolerance)
            )

        def segmentsIntersect(a, b, c, d):
            abC, abD = cross(a, b, c), cross(a, b, d)
            cdA, cdB = cross(c, d, a), cross(c, d, b)

            # Proper crossing.
            if (
                ((abC > tolerance and abD < -tolerance) or
                 (abC < -tolerance and abD > tolerance))
                and
                ((cdA > tolerance and cdB < -tolerance) or
                 (cdA < -tolerance and cdB > tolerance))
            ):
                return True

            # Also reject touching or overlapping nonadjacent edges.
            return (
                onSegment(a, b, c) or onSegment(a, b, d)
                or onSegment(c, d, a) or onSegment(c, d, b)
            )

        # Check that nonadjacent edges never intersect.
        count = len(remaining)

        for first in range(count):
            a = points[remaining[first]]
            b = points[remaining[(first + 1) % count]]

            for second in range(first + 1, count):
                # Adjacent edges are allowed to share their endpoint.
                if second == first + 1 or (first == 0 and second == count - 1):
                    continue

                c = points[remaining[second]]
                d = points[remaining[(second + 1) % count]]

                if segmentsIntersect(a, b, c, d):
                    return invalidPolygon("nonadjacent edges intersect or touch.")

        # Determine winding and reject polygons with effectively zero area.
        boundary = points[remaining]
        twiceArea = np.sum(
            boundary[:, 0] * np.roll(boundary[:, 1], -1)
            - boundary[:, 1] * np.roll(boundary[:, 0], -1)
        )

        if abs(twiceArea) <= tolerance:
            return invalidPolygon("polygon area is zero or too small.")

        winding = 1 if twiceArea > 0 else -1
        triangles = []

        # An ear is a convex corner whose triangle contains no other vertex.
        while len(remaining) > 3:
            foundEar = False

            for position, current in enumerate(remaining):
                previous = remaining[position - 1]
                following = remaining[(position + 1) % len(remaining)]
                a, b, c = points[[previous, current, following]]

                if winding * cross(a, b, c) <= tolerance:
                    continue

                containsVertex = False

                for candidate in remaining:
                    if candidate in (previous, current, following): continue
                    point = points[candidate]

                    if (
                        winding * cross(a, b, point) >= -tolerance
                        and winding * cross(b, c, point) >= -tolerance
                        and winding * cross(c, a, point) >= -tolerance
                    ):
                        containsVertex = True
                        break

                if containsVertex: continue

                triangles.append([previous, current, following])
                remaining.pop(position)
                foundEar = True
                break

            if not foundEar:
                return invalidPolygon("could not find a nondegenerate triangulation.")

        a, b, c = points[remaining]
        if winding * cross(a, b, c) <= tolerance:
            return invalidPolygon("final triangle is degenerate.")

        triangles.append(remaining.copy())
        return vertices[np.asarray(triangles, dtype=np.intp)]    
    
    ################################################################
    # polygon
    ################################################################
    def polygon(self, vertices, color=None, units=None):
        """
        Draw a filled simple polygon, including concave polygons.

        Args:
            vertices: Numeric n x 2 coordinates ordered around the perimeter.
            color: RGB color or grayscale scalar.
            units: None or "deg" for visual-angle coordinates, "pix" for
                pixels, or "device" to skip Python coordinate conversion.

        Returns:
            bool: False if validation fails; True after submitting drawing.
                True does not independently verify GPU completion.
        """
        if not self.isOpen():
            pglMessages.warning("No screen is open.")
            return False

        # Triangulate before sending any drawing commands.
        triangles = self.triangulatePolygon(vertices)
        if triangles is None: return False

        # quad() treats None as visual-angle coordinates.
        if isinstance(units, str) and units.lower() == "deg": units = None

        # One real triangle and one degenerate triangle per quad.
        quadVertices = triangles[:, [0, 1, 2, 0], :]
        self.quad(quadVertices, color=color, units=units)
        return True
            

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
            color = np.repeat(color, 3, axis=1)
            color[:,1] = color[:,0]
            color[:,2] = color[:,0]

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
    # bullseye
    ####################################################
    def bullseye(self):
        '''
        Draw a bullseye stimulus. Used for visualizing screen dimensions
        '''
        # get width and height of screen in degrees
        maxRadius = math.ceil(math.sqrt(self.screenWidth.deg**2 + self.screenHeight.deg**2)/2) + 1
        
        # make concentric circles
        radiusSteps = 0.5
        for iRadius, innerRadius in enumerate(np.arange(-0.25, maxRadius, radiusSteps)):
            # compute outerRadius
            outerRadius = innerRadius + radiusSteps
            # alternate colors for each ring of the bullseye
            color = [0, 0, 0] if iRadius % 2 else [0.2, 0.2, 0.2]
            # if a multiple of 5, make it red to help visualize the dimensions
            if iRadius % 5 == 0: color = [0.2, 0, 0]
            if iRadius % 10 == 0: color = [0.3, 0, 0]
            # draw the circle
            self.arc(x=0, y=0, innerRadius=innerRadius, outerRadius=outerRadius, startAngle=0, stopAngle=2*np.pi, color=color, borderSize=0)

        # fontSize
        fontSize = 40
        
        # display vertical text labels
        for iRadius, radius in enumerate(np.arange(2.5, maxRadius, 2.5)):
            color = [0.8, 0.8, 0.8]
            if iRadius % 2:
                self.text(f"{radius:.1f}", x=0, y=radius, color=color, fontSize=fontSize, yAlign=0, xAlign=-1)
                self.text(f"-{radius:.1f}", x=0, y=-radius, color=color, fontSize=fontSize, yAlign=0, xAlign=-1)
                self.text(f"{radius:.1f}", x=radius, y=0, color=color, fontSize=fontSize, yAlign=-1, xAlign=0)
                self.text(f"-{radius:.1f}", x=-radius, y=0, color=color, fontSize=fontSize, yAlign=-1, xAlign=0)
            else:
                self.text(f"{radius:.1f}", x=0, y=radius, color=color, fontSize=fontSize, yAlign=0, xAlign=1)
                self.text(f"-{radius:.1f}", x=0, y=-radius, color=color, fontSize=fontSize, yAlign=0, xAlign=1)
                self.text(f"{radius:.1f}", x=radius, y=0, color=color, fontSize=fontSize, yAlign=1, xAlign=0)
                self.text(f"-{radius:.1f}", x=-radius, y=0, color=color, fontSize=fontSize, yAlign=1, xAlign=0)
        
        self.text(f"[distance: {self.distanceToScreen.cm:.1f} cm]", line=1, xAlign=1, x=self.screenWidth.deg/2)
        self.text(f"[{self.screenWidth.cm} x {self.screenHeight.cm}] cm", line=2, xAlign=1, x=self.screenWidth.deg/2)
        self.text(f"[{self.screenWidth.deg:.1f} x {self.screenHeight.deg:.1f}] deg", line=3, xAlign=1, x=self.screenWidth.deg/2)
        self.text(f"[{self.screenWidth.pix} x {self.screenHeight.pix}] pixels frameRate: {self.frameRate}", line=4, xAlign=1, x=self.screenWidth.deg/2)

        # draw polar angle lines
        for angle in np.arange(0, 360, 15):
            angleRad = np.deg2rad(angle)
            x = maxRadius * np.cos(angleRad)
            y = maxRadius * np.sin(angleRad)
            color = 0.8 if angle % 45 == 0 else 0.5
            self.line(0, 0, x, y, color=color)
            
        return

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
        color = color.flatten()

        # get vertex data
        vertexData = [x, y, 0, *color, innerRadius, outerRadius, innerRadius, outerRadius, startAngle, stopAngle, borderSize]
        
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
    # textSetRenderingParameters
    ######################################################
    def textSetRenderingParameters(self,fontName="Helvetica",fontSize=40,paddingPixels=11,foregroundColor=(1.0, 1.0, 1.0, 1.0),backgroundColor=(0.0, 0.0, 0.0, 0.0)):
        '''
        Set the server-side parameters used by subsequent _textCreate() calls.

        Args:
            fontName: Name of the Mac/Core Text font to use.
            fontSize: Font size in pixels/points.
            paddingPixels: Transparent or background-colored padding around text.
            foregroundColor: RGB or RGBA text color, with values from 0 to 1.
            backgroundColor: RGB or RGBA texture background color, with values
                from 0 to 1.

        Returns:
            True when the server accepts the new parameters, otherwise False.
        '''

        # Validate the font name.
        if not isinstance(fontName, str) or not fontName:
            pglMessages.warning("fontName must be a nonempty string.")
            return False

        # Validate the font size.
        if not isinstance(fontSize, (int, float, np.integer, np.floating)) or fontSize <= 0:
            pglMessages.warning("fontSize must be a positive number.")
            return False

        # Validate the requested padding.
        if not isinstance(paddingPixels, (int, np.integer)) or paddingPixels < 0:
            pglMessages.warning("paddingPixels must be a nonnegative integer.")
            return False

        # Convert incoming colors to four-component RGBA float arrays.
        foregroundColor = self.validateColor(foregroundColor).flatten()
        backgroundColor = self.validateColor(backgroundColor).flatten()

        if foregroundColor.size != 4 or backgroundColor.size != 4:
            pglMessages.warning("foregroundColor and backgroundColor must resolve to RGBA colors.")
            return False

        # Request replacement of the current server-side text-rendering parameters.
        self.s.writeCommand("mglSetTextRenderingParameters")
        ackTime = self.s.readAck()

        if ackTime is None:
            pglMessages.message("Did not receive command acknowledgment.")
            return False

        # Send font and layout settings.
        self.s.write(fontName)
        self.s.write(np.float32(fontSize))
        self.s.write(np.uint32(paddingPixels))

        # Send text foreground color.
        self.s.write(np.float32(foregroundColor[0]))
        self.s.write(np.float32(foregroundColor[1]))
        self.s.write(np.float32(foregroundColor[2]))
        self.s.write(np.float32(foregroundColor[3]))

        # Send texture background color.
        self.s.write(np.float32(backgroundColor[0]))
        self.s.write(np.float32(backgroundColor[1]))
        self.s.write(np.float32(backgroundColor[2]))
        self.s.write(np.float32(backgroundColor[3]))

        # This command has only normal command completion results.
        self.commandResults = self.s.readCommandResults(ackTime)

        if self.commandResults is None:
            pglMessages.message("Did not receive command results.")
            return False

        success = self.commandResults["success"]

        if np.any(np.asarray(success) == 0):
            pglMessages.message("Server did not accept text-rendering parameters.")
            return False

        # Remember the parameters now active on the Swift server.
        self.textRenderingParameters = {
            "fontName": fontName,
            "fontSize": float(fontSize),
            "paddingPixels": int(paddingPixels),
            "foregroundColor": tuple(float(value) for value in foregroundColor),
            "backgroundColor": tuple(float(value) for value in backgroundColor)
        }

        return True
    
    ######################################################
    # _setTextRenderingParametersIfNeeded
    ######################################################
    def _setTextRenderingParametersIfNeeded(self,fontName,fontSize,paddingPixels,foregroundColor,backgroundColor):
        '''
        Update Swift text-rendering parameters only when they differ from
        the parameters currently active on the server.
        '''

        foregroundColor = tuple(float(value) for value in foregroundColor)
        backgroundColor = tuple(float(value) for value in backgroundColor)

        requestedParameters = {
            "fontName": fontName,
            "fontSize": float(fontSize),
            "paddingPixels": int(paddingPixels),
            "foregroundColor": foregroundColor,
            "backgroundColor": backgroundColor
        }

        if requestedParameters == self.textRenderingParameters:
            return True

        return self.textSetRenderingParameters(
            fontName=fontName,
            fontSize=fontSize,
            paddingPixels=paddingPixels,
            foregroundColor=foregroundColor,
            backgroundColor=backgroundColor
        )    
    
    ######################################################
    # _textCreate
    ######################################################
    def _textCreate(self, textString, color=None, fontSize=40, fontName="Helvetica"):
        '''
        Create a text texture on the mglMetal server.

        Text parameters are sent only when they differ from cached settings.
        During command recording, retain the returned texture ID so replay
        can map original IDs to IDs assigned by its own renderer.
        '''
        if not isinstance(textString, str):
            pglMessages.warning("textString must be a string.")
            return None

        # Empty string: nothing to create.
        if not textString: return None

        # Convert the requested text color to RGBA.
        # validateColor(None) provides the normal default text color.
        foregroundColor = self.validateColor(color).flatten()
        if foregroundColor.size != 4:
            pglMessages.warning("color must resolve to an RGBA color.")
            return None

        # Use the current standard text texture layout defaults.
        paddingPixels = 11
        backgroundColor = (0.0, 0.0, 0.0, 0.0)

        # Apply requested settings only if they differ from the active settings.
        if not self._setTextRenderingParametersIfNeeded(
            fontName=fontName,
            fontSize=fontSize,
            paddingPixels=paddingPixels,
            foregroundColor=foregroundColor,
            backgroundColor=backgroundColor,
        ):
            pglMessages.message("Could not set text-rendering parameters.")
            return None

        # Request a texture containing the server-rendered string.
        if not self.s.writeCommand("mglCreateTextTexture"): return None
        commandIndex = self.commandRecordIndex() if self.commandRecording else -1
        ackTime = self.s.readAck()

        if ackTime is None:
            pglMessages.message("Did not receive command acknowledgment.")
            return None

        # Send the string for server-side measurement and rendering.
        self.s.write(textString)

        # Read text-specific query results.
        result = self.s.read(np.float64)
        if result is None:
            pglMessages.message("Did not receive text texture results.")
            return None
        if result < 0:
            pglMessages.message(f"Failed to create text texture for '{textString}'.")
            return None

        textWidthPixels = self.s.read(np.uint32)
        textHeightPixels = self.s.read(np.uint32)
        textureNumber = self.s.read(np.uint32)
        textureCount = self.s.read(np.uint32)

        if any(value is None for value in (textWidthPixels, textHeightPixels, textureNumber, textureCount)):
            pglMessages.message("Did not receive complete text texture data.")
            return None

        # Preserve the original resource identity before reading completion
        # results, so an interruption there does not lose the received ID.
        if commandIndex != -1:
            self.commandRecordResource(
                commandIndex,
                "texture",
                textureNumber,
                source="serverText",
                widthPixels=int(textWidthPixels),
                heightPixels=int(textHeightPixels),
                textureCount=int(textureCount),
            )

        self.commandResults = self.s.readCommandResults(ackTime)

        return pglImageInstance(
            int(textureNumber),
            int(textWidthPixels),
            int(textHeightPixels),
            self,
        )

    ######################################################
    # text
    ######################################################
    def text(self, textString, x=0, y=None, color=None, fontSize=40, fontName="Helvetica", line=None, xAlign=0, yAlign=0):
        """
        Draw text on the screen.

        Args:
            textString (str): The text to draw. For strings you want to draw multiple times, you
                       can pass in the return value of textCreate
            x (float): The x coordinate of the text. 
            y (float): The y coordinate of the text. If omitted and line is not set
                        then the text will be drawn at the top of the screen and
                        each subsequent call will draw text one line below until
                        the bottom of the screen is reached, upon which it will
                        revert to the top of the screen.
            xAlign (int): The horizontal alignment of the text. -1 = left, 0 = center, 1 = right.
            yAlign (int): The vertical alignment of the text. -1 = top, 0 = middle, 1 = bottom.
            line (int or str): The line number to draw the text on. IF this
                        is set, y will be ignored and instead the text
                        will be drawn on the specified line from top of screen.
                        If line is negative, then from the bottom of the screen.
                        If set to "center" will draw at the center of the secreen
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            fontSize (int): The size of the font.
            fontName (str): The name of the font.


        Returns:
            None
        """
        if isinstance(textString, pglImageInstance):
            textImage = textString
        elif isinstance(textString, str):
            if textString:
                # create the text image
                textImage = self._textCreate(textString, color, fontSize, fontName)
            else:
                # empty string
                return
        else:
            print("(pglDraw:text) textString must be a string or pglImageInstance returned from _textCreate.")
            return None

        # get text height in degrees
        textHeight = textImage.height.pix * self.yPix2Deg
        padding = 0
        lineHeight = textHeight 
        linesPerScreen = int(self.screenHeight.deg / lineHeight)
        
        if line is None and y is None:
            # get current line number
            line = self.currentLine
            # and update
            self.currentLine += 1

        # center value
        if line is not None and isinstance(line, str) and line.lower() == "center":
            # get the middle line
            line = linesPerScreen//2
           
        # negative line numbers
        if line is not None and line < 0: line = linesPerScreen + line + 1

        # calculate line if necessary
        if line is not None:
            # if line is specified, calculate y based on line number
            y =  self.screenHeight.deg / 2 - textHeight/2 - (line - 1) * (textHeight + padding * 2) - padding
            self.currentLine = line+1

        # check to see if we will go over the bottom
        if self.currentLine > linesPerScreen: self.currentLine = 1
   
        # display the text image
        textImage.display(x,y,xAlign=xAlign,yAlign=yAlign)

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


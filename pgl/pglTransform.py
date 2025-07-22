################################################################
#   filename: pglTransform.py
#    purpose: Implements coordinate transforms
#    License: MIT License — see LICENSE file for details.
#         by: JLG
#       date: July 9, 2025
################################################################

#############
# Import modules
#############
import math
import numpy as np


#############
# Main class
#############
class pglTransform:
    xform = np.identity(4)
    calibProportion = None
    xPix2Deg = None
    yPix2Deg = None
    xDeg2Pix = None
    yDeg2Pix = None
    ################################################################
    # Transform to visual angle coordinates
    ################################################################
    def visualAngle(self, distanceToScreenCentimeters, screenWidthCentimeters, screenHeightCentimeters, calibProportion=0.36):
        """
        Transform screen coordinates (x, y) to visual angle coordinates.

        Args:
            distanceToScreenCentimeters (float): Distance from the viewer to the screen in cm.
            screenWidthCentimeters (float): Width of the screen in cm.
            screenHeightCentimeters (float): Height of the screen in cm.
            calibProportion (float): Proportion of the screen for which the calibration
                                     will be exact. Note that the screen is flat, but
                                     the linear transform basically assumes a curved
                                     surface, so you have to decide which location on the
                                     screen you want to be exact. The default is 0.36, 
                                     which gives the best absolute error over all locations

        Returns:
            None: This function sets the transform so that all drawing
            functions will use visual angle coordinates.
        """
        # Validate inputs
        if not isinstance(distanceToScreenCentimeters, (int, float)) or distanceToScreenCentimeters <= 0:
            print("(pgl:pglTransform:visualAngleCoordinates) distanceToScreen must be a positive number.")  
            return 
        if not isinstance(screenWidthCentimeters, (int, float)) or screenWidthCentimeters <= 0:
            print("(pgl:pglTransform:visualAngleCoordinates) screenWidthCentimeters must be a positive number.")
            return 
        if not isinstance(screenHeightCentimeters, (int, float)) or screenHeightCentimeters <= 0:
            print("(pgl:pglTransform:visualAngleCoordinates) screenHeightCentimeters must be a positive number.")
            return
        if not isinstance(calibProportion, (int, float)) or calibProportion <= 0:
            print("(pgl:pglTransform:visualAngleCoordinates) calibProportion must be a positive number.")
            return

        # save paseed in vlaues
        self.screenWidth.cm = screenWidthCentimeters
        self.screenHeight.cm = screenHeightCentimeters
        self.distanceToScreen.cm = distanceToScreenCentimeters
        
        # Calculate the visual angle in degrees of the screen
        self.screenWidth.deg = (1/calibProportion)*math.atan(calibProportion*self.screenWidth.cm/self.distanceToScreen.cm)/math.pi*180
        self.screenHeight.deg = (1/calibProportion)*math.atan(calibProportion*self.screenHeight.cm/self.distanceToScreen.cm)/math.pi*180

        # get the pixel width and height
        self.getWindowFrameInDisplay()

        # calculate conversion factors
        self.xPix2Deg = self.screenWidth.deg / self.screenWidth.pix
        self.yPix2Deg = self.screenHeight.deg / self.screenHeight.pix
        self.xDeg2Pix = self.screenWidth.pix / self.screenWidth.deg
        self.yDeg2Pix = self.screenHeight.pix / self.screenHeight.deg

        # set scale 
        self.setTransformScale(2.0 / self.screenWidth.deg, 2.0 / self.screenHeight.deg)

    ################################################################
    # Transform to screen coordinates
    ################################################################
    def screenCoordinates(self):
        pass


    ################################################################
    # Set scale of transform
    ################################################################
    def setTransformScale(self, xScale = 1.0, yScale = 1.0, zScale = 1.0, keepCurrent=False):
        '''
        '''
        # Create a xform that scales
        xformScale = np.eye(4)
        xformScale[0,0] = xScale
        xformScale[1,1] = yScale
        xformScale[2,2] = zScale
        # multiply with current transform
        if keepCurrent:
            self.xform = np.matmul(xformScale,self.xform)
        else:
            self.xform = xformScale
        # now update the xform on the application
        self.setTransform(self.xform)

    ################################################################
    # Set translation of transform
    ################################################################
    def setTransformOffset(self, xOffset = 0.0, yOffset = 0.0, zOffset = 0.0):
        '''
        '''
        # Create a xform that shifts the offset
        xformOffset= np.eye(4)
        xformOffset[0,3] = xOffset
        xformOffset[1,3] = yOffset
        xformOffset[2,3] = zOffset
        # multiply with current transform
        self.xform = np.matmul(xformScale,self.xform)
        # now update the xform on the application
        self.setTransform(self.xform)
        

    ################################################################
    # Set transform
    ################################################################
    def setTransform(self, xform):
        '''
        '''
        self.s.writeCommand("mglSetXform")
        
        # send the color data
        self.s.write(np.array(xform.T.astype(np.float32).flatten(), dtype=np.float32))
        
        # Read the command results
        self.commandResults = self.s.readCommandResults()




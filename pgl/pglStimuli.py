################################################################
#   filename: pglStimuli.py
#    purpose: Class that implements common psychophysical stimuli
#             such as random dots and gabors
#         by: JLG
#       date: July 26, 2025
################################################################

#############
# Import modules
#############
import numpy as np

#############
# Stimuli class
#############
class pglStimuli:
    '''
    Class for generating common psychophysical stimuli.

    '''
    def grating(self, width=None, height=None, spatialFrequency=1.0, orientation=0.0, contrast=1.0, phase=0.0, temporalFrequency=0.0, direction = 1, returnAsMatrix = False):
        '''
        Generate a sinusoidal grating

        Parameters:
        - spatialFrequency (float): Frequency of the grating in cycles per degree.
        - temporalFrequency (float): Temporal frequency of the grating in Hz. (default is 0)
        - phase (float): Phase of the grating in degrees.
        - orientation (float): Orientation of the grating in degrees.
        - contrast (float): Contrast of the grating (0 to 1).
        - width (float): Width of the grating in degrees (default is screenWidth).
        - height (float): Height of the grating in degrees (default is screenHeight).

        Returns:
        - A numpy array representing the grating stimulus.
        '''
        if self.coordinateFrame != "visualAngle":
            print("(pgl:pglStimuli:gaussian) Error: gaussian can only be generated in visualAngle coordinates. run visualAngle() ")
            return None

        # Validate inputs
        if not isinstance(spatialFrequency, (int, float)) or spatialFrequency <= 0:
            print("(pgl:pglStimuli:gaussian) spatialFrequency must be a positive number.")
            return None
        if not isinstance(phase, (int, float)):
            print("(pgl:pglStimuli:gaussian) phase must be a number.")
            return None
        if not isinstance(orientation, (int, float)):
            print("(pgl:pglStimuli:gaussian) orientation must be a number.")
            return None
        if not isinstance(contrast, (int, float)) or not (0 <= contrast <= 1):
            print("(pgl:pglStimuli:gaussian) contrast must be a number between 0 and 1.")
            return None 

        if width is None: width = self.screenWidth.deg
        if height is None: height = self.screenHeight.deg

        # create a squence of frames for this temporal frequency
        if temporalFrequency != 0:
            # get deltaT of monitor
            deltaT = 1 / self.getFrameRate()
            # calculate on period
            period = direction / temporalFrequency
            # get time points to compute images from
            phasePoints = np.arange(0, period, deltaT)
            nPhase = len(phasePoints)
            # Now, prealocate array
            grating = np.zeros((int(height * self.yDeg2Pix), int(width * self.xDeg2Pix), nPhase), dtype=np.float32)
            # for each phasePoint, compute the grating
            for iPhase, phaseValue in enumerate(phasePoints):
                # get the phase for this frame
                thisPhase = phase + 360 * (iPhase / nPhase)
                # compute frame
                grating[..., iPhase] = self.grating(width, height, spatialFrequency, orientation, contrast, thisPhase)
            if returnAsMatrix: return grating
            # create a pglImageStimulus
            gratingStimulus = pglImageStimulus(self)
            for iPhase in range(nPhase):
                gratingStimulus.addImage(grating[..., iPhase])
            return gratingStimulus
            
        # Create a grid of coordinates with resolution
        # set by the current screen resolution
        x = np.linspace(-width/2, width/2, int(width * self.xDeg2Pix))
        y = np.linspace(-height/2, height/2, int(height * self.yDeg2Pix))
        X, Y = np.meshgrid(x, y)

        # Convert orientation to radians
        theta = np.deg2rad(orientation)
        phase = np.deg2rad(phase)

        # Calculate the grating
        grating = contrast * np.sin(2 * np.pi * spatialFrequency * (X * np.cos(theta) + Y * np.sin(theta)) + phase)
        return grating

    def gaussian(self, width=None, height=None, stdX=None, stdY=None, centerX=0, centerY=0, orientation=0.0, contrast=1.0):
        '''
        Generate a Gaussian 

        Parameters:
        - width: Width of the grating in degrees (default is screenWidth).
        - height: Height of the grating in degrees (default is screenHeight).
        - stdX: Standard deviation in the X direction (default is 1/8 width).
        - stdY: Standard deviation in the Y direction (default is 1/8 height).
        - centerX: Center position in the X direction (default is 0).
        - centerY: Center position in the Y direction (default is 0).
        - orientation: Orientation of the Gaussian in degrees (default is 0).

        Returns:
        - A numpy array representing the Gaussian grating stimulus.
        '''
        if self.coordinateFrame != "visualAngle":
            print("(pgl:pglStimuli:gaussian) Error: gaussian can only be generated in visualAngle coordinates. run visualAngle() ")
            return None
    
        # Validate inputs
        if not isinstance(centerX, (int, float)):
            print("(pgl:pglStimuli:gaussian) centerX must be a number.")
            return None
        if not isinstance(centerY, (int, float)):
            print("(pgl:pglStimuli:gaussian) centerY must be a number.")
            return None
        if not isinstance(orientation, (int, float)):
            print("(pgl:pglStimuli:gaussian) orientation must be a number.")
            return None
        if not isinstance(contrast, (int, float)) or not (0 <= contrast <= 1):
            print("(pgl:pglStimuli:gaussian) contrast must be a number between 0 and 1.")
            return None
        if width is None: width = self.screenWidth.deg
        if height is None: height = self.screenHeight.deg
        if stdX is None: stdX = width / 8
        if stdY is None: stdY = height / 8
        if height is None: height = self.screenHeight.deg
        if not isinstance(stdX, (int, float)) or stdX <= 0:
            print("(pgl:pglStimuli:gaussian) stdX must be a positive number.")
            return None
        if not isinstance(stdY, (int, float)) or stdY <= 0:
            print("(pgl:pglStimuli:gaussian) stdY must be a positive number.")
            return None

        # Create a grid of coordinates with resolution
        # set by the current screen resolution
        x = np.linspace(-width/2, width/2, int(width * self.xDeg2Pix))
        y = np.linspace(-height/2, height/2, int(height * self.yDeg2Pix))
        X, Y = np.meshgrid(x, y)

        # Convert to radians
        theta = np.deg2rad(orientation) 

        # Rotation coefficients
        a = np.cos(theta)**2 / (2 * stdX**2) + np.sin(theta)**2 / (2 * stdY**2)
        b = -np.sin(2 * theta) / (4 * stdX**2) + np.sin(2 * theta) / (4 * stdY**2)
        c = np.sin(theta)**2 / (2 * stdX**2) + np.cos(theta)**2 / (2 * stdY**2)

        # Gaussian function
        gaussian = np.exp(-(a * (X - centerX)**2 + 2 * b * (X - centerX) * (Y - centerY) + c * (Y - centerY)**2))

        return gaussian
    
    def gabor(self, width=None, height=None, spatialFrequency=1.0, orientation=0.0, contrast=1.0, phase=0.0, stdX=None, stdY=None, returnAsMatrix = False):
        '''
        Generate a Gabor patch.

        Parameters:
        - spatialFrequency: Frequency of the grating in cycles per degree.
        - phase: Phase of the grating in degrees.
        - orientation: Orientation of the grating in degrees.
        - contrast: Contrast of the grating (0 to 1).
        - width: Width of the Gabor patch in degrees (default is screenWidth).
        - height: Height of the Gabor patch in degrees (default is screenHeight).
        - stdX: Standard deviation in the X direction (default is 1/8 width).
        - stdY: Standard deviation in the Y direction (default is 1/8 height).
        - returnAsMatrix (False): If True, return as a numpy matrix. If False, 
            return as a pglImageStimulus

        Returns:
        - A pglImageStimulus representing the Gabor stimulus. Use the display() method to show it.

        e.g.:
        gabor = pgl.gabor(width=5, height=5, spatialFrequency=1.0, orientation=0.0, contrast=1.0, phase=0.0, stdX=0.5, stdY=0.5)
        gabor.display()
        pgl.flush()
        '''
        if self.coordinateFrame != "visualAngle":
            print("(pgl:pglStimuli:gabor) Error: gabor can only be generated in visualAngle coordinates. run visualAngle() ")
            return None

        # Validate inputs
        if not isinstance(spatialFrequency, (int, float)) or spatialFrequency <= 0:
            print("(pgl:pglStimuli:gabor) spatialFrequency must be a positive number.")
            return None
        if not isinstance(phase, (int, float)):
            print("(pgl:pglStimuli:gabor) phase must be a number.")
            return None
        if not isinstance(orientation, (int, float)):
            print("(pgl:pglStimuli:gabor) orientation must be a number.")
            return None
        if not isinstance(contrast, (int, float)) or not (0 <= contrast <= 1):
            print("(pgl:pglStimuli:gabor) contrast must be a number between 0 and 1.")
            return None 

        # create the gabor
        grating = self.grating(width, height, spatialFrequency, orientation, contrast, phase)
        gaussian = self.gaussian(width, height, stdX, stdY)
        gabor = ((grating * gaussian)+1)/2
        grating = self.grating(width, height, spatialFrequency, orientation+45, contrast, phase)
        gaussian = self.gaussian(width, height, stdX, stdY)
        gabor2 = ((grating * gaussian)+1)/2
        if returnAsMatrix: return gabor
        
        # create a pglImageStimulus
        gaborStimulus = pglImageStimulus(self)
        gaborStimulus.addImage(gabor)
        gaborStimulus.addImage(gabor2)
        gaborStimulus.addImage(gabor)
        gaborStimulus.addImage(gabor2)
        gaborStimulus.print()
        return gaborStimulus


class _pglStimulus:
    '''
    Base class for all stimuli.
    This class is not meant to be instantiated directly.
    '''
    pgl = None
    def __init__(self, pgl):
        self.pgl = pgl

    def display(self):
        '''
        Display the stimulus.
        This method should be implemented by subclasses.
        '''
        raise NotImplementedError("(_pglStimulus) Subclasses must implement this method.")
    def print(self):
        '''
        Print information about the stimulus.
        This method should be implemented by subclasses.
        '''
        raise NotImplementedError("(_pglStimulus) Subclasses must implement this method.")

class pglImageStimulus(_pglStimulus):
    '''
    Base class for image stimuli.
    This class is not meant to be instantiated directly.
    '''
    def __init__(self, pgl):
        '''
        Initialize the image stimulus with an image instance.
        
        Args:
        '''
        super().__init__(pgl)
        self.nImages = 0
        self.currentImage = 0
        self.imageList = []
    
    def addImage(self, imageData):
        '''
        Add an image to the stimulus.
        
        Args:
        - imageData: The image data to be added.
        '''
        (tf,imageData) = self.pgl.imageValidate(imageData)
        if not tf: return None
        
        # make into a pgl image
        imageInstance = self.pgl.imageCreate(imageData)
        if self.pgl.verbose>1: print(f"(pgl:pglStimulus:addImage) Adding image {imageInstance.imageNum} ({imageInstance.imageWidth}x{imageInstance.imageHeight})")
        self.imageList.append(imageInstance)
        # update count
        self.nImages += 1

    def display(self):
        '''
        Display the current image.
        '''
        if self.nImages == 0:
            print("(pgl:pglStimulus:display) No images to display.")
            return None
        
        # Print information about the stimulus
        if self.pgl.verbose>1: print(f"(pgl:pglStimulus:display) Displaying image {self.currentImage} of {self.nImages}.")

        # display current image        
        self.imageList[self.currentImage].display()

        # Increment the current image index
        self.currentImage = (self.currentImage + 1) % self.nImages
        
    def print(self):
        '''
        Print information about the stimulus.
        '''
        print(f"(pgl:pglStimulus:print) Image {self.currentImage}/{self.nImages}. ")
        # print info on each image
        for iImage in range(self.nImages):
            self.imageList[iImage].print()
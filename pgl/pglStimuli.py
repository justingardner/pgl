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
    def grating(self, width=None, height=None, spatialFrequency=1.0, orientation=0.0, contrast=1.0, phase=0.0):
        '''
        Generate a sinusoidal grating

        Parameters:
        - spatialFrequency: Frequency of the grating in cycles per degree.
        - phase: Phase of the grating in degrees.
        - orientation: Orientation of the grating in degrees.
        - contrast: Contrast of the grating (0 to 1).
        - width: Width of the grating in degrees (default is screenWidth).
        - height: Height of the grating in degrees (default is screenHeight ).

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
    
    def gabor(self, width=None, height=None, spatialFrequency=1.0, orientation=0.0, contrast=1.0, phase=0.0, stdX=None, stdY=None):
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

        Returns:
        - A numpy array representing the Gabor stimulus.
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

        # create a stimulus
        gaborStimulus = pglImageStimulus(self)
        rgba = np.zeros((gabor.shape[0], gabor.shape[1], 4), dtype=np.float32)
        rgba[..., 0] = gabor
        rgba[..., 1] = gabor
        rgba[..., 2] = gabor
        rgba[..., 3] = gaussian
        gaborStimulus.addImage(rgba)
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
        raise NotImplementedError("Subclasses must implement this method.")

class pglImageStimulus(_pglStimulus):
    '''
    Base class for image stimuli.
    This class is not meant to be instantiated directly.
    '''
    nImages = 0
    currentImage = 0
    imageList = []
    def __init__(self, pgl):
        '''
        Initialize the image stimulus with an image instance.
        
        Args:
        '''
        super().__init__(pgl)
    
    def addImage(self, imageData):
        '''
        Add an image to the stimulus.
        
        Args:
        - imageData: The image data to be added.
        '''
        if not isinstance(imageData, np.ndarray) or imageData.ndim != 3 or imageData.shape[2] != 4:
            print("(pgl:pglStimulus:addImage) Error: imageData must be a nxmx4 numpy matrix.")
            return None
        
        # make into a pgl image
        imageInstance = self.pgl.imageCreate(imageData)
        self.imageList.append(imageInstance)
        # update count
        self.nImages += 1

    def display(self):
        '''
        Display the current image.
        '''
        if self.nImages == 0:
            print("(pgl:pglStimulus:display) Error: No images to display.")
            return None

        currentImage = self.imageList[self.currentImage]
        currentImage.display()
        currentImage.print()
        # Increment the current image index
        self.currentImage = (self.currentImage + 1) % self.nImages
        print(f"(pgl:pglStimulus:display) Displaying image {self.currentImage} of {self.nImages}.")
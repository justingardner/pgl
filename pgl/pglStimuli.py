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
        if not isinstance(contrast, (int, float)) or not (-1 <= contrast <= 1):
            print("(pgl:pglStimuli:gaussian) contrast must be a number between -1 and 1.")
            return None
        if not isinstance(direction, int) or direction not in [-1, 0, 1]:
            print("(pgl:pglStimuli:gaussian) direction must be -1, 0, or 1.")
            return None

        if width is None: width = self.screenWidth.deg
        if height is None: height = self.screenHeight.deg

        # create a squence of frames for this temporal frequency
        if temporalFrequency != 0:
            # get deltaT of monitor
            deltaT = 1 / self.getFrameRate()
            # calculate on period
            period = 1 / temporalFrequency
            # get time points to compute images from
            phasePoints = np.arange(0, period, deltaT)
            # set direction
            if direction == -1: phasePoints = phasePoints[::-1]
            nPhase = len(phasePoints)
            # Now, preallocate array
            grating = np.zeros((int(height * self.yDeg2Pix), int(width * self.xDeg2Pix), nPhase), dtype=np.float32)
            # for each phasePoint, compute the grating
            for iPhase, phaseValue in enumerate(phasePoints):
                # get the phase for this frame
                thisPhase = phase + direction * 360 * iPhase / nPhase
                # if direction = 0, this is a contrast reversing grating
                if direction == 0:
                    thisContrast = contrast * np.cos(2 * np.pi * iPhase / nPhase)
                else:
                    thisContrast = contrast
                # compute frame
                grating[..., iPhase] = self.grating(width, height, spatialFrequency, orientation, thisContrast, thisPhase)
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
    
    def randomDots(self, width=10, height=10, color=[1,1,1], aperture='elliptical', density=10, dotSize=0.1, dotShape=1, dotAntialiasingBorder=0, noiseType='randomwalk'):
        '''
        Generate a random dot stimulus.

        Parameters:
        - width: Width of the stimulus in degrees (default is 10).
        - height: Height of the stimulus in degrees (default is 10).
        - aperture: Shape of the aperture ('elliptical' or 'rectangular', default is 'elliptical').
        - density: Density of dots per square degree (default is 10).
        - dotSize: Size of the dots in degrees (default is 1).
        - dotShape: Shape of the dots (0 for rectangular, 1 for circular, default is 1).
        - dotAntialiasingBorder: Antialiasing border size in pixels (default is 0).

        Returns:
        - A pglRandomDotsStimulus instance.
        '''
        rdk = pglRandomDotsStimulus(self, width, height, color, aperture, density, dotSize, dotShape, dotAntialiasingBorder, noiseType)
        return rdk


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

class pglRandomDotsStimulus(_pglStimulus):
    '''
    Base class for random dot stimuli.
    '''
    def __init__(self, pgl, width=10, height=10, color=[1,1,1], aperture='elliptical', density=10, dotSize=1, dotShape=1, dotAntialiasingBorder=0, noiseType='randomwalk'):
        # call init function of parent class
        super().__init__(pgl)
        self.width = width
        self.height = height
        self.aperture = aperture
        self.density = density
        self.dotSize = dotSize
        self.dotShape = dotShape
        self.dotAntialiasingBorder = dotAntialiasingBorder

        if noiseType.lower() in ['randomwalk', 'random', 'random walk']:
            self.noiseType = 0
        elif noiseType.lower() in ['movshon', 'replot']:
            self.noiseType = 1
        else:
            print(f"(pgl:pglStimulus:init) Unknown noise type '{noiseType}'. Defaulting to 'randomwalk'.")
            self.noiseType = 0

        # calculate number of dots
        self.n = int(self.width * self.height * self.density)
        if self.pgl.verbose>1: print(f"(pgl:pglStimulus:init) Creating random dot stimulus with {self.n} dots, size {self.dotSize}, shape {self.dotShape}, aperture {self.aperture}.")
        
        # validate color
        self.color = pgl.validateColor(color, n=self.n, forceN=True)
        self.color[:,3] = 1

        # create arrays of random positions
        rx = self.width / 2
        ry = self.height / 2
        self.x = np.random.uniform(-rx, rx, self.n)
        self.y = np.random.uniform(-ry, ry, self.n)
        self.z = np.zeros(self.n)

        # compute the radii squared for checking against oval apertures
        self.rx2 = rx * rx
        self.ry2 = ry * ry

        # default aperture check type
        if aperture not in ['elliptical', 'rectangular']:
            print(f"(pgl:pglStimulus:init) Error: Unknown aperture type '{aperture}'. Use 'elliptical' or 'rectangular'. Defaulting to elliptical")
            aperture = 'elliptical'

        if aperture == 'elliptical':
            self.apertureCheck = lambda x, y: (x**2 / self.rx2) + (y**2 / self.ry2) > 1
        elif aperture == 'rectangular':
            self.apertureCheck = lambda x, y: (np.abs(x) > rx) | (np.abs(y) > ry)
    
    def __repr__(self):
        return f"<pglRandomDotStimulus: {self.n} dots, size={self.dotSize}, shape={self.dotShape}, aperture={self.aperture}>"

    def display(self, direction=0, coherence=1.0, speed=1.0):
        '''
        Display the random dot stimulus

        Parameters:
        - direction: Direction of motion in degrees (default is 0).
        - coherence: Coherence of the motion (0 to 1, default is 1.0).
        - speed: Speed of the motion in degrees per second (default is 1.0
        '''
        # check if the dots are within the aperture
        invalidDots = self.apertureCheck(self.x, self.y)
        # make all dots visible
        self.color[:,3] = 1
        if np.any(invalidDots):
            # make dots outside of the aperture invisible
            self.color[invalidDots, 3] = 0
            self.x[invalidDots] = -self.x[invalidDots]
            self.y[invalidDots] = -self.y[invalidDots]
            self.z[invalidDots] = 0

        # draw the dots
        self.pgl.dots(self.x, self.y, self.z, color=self.color, dotSize=self.dotSize, dotShape=self.dotShape, dotAntialiasingBorder=self.dotAntialiasingBorder)

        # convert direction into an array of directions
        direction = np.deg2rad(direction)
        direction = np.tile(direction, self.n)

        # for incoherent motion, set the proportion of incoherent dots
        # to have a random direction
        incoherentDots = np.random.rand(self.n) > coherence
        if np.any(incoherentDots):
            if self.noiseType == 0:
                # set the direction of incoherent dots to a random direction
                randomDirections = np.random.uniform(0, 2 * np.pi, np.sum(incoherentDots))
                direction[incoherentDots] = randomDirections
            else:
                numIncoherentDots = np.sum(incoherentDots)
                # replot randomly
                self.x[incoherentDots] = np.random.uniform(-self.width/2, self.width/2, numIncoherentDots)
                self.y[incoherentDots] = np.random.uniform(-self.height/2, self.height/2, numIncoherentDots)
                

        # update positions based on direction and speed
        if speed != 0:
            self.x += (speed * np.cos(direction))/self.pgl.frameRate
            self.y += (speed * np.sin(direction))/self.pgl.frameRate
class pglImageStimulus(_pglStimulus):
    '''
    Base class for image stimuli.
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
    
    def __repr__(self):
        return f"<pglImageStimulus: {self.currentImage+1}/{self.nImages}>"

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
        print(self.__repr__())
        # print info on each image
        for iImage in range(self.nImages):
            self.imageList[iImage].print()
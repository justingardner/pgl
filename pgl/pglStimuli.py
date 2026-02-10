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
import matplotlib.pyplot as plt

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
            # create a pglStimulusImage
            gratingStimulus = pglStimulusImage(self)
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
            return as a pglStimulusImage.

        Returns:
        - A pglStimulusImage representing the Gabor stimulus. Use the display() method to show it.

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
        
        # create a  pglStimulusImage
        gaborStimulus = pglStimulusImage(self)
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
        - A pglStimulusRandomDots instance.
        '''
        rdk = pglStimulusRandomDots(self, width, height, color, aperture, density, dotSize, dotShape, dotAntialiasingBorder, noiseType)
        return rdk

    ####################################################
    # checkerboard
    ####################################################
    def checkerboard(self, x=0, y=0, width=None, height=None, checkWidth=1.0, checkHeight = 1.0, temporalFrequency = 1.0, color=None, type='sliding', temporalSquareWave=True):
        """
        Checkerboard stimulus 
        Args:
            x (float): x coordinate of the center of the checkerboard.
            y (float): y coordinate of the center of the checkerboard.
            width (float, optional): Width of the checkerboard in degrees. If None, will use screenWidth.
            height (float, optional): Height of the checkerboard in degrees. If None, will use screenHeight.
            checkWidth (float): Width of each checker square in degrees. Default=1.0
            checkHeight (float): Height of each checker square in degrees. Default=1.0
            temporalFrequency (float): Temporal frequency of the checkerboard in Hz. Default=1.0
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            type (str, optional): Type of checkerboard ('sliding' or 'flickering'). Default='sliding'. When temporalFrequency
                                  is not zero will either flicker or slide. For bars used in pRF mapping, we use sliding
                                  type.
            temporalSquareWave (bool, optional): If True, the checkerboard will use a square wave temporal profile.
                                                  Default is True. This only changes how flickering stimuli are presented
        """
        if self.coordinateFrame != "visualAngle":
            print("(pgl:pglStimuli:checkerboard) Error: checkerboard can only be generated in visualAngle coordinates. run visualAngle() ")
            return None

        # Validate inputs
        if not isinstance(x, (int, float)):
            print("(pgl:pglStimuli:checkerboard) ❌ x must be a number.")
            return None
        if not isinstance(y, (int, float)):
            print("(pgl:pglStimuli:checkerboard) ❌ y must be a number.")
            return None
        if not isinstance(checkWidth, (int, float)) or checkWidth <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ checkWidth must be a positive number.")
            return None
        if not isinstance(checkHeight, (int, float)) or checkHeight <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ checkHeight must be a positive number.")
            return None
        if not isinstance(temporalFrequency, (int, float)) or temporalFrequency < 0:
            print("(pgl:pglStimuli:checkerboard) ❌ temporalFrequency must be a non-negative number.")
            return None
        if not isinstance(width, (int, float)) and width is not None:
            print("(pgl:pglStimuli:checkerboard) ❌ width must be a positive number.")
            return None
        if not isinstance(height, (int, float)) and height is not None:
            print("(pgl:pglStimuli:checkerboard) ❌ height must be a positive number.")
            return None
        if type not in ['sliding', 'flickering','flicker','slide']:
            print("(pgl:pglStimuli:checkerboard) ❌ type must be 'sliding' or 'flickering'.")
            return None
        if temporalSquareWave not in [True, False]:
            print("(pgl:pglStimuli:checkerboard) ❌ temporalSquareWave must be a boolean.")
            return None

        # Create the checkerboard stimulus
        if type in ['sliding', 'slide']:
            checkerboardStimulus = pglStimulusCheckerboardSliding(self, x=x, y=y, width=width, height=height,
                                                                 checkWidth=checkWidth, checkHeight=checkHeight,
                                                                 temporalFrequency=temporalFrequency, color=color)
        else:
            checkerboardStimulus = pglStimulusCheckerboardFlickering(self, x=x, y=y, width=width, height=height,
                                                                   checkWidth=checkWidth, checkHeight=checkHeight,
                                                                   temporalFrequency=temporalFrequency, color=color,
                                                                   temporalSquareWave=temporalSquareWave)
        return checkerboardStimulus
    
    ####################################################
    # bar
    ####################################################
    def bar(self, width=1.0, speed=1.0, dir=0):
        '''
        Create a bar stimulus. Used for pRF mapping
        
        Args:
            width (float): The width of the bar stimulus in degrees
            speed (float): The speed in deg/s the bar stimulus moves across screen
        '''
        # Validate inputs
        if not isinstance(width, (int, float)) or width <= 0:
            print("(pgl:pglStimuli:bar) ❌ width must be a positive number.")
            return None

        if not isinstance(speed, (int, float)) or speed <= 0:
            print("(pgl:pglStimuli:bar) ❌ speed must be a positive number.")
            return None

        # Create the bar stimulus
        barStimulus = pglStimulusBar(self, width=width, speed=speed, dir=dir)
        return barStimulus

    ####################################################
    # movie
    ####################################################
    def movie(self, filename, x=0, y=0, displayWidth=0, displayHeight=0, xAlign=0, yAlign=0):
        '''
        Create a movie stimulus. 

        Args:
            filename (str): The file name of the movie file
        '''
        # Validate inputs
        if not isinstance(filename, str):
            print("(pgl:pglStimuli:movie) ❌ filename must be a string.")
            return None

        # Create the movie stimulus
        movieStimulus = pglStimulusMovie(self, filename=filename, x=x, y=y, displayWidth=displayWidth, displayHeight=displayHeight, xAlign=xAlign, yAlign=yAlign)
        return movieStimulus

    ####################################################
    # radialCheckerboard
    ####################################################
    def radialCheckerboard(self, pgl, x=0, y=0, radialWidth=360, theta=0, outerRadius=None, innerRadius=None, checkRadialWidth=15.0, checkRadialLength=1.0, temporalFrequency=1.0, temporalSquareWave=True, color=None, type='flickering'):
        """
        Radial checkerboard stimulus
        Args:
            x (float): x coordinate of the center of the checkerboard.
            y (float): y coordinate of the center of the checkerboard.
            radialWidth (float, optional): Width of the checkerboard in degrees. If None, will use screenWidth.
            theta (float, optional): Rotation angle of the checkerboard in degrees. Default=0.
            innerRadius (float, optional): inner radius of checkerboard pattern
            outerRadius (float, optional): outer radius of checkerboard pattern
            checkRadialWidth (float): radial width of checkerboard pattern (degrees around circle)
            checkRadialLength (float): radial length of checks in degrees of visual angle
            temporalFrequency (float): Temporal frequency of the checkerboard in Hz. Default=1.0
            color (list or tuple, optional): RGB color values as a list or tuple of three floats in the range [0, 1].
            type (str, optional): Type of checkerboard ('sliding' or 'flickering'). Default='sliding'. When temporalFrequency
                                  is not zero will either flicker or slide. For bars used in pRF mapping, we use sliding
                                  type.
            temporalSquareWave (bool, optional): If True, the checkerboard will use a square wave temporal profile.
                                                  Default is True. This only changes how flickering stimuli are presented
        """
        if self.coordinateFrame != "visualAngle":
            print("(pgl:pglStimuli:checkerboard) Error: checkerboard can only be generated in visualAngle coordinates. run visualAngle() ")
            return None

        # Validate inputs
        if not isinstance(x, (int, float)):
            print("(pgl:pglStimuli:checkerboard) ❌ x must be a number.")
            return None
        if not isinstance(y, (int, float)):
            print("(pgl:pglStimuli:checkerboard) ❌ y must be a number.")
            return None
        if not isinstance(checkRadialWidth, (int, float)) or checkRadialWidth <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ checkRadialWidth must be a positive number.")
            return None
        if not isinstance(checkRadialLength, (int, float)) or checkRadialLength <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ checkRadialLength must be a positive number.")
            return None
        if not isinstance(temporalFrequency, (int, float)) or temporalFrequency < 0:
            print("(pgl:pglStimuli:checkerboard) ❌ temporalFrequency must be a non-negative number.")
            return None
        if not isinstance(radialWidth, (int, float)) and radialWidth is not None:
            print("(pgl:pglStimuli:checkerboard) ❌ radialWidth must be a positive number.")
            return None
        if not isinstance(innerRadius, (int, float)) and innerRadius is not None:
            print("(pgl:pglStimuli:checkerboard) ❌ innerRadius must be a positive number.")
            return None
        if not isinstance(outerRadius, (int, float)) and outerRadius is not None:
            print("(pgl:pglStimuli:checkerboard) ❌ outerRadius must be a positive number.")
            return None
        if type not in ['sliding', 'flickering','flicker','slide']:
            print("(pgl:pglStimuli:checkerboard) ❌ type must be 'sliding' or 'flickering'.")
            return None
        if temporalSquareWave not in [True, False]:
            print("(pgl:pglStimuli:checkerboard) ❌ temporalSquareWave must be a boolean.")
            return None

        # Create the checkerboard stimulus
        if type in ['sliding', 'slide']:
            checkerboardStimulus = pglStimulusRadialCheckerboardSliding(self, x=x, y=y, radialWidth=radialWidth, theta=theta,
                                                                        outerRadius=outerRadius, innerRadius=innerRadius,
                                                                        checkRadialWidth=checkRadialWidth, checkRadialLength=checkRadialLength,
                                                                        temporalFrequency=temporalFrequency, color=color)
        else:
            checkerboardStimulus = pglStimulusRadialCheckerboardFlickering(self, x=x, y=y, radialWidth=radialWidth, theta=theta,
                                                                          outerRadius=outerRadius, innerRadius=innerRadius,
                                                                          checkRadialWidth=checkRadialWidth, checkRadialLength=checkRadialLength,
                                                                          temporalFrequency=temporalFrequency, color=color,
                                                                          temporalSquareWave=temporalSquareWave)
        return checkerboardStimulus

    ####################################################
    # flicker
    ####################################################
    def flicker(self, pgl, temporalFrequency=None, x=None, y=None, width=None, height=None, type='square', phase=0.0):
        '''
        Full screen flicker stimulus
        
        Args:
            temporalFrequency (float): Temporal frequency of the flicker in Hz. If None, defaults to half the frame rate.
            type (str): Type of flicker ('square' or 'sinusoidal'). Default is 'square'.
            x (float): X position of the center of the flicker in degrees. If None, defaults to 0.
            y (float): Y position of the center of the flicker in degrees. If None, defaults to 0.
            width (float): Width of the flicker in degrees. If None, defaults to screen width.
            height (float): Height of the flicker in degrees. If None, defaults to screen height.
            phase (float): Phase of the flicker in degrees. Default is 0.
        '''
        flickerStimulus = pglStimulusFlicker(pgl, temporalFrequency, type, x, y, width, height, phase)
        return flickerStimulus

################################################################
# Basre stimulus class
################################################################
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

################################################################
# Full screen flicker stimulus
################################################################
class pglStimulusFlicker(_pglStimulus):
    '''
    Base class for random dot stimuli.
    
    Args:
        pgl: pgl instance
        temporalFrequency (float): Temporal frequency of the flicker in Hz. If None, defaults to half the frame rate.
        type (str): Type of flicker ('square' or 'sinusoidal'). Default is 'square'.
        x (float): X position of the center of the flicker in degrees. If None, defaults to 0.
        y (float): Y position of the center of the flicker in degrees. If None, defaults to 0.
        width (float): Width of the flicker in degrees. If None, defaults to screen width.
        height (float): Height of the flicker in degrees. If None, defaults to screen height.
        phase (float): Phase of the flicker in degrees. Default is 0.
    '''
    def __init__(self, pgl, temporalFrequency=None, type='square', x=None, y=None, width=None, height=None, phase=0.0):
        # call init function of parent class
        super().__init__(pgl)
        self.temporalFrequency = temporalFrequency
        self.type = type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.phase = 2*np.pi*phase/360
        
        # get frame rate
        self.frameRate = pgl.getFrameRate()

        # set temporal frequency
        if temporalFrequency is None:
            self.temporalFrequency = self.frameRate / 2
        if width is None:
            self.width = pgl.screenWidth.deg
        if height is None:
            self.height = pgl.screenHeight.deg
        if x is None:
            self.x = 0
        if y is None:
            self.y = 0
            
        # interpret type
        if type.lower() in ['sinusoidal', 'sine', 'sin']:
            self.type = 0
        elif type.lower() in ['square']:
            self.type = 1
        else:
            print(f"(pgl:pglStimulusFlicker:init) Unknown type '{type}'. Defaulting to square")
            self.type = 1
            
        # initialize start time
        self.startTime = -1

        
    def __repr__(self):
        return f"<pglStimulusFlicker: temporalFrequency={self.temporalFrequency}, type={self.type}, x={self.x}, y={self.y}, width={self.width}, height={self.height}>"

    def display(self):
        '''
        Display the flicker stimulus
    
        Returns:
            bool: True if a new cycle just started on this frame, False otherwise
        '''
        if self.startTime == -1:
            self.startTime = self.pgl.getSecs()
            self.lastCycleCount = 0
    
        # get elapsed time
        elapsedTime = self.pgl.getSecs() - self.startTime
    
        # compute phase and cycle count
        phase = self.phase + 2 * np.pi * (elapsedTime * self.temporalFrequency % 1.0)
        currentCycleCount = int(elapsedTime * self.temporalFrequency)
    
        # detect new cycle
        newCycle = (currentCycleCount > self.lastCycleCount)
        self.lastCycleCount = currentCycleCount
    
        if self.type == 0:
            # sinusoidal flicker
            intensity = (np.sin(phase) + 1) / 2
        else:
            # square flicker
            intensity = 1.0 if phase < np.pi else 0.0
    
        # draw the rectangle
        self.pgl.rect(self.x, self.y, self.width, self.height, color=[intensity, intensity, intensity])
    
        return newCycle          
################################################################
# Random dot stimulus class
################################################################
class pglStimulusRandomDots(_pglStimulus):
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
        return f"<pglStimulusRandomDot: {self.n} dots, size={self.dotSize}, shape={self.dotShape}, aperture={self.aperture}>"

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

################################################################
# Image stimulus class
################################################################
class pglStimulusImage(_pglStimulus):
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
        return f"<pglStimulusImage: {self.currentImage+1}/{self.nImages}>"

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
        if self.pgl.verbose>1: print(f"(pgl:pglStimulus:addImage) Adding image {imageInstance.imageNum} ({imageInstance.width}x{imageInstance.height})")
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

################################################################
# Checkerboard stimulus class
################################################################
class _pglStimulusCheckerboard(_pglStimulus):
    '''
    Base class for checkerboard stimuli.
    '''
    def __init__(self, pgl,x=0,y=0,width=None,height=None,checkWidth=1.0,checkHeight=1.0,temporalFrequency=1.0,color=None,temporalSquareWave=True):
        '''
        Initialize the checkerboard stimulus.
        '''
        super().__init__(pgl)
        self.x = x
        self.y = y
        self.width = pgl.screenWidth.deg if width is None else width
        self.height = pgl.screenHeight.deg if height is None else height
        self.checkWidth = checkWidth
        self.checkHeight = checkHeight
        self.color = pgl.validateColor(color, n=2, forceN=True, withAlpha=False) if color is not None else np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
        self.startTime = pgl.getSecs()
        self.temporalFrequency = temporalFrequency
        self.temporalSquareWave = temporalSquareWave

        if self.width <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ width must be greater than 0, resetting to screen width.")
            self.width = pgl.screenWidth.deg

        if self.height <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ height must be greater than 0, resetting to screen height.")
            self.height = pgl.screenHeight.deg

    def __repr__(self):
        return f"<pglStimulusCheckerboard: center: ({self.x}, {self.y}) size: {self.width}x{self.height} checkSize: {self.checkWidth}x{self.checkHeight} color: {self.color}>"

################################################################
# Flickering Checkerboard stimulus class
################################################################
class pglStimulusCheckerboardFlickering(_pglStimulusCheckerboard):
    def display(self, stimulusPhase=0):
        '''
        Display the checkerboard stimulus
        '''
        # x coordinates
        xMin = self.x - self.width / 2
        xMax = self.x + self.width / 2
        
        # calculate the phase of the checkerboard
        phase = stimulusPhase + self.temporalFrequency * 2 * np.pi * (self.pgl.getSecs() - self.startTime)
        
        # Make the temporal pattern a square wave if set
        # otherwise sinusoidal motion
        if self.temporalSquareWave:
            cosPhase = np.where(np.cos(phase) >= 0, 1, 0)
        else:
          cosPhase = np.cos(phase)
        
        # calculate x coordinates
        xCoords = np.arange(xMin + cosPhase*self.checkWidth, xMax, self.checkWidth)
        if (xCoords[-1] < xMax): xCoords = np.append(xCoords, xMax)
        if (xCoords[0] < xMin): xCoords[0] = xMin
        xCoords = np.insert(xCoords, 0, xMin)
        # calculate the number of x coordinates
        Nx = len(xCoords) - 1

        # y coordinates
        yMin = self.y - self.height / 2
        yMax = self.y + self.height / 2
        yCoords = np.arange(yMin, yMax, self.checkHeight)
        if (yCoords[-1] < yMax): yCoords = np.append(yCoords, yMax)
        if (yCoords[0] > yMin): yCoords = np.insert(yCoords, 0, yMin)
        Ny = len(yCoords) - 1

        # create a checkerboard pattern
        iQuad = 0
        quad = np.zeros((Nx * Ny, 4, 2), dtype=np.float32)
        colors = np.zeros((Nx * Ny, 3), dtype=np.float32)
        for jCoord in range(len(yCoords)-1):
            colorIndex = jCoord % 2
            for iCoord in range(len(xCoords)-1):
                quad[iQuad,:,:] = np.array([[xCoords[iCoord], yCoords[jCoord]],
                                         [xCoords[iCoord+1], yCoords[jCoord]],
                                         [xCoords[iCoord+1], yCoords[jCoord+1]],
                                         [xCoords[iCoord], yCoords[jCoord+1]]])
                colors[iQuad,:] = self.color[colorIndex % 2]
                colorIndex += 1
                iQuad += 1
        # draw the checkerboard
        self.pgl.quad(quad, color=colors)

################################################################
# Sliding Checkerboard stimulus class
################################################################
class pglStimulusCheckerboardSliding(_pglStimulusCheckerboard):
    def display(self, stimulusPhase=0):
        '''
        Display the checkerboard stimulus
        '''
        # x coordinates
        xMin = self.x - self.width / 2
        xMax = self.x + self.width / 2
        
        # calculate the phase of the checkerboard
        phase = -1 + 2 * ((stimulusPhase + self.temporalFrequency * (self.pgl.getSecs() - self.startTime)) % 1)

        # calculate x coordinates that slide in one direction 
        xCoordsOdd = np.arange(xMin + phase*self.checkWidth, xMax, self.checkWidth)        
        if (xCoordsOdd[-1] < xMax): xCoordsOdd = np.append(xCoordsOdd, xMax)
        if (phase <= 0): xCoordsOdd[0] = xMin        
        xCoordsOdd = np.insert(xCoordsOdd, 0, xMin)

        # calculate the number of x coordinates
        NxOdd = len(xCoordsOdd) - 1

        # calculate x coordinates that slide in the other direction 
        xCoordsEven = np.arange(xMin - phase*self.checkWidth, xMax, self.checkWidth)        
        if (xCoordsEven[-1] < xMax): xCoordsEven = np.append(xCoordsEven, xMax)
        if (phase > 0): xCoordsEven[0] = xMin        
        xCoordsEven = np.insert(xCoordsEven, 0, xMin)

        # calculate the number of x coordinates
        NxEven = len(xCoordsEven) - 1

        # y coordinates
        yMin = self.y - self.height / 2
        yMax = self.y + self.height / 2
        yCoords = np.arange(yMin, yMax, self.checkHeight)
        if (yCoords[-1] < yMax): yCoords = np.append(yCoords, yMax)
        if (yCoords[0] > yMin): yCoords = np.insert(yCoords, 0, yMin)
        Ny = len(yCoords) - 1

        Nx = max(NxOdd, NxEven)
        # create a checkerboard pattern
        iQuad = 0
        quad = np.zeros((Nx * Ny, 4, 2), dtype=np.float32)
        colors = np.zeros((Nx * Ny, 3), dtype=np.float32)
        for jCoord in range(len(yCoords)-1):
            if jCoord % 2 == 0:
                # even rows use xCoordsEven
                xCoords = xCoordsEven
            else:
                # odd rows use xCoordsOdd
                xCoords = xCoordsOdd
            colorIndex = jCoord % 2
            for iCoord in range(len(xCoords)-1):
                quad[iQuad,:,:] = np.array([[xCoords[iCoord], yCoords[jCoord]],
                                            [xCoords[iCoord+1], yCoords[jCoord]],
                                            [xCoords[iCoord+1], yCoords[jCoord+1]],
                                            [xCoords[iCoord], yCoords[jCoord+1]]])
                colors[iQuad,:] = self.color[colorIndex % 2]
                colorIndex += 1
                iQuad += 1
        # draw the checkerboard
        self.pgl.quad(quad, color=colors)
 
################################################################
# Radial Checkerboard stimulus class
################################################################
class _pglStimulusRadialCheckerboard(_pglStimulus):
    '''
    Base class for radial checkerboard stimuli.
    '''
    def __init__(self, pgl, x=0, y=0, radialWidth=360, theta=0, outerRadius=None, innerRadius=None, checkRadialWidth = 15.0, checkRadialLength=1.0, temporalFrequency = 1.0, temporalSquareWave=True, color=None):
        '''
        Initialize the checkerboard stimulus.
        '''
        super().__init__(pgl)
        self.x = x
        self.y = y
        self.radialWidth = np.deg2rad(radialWidth)
        self.theta = np.deg2rad(theta)

        minRadius = min(self.pgl.screenWidth.deg,self.pgl.screenHeight.deg)/2 
        self.innerRadius = 0 if innerRadius is None else innerRadius
        self.outerRadius = minRadius if outerRadius is None else outerRadius

        self.checkRadialWidth = np.deg2rad(checkRadialWidth)
        self.checkRadialLength = checkRadialLength
        self.color = pgl.validateColor(color, n=2, forceN=True, withAlpha=False) if color is not None else np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
        
        self.startTime = pgl.getSecs()
        self.temporalFrequency = temporalFrequency
        self.temporalSquareWave = temporalSquareWave

        if self.radialWidth <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ radialWidth must be greater than 0, resetting to 360 degrees.")
            self.radialWidth = np.deg2rad(360)

        if self.innerRadius < 0:
            print("(pgl:pglStimuli:checkerboard) ❌ innerRadius must be greater than or equal to 0, resetting to 0.")
            self.innerRadius = 0

        if self.outerRadius <= 0:
            print("(pgl:pglStimuli:checkerboard) ❌ outerRadius must be greater than 0, resetting to minRadius.")
            self.outerRadius = minRadius

    def __repr__(self):
        return f"<pglStimulusCheckerboard: center: ({self.x}, {self.y}) theta: {np.rad2deg(self.theta)} radius: {self.innerRadius}->{self.outerRadius} radialWidth: {np.rad2deg(self.radialWidth)} checkSize: {np.rad2deg(self.checkRadialWidth)}x{self.checkRadialLength} color: {self.color}>"

################################################################
# Flickering Radial Checkerboard stimulus class
################################################################
class pglStimulusRadialCheckerboardFlickering(_pglStimulusRadialCheckerboard):
    def display(self, stimulusPhase=0):
        '''
        Display the checkerboard stimulus
        '''

        # calculate the phase of the checkerboard
        phase = stimulusPhase + self.temporalFrequency * 2 * np.pi * (self.pgl.getSecs() - self.startTime)
        
        # Make the temporal pattern a square wave if set
        # otherwise sinusoidal motion
        if self.temporalSquareWave:
            cosPhase = np.where(np.cos(phase) >= 0, 1, 0)
        else:
          cosPhase = np.cos(phase)

        # calculate r coordinates
        rCoords = np.arange(self.innerRadius + cosPhase*self.checkRadialLength, self.outerRadius, self.checkRadialLength)
        if (rCoords[-1] < self.outerRadius): rCoords = np.append(rCoords, self.outerRadius)
        if (rCoords[0] < self.innerRadius): rCoords[0] = self.innerRadius
        rCoords = np.insert(rCoords, 0, self.innerRadius)
        # calculate the number of r coordinates
        Nr = len(rCoords) - 1

        # theta coordinates
        thetaMin = self.theta - self.radialWidth / 2
        thetaMax = self.theta + self.radialWidth / 2
        thetaCoords = np.arange(thetaMin, thetaMax + self.checkRadialWidth, self.checkRadialWidth)
        Ntheta = len(thetaCoords) - 1

        # create a checkerboard pattern
        iQuad = 0
        quad = np.zeros((Nr * Ntheta, 4, 2), dtype=np.float32)
        colors = np.zeros((Nr * Ntheta, 3), dtype=np.float32)
        for jCoord in range(len(thetaCoords)-1):
            colorIndex = jCoord % 2
            for iCoord in range(len(rCoords)-1):
                quad[iQuad,:,:] = np.array([[self.x + rCoords[iCoord] * np.cos(thetaCoords[jCoord]), self.y + rCoords[iCoord] * np.sin(thetaCoords[jCoord])],
                                         [self.x + rCoords[iCoord+1] * np.cos(thetaCoords[jCoord]), self.y + rCoords[iCoord+1] * np.sin(thetaCoords[jCoord])],
                                         [self.x + rCoords[iCoord+1] * np.cos(thetaCoords[jCoord+1]), self.y + rCoords[iCoord+1] * np.sin(thetaCoords[jCoord+1])],
                                         [self.x + rCoords[iCoord] * np.cos(thetaCoords[jCoord+1]), self.y + rCoords[iCoord] * np.sin(thetaCoords[jCoord+1])]])
                colors[iQuad,:] = self.color[colorIndex % 2]
                colorIndex += 1
                iQuad += 1
        # draw the checkerboard
        self.pgl.quad(quad, color=colors)

################################################################
# Sliding Radial checkerboard stimulus class
################################################################
class pglStimulusRadialCheckerboardSliding(_pglStimulusRadialCheckerboard):
    def display(self, stimulusPhase=0):
        '''
        Display the checkerboard stimulus
        '''
        # calculate the phase of the checkerboard
        phase = -1 + 2 * ((stimulusPhase + self.temporalFrequency * (self.pgl.getSecs() - self.startTime)) % 1)

        # calculate r coordinates that slide in one direction 
        rCoordsOdd = np.arange(self.innerRadius + phase*self.checkRadialLength, self.outerRadius, self.checkRadialLength)
        if (rCoordsOdd[-1] < self.outerRadius): rCoordsOdd = np.append(rCoordsOdd, self.outerRadius)
        if (phase <= 0): rCoordsOdd[0] = self.innerRadius        
        rCoordsOdd = np.insert(rCoordsOdd, 0, self.innerRadius)

        # calculate the number of r coordinates
        NrOdd = len(rCoordsOdd) - 1

        # calculate r coordinates that slide in the other direction 
        rCoordsEven = np.arange(self.innerRadius - phase*self.checkRadialLength, self.outerRadius, self.checkRadialLength)        
        if (rCoordsEven[-1] < self.outerRadius): rCoordsEven = np.append(rCoordsEven, self.outerRadius)
        if (phase > 0): rCoordsEven[0] = self.innerRadius        
        rCoordsEven = np.insert(rCoordsEven, 0, self.innerRadius)

        # calculate the number of r coordinates
        NrEven = len(rCoordsEven) - 1

        # theta coordinates
        thetaMin = self.theta - self.radialWidth / 2
        thetaMax = self.theta + self.radialWidth / 2
        thetaCoords = np.arange(thetaMin, thetaMax + self.checkRadialWidth, self.checkRadialWidth)
        Ntheta = len(thetaCoords) - 1

        Nr = max(NrOdd, NrEven)
        # create a checkerboard pattern
        iQuad = 0
        quad = np.zeros((Nr * Ntheta, 4, 2), dtype=np.float32)
        colors = np.zeros((Nr * Ntheta, 3), dtype=np.float32)
        for jCoord in range(len(thetaCoords)-1):
            if jCoord % 2 == 0:
                # even rows use rCoordsEven
                rCoords = rCoordsEven
            else:
                # odd rows use rCoordsOdd
                rCoords = rCoordsOdd
            colorIndex = jCoord % 2
            for iCoord in range(len(rCoords)-1):
                quad[iQuad,:,:] = np.array([[self.x + rCoords[iCoord] * np.cos(thetaCoords[jCoord]), self.y + rCoords[iCoord] * np.sin(thetaCoords[jCoord])],
                                            [self.x + rCoords[iCoord+1] * np.cos(thetaCoords[jCoord]), self.y + rCoords[iCoord+1] * np.sin(thetaCoords[jCoord])],
                                            [self.x + rCoords[iCoord+1] * np.cos(thetaCoords[jCoord+1]), self.y + rCoords[iCoord+1] * np.sin(thetaCoords[jCoord+1])],
                                            [self.x + rCoords[iCoord] * np.cos(thetaCoords[jCoord+1]), self.y + rCoords[iCoord] * np.sin(thetaCoords[jCoord+1])]])
                colors[iQuad,:] = self.color[colorIndex % 2]
                colorIndex += 1
                iQuad += 1
        # draw the checkerboard
        self.pgl.quad(quad, color=colors)
       
################################################################
# Bar stimulus class
################################################################
class pglStimulusBar(_pglStimulus):
    def __init__(self, pgl, width=1.0, speed=1.0, dir=0, stepPosition=True):
        super().__init__(pgl)
        
        # save parameters
        self.width = width
        self.speed = speed
        self.dir = dir
        self.stepPosition = stepPosition
        # set the height to the diagonal length (so that the bars
        # display full screen even if they are moving in an oblique direction)
        self.height = np.sqrt(pgl.screenHeight.deg**2 + pgl.screenWidth.deg**2)
                
        # create the bar stimulus
        self.barStimulus = pglStimulusCheckerboardSliding(pgl, width=self.width, height=self.height)

        # start the pass
        self.initPass()
        
    def display(self, dir=0):
        '''
        Display the bar stimulus.
        '''
        # Update the bar stimulus position
        if self.stepPosition:
            # only update, every time the stimulus moves a full width
            self.thisX = self.passStartX + self.speed * (self.pgl.getSecs() - self.passStartTime)
            if self.thisX - self.barStimulus.x > self.width:
                self.barStimulus.x = self.thisX
        else:
            # smoothly move
            self.barStimulus.x = self.passStartX + self.speed * (self.pgl.getSecs() - self.passStartTime)

        # display the bar
        self.barStimulus.display()
        
        # check for end of pass
        if self.barStimulus.x > self.passEndX:
            # reset for next pass
            self.initPass()
                  
    def initPass(self):
        '''
        Initialize the pass parameters for the bar stimulus.
        '''
        # save direction
        self.passDir = self.dir

        # set time
        self.passStartTime = self.pgl.getSecs()
        
        # set the x, y start and end positions
        self.passStartX = -self.pgl.screenWidth.deg/2 - self.width/2
        self.passEndX = self.pgl.screenWidth.deg/2 + self.width/2
        self.passStartY = 0
        self.passEndY = 0
        
        # set the bar stimulus start position            
        self.barStimulus.x = self.passStartX
        self.barStimulus.y = self.passStartY
        
        # rotate coordinate frame accordingly
        self.pgl.setTransformRotation(self.passDir)


################################################################
# Movie stimulus class
################################################################
class pglStimulusMovie(_pglStimulus):
    
    '''
    Base class for movie stimuli.
    '''
    # error codes
    movieError = {-1.0: "File not found",
                  -2.0: "No permission for file",
                  -3.0: "Invalid format",
                  -4.0: "Error reading vertices",
                  -5.0: "Error creating movie",
                  -6.0: "Error adding movie"}
    # default dimensions
    width = 0
    height = 0
    
    # arrays of timestamps to be filled in by drawFrames
    drawFrameTimes = []
    targetPresentationTimestamps = []
    movieTimes = []
    
    def __init__(self, pgl, filename="", x=0, y=0, displayWidth=0, displayHeight=0, xAlign=0, yAlign=0):
        '''
        Initialize the movie stimulus with a movie instance.

        Args:
        '''
        super().__init__(pgl)
        self.pgl = pgl
        self.filename = filename

        # make sure that a screen is open
        if self.pgl.isOpen() is False: 
            print(f"(pglStimulusMovie) ❌ No screen is open")
            return False
        

        self.pgl.s.writeCommand("mglMovieCreate")
        ackTime = self.pgl.s.readAck()
        self.pgl.s.write(filename)
        
        result = self.pgl.s.read(np.float64)
        if (result < 0): 
            print(f"(pglStimulusMovie:init) Error creating movie: {self.movieError.get(int(result),"Unrecogonized Error")}")
            self.commandResults = self.pgl.s.readCommandResults(ackTime)
            return None
        
        # means we have details for mvoe
        if result>1.0:
            # read details of movie
            self.frameRate = self.pgl.s.read(np.float64)
            self.duration = self.pgl.s.read(np.float64)
            self.totalFrames = self.pgl.s.read(np.uint32)
            self.width = self.pgl.s.read(np.uint32)
            self.height = self.pgl.s.read(np.uint32)
            print(self)

        # Read the movie number
        self.movieNum = self.pgl.s.read(np.uint32)
        nMovies = self.pgl.s.read(np.uint32)      
        if self.pgl.verbose>0:  
            print(f"(pglStimulusMovie:init) Created movie {self.movieNum} ({nMovies} total movies)")
        
        # get command results
        self.commandResults = self.pgl.s.readCommandResults(ackTime)
        
        # set the display position
        self.setDisplayPosition(x, y, displayWidth, displayHeight, xAlign, yAlign)


    def __repr__(self):
        return f"<pglStimulusMovie: {self.filename} {self.width}x{self.height}pix, duration={self.duration}s, {self.frameRate}fps, totalFrames={self.totalFrames}>"

    def setDisplayPosition(self, x = 0, y = 0, displayWidth = 0, displayHeight = 0, xAlign = 0, yAlign = 0):
        '''
        Sets the display position
        
        args:
        -  x: x location for movie (default = 0)
        -  y: y location for movie (default = 0)
        -  displayWidth: width for movie (default = 0) If height is set, and width is 0 
            then width will default to correct value to maintain aspect ratio if the movie
            information has been loaded. Otherwise will default to screenWidth
        -  displayHeight: height for movie (default = 0) Behaves the same as displayWidth
        -  xAlign: xAlginment (-1 left, 0 center, 1 right: default = 0)
        -  yAlign: xAlginment (-1 top, 0 center, 1 bottom: default = 0)
        '''
        # handle width and height defaults
        if displayWidth == 0 and displayHeight == 0:
            # no width or height, maximize to fill screen
            displayWidth = self.pgl.screenWidth.deg
            displayHeight = self.pgl.screenHeight.deg  
        elif displayWidth == 0:
            # get width based on aspect ration of movie
            if width == 0 and height == 0:
                displayWidth = self.pgl.screenWidth.deg
            else:
                # set according to aspect ratio
                displayWidth = (width / height) * displayHeight
        elif displayHeight == 0:
            # get width based on aspect ration of movie
            if width == 0 and height == 0:
                displayHeight = self.pgl.screenHeight.deg
            else:
                # set according to aspect ratio
                displayHeight = (height / width) * displayWidth               
        
        # vertex coordinates in device coordinates
        displayLeft = x - (xAlign + 1) / 2 * displayWidth
        displayRight = displayLeft + displayWidth
        displayTop = y + (yAlign + 1) / 2 * displayHeight
        displayBottom = displayTop - displayHeight

        # no z coordinate
        z = 0

        # texture coordinates which map to vertex coordinates
        texRight = 1
        texLeft = 0
        texTop = 0
        texBottom = 1
        
        # create the two triangles which map the texture (ie image)
        # to vertices in device coordinates
        vertices = np.array([
            [displayRight, displayTop, z, texRight, texTop],
            [displayLeft, displayTop, z, texLeft, texTop],
            [displayLeft, displayBottom, z, texLeft, texBottom],

            [displayRight, displayTop, z, texRight, texTop],
            [displayLeft, displayBottom, z, texLeft, texBottom],
            [displayRight, displayBottom, z, texRight, texBottom]
        ], dtype=np.float32) 
        nVertices = np.float32(vertices.shape[0])

        # write command for display position
        self.pgl.s.writeCommand("mglMovieSetDisplayPosition")
        ackTime = self.pgl.s.readAck()
        self.pgl.s.write(np.uint32(self.movieNum))
        
        # write the vertices
        self.pgl.s.write(np.uint32(nVertices))
        self.pgl.s.write(vertices)
        
        # get command results
        self.commandResults = self.pgl.s.readCommandResults(ackTime)

    def play(self):
        '''
        play the Movie.
        
        '''
        self.pgl.s.writeCommand("mglMoviePlay")
        ackTime = self.pgl.s.readAck()
        self.pgl.s.write(np.uint32(self.movieNum))
        
        # read drawable presentedTimes
        self.presentedTimes = self.pgl.s.readArray(np.float64)
        
        # read corresponding movie times - this is the corresponding time in the movie
        self.movieTimes = self.pgl.s.readArray(np.float64)
        
        # read corresponding targetPresentationTimestamps (the time the os thinks the frame will be displayed)
        self.targetPresentationTimestamps = self.pgl.s.readArray(np.float64)

        # read corresponding draw frame times (the time at which the code to draw the frame
        # ran - which is different from when it was displayed which is presentedTimes)
        self.drawFrameTimes = self.pgl.s.readArray(np.float64)

        # command Results
        self.commandResults = self.pgl.s.readCommandResults(ackTime)
        
        # print summary if verbose
        if self.pgl.verbose>0:
            print(f"(pglStimulusMovie:play) Movie played {len(self.presentedTimes)} frames.")

        # display frame statistics if versbose is set
        if self.pgl.verbose>0:
            self.displayFrameStatistics()

    # display frame statistics    
    def displayFrameStatistics(self):
        
        # offset all time values to the first non-zero presentation timestamp
        offsetValue = next((v for v in self.targetPresentationTimestamps if v != 0), 0)
        drawFrameTimes = 1000*(self.drawFrameTimes - offsetValue)
        targetPresentationTimestamps = 1000*(self.targetPresentationTimestamps - offsetValue)
        presentedTimes = np.where((self.presentedTimes == -1) | (self.presentedTimes == 0), -1, 1000 * (self.presentedTimes - offsetValue))
        
        # convert movie time to ms
        movieTimes = np.where((self.movieTimes == -1), -1, 1000 * self.movieTimes)
        
        # compute how long each frame was presented for, i.e. the difference
        # between consecutive presentedTimes, but put -1 if either value is -1
        frameLen = np.array([
            presentedTimes[i] - presentedTimes[i-1] if presentedTimes[i] != -1 and presentedTimes[i-1] != -1 else -1
            for i in range(1, len(presentedTimes))
        ])
        
        # compute the discrepancy for all frames in frameTimes
        # that have a value with presentedTimes. We
        # will use this to determine how closely the presented times
        # match the frame times
        lastValidMovieTime = None
        lastPresentedTime = None
        delay = []
        for i in range(len(movieTimes)):
            if movieTimes[i] != -1:
                lastValidMovieTime = movieTimes[i]
            # Use last valid frame if current is -1
            movieTime = lastValidMovieTime if lastValidMovieTime is not None else 0
            # if presentedTiems is -1 it is a dropped frame
            if presentedTimes[i] == -1 or presentedTimes[i] == 0:
                # this means a frame was dropped, so the last presented frame
                # is still displaying, so compute the delay relative to that frame
                if lastPresentedTime != None:
                    delay.append(lastPresentedTime - movieTime)
                else:
                    # if there is no lastPresentedTime, then nan
                    delay.append(np.nan)
            else:
                # compute difference betwen the presentedTime and the frameTime
                delay.append(presentedTimes[i] - movieTime)
                # keep this presented time, in case there is a dropped frame
                lastPresentedTime = presentedTimes[i]


        # Determine the maximum width dynamically to represent the numbers
        maxWidth = max(len(f"{v:.1f}") for v in drawFrameTimes) + 1  # +1 for spacing

        # ok, print out timeline. 
        # frame num: Is the frame number from start of movie play (starting at 0)
        # drawFrame: Is the time at which drawFrame is called
        # target: Is the targetPresentationTimestamp which is the predicted time of frame display
        # presented: is the time that the frame was actually presented
        # frameLen: Is how long the frame was shown for
        # Movie time: is the time in the movie that is being displayed
        # Delay: is the difference in time between movie time and presented time (negative
        # values mean that the frame was displayed before the movie time)
        print("frame num: " + " ".join(f"{v:{maxWidth}.0f}" for v in np.arange(len(presentedTimes))))            
        print("drawFrame: " + " ".join(f"{v:{maxWidth}.1f}" for v in drawFrameTimes))            
        print("target:    " + " ".join(f"{v:{maxWidth}.1f}" for v in targetPresentationTimestamps))            
        print("presented: " + " ".join(
            f"{v:{maxWidth}.1f}" if v != -1 else f"{'-':>{maxWidth}}"
            for v in presentedTimes
        ))    
        print("frameLen:  " + " ".join(
            f"{v:{maxWidth}.1f}" if v != -1 else f"{'-':>{maxWidth}}"
            for v in frameLen
        ))    
        print("Movie time:" + " ".join(
            f"{v:{maxWidth}.1f}" if v != -1 else f"{'-':>{maxWidth}}"
            for v in movieTimes
        ))    
        print("Delay:     " + " ".join(
            f"{v:{maxWidth}.1f}" if v != -1 else f"{'-':>{maxWidth}}"
            for v in delay
        ))  
        print(f"Delay: {np.nanmean(delay):.1f} ± {np.nanstd(delay):.1f} ms")        
  
        # report dropped frames (i.e when presentedTimes = -1)
        droppedFrameIndexes = [i for i, t in enumerate(presentedTimes) if t == -1]
        print(f"{len(droppedFrameIndexes)} dropped frames:" + " ".join(str(i) for i in droppedFrameIndexes))

        # Compute differences between consecutive elements
        presentedTimesNoZeros = [t for t in presentedTimes if t != -1]
        frameLens = np.diff(presentedTimesNoZeros)  # diffs[i] = times[i+1] - times[i]

        # now we will compute how many frames are in what frameRate, 
        # we will test against the expected frameTime (e.g. 1/120) with
        # this tolerance in %
        tolerance = 0.05

        # Expected frame durations, based on multiples of the refresh rate
        multiples = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])
        refreshRate = self.pgl.getFrameRate()
        expectedFrameDuration = 1000 * multiples / refreshRate  # milliseconds per frame
        labels = [f"{int(refreshRate / m)}Hz" for m in multiples]

        # Prepare classification dict
        classified = {label: [] for label in labels}
        unclassified = []

        for idx, t in enumerate(frameLens):
            matched = False
            for label, expected in zip(labels, expectedFrameDuration):
                # check length against expected duration with tolerance
                if abs(t - expected) / expected <= tolerance:                    
                    classified[label].append(idx)
                    matched = True
                    break
            if not matched:
                unclassified.append((idx, t))

        totalFrames = len(frameLens)

        # Print report of how many frames were classified into each category
        for label in labels:
            indices = classified[label]
            count = len(indices)
            if count > 0:
                percent = count / totalFrames * 100
                print(f"{count} frames ({percent:.1f}%) at {label}: {' '.join(str(i) for i in indices)}")

        # Print unclassified
        if unclassified:
            count = len(unclassified)
            percent = count / totalFrames * 100
            unclassifiedString = " ".join(f"{i} ({t*1000:.2f}ms)" for i, t in unclassified)
            print(f"{count} frames ({percent:.1f}%) unclassified: {unclassifiedString}")
        
    def drawFrame(self):
        '''
        draw the current frame of the movie (requires a pgl.flush() to display).
        '''
        self.pgl.s.writeCommand("mglMovieDrawFrame")
        ackTime = self.pgl.s.readAck()
        self.pgl.s.write(np.uint32(self.movieNum))

        self.drawFrameTimes.append(self.pgl.s.read(np.float64))
        self.targetPresentationTimestamps.append(self.pgl.s.read(np.float64))
        self.movieTimes.append(self.pgl.s.read(np.float64))

        self.commandResults = self.pgl.s.readCommandResults(ackTime)

    def status(self):
        '''
        get the status of the Movie.
        
        '''
        self.pgl.s.writeCommand("mglMovieStatus")
        ackTime = self.pgl.s.readAck()
        self.pgl.s.write(np.uint32(self.movieNum))

        status = self.pgl.s.read(np.float64)
        print(f"(pglStimulusMovie:status) Movie status: {status}")
        if status>0:
            # read details of movie
            self.frameRate = self.pgl.s.read(np.float64)
            self.duration = self.pgl.s.read(np.float64)
            self.totalFrames = self.pgl.s.read(np.uint32)
            self.width = self.pgl.s.read(np.uint32)
            self.height = self.pgl.s.read(np.uint32)


        self.commandResults = self.pgl.s.readCommandResults(ackTime)
        print(self.pgl.commandResults)

        
    def print(self):
        '''
        Print information about the stimulus.
        '''
        print(self.__repr__())

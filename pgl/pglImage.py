################################################################
#   filename: pglImage.py
#    purpose: Module for displaying images
#         by: JLG
#       date: July 22, 2025
################################################################

#############
# Import modules
#############
import numpy as np
from types import SimpleNamespace
from .pglMessages import pglMessages
from .pglBase import pglBase
from .pglSettings import pglTraitSettings
from PIL import Image, UnidentifiedImageError
from traitlets import HasTraits, Float, Int, List, Tuple, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from fsspec import AbstractFileSystem
import posixpath
import os
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#############
# Image class
#############
class pglImage:
    '''
    Class for displaying images.

    '''

    def imageCreate(self, imageData):
        '''
            Creates a new image from the provided image data.

            Args:
                imageData: The image data to create the image from. Can be HxW grayscale,
                 HxWx3 RGB, or HxWx4 RGBA data.

            Returns:
                An instance of pglImageInstance or None if the image could not be created.
                The pglImageInstance can be displayed by calling imageDisplay(). This is used
                if you want to display an image multiple times and want to save the overhead
                of creating the image each type. If you want to just display once, you can
                just call imageDisplay().

            Example:
                pgl.open()
                imageData = np.random.rand(480, 640, 3)
                imageInstance = pgl.imageCreate(imageData)
                if imageInstance is not None:
                    pgl.imageDisplay(imageInstance)

        '''
     
        # check dimensions of imageData
        (tf, imageData) = self.imageValidate(imageData)
        if not tf: return None
    
        # get image width and height
        imageWidth = imageData.shape[1]
        imageHeight = imageData.shape[0]

        # flatten and convert to float32
        imageData = imageData.astype(np.float32).flatten()

        # send the createTexture command
        self.s.writeCommand("mglCreateTexture")
        ackTime = self.s.readAck()

        # send the image width, height and data
        self.s.write(np.uint32(imageWidth))
        self.s.write(np.uint32(imageHeight))
        if self.verbose>1: print(f"(pglImage:imageCreate) Creating image {imageWidth}x{imageHeight}")
        self.s.write(imageData)

        # read the imageNum
        result = self.s.read(np.float64)
        if (result < 0): 
            print("(pglImage:imageCreate) Error creating image")
            return None
    
        # create an imageInstance with all the info
        imageNum = self.s.read(np.uint32)
        nImages = self.s.read(np.uint32)
        if self.verbose>1: print(f"(pglImage:imageCreate) Created image {imageNum} ({nImages} total images)")
        self.s.readCommandResults(ackTime)

        # create an instance of pglImageInstance
        imageInstance = pglImageInstance(imageNum, imageWidth, imageHeight, self)
        return imageInstance
    
    def imageDisplay(self, imageInstance, x=None, y=None, width=None, height=None, xAlign=0, yAlign=0):
        '''
        Displays an image. If you want to display an image multiple times and want
        to avoid the overhead of creating the image each time, you can use the
        pglImageInstance returned by imageCreate.

        Args:
            imageInstance: Either what is returned by imageCreate or a numpy matrix
            x,y: The location to display the image. Defaults to 0,0
            width,height: The size to display the image. If only one is set, then the
                other one will be set to maintain the aspect ratio. If neither are set, then
                will display at full resolution given the pixel dimensions or if the image
                is bigger then the window to fit within the window         
            xAlign: Horizontal alignment of the image relative to x. -1=left, 0=center, 1=right
            yAlign: Vertical alignment of the image relative to y. -1=top, 0=center, 1=bottom

        Returns:
            None

        Example:
            pgl.open()
            imageData = np.random.rand(480, 640, 3)
            imageInstance = pgl.imageCreate(imageData)
            if imageInstance is not None:
                pgl.imageDisplay(imageInstance, x=100, y=100, width=200, height=200)

        '''
        if self.isOpen() == False:
            print("(pgl:pglStimulus:display) pgl is not open. Cannot display image.")
            return None
        
        # check for image passed in
        if not isinstance(imageInstance, pglImageInstance):
            imageInstance = self.imageCreate(imageInstance)
            if imageInstance is None: return None

        if width is None:
            if height is None:
                widthRatio = imageInstance.width.pix / self.screenWidth.pix
                heightRatio = imageInstance.height.pix / self.screenHeight.pix
                maxRatio = max(widthRatio, heightRatio)
                if maxRatio > 1:
                    # scale down to fit within window
                    width = imageInstance.width.pix * self.xPix2Deg / maxRatio
                    height = imageInstance.height.pix * self.yPix2Deg / maxRatio
                else:
                    # default size is image size
                    width = imageInstance.width.pix * self.xPix2Deg
                    height = imageInstance.height.pix * self.yPix2Deg
            else:
                # make width proportional to height
                width = height * (imageInstance.width.pix / imageInstance.height.pix)
        if height is None:
            # default location is center of screen
            height = width * (imageInstance.height.pix / imageInstance.width.pix)
        if x is None: x = 0
        if y is None: y = 0

        # vertex coordinates in device coordinates
        imageInstance.displayLeft = x - (xAlign + 1) / 2 * width
        imageInstance.displayRight = imageInstance.displayLeft + width
        imageInstance.displayTop = y + (yAlign + 1) / 2 * height
        imageInstance.displayBottom = imageInstance.displayTop - height
        imageInstance.displayWidth = width
        imageInstance.displayHeight = height

        # keep this coordinates for reference
        imageInstance.displayed = True
        imageInstance.displayTime = self.getDateAndTime()

        if self.verbose>1:
            print(f"(pglImage:imageDisplay) Displaying image {imageInstance.imageNum} at {displayLocation} with size {displaySize}.")
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
            [imageInstance.displayRight, imageInstance.displayTop, z, texRight, texTop],
            [imageInstance.displayLeft, imageInstance.displayTop, z, texLeft, texTop],
            [imageInstance.displayLeft, imageInstance.displayBottom, z, texLeft, texBottom],

            [imageInstance.displayRight, imageInstance.displayTop, z, texRight, texTop],
            [imageInstance.displayLeft, imageInstance.displayBottom, z, texLeft, texBottom],
            [imageInstance.displayRight, imageInstance.displayBottom, z, texRight, texBottom]
        ], dtype=np.float32) 
        nVertices = np.float32(vertices.shape[0])

        self.s.writeCommand("mglBltTexture")
        ackTime = self.s.readAck()
        self.s.write(np.uint32(imageInstance.minMagFilter))
        self.s.write(np.uint32(imageInstance.mipFilter))
        self.s.write(np.uint32(imageInstance.addressMode))
        self.s.write(np.uint32(nVertices))
        self.s.write(vertices)
        self.s.write(np.float32(imageInstance.phase))
        self.s.write(np.uint32(imageInstance.imageNum))
        self.commandResults = self.s.readCommandResults(ackTime)
 
    def imageDelete(self, imageInstance):
        '''
            Delete the specified image instance. This returns memory from the mglMetal app
            which is allocated to hold the image.
        '''
        if self.isOpen() == False:
            return
        if not isinstance(imageInstance, pglImageInstance):
            print("(pglImage:imageDelete) imageInstance should be an instance of pglImageInstance.")
            return
        
        # Delete texture
        if self.verbose>1: print(f"(pglImage:imageDelete) Deleting image {imageInstance.imageNum} ({imageInstance.width.pix}x{imageInstance.height.pix})")
        # send the deleteTexture command
        self.s.writeCommand("mglDeleteTexture")
        self.s.write(np.uint32(imageInstance.imageNum))
        self.commandResults = self.s.readCommandResults()

    def imageValidate(self, imageData):
        '''
        Validate the image data and return a tuple of (True, imageData) if valid,
        or (False, None) if invalid. This will insure that images are WxHx4 numpy matrices.
        '''
        imageData = np.array(imageData)
        if not isinstance(imageData, np.ndarray) or imageData.ndim < 2 or imageData.ndim > 3:
            print("(pglImage:imageValidate) imageData should be a numpy matrix either WxH, WxHx3 or WxHx4.")
            return (False, None)
                
        # make float32
        imageData = imageData.astype(np.float32)

        # check if any alues are less than 0
        if np.any(imageData < 0):
            # if all values are between -1 and 1, we can scale them
            if np.all(imageData >= -1) and np.all(imageData <= 1):
                imageData = (imageData + 1) / 2
                if self.verbose>1: print(f"(pglImage:imageValidate) imageData values were scaled from [{-1}, {1}] to [0, 1].")
            else:
                # scale between min and max
                minVal = np.min(imageData)
                maxVal = np.max(imageData)
                imageData = (imageData - minVal) / (maxVal - minVal)
                if self.verbose>1: print(f"(pglImage:imageValidate) imageData values were scaled from [{minVal}, {maxVal}] to [0, 1].")
        # check if any values are greater than 1
        elif np.any(imageData > 1):
            # if all values are whole numbers between 0 and 255, this is an 8 bit image
            if np.all(np.floor(imageData) == imageData) and np.all((imageData>=0) & (imageData<=255)):
                if self.verbose>1: print(f"(pglImage:imageValidate) imageData values were scaled from [0, 255] to [0, 1].")
                imageData = imageData / 255.0
            else:
                # scale between min and max
                minVal = np.min(imageData)
                maxVal = np.max(imageData)
                imageData = (imageData - minVal) / (maxVal - minVal)
                if self.verbose>1: print(f"(pglImage:imageValidate) imageData values were scaled from [{minVal}, {maxVal}] to [0, 1].")

        # check dimensions
        if imageData.ndim == 2:
            # assume grayscale image, convert to RGBA
            imageData = np.stack((imageData,)*4, axis=-1)
            # set alpha channel to 1
            imageData[..., 3] = 1
        elif imageData.ndim == 3 and imageData.shape[2] == 3:
            # assume RGB image, convert to RGBA
            imageData = np.concatenate((imageData, np.ones((imageData.shape[0], imageData.shape[1], 1), dtype=imageData.dtype)), axis=-1)
        elif imageData.ndim == 3 and imageData.shape[2] != 4:
            print("(pglImage:imageValidate) imageData should be a WxHx3 or WxHx4 numpy matrix.")
            return (False, None)
        
        return (True, imageData)
       

 
#container class that holds image reference
class pglImageInstance:
    # minMagFilter -- optional value to choose sampler filtering:
    #   0: nearest
    #   1: linear (default)
    minMagFilter = 1
    # mipFilter -- optional value to choose sampler filtering:
    #   0: not mipmapped
    #   1: nearest
    #   2: linear (default)
    mipFilter = 2
    # addressMode -- optional value to choose sampler addressing:
    #   0: clamp to edge
    #   1: mirror clamp to edge
    #   2: repeat (default)
    #   3: mirror repeat
    #   4: clamp to zero
    #   5: clamp to border color
    addressMode = 2
    # phase -- optional value to choose sampler phase:
    #   0: phase (default)
    phase = 0
    def __init__(self, imageNum, imageWidth, imageHeight, pgl):
        
        # keep reference to pgl 
        self.pgl = pgl
        
        # and image info
        self.width = SimpleNamespace(pix=imageWidth)
        self.height = SimpleNamespace(pix=imageHeight)
        self.imageNum = imageNum
        self.displayed = None
        if pgl.verbose>1: 
            print(f"(pglImage:pglImageInstance) Created image instance with: {self.imageNum} ({self.width.pix}x{self.height.pix})")
    def __del__(self):
        # call the pgl function 
        self.pgl.imageDelete(self)
    def display(self, x=None, y=None, width=None, height=None, xAlign=0, yAlign=0):
        '''
          Display the image at the specified location and size.
        '''
        # call the pgl function to display
        self.pgl.imageDisplay(self, x, y, width, height, xAlign, yAlign)
    def print(self):
       if self.displayed is not None:
            print(f"Image {self.imageNum} ({self.width.pix}x{self.height.pix}) displayed: left={self.displayLeft} right={self.displayRight} bottom={self.displayBottom} top={self.displayTop} time={self.displayTime}")
       else:
           print(f"Image: {self.imageNum} ({self.width.pix}x{self.height.pix})")

################################################
# pglStimulusFile
################################################
class pglStimulusFile(pglTraitSettings):
    # filesystem, name and prefix for where stimulus is saved
    name = Unicode(help="name of image")
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    filename= Unicode(allow_none=True, default_value="", help="Full path to images", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    caption = Unicode(default_value="", help="Image caption")
    condition = Unicode(default_value="", help="condition")
    conditionNum = Int(allow_none=True, default_value=None, help="conditionNum")


################################################
# pglImageFile
################################################
class pglImageFile(pglStimulusFile):

    # filesystem, name and prefix for where the images were loaded from
    _size = Tuple(Int, Int, labels=("width","height"), property="size", enabled=False, allow_none=True, default_value=None, help="Image width x height")
    _mode = Unicode(allow_none=True, default_value=None, property="mode", enabled=False, help="Image mode")
    _format = Unicode(allow_none=True, default_value=None, property="format", enabled=False, help="Image format")
    _img = Instance(Image.Image, allow_none=True, default_value=None, serialize=False, visible=False, help="The loaded PIL image")
    
    def _loadMetadata(self):
        pglMessages.message(f"Loading image metadata: {self.filesystemPrefix}/{self.filename}")
        filesystem, filename, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.filename, filesystemPrefix=self.filesystemPrefix)
        with filesystem.open(filename, "rb") as h:
            img = Image.open(h)
            self._size = img.size
            self._mode = img.mode
            self._format = img.format

    def _loadImage(self):
        pglMessages.message(f"Loading image: {self.filesystemPrefix}/{self.filename}")
        filesystem, filename, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.filename, filesystemPrefix=self.filesystemPrefix)
        with filesystem.open(filename, "rb") as h:
            img = Image.open(h)
            img.load()
            self._img = img
            # also cache metadata if not already set
            if self._size is None: self._size = img.size
            if self._mode is None: self._mode = img.mode
            if self._format is None: self._format = img.format

    @property
    def size(self):
        '''imageSize.'''
        if self._size is None: self._loadMetadata()
        return self._size

    @size.setter
    def size(self, value):
        self._size = value    

    @property
    def mode(self):
        '''imageMode.'''
        if self._mode is None: self._loadMetadata()
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value    

    @property
    def format(self):
        '''imageFormat.'''
        if self._format is None: self._loadMetadata()
        return self._format

    @format.setter
    def format(self, value):
        self._format = value    
        
    @property
    def img(self):
        '''image'''
        if self._img is None: self._loadImage()
        return self._img

    @img.setter
    def img(self, value):
        self._img = value    

    def __repr__(self):
        return f"{os.path.basename(self.filename)} {self.size[0]}x{self.size[1]} {self.mode}"
    
    def print(self):
        print(self.__repr__())
        
    def display(self, fig=None, ax=None):
        if fig:
            ax = fig.subplots()
        if ax is None:
            fig, ax = plt.subplots()
        # self.img triggers _loadImage (full pixel decode) — only now
        img = self.img
        if self.mode in ("L", "I", "F", "I;16"):
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(self.img)
        ax.set_title(repr(self))
        ax.axis("off")
        return ax
    
################################################
# pglMovieFile
################################################
class pglMovieFile(pglStimulusFile):

    def get(self):
        '''
        Get movie-stimulus reader

        Args:
            stimulusNum (int): number of stimulus to get

        Returns:
            imageio reader, or None if unavailable/nonlocal.
        '''
        filesystem, filename, _ = pglBase.validateFilesystem(filesystem=self.filesystem, dataPath=self.filename, filesystemPrefix=self.filesystemPrefix)

        # fsspec filesystems generally identify local files with protocol="file".
        protocol = filesystem.protocol
        if isinstance(protocol, (tuple, list)):
            isLocal = "file" in protocol
        else:
            isLocal = protocol == "file"

        if not isLocal:
            pglMessages.warning(
                f"Cannot yet read movie from nonlocal filesystem "
                f"({protocol}): {filename}"
            )
            return None

        # Do not open it with filesystem.open(): imageio/FFmpeg wants the
        # actual local filename, and it manages the file handle itself.
        localFilename = filesystem._strip_protocol(filename)

        return imageio.get_reader(localFilename)

    def display(self, ax=None):
        '''
        Display movie once in `ax`, without blocking.

        If ax is omitted, create a new figure and axes.

        Returns:
            animation: Keep this reference if you want to control the animation.
        '''
        if ax is None:
            fig, ax = plt.subplots()

        elif isinstance(ax, plt.Figure):
            fig = ax
            ax = fig.add_subplot(111)

        else:
            fig = ax.figure

        reader = self.get()
        if reader is None:
            return None

        fps = reader.get_meta_data().get("fps", 30)

        # Read only the first frame now, so Matplotlib has something to display.
        frameIterator = reader.iter_data()
        firstFrame = next(frameIterator)

        imageArtist = ax.imshow(firstFrame)
        ax.axis("off")

################################################
# pglStimulusDaabase
################################################
class pglStimulusDatabase(pglTraitSettings):
    
    # filesystem, name and prefix for where the images were loaded from
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    dataPath = Unicode(allow_none=True, default_value="", help="Full path to stimuli", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    stimuli = List(Instance(pglStimulusFile), default_value=[], settingsListKey="name", traitDisplayName="Choose stimulus", hasPlotButton=True, buttonFunction="display", help="List of stimuli in database")
    nStimuli = Int(0, help="Number of stimuli", enable=False)
    
    def __init__(self, dataPath=None, knownFileExtensions=None, filesystem=None, stimulusFileClass=pglStimulusFile):
        '''
        Initialize by pointing to a directory where images live
        '''
        if dataPath:
    
            # parse filesystem
            self.filesystem, self.dataPath, self.filesystemPrefix = pglBase.validateFilesystem(filesystem, dataPath)
            if not self.filesystem:
                pglMessages.warning("Could not resolve path to images")
                return
                            
            # look for stimuli in directory
            for f in self.filesystem.ls(self.dataPath, detail=True):
                # only open files
                if f["type"] != "file": continue
                if os.path.splitext(f["name"])[1].lower() in knownFileExtensions:
                    self.stimuli.append(stimulusFileClass(
                        name = os.path.basename(f["name"]),
                        filename = f["name"],
                        filesystem = self.filesystem,
                        filesystemPrefix = self.filesystemPrefix
                    ))
            # store number of images                    
            self.nStimuli = len(self.stimuli)
            
            # alphabetically sort the images by name
            self.stimuli.sort(key=lambda x: x.name.lower())
            
            # and let the world know
            pglMessages.message(f"Found {self.nStimuli} stimulus files in {os.path.basename(dataPath)}")

    def useManifest(self, filenameColumn="filename", indexColumn="index", captionColumn=None, conditionColumn=None, conditionNumColumn=None):
        """
        Load stimulus ordering and optional metadata from a CSV manifest.

        The manifest must contain filenameColumn and indexColumn. Optional
        caption, condition, and conditionNum columns are copied onto each
        stimulus object.
        """

        # Filenames of stimuli that were actually loaded
        filenames = [stimulus.name for stimulus in self.stimuli]

        # Find a CSV manifest
        self.manifest = None
        
        csvFilenames=self.filesystem.glob(os.path.join(self.dataPath, "*.csv"))
        if len(csvFilenames) == 0:
            pglMessages.warning(f"Could not find a manifest in {self.dataPath}. Must be a file with .csv extension")
            return
        elif len(csvFilenames) > 1:
            pglMessages.warning(f"Found multiple manifests in {self.dataPath}. Must be ONLY ONE file with .csv extension")
            return

        for csvFilename in csvFilenames:
            try:
                with self.filesystem.open(csvFilename) as f:
                    dataFrame = pd.read_csv(f)

                # Check required columns
                if filenameColumn not in dataFrame.columns or indexColumn not in dataFrame.columns:
                    pglMessages.message(
                        f"Could not find {filenameColumn} and/or {indexColumn} "
                        f"in {csvFilename}. Using alphabetical sort order"
                    )
                    return

                # Make sure the manifest contains exactly the loaded stimuli
                manifestFilenames = dataFrame[filenameColumn]

                if set(manifestFilenames) != set(filenames):
                    pglMessages.message(
                        f"Manifest {csvFilename} does not contain exactly the "
                        f"loaded stimulus files. Using alphabetical sort order"
                    )
                    return

                # Keep the manifest
                self.manifest = dataFrame

                # Sort stimuli according to the manifest index
                indexByFilename = dict(
                    zip(manifestFilenames, dataFrame[indexColumn])
                )

                self.stimuli.sort(
                    key=lambda stimulus: indexByFilename[stimulus.name]
                )

                # Create lookup table by filename
                manifestByFilename = dataFrame.set_index(filenameColumn)

                # Copy optional metadata onto each stimulus
                for stimulus in self.stimuli:
                    row = manifestByFilename.loc[stimulus.name]

                    if captionColumn is not None:
                        stimulus.caption = row[captionColumn]

                    if conditionColumn is not None:
                        stimulus.condition = row[conditionColumn]

                    if conditionNumColumn is not None:
                        value = row[conditionNumColumn]
                        stimulus.conditionNum = (
                            None if pd.isna(value) else int(value)
                        )

                pglMessages.message(
                    f"Sorted and loaded stimulus metadata from "
                    f"{os.path.basename(csvFilename)}"
                )

                return

            except Exception as e:
                pglMessages.warning(
                    f"Error loading manifest {csvFilename}: {e}",
                    level=1
                )   

    def validateStimulusNum(self, stimulusNum):
        '''
        validate stimulus Num
        
        Args:
            stimulusNum (int): stimulusNum to validate

        Returns Bool for whether stimulusNum is valid or not
        '''
        if stimulusNum < 0 or stimulusNum >= self.nStimuli:
            pglMessages.warning(f"Could not preload image: stimulusNum={stimulusNum} out of range [0,{self.nStimuli})")
            return False
        return True
    
    def preload(self,stimulusNum):
        '''
        preload - to be defined by subclasses, will do any preloading necessary for the asset type
                  this default function validates stimulusNum and just returns whatever get returns
        Args:
            stimulusNum(int): Number of stimulus to preload
        '''
        if not self.validateStimulusNum(stimulusNum): return
        return self.get(stimulusNum)
        
    def get(self, stimulusNum):
        '''
        get stimulus from database. define in subclass
        
        Args:
            stimulusNum (int): number of stimulus to get
        '''
        pass
    
    def display(self, ax=None):
        ''' 
        display a representation of stimulus database. define in subclass
        '''
        pass
    
    def print(self):
        '''
        print a representation of stimulus database. 
        '''
        pglMessages.message(" ".join(f"{iImage}: {image.name}" for iImage, image in enumerate(self.stimuli)))      

############################################
# pglMovieDatabse
############################################
class pglMovieDatabase(pglStimulusDatabase):
    def __init__(self, dataPath=None, filesystem=None):
        '''
        Initialize a movie database
        '''
        # call super
        super().__init__(dataPath=dataPath, knownFileExtensions=['.mp4'], filesystem=filesystem, stimulusFileClass=pglMovieFile)

    def preload(self,stimulusNum):
        '''
        preloadImage from database. This will preload the image into memory
        '''
        pass
        
    def get(self, stimulusNum):
        '''
        Get movie-stimulus reader from database.

        Args:
            stimulusNum (int): number of stimulus to get

        Returns:
            imageio reader, or None if unavailable/nonlocal.
        '''
        if not self.validateStimulusNum(stimulusNum):
            return None

        return self.stimuli[stimulusNum].get()
    
    def display(self, stimulusNum=0, ax=None):
        self.stimuli[stimulusNum].display(ax=ax)

############################################
# pglImageDatabse
############################################
class pglImageDatabase(pglStimulusDatabase):
    def __init__(self, dataPath=None, filesystem=None):
        '''
        Initialize an image database
        '''
        # get known image extensions
        Image.init()
        imageExtensions = set(Image.registered_extensions().keys())
            
        # call super
        super().__init__(dataPath=dataPath, knownFileExtensions=imageExtensions, filesystem=filesystem, stimulusFileClass=pglImageFile)

    def preload(self,stimulusNum):
        '''
        preloadImage from database. This will preload the image into memory
        '''
        if not self.validateStimulusNum(stimulusNum): return
        return self.stimuli[stimulusNum].img
        
    def get(self,stimulusNum):
        '''
        get image stimulus from database.
        
        Args:
            stimulusNum (int): number of stimulus to get
        '''
        if not self.validateStimulusNum(stimulusNum): return
        return(self.stimuli[stimulusNum].img)
    
    def display(self, ax=None):
        ''' 
        display a representation of stimulus at top of stimulus list. define in subclass
        '''
        pass
    


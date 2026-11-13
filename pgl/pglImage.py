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
from .pglStorage import pglStorage
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
from contextlib import contextmanager
import h5py
from numbers import Integral
from datetime import datetime, timezone
from traitlets import Bool
import hashlib
import io
import json
import os
import posixpath
import tempfile
import uuid
from pathlib import Path

import numpy as np
from traitlets import Dict

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
            imageData: The image data to create the image from. Can be HxW
                grayscale, HxWx3 RGB, or HxWx4 RGBA data.

        Returns:
            An instance of pglImageInstance or None if the image could not
            be created. Use imageDisplay() or the instance's display()
            method to display it repeatedly without uploading it again.

        During command recording, the outgoing command already retains
        the uploaded pixels. Record the returned texture ID separately
        so later reconstruction can map resource identities without
        duplicating pixel data.

        Example:
            pgl.open()
            imageData = np.random.rand(480, 640, 3)
            imageInstance = pgl.imageCreate(imageData)
            if imageInstance is not None:
                pgl.imageDisplay(imageInstance)
        '''
        # Check dimensions and convert to supported RGBA image data.
        tf, imageData = self.imageValidate(imageData)
        if not tf: return None

        # Get image width and height.
        imageWidth = imageData.shape[1]
        imageHeight = imageData.shape[0]

        # Flatten and convert to float32.
        imageData = imageData.astype(np.float32).flatten()

        # Send the createTexture command and capture its recorded index.
        if not self.s.writeCommand("mglCreateTexture"): return None
        commandIndex = self.commandRecordIndex() if self.commandRecording else -1
        ackTime = self.s.readAck()

        # Send the image width, height and data. Command capture retains
        # the packed bytes through the existing socket write hooks.
        self.s.write(np.uint32(imageWidth))
        self.s.write(np.uint32(imageHeight))
        if self.verbose > 1:
            print(f"(pglImage:imageCreate) Creating image {imageWidth}x{imageHeight}")
        self.s.write(imageData)

        # Read the existing texture-specific result.
        result = self.s.read(np.float64)
        if result is None:
            pglMessages.warning("Did not receive image creation result.")
            return None
        if result < 0:
            print("(pglImage:imageCreate) Error creating image")
            return None

        # Preserve the returned identity before reading the texture count
        # or ordinary completion results. This records receipt of an ID,
        # not proof that command completion succeeded.
        imageNum = self.s.read(np.uint32)
        if imageNum is not None and commandIndex != -1:
            self.commandRecordResource(
                commandIndex,
                "texture",
                imageNum,
                source="uploadedPixels",
                widthPixels=int(imageWidth),
                heightPixels=int(imageHeight),
            )

        nImages = self.s.read(np.uint32)
        if self.verbose > 1:
            print(f"(pglImage:imageCreate) Created image {imageNum} ({nImages} total images)")
        self.s.readCommandResults(ackTime)

        # Do not construct an image reference without a returned texture ID.
        if imageNum is None:
            pglMessages.warning("Did not receive image texture ID.")
            return None

        # Create an instance of pglImageInstance.
        return pglImageInstance(imageNum, imageWidth, imageHeight, self)
    
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
        Delete an image texture from the renderer.

        Successful deletion clears imageNum so a later explicit deletion
        or pglImageInstance destructor cannot delete the same texture twice.

        Returns:
            True if already deleted or deletion reports success.
            False if the renderer is unavailable or deletion is unconfirmed.
        '''
        if not isinstance(imageInstance, pglImageInstance):
            print("(pglImage:imageDelete) imageInstance should be an instance of pglImageInstance.")
            return False

        # An explicitly deleted image may subsequently reach its destructor.
        if imageInstance.imageNum is None: return True
        if not self.isOpen(): return False

        imageNum = imageInstance.imageNum
        if self.verbose > 1:
            print(
                f"(pglImage:imageDelete) Deleting image {imageNum} "
                f"({imageInstance.width.pix}x{imageInstance.height.pix})"
            )

        # Send the existing texture deletion protocol.
        if not self.s.writeCommand("mglDeleteTexture"): return False
        self.s.write(np.uint32(imageNum))
        self.commandResults = self.s.readCommandResults()

        # Do not claim successful deletion when completion is missing.
        success = None if self.commandResults is None else self.commandResults.get("success")
        if success is None:
            pglMessages.warning("Texture deletion returned no success result.")
            return False

        success = np.asarray(success)
        if success.size != 1 or not bool(np.all(success == 1)):
            pglMessages.warning("Texture deletion did not report success=1.")
            return False

        # Prevent a later destructor from deleting this ID again.
        imageInstance.imageNum = None
        return True
    
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
            pglMessages.message(f"Created image instance with: {self.imageNum} ({self.width.pix}x{self.height.pix})")
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
    # Preserve image identity and original provenance across run export/reload.
    stimulusId = Unicode(default_value="", help="Stable stimulus identifier")
    sourceInfo = Dict(default_value={}, help="Original stimulus source provenance")
    # Complete image-level manifest row, separate from presentation-specific metadata.
    metadata = Dict(default_value={}, help="Complete stimulus manifest metadata")

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
        filesystem, filename, _ = pglStorage.validateFilesystem(filesystem=self.filesystem, dataPath=self.filename, filesystemPrefix=self.filesystemPrefix)
        with filesystem.open(filename, "rb") as h:
            img = Image.open(h)
            self._size = img.size
            self._mode = img.mode
            self._format = img.format

    def _loadImage(self):
        pglMessages.message(f"Loading image: {self.filesystemPrefix}/{self.filename}")
        filesystem, filename, _ = pglStorage.validateFilesystem(filesystem=self.filesystem, dataPath=self.filename, filesystemPrefix=self.filesystemPrefix)
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
# pglHdf5Source
################################################
class pglH5Source:

    def __init__(self, filename, filesystem=None, filesystemPrefix=""):
        '''
        Own a shared filesystem stream and HDF5 handle.

        Opening is lazy. After close(), a subsequent open() reopens the source.
        Live source objects must not be shared across worker processes.
        '''
        self.filename = filename
        self.filesystem = filesystem
        self.filesystemPrefix = filesystemPrefix
        self._fileHandle = None
        self._hdfFile = None

    @property
    def isOpen(self):
        '''Whether this source currently owns an open HDF5 handle.'''
        return self._hdfFile is not None and bool(self._hdfFile.id.valid)

    def open(self):
        '''
        Return the shared HDF5 handle, opening it only when necessary.

        Callers borrow this handle; only the source owner should close it.
        '''
        if self.isOpen: return self._hdfFile

        # Clear any stale resources before opening a new stream.
        self.close()
        filesystem, filename, _ = pglStorage.validateFilesystem(
            filesystem=self.filesystem,
            dataPath=self.filename,
            filesystemPrefix=self.filesystemPrefix
        )
        if filesystem is None: raise ValueError(f"Could not resolve HDF5 file: {self.filename}")

        fileHandle = filesystem.open(filename, "rb")
        try:
            hdfFile = h5py.File(fileHandle, "r")
        except Exception:
            # A failed HDF5 open must not leave the filesystem stream open.
            fileHandle.close()
            raise

        self._fileHandle = fileHandle
        self._hdfFile = hdfFile
        return self._hdfFile

    def close(self):
        '''
        Release both handles in the required order.

        Safe to call repeatedly. Already loaded PIL images remain usable.
        '''
        hdfFile, fileHandle = self._hdfFile, self._fileHandle
        self._hdfFile = None
        self._fileHandle = None

        try:
            if hdfFile is not None: hdfFile.close()
        finally:
            if fileHandle is not None: fileHandle.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exceptionType, exceptionValue, traceback):
        self.close()

################################################
# pglImageFileH5
################################################
class pglImageFileH5(pglImageFile):

    # Identify one image inside an HDF5 dataset.
    datasetPath = Unicode(default_value="", help="Dataset path inside the HDF5 file")
    imageIndex = Int(default_value=0, min=0, help="Zero-based image index inside the dataset")
    imageAxis = Int(default_value=0, min=0, help="Dataset axis that indexes images")
    # Optional shared runtime source; retain existing locator fields for serialization.
    source = Instance(pglH5Source, allow_none=True, default_value=None, serialize=False, visible=False, help="Shared HDF5 source")
    
    @contextmanager
    def _openDataset(self):
        '''
        Borrow a dataset from a shared source, or open a temporary source.

        Shared handles remain open between image reads. Standalone images
        retain per-operation cleanup. Dataset handles must not escape this
        context; read pixels into memory before returning.
        '''
        if not self.datasetPath: raise ValueError("datasetPath must identify an HDF5 image dataset")

        # A standalone image owns its temporary source; shared images do not.
        ownsSource = self.source is None
        source = self.source if self.source is not None else pglH5Source(
            filename=self.filename,
            filesystem=self.filesystem,
            filesystemPrefix=self.filesystemPrefix
        )

        try:
            hdfFile = source.open()
            if self.datasetPath not in hdfFile:
                raise KeyError(f"Dataset '{self.datasetPath}' not found in {source.filename}")

            dataset = hdfFile[self.datasetPath]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"'{self.datasetPath}' is not an HDF5 dataset")

            yield dataset
        finally:
            # Only the owner closes a shared source, not individual images.
            if ownsSource: source.close()

    def _getImageMetadata(self, dataset):
        '''
        Validate the default layout and return PIL-compatible size and mode.

        Removing imageAxis must leave (height, width), (height, width, 3),
        or (height, width, 4). Subclasses can override this layout hook.
        '''
        if self.imageAxis >= dataset.ndim:
            raise ValueError(f"imageAxis={self.imageAxis} is invalid for dataset shape {dataset.shape}")

        nImages = dataset.shape[self.imageAxis]
        if not 0 <= self.imageIndex < nImages:
            raise IndexError(f"imageIndex={self.imageIndex} out of range [0, {nImages})")

        # Inspect shape and dtype only; do not read image pixels.
        imageShape = dataset.shape[:self.imageAxis] + dataset.shape[self.imageAxis + 1:]
        if dataset.dtype.name != "uint8":
            raise TypeError(f"Default HDF5 image loading requires uint8 data, got {dataset.dtype}")

        if len(imageShape) == 2:
            mode = "L"
        elif len(imageShape) == 3 and imageShape[2] in (3, 4):
            mode = "RGB" if imageShape[2] == 3 else "RGBA"
        else:
            raise ValueError(f"Unsupported image shape {imageShape}; expected (H, W), (H, W, 3), or (H, W, 4)")

        if any(dimension == 0 for dimension in imageShape):
            raise ValueError(f"Image dimensions must be nonzero, got {imageShape}")

        # PIL reports size as (width, height), not array order.
        return (imageShape[1], imageShape[0]), mode

    def _readImage(self, dataset):
        '''
        Read only the selected image.

        Subclasses can override this hook for axis reordering or decoding.
        '''
        selection = [slice(None)] * dataset.ndim
        selection[self.imageAxis] = self.imageIndex
        return dataset[tuple(selection)]

    def _loadMetadata(self):
        '''Cache image metadata without loading image pixels.'''
        pglMessages.message(f"Loading HDF5 image metadata: {self.filename}:{self.datasetPath}[{self.imageIndex}]")

        with self._openDataset() as dataset:
            size, mode = self._getImageMetadata(dataset)

        self._size = size
        self._mode = mode
        self._format = "HDF5"

    def _loadImage(self):
        '''Read one image and cache it as a PIL image, matching pglImageFile.'''
        pglMessages.message(f"Loading HDF5 image: {self.filename}:{self.datasetPath}[{self.imageIndex}]")

        with self._openDataset() as dataset:
            size, mode = self._getImageMetadata(dataset)
            imageData = self._readImage(dataset)

        # imageData is now in memory and independent of the closed HDF5 file.
        img = Image.fromarray(imageData)
        if img.size != size or img.mode != mode:
            raise ValueError(f"Decoded image does not match expected size={size}, mode={mode}")

        self._img = img
        self._size = img.size
        self._mode = img.mode
        self._format = "HDF5"

    def __repr__(self):
        return f"{os.path.basename(self.filename)}:{self.datasetPath}[{self.imageIndex}] {self.size[0]}x{self.size[1]} {self.mode}"


################################################
# pglImageFileNSD-mini
################################################
class pglImageFileNSDMini(pglImageFileH5):

    # NSDmini stores images in this dataset by default.
    datasetPath = Unicode(default_value="imgBrick", help="NSDmini image dataset path")

    def _getImageMetadata(self, dataset):
        '''
        Validate NSDmini's (image, height, width, RGB channel) layout.

        Do not hardcode image count or spatial dimensions.
        '''
        if self.imageAxis != 0:
            raise ValueError("NSDmini images must use imageAxis=0")

        if dataset.ndim != 4 or dataset.shape[-1] != 3:
            raise ValueError(f"Expected NSDmini dataset shape (N, H, W, 3), got {dataset.shape}")

        return super()._getImageMetadata(dataset)
    
    @classmethod
    def discoverImages(cls, source, datasetPath=None):
        '''
        Create unloaded image objects sharing one HDF5 source.

        The selected class supplies dataset defaults and layout validation.
        The caller owns the source and is responsible for closing it.
        '''
        imageSettings = {
            "filename": source.filename,
            "filesystem": source.filesystem,
            "filesystemPrefix": source.filesystemPrefix,
            "source": source
        }
        if datasetPath is not None: imageSettings["datasetPath"] = datasetPath

        # Use a temporary image object to apply this class's format knowledge.
        probe = cls(imageIndex=0, **imageSettings)
        with probe._openDataset() as dataset:
            # Validate the layout and first image without loading pixels.
            # Empty datasets are rejected because they contain no valid first image.
            probe._getImageMetadata(dataset)
            nImages = dataset.shape[probe.imageAxis]

        # Stable names and source indices are independent of later trial ordering.
        namePrefix = probe.datasetPath.strip("/").replace("/", "_")
        indexWidth = max(4, len(str(nImages - 1)))

        return [
            cls(
                name=f"{namePrefix}_{imageIndex:0{indexWidth}d}",
                imageIndex=imageIndex,
                **imageSettings
            )
            for imageIndex in range(nImages)
        ]


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
        filesystem, filename, _ = pglStorage.validateFilesystem(filesystem=self.filesystem, dataPath=self.filename, filesystemPrefix=self.filesystemPrefix)

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
# Trial stimulus conventions
################################################
blankStimulusNum = -1


################################################
# pglStimulusTrial
################################################
class pglStimulusTrial:

    def __init__(self, trialNum, stimulusNum, stimulus=None, metadata=None):
        '''
        Represent one planned presentation.

        Trial numbers are zero-based. The initial trial ID equals the trial
        number; it identifies this presentation, not the underlying image.
        '''
        self.trialNum = trialNum
        self.trialId = trialNum
        self.stimulusNum = stimulusNum
        self.stimulus = stimulus
        self.metadata = {} if metadata is None else dict(metadata)
        self.status = "pending"
        self.completedAt = None

    @property
    def isBlank(self):
        return self.stimulusNum == blankStimulusNum

    @property
    def img(self):
        '''
        Retrieve an image lazily through the referenced image stimulus.

        Blank trials have no image. For other stimulus types, use stimulus
        and its corresponding loading interface instead.
        '''
        return None if self.isBlank else self.stimulus.img
    
################################################
# pglStimulusDaabase
################################################
class pglStimulusDatabase(pglTraitSettings):
    
    # filesystem, name and prefix for where the images were loaded from
    filesystem = Instance(AbstractFileSystem, allow_none=True, serialize=False, help="filesystem for serialization")
    dataPath = Unicode(allow_none=True, default_value="", help="Full path to stimuli", visible=False)
    filesystemPrefix = Unicode(allow_none=True, default_value="", help="Prefix like ssh:// used for accessing filesystem", visible=False)
    stimuli = List(Instance(pglStimulusFile), default_value=[], settingsListKey="name", traitDisplayName="Choose stimulus", hasPlotButton=True, buttonFunction="display", help="List of stimuli in database")
    # Trial objects are runtime state; explicit run export/reload will handle persistence.
    trials = List(Instance(pglStimulusTrial), default_value=[], serialize=False, visible=False, help="Ordered trial presentations")
    _runStarted = Bool(default_value=False, serialize=False, visible=False, help="Whether run tracking has started")
    nStimuli = Int(0, help="Number of stimuli", enable=False)
    
    def __init__(self, dataPath=None, knownFileExtensions=None, filesystem=None, stimulusFileClass=pglStimulusFile):
        '''
        Initialize by pointing to a directory where images live
        '''
        if dataPath:
    
            # parse filesystem
            self.filesystem, self.dataPath, self.filesystemPrefix = pglStorage.validateFilesystem(filesystem, dataPath)
            if not self.filesystem:
                pglMessages.warning(f"Could not resolve path to images: {self.dataPaths}")
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
            pglMessages.message(f"Found {self.nStimuli} stimulus files in {dataPath}")

    def useManifest(self, filenameColumn="filename", indexColumn="index", captionColumn=None, conditionColumn=None, conditionNumColumn=None):
        '''
        Load image ordering and complete image metadata from a CSV manifest.

        Every CSV column is retained in stimulus.metadata, including filename
        and index. Optional columns also populate the existing convenience
        fields: caption, condition, and conditionNum.

        Call before setTrialOrder(), because this method sorts the inventory.
        Invalid input leaves the existing inventory and metadata unchanged.
        '''
        # Reordering would invalidate the numeric references already assigned to trials.
        if self.trials:
            pglMessages.warning("Load the image manifest before setting trial order; existing trials were left unchanged")
            return

        if self.filesystem is None or not self.dataPath:
            pglMessages.warning("Cannot load an image manifest without a filesystem and dataPath")
            return

        # This method handles directory-backed CSV manifests, not container trial order.
        if not self.filesystem.isdir(self.dataPath):
            pglMessages.warning("useManifest() requires an image directory containing one CSV manifest")
            return

        csvFilenames = self.filesystem.glob(os.path.join(self.dataPath, "*.csv"))
        if len(csvFilenames) != 1:
            pglMessages.warning(
                f"Expected exactly one CSV manifest in {self.dataPath}; found {len(csvFilenames)}"
            )
            return

        csvFilename = csvFilenames[0]

        try:
            # Preserve filename strings, including numeric-looking names.
            with self.filesystem.open(csvFilename, "rb") as fileHandle:
                dataFrame = pd.read_csv(fileHandle, dtype={filenameColumn: str})

            requiredColumns = [filenameColumn, indexColumn]
            requiredColumns.extend(
                column for column in (captionColumn, conditionColumn, conditionNumColumn)
                if column is not None
            )
            missingColumns = [column for column in requiredColumns if column not in dataFrame.columns]
            if missingColumns:
                raise ValueError(f"Missing manifest columns: {missingColumns}")

            filenames = [stimulus.name for stimulus in self.stimuli]
            manifestFilenames = dataFrame[filenameColumn]

            # A unique manifest row must correspond to each unique stimulus name.
            if len(set(filenames)) != len(filenames):
                raise ValueError("Loaded stimulus names are not unique")
            if manifestFilenames.isna().any() or manifestFilenames.duplicated().any():
                raise ValueError("Manifest filenames must be present and unique")
            if set(manifestFilenames) != set(filenames):
                raise ValueError("Manifest must contain exactly the loaded stimulus filenames")
            if dataFrame[indexColumn].isna().any():
                raise ValueError("Manifest ordering indices must not be missing")

            # Convert each parsed row to plain scalar values suitable for JSON.
            # to_dict('records') avoids the cross-column coercion of iterrows().
            metadataByFilename = {}
            for row in dataFrame.to_dict(orient="records"):
                metadata = {}
                for key, value in row.items():
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, np.generic):
                        value = value.item()
                    metadata[key] = value

                metadataByFilename[metadata[filenameColumn]] = metadata

            # Prepare convenience fields and validate them before changing objects.
            preparedUpdates = []
            for stimulus in self.stimuli:
                metadata = metadataByFilename[stimulus.name]
                convenienceFields = {}

                if captionColumn is not None:
                    value = metadata[captionColumn]
                    convenienceFields["caption"] = "" if value is None else str(value)

                if conditionColumn is not None:
                    value = metadata[conditionColumn]
                    convenienceFields["condition"] = "" if value is None else str(value)

                if conditionNumColumn is not None:
                    value = metadata[conditionNumColumn]
                    if value is None:
                        convenienceFields["conditionNum"] = None
                    else:
                        conditionNum = int(value)
                        if not isinstance(value, str) and conditionNum != value:
                            raise ValueError(f"Noninteger condition number for {stimulus.name}: {value}")
                        convenienceFields["conditionNum"] = conditionNum

                preparedUpdates.append((stimulus, metadata, convenienceFields))

            # Calculate ordering before committing metadata or inventory changes.
            sortedStimuli = sorted(
                self.stimuli,
                key=lambda stimulus: metadataByFilename[stimulus.name][indexColumn]
            )

        except Exception as error:
            pglMessages.warning(f"Could not load manifest {csvFilename}: {error}", level=1)
            return

        # Commit only after parsing and validation have succeeded.
        for stimulus, metadata, convenienceFields in preparedUpdates:
            stimulus.metadata = metadata
            for fieldName, value in convenienceFields.items():
                setattr(stimulus, fieldName, value)

        self.stimuli = sortedStimuli
        self.nStimuli = len(sortedStimuli)
        self.manifest = dataFrame

        pglMessages.message(
            f"Sorted {self.nStimuli} stimuli and retained all {len(dataFrame.columns)} "
            f"manifest columns from {os.path.basename(csvFilename)}"
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
    
    def preload(self,stimulusNum=None):
        '''
        preload - to be defined by subclasses, will do any preloading necessary for the asset type
                  this default function validates stimulusNum and just returns whatever get returns
        Args:
            stimulusNum(int): Number of stimulus to preload
        '''
        if stimulusNum is None:
            for stimulusNum in range(self.nStimuli):
                self.preload(stimulusNum)
            return
        if not self.validateStimulusNum(stimulusNum): return
        return self.get(stimulusNum)
        
    def get(self, stimulusNum):
        '''
        get stimulus from database. define in subclass
        
        Args:
            stimulusNum (int): number of stimulus to get
        '''
        pass
    
    @property
    def nTrials(self):
        return len(self.trials)

    def setTrialOrder(self, imageOrder, metadata=None):
        '''
        Set a planned trial sequence without rearranging the stimulus inventory.

        Args:
            imageOrder: Iterable of zero-based stimulus numbers; -1 means blank.
            metadata: Optional iterable of dictionaries, one per trial.

        Calling this method replaces the previous trial sequence and resets
        trial status. Invalid input leaves the existing sequence unchanged.
        '''
        # Do not silently discard an unfinished run.
        if self._runStarted and any(trial.status == "pending" for trial in self.trials):
            pglMessages.warning("Cannot replace trial order during an unfinished run")
            return
        
        imageOrder = list(imageOrder)

        # Each trial gets its own metadata dictionary, including repeated images.
        if metadata is None:
            trialMetadata = [{} for _ in imageOrder]
        else:
            trialMetadata = list(metadata)
            if len(trialMetadata) != len(imageOrder):
                raise ValueError("metadata must contain one dictionary per trial")

        # Build locally so validation failures do not partially replace trials.
        trials = []
        for trialNum, (stimulusNum, trialInfo) in enumerate(zip(imageOrder, trialMetadata)):
            if isinstance(stimulusNum, bool) or not isinstance(stimulusNum, Integral):
                raise TypeError(f"Trial {trialNum}: stimulusNum must be an integer, got {stimulusNum!r}")

            stimulusNum = int(stimulusNum)
            if stimulusNum != blankStimulusNum and not 0 <= stimulusNum < len(self.stimuli):
                raise ValueError(
                    f"Trial {trialNum}: stimulusNum={stimulusNum} must be "
                    f"{blankStimulusNum} or in [0, {len(self.stimuli)})"
                )

            if not isinstance(trialInfo, dict):
                raise TypeError(f"Trial {trialNum}: metadata must be a dictionary")

            # Handle the blank sentinel before indexing: -1 must never select an image.
            stimulus = None if stimulusNum == blankStimulusNum else self.stimuli[stimulusNum]
            trials.append(pglStimulusTrial(
                trialNum=trialNum,
                stimulusNum=stimulusNum,
                stimulus=stimulus,
                metadata=trialInfo
            ))

        self.trials = trials
        self._runStarted = False

    def getTrial(self, trialNum):
        '''
        Return a trial by its zero-based position without advancing progress.

        Retrieving the trial does not load pixels; accessing trial.img does.
        '''
        if isinstance(trialNum, bool) or not isinstance(trialNum, Integral):
            raise TypeError("trialNum must be an integer")

        trialNum = int(trialNum)
        if not 0 <= trialNum < self.nTrials:
            raise IndexError(f"trialNum={trialNum} out of range [0, {self.nTrials})")

        return self.trials[trialNum]
    
    def startRun(self, reset=False):
        '''
        Start or restart tracking the configured trial sequence.

        Restarting clears completion status and timestamps. Warn unless
        reset=True explicitly acknowledges that progress will be discarded.
        '''
        if not self.trials:
            raise RuntimeError("Set a nonempty trial order before starting a run")

        if self._runStarted and not reset:
            pglMessages.warning("Restarting trial sequence; previous completion status and timestamps will be cleared")

        for trial in self.trials:
            trial.status = "pending"
            trial.completedAt = None

        self._runStarted = True
        
    def nextTrial(self):
        '''
        Return the first pending trial without advancing or loading its image.

        Return None when the run has no pending trials.
        '''
        if not self._runStarted:
            raise RuntimeError("Call startRun() before requesting the next trial")

        return next((trial for trial in self.trials if trial.status == "pending"), None)

    def completeTrial(self, trialId):
        '''
        Complete the current pending trial and record its UTC completion time.

        Reject duplicate, out-of-order, and invalid completion requests.
        Return the completed trial without loading its image.
        '''
        if not self._runStarted:
            raise RuntimeError("Call startRun() before completing a trial")
        if isinstance(trialId, bool) or not isinstance(trialId, Integral):
            raise TypeError("trialId must be an integer")

        trialId = int(trialId)
        trial = next((trial for trial in self.trials if trial.trialId == trialId), None)
        if trial is None:
            raise ValueError(f"Unknown trialId={trialId}")
        if trial.status != "pending":
            raise RuntimeError(f"Trial {trialId} is already {trial.status}")

        pendingTrial = self.nextTrial()
        if trial is not pendingTrial:
            raise RuntimeError(f"Cannot complete trial {trialId}; next pending trial is {pendingTrial.trialId}")

        # Advancing is implicit: nextTrial() now finds the next pending record.
        trial.completedAt = datetime.now(timezone.utc).isoformat()
        trial.status = "completed"
        return trial    
    
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
# pglImageDatabase
############################################
class pglImageDatabase(pglStimulusDatabase):

    # Runtime resource only; open handles must never be serialized.
    _h5Source = Instance(pglH5Source, allow_none=True, default_value=None, serialize=False, visible=False, help="Shared HDF5 source")

    def __init__(self, dataPath=None, filesystem=None, stimulusFileClass=None, datasetPath=None):
        '''
        Initialize from an image directory or an HDF5 container.

        Directories retain the existing behavior. For HDF5 containers, select
        an HDF5 image class; format-specific subclasses supply sensible defaults.
        '''
        if stimulusFileClass is None: stimulusFileClass = pglImageFile
        if not issubclass(stimulusFileClass, pglImageFile):
            raise TypeError("stimulusFileClass must inherit from pglImageFile")

        # Preserve construction of an empty database.
        if not dataPath:
            super().__init__()
            return

        isH5Class = issubclass(stimulusFileClass, pglImageFileH5)

        if not isH5Class:
            if datasetPath is not None:
                raise ValueError("datasetPath requires an HDF5 image class")

            # Preserve ordinary image-directory discovery and alphabetical ordering.
            Image.init()
            imageExtensions = set(Image.registered_extensions().keys())
            super().__init__(
                dataPath=dataPath,
                knownFileExtensions=imageExtensions,
                filesystem=filesystem,
                stimulusFileClass=stimulusFileClass
            )
            return

        # Initialize without invoking directory discovery for a container file.
        super().__init__()
        self.filesystem, self.dataPath, self.filesystemPrefix = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=dataPath
        )
        if self.filesystem is None:
            raise ValueError(f"Could not resolve HDF5 path: {dataPath}")
        if not self.filesystem.isfile(self.dataPath):
            raise ValueError(f"Expected an HDF5 container file: {dataPath}")

        source = pglH5Source(
            filename=self.dataPath,
            filesystem=self.filesystem,
            filesystemPrefix=self.filesystemPrefix
        )

        try:
            # Discovery reads metadata only; all images share this source.
            stimuli = stimulusFileClass.discoverImages(source, datasetPath=datasetPath)
            self.stimuli = stimuli
            self.nStimuli = len(stimuli)
            self._h5Source = source
        except Exception:
            # Constructor failures must not leak the shared file handles.
            source.close()
            raise

        # Keep dataset index order rather than applying directory-style sorting.
        pglMessages.message(f"Found {self.nStimuli} images in {dataPath}")

    def preload(self, stimulusNum=None):
        '''
        Preload one image, or all images when stimulusNum is omitted.
        '''
        if stimulusNum is None:
            for stimulusNum in range(self.nStimuli): self.preload(stimulusNum)
            return

        if not self.validateStimulusNum(stimulusNum): return
        return self.stimuli[stimulusNum].img

    def get(self, stimulusNum):
        '''
        Get a PIL image by its zero-based database stimulus number.
        '''
        if not self.validateStimulusNum(stimulusNum): return
        return self.stimuli[stimulusNum].img

    def close(self):
        '''
        Close the shared source without clearing cached images.

        Accessing an uncached image later will reopen the source.
        Directory-backed databases have no persistent source to close.
        '''
        if self._h5Source is not None: self._h5Source.close()

    def __enter__(self):
        return self

    def __exit__(self, exceptionType, exceptionValue, traceback):
        self.close()
        
    @staticmethod
    def _manifestJsonDefault(value):
        '''
        Convert NumPy values commonly found in trial metadata to JSON values.

        Unsupported objects raise rather than being silently converted to text.
        '''
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(f"Cannot save {type(value).__name__} in stimulus metadata")

    @staticmethod
    def _writeRunFile(filename, contents):
        '''
        Write a local file through a temporary file in the same directory.

        Replacement prevents readers from seeing a partially written file.
        This is not a transaction covering the entire stimulus bundle.
        '''
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        temporaryPath = None

        try:
            with tempfile.NamedTemporaryFile(dir=filename.parent, prefix=".pgl-", delete=False) as fileHandle:
                temporaryPath = Path(fileHandle.name)
                fileHandle.write(contents)
                fileHandle.flush()
                os.fsync(fileHandle.fileno())

            os.replace(temporaryPath, filename)
        finally:
            if temporaryPath is not None and temporaryPath.exists(): temporaryPath.unlink()

    def saveRun(self, taskDirectory):
        '''
        Export a self-contained stimulus bundle into a local task directory.

        Save all referenced images, including pending trials. Repeated trials
        share one stimulus entry and image file. Blank trials have no image.

        Use readable filenames and retain SHA-256 hashes in the manifest.
        Publish the manifest last; never overwrite an existing image with
        different contents.

        Return the manifest path. Errors propagate for experiment recovery.
        '''
        taskDirectory = Path(taskDirectory).expanduser()
        bundlePath = taskDirectory / "stimuli"
        imagePath = bundlePath / "images"

        stimulusRecords = []
        trialRecords = []
        referencedStimuli = {}

        # Validate trial references before writing anything.
        for trialNum, trial in enumerate(self.trials):
            if trial.trialNum != trialNum:
                raise ValueError("Trial numbers must match their positions in the sequence")
            if trial.status not in ("pending", "completed"):
                raise ValueError(f"Unsupported status for trial {trialNum}: {trial.status}")

            if trial.isBlank:
                if trial.stimulus is not None:
                    raise ValueError(f"Blank trial {trialNum} unexpectedly references a stimulus")
            else:
                stimulusNum = trial.stimulusNum
                if not 0 <= stimulusNum < len(self.stimuli):
                    raise ValueError(f"Trial {trialNum} references an invalid stimulus")
                if trial.stimulus is not self.stimuli[stimulusNum]:
                    raise ValueError("Stimulus inventory changed after trial ordering was configured")
                referencedStimuli[stimulusNum] = trial.stimulus

            trialRecords.append({
                "trialNum": trial.trialNum,
                "trialId": trial.trialId,
                "stimulusNum": trial.stimulusNum,
                "status": trial.status,
                "completedAt": trial.completedAt,
                "metadata": trial.metadata
            })

        trialIds = [record["trialId"] for record in trialRecords]
        if len(set(trialIds)) != len(trialIds):
            raise ValueError("Trial IDs must be unique")

        # Prepare one inventory entry per referenced stimulus.
        for stimulusNum, stimulus in sorted(referencedStimuli.items()):
            if not stimulus.stimulusId: stimulus.stimulusId = uuid.uuid4().hex

            sourceInfo = dict(stimulus.sourceInfo)
            if not sourceInfo:
                sourceInfo = {
                    "filename": stimulus.filename,
                    "filesystemPrefix": stimulus.filesystemPrefix,
                    "originalStimulusNum": stimulusNum
                }
                if isinstance(stimulus, pglImageFileH5):
                    sourceInfo.update({
                        "datasetPath": stimulus.datasetPath,
                        "imageIndex": stimulus.imageIndex,
                        "imageAxis": stimulus.imageAxis
                    })

            stimulusRecords.append({
                "stimulusNum": stimulusNum,
                "stimulusId": stimulus.stimulusId,
                "name": stimulus.name,
                "metadata": dict(stimulus.metadata),
                "caption": stimulus.caption,
                "condition": stimulus.condition,
                "conditionNum": stimulus.conditionNum,
                "sourceInfo": sourceInfo
            })

        stimulusIds = [record["stimulusId"] for record in stimulusRecords]
        if len(set(stimulusIds)) != len(stimulusIds):
            raise ValueError("Referenced stimulus inventory contains duplicate stimulus IDs")

        manifest = {
            "formatName": "pglImageDatabaseRun",
            "formatVersion": 2,
            "runStarted": self._runStarted,
            "stimuli": stimulusRecords,
            "trials": trialRecords
        }

        # Validate metadata before reading pixels or writing image files.
        json.dumps(manifest, default=self._manifestJsonDefault, allow_nan=False)
        imagePath.mkdir(parents=True, exist_ok=True)

        # Track names case-insensitively for portable exports.
        existingNames = {filename.name.casefold(): filename for filename in imagePath.iterdir()}

        for record in stimulusRecords:
            stimulusNum = record["stimulusNum"]
            stimulus = referencedStimuli[stimulusNum]
            img = stimulus.img

            if img.mode not in ("L", "RGB", "RGBA"):
                raise ValueError(f"Cannot export image mode {img.mode!r} losslessly with this exporter")

            with io.BytesIO() as buffer:
                img.save(buffer, format="PNG")
                imageBytes = buffer.getvalue()

            imageHash = hashlib.sha256(imageBytes).hexdigest()
            preferredName = self._getRunImageFilename(stimulus, stimulusNum)
            imageStem = posixpath.splitext(preferredName)[0]
            candidateName = preferredName
            suffix = 2

            # Reuse matching files; preserve differing files with a numeric suffix.
            while True:
                existingPath = existingNames.get(candidateName.casefold())
                if existingPath is None:
                    filename = imagePath / candidateName
                    self._writeRunFile(filename, imageBytes)
                    existingNames[candidateName.casefold()] = filename
                    break

                if existingPath.is_file() and hashlib.sha256(existingPath.read_bytes()).hexdigest() == imageHash:
                    filename = existingPath
                    break

                candidateName = f"{imageStem}_{suffix}.png"
                suffix += 1

            record.update({
                "filename": f"images/{filename.name}",
                "sha256": imageHash,
                "size": list(img.size),
                "mode": img.mode
            })

        # Publish only after all referenced images are available.
        manifestBytes = (
            json.dumps(manifest, indent=2, default=self._manifestJsonDefault, allow_nan=False) + "\n"
        ).encode("utf-8")
        manifestPath = bundlePath / "manifest.json"
        self._writeRunFile(manifestPath, manifestBytes)

        pglMessages.message(
            f"Saved {len(trialRecords)} trials and {len(stimulusRecords)} referenced stimuli to {bundlePath}"
        )
        return manifestPath

    @classmethod
    def fromRun(cls, taskDirectory, filesystem=None, filesystemPrefix=None):
        '''
        Load a saved stimulus bundle without accessing the original image source.

        Read the manifest and check image-file existence, but leave image pixels
        unloaded. Exported stimulus numbers are mapped to a compact local inventory.
        '''
        filesystem, taskDirectory, filesystemPrefix = pglStorage.validateFilesystem(
            filesystem=filesystem,
            dataPath=str(taskDirectory),
            filesystemPrefix=filesystemPrefix
        )
        if filesystem is None:
            raise ValueError(f"Could not resolve task directory: {taskDirectory}")

        bundlePath = posixpath.join(taskDirectory, "stimuli")
        manifestPath = posixpath.join(bundlePath, "manifest.json")
        with filesystem.open(manifestPath, "rt", encoding="utf-8") as fileHandle:
            manifest = json.load(fileHandle)

        if not isinstance(manifest, dict):
            raise ValueError("Stimulus manifest must be a JSON object")
        if manifest.get("formatName") != "pglImageDatabaseRun" or manifest.get("formatVersion") not in (1, 2):
            raise ValueError("Unsupported stimulus manifest format or version")

        stimulusRecords = manifest.get("stimuli")
        trialRecords = manifest.get("trials")
        runStarted = manifest.get("runStarted")
        if not isinstance(stimulusRecords, list) or not isinstance(trialRecords, list):
            raise ValueError("Stimulus manifest must contain stimulus and trial lists")
        if not isinstance(runStarted, bool):
            raise ValueError("Manifest runStarted must be Boolean")

        database = cls()
        database.filesystem = filesystem
        database.dataPath = bundlePath
        database.filesystemPrefix = filesystemPrefix

        stimuli = []
        localNumBySavedNum = {}
        stimulusIds = set()

        for record in stimulusRecords:
            savedNum = record["stimulusNum"]
            if type(savedNum) is not int or savedNum < 0 or savedNum in localNumBySavedNum:
                raise ValueError(f"Invalid or duplicate saved stimulus number: {savedNum}")

            stimulusId = record["stimulusId"]
            if not isinstance(stimulusId, str) or not stimulusId or stimulusId in stimulusIds:
                raise ValueError("Stimulus IDs must be unique nonempty strings")
            stimulusIds.add(stimulusId)

            # Both versions retain hashes; version 2 permits readable filenames.
            imageHash = record["sha256"]
            if not isinstance(imageHash, str) or len(imageHash) != 64 or any(c not in "0123456789abcdef" for c in imageHash):
                raise ValueError("Invalid image SHA-256")

            relativePath = record["filename"]
            if not isinstance(relativePath, str):
                raise ValueError("Exported image path must be a string")

            # Only allow a direct PNG child of images/, never arbitrary paths.
            pathParts = relativePath.split("/")
            if (
                len(pathParts) != 2
                or pathParts[0] != "images"
                or not pathParts[1].lower().endswith(".png")
                or "\\" in relativePath
                or "\x00" in relativePath
                or ":" in pathParts[1]
            ):
                raise ValueError("Invalid exported image path")

            if manifest["formatVersion"] == 1 and relativePath != f"images/{imageHash}.png":
                raise ValueError("Invalid version 1 exported image path")

            filename = posixpath.join(bundlePath, relativePath)
            if not filesystem.isfile(filename):
                raise FileNotFoundError(f"Exported image is missing: {filename}")

            localNumBySavedNum[savedNum] = len(stimuli)
            stimuli.append(pglImageFile(
                name=record["name"],
                metadata=record.get("metadata", {}),
                filename=filename,
                filesystem=filesystem,
                filesystemPrefix=filesystemPrefix,
                stimulusId=stimulusId,
                sourceInfo=record["sourceInfo"],
                caption=record["caption"],
                condition=record["condition"],
                conditionNum=record["conditionNum"]
            ))

        database.stimuli = stimuli
        database.nStimuli = len(stimuli)

        imageOrder = []
        metadata = []
        trialIds = set()

        for trialNum, record in enumerate(trialRecords):
            if type(record["trialNum"]) is not int or record["trialNum"] != trialNum:
                raise ValueError("Saved trial numbers must be consecutive and zero-based")

            trialId = record["trialId"]
            if type(trialId) is not int or trialId < 0 or trialId in trialIds:
                raise ValueError("Trial IDs must be unique nonnegative integers")
            trialIds.add(trialId)

            savedNum = record["stimulusNum"]
            if type(savedNum) is not int:
                raise ValueError("Saved stimulus numbers must be integers")
            if savedNum == blankStimulusNum:
                imageOrder.append(blankStimulusNum)
            elif savedNum in localNumBySavedNum:
                imageOrder.append(localNumBySavedNum[savedNum])
            else:
                raise ValueError(f"Trial {trialNum} references an unknown stimulus")

            status = record["status"]
            completedAt = record["completedAt"]
            if status not in ("pending", "completed"):
                raise ValueError(f"Unsupported trial status: {status}")
            if status == "pending" and completedAt is not None:
                raise ValueError("Pending trials must not have completion timestamps")
            if status == "completed":
                if not runStarted or not isinstance(completedAt, str):
                    raise ValueError("Completed trials require a started run and completion timestamp")
                datetime.fromisoformat(completedAt)

            metadata.append(record["metadata"])

        # Restore trial relationships, then restore progress without calling startRun().
        database.setTrialOrder(imageOrder, metadata=metadata)
        for trial, record in zip(database.trials, trialRecords):
            trial.trialId = record["trialId"]
            trial.status = record["status"]
            trial.completedAt = record["completedAt"]

        database._runStarted = runStarted
        return database
    
    @staticmethod
    def _getRunImageFilename(stimulus, stimulusNum):
        '''
        Choose a readable PNG filename.

        Keep ordinary image filenames where possible. Container images use
        their original image index, displayed as a one-based number.
        '''
        sourceInfo = stimulus.sourceInfo
        isContainerImage = isinstance(stimulus, pglImageFileH5) or "datasetPath" in sourceInfo

        if isContainerImage:
            imageIndex = sourceInfo.get("imageIndex", getattr(stimulus, "imageIndex", stimulusNum))
            return f"image{imageIndex + 1:04d}.png"

        # For reloaded exports, use original provenance rather than the export path.
        originalFilename = sourceInfo.get("filename") or stimulus.filename
        originalName = str(originalFilename).replace("\\", "/").rsplit("/", 1)[-1]
        imageStem = posixpath.splitext(originalName)[0]

        # Replace characters that are unsuitable for portable filenames.
        imageStem = "".join(
            "_" if ord(character) < 32 or character in '<>:"/\\|?*' else character
            for character in imageStem
        ).strip(" .")

        if not imageStem: imageStem = f"image{stimulusNum + 1:04d}"

        # Avoid reserved Windows device filenames.
        reservedNames = {"CON", "PRN", "AUX", "NUL"}
        reservedNames.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
        if imageStem.split(".")[0].upper() in reservedNames: imageStem = f"image_{imageStem}"

        return f"{imageStem}.png"@staticmethod
    
    @staticmethod
    def _getRunImageFilename(stimulus, stimulusNum):
        '''
        Choose a readable PNG filename.

        Keep ordinary image filenames where possible. Container images use
        their original image index, displayed as a one-based number.
        '''
        sourceInfo = stimulus.sourceInfo
        isContainerImage = isinstance(stimulus, pglImageFileH5) or "datasetPath" in sourceInfo

        if isContainerImage:
            imageIndex = sourceInfo.get("imageIndex", getattr(stimulus, "imageIndex", stimulusNum))
            return f"image{imageIndex + 1:04d}.png"

        # For reloaded exports, use original provenance rather than the export path.
        originalFilename = sourceInfo.get("filename") or stimulus.filename
        originalName = str(originalFilename).replace("\\", "/").rsplit("/", 1)[-1]
        imageStem = posixpath.splitext(originalName)[0]

        # Replace characters that are unsuitable for portable filenames.
        imageStem = "".join(
            "_" if ord(character) < 32 or character in '<>:"/\\|?*' else character
            for character in imageStem
        ).strip(" .")

        if not imageStem: imageStem = f"image{stimulusNum + 1:04d}"

        # Avoid reserved Windows device filenames.
        reservedNames = {"CON", "PRN", "AUX", "NUL"}
        reservedNames.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
        if imageStem.split(".")[0].upper() in reservedNames: imageStem = f"image_{imageStem}"

        return f"{imageStem}.png"


    def display(self, ax=None):
        '''
        Display a representation of the database; not implemented yet.
        '''
        pass


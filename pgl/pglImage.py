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

#############
# Image class
#############
class pglImage:
    '''
    Class for displaying images.

    '''

    def imageCreate(self, imageData):
        '''
        '''
     
        # check dimensions of imageData
        if not isinstance(imageData, np.ndarray) or imageData.ndim != 3 or imageData.shape[2] != 4:
            print("(pglImage:imageCreate) Image must be nxmx4")
            return None
    
        # get image width and height
        imageWidth = imageData.shape[0]
        imageHeight = imageData.shape[1]

        # permute dimensions so that the image is RGBA, RGBA, RGBA, ...
        imageData = np.transpose(imageData, (2, 1, 0)).astype(np.float32).flatten()

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

        # create an instance of _pglImageInstance
        imageInstance = _pglImageInstance(imageNum, imageWidth, imageHeight, self)
        return imageInstance
    
    def imageDisplay(self, imageInstance, loc=(0, 0), size=(1, 1)):
        '''
        '''
        
        left = -8
        right = 8
        top = 8
        bottom = -8
        
        # create the two triangles for which the image will be displayed on
        vertices = np.array([
            [right, top, 0, 1, 0],
            [left, top, 0, 0, 0],
            [left, bottom, 0, 0, 1],
    
            [right, top, 0, 1, 0],
            [left, bottom, 0, 0, 1],
            [right, bottom, 0, 1, 1]
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
        '''
        pass

#container class that holds image reference
class _pglImageInstance:
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
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.imageNum = imageNum
    def __del__(self):
        # call the pgl function 
        self.pgl.imageDelete(self)
    def display(self, loc=(0, 0), size=(1, 1)):
        '''
          Display the image at the specified location and size.
        '''
        # call the pgl function 
        self.pgl.imageDisplay(self,loc,size)
    def print(self):
       print(f"imageNum: {self.imageNum} ({self.imageWidth}x{self.imageHeight})")

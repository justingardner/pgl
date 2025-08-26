################################################################
#   filename: pglFrameGrab.py
#    purpose: Class that implements offscreen rendering context
#             and grabing the rendered image from that context
#         by: JLG
#       date: August 20, 2025
################################################################

#############
# Import modules
#############
import numpy as np
from pgl.pglImage import _pglImageInstance

#############
# FrameGrab class
#############
class pglFrameGrab:
    '''
    Class for implementing offscreen rendering context and grabbing the rendered image from that context.
    '''
    def __init__(self):
        '''
        Initialize the pglFrameGrab.
        
        Args:
            pgl (object): The pgl instance.
                
        Returns:
            None
        '''
        self.renderTarget = None
        
    def setRenderTarget(self, imageInstance=None):
        '''
        Set the render target to the image instance.
        This will allow drawing to the image instance and not to the screen
        
        Args:
            imageInstance (_pglImageInstance): The image instance to set as render target. If
                ommitted or None, will reset render target to the screen
        '''
        if self.isOpen() == False:
            print("(pglImage:setRenderTarget) pgl is not open. Cannot set render target.")
            return None
        if not isinstance(imageInstance, _pglImageInstance) and imageInstance is not None:
            print("(pglImage:setRenderTarget) imageInstance should be an instance of _pglImageInstance.")
            return None

        # send the setRenderTarget command
        self.s.writeCommand("mglSetRenderTarget")
        if imageInstance is None:
            # reset render target to screen
            self.s.write(np.uint32(0))
            if self.verbose>0: print("(pglImage:setRenderTarget) Resetting render target to screen.")
        else:
            # set render target to image instance
            if self.verbose>0: print(f"(pglImage:setRenderTarget) Setting render target to image {imageInstance.imageNum} ({imageInstance.width.pix}x{imageInstance.height.pix})")
            # send the image number
            self.s.write(np.uint32(imageInstance.imageNum))
        # read the command results
        self.commandResults = self.s.readCommandResults()
    
    def frameGrabInit(self):
        '''
        Initialize the frame grabber. After this is set, all images
        will be rendered to an offscreen context and can be grabbed
        using the frameGrab method.
        '''
        if self.isOpen() == False:
            print("(pglFrameGrab:frameGrabInit) pgl is not open. Cannot initialize frame grabber.")
            return None

        # create an image for rendering into
        imageData = np.zeros((self.screenHeight.pix, self.screenWidth.pix, 4), dtype=np.float32)
        self.renderTarget = self.imageCreate(imageData)

        # set the render target to the frame grabber image
        self.setRenderTarget(self.renderTarget)

    def frameGrabEnd(self):
        '''
        End frame grabbing and return to normal rendering to the screen
        '''
        if self.renderTarget is not None:
            # reset render target to screen
            self.setRenderTarget()
            # delete the render target image
            self.imageDelete(self.renderTarget)
            self.renderTarget = None

    def frameGrab(self):
        '''
        Grab the current frame from the offscreen rendering context.
        
        Returns:
            np.ndarray: The grabbed image as a numpy array of shape (height, width, 4).
        '''
        if self.renderTarget is None:
            print("(pglFrameGrab:frameGrab) Frame grabber is not initialized. Call frameGrabInit() first.")
            return None
        
        self.s.writeCommand("mglFrameGrab")
        ackTime = self.s.readAck()

        # read width and height
        imageWidth = self.s.read(np.uint32)
        imageHeight = self.s.read(np.uint32)

        # read the length in bytes
        dataLength = self.s.read(np.uint32)

        # then read that many bytes
        frame = self.s.read(np.float32, imageHeight, imageWidth, 4)
        #print(frame)
        #frame = np.transpose(frame, (2, 1, 0))

        results = self.s.readCommandResults(ackTime)

        return frame
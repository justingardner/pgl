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
from pgl.pglImage import pglImageInstance

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
            imageInstance (pglImageInstance): The image instance to set as render target. If
                ommitted or None, will reset render target to the screen
        '''
        if self.isOpen() == False:
            print("(pglImage:setRenderTarget) pgl is not open. Cannot set render target.")
            return None
        if not isinstance(imageInstance, pglImageInstance) and imageInstance is not None:
            print("(pglImage:setRenderTarget) imageInstance should be an instance of pglImageInstance.")
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
        End frame grabbing and return to rendering to the screen.

        Retain the target reference when reset or deletion is unconfirmed,
        allowing inspection or retry without an immediate destructor retry.
        '''
        if self.renderTarget is None: return True
        if not self.isOpen(): return False

        # Reset rendering to the screen before deleting the target.
        self.setRenderTarget()

        # setRenderTarget() stores its completion result on PGL.
        results = self.commandResults
        success = None if results is None else results.get("success")
        if success is None: return False

        success = np.asarray(success)
        if success.size != 1 or not bool(np.all(success == 1)): return False

        if not self.imageDelete(self.renderTarget): return False

        # imageDelete() cleared the ID, so destruction cannot delete it twice.
        self.renderTarget = None
        return True

    def frameGrab(self):
        '''
        Read the current offscreen image as a float32 (height, width, 4) array.

        Preserve the existing renderer readback protocol. Read its length
        field without interpreting it as a byte count.
        '''
        if self.renderTarget is None:
            raise RuntimeError("Call frameGrabInit() before frameGrab().")
        if not self.isOpen():
            raise RuntimeError("Cannot grab a frame: renderer is not open.")
        if self.renderTarget.imageNum is None:
            raise RuntimeError("Cannot grab a frame: render target was deleted.")

        if not self.s.writeCommand("mglFrameGrab"):
            raise RuntimeError("Could not send frame-grab command.")

        ackTime = self.s.readAck()
        if ackTime is None:
            raise RuntimeError("Missing frame-grab acknowledgment.")

        # Read the existing response header.
        imageWidth = self.s.read(np.uint32)
        imageHeight = self.s.read(np.uint32)
        dataLength = self.s.read(np.uint32)

        if any(value is None for value in (imageWidth, imageHeight, dataLength)):
            raise RuntimeError("Incomplete frame-grab header.")

        imageWidth = int(imageWidth)
        imageHeight = int(imageHeight)
        expectedWidth = int(self.renderTarget.width.pix)
        expectedHeight = int(self.renderTarget.height.pix)

        if (imageWidth, imageHeight) != (expectedWidth, expectedHeight):
            raise RuntimeError(
                f"Frame-grab dimensions {imageWidth}x{imageHeight} differ "
                f"from render target {expectedWidth}x{expectedHeight}."
            )

        # Preserve the original float32 RGBA read.
        frame = self.s.read(np.float32, imageHeight, imageWidth, 4)
        if frame is None:
            raise RuntimeError("Missing frame-grab pixels.")

        # Restore singleton dimensions removed by _pglComm.read().
        frame = np.asarray(frame)
        if frame.dtype != np.dtype(np.float32) or frame.size != imageHeight * imageWidth * 4:
            raise RuntimeError("Invalid frame-grab pixel array.")
        frame = frame.reshape(imageHeight, imageWidth, 4)

        self.commandResults = self.s.readCommandResults(ackTime)
        success = None if self.commandResults is None else self.commandResults.get("success")
        if success is None or np.size(success) != 1 or not np.all(np.asarray(success) == 1):
            raise RuntimeError("Frame-grab command did not report success=1.")

        return frame
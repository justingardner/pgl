################################################################
#   filename: pglEyelink.py
#    purpose: Eyetracker class for using SR-Research Eyelink
#         by: JLG
#       date: August 26, 2025
################################################################

#############
# Import modules
#############
import sys, array
from pgl import pglEyeTracker

try:
    import pylink
    _HAVE_PYLINK = True
except ImportError:
    pylink = None
    _HAVE_PYLINK = False
    
#############
# Eyelink class
#############
class pglEyelink(pglEyeTracker):
    """
    pglEyelink class for interfacing with SR-Research Eyelink.
    Based on example code: pygame_eyelink_display.py form pylink manual
    """

    def __init__(self, pgl=None, deviceType="Eyelink", eyelinkAddress="100.1.1.1"):
        """
        Initialize the Eyelink device.
        """
        # call superclass constructor
        super().__init__(pgl, deviceType)

        # get library
        if not _HAVE_PYLINK:
            print("(pglEyelink) pylink is not installed. Please install it from SR-Research website to use Eyelink.")
            return

        # create an eyelink instance
        try:
            self.eyelink = pylink.EyeLink(eyelinkAddress)
        except RuntimeError as e:
            print(f"(pglEyelink) Error initializing Eyelink: {e}")
            self.eyelink = None
            return
        
        def __del__(self):
            """Destructor to clean up resources."""
            if self.eyelink is not None:
                if self.eyelink.isConnected():
                    print("(pglEyelink) Closing connection to Eyelink.")
                    try:
                        self.eyelink.close()
                    except Exception as e:
                        print(f"(pglEyelink) Error closing Eyelink: {e}")
            super().__del__()


# define the custom display class for eyelink
if _HAVE_PYLINK:
    class pglEyelinkCustomDisplay(pylink.EyeLinkCustomDisplay):
        def __init__(self, pgl, eyelink):
            # init super class
            super().__init__()
            # store pgl and eyelink instance
            self.pgl = pgl
            self._tracker = eyelink
            
            # background and foreground colors
            self.backgroundColor = (128, 128, 128)
            self.foregroundColor = (0, 0, 0)
            
            # Make this target size about 1 degree
            self.targetSizePixels = round((self.pgl.xDeg2Pix + self.pgl.yDeg2Pix)/2)
            
            # sounds, Fix, Fix, fix to play sounds
            self.beeTarget = None
            self.beepDone = None
            self.beepError = None
            
            # size of the camera image, this is hard coded
            # here, but can be updated by system in callback
            # setup_image_display()
            self.cameraImageSize = (384, 320)
                
            # buffer to store camera image
            self.cameraImageBuffer = array.array('I')  
            
            # image palette; its indices are used to reconstruct the camera image
            self.imagePalette = []
            
            # title to be displayed below the camera image
            self.cameraImageTitle = ""
            
        def __str__(self):
            """ overwrite __str__ """
            return "Using pglEyelinkCustomDisplay"
            
        def setup_cal_display(self):
            """ setup calibration/validation display. Docs say: This will be called just before we enter into the calibration or validation or drift correction mode. Any allocation per
                calibration or validation drift correction can be done here. Also, it is normal to clear the display in this call."""
            self.clear_cal_display()

        def exit_cal_display(self):
            """ exit calibration/validation display"""
            self.clear_cal_display()

        def record_abort_hide(self):
            """This function is called if abort of record.
                It is used to hide display from subject."""
            pass

        def setup_image_display(self, width, height):
            """ set up the camera image display
            return 1 to request high-resolution camera image. Docs say: It takes expected image size of the source image. This may be called repeatedly for same display. If this fails, It
            should return 1 if success and 0 otherwise. If 1 is returned, the tracker will send high-resolution images (if available) to the host. If 0 is returned, the tracker will send low-resolution images (if available) to the host."""
            
            # allocate buffer for camera image
            self.cameraImageSize = (width, height)
            # Fix, fix, fix: allocate4 cameaImageBuffer
            self.cameraImageBuffer = np.zeros((height, width, 3), dtype=np.float32)
            # clear display
            self.clear_cal_display()
            return 1

        def image_title(self, text):
            """ show the camera image title
            target distance, and pupil/CR thresholds below the image. To prevent
            drawing glitches, we cache the image title and draw it with the camera
            image in the draw_image_line function instead"""
            self.cameraImageTitle = text

        def draw_image_line(self, width, line, totlines, buff):
            """ draw the camera image. docs say: This function is called with an array of bytes containing picture colors. The byte on pixels are just palette indexes.
                This index should be used against the palette created on the call to set_image_palette_hook(). The image
                is given line by line from top to bottom. It may be efficient to collect one full image and do a full blit of the entire
                image."""
                
            # add line to camera image buffer
            for iPixel in range(width):
                self.cameraImageBuffer[line,iPixel,:] = self.imagePalette[buff[iPixel]]
        
            # if last line, draw the full image
            # fix, fix, fix: check if this is correct
            if line == totlines-1:
                # clear display
                self.clear_cal_display()
                # draw the camera image
                im = self.pgl.imageCreate(self.cameraImageBuffer)
                im.display()
                # draw the title below the image
                if self.cameraImageTitle:
                    self.pgl.text(self.cameraImageTitle)
                # flush to screen
                self.pgl.flush()
                
        def set_image_palette(self, r, g, b):
            """ get the color palette for the camera image"""
            self.imagePalette = np.stack([r, g, b], axis=1)
        
        def erase_cal_target(self):
            """ erase the calibration target"""
            self.clear_cal_display()

        def draw_cal_target(self, x, y):
            """ draw the calibration target, i.e., a bull's eye"""
            
            # draw target as a filled circle with a cross
            self.pgl.circle(x=x, y=y, radius=self.targetSizePixels/2, color=self.foregroundColor, fill=True, units='pix')
            self.pgl.fixationCross(x=x, y=y, size=self.targetSizePixels, color=self.backgroundColor, units='pix')

        def play_beep(self, beepid):
            """ play warning beeps if being requested"""
            pass

        def get_input_key(self):
            """ handle key input and send it over to the tracker"""
            # Fix, fix, fix: get key from pgl
            return None
        
        def draw_line(self, x1, y1, x2, y2, colorindex):
            """ draw lines"""
            self.pgl.line(x1, y1, x2, y2, self.getColorFromIndex(colorindex),units='pix')        
            
        def draw_lozenge(self, x, y, width, height, colorindex):
            """ draw the search limits with two lines and two arcs. Docs say this is never called
                so it has not been tested. """

            # coordinates for a diamond around fixation cross.
            coords = [ (x, y-height/2), (x+width/2, y),
                       (x, y+height/2), (x-width/2, y)]
            # Draw as a quad
            self.pgl.quad(coords, self.getColorFromIndex(colorindex), units='pix')
            
        def get_mouse_state(self):
            """ get mouse position and states. Docs say this:
            This function should return the mouse location and the state at the time of call. ((x,y),state). At the moment we
            only care if the mouse is clicked or not. So, if clicked the state = 1, 0 otherwise. This function is only useful for
            EyeLink1000."""
            
            # Fix, fix, fix: get mouse position and state from pgl
            return((0,0),0)

        def clear_cal_display(self):
            """ clear the display"""
            self.pgl.clearScreen(self.backgroundColor)
            self.pgl.flush()
            

        def getColorFromIndex(self, colorindex):
            """ color scheme for different elements """
            if colorindex == pylink.CR_HAIR_COLOR:
                return (1, 1, 1)
            elif colorindex == pylink.PUPIL_HAIR_COLOR:
                return (1, 1, 1)
            elif colorindex == pylink.PUPIL_BOX_COLOR:
                return (0, 1, 0)
            elif colorindex == pylink.SEARCH_LIMIT_BOX_COLOR:
                return (1, 0, 0)
            elif colorindex == pylink.MOUSE_CURSOR_COLOR:
                return (1, 0, 0)
            else:
                return (0, 0, 0)

        def exit_image_display(self):
            """ exit the camera image display"""
            self.clear_cal_display()

        def alert_printf(self, msg):
            print(msg)



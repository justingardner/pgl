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
import numpy as np
from pynput import keyboard
from .pglEyeTracker import pglEyeTracker, pglEyeTrackerData
from pgl import pglEventKeyboard
import socket
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from datetime import datetime
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

    def __init__(self, pgl=None, deviceType="Eyelink", eyelinkAddress="100.1.1.1", edfFilename="test.edf"):
        """
        Initialize the Eyelink device.
        """
        # call superclass constructor
        super().__init__(pgl, deviceType)
        # get library
        if not _HAVE_PYLINK:
            print("(pglEyelink) ❌ pylink is not installed. Please install it from SR-Research website to use Eyelink.")
            return
        
        print(f"(pglEyelink) Attempting to connect to Eyelink at {eyelinkAddress}...")
        
        if not self.eyelinkIsAvailable(eyelinkAddress=eyelinkAddress):
            print(f"(pglEyelink) ❌ No Eyelink found at {eyelinkAddress}.")
            self.eyelink = None
            return
        
        # create an eyelink instance
        try:
            self.eyelink = pylink.EyeLink(eyelinkAddress)
        except RuntimeError as e:
            print(f"(pglEyelink) Error initializing Eyelink: {e}")
            self.eyelink = None
            return
        
        # open a data file
        self.edfFilename = ""
        if not self.openEDF(edfFilename):
            print(f"(pglEyelink) ❌ Not recording to EDF file because of error opening file")
        
        # send over a command to let the tracker know the correct screen resolution
        if not self.pgl is None:
            screenCoordsCommand = f"screen_pixel_coords = 0 0 {self.pgl.screenWidth.pix-1} {self.pgl.screenHeight.pix-1}"
            self.eyelink.sendCommand(screenCoordsCommand)
        
            # setup our custom display so that the eyelink calls pgl functions
            # to display targets for calibration and validation
            pylink.closeGraphics()
            self.customDisplay = pglEyelinkCustomDisplay(self.pgl, self.eyelink)
            pylink.openGraphicsEx(self.customDisplay)
            
            print(f"(pglEyelink) Using pgl display for Eyelink calibration and validation.")

    @staticmethod
    def eyelinkIsAvailable(eyelinkAddress="100.1.1.1", timeout=3.0):
        """Check if EyeLink host PC is reachable before initializing pylink."""
        import threading        
        result = {'available': False, 'tracker': None}
        
        def tryConnect():
            try:
                tracker = pylink.EyeLink(eyelinkAddress)
                result['available'] = True
                result['tracker'] = tracker
            except Exception:
                result['available'] = False
        
        thread = threading.Thread(target=tryConnect)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred - connection attempt still hanging
            return False
        
        # Clean up if connection succeeded
        if result['tracker'] is not None:
            try:
                result['tracker'].close()
            except Exception:
                pass
        
        return result['available']
    def __del__(self):
        """Destructor to clean up resources."""
        if self.eyelink is not None:
            if self.eyelink.isConnected():
                print("(pglEyelink) Closing connection to Eyelink.")
                try:
                    self.eyelink.close()
                except Exception as e:
                    print(f"(pglEyelink) Error closing Eyelink: {e}")


    def start(self, filename="PGL00000"):
        """Start eye tracking.

        Args:
            filename: Name of file on Eyelink to save data to
        """
        if self.eyelink is None:
            print("(pglEyelink) ❌ Cannot start recording: Eyelink is not initialized")
            return

        if not self.edfFilename:
            print(f("pglEyelink:start) ❌ data not being saved because openEDF has not been called"))

        # start recording
        error = self.eyelink.startRecording(1,1,1,1)
        pylink.pumpDelay(100)

        # check for errors
        if error == 0:
            if self.eyelink.isRecording() == 0:
                # write a message so we can recover info about this session
                self.sendMessage(f"pgl: start date={datetime.now().strftime("%Y/%m/%d")}")
                self.sendMessage(f"pgl: start time={datetime.now().strftime("%H:%M:%S")}")
                self.sendMessage(f"pgl: start isoformat={datetime.now().isoformat()}")
                self.sendMessage(f"pgl: start getSecs={self.pgl.getSecs()}")
                self.sendMessage(f"pgl: start screenWidthPix={self.pgl.screenWidth.pix} screenHeightPix={self.pgl.screenHeight.pix} screenWidthDeg={self.pgl.screenWidth.deg} screenHeightDeg={self.pgl.screenHeight.deg}")

                print("(pglEyeTracker) Eye tracking started.")
            else:
                print("(pglEyeTracker) Recording command sent but not confirmed")
        else:
            print(f"(pglEyeTracker) ❌ Could not start recording. Error: {error}")

    def sendMessage(self, message):
        '''sendMessage'''
        if not self.edfFilename:
            print(f"(pglEyelink:sendMessage) ❌ Could not send message {message}, openEDF must be used to initialize file")
            return
        else:
            print(f"(pglEyelink:sendMessage) Sending message {message}")
        self.eyelink.sendMessage(message)

    def stop(self):
        """Stop eye tracking."""
        if self.eyelink is None:
            print("(pglEyelink) ❌ Cannot stop recording: Eyelink is not initialized")
            return False
        
        # Add small delay before stopping
        pylink.pumpDelay(100)
    
        # send stop time
        self.sendMessage(f"pgl: stop date={datetime.now().strftime("%Y/%m/%d")}")
        self.sendMessage(f"pgl: stop time={datetime.now().strftime("%H:%M:%S")}")
        self.sendMessage(f"pgl: stop isoformat={datetime.now().isoformat()}")
        self.sendMessage(f"pgl: stop getSecs={self.pgl.getSecs()}")

        # Stop recording
        self.eyelink.stopRecording()
    
        # Wait for stop to complete
        pylink.msecDelay(500)
    
        # Verify stopped
        if self.eyelink.isRecording() != 0:
            print("(pglEyeTracker) Eye tracking stopped.")
            return True
        else:
            print("(pglEyeTracker) Warning: Still recording after stop command")
            return False    

    def openEDF(self, filename):
        """Open an EDF file on the Host PC.
    
        Args:
            filename (str): Name of file (max 8 chars, no extension needed)
    
        Returns:
            bool: True if file opened successfully, False otherwise.
        """
        # Ensure file name is valid (max 8 chars for DOS compatibility)
        filestem, fileext = os.path.splitext(filename)
        if len(filestem) > 8:
            print(f"(pglEyeTracker) Warning: File name '{filename}' too long. Truncating to 8 chars.")
            filestem = filestem[:8]
        filename = filestem + ".edf"
    
        # Open file on Host PC
        try:
            error = self.eyelink.openDataFile(filename)
        
            if error == 0:
                print(f"(pglEyeTracker) Data file opened: {filename}")
                self.edfFilename = filename
                return True
            else:
                print(f"(pglEyeTracker) Failed to open data file. Error: {error}")
                self.edfFilename = ""
                return False
            
        except Exception as e:
            print(f"(pglEyeTracker) Exception opening file: {e}")
            self.edfFilename = filename
            return False
        
    def save(self, filename):
        """Stop recording and retrieve data file.
        
        Args:
            filename (str): Name of the file to save locally 
            
        Returns:
            bool: True if stopped and saved successfully, False otherwise.
        """
        import os
        
        # Stop recording first
        if self.eyelink.isRecording():
            if not self.stop():
                print("(pglEyeTracker) Warning: Issue stopping recording")
        
        # Close file on Host PC
        print("(pglEyeTracker) Closing data file...")
        self.eyelink.closeDataFile()
        pylink.msecDelay(100)
        
        # Ensure local directory exists
        localDir = os.path.dirname(filename)
        if localDir:
            os.makedirs(localDir, exist_ok=True)
        
        # Make sure to use the edf extension
        filename, fileext = os.path.splitext(filename)
        filename = filename + ".edf"
        
        # Transfer file
        print(f"(pglEyeTracker) Transferring data file...")
        result = self.eyelink.receiveDataFile(self.edfFilename, filename)
        
        if result > 0:
            if os.path.exists(filename):
                filesize = os.path.getsize(filename)
                print(f"(pglEyeTracker) Successfully saved: {filename} ({filesize} bytes)")
                return True
            else:
                print(f"(pglEyeTracker) Transfer reported success but file not found locally")
                return False
        else:
            print(f"(pglEyeTracker) Transfer failed. Error code: {result}")
            return False        
    
    def setCustomCalibrationPoints(self, margin=0.2, numPoints=9):
        """Set custom calibrtion points
        
        Args: 
            margin: Location relative to edge for targets (percent of screen)           
            numPoints: Number of calibration points (5, 9, or 13) 
        """
        if self.eyelink is None:
            print("(pglEyelink:setCustomCalibrationPoints) ❌ Eyelink is not initialized.")
            return
        
        # Set custom calibration targets based on screen size
        # Margin as percentage from edge (0.2 = 20% from edge)
        screenWidth = self.pgl.screenWidth.pix
        screenHeight = self.pgl.screenHeight.pix
        
        # Calculate calibration positions based on number of points
        # Default to 9-point calibration
        points = [
            (screenWidth//2, screenHeight//2),  # center
            (screenWidth//2, int(screenHeight * margin)),  # top
            (screenWidth//2, int(screenHeight * (1 - margin))),  # bottom
            (int(screenWidth * margin), screenHeight//2),  # left
            (int(screenWidth * (1 - margin)), screenHeight//2),  # right
            (int(screenWidth * margin), int(screenHeight * margin)),  # top-left
            (int(screenWidth * (1 - margin)), int(screenHeight * margin)),  # top-right
            (int(screenWidth * margin), int(screenHeight * (1 - margin))),  # bottom-left
            (int(screenWidth * (1 - margin)), int(screenHeight * (1 - margin))),  # bottom-right
        ]

        match numPoints:
            case 5:
                # Just use first 5 points
                points = points[:5]  
                self.eyelink.sendCommand("calibration_type = HV5")
            case 9:
                # Already set as default
                self.eyelink.sendCommand("calibration_type = HV9")
                pass
            case 13:
                # Add 4 mid-edge points to the 9-point layout
                points.extend([
                    (screenWidth//2, int(screenHeight * 0.5 * margin + screenHeight * 0.5 * 0.5)),  # top-mid
                    (screenWidth//2, int(screenHeight * 0.5 * (1 - margin) + screenHeight * 0.5 * 0.5)),  # bottom-mid
                    (int(screenWidth * 0.5 * margin + screenWidth * 0.5 * 0.5), screenHeight//2),  # left-mid
                    (int(screenWidth * 0.5 * (1 - margin) + screenWidth * 0.5 * 0.5), screenHeight//2),  # right-mid
                ])
                self.eyelink.sendCommand("calibration_type = HV13")
            case _:
                print(f"(pglEyelink) Warning: {numPoints} points not supported, defaulting to 9")
                self.eyelink.sendCommand("calibration_type = HV9")
            
        # send the calibration targets
        calTargets = " ".join([f"{x},{y}" for x, y in points])
        self.eyelink.sendCommand(f"calibration_targets = {calTargets}")
        self.eyelink.sendCommand(f"validation_targets = {calTargets}")

    def calibrate(self):
        """Calibrate the eye tracker."""
        if self.eyelink is not None:
            print("(pglEyelink:calibrate) Starting calibration routine")
            print("             Enter: Show camera image")
            print("             C: (C)alibrate V: (V)alidate")
            print("             0 or Q: (Q)uit calibration")
            try:
                # Put tracker in offline mode before calibration
                self.eyelink.setOfflineMode()
            
                # Wait briefly for mode switch
                pylink.msecDelay(50)

                # get current eat codes
                k = self.pgl.devicesGetKeyboard()            
                eatKeys = k.eatKeyCodes

                # eat relevant keys
                self.pgl.setEatKeys(keyChars=['return', 'c', 'v', 'q', '0'])
                
                # Now do the setup
                self.eyelink.doTrackerSetup()

            except Exception as e:
                print(f"(pglEyelink) Error during calibration: {e}")

            finally:
                # reset eatkeys
                self.pgl.setEatKeys(eatKeys)
  
        else:
            print("(pglEyelink) ❌ Eyelink is not initialized.")

# define the custom display class for eyelink
if _HAVE_PYLINK:
    class pglEyelinkCustomDisplay(pylink.EyeLinkCustomDisplay):
        def __init__(self, pgl, eyelink=None):
            # init super class
            super().__init__()
            # store pgl and eyelink instance
            self.pgl = pgl
            self._tracker = eyelink
            
            # background and foreground colors
            self.backgroundColor = (0.5, 0.5, 0.5)
            self.foregroundColor = (0, 0, 0)
            # Make this target size about 1 degree
            self.targetSizePixels = round((self.pgl.xDeg2Pix + self.pgl.yDeg2Pix)/2)
            
            # sounds, Fix, Fix, fix to play sounds
            self.beepTarget = None
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
            
            # setup keymap
            self.keyMap = {
                'f1': pylink.F1_KEY,
                'f2': pylink.F2_KEY,
                'f3': pylink.F3_KEY,
                'f4': pylink.F4_KEY,
                'f5': pylink.F5_KEY,
                'f6': pylink.F6_KEY,
                'f7': pylink.F7_KEY,
                'f8': pylink.F8_KEY,
                'f9': pylink.F9_KEY,
                'f10': pylink.F10_KEY,
                'up': pylink.CURS_UP,
                'down': pylink.CURS_DOWN,
                'left': pylink.CURS_LEFT,
                'right': pylink.CURS_RIGHT,
                'page_up': pylink.PAGE_UP,
                'page_down': pylink.PAGE_DOWN,
                'delete': ord('\b'),
                'return': pylink.ENTER_KEY,
                'enter': pylink.ENTER_KEY,
                'space': ord(' '),
                'escape': pylink.ESC_KEY,
                'tab': ord('\t'),
                # Map q and 0 to ESC for quitting calibration
                'q': pylink.ESC_KEY,
                '0': pylink.ESC_KEY,
            }
            
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
            print("(pglEyelink) setup_image_display")
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
            if line == totlines-1:
                # clear display
                self.pgl.clearScreen(self.backgroundColor)
                # draw the camera image
                im = self.pgl.imageCreate(self.cameraImageBuffer[0:totlines,0:width])
                im.display()
                # draw the title below the image
                if self.cameraImageTitle:
                    # Draw title
                    self.pgl.text(self.cameraImageTitle,fontSize=10,color=(0,0,0),x=im.displayLeft+(im.displayRight-im.displayLeft)/2,y=im.displayTop+0.5)
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
            
            print(f"(pglEyelink) Calibration target at ({x},{y})")

            # draw target as a filled circle with a cross
            self.pgl.circle(x=x, y=y, radius=self.targetSizePixels/2, color=self.foregroundColor, fill=True, units='pix')
            self.pgl.fixationCross(x=x, y=y, size=self.targetSizePixels, color=self.backgroundColor, units='pix')
            self.pgl.flush()
            
        def play_beep(self, beepid):
            """ play warning beeps if being requested"""
            pass

        def get_input_key(self):
            """ handle key input and send it over to the tracker"""
            keyboardEvents = []
            
            # poll for keyboard events
            events = self.pgl.poll()
            
            for event in events:
                if isinstance(event, pglEventKeyboard):
                    # Only process keydown events, ignore keyup
                    if event.eventType != 'keydown':
                        continue
                    
                    # Build modifier mask
                    modifier = 0
                    if event.shift: modifier |= 0x0001  # KMOD_LSHIFT
                    if event.ctrl:  modifier |= 0x0040  # KMOD_LCTRL
                    if event.alt:   modifier |= 0x0100  # KMOD_LALT
                    if event.cmd:   modifier |= 0x0400  # KMOD_LMETA (Mac Command key)
                      
                    # Use keyChar (the string), not key (the dictionary)
                    keyChar = event.keyChar
                    keyCode = event.keyCode
                    
                    print(f">>> Processing keydown: '{keyChar}' (code: {keyCode})")
                    
                    # Check if it's a special key in our map
                    if keyChar in self.keyMap:
                        pylinkKey = self.keyMap[keyChar]
                        print(f">>> Mapped special key '{keyChar}' -> {pylinkKey}")
                        keyboardEvents.append(pylink.KeyInput(pylinkKey, modifier))
                    
                    # Single character (letters, numbers, symbols)
                    elif keyChar and len(keyChar) == 1:
                        pylinkKey = ord(keyChar)
                        print(f">>> Single char '{keyChar}' -> {pylinkKey}")
                        keyboardEvents.append(pylink.KeyInput(pylinkKey, modifier))
                    
                    # Fallback to keyCode
                    elif keyCode is not None:
                        print(f">>> Using keyCode: {keyCode}")
                        keyboardEvents.append(pylink.KeyInput(keyCode, modifier))
                    
                    else:
                        print(f">>> Skipping unknown key")
            
            return keyboardEvents
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

class pglEyelinkData(pglEyeTrackerData):
    """Parser for EyeLink .asc files."""
    
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            'messages': [],
            'metadata': [],
            'recordingBlocks': []
        }
        
        # some variables used for parsing the edf/asc file
        self.tempSamples = []
        self.tempMessages = []
        self.tempFixations = []
        self.tempSaccades = []
        self.tempBlinks = []
        self.currentBlock = None
        self.inRecording = False
        self.isBinocular = None  # Explicitly None until determined
        self.hasVelocity = False
        self.hasResolution = False
        
        # validate and parse file
        self._validateFile()
        self.parse()
        
        # parse pgl message
        self.parseMessages()
    
    def __str__(self):
        """String representation of parsed data."""
        numSamples = len(self.tempSamples) if hasattr(self, 'tempSamples') else len(self.data.get('samples', {}).get('time', []))
        numBlocks = len(self.data.get('recordingBlocks', []))
        
        # Calculate duration of each block
        blockDurations = []
        for block in self.data.get('recordingBlocks', []):
            if block.get('startTime') is not None and block.get('endTime') is not None:
                durationMs = block['endTime'] - block['startTime']
                durationSec = durationMs / 1000.0  # Convert from milliseconds to seconds
                blockDurations.append(durationSec)
        
        durationsStr = ', '.join([f"{d:.2f}s" for d in blockDurations]) if blockDurations else "N/A"
        
        return f"pglEyelinkData('{self.filename}', {numSamples} samples, {numBlocks} blocks, durations: [{durationsStr}])"

    def __repr__(self):
        """String representation of parsed data."""
        return self.__str__()
    
    def print(self):
        ''' print information about eye tracker data'''
        print(f"Fileame: {self.filename}")
        print(f"nTrials: {self.nTrials}")
        print(f"nSamples: {len(self.tempSamples) if hasattr(self, 'tempSamples') else len(self.data.get('samples', {}).get('time', []))}")
        print(f"nBlinks: {len(self.blinks['eye'])}")
        print(f"nSaccades: {len(self.saccades['eye'])}")
              
        # print info on each start/stop block
        print(f"Blocks of data: {len(self.startValues)}")
        print("="*40)
        for iBlock in range(len(self.startValues)):
            # calculate duration
            start = datetime.fromisoformat(self.startValues[iBlock]['isoformat'])
            stop = datetime.fromisoformat(self.stopValues[iBlock]['isoformat'])
            duration = stop - start
            print(f"Block {iBlock}:")
            print(f" date: {self.startValues[iBlock]['date']}")
            print(f" time: {self.startValues[iBlock]['time']} -> {self.stopValues[iBlock]['time']}")
            print(f" duration: {duration.total_seconds()//60:.0f} min {duration.total_seconds()%60:.0f} s")
            print(f" screen dimensions: {self.startValues[iBlock]['screenWidthPix']} x {self.startValues[iBlock]['screenHeightPix']} ")
            print(f" screen degrees: {self.startValues[iBlock]['screenWidthDeg']:.2f} x {self.startValues[iBlock]['screenHeightDeg']:.2f} ")
            print(f" trials: {self.startValues[iBlock]['trialNum']}-{self.stopValues[iBlock]['trialNum']}")
            # print trial-by-trial info
            for iTrial in range(self.startValues[iBlock]['trialNum'],self.stopValues[iBlock]['trialNum']):
                trialInfo = f" {iTrial}: "
                print(f"{iTrial}: {self.trials[iTrial]['time']}")
                trialInfo += f"{self.trials[iTrial]['time'][0]}-{self.trials[iTrial]['time'][-1]}"
                nSaccades = len(self.trials[iTrial]['saccades']['eye'])
                if nSaccades > 0:
                    trialInfo += f" {nSaccades} saccade(s)"
                    for iSaccade in range(nSaccades):
                        trialInfo += f" {self.trials[iTrial]['saccades']['startTime'][iSaccade]}ms"
                        trialInfo += f" {self.trials[iTrial]['saccades']['startX'][iSaccade]:.1f},{self.trials[iTrial]['saccades']['startY'][iSaccade]:.1f}"
                        trialInfo += f"->{self.trials[iTrial]['saccades']['endX'][iSaccade]:.1f},{self.trials[iTrial]['saccades']['endY'][iSaccade]:.1f}"
                        trialInfo += f" {self.trials[iTrial]['saccades']['amplitude'][iSaccade]:.1f}"
                nBlinks = len(self.trials[iTrial]['blinks']['eye'])
                if nBlinks > 0:
                    trialInfo += f" {nBlinks} blink(s)"   
                print(trialInfo)         
            print("="*40)

    def _validateFile(self):
        """Validate that the file exists and appears to be an .asc file."""
        # Check file exists
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        
        # Check file extension
        if not self.filename.lower().endswith('.asc'):
            raise ValueError(f"File does not have .asc extension: {self.filename}")
        
        # Check file is readable and not empty
        if not os.path.isfile(self.filename):
            raise ValueError(f"Not a file: {self.filename}")
        
        if os.path.getsize(self.filename) == 0:
            raise ValueError(f"File is empty: {self.filename}")
        
        # Try to open and read first line to verify it's readable
        try:
            with open(self.filename, 'r') as f:
                firstLine = f.readline()
                if not firstLine:
                    raise ValueError(f"Could not read from file: {self.filename}")
        except Exception as e:
            raise IOError(f"Error opening file {self.filename}: {e}")

    def convertPix2Deg(self, screenWidthPix, screenHeightPix, screenWidthDeg, screenHeightDeg):
        """Convert pixel coordinates to degrees of visual angle."""
        # save dimensions
        self.screenWidthPix = screenWidthPix
        self.screenHeightPix = screenHeightPix
        self.screenWidthDeg = screenWidthDeg
        self.screenHeightDeg = screenHeightDeg
        
        # calculate degrees per pixel
        self.xDegPerPix = screenWidthDeg / screenWidthPix
        self.yDegPerPix = screenHeightDeg / screenHeightPix
        
        # now apply the transformation, this will add deg fields
        self.applyPix2Deg(self.samples)
        self.applyPix2Deg(self.saccades)

    def applyPix2Deg(self, data):
        # possible fields and info on how to convert them to degrees
        keys = [
            ('x', 'coordX'),
            ('y', 'coordY'),
            ('pupil', 'pupil'),
            ('xLeft', 'coordX'),
            ('yLeft', 'coordY'),
            ('pupilLeft', 'pupil'),
            ('xRight', 'coordX'),
            ('yRight', 'coordY'),
            ('pupilRight', 'pupil'),
            ('xVelLeft', 'velX'),
            ('yVelLeft', 'velY'),
            ('xVelRight', 'velX'),
            ('yVelRight', 'velY'),
            ('startX', 'coordX'),
            ('startY', 'coordY'),
            ('endX', 'coordX'),
            ('endY', 'coordY'),
            ('amplitude','radial'),
            ('peakVel','radial')
        ]
        
        # now look for each key and convert to degrees if found
        for key, transformType in keys:
            if key in data:
                # convert x and y coordiantes to degrees of visual angle relative to center of screen
                if transformType == 'coordX':
                    data[key + 'Deg'] = (data[key] - self.screenWidthPix / 2) * self.xDegPerPix
                elif transformType == 'coordY':
                    data[key + 'Deg'] = (data[key] - self.screenHeightPix / 2) * self.yDegPerPix
                # convert pupil size to degrees of visual angle using geometric mean of x and y degrees per pixel
                elif transformType == 'pupil':
                    data[key + 'Deg'] = data[key] * (self.xDegPerPix * self.yDegPerPix)
                # convert velocity to degrees per second
                elif transformType == 'velX':
                    data[key + 'Deg'] = data[key] * self.xDegPerPix
                elif transformType == 'velY':
                    data[key + 'Deg'] = data[key] * self.yDegPerPix
                # radial transformation
                elif transformType == 'radial':
                    data[key + 'Deg'] = data[key] * (self.xDegPerPix**2 + self.yDegPerPix**2) ** 0.5
                    
    def parseMessages(self):
        """Parse the PGL messages and restructure data into trials."""
        
        # init variables
        self.nTrials = 0
        trialStartTimes = []
        trialEndTimes = []

        # time of messages (in Eyelink clock)
        messageTimesEyelink = self.messages['time']

        # initialize dictionaries with start / stop values
        currentValues = {'trialNum': 1}
        self.startValues = []
        self.stopValues = []
        processingStart = False
        processingStop = False

        # parse PGL messages
        for i, messageText in enumerate(self.messages['text']):
            # convert message text to a structured format
            messageValues = self.parseMessageLine(messageText)
            # check for vaild pgl message
            if messageValues:
                # if it's a start message then collect its values
                if messageValues.get('messageType') == 'start':
                    # append the last values on to stop
                    if processingStop:
                        # set endTrial
                        currentValues['trialNum'] = self.nTrials-1
                        self.stopValues.append(currentValues)
                        currentValues['trialNum'] = self.nTrials
                        processingStop = False
                    processingStart = True
                    currentValues.update(messageValues)

                # if it's a stop message then collect its values
                if messageValues.get('messageType') == 'stop':
                    # append the last values on to start
                    if processingStart:
                        self.startValues.append(currentValues)
                        currentValues = {}
                        processingStart = False
                    processingStop = True
                    currentValues.update(messageValues)
                    
                # if it's a trial start message
                if messageValues.get('messageType') == 'trial':
                    # increment trial count
                    self.nTrials += 1
                    # save trial start time
                    trialStartTimes.append(messageTimesEyelink[i])
        
        # keep the stop values
        if processingStop:
            currentValues['trialNum'] = self.nTrials-1
            self.stopValues.append(currentValues)
        self.convertPix2Deg(self.startValues[0]['screenWidthPix'], self.startValues[0]['screenHeightPix'], self.startValues[0]['screenWidthDeg'], self.startValues[0]['screenHeightDeg'])

        # compute trial end times as next trial start time
        if len(trialStartTimes) > 1:
            trialEndTimes = trialStartTimes[1:]
            trialStartTimes = trialStartTimes[:-1]
            self.nTrials -= 1
        
        # create trial array with relative times
        self.trials = []

        for trialNum in range(self.nTrials):
            # get trial start and end times
            startTime = trialStartTimes[trialNum]
            endTime = trialEndTimes[trialNum]
            
            # find samples within this trial's time window
            trialMask = (self.samples['time'] >= startTime) & (self.samples['time'] < endTime)
            
            # find saccades within this trial's time window
            saccadeMask = (self.saccades['startTime'] >= startTime) & (self.saccades['startTime'] < endTime)
            
            # find blinks within this trial's time window
            blinkMask = (self.blinks['startTime'] >= startTime) & (self.blinks['startTime'] < endTime)
            
            # extract trial data
            trialData = {
                'x': self.samples['x'][trialMask],
                'y': self.samples['y'][trialMask],
                'pupil': self.samples['pupil'][trialMask],
                'time': self.samples['time'][trialMask],
                'saccades': {
                    'eye': self.saccades['eye'][saccadeMask],
                    'startTime': self.saccades['startTime'][saccadeMask]-startTime,
                    'endTime': self.saccades['endTime'][saccadeMask]-startTime,
                    'duration': self.saccades['duration'][saccadeMask],
                    'startX': self.saccades['startXDeg'][saccadeMask],
                    'startY': self.saccades['startYDeg'][saccadeMask],
                    'endX': self.saccades['endXDeg'][saccadeMask],
                    'endY': self.saccades['endYDeg'][saccadeMask],
                    'amplitude': self.saccades['amplitudeDeg'][saccadeMask],
                    'peakVel': self.saccades['peakVelDeg'][saccadeMask]
                },
                'blinks': {
                    'eye': self.blinks['eye'][blinkMask],
                    'startTime': self.blinks['startTime'][blinkMask]-startTime,
                    'endTime': self.blinks['endTime'][blinkMask]-startTime,
                    'duration': self.blinks['duration'][blinkMask],
                }
            }
            
            self.trials.append(trialData)
            
        print(f"(pglEyelinkData) Parsed {self.nTrials} trials from messages.")

    def displayTrials(self):
        """
        Display eye tracking data trial by trial with rainbow colors
        
        Each trial shows:
        - x,y samples as small dots
        - saccade endpoints as larger filled circles with white outline
        """
        # create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # create rainbow color palette
        colors = cm.rainbow(np.linspace(0, 1, self.nTrials))
        
        # plot each trial
        for trialNum in range(self.nTrials):
            trial = self.trials[trialNum]
            color = colors[trialNum]
            
            # plot x,y samples as small dots
            ax.plot(trial['x'], trial['y'], '.', color=color, markersize=2, 
                    label=f'Trial {trialNum}')
            
            # plot saccade endpoints as larger filled circles with white outline
            if len(trial['saccades']['endX']) > 0:
                ax.plot(trial['saccades']['endX'], trial['saccades']['endY'], 
                    'o', color=color, markersize=8, markerfacecolor=color,
                    markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Eye Tracking Data by Trial')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def parseMessageLine(self,line):
        """
        Parse pgl log line and extract messageType and key=value pairs
        
        Parameters
        ----------
        line : str
            Log line like 'pgl: segmentStart segmentNum=0 trialNum=10 timestamp=1152971.4257527918'
        
        Returns
        -------
        dict
            Dictionary with messageType and extracted values, e.g., 
            {'messageType': 'segmentStart', 'segmentNum': 0, 'trialNum': 10, 'timestamp': 1152971.4257527918}
            Returns empty dict if first token is not 'pgl:'
        """
        tokens = line.split()
        
        # Check if first token is 'pgl:'
        if not tokens or tokens[0] != 'pgl:':
            return {}
        
        # Need at least two tokens (pgl: and messageType)
        if len(tokens) < 2:
            return {}
        
        result = {'messageType': tokens[1]}
        
        # Parse remaining tokens for key=value pairs
        for token in tokens[2:]:
            if '=' in token:
                key, value = token.split('=', 1)
                try:
                    result[key] = int(value)
                except ValueError:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
        
        return result
    
    def parse(self):
        """Parse the .asc file and return structured data."""
        try:
            lineCount = 0
            sampleLineCount = 0
            
            with open(self.filename, 'r') as f:
                for line in f:
                    lineCount += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = line.split()
                    if not tokens:
                        continue
                    
                    # Debug: check if we're finding sample lines
                    if tokens[0].isdigit():
                        sampleLineCount += 1
                    
                    self._parseLine(tokens, line)
            
            
        except Exception as e:
            raise IOError(f"Error parsing file {self.filename}: {e}")
            
        # Finalize last block if still recording
        if self.inRecording:
            self._finalizeCurrentBlock()
        
        # Verify that we determined the recording format
        if self.isBinocular is None and len(self.tempSamples) > 0:
            raise ValueError("Could not determine if recording is monocular or binocular. Missing SAMPLES configuration line.")
        
        # Convert lists to arrays (for the entire file)
        self.data['samples'] = self._listsToArrays(self.tempSamples)
        self.data['fixations'] = self._listsToArrays(self.tempFixations)
        self.data['saccades'] = self._listsToArrays(self.tempSaccades)
        self.data['blinks'] = self._listsToArrays(self.tempBlinks)
        self.data['messages'] = self._listsToArrays(self.tempMessages) 
        
        return self.data
    
    def _parseLine(self, tokens, rawLine):
        """Route line to appropriate parser based on first token."""
        firstToken = tokens[0]
        
        if firstToken.startswith('**'):
            self.data['metadata'].append(rawLine)
            
        elif firstToken == 'START':
            self._handleStart(tokens)
            
        elif firstToken == 'END':
            self._handleEnd(tokens)
            
        elif firstToken == 'SAMPLES':
            self._parseSamplesConfig(tokens)
            
        elif firstToken == 'EVENTS':
            self._parseEventsConfig(tokens)
            
        elif firstToken == 'MSG':
            self.tempMessages.append(self._parseMsgLine(tokens))  
            
        elif firstToken == 'EFIX':
            if self.inRecording:
                self.tempFixations.append(self._parseFixationLine(tokens))
            
        elif firstToken == 'ESACC':
            if self.inRecording:
                self.tempSaccades.append(self._parseSaccadeLine(tokens))
            
        elif firstToken == 'EBLINK':
            if self.inRecording:
                self.tempBlinks.append(self._parseBlinkLine(tokens))
            
        elif firstToken in ['SFIX', 'SSACC', 'SBLINK']:
            pass
            
        elif firstToken.isdigit():
            if self.inRecording:
                # Check if we know the format before parsing samples
                if self.isBinocular is None:
                    raise ValueError(f"Encountered sample data before SAMPLES configuration line at: {rawLine}")
                self.tempSamples.append(self._parseSampleLine(tokens))
            
        elif firstToken in ['PRESCALER', 'VPRESCALER', 'PUPIL']:
            # Recording configuration lines
            self.data['metadata'].append(rawLine)
            
        else:
            self.data['metadata'].append(rawLine)
    
    def _handleStart(self, tokens):
        """Handle START line: START timestamp eye SAMPLES EVENTS"""
        self.inRecording = True
        self.currentBlock = {
            'startTime': int(tokens[1]) if len(tokens) > 1 else None,
            'eye': tokens[2] if len(tokens) > 2 else None,
            'startIndex': len(self.tempSamples)
        }
    
    def _handleEnd(self, tokens):
        """Handle END line: END timestamp eye SAMPLES EVENTS RES..."""
        if self.inRecording:
            self._finalizeCurrentBlock()
            if self.currentBlock:
                self.currentBlock['endTime'] = int(tokens[1]) if len(tokens) > 1 else None
                self.currentBlock['endIndex'] = len(self.tempSamples)
                self.data['recordingBlocks'].append(self.currentBlock)
            self.currentBlock = None
            self.inRecording = False
    
    def _finalizeCurrentBlock(self):
        """Finalize the current recording block."""
        if self.currentBlock:
            self.currentBlock['endIndex'] = len(self.tempSamples)
    
    def _parseSamplesConfig(self, tokens):
        """Parse SAMPLES configuration line to determine data format."""
        config = {
            'dataType': tokens[1] if len(tokens) > 1 else None,  # GAZE, HREF, etc.
            'eyes': [],
            'rate': None,
            'tracking': None,
            'filter': None
        }
        
        # Find which eyes are recorded
        eyesFound = []
        if 'LEFT' in tokens:
            eyesFound.append('LEFT')
        if 'RIGHT' in tokens:
            eyesFound.append('RIGHT')
        
        config['eyes'] = eyesFound
        
        # Set binocular flag - must have exactly 2 eyes
        if len(eyesFound) == 2:
            self.isBinocular = True
        elif len(eyesFound) == 1:
            self.isBinocular = False
        else:
            raise ValueError(f"Could not determine eye configuration from SAMPLES line: {' '.join(tokens)}")
        
        # Get sample rate
        if 'RATE' in tokens:
            rateIdx = tokens.index('RATE')
            if rateIdx + 1 < len(tokens):
                config['rate'] = float(tokens[rateIdx + 1])
        
        # Get tracking mode
        if 'TRACKING' in tokens:
            trackIdx = tokens.index('TRACKING')
            if trackIdx + 1 < len(tokens):
                config['tracking'] = tokens[trackIdx + 1]
        
        # Get filter level
        if 'FILTER' in tokens:
            filterIdx = tokens.index('FILTER')
            if filterIdx + 1 < len(tokens):
                config['filter'] = int(tokens[filterIdx + 1])
        
        self.data['samplesConfig'] = config
    
    def _parseEventsConfig(self, tokens):
        """Parse EVENTS configuration line."""
        config = {
            'dataType': tokens[1] if len(tokens) > 1 else None,
            'eyes': [],
            'rate': None,
            'tracking': None,
            'filter': None
        }
        
        # Find which eyes are recorded
        if 'LEFT' in tokens:
            config['eyes'].append('LEFT')
        if 'RIGHT' in tokens:
            config['eyes'].append('RIGHT')
        
        # Get event rate
        if 'RATE' in tokens:
            rateIdx = tokens.index('RATE')
            if rateIdx + 1 < len(tokens):
                config['rate'] = float(tokens[rateIdx + 1])
        
        # Get tracking mode
        if 'TRACKING' in tokens:
            trackIdx = tokens.index('TRACKING')
            if trackIdx + 1 < len(tokens):
                config['tracking'] = tokens[trackIdx + 1]
        
        # Get filter level
        if 'FILTER' in tokens:
            filterIdx = tokens.index('FILTER')
            if filterIdx + 1 < len(tokens):
                config['filter'] = int(tokens[filterIdx + 1])
        
        self.data['eventsConfig'] = config
    
    def _parseSampleLine(self, tokens):
        """Parse a sample line into a dict based on monocular/binocular format."""        # Remove trailing '...' if present
        # Remove trailing '...' if present (must check before counting tokens)
        if len(tokens) > 0 and tokens[-1] == '...':
            tokens = tokens[:-1]
    
        
        numTokens = len(tokens)
        sample = {}
        
        # Parse timestamp (always first)
        sample['time'] = int(tokens[0])
        
        if self.isBinocular:
            # Binocular formats
            if numTokens == 7:
                # Binocular: time xpl ypl psl xpr ypr psr
                sample['xLeft'] = self.toFloat(tokens[1])
                sample['yLeft'] = self.toFloat(tokens[2])
                sample['pupilLeft'] = self.toFloat(tokens[3])
                sample['xRight'] = self.toFloat(tokens[4])
                sample['yRight'] = self.toFloat(tokens[5])
                sample['pupilRight'] = self.toFloat(tokens[6])
                
            elif numTokens == 11:
                # Binocular with velocity: time xpl ypl psl xpr ypr psr xvl yvl xvr yvr
                sample['xLeft'] = self.toFloat(tokens[1])
                sample['yLeft'] = self.toFloat(tokens[2])
                sample['pupilLeft'] = self.toFloat(tokens[3])
                sample['xRight'] = self.toFloat(tokens[4])
                sample['yRight'] = self.toFloat(tokens[5])
                sample['pupilRight'] = self.toFloat(tokens[6])
                sample['xVelLeft'] = self.toFloat(tokens[7])
                sample['yVelLeft'] = self.toFloat(tokens[8])
                sample['xVelRight'] = self.toFloat(tokens[9])
                sample['yVelRight'] = self.toFloat(tokens[10])
                self.hasVelocity = True
                
            elif numTokens == 9:
                # Binocular with resolution: time xpl ypl psl xpr ypr psr xr yr
                sample['xLeft'] = self.toFloat(tokens[1])
                sample['yLeft'] = self.toFloat(tokens[2])
                sample['pupilLeft'] = self.toFloat(tokens[3])
                sample['xRight'] = self.toFloat(tokens[4])
                sample['yRight'] = self.toFloat(tokens[5])
                sample['pupilRight'] = self.toFloat(tokens[6])
                sample['xRes'] = self.toFloat(tokens[7])
                sample['yRes'] = self.toFloat(tokens[8])
                self.hasResolution = True
                
            elif numTokens == 13:
                # Binocular with velocity and resolution: time xpl ypl psl xpr ypr psr xvl yvl xvr yvr xr yr
                sample['xLeft'] = self.toFloat(tokens[1])
                sample['yLeft'] = self.toFloat(tokens[2])
                sample['pupilLeft'] = self.toFloat(tokens[3])
                sample['xRight'] = self.toFloat(tokens[4])
                sample['yRight'] = self.toFloat(tokens[5])
                sample['pupilRight'] = self.toFloat(tokens[6])
                sample['xVelLeft'] = self.toFloat(tokens[7])
                sample['yVelLeft'] = self.toFloat(tokens[8])
                sample['xVelRight'] = self.toFloat(tokens[9])
                sample['yVelRight'] = self.toFloat(tokens[10])
                sample['xRes'] = self.toFloat(tokens[11])
                sample['yRes'] = self.toFloat(tokens[12])
                self.hasVelocity = True
                self.hasResolution = True
            else:
                raise ValueError(f"Unexpected number of tokens ({numTokens}) in binocular sample line: {' '.join(tokens)}")
                
        else:
            # Monocular formats
            if numTokens == 4:
                # Monocular: time xp yp ps
                sample['x'] = self.toFloat(tokens[1])
                sample['y'] = self.toFloat(tokens[2])
                sample['pupil'] = self.toFloat(tokens[3])
                
            elif numTokens == 6:
                # Monocular with velocity: time xp yp ps xv yv
                # OR Monocular with resolution: time xp yp ps xr yr
                # Ambiguous - default to velocity (more common)
                sample['x'] = self.toFloat(tokens[1])
                sample['y'] = self.toFloat(tokens[2])
                sample['pupil'] = self.toFloat(tokens[3])
                sample['xVel'] = self.toFloat(tokens[4])
                sample['yVel'] = self.toFloat(tokens[5])
                self.hasVelocity = True
                
            elif numTokens == 8:
                # Monocular with velocity and resolution: time xp yp ps xv yv xr yr
                sample['x'] = self.toFloat(tokens[1])
                sample['y'] = self.toFloat(tokens[2])
                sample['pupil'] = self.toFloat(tokens[3])
                sample['xVel'] = self.toFloat(tokens[4])
                sample['yVel'] = self.toFloat(tokens[5])
                sample['xRes'] = self.toFloat(tokens[6])
                sample['yRes'] = self.toFloat(tokens[7])
                self.hasVelocity = True
                self.hasResolution = True
            else:
                raise ValueError(f"Unexpected number of tokens ({numTokens}) in monocular sample line: {' '.join(tokens)}")
        
        return sample
    
    def toInt(self, value):
        """Convert a value to int. Return NaN if conversion fails."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return np.nan

    def toFloat(self, value):
        """Convert a value to float. Return NaN for missing or invalid values."""
        if value in (None, ".", ""):
            return float("nan")
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")
    
    def _parseFixationLine(self, tokens):
        """Parse EFIX line into a dict."""
        # Format: EFIX eye startTime endTime duration avgX avgY avgPupil [xRes yRes]
        if len(tokens) < 8:
            raise ValueError(f"EFIX line has too few tokens: {' '.join(tokens)}")
        
        fixation = {}
        
        fixation['eye'] = tokens[1]  # L or R
        
        try:
            fixation['startTime'] = self.toInt(tokens[2])
            fixation['endTime'] = self.toInt(tokens[3])
            fixation['duration'] = self.toInt(tokens[4])
            fixation['avgX'] = self.toFloat(tokens[5])
            fixation['avgY'] = self.toFloat(tokens[6])
            fixation['avgPupil'] = self.toFloat(tokens[7])
            
            # Check if resolution data is included
            if len(tokens) == 10:
                # EFIX with resolution: eye stime etime dur axp ayp aps xr yr
                fixation['xRes'] = self.toFloat(tokens[8])
                fixation['yRes'] = self.toFloat(tokens[9])
            elif len(tokens) == 8:
                # EFIX without resolution: eye stime etime dur axp ayp aps
                pass
            else:
                raise ValueError(f"Unexpected number of tokens ({len(tokens)}) in EFIX line")
                
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing EFIX line: {' '.join(tokens)}, error: {e}")
        
        return fixation    
    
    def _parseSaccadeLine(self, tokens):
        """Parse ESACC line into a dict."""
        # Format: ESACC eye startTime endTime duration startX startY endX endY amplitude peakVel [xRes yRes]
        if len(tokens) < 11:
            raise ValueError(f"ESACC line has too few tokens: {' '.join(tokens)}")
        
        saccade = {}
        
        saccade['eye'] = tokens[1]  # L or R
        
        try:
            saccade['startTime'] = self.toInt(tokens[2])
            saccade['endTime'] = self.toInt(tokens[3])
            saccade['duration'] = self.toInt(tokens[4])
            saccade['startX'] = self.toFloat(tokens[5])
            saccade['startY'] = self.toFloat(tokens[6])
            saccade['endX'] = self.toFloat(tokens[7])
            saccade['endY'] = self.toFloat(tokens[8])
            saccade['amplitude'] = self.toFloat(tokens[9])
            saccade['peakVel'] = self.toFloat(tokens[10])
            
            # Check if resolution data is included
            if len(tokens) == 13:
                # ESACC with resolution: eye stime etime dur sxp syp exp eyp ampl pv xr yr
                saccade['xRes'] = self.toFloat(tokens[11])
                saccade['yRes'] = self.toFloat(tokens[12])
            elif len(tokens) == 11:
                # ESACC without resolution: eye stime etime dur sxp syp exp eyp ampl pv
                pass
            else:
                raise ValueError(f"Unexpected number of tokens ({len(tokens)}) in ESACC line")
                
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing ESACC line: {' '.join(tokens)}, error: {e}")
        
        return saccade    

    def _parseBlinkLine(self, tokens):
        """Parse EBLINK line into a dict."""
        # Format: EBLINK eye startTime endTime duration
        if len(tokens) < 5:
            raise ValueError(f"EBLINK line has too few tokens: {' '.join(tokens)}")
        
        blink = {}
        
        blink['eye'] = tokens[1]  # L or R
        
        try:
            blink['startTime'] = self.toInt(tokens[2])
            blink['endTime'] = self.toInt(tokens[3])
            blink['duration'] = self.toInt(tokens[4])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing EBLINK line: {' '.join(tokens)}, error: {e}")
        
        return blink

    def _parseMsgLine(self, tokens):
        """Parse MSG line into a dict."""
        # Format: MSG timestamp message_text...
        if len(tokens) < 2:
            raise ValueError(f"MSG line has too few tokens: {' '.join(tokens)}")
        
        msg = {}
        
        # Parse timestamp
        try:
            msg['time'] = self.toInt(tokens[1])
        except ValueError:
            raise ValueError(f"Could not parse timestamp from MSG line: {tokens[1]}")
        
        # Join remaining tokens as the message text
        if len(tokens) > 2:
            msg['text'] = ' '.join(tokens[2:])
        else:
            msg['text'] = ''
        
        return msg    
    def _listsToArrays(self, listOfDicts):
        """Convert list of dicts to dict of numpy arrays."""
        # Handle empty list
        if not listOfDicts:
            return {}
        
        # Filter out None values (in case any snuck through)
        listOfDicts = [d for d in listOfDicts if d is not None]
        
        # Check again after filtering
        if not listOfDicts:
            return {}
        
        result = {}
        keys = listOfDicts[0].keys()
        for key in keys:
            result[key] = np.array([d[key] for d in listOfDicts])
        
        return result
    @property
    def samples(self):
        return self.data.get('samples', {})

    @property
    def messages(self):
        return self.data.get('messages', {})

    @property
    def fixations(self):
        return self.data.get('fixations', {})

    @property
    def saccades(self):
        return self.data.get('saccades', {})

    @property
    def blinks(self):
        return self.data.get('blinks', {})

    
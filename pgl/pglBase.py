################################################################
#   filename: pglBase.py
#    purpose: Base module for the pgl psychophysics and experiment library
#         by: JLG
#       date: July 9, 2025
################################################################

#############
# Import modules
#############
import platform, subprocess, pprint, random, string, os, pprint
import numpy as np
from ._socket import _socket
from . import _displayInfo
from types import SimpleNamespace

#############
# Main class
#############
class pglBase:
    ################################################################
    # Variables
    ################################################################
    _verbose = 1 # verbosity level, 0 = silent, 1 = normal, 2 = verbose
    macOSversion = None
    hardwareInfo = None
    gpuInfo = None
    screenX = SimpleNamespace(pix = 0)
    screenY = SimpleNamespace(pix = 0)
    screenWidth = SimpleNamespace(pix = 0, cm = 0.0, deg = 0.0)
    screenHeight = SimpleNamespace(pix = 0, cm = 0.0, deg = 0.0)
    distanceToScreen = SimpleNamespace(cm = 0.0)

    ################################################################
    # Init Function
    ################################################################
    def __init__(self):
        # check os
        if not self.checkOS():
            raise Exception("(pglBase) Unsupported OS")
                
        # Init verbose level to normal
        self.verbose = 1

        # print what we are doing
        if self.verbose > 0: print("(pglBase) Main library instance created")
    
    ################################################################
    # Delete Function
    ################################################################
    def __del__(self):

        self.close()  # close the socket if it exists
        # print what we are doing
        if self.verbose > 0: print("(pglBase) Main library closed")
    
    ################################################################
    # Verbose property
    ################################################################
    @property
    def verbose(self):
        # Get the current verbosity level.
        return self._verbose
    @verbose.setter
    def verbose(self, level):
        # Set the verbosity level.
        if level < 0 or level > 2:
            print("(pglBase) Verbosity level must be between 0 and 2")
        else:
            # set the verbosity level
            self._verbose = level
            # tell the displayInfo c-code library to set the verbosity level
            _displayInfo.setVerbose(level)
            # if we have a socket, set the verbosity level there too
            if hasattr(self, 's') and self.s:
                self.s.verbose = level

        # Print the new verbosity level
        if self._verbose > 0: print(f"(pglBase) Verbosity level set to {self._verbose}")

    ################################################################
    # Open a screen
    ################################################################
    def open(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        # Print what we are doing
        if self.verbose > 0: print(f"(pglBase:open) Opening screen {whichScreen} with dimensions {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")
        # for now hard-code these
        self.metalAppName = "/Users/justin/proj/mgl/metal/binary/stable/mglMetal.app"
        self.metalSocketPath = "/Users/justin/Library/Containers/gru.mglMetal/Data"
        
        # get a randomized socket name
        self.metalSocketName = "pglMetal.socket."+''.join(random.choices(string.ascii_letters + string.digits, k=10))
        
        # create the socket path
        socketName = os.path.join(self.metalSocketPath, self.metalSocketName)

        # start up mglMetal application
        if not os.path.exists(self.metalAppName):
            print(f"(pglBase:open) Error: mglMetal application not found at {self.metalAppName}")
            return False
        else:
            if self.verbose > 0: 
                print(f"(pglBase:open) Starting mglMetal application: {self.metalAppName}")
                print(f"(pglBase:open) Using socket with address: {socketName}")
            # start the mglMetal application
            try:
                result = subprocess.run([
                    "open", "-g", "-n", self.metalAppName,
                    "--args", "-mglConnectionAddress", socketName
                    ], check=True)
            except Exception as e:
                print(f"(pglBase:open) Error starting mglMetal application: {e}")
                return False
        
        # now try to connect to the socket
        self.s = _socket(socketName)

        # and parse command types
        self.s.parseCommandTypes()

        # get the window location
        self.getWindowFrameInDisplay()

        self.clearScreen([0.4, 0.2, 0.5])
        self.flush()
    def setWindowFrameInDisplay(self, whichScreen, screenX, screenY, screenWidth, screenHeight):
        """
        Set the window frame location and size
 
        Args:
            whichScreen (int): The screen number to set the window frame for.
            screenX (int): The x-coordinate of the window frame.
            screenY (int): The y-coordinate of the window frame.
            screenWidth (int): The width of the window frame.
            screenHeight (int): The height of the window frame.
        """
        self.s.writeCommand("mglSetWindowFrameInDisplay")
        self.s.write(np.uint32(whichScreen))
        self.s.write(np.uint32(screenX))
        self.s.write(np.uint32(screenY))
        self.s.write(np.uint32(screenWidth))
        self.s.write(np.uint32(screenHeight))
        commandResults = self.s.readCommandResults()
        # save pixel dimensions
        self.screenWidth.pix = screenWidth
        self.screenHeight.pix = screenHeight
    
    def getWindowFrameInDisplay(self):
        """
        Get the current window frame location and size.

        Returns:
            dict: A dictionary containing the window frame information.
            - 'whichScreen' (int): The screen number where the window frame is located.
            - 'screenX' (int): The x-coordinate of the window frame in pixels.
            - 'screenY' (int): The y-coordinate of the window frame in pixels.
            - 'screenWidth' (int): The width of the window frame in pixels.
            - 'screenHeight' (int): The height of the window frame in pixels.
        """
        self.s.writeCommand("mglGetWindowFrameInDisplay")
        ack = self.s.readAck()
        responseIncoming = self.s.read(np.double)
        if responseIncoming < 0:
            print(f"(pglBase:getWindowFrameInDisplay) ❌ Error getting window frame size")
            windowLocation = {}
        else:
            windowLocation = {'whichScreen': self.s.read(np.uint32),
                              'screenX': self.s.read(np.uint32),
                              'screenY': self.s.read(np.uint32),
                              'screenWidth': self.s.read(np.uint32),
                              'screenHeight': self.s.read(np.uint32)} 
        commandResults = self.s.readCommandResults(ack)

        # update the stored values
        self.whichScreen = windowLocation.get('whichScreen', 0)
        self.screenX.pix = windowLocation.get('screenX', 0)
        self.screenY.pix = windowLocation.get('screenY', 0)
        self.screenWidth.pix = windowLocation.get('screenWidth', 0)
        self.screenHeight.pix = windowLocation.get('screenHeight', 0)

        return windowLocation

    def fullscreen(self, goFullScreen=True):
        """
        Set the window to fullscreen mode.

        Args:
            goFullScreen (bool): If True, set the window to fullscreen. If False, exit fullscreen.
        """
        if goFullScreen:
            self.s.writeCommand("mglFullscreen")
        else:
            self.s.writeCommand("mglWindowed")
        commandResults = self.s.readCommandResults()
        if commandResults.get('success',False) is False:
            print("(pglBase:fullscreen) ❌ Error setting fullscreen mode")
            return False
        return True

    def close(self):
        """
        Close the connection to the mglMetal application and clean up.

        This function sends a close command to the mglMetal application, waits for a response,
        and then closes the socket connection.

        Returns:
            bool: True if the connection was closed successfully, False otherwise.
        """
        # Print what we are doing
        if self.verbose > 0: print("(pglBase:close) Closing connection to mglMetal application")

        # Check if the socket is connected
        if not self.s:
            print("(pglBase:close) ❌ Not connected to socket")
            return False
        
        # get the PID of the mglMetal application
        pid = self.s.getPID()
        if pid is None:
            print("(pglBase:close) ❌ Could not find PID of mglMetal application")
            return False
        
        # close the application
        if self.verbose > 0: print(f"(pglBase:close) Closing mglMetal application with PID {pid}")
        try:
            subprocess.run(["kill", "-9", str(pid)], check=True)
            if self.verbose > 0: print(f"(pglBase:close) mglMetal application with PID: {pid} was killed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"(pglBase:close) Error killing mglMetal application with PID: {pid} : {e}")

        # Close the socket
        self.s.close()
        
        return True
    def printCommandResults(self, commandResults, relativeToTime=None):
        """
        Print the results of a command.

        Args:
            commandResults (dict): The command results to print.
        """
        print(f"(pglBase:printCommandResults) Command results:")
        if relativeToTime is None:
            relativeToTime = commandResults['ack']
            print(f"(pglBase:printCommandResults) Ack: {commandResults['ack']:.3f} (absolute time in seconds)")
        else:
            print(f"(pglBase:printCommandResults) Ack: {(commandResults['ack'] - relativeToTime)*1000.0:.3f} ms (relative to {relativeToTime})")
        print(f"(pglBase:printCommandResults) Command Code: {commandResults['commandCode']}")
        print(f"(pglBase:printCommandResults) Success: {commandResults['success']}")
        if commandResults['vertexStart'] != 0:
            print(f"(pglBase:printCommandResults) Vertex Start: {(commandResults['vertexStart'] - relativeToTime)*1000.0:.3f} ms")
        if commandResults['vertexEnd'] != 0:
            print(f"(pglBase:printCommandResults) Vertex End: {(commandResults['vertexEnd'] - relativeToTime)*1000.0:.3f} ms")
        if commandResults['fragmentStart'] != 0:
            print(f"(pglBase:printCommandResults) Fragment Start: {(commandResults['fragmentStart'] - relativeToTime)*1000.0:.3f} ms")
        if commandResults['fragmentEnd'] != 0:
            print(f"(pglBase:printCommandResults) Fragment End: {(commandResults['fragmentEnd'] - relativeToTime)*1000.0:.3f} ms")
        if commandResults['drawableAcquired'] != 0:
            print(f"(pglBase:printCommandResults) Drawable Acquired: {(commandResults['drawableAcquired'] - relativeToTime)*1000.0:.3f} ms")
        if commandResults['drawablePresented'] != 0:
            print(f"(pglBase:printCommandResults) Drawable Presented: {(commandResults['drawablePresented'] - relativeToTime)*1000.0:.3f} ms")
        if commandResults['processedTime'] != 0:
            print(f"(pglBase:printCommandResults) Processed Time: {(commandResults['processedTime'] - relativeToTime)*1000.0:.3f} ms")


    def clearScreen(self, color):
        """
        Clear the screen with a specified color.

        Args:
            color (list or tuple): RGB color values as a list or tuple of three floats in the range [0, 1].

        Returns:
            bool: True if the screen was cleared successfully, False otherwise.
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:clearScreen) Clearing screen with color {color}")

        # Check if the socket is connected
        if not self.s:
            print("(pgl:clearScreen) ❌ Not connected to socket")
            return False
        
        # Send the clear command
        self.s.writeCommand("mglSetClearColor")
        # send the color data
        self.s.write(np.array(color, dtype=np.float32))
        # Read the command results
        commandResults = self.s.readCommandResults()
        if self.verbose > 0: self.printCommandResults(commandResults)
        
    def flush(self):
        """        
        Flush the drawing commands to the screen.

        This function sends a flush command to the mglMetal application to ensure that all
        drawing commands are executed.

        Args:
            None

        Returns:
            bool: True if the flush command was sent successfully, False otherwise.
        """
        self.s.writeCommand("mglFlush")
        commandResults = self.s.readCommandResults()
        if self.verbose > 0: self.printCommandResults(commandResults)
        
        # success
        return True
    
    ################################################################
    # Check OS compatibility
    ################################################################
    def checkOS(self):
        """
        Check if the current operating system is macOS and retrieves system information.

        This method verifies if the code is running on macOS (Darwin). If so, it obtains the macOS
        version and hardware information by calling the system profiler. The hardware info is parsed
        into a dictionary for easy access.

        Returns:
            bool: True if running on macOS and information was retrieved (or attempted),
                  False if running on a non-macOS system.

        Verbose Mode:
            Module-level 'verbose' (pgl.verbose) can be set to display:
                - 1 MacOS version
                - 2 Hardware information

        Author:
            JLG

        Date:
            July 9, 2025
        """
        if platform.system() == "Darwin":
            # get version
            self.macOSversion = platform.mac_ver()
            # get hardware and gpu info
            try:
                # hardware info
                hardwareInfo = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # parse into a dict for easier access
                lines = hardwareInfo.stdout.splitlines()
                self.hardwareInfo = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        # add to dict
                        self.hardwareInfo[key] = value
                # gpu info
                gpuInfo = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # parse into a dict for easier access
                self.gpuInfo = parseGPUInfo(gpuInfo.stdout) 

            except subprocess.CalledProcessError as e:
                self.hardwareInfo["error"] = f"Error retrieving hardware info: {e}"
                self.gpuInfo["error"] = f"Error retrieving gpu info: {e}"
            # Print the macOS version and hardware info
            if self.verbose > 0:
                modelName = self.hardwareInfo.get("Model Name", "Unknown Model").strip()
                modelID = self.hardwareInfo.get("Model Identifier", "Unknown Identifier").strip()
                osVersion = self.macOSversion[0].strip()
                print(f"(pgl:checkOS) Running on {modelName} ({modelID}) with macOS version: {osVersion}")
            if self.verbose > 0: print("(pgl:checkOS)",
                                        self.hardwareInfo.get("Chip", "Unknown "),
                                        "Cores:",
                                        self.hardwareInfo.get("Total Number of Cores", "Unknown "),
                                        "Memory:",
                                        self.hardwareInfo.get("Memory", "Unknown "))
            # Print GPU info
            if self.verbose > 0:
                for gpuName, gpuInfo in self.gpuInfo.items():
                    gpuChipset = gpuInfo.get("Chipset Model", "Unknown")
                    gpuBus = gpuInfo.get("Bus", "Unknown")
                    gpuMetalSupport = gpuInfo.get("Metal Support", "Unknown metal")
                    gpuNumCores = gpuInfo.get("Total Number of Cores", "Unknown")
                    print(f"(pgl:checkOS) GPU: {gpuChipset} ({gpuBus}) {gpuNumCores} cores, {gpuMetalSupport} support" )
                    displays = gpuInfo.get("Displays", [])
                    for display in displays:
                        displayName = display.get("DisplayName", "Unnamed")
                        displayResolution = display.get("Resolution", "Unknown resolution")
                        displayType = display.get("Display Type", "Unknown type")
                        if display.get("Main Display", "No") == "Yes":
                            print(f"(pgl:checkOS)   {displayName} [Main Display]: {displayResolution} ({displayType})")
                        else:
                            print(f"(pgl:checkOS)   {displayName}: {displayResolution} ({displayType})")

            if self.verbose > 1:
                # print detailed information
                print("(pgl:checkOS) Hardware info")
                pprint.pprint(self.hardwareInfo)
                print("(pgl:checkOS) GPU info")
                pprint.pprint(self.gpuInfo)
            return True
        else:
            # not macOS
            print("(pgl:checkOS) PGL is only supported on macOS")
            return False
    
    ################################################################
    # Get the display resolution
    ################################################################
    def getResolution(self, whichScreen):
        """
        Get the resolution and display settings for a given screen.

        This function retrieves the width, height, refresh rate, and bit depth of the specified
        display using the underlying `_displayInfo` compiled extension. 

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays.

        Returns:
            tuple[int, int, int, int]: A 4-tuple containing:
                - width (int): Screen width in pixels.
                - height (int): Screen height in pixels.
                - refresh_rate (int): Refresh rate in Hz.
                - bit_depth (int): Color depth in bits per pixel.

        Raises:
            None. Errors are signaled by a return value of (-1, -1, -1, -1)

        Verbose Mode:
            Module-level 'verbose' (pgl.verbose) can be set to display:
                - 1 Screen resolution, refresh rate and bit depth
                - 2 all available modes for the display
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:getResolution) Getting resolution for screen {whichScreen}")

        # Call the C function to get the display info
        return _displayInfo.getResolution(whichScreen)
    
    ################################################################
    # Set the display resolution
    ################################################################
    def setResolution(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        """
        Set the resolution and display settings for a given screen.

        This function sets the width, height, refresh rate, and bit depth of the specified
        display using the underlying `_displayInfo` compiled extension. 

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays.
            screenWidth (int): Desired screen width in pixels.
            screenHeight (int): Desired screen height in pixels.
            screenRefreshRate (int): Desired refresh rate in Hz.
            screenColorDepth (int): Desired color depth in bits per pixel (e.g., 32 for 32-bit color).

        Returns:
            None: The function does not return a value, but it will print the new resolution if successful.

        Author:
            JLG

        Date:
            July 9, 2025
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:setResolution) Setting resolution for screen {whichScreen} to {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")

        # Call the C function to set the display info
        if _displayInfo.setResolution(whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
            # print what resolution the display was set to
            self.getResolution(whichScreen)
    ################################################################
    # Get the number of displays and the default display
    ################################################################
    def getNumDisplaysAndDefault(self):
        """
        Get the number of displays and the default display index.

        This function retrieves the total number of active displays and identifies the
        default display (usually the primary display) using the underlying `_displayInfo`
        compiled extension.

        Args:
            None

        Returns:
            tuple[int, int]: A 2-tuple containing:
                - numDisplays (int): The total number of active displays.
                - defaultDisplay (int): The index of the default display (0 = primary).

        Author:
            JLG

        Date:
            July 9, 2025
        """
        # Print what we are doing
        if self.verbose > 1: print("(pgl:getNumDisplaysAndDefault) Getting number of displays and default display")

        # Call the C function to get the number of displays and the default display
        return _displayInfo.getNumDisplaysAndDefault()

################
# parseGPUInfo #
################
def parseGPUInfo(text):
    """
    Parse the output of `system_profiler SPDisplaysDataType` into a structured dictionary.

    Supports multiple GPUs, nested display info

    Args:
        text (str): Raw text output from system_profiler.

    Returns:
        dict: A dictionary mapping GPU names to their attributes and associated displays.
    """
    try:
        lines = text.splitlines()
        result = {}
        currentGpu = None
        currentDisplay = None
        displayList = []
        inDisplaysSection = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            indent = len(line) - len(line.lstrip())

            # Detect top-level GPU name
            if indent == 4 and line.endswith(":") and not stripped.startswith("Displays:"):
                currentGpu = stripped.rstrip(":")
                result[currentGpu] = {}
                displayList = []
                inDisplaysSection = False

            # Start of Displays section
            elif indent == 6 and stripped == "Displays:":
                inDisplaysSection = True
                result[currentGpu]["Displays"] = displayList

            # GPU metadata
            elif indent == 6 and ":" in stripped and not inDisplaysSection:
                key, value = map(str.strip, stripped.split(":", 1))
                result[currentGpu][key] = value

            # New display name
            elif indent == 8 and stripped.endswith(":") and inDisplaysSection:
                currentDisplay = {"DisplayName": stripped.rstrip(":")}
                displayList.append(currentDisplay)

            # Display metadata
            elif indent >= 10 and ":" in stripped and currentDisplay is not None:
                key, value = map(str.strip, stripped.split(":", 1))
                currentDisplay[key] = value

        return result

    except Exception as e:
        print(f"(parseGpuDisplayInfo) Warning: Parsing failed with error: {e}")
        return {}

################################################################
# # pgl.py: Main module for the pgl psychophysics and experiment library
################################################################

#############
# Import modules
#############
import platform, subprocess, pprint

from ._socket import _socket
from . import _displayInfo

#############
# Main class
#############
class pgl:
    ################################################################
    # Variables
    ################################################################
    _verbose = 1 # verbosity level, 0 = silent, 1 = normal, 2 = verbose
    macOSversion = None
    hardwareInfo = None
    gpuInfo = None

    ################################################################
    # Init Function
    ################################################################
    def __init__(self):
        # check os
        if not self.checkOS():
            raise Exception("(pgl) Unsupported OS")
                
        # start socket library
        self.s = _socket()

        # Init verbose level to normal
        self.verbose = 1

        # print what we are doing
        if self.verbose > 0: print("(pgl) Main library instance created")
    
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
            print("(pgl) Verbosity level must be between 0 and 2")
        else:
            # set the verbosity level
            self._verbose = level
            # tell the displayInfo c-code library to set the verbosity level
            _displayInfo.setVerbose(level)

        # Print the new verbosity level
        if self._verbose > 0: print(f"(pgl) Verbosity level set to {self._verbose}")

    ################################################################
    # Open a screen
    ################################################################
    def open(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        # Print what we are doing
        if self.verbose > 0: print(f"(pgl:open) Opening screen {whichScreen} with dimensions {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")
        # Here you would add the code to actually open the screen using a graphics library
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

        Author:
            JLG

        Date:
            July 9, 2025
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:getResolution) Getting resolution for screen {whichScreen}")

        # Call the C function to get the display info
        return _displayInfo.getResolution(whichScreen)
    
    ################################################################
    # Set the display resolution
    ################################################################
    def setResolution(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
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
        # Print what we are doing
        if self.verbose > 1: print("(pgl:getNumDisplaysAndDefault) Getting number of displays and default display")
        
        # Call the C function to get the number of displays and the default display
        return _displayInfo.getNumDisplaysAndDefault()     


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
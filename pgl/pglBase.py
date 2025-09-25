################################################################
#   filename: pglBase.py
#    purpose: Base module for the pgl psychophysics and experiment library
#         by: JLG
#       date: July 9, 2025
################################################################

#############
# Import modules
#############
from datetime import datetime
import glob
import inspect
import platform, subprocess, random, string, os
from pprint import pprint
import sys
import numpy as np
from . import _pglComm as pglComm
from . import _resolution
from types import SimpleNamespace
import signal
import glob
import psutil

#############
# Main class
#############
class pglBase:
    ################################################################
    # Variables
    ################################################################
    _verbose = 1 # verbosity level, 0 = silent, 1 = normal, 2 = verbose
    macOSversion = None
    cpuInfo = None
    gpuInfo = None
    commandResults = None
    s = None  # socket connection to mglMetal application
    screenX = SimpleNamespace(pix = 0)
    screenY = SimpleNamespace(pix = 0)
    screenWidth = SimpleNamespace(pix = 0, cm = 0.0, deg = 0.0)
    screenHeight = SimpleNamespace(pix = 0, cm = 0.0, deg = 0.0)
    distanceToScreen = SimpleNamespace(cm = 0.0)

    ################################################################
    # Init Function
    ################################################################
    def __init__(self):
        
        self.printHeader("pglBase: init")
        
        # check os
        if not self.checkOS():
            raise Exception("(pglBase) Unsupported OS")
                
        # get some directories
        self.homeDir = os.path.expanduser("~")
        pglDir = inspect.getfile(self.__class__)
        self.pglDir = os.path.dirname(os.path.dirname(pglDir))
        
        # get socket path
        self.metalSocketPath = os.path.join(self.homeDir, "Library/Containers/gru.mglMetal/Data")

        # print what we are doing
        if self.verbose > 0: 
            print("(pglBase) Main library instance created")
            self.printHeader()
    
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
            _resolution.setVerbose(level)
            # if we have a socket, set the verbosity level there too
            if hasattr(self, 's') and self.s:
                self.s.verbose = level

        # Print the new verbosity level
        if self._verbose > 0: print(f"(pglBase) Verbosity level set to {self._verbose}")

    ################################################################
    # Open a screen
    ################################################################
    def open(self, whichScreen=None, screenWidth=None, screenHeight=None, screenX=None, screenY=None, stable=False, mglMetalPath=None):
        """
        Open a screen on the specified display.

        Args:
            whichScreen (int): The screen number to open, 0 is the primary display
                               the default is to open the non-primary display
            screenWidth (int, optional): The width of the screen in pixels.
            screenHeight (int, optional): The height of the screen in pixels.
            If screenWidth and screenHeight are not provided, the screen will open full screen
            screenX (int, optional): The x-coordinate of the screen in pixels.
            screenY (int, optional): The y-coordinate of the screen in pixels

            Advanced arguments for debugging:
            stable (bool, optional): If True, forces the use of a stable version of the mglMetal application,
                                     rather than looking for a later compiled version.
            mglMetalPath (str, optional): The file path to the mglMetal application, if omitted will search in the pgl directory

        Returns:
            bool: True if the screen was opened successfully, False otherwise.
        """
        self.printHeader("pglBase:open")
        # get how many displays we have
        (numDisplays, defaultDisplay) = self.getNumDisplaysAndDefault()
        if whichScreen is None: whichScreen = defaultDisplay

        # Check if the screen number is valid
        if whichScreen < 0 or whichScreen >= numDisplays:
            print(f"(pglBase:open) ❌ Error: Invalid screen number {whichScreen}. Must be between 0 and {numDisplays-1}.")
            return False

        # Check whether any screen positioning was provided, in which
        # case we will not open full screen
        if (screenWidth, screenHeight, screenX, screenY) == (None, None, None, None):
            fullScreen = True
        else:
            fullScreen = False

        # Set ddefault values
        screenWidth = screenWidth if screenWidth is not None else 800
        screenHeight = screenHeight if screenHeight is not None else 600
        screenX = screenX if screenX is not None else 100
        screenY = screenY if screenY is not None else 100            

        # get metal app name
        self.metalAppName = self.getMetalAppName(stable=stable, mglMetalPath=mglMetalPath)

        # get a socket name that incorporated date, time and a random string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        randomString = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        self.metalSocketName = f"pglMetal.socket.{timestamp}.{randomString}"

        # create the socket path
        socketName = os.path.join(self.metalSocketPath, self.metalSocketName)

        # start up mglMetal application
        if not os.path.exists(self.metalAppName):
            print(f"(pglBase:open) ❌ Error: mglMetal application not found at {self.metalAppName}")
            return False
        else:
            if self.verbose > 0: 
                print(f"(pglBase:open) Starting mglMetal application: {self.metalAppName}")
                print(f"(pglBase:open) Using socket with address: {socketName}")
            try:
                # start the mglMetal application
                result = subprocess.run([
                    "open", "-g", "-n", self.metalAppName,
                    "--args", "-mglConnectionAddress", socketName
                    ], check=True)
            except Exception as e:
                print(f"(pglBase:open) ❌ Error starting mglMetal application: {e}")
                return False
        
        # now try to connect to the socket
        self.s = pglComm._pglComm(socketName,self)

        if not self.s.isOpen():
            print("(pglBase:open) ❌ Error: Could not connect to mglMetal application.")
            self.s = None
            return False

        # and parse command types
        commandTypesFilename = os.path.join(self.pglDir, "metal/mglCommandTypes.h")
        self.s.parseCommandValues(commandTypesFilename)
        if not self.s.isOpen():
            print("(pglBase:open) ❌ Error: Could not parse command types.")
            self.s = None
            return False

        # set the window location and size
        self.setWindowFrameInDisplay(whichScreen, screenX, screenY, screenWidth, screenHeight)

        # set full screen if requested
        if fullScreen: 
            self.fullScreen(True)
            #self.waitSecs(0.1)
        
        # get the window location
        self.getWindowFrameInDisplay()

        # get frame rate
        self.frameRate = self.getFrameRate(whichScreen)

        # clear screen
        self.clearScreen([0.4, 0.2, 0.5])
        self.flush()
        
        # print how you can get error log
        print("(pgl:open) mglMetal error log can be viewed in MacOS Console app by searching for PROCESS mglMetal or in a terminal with:")
        print("           log stream --level info --process mglMetal")
        self.printHeader()
        # success
        return True
 
    ################################################################
    # close
    ################################################################
    def close(self):
        """
        Close the connection to the mglMetal application and clean up.

        This function sends a close command to the mglMetal application, waits for a response,
        and then closes the socket connection.

        Returns:
            bool: True if the connection was closed successfully, False otherwise.
        """
         # make sure that a screen is open
        if self.isOpen() is False: return True
        
        # Print what we are doing
        if self.verbose > 0: 
            self.printHeader("pglBase:close")
            print("(pglBase:close) Closing connection to mglMetal application")

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
            print(f"(pglBase:close) ❌ Error killing mglMetal application with PID: {pid} : {e}")

        # Close the socket
        self.s.close()
        self.s = None
        if self.verbose>0: self.printHeader()
        return True
    
    ################################################################
    # flush
    ################################################################
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
        # make sure that a screen is open
        if self.isOpen() is False: 
            print(f"(pglBase:flush) ❌ No screen is open")
            return False
        self.s.writeCommand("mglFlush")
        self.commandResults = self.s.readCommandResults()
        
        # keep profile information if profileMode is set
        if self.profileMode > 0:
            # check if we need to reallocate the buffer
            if self.profileModeBufferIndex >= self.profileModeBufferSize:
                # reallocate the buffer
                self.profileModeBufferSize *= 2
                self.profileModeFlushBuffer = np.resize(self.profileModeFlushBuffer, self.profileModeBufferSize)
                # reallocate the commandResults buffer
                if self._profileMode >= 2:
                    self.profileModeCommandResults.extend([{} for _ in range(self.profileModeBufferSize - len(self.profileModeCommandResults))])
            # store the results in the buffer
            self.profileModeFlushBuffer[self.profileModeBufferIndex] = self.commandResults[self.profileCommandResultsField]
            # save the whole command structure if needed
            if self._profileMode >= 2:
                self.profileModeCommandResults[self.profileModeBufferIndex] = self.commandResults 
            self.profileModeBufferIndex += 1
        
        # reset line counter for pglDraw:text
        self.currentLine = 1
        
        # success
        return True
    
    ################################################################
    # setWindowFrameInDisplay
    ################################################################
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
        # make sure that a screen is open
        if self.s is None: 
            print(f"(pglBase:setWindowFrameInDisplay) ❌ No screen is open")
            return
        try:
            # pause interrupts so we don't get interrupted by Ctrl-C
            self.pauseInterrupts()
            # send the commands
            self.s.writeCommand("mglSetWindowFrameInDisplay")
            self.s.write(np.uint32(whichScreen+1))  # whichScreen is 0-indexed in Python, but 1-indexed in mglMetal
            self.s.write(np.uint32(screenX))
            self.s.write(np.uint32(screenY))
            self.s.write(np.uint32(screenWidth))
            self.s.write(np.uint32(screenHeight))
            self.commandResults = self.s.readCommandResults()
        finally:
            # restore interrupts
            self.restoreInterrupts()

        # save pixel dimensions
        self.screenWidth.pix = screenWidth
        self.screenHeight.pix = screenHeight
    
    ################################################################
    # getWindowFrameInDisplay
    ################################################################
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
        # make sure that a screen is open
        if self.s is None: 
            print(f"(pglBase:getWindowFrameInDisplay) ❌ No screen is open")
            return {}

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
        self.commandResults = self.s.readCommandResults(ack)

        # update the stored values
        self.whichScreen = windowLocation.get('whichScreen', 0)
        self.screenX.pix = int(windowLocation.get('screenX', 0))
        self.screenY.pix = int(windowLocation.get('screenY', 0))
        self.screenWidth.pix = int(windowLocation.get('screenWidth', 0))
        self.screenHeight.pix = int(windowLocation.get('screenHeight', 0))

        return windowLocation

    ################################################################
    # fullScreen
    ################################################################
    def fullScreen(self, goFullScreen=True):
        """
        Set the window to fullScreen mode.

        Args:
            goFullScreen (bool): If True, set the window to fullscreen. If False, exit fullscreen.

        Returns:
            bool: True if the fullscreen mode was set successfully, False otherwise.
        """
        # make sure that a screen is open
        if self.s is None: 
            print(f"(pglBase:fullScreen) ❌ No screen is open")
            return False

        if goFullScreen:
            self.s.writeCommand("mglFullscreen")
        else:
            self.s.writeCommand("mglWindowed")
        self.commandResults = self.s.readCommandResults()
        if self.commandResults.get('success',False) is False:
            print("(pglBase:fullscreen) ❌ Error setting fullscreen mode")
            return False
        return True

    ################################################################
    # getTimestamps
    ################################################################
    def getTimestamps(self):
        """
        Get the timestamps for the cpu and gpu

        Returns:
            tuple: (cpuTime, gpuTime)
        """
        if self.isOpen() is False:
            print(f"(pglBase:getTimestamps) ❌ No screen is open")
            return {}

        self.s.writeCommand("mglSampleTimestamps")
        ack = self.s.readAck()
        cpuTime = self.s.read(np.double)
        gpuTime = self.s.read(np.double)
        self.commandResults = self.s.readCommandResults(ack)

        return (cpuTime, gpuTime)

    ################################################################
    # isOpen
    ################################################################
    def isOpen(self):
        """
        Check if a screen is currently open.

        Returns:
            bool: True if a screen is open, False otherwise.
        """
        return self.s is not None
    ################################################################
    # printCommandResults
    ################################################################
    def printCommandResults(self, commandResults=None, relativeToTime=None, prefix="(pglBase:printCommandResults)", index=0):
        """
        Print the results of a command.

        Args:
            commandResults (dict): The command results to print.
        """
        if commandResults is None: commandResults = self.commandResults

        # fieldnames that have special printing
        commandsInt = {'commandCode','success'}
        commandsGPUTime = {'vertexStart','vertexEnd','fragmentStart','fragmentEnd','drawableAcquired','drawablePresented'}
        commandsCPUTime = {'ack','processedTime'}
        
        # extract all valid values from the commandReults
        # that is, ones where the field has the indexed
        # value and put it into a new dict for easy access
        extractedValues = {}
        for field in commandResults.keys():
            # get the field value
            value = commandResults.get(field)
            # convert to a float array
            value = np.array([value], dtype=np.float32).flatten()
            # if it is None, we will just ignore
            if value is not None:
                if isinstance(value, np.ndarray):
                    if index < value.size:
                        value = value[index]
                    else:
                        print(f"Index {index} out of bounds for {field} array")
                        value = None
                        continue
                # get the value and save it to extractedValues
                if value != 0: extractedValues[field] = value
        # get relativeTime if not set
        postfix = ''
        if relativeToTime is None:
            ack = extractedValues.get('ack',None)
            if ack is None:
                postfix = '(absolute time)'
            else:
                postfix = f"(relative to {ack})"
                relativeToTime = ack
            
        # print everything that made it to extractedValues
        for field in extractedValues.keys():
            value = extractedValues[field]
            if field in commandsInt:
                print(f"{prefix} {field}: {int(value)}")
            elif field in commandsCPUTime:
                print(f"{prefix} {field}: {(value * 1000.0 - relativeToTime):0.3f} ms {postfix}")
            elif field in commandsGPUTime:
                print(f"{prefix} {field}: {((value / 1000000.0)-relativeToTime):.3f} ms")
            else:
                print(f"{prefix} {field}: {value}")

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
        # get python information
        self.pythonVersion = sys.version
        print(f"(pgl:checkOS) Python version: {self.pythonVersion}")
        
        # check keyboard
        
        if platform.system() == "Darwin":
            
            # get version
            self.macOSversion = platform.mac_ver()

            # get hardware and gpu info
            try:
                # get cpu info and parse into a dict for easier access
                self.cpuInfo = getCPUInfo()
                # get gpu info and parse into a dict for easier access
                self.gpuInfo = getGPUInfo() 

            except subprocess.CalledProcessError as e:
                self.cpuInfo["error"] = f"Error retrieving hardware info: {e}"
                self.gpuInfo["error"] = f"Error retrieving gpu info: {e}"
            # Print the macOS version and hardware info
            if self.verbose > 0:
                modelName = self.cpuInfo.get("Model Name", "Unknown Model").strip()
                modelID = self.cpuInfo.get("Model Identifier", "Unknown Identifier").strip()
                osVersion = self.macOSversion[0].strip()
                print(f"(pgl:checkOS) Running on {modelName} ({modelID}) with macOS version: {osVersion}")
            if self.verbose > 0: print("(pgl:checkOS)",
                                        self.cpuInfo.get("Processor", "Unknown "),
                                        "Cores:",
                                        self.cpuInfo.get("Total Number of Cores", "Unknown "),
                                        "Memory:",
                                        self.cpuInfo.get("Memory", "Unknown "))
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
    # Clean up open windows (which may be orphaned) and there socket connections
    ################################################################
    def cleanUp(self):
        '''
        Clean up open windows (which may be orphaned) and their socket connections
        '''
        if self.isOpen(): self.close()
        self.shutdownAll()
        self.removeOrphanedSockets()

    ################################################################
    # Shutdown all mglMetal processes
    ################################################################
    def shutdownAll(self):
        '''
        Shutdown all mglMetal processes
        '''
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'mglMetal':
                print(f"(pglBase:shutdownAll) Shutting down mglMetal process: {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except TimeoutExpired:
                    print(f"(pglBase:shutdownAll) Forcefully killing mglMetal process: {proc.pid}")
                    proc.kill()

    ################################################################
    # Remove orphaned sockets
    ################################################################
    def removeOrphanedSockets(self):
        '''
        Remove orphaned sockets
        '''
        if not hasattr(self, 'metalSocketPath'):
            print("(pglBase:removeOrphanedSockets) No metalSocketPath defined, cannot remove orphaned sockets")
            return
        
        socketPattern = os.path.join(self.metalSocketPath, "pglMetal.socket.*")

        # check for all mglMetal processes that are running, what their socket address are
        openMetalSockets = []
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] == 'mglMetal':
                    # check UNIX socket connections for this process
                    for conn in proc.connections(kind='unix'):
                        if conn.laddr:  # laddr is the socket path
                            openMetalSockets.append(conn.laddr)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue  # skip processes we can't inspect
        
        # print out open sockets
        for socket in list(set(openMetalSockets)):
            print(f"(pglBase:removeOrphanedSockets) Found open socket: {socket}")

        nRemovedSockets = 0
        for socketPath in glob.glob(socketPattern):
            # check if it is one that mglMetal is using
            if not socketPath in openMetalSockets:
                print(f"(pglBase:removeOrphanedSockets) Removing orphaned socket: {socketPath}")
                try:
                    os.remove(socketPath)
                    nRemovedSockets += 1
                except OSError as e:
                    print(f"(pglBase:removeOrphanedSockets) Failed to remove {socketPath}: {e}")
        
        # Display how many sockets were removed
        if nRemovedSockets > 0:
            print(f"(pglBase:removeOrphanedSockets) Removed {nRemovedSockets} orphaned sockets")
        else:
            print(f"(pglBase:removeOrphanedSockets) No orphaned sockets found in {self.metalSocketPath}")
    #################################################################
    # Pause interrupts
    #################################################################
    def pauseInterrupts(self):
        """
        Pause interrupts by ignoring the SIGINT signal (Ctrl-C).
        
        Used for when running communication to mglMetal
        """
        # keep original handler so we can restore it later
        self.originalHandler = signal.getsignal(signal.SIGINT)
        # ignore SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    #################################################################
    # Pause interrupts
    #################################################################
    def restoreInterrupts(self):
        """
        Restore interrupts by re-enabling the SIGINT signal (Ctrl-C).
        """
        signal.signal(signal.SIGINT, self.originalHandler)

    #################################################################
    # Print a header
    #################################################################
    def printHeader(self, str="", len=80, fillChar="="):
        '''
        Print a header with a given string centered
        '''
        if str == "":
            print(fillChar * len)
        else:
            print(f" {str} ".center(len, fillChar))

    ###################################
    # get the name of the mglMetalApp
    ###################################
    def getMetalAppName(self, stable=False, mglMetalPath=None):
        '''
        Get the name of the mglMetal application. This will search in the directory where pgl is installed
        as well as the Xcode DerivedData directory where new versions get compiled. If there is a newer
        version available, it will be used instead of the stable version (this is helpful for debugging). You
        can avoid this behavior and always use the stable version by setting stable here, or in open set
        forceStableMGLMetal=True.

        Args:
            stable (bool): Whether to use the stable version of the app.
            mglMetalPath (str, optional): The file path to the mglMetal application, if you want to force a specific path   
        '''
        # if forcing a path, just return that
        if mglMetalPath is not None: return mglMetalPath
    
        # paths to stable version and derivedData for recently compiled versions
        stableAppPath = os.path.join(self.pglDir, "metal/mglMetal.app")
        derivedDataDirectory = os.path.join(self.homeDir, "Library/Developer/Xcode/DerivedData")

        # If runStable is True, always return the stable app path
        if stable: return stableAppPath

        latestBuildPath = None
        latestBuildTime = 0

        # Search for all mglMetal.app instances in DerivedData
        for dirPath, dirNames, fileNames in os.walk(derivedDataDirectory):
            for dirName in dirNames:
                if dirName.endswith(".app") and "mglMetal" in dirName:
                    appPath = os.path.join(dirPath, dirName)
                    modificationTime = os.path.getmtime(appPath)
                    if modificationTime > latestBuildTime:
                        latestBuildTime = modificationTime
                        latestBuildPath = appPath

        # Determine which app to return
        if latestBuildPath:
            return latestBuildPath
        else:
            return stableAppPath


################
# getCPUInfo   #
################
def getCPUInfo():
    """
    Get and Parse the output of `system_profiler SPHardwareDataType` into a structured dictionary.

    Returns:
        dict: A dictionary containing CPU information.
    """
    try:
        systemProfilerOutput = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=True
        )
        # parse into a dict for easier access
        lines = systemProfilerOutput.stdout.splitlines()
        cpuInfo = {"system_profiler_output": systemProfilerOutput.stdout}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # add to dict
                cpuInfo[key] = value

        # extract processor name, which can be named different things on different systems
        processor = next((cpuInfo.get(k) for k in ("Chip", "Processor Name", "CPU Type", "CPU") if cpuInfo.get(k)), "Unknown")

        # add on procesor speed if it exists
        processor += (f" {cpuInfo.get('Processor Speed', '')}" if "Processor Speed" in cpuInfo else "")
        
        # add to processor entry
        cpuInfo["Processor"] = processor
        return cpuInfo
    except Exception as e:
        print(f"(pglBase:getCPUInfo) Warning: {e}")
        return {}

################
# getGPUInfo   #
################
def getGPUInfo():
    """
    Get and Parse the output of `system_profiler SPDisplaysDataType` into a structured dictionary.

    Supports multiple GPUs, nested display info

    Args:
        text (str): Raw text output from system_profiler.

    Returns:
        dict: A dictionary mapping GPU names to their attributes and associated displays.
    """
    try:
        # gpu info
        systemProfilerOutput = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True
        )
        # split into lines
        lines = systemProfilerOutput.stdout.splitlines()
        gpuInfo = {}
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
                gpuInfo[currentGpu] = {"system_profiler_output": systemProfilerOutput.stdout}
                displayList = []
                inDisplaysSection = False

            # Start of Displays section
            elif indent == 6 and stripped == "Displays:":
                inDisplaysSection = True
                gpuInfo[currentGpu]["Displays"] = displayList

            # GPU metadata
            elif indent == 6 and ":" in stripped and not inDisplaysSection:
                key, value = map(str.strip, stripped.split(":", 1))
                gpuInfo[currentGpu][key] = value

            # New display name
            elif indent == 8 and stripped.endswith(":") and inDisplaysSection:
                currentDisplay = {"DisplayName": stripped.rstrip(":")}
                displayList.append(currentDisplay)

            # Display metadata
            elif indent >= 10 and ":" in stripped and currentDisplay is not None:
                key, value = map(str.strip, stripped.split(":", 1))
                currentDisplay[key] = value

        return gpuInfo

    except Exception as e:
        print(f"(pglBase:getGPUInfo) Warning: Parsing failed with error: {e}")
        return {}

def printHeader(str="", len=80, fillChar="="):
    '''
    Print a header with a given string centered
    '''
    if str == "":
        print(fillChar * len)
    else:
        print(f" {str} ".center(len, fillChar))

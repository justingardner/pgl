################################################################
#   filename: pglDevice.py
#    purpose: Device class
#         by: JLG
#       date: July 28, 2025
################################################################

###########
# # Import
##########
from asyncio import subprocess
from pgl import pglTimestamp
from .pglEvent import pglEvent

#################################################################
# Parent class for devices
#################################################################
class pglDevice:
    """
    Parent class for all pglDevice types
    """
    def __init__(self, deviceType):  
        '''
        Initialize the _pglDevice instance.
        
        Args:
            pgl (object): The pgl instance.
            type (str): The type of the device.
                
        Returns:
            None
        '''
        # set the device type
        self.deviceType = deviceType
        # set the initialization time
        self.pglTimestamp = pglTimestamp()
        self.startTime = self.pglTimestamp.getDateAndTime()
        # set the device status
        self.currentStatus = 0
        # some fields about the device that will be set by subclasses
        self.device = None
        self.deviceAttributes = {}
        # set verbosity
        self.verbose = 1


    def __repr__(self):
        return f"<pglDevice type={self.deviceType}>"
    
    def __del__(self):
        """
        Clean up the _pglDevice instance.
        """
        # Perform any necessary cleanup here
        print(f"(pglDevice) Cleaning up device of type {self.deviceType}")
        pass

    def poll(self):
        """
        Poll the event.

        This method is used to poll the event for any updates or changes.
        Should be implemented in subclasses.

        """
        # Implement polling logic here
        return f"(pglDevice) Device {self.deviceType}: poll not implemented"
    
    def status(self):
        """
        Get the status of the device.

        This method retrieves the current status of the device.
        Should be implemented in subclasses.

        Returns:
            str: A string representing the current status of the device.
        """
        # Implement status retrieval logic here
        return f"(pglDevice) Device {self.deviceType}: status not implemented"

#################################################################
# pglDevices is mixed into pgl and handles multiple pglDevice instances
#################################################################
class pglDevices:
    """
    Class to manage multiple pglDevice instances.
    """
    
    def __init__(self):
        """
        Initialize the pglDevices instance.
        """
        self.devices = []

    def devicesAdd(self, device):
        """
        Add a pglDevice instance to the list of devices.

        Args:
            device (pglDevice): The device to add.
        """
        if isinstance(device, pglDevice):
            self.devices.append(device)
            print(f"(pglDevices) Added device: {device.deviceType}")
        else:
            print("(pglDevices) Error: Device must be an instance of pglDevice.")

    def devicesPoll(self):
        """
        Poll all devices for updates.

        This method iterates through all devices and calls their poll method.
        """

        for device in self.devices: 
            # poll each device for events
            eventList = device.poll()
            # add them to the events list
            self.eventsAdd(eventList)
        # return the eventList
        return eventList

#############################
# keyboard device
#############################
from pynput import keyboard
from queue import Queue
import threading

class pglKeyboard(pglDevice):
    def __init__(self): 
        super().__init__(deviceType="pglKeyboard")

        if not self.checkAccessibilityPermission():
            print("(pglKeyboard) ❌ This app is not authorized for Accessibility input monitoring. No keyboard events will be detected!!")
            print("              Go to System Settings → Privacy & Security → Accessibility and add this app.")
            return

        # Create a thread-safe queue
        self.keyQueue = Queue()

        # Store listener reference
        self.listener = keyboard.Listener(
            on_press=self.onPress,
            on_release=self.onRelease
        )

        # Start the keyboard listener thread
        self.listenerThread = threading.Thread(target=self.listener.run, daemon=True)
        self.listenerThread.start()
        
        # for getting time of events
        self.pglTimestamp = pglTimestamp()

        # initialize the modifier keys
        self.shift = False
        self.ctrl = False
        self.alt = False
        self.cmd = False

        print("(pglKeyboard) Keyboard listener initialized.")
    def checkAccessibilityPermission(self):
        """
        Returns True if the process is trusted for accessibility events.
        Works on macOS using tccutil database query.
        """
        try:
            # This command checks if the current app is listed in the accessibility DB
            result = subprocess.run(
                ["sqlite3", "~/Library/Application Support/com.apple.TCC/TCC.db", 
                "SELECT allowed FROM access WHERE client='Python' AND service='kTCCServiceAccessibility';"],
                capture_output=True,
                text=True,
                shell=True
            )
            output = result.stdout.strip()
            return output == "1"
        except Exception:
            return False


    def __del__(self):
        self.stopListener()

    # Callback function for key presses
    def onPress(self, key):
        # check if we have a modifier key
        if key in [keyboard.Key.shift, keyboard.Key.shift_r]:
            self.shift = True
        elif key in [keyboard.Key.ctrl, keyboard.Key.ctrl_r]:
            self.ctrl = True
        elif key in [keyboard.Key.alt, keyboard.Key.alt_r]:
            self.alt = True
        elif key in [keyboard.Key.cmd, keyboard.Key.cmd_r]:
            self.cmd = True
        else:
            # if not, then put the key into the queue
            self.keyQueue.put((key,self.pglTimestamp.getSecs(),self.shift,self.ctrl,self.alt,self.cmd))

    # Callback function for key releases (optional)
    def onRelease(self, key):
        # check if we have a modifier key
        if key in [keyboard.Key.shift, keyboard.Key.shift_r]:
            self.shift = False
        elif key in [keyboard.Key.ctrl, keyboard.Key.ctrl_r]:
            self.ctrl = False
        elif key in [keyboard.Key.alt, keyboard.Key.alt_r]:
            self.alt = False
        elif key in [keyboard.Key.cmd, keyboard.Key.cmd_r]:
            self.cmd = False
        #elif key == keyboard.Key.esc:
        #    print("(pglKeyboard) Esc key released, ending keyboard listener")
        #    return False  # stops listener

    # Proper stop method
    def stopListener(self): 
        '''
        Stop the keyboard listener.
        '''
         # stop the keyboard listener
        if self.isRunning(): self.listener.stop() 
        # stop the thread
        if hasattr(self, 'listenerThread') and self.listenerThread.is_alive():
            self.listenerThread.join(timeout=1)
        print("(pglKeyboard) Listener thread stopped")

    def isRunning(self):
        '''
        Check if the keyboard listener is running.
        '''
        return hasattr(self, 'listener') and self.listener.running

    def poll(self): 
        '''
        Poll the key queue for events.
        '''
        eventList = []
        if not self.isRunning: return eventList

        while not self.keyQueue.empty():
            key, timestamp, shift, ctrl, alt, cmd = self.keyQueue.get()
            # get string representation
            try:
                # normal key
                keyChar = key.char 
                keyCode = ord(key.char)
            except AttributeError:
                # special key
                keyChar = str(key) 
                keyCode = None
            # put in event list
            eventList.append(pglEventKeyboard(keyChar, keyCode, key, timestamp, shift, ctrl, alt, cmd))
        return eventList

###################################
# Keyboard event
###################################
class pglEventKeyboard(pglEvent):
    """
    Represents a keyboard event for the pglKeyboard device.

    """

    def __init__(self, keyStr, keyCode, key, deviceTime, shift, ctrl, alt, cmd):
        '''
        Initialize the pglEventKeyboard instance.
        Args:
            keyStr(str): The key that was pressed.
            keyCode (int): The key code of the pressed key.
            key (Key): The key object.
            deviceTime (float): The device time.
            shift (bool): Whether the shift key was held down.
            ctrl (bool): Whether the ctrl key was held down.
            alt (bool): Whether the alt key was held down.
            cmd (bool): Whether the cmd key was held down.
        Returns:
            None
        '''
        super().__init__("Keyboard")
        self.keyStr = keyStr
        self.keyCode = keyCode
        self.key = key
        self.deviceTime = deviceTime
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        self.cmd = cmd

    def __repr__(self):
        '''
        Return a string representation of the pglEventKeyboard instance.
        Returns:
            str: String representation of the instance.
        '''
        modifierStr = ""
        if self.shift: modifierStr += "Shift "
        if self.ctrl: modifierStr += "Ctrl "
        if self.alt: modifierStr += "Alt "
        if self.cmd: modifierStr += "Cmd "
        return f"(pglEventKeyboard) Key: {self.keyStr}, KeyCode: {self.keyCode}, Device Time: {self.deviceTime}, Modifiers: {modifierStr.strip()}"

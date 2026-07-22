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
import io
import sys
from time import sleep
from typing import Optional
from pgl import pglTimestamp
from .pglEvent import List, pglEvent
from .pglEventListener import pglEventListener
from dataclasses import dataclass
from .pglSerialize import pglSerialize
import matplotlib.pyplot as plt
import numpy as np

#################################################################
# Parent class for devices
#################################################################
class pglDevice:
    """
    Parent class for all pglDevice types
    """
    def __init__(self, deviceType, deviceDescription=None):  
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
        if deviceDescription == None:
            self.deviceDescription = deviceType
        else:
            self.deviceDescription = deviceDescription
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
    
    def start(self):
        pass
    def stop(self):
        pass
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

    def devicesGet(self, deviceType):
        '''
        Get a pglDevice instance by its type.

        Args:
            deviceType (str): The type of the device to retrieve.

        Returns:
            pglDevice: The device instance if found, None otherwise.
        '''
        return [d for d in self.devices if isinstance(d, deviceType)]

    def devicesGetKeyboard(self):
        '''
        Get a pglKeyboardMouse device from pgl. This assumes that there is only
        one pglKeyboardMouse device in the devices list.

        Returns:
            pglDevice: The device instance if found, None otherwise.
        '''
        from .pglKeyboardMouse import pglKeyboardMouse
        d = self.devicesGet(pglKeyboardMouse)
        return d[0] if len(d) > 0 else None

    def setEatKeys(self, keyCodes=None, keyChars=None):
        '''
        Set eat keys for keyboard device. This calls the function setEatKeys on the pglKeyboardMouse device if it exists.

        Args:
            keyCodes (list, optional): List of key codes to eat. Defaults to None.
            keyChars (list, optional): List of key characters to eat. Defaults to None.
        '''
        keyboardDevice = self.devicesGetKeyboard()
        if keyboardDevice is not None:
            keyboardDevice.setEatKeys(keyCodes=keyCodes, keyChars=keyChars)

    def poll(self):
        """
        Poll all devices for updates.

        This method iterates through all devices and calls their poll method.
        """
        eventList = []
        for device in self.devices: 
            # poll each device for events
            eventList = device.poll()
            # add them to the events list
            self.eventsAdd(eventList)
        # return the eventList
        return eventList

#############################
# keyboard and mouse device 
# Uses pglEventListener which
# calls the _pglEventListener C extension
# to listen to events and gets
# hardware precise timestamps for
# keyboard and mouse events
##############################
class pglKeyboardMouse(pglDevice):
    def __init__(self, eatKeys=None):
        super().__init__(deviceType="pglKeyboard")

        if not self.checkAccessibilityPermission():
            print("(pglKeyboardMouse) ❌ This app is not authorized for Accessibility input monitoring. No keyboard events will be detected!!")
            print("  Go to System Settings → Privacy & Security → Accessibility and add this app.")
            print("  If you are running VS Code and it already has permissions granted, try running directly from a terminal with:")
            print("  /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Electron")
            return

        self.start(eatKeys)

    def start(self, eatKeys=None):
        '''
        Start the keyboard listener.
        '''
        if self.isRunning(): return
        print(f"(pglKeyboardMouse:start) Starting keyboard and mouse event listener.")
        
        # start the listener
        self.listener = pglEventListener()
        self.listener.start()
        
        # if eatKeys are passed in, set them
        if eatKeys is not None:
            self.listener.setEatKeys(keyString=eatKeys)
    
    def stop(self):
        '''
        Stop the keyboard listener.
        '''
        if self.isRunning(): 
            self.listener.stop()
        print(f"(pglKeyboardMouse:stop) Stopping keyboard listener.")
        
    def checkAccessibilityPermission(self):
        """
        Returns True if the process is trusted for accessibility events.
        Works on macOS using tccutil database query.
        """
        # if already running, return True
        if self.isRunning():
            return True
        
        accessibilityPermission = False
        listener = None

        # capture stdout and stderr
        error = None
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Start a temporary listener
            listener = pglEventListener()
            # check if one is already running (could have
            # been started by another pglEventListener instance)
            if not listener.isRunning():
                listener.start()
                sleep(0.1)
                # if it is not running, then problem.
                if not listener.isRunning(): error = "Listener did not start properly"

        except Exception as e:
            error = str(e)

        finally:
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass

        if error:
            print(f"(pglKeyboardMouse) ❌ {error.rstrip('\n')}")
        else:
            accessibilityPermission = True
        
        return(accessibilityPermission)

    def __del__(self):
        if self.isRunning():
            self.listener.stop()

    def isRunning(self):
        '''
        Check if the keyboard listener is running.
        '''
        return hasattr(self, 'listener') and self.listener.isRunning()
    
    def clear(self):
        '''
        Clear any pending events in the queue.
        '''
        if self.isRunning():
            self.listener.clearQueues()

    def poll(self): 
        '''
        Poll the key queue for events.
        '''
        eventList = []
        if not self.isRunning(): return eventList

        # get all keyEvents from listener
        keyEvents = self.listener.getAllKeyboardEvents()
        
        # extract fields from keyEvents
        for keyEvent in keyEvents:
            # Extract fields from event dictionary
            timestamp = keyEvent['timestamp']
            keyCode = keyEvent['keyCode']
            eventType = keyEvent['eventType']
            
            # Extract modifier keys
            shift = keyEvent.get('shift', False)
            ctrl = keyEvent.get('control', False)
            alt = keyEvent.get('alt', False)
            cmd = keyEvent.get('command', False)
            
            # Convert keycode to character (if possible)
            keyChar = self.keyCodeToChar(keyCode, shift)
            
            # Create event object
            eventList.append(pglEventKeyboard(
                keyChar=keyChar,
                keyCode=keyCode,
                key=keyEvent,
                eventType=eventType,
                timestamp=timestamp,
                shift=shift,
                ctrl=ctrl,
                alt=alt,
                cmd=cmd
            ))

        return eventList
    
    def setEatKeys(self, keyCodes):
        '''
        Set keys to eat so they don't propagate to the OS.

        Args:
            keyCodes (list): List of key codes to eat.
        '''
        if self.isRunning():
            self.listener.setEatKeys(keyCodes)
    
    def keyCodeToChar(self, keyCode: int, shift: bool = False) -> Optional[str]:
        """
        Convert macOS keycode to character representation.
        
        Args:
            keyCode: The macOS keycode
            shift: Whether shift is pressed
            
        Returns:
            Character string or special key name
        """
        # Lowercase letter keycodes (without shift)
        keycodeMapLower = {
            0: 'a', 11: 'b', 8: 'c', 2: 'd', 14: 'e', 3: 'f', 5: 'g', 4: 'h',
            34: 'i', 38: 'j', 40: 'k', 37: 'l', 46: 'm', 45: 'n', 31: 'o',
            35: 'p', 12: 'q', 15: 'r', 1: 's', 17: 't', 32: 'u', 9: 'v',
            13: 'w', 7: 'x', 16: 'y', 6: 'z',
        }
        
        # Uppercase letters (with shift)
        keycodeMapUpper = {
            0: 'A', 11: 'B', 8: 'C', 2: 'D', 14: 'E', 3: 'F', 5: 'G', 4: 'H',
            34: 'I', 38: 'J', 40: 'K', 37: 'L', 46: 'M', 45: 'N', 31: 'O',
            35: 'P', 12: 'Q', 15: 'R', 1: 'S', 17: 'T', 32: 'U', 9: 'V',
            13: 'W', 7: 'X', 16: 'Y', 6: 'Z',
        }
        
        # Numbers (without shift)
        numberMap = {
            29: '0', 18: '1', 19: '2', 20: '3', 21: '4',
            23: '5', 22: '6', 26: '7', 28: '8', 25: '9',
        }
        
        # Numbers with shift (symbols)
        numberShiftMap = {
            29: ')', 18: '!', 19: '@', 20: '#', 21: '$',
            23: '%', 22: '^', 26: '&', 28: '*', 25: '(',
        }
        
        # Punctuation without shift
        punctuationMap = {
            27: '-', 24: '=', 33: '[', 30: ']', 42: '\\',
            41: ';', 39: "'", 43: ',', 47: '.', 44: '/', 50: '`',
        }
        
        # Punctuation with shift
        punctuationShiftMap = {
            27: '_', 24: '+', 33: '{', 30: '}', 42: '|',
            41: ':', 39: '"', 43: '<', 47: '>', 44: '?', 50: '~',
        }
        
        # Special keys
        specialKeys = {
            49: 'space',
            36: 'return',
            48: 'tab',
            51: 'delete',
            53: 'escape',
            76: 'enter',  # Numpad enter
            123: 'left',
            124: 'right',
            125: 'down',
            126: 'up',
            122: 'f1', 120: 'f2', 99: 'f3', 118: 'f4', 96: 'f5', 97: 'f6',
            98: 'f7', 100: 'f8', 101: 'f9', 109: 'f10', 103: 'f11', 111: 'f12',
        }
       
        # Check special keys first
        if keyCode in specialKeys:
            return specialKeys[keyCode]
        
        # Check letters
        if shift:
            if keyCode in keycodeMapUpper:
                return keycodeMapUpper[keyCode]
        else:
            if keyCode in keycodeMapLower:
                return keycodeMapLower[keyCode]
        
        # Check numbers and symbols
        if shift:
            if keyCode in numberShiftMap:
                return numberShiftMap[keyCode]
            if keyCode in punctuationShiftMap:
                return punctuationShiftMap[keyCode]
        else:
            if keyCode in numberMap:
                return numberMap[keyCode]
            if keyCode in punctuationMap:
                return punctuationMap[keyCode]
        
        # Unknown keycode
        return f'<keycode:{keyCode}>'
    
    def charToKeyCode(self, char: str) -> Optional[int]:
        """
        Convert character representation to macOS keycode.
        
        Args:
            char: Character string or special key name
            
        Returns:
            The macOS keycode, or None if not found
        """
        # Letter keycodes (case-insensitive)
        charMapLetters = {
            'a': 0, 'b': 11, 'c': 8, 'd': 2, 'e': 14, 'f': 3, 'g': 5, 'h': 4,
            'i': 34, 'j': 38, 'k': 40, 'l': 37, 'm': 46, 'n': 45, 'o': 31,
            'p': 35, 'q': 12, 'r': 15, 's': 1, 't': 17, 'u': 32, 'v': 9,
            'w': 13, 'x': 7, 'y': 16, 'z': 6,
            'A': 0, 'B': 11, 'C': 8, 'D': 2, 'E': 14, 'F': 3, 'G': 5, 'H': 4,
            'I': 34, 'J': 38, 'K': 40, 'L': 37, 'M': 46, 'N': 45, 'O': 31,
            'P': 35, 'Q': 12, 'R': 15, 'S': 1, 'T': 17, 'U': 32, 'V': 9,
            'W': 13, 'X': 7, 'Y': 16, 'Z': 6,
        }
        
        # Numbers and symbols
        charMapNumbers = {
            '0': 29, '1': 18, '2': 19, '3': 20, '4': 21,
            '5': 23, '6': 22, '7': 26, '8': 28, '9': 25,
            ')': 29, '!': 18, '@': 19, '#': 20, '$': 21,
            '%': 23, '^': 22, '&': 26, '*': 28, '(': 25,
        }
        
        # Punctuation
        charMapPunctuation = {
            '-': 27, '_': 27, '=': 24, '+': 24,
            '[': 33, '{': 33, ']': 30, '}': 30,
            '\\': 42, '|': 42, ';': 41, ':': 41,
            "'": 39, '"': 39, ',': 43, '<': 43,
            '.': 47, '>': 47, '/': 44, '?': 44,
            '`': 50, '~': 50,
        }
        
        # Special keys (case-insensitive)
        charMapSpecial = {
            'space': 49, 'return': 36, 'tab': 48, 'delete': 51,
            'escape': 53, 'esc': 53, 'enter': 76,
            'left': 123, 'right': 124, 'down': 125, 'up': 126,
            'f1': 122, 'f2': 120, 'f3': 99, 'f4': 118, 'f5': 96, 'f6': 97,
            'f7': 98, 'f8': 100, 'f9': 101, 'f10': 109, 'f11': 103, 'f12': 111,
        }
        
        # Try letters first
        if char in charMapLetters:
            return charMapLetters[char]
        
        # Try numbers/symbols
        if char in charMapNumbers:
            return charMapNumbers[char]
        
        # Try punctuation
        if char in charMapPunctuation:
            return charMapPunctuation[char]
        
        # Try special keys (case-insensitive)
        if char.lower() in charMapSpecial:
            return charMapSpecial[char.lower()]
        
        return None
#############################
# keyboard device (pynput implementation)
# Pynput does not provide ability to
# eat specific keys (as of 2/16/2026)
#############################
from pynput import keyboard
from queue import Queue
import threading

class pglPynputKeyboard(pglDevice):
    def __init__(self, eatKeys=False): 
        super().__init__(deviceType="pglKeyboard")

        if not self.checkAccessibilityPermission():
            print("(pglKeyboard) ❌ This app is not authorized for Accessibility input monitoring. No keyboard events will be detected!!")
            print("              Go to System Settings → Privacy & Security → Accessibility and add this app.")
            print("              If you are running VS Code and it already has permissions granted, try running directly from a terminal with:")
            print("              /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Electron")
            return

        self.start(eatKeys)

    def start(self, eatKeys=False):
        '''
        Start the keyboard listener.
        '''
        if self.isRunning(): return
        print(f"(pglKeyboard:start) Starting keyboard listener.")
        
        # Create a thread-safe queue
        self.keyQueue = Queue()

        # Store listener reference
        self.listener = keyboard.Listener(
            on_press=self.onPress,
            on_release=self.onRelease,
            suppress=eatKeys
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
    
    def stop(self):
        '''
        Stop the keyboard listener.
        '''
        if self.isRunning(): self.stopListener()
        print(f"(pglKeyboard:stop) Stopping keyboard listener.")

        
    def checkAccessibilityPermission(self):
        """
        Returns True if the process is trusted for accessibility events.
        Works on macOS using tccutil database query.
        """
        accessibilityPermission = False

        # capture stdout and stderr
        error = None
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Redirect stdout and stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture

            # Start a temporary listener
            listener = keyboard.Listener()
            listener.start()  # non-blocking
            sleep(0.1)   # give it a moment to initialize

            if not listener.isRunning(): error = "Listener did not start properly"

        except Exception as e:
            error = str(e)

        finally:
            listener.stop()
            # Restore stdout and stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr

        # Get captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        if error:
            print(f"(pglKeyboard) ❌ {error.rstrip('\n')}")
        elif stdout_output:
            print(f"(pglKeyboard) ❌ {stdout_output.rstrip('\n')}")
        elif stderr_output:
            print(f"(pglKeyboard) ❌ {stderr_output.rstrip('\n')}")
        else:
            accessibilityPermission = True
        
        return(accessibilityPermission)


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
            eventList.append(pglEventKeyboard(keyChar, keyCode, key, timestamp, shift, ctrl, alt, cmd, "keydown"))
        return eventList

###################################
# Keyboard event
###################################
class pglEventKeyboard(pglEvent):
    """
    Represents a keyboard event for the pglKeyboard device.

    """

    def __init__(self, keyChar, keyCode, key, timestamp, shift, ctrl, alt, cmd, eventType = None):
        '''
        Initialize the pglEventKeyboard instance.
        Args:
            keyChar(str): The key that was pressed.
            keyCode (int): The key code of the pressed key.
            key (Key): The key object.
            timestamp (double): The device time.
            shift (bool): Whether the shift key was held down.
            ctrl (bool): Whether the ctrl key was held down.
            alt (bool): Whether the alt key was held down.
            cmd (bool): Whether the cmd key was held down.
            eventType (str): The type of event ('keydown' or 'keyup').
        Returns:
            None
        '''
        super().__init__("keyboard")
        self.keyChar = keyChar
        self.keyCode = keyCode
        self.key = key
        self.timestamp = timestamp
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        self.cmd = cmd
        self.eventType = eventType
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
        return f"(pglEventKeyboard) Key: {self.keyChar}, KeyCode: {self.keyCode}, Timestamp: {self.timestamp}, Modifiers: {modifierStr.strip()}, Event Type: {self.eventType}"

################################################################
#  Abstract base class defining the interface for
#  digital IO devices. Concrete devices (e.g.,
#  pglLabJack) should inherit from this and implement
#  the stubbed methods.
################################################################
class pglDigitalIODevice(pglDevice):
    '''
    Abstract base class for digital IO devices.

    Any concrete digital IO device (e.g., a LabJack, an Arduino, a
    National Instruments DAQ, etc.) should inherit from this class and
    implement the stubbed methods below. This defines the common
    interface that the rest of pgl can rely on regardless of the
    underlying hardware.

    Subclasses are expected to:
        - Establish/close the hardware connection (in __init__ / stop)
        - Set self.digitalOutputConfigured appropriately
        - Implement all methods marked as NotImplementedError below
    '''

    def __init__(self, deviceType="DigitalIODevice"):
        # whether a digital output channel has been configured
        self.digitalOutputConfigured = False
        super().__init__(deviceType=deviceType)

    def __repr__(self):
        return f"<pglDigitalIODevice deviceType={getattr(self, 'deviceType', 'Unknown')}>"

    ################################################################
    # Digital output interface
    ################################################################
    def setupDigitalOutput(self, channel=0):
        '''
        Configure a digital output channel.

        Implementations should convert the channel argument into the
        hardware-specific representation, set the channel as an output,
        initialize it to a known state (typically LOW), and set
        self.digitalOutputConfigured = True on success (False on failure).

        Args:
            channel (int or str): Digital channel number or name
        '''
        raise NotImplementedError("(pglDigitalIODevice:setupDigitalOutput) Subclass must implement setupDigitalOutput().")

    def digitalOutput(self, state, pulseLen=None):
        '''
        Set the digital output state immediately.

        Implementations should check self.digitalOutputConfigured first.
        If pulseLen is provided, the output should return to the opposite
        state after pulseLen milliseconds.

        Args:
            state (bool): True for HIGH, False for LOW
            pulseLen (float or None): Pulse length in milliseconds

        Returns:
            timestamp (float or None): Timestamp when output was set,
                                       or None on error.
        '''
        raise NotImplementedError("(pglDigitalIODevice:digitalOutput) Subclass must implement digitalOutput().")

    def digitalOutputAtTime(self, targetTime, state, pulseLen=None):
        '''
        Set the digital output state at a specified future time.

        Implementations should check self.digitalOutputConfigured first
        and validate that targetTime is in the future.

        Args:
            targetTime (float): Timestamp (in seconds) when the pulse
                                should be delivered.
            state (bool): True for HIGH, False for LOW
            pulseLen (float or None): Pulse length in milliseconds

        Returns:
            bool: True if the pulse was successfully scheduled, False otherwise
        '''
        raise NotImplementedError("(pglDigitalIODevice:digitalOutputAtTime) Subclass must implement digitalOutputAtTime().")

    ################################################################
    # Analog input interface
    ################################################################
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000, range=10.0):
        '''
        Start analog input reading from specified channels.

        Args:
            duration (float): Duration of recording in seconds
            channels (list): List of channel numbers or names
            scanRate (int): Sampling rate in Hz
            scansPerRead (int): Number of scans per read operation
            range (float): Voltage range for analog inputs
        '''
        raise NotImplementedError("(pglDigitalIODevice:startAnalogRead) Subclass must implement startAnalogRead().")

    def stopAnalogRead(self, waitToFinish=False, doNotTruncate=False):
        '''
        Stop the analog reading and return time and data arrays.

        Args:
            waitToFinish (bool): If True, wait for acquisition to finish.
            doNotTruncate (bool): If True, do not truncate to exact sample count.

        Returns:
            tuple: (time, data) arrays, or (None, None) on error.
        '''
        raise NotImplementedError("(pglDigitalIODevice:stopAnalogRead) Subclass must implement stopAnalogRead().")

    ################################################################
    # Lifecycle interface
    ################################################################
    def start(self):
        '''
        Start the device.
        '''
        raise NotImplementedError("(pglDigitalIODevice:start) Subclass must implement start().")

    def stop(self):
        '''
        Stop the device.
        '''
        raise NotImplementedError("(pglDigitalIODevice:stop) Subclass must implement stop().")
    
class pglAnalogTraceData(pglSerialize):
    # Stores short trains of analog data created by pglDigitalIODevice
    # and offers functions for display
    def __init__(self, time, data, channelNames=None):
        '''
        store the time and data
        '''
        self.time = time
        self.data = data
        
        # compute number of channels
        self.numChannels = 1 if data.ndim == 1 else data.shape[1]
        
        # Handle channel names
        if channelNames is None:
            channelNames = []
        else:
            channelNames = list(channelNames)

        # figure out zero-padding width based on channel count
        padWidth = max(3, len(str(self.numChannels - 1)))

        # build a normalized list of channelNames (if any are missing)
        # Will add channelNames like: CH001
        normalizedChannelNames = []
        for i in range(self.numChannels):
            if i < len(channelNames) and channelNames[i] is not None and str(channelNames[i]).strip() != "":
                # coerce existing entry to string
                normalizedChannelNames.append(str(channelNames[i]))
            else:
                # fill missing/blank entry with CHnnn
                normalizedChannelNames.append(f"CH{i:0{padWidth}d}")

        # warn if caller supplied more names than there are channels
        if len(channelNames) > self.numChannels:
            print(f"(pglAnalogTraceData) Warning: {len(channelNames)} names given but only {self.numChannels} channels. Extra names ignored.")
        
        # store the normalized channel names
        self.channelNames = normalizedChannelNames
        
    def __len__(self):
        if self.time is not None:
            return len(self.time)
        else:
            return 0
    
    @property
    def nSamples(self):
        return self.__len__()
    
    def getCycles(self, cycleLen=None, digitalSyncChannel=None, digitalSyncThreshold=None, ignoreInitial=None):
        '''
        Extract cycles from analog data based on fixed cycle length or digital sync triggers.
        
        Args:
            cycleLen (float): Fixed cycle length in seconds (used if digitalSyncChannel is None)
            digitalSyncChannel (int): Channel index to use for digital sync detection
            digitalSyncThreshold (float): Voltage threshold for detecting digital pulse rising edge
            ignoreInitial (float): Number of seconds to ignore from the beginning of data.
                                  If None (default), no data is ignored. Must be non-negative.
            
        Returns:
            dict: Dictionary containing:
                - 'cycles': list of arrays, one per channel, each array is (numCycles, samplesPerCycle)
                - 'cycleTime': time array for one cycle
                - 'mean': list of mean cycles per channel
                - 'std': list of std cycles per channel
                - 'median': list of median cycles per channel
                - 'numCycles': number of cycles detected
                - 'cycleLen': actual cycle length used
                - 'ignoredSamples': number of samples ignored from the beginning
        '''
        if self.time is None or self.data is None:
            print("(pglLabJack:getCycles) No data provided.")
            return None
        
        # Validate ignoreInitial parameter
        if ignoreInitial is not None:
            if not isinstance(ignoreInitial, (int, float)):
                print(f"(pglLabJack:getCycles) Error: ignoreInitial must be a number or None, got {type(ignoreInitial).__name__}")
                return None
            if ignoreInitial < 0:
                print(f"(pglLabJack:getCycles) Error: ignoreInitial must be non-negative, got {ignoreInitial}")
                return None
            if ignoreInitial >= self.time[-1] - self.time[0]:
                print(f"(pglLabJack:getCycles) Error: ignoreInitial ({ignoreInitial}s) is greater than or equal to total data duration ({self.time[-1] - self.time[0]:.3f}s)")
                return None
        
        # Filter data if ignoreInitial is specified
        ignoredSamples = 0
        time=self.time
        data=self.data
        if ignoreInitial is not None and ignoreInitial > 0:
            # Find the index where time exceeds ignoreInitial seconds from start
            startTime = self.time[0] + ignoreInitial
            maskIndices = np.where(self.time >= startTime)[0]
            
            if len(maskIndices) == 0:
                print(f"(pglLabJack:getCycles) Error: No data remains after ignoring initial {ignoreInitial}s")
                return None
            
            startIdx = maskIndices[0]
            ignoredSamples = startIdx
            
            # Slice the data
            time = self.time[startIdx:]
            if data.ndim == 1:
                data = self.data[startIdx:]
            else:
                data = self.data[startIdx:, :]
            
            print(f"(pglLabJack:getCycles) Ignoring first {ignoreInitial}s ({ignoredSamples} samples)")
        
        # Handle single or multi-channel data
        if data.ndim == 1:
            dataToProcess = data.reshape(-1, 1)
            numChannels = 1
        else:
            dataToProcess = data
            numChannels = data.shape[1]
        
        # Determine cycle start indices and samples per cycle
        if digitalSyncChannel is not None and digitalSyncThreshold is not None:
            # Use digital sync channel to detect cycle starts
            syncData = dataToProcess[:, digitalSyncChannel]
            
            # Detect rising edges (when signal crosses threshold from below)
            aboveThreshold = syncData > digitalSyncThreshold
            risingEdges = np.where(np.diff(aboveThreshold.astype(int)) > 0)[0] + 1
            
            if len(risingEdges) < 2:
                print(f"(pglLabJack:getCycles) Warning: Found {len(risingEdges)} rising edges. Need at least 2 for cycle analysis.")
                return None
            
            # Calculate cycle length from detected triggers
            cycleLengths = np.diff(risingEdges)
            samplesPerCycle = int(np.median(cycleLengths))
            cycleLen = samplesPerCycle * np.median(np.diff(time))
            
            # Use detected trigger indices as cycle starts
            cycleStarts = risingEdges[:-1]  # Exclude last one to ensure complete cycles
            
        else:
            # Use fixed cycleLen
            if cycleLen is None:
                print("(pglLabJack:getCycles) Must provide either cycleLen or digitalSyncChannel/digitalSyncThreshold.")
                return None
                
            dt = np.mean(np.diff(time))
            samplesPerCycle = int(cycleLen / dt)
            
            # Check if data is long enough for at least one cycle
            if len(time) < samplesPerCycle:
                print(f"(pglLabJack:getCycles) Warning: Data length ({len(time)} samples) is shorter than one cycle ({samplesPerCycle} samples).")
                return None
            
            # Generate regular cycle starts
            numCycles = len(dataToProcess) // samplesPerCycle
            cycleStarts = np.arange(numCycles) * samplesPerCycle
        
        # Create cycle time array
        cycleTime = np.linspace(0, cycleLen, samplesPerCycle)
        
        # Extract cycles for each channel
        allCycles = []
        allMeans = []
        allStds = []
        allMedians = []
        
        for ch in range(numChannels):
            channelData = dataToProcess[:, ch]
            cycles = []
            
            for startIdx in cycleStarts:
                endIdx = startIdx + samplesPerCycle
                
                # Skip if cycle extends beyond data
                if endIdx > len(channelData):
                    continue
                
                cycle = channelData[startIdx:endIdx]
                
                # Pad or trim to exact samplesPerCycle length (for digital sync with varying lengths)
                if len(cycle) < samplesPerCycle:
                    cycle = np.pad(cycle, (0, samplesPerCycle - len(cycle)), mode='edge')
                elif len(cycle) > samplesPerCycle:
                    cycle = cycle[:samplesPerCycle]
                    
                cycles.append(cycle)
            
            if len(cycles) == 0:
                print(f"(pglLabJack:getCycles) No complete cycles found for channel {ch}.")
                return None
            
            # Convert to array (numCycles, samplesPerCycle)
            cycles = np.array(cycles)
            
            # Calculate statistics
            meanCycle = np.mean(cycles, axis=0)
            stdCycle = np.std(cycles, axis=0)
            medianCycle = np.median(cycles, axis=0)
            
            allCycles.append(cycles)
            allMeans.append(meanCycle)
            allStds.append(stdCycle)
            allMedians.append(medianCycle)
        
        return {
            'cycles': allCycles,
            'cycleTime': cycleTime,
            'mean': allMeans,
            'std': allStds,
            'median': allMedians,
            'numCycles': len(cycles),
            'cycleLen': cycleLen,
            'ignoredSamples': ignoredSamples
        }
    
    def display(self, cycleLen=None, digitalSyncChannel=None, digitalSyncThreshold=3, ignoreInitial=None, displayStartEnd=None, fig=None):
        '''
        Plot the analog read data

        Args:
            cycleLen (float): If provided, creates a second subplot showing cycle-averaged data
            digitalSyncChannel (int): Channel index to use for digital sync detection
            digitalSyncThreshold (float): Voltage threshold for detecting digital pulse rising edge
            ignoreInitial (float): Time in seconds to ignore at the beginning of the recording for
                displaying cycles (e.g., to exclude initial transients). If None, no data is ignored. Must be non-negative.
            displayStartEnd (float): If not None, will display the first displayStartEnd seconds as separate graphs  
            fig (matplotlib fig): If not none, will plot into the supplied fig
            
        Returns:
            dict: Dictionary containing:
            - 'fig': Figure object
            - 'cycleData': dict from getCycles function (if cycleLen or digitalSyncChannel is provided), otherwise None
            - 'axes': Dictionary of axis objects with keys:
                - 'fullTrace': Full trace axis (always present)
                - 'start': Start segment axis (if displayStartEnd is set)
                - 'end': End segment axis (if displayStartEnd is set)
                - 'cycle': Cycle-averaged axis (if cycleLen or digitalSyncChannel is set)
   
        '''
        retval = {}

        if self.time is None or self.data is None:
            print("(pglLabJack:plotAnalogRead) No data to plot.")
            return
        
        # Determine number of rows needed
        numRows = 1  # Always have full trace row
        if displayStartEnd is not None:
            numRows += 1  # Add row for start/end segments
        if cycleLen is not None or digitalSyncChannel is not None:
            numRows += 1  # Add row for cycle-averaged data
        
        # Determine grid layout
        if displayStartEnd is not None or (cycleLen is not None or digitalSyncChannel is not None):
            if fig is not None:
                fig.clear()
            else:
                fig = plt.figure(figsize=(16, 6 * numRows / 2))
            gs = fig.add_gridspec(numRows, 2, hspace=0.3, wspace=0.3)
        else:
            if fig is not None:
                fig.clear()
            else:
                fig = plt.figure(figsize=(16, 6))
            gs = fig.add_gridspec(1, 1)
        retval['fig'] = fig
        currentRow = 0
        
        # First row: Full analog trace (spans both columns)
        axFullTrace = fig.add_subplot(gs[currentRow, :])
        retval['fullTrace'] = axFullTrace
        if self.data.ndim == 1:
            # Single channel
            axFullTrace.plot(self.time, self.data, label=self.channelNames[0])
        else:
            # Multiple channels
            for i in range(self.data.shape[1]):
                axFullTrace.plot(self.time, self.data[:, i], label=self.channelNames[i])
        
        axFullTrace.set_xlabel("Time (s)")
        axFullTrace.set_ylabel("Voltage (V)")
        axFullTrace.set_title("Analog Trace Data")
        axFullTrace.legend()
        axFullTrace.grid(True)
        currentRow += 1
        
        # Second row: Start and end segments (if requested)
        if displayStartEnd is not None:
            # Determine if we should use ms or s for display
            useMilliseconds = displayStartEnd < 1.0
            timeMultiplier = 1000 if useMilliseconds else 1
            timeUnit = "ms" if useMilliseconds else "s"
            
            # Left column: First displayStartEnd seconds
            axStart = fig.add_subplot(gs[currentRow, 0])
            retval['start'] = axStart
            startIdx = int(displayStartEnd * self.scanRate)
            timeStart = self.time[:startIdx] * timeMultiplier
            if self.data.ndim == 1:
                axStart.plot(timeStart, self.data[:startIdx], label=self.channelNames[0])
            else:
                for ch in range(self.data.shape[1]):
                    axStart.plot(timeStart, self.data[:startIdx, ch], label=self.channelNames[ch])
            
            axStart.set_xlabel(f"Time ({timeUnit})")
            axStart.set_ylabel("Voltage (V)")
            axStart.set_title(f"First {displayStartEnd} seconds")
            axStart.legend()
            axStart.grid(True)
            
            # Right column: Last displayStartEnd seconds
            axEnd = fig.add_subplot(gs[currentRow, 1])
            retval['end'] = axEnd
            endIdx = int(displayStartEnd * self.scanRate)
            timeEnd = self.time[-endIdx:] * timeMultiplier
            if data.ndim == 1:
                axEnd.plot(timeEnd, self.data[-endIdx:], label=self.channelNames[0])
            else:
                for ch in range(self.data.shape[1]):
                    axEnd.plot(timeEnd, self.data[-endIdx:, ch], label=self.channelNames[ch])
            
            axEnd.set_xlabel(f"Time ({timeUnit})")
            axEnd.set_ylabel("Voltage (V)")
            axEnd.set_title(f"Last {displayStartEnd} seconds")
            axEnd.legend()
            axEnd.grid(True)
            currentRow += 1
        
        # Next row: Cycle-averaged data (if requested, spans both columns)
        if cycleLen is not None or digitalSyncChannel is not None:
            axCycle = fig.add_subplot(gs[currentRow, :])
            retval['cycle'] = axCycle
            
            # Get cycles using the getCycles function
            cycleData = self.getCycles(cycleLen, digitalSyncChannel, digitalSyncThreshold, ignoreInitial)
            retval['cycleData'] = cycleData
            
            if cycleData is None:
                axCycle.text(0.5, 0.5, 'Unable to extract cycles', 
                           ha='center', va='center', transform=axCycle.transAxes)
                axCycle.set_xlabel("Time in Cycle (s)")
                axCycle.set_ylabel("Voltage (V)")
                axCycle.set_title("Cycle-Averaged Data")
            else:
                cycleTime = cycleData['cycleTime']
                numChannels = len(cycleData['cycles'])
                
                # Convert time to ms, if cycle time is less than 1 second
                if max(cycleTime) < 1.0:
                    cycleTime = cycleTime * 1000
                    xAxisLabel = "Time in Cycle (ms)"
                else:
                    xAxisLabel = "Time in Cycle (s)"
                    
                for ch in range(numChannels):
                    cycles = cycleData['cycles'][ch]
                    medianCycle = cycleData['median'][ch]
                    
                    # Plot individual trials as thin lines in background
                    for i in range(cycles.shape[0]):
                        axCycle.plot(cycleTime, cycles[i, :], color=f'C{ch}', alpha=0.2, linewidth=0.5)
                                        
                    # Plot median as solid line
                    axCycle.plot(cycleTime, medianCycle, color=f'C{ch}', 
                               linewidth=2, label=self.channelNames[ch])
                
                titleStr = "Trigger-Averaged Data" if digitalSyncChannel is not None else "Cycle-Averaged Data"
                axCycle.set_xlabel(xAxisLabel)
                axCycle.set_ylabel("Voltage (V)")
                axCycle.set_title(f"{titleStr} (n={cycleData['numCycles']} cycles)")
                axCycle.legend()
                axCycle.grid(True)
        
        #plt.tight_layout()
        #plt.show(block=False)    
        
        # Return figure and axes dictionary
        return retval
 
        
        
        
    


    
################################################################
#   filename: pglKeyboardMouse.py
#    purpose: Keyboard and Mouse handling for PGL
#         by: JLG
#       date: March 1, 2026
################################################################

#############
# Import modules
#############
import io
import sys
from time import sleep
from typing import Optional
from .pglEvent import pglEvent
from .pglEventListener import pglEventListener, keyCodeToChar, charToKeyCode
from .pglDevice import pglDevice

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
            self.eatKeyCodes = self.listener.setEatKeys(keyString=eatKeys)
        else:
            self.eatKeyCodes = []
    
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
            keyChar = keyCodeToChar(keyCode, shift)
            
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
    
    def setEatKeys(self, keyCodes=None, keyChars=None):
        '''
        Set keys to eat so they don't propagate to the OS. 

        Args:
            keyCodes (list): List of key codes to eat.
            keyChars (list): List of characters to eat (e.g., ["a", "b", "c", "space"]). Each character is converted to its key code.
        '''
        if self.isRunning():
            self.eatKeyCodes = self.listener.setEatKeys(keyCodes, keyChars)
    
    def charToKeyCode(self, char):
        '''
        Convert a character to a key code using the charToKeyCode function.
        '''
        return charToKeyCode(char)

    def keyCodeToChar(self, keyCode, shift=False):
        '''
        Convert a key code to a character using the keyCodeToChar function.
        '''
        return keyCodeToChar(keyCode, shift)
        

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
        if not self.isRunning(): return eventList

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


######################################
# Keyboard input buffer for text entry
######################################
class pglKeyBuffer:
    def __init__(self):
        self.buffer = ""
        self.cursorPosition = 0
        
        # MacOS key codes
        self.keyCodeMap = {
            # Letters
            0: 'a', 1: 's', 2: 'd', 3: 'f', 4: 'h', 5: 'g', 6: 'z', 7: 'x',
            8: 'c', 9: 'v', 11: 'b', 12: 'q', 13: 'w', 14: 'e', 15: 'r',
            16: 'y', 17: 't', 31: 'o', 32: 'u', 34: 'i', 35: 'p', 37: 'l',
            38: 'j', 40: 'k', 45: 'n', 46: 'm',
            
            # Numbers
            18: '1', 19: '2', 20: '3', 21: '4', 23: '5', 22: '6', 26: '7',
            28: '8', 25: '9', 29: '0',
            
            # Special characters
            27: '-', 24: '=', 33: '[', 30: ']', 41: ';', 39: "'", 42: '\\',
            43: ',', 47: '.', 44: '/',
            
            # Space
            49: ' ',
            
            # Keypad
            65: '.', 67: '*', 69: '+', 75: '/', 78: '-', 81: '=',
            82: '0', 83: '1', 84: '2', 85: '3', 86: '4', 87: '5',
            88: '6', 89: '7', 91: '8', 92: '9',
            
            # Special keys (Return, Tab, etc.)
            36: '\n',  # Return
            76: '\n',  # Enter (keypad)
            48: '\t',  # Tab
        }
        
        # Shift-modified characters
        self.shiftMap = {
            '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
            '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
            '-': '_', '=': '+', '[': '{', ']': '}', '\\': '|',
            ';': ':', "'": '"', ',': '<', '.': '>', '/': '?',
        }
        
        # Special function key codes
        self.deleteKey = 51        # Backspace
        self.forwardDeleteKey = 117 # Delete (forward)
        self.leftArrowKey = 123
        self.rightArrowKey = 124
        self.upArrowKey = 126
        self.downArrowKey = 125
        self.homeKey = 115
        self.endKey = 119
        
    def processEvent(self, event):
        """
        Process a keyboard event
        """
        modifiers = {'shift': event.shift, 'command': event.cmd, 'option': event.alt}
        self.processKeyCode(event.keyCode, modifiers)
        
    def processKeyCode(self, keyCode, modifiers=None):
        """
        Process a key code and update the buffer.
        
        Args:
            keyCode: The MacOS key code
            modifiers: Optional dict with modifier states (shift, command, etc.)
        
        Returns:
            bool: True if buffer was modified, False otherwise
        """
        if modifiers is None:
            modifiers = {'shift': False, 'command': False, 'option': False}
        
        # Handle special editing keys
        if keyCode == self.deleteKey:
            return self.backspace()
        elif keyCode == self.forwardDeleteKey:
            return self.delete()
        elif keyCode == self.leftArrowKey:
            return self.moveCursorLeft(modifiers.get('command', False))
        elif keyCode == self.rightArrowKey:
            return self.moveCursorRight(modifiers.get('command', False))
        elif keyCode == self.homeKey:
            return self.moveCursorToStart()
        elif keyCode == self.endKey:
            return self.moveCursorToEnd()
        
        # Handle regular character input
        elif keyCode in self.keyCodeMap:
            char = self.keyCodeMap[keyCode]
            
            # Apply shift modifier for uppercase and special chars
            if modifiers.get('shift', False):
                if char.isalpha():
                    char = char.upper()
                elif char in self.shiftMap:
                    char = self.shiftMap[char]
            
            return self.insertCharacter(char)
        
        return False
    
    def insertCharacter(self, char):
        """Insert a character at the cursor position."""
        self.buffer = (self.buffer[:self.cursorPosition] + 
                      char + 
                      self.buffer[self.cursorPosition:])
        self.cursorPosition += len(char)
        return True
    
    def backspace(self):
        """Delete character before cursor."""
        if self.cursorPosition > 0:
            self.buffer = (self.buffer[:self.cursorPosition - 1] + 
                          self.buffer[self.cursorPosition:])
            self.cursorPosition -= 1
            return True
        return False
    
    def delete(self):
        """Delete character after cursor."""
        if self.cursorPosition < len(self.buffer):
            self.buffer = (self.buffer[:self.cursorPosition] + 
                          self.buffer[self.cursorPosition + 1:])
            return True
        return False
    
    def moveCursorLeft(self, jumpToStart=False):
        """Move cursor left by one position, or to start if jumpToStart."""
        if jumpToStart:
            self.cursorPosition = 0
        elif self.cursorPosition > 0:
            self.cursorPosition -= 1
        else:
            return False
        return True
    
    def moveCursorRight(self, jumpToEnd=False):
        """Move cursor right by one position, or to end if jumpToEnd."""
        if jumpToEnd:
            self.cursorPosition = len(self.buffer)
        elif self.cursorPosition < len(self.buffer):
            self.cursorPosition += 1
        else:
            return False
        return True
    
    def moveCursorToStart(self):
        """Move cursor to the start of the buffer."""
        if self.cursorPosition != 0:
            self.cursorPosition = 0
            return True
        return False
    
    def moveCursorToEnd(self):
        """Move cursor to the end of the buffer."""
        if self.cursorPosition != len(self.buffer):
            self.cursorPosition = len(self.buffer)
            return True
        return False
    
    def getText(self):
        """Get the current buffer text."""
        return self.buffer
    
    def getTextWithCursor(self, cursorChar='|'):
        """Get the buffer text with a cursor indicator."""
        return (self.buffer[:self.cursorPosition] + 
                cursorChar + 
                self.buffer[self.cursorPosition:])
    
    def clear(self):
        """Clear the buffer and reset cursor."""
        self.buffer = ""
        self.cursorPosition = 0
    
    def setText(self, text):
        """Set the buffer to a specific text."""
        self.buffer = text
        self.cursorPosition = len(text)
    
    def getCursorPosition(self):
        """Get the current cursor position."""
        return self.cursorPosition
    
    def setCursorPosition(self, position):
        """Set the cursor position (clamped to valid range)."""
        self.cursorPosition = max(0, min(position, len(self.buffer)))

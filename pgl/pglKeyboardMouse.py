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
from .pglEventListener import pglEventListener
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

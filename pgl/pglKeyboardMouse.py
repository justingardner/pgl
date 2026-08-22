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
from .pglMessages import pglMessages

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
            warningMessage = "Accessibility permission not granted for keyboard/mouse access.\n" + \
                "On macOS, go to System Preferences -> Security & Privacy -> Privacy -> Accessibility\n" + \
                "and add your terminal application (e.g. Terminal, iTerm, etc) to the list of apps allowed to control your computer.\n" + \
                "If you are running VS Code and it already has permissions granted, try running directly from a terminal with:\n" + \
                "    /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Code"
            pglMessages.warning(warningMessage)
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
    
    def setEatAllKeys(self, eatAllKeys=False):
        '''
        Set to eat all keys or not. If set to False, will still eat any keys that setEatKeys is set to.

        Args:
            eatAllKeys (Bool): True/False eat all keys or not
        '''
        if self.isRunning():
            self.listener.setEatAllKeys(eatAllKeys)

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
            eventList.append(pglEventKeyboard(keyChar=keyChar, keyCode=keyCode, key=key, timestamp=timestamp, shift=shift, ctrl=ctrl, alt=alt, cmd=cmd, eventType="keydown"))
        return eventList
    
    

###################################
# Keyboard event
###################################
class pglEventKeyboard(pglEvent):
    """
    Represents a keyboard event for the pglKeyboard device.

    """

    def __init__(self, keyChar=None, keyCode=None, timestamp=None, key=None, shift=False, ctrl=False, alt=False, cmd=False, eventType=None):
        '''
        Initialize the pglEventKeyboard instance.
        Args:
            keyChar(str): The key that was pressed. If not passed in, but keyCode is then is derived from keyCode
            keyCode (int): The key code of the pressed key. If not passed in, but keyChar is then is derived from keyChar
            key (Key): The key object. Can be omitted
            timestamp (double): The device time. 
            shift (bool): Whether the shift key was held down. Default False
            ctrl (bool): Whether the ctrl key was held down. Default False
            alt (bool): Whether the alt key was held down. Default False
            cmd (bool): Whether the cmd key was held down. Default False
            eventType (str): The type of event ('keydown' or 'keyup'). Defaults to keydown
        Returns:
            None
        '''
        super().__init__("keyboard")

        # if keyChar is not passed in but keyCode, is then dervie it from keyCode
        if keyChar is None and keyCode is not None:
            self.keyChar = keyCodeToChar(keyCode)
        else:
            self.keyChar = keyChar
        # get keyCode (either derive from keyChar or passed in)
        if keyCode is None and keyChar is not None:
            self.keyCode = charToKeyCode(keyChar)
        else:
            self.keyCode = keyCode
        self.key = key
        self.timestamp = timestamp
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        self.cmd = cmd
        # default to keydown events
        if eventType is not None:
            self.eventType = eventType
        else:
            self.eventType = 'keydown'
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
    def __init__(self, maxLineLength=40, wrapTolerance=3):
        """
        Keyboard input buffer for text entry with automatic word wrapping.

        Args:
            maxLineLength (int):
                Preferred maximum number of characters per line.

            wrapTolerance (int):
                Number of characters that a line may be under or over
                maxLineLength when deciding where to wrap.

                Example:
                    maxLineLength=80
                    wrapTolerance=3

                The wrapper will prefer a space near character 80,
                searching within approximately 77-83 characters.

        Notes:
            - Newlines explicitly entered by the user are HARD breaks.
            - Automatic wrapping does not cross hard breaks.
            - Words are wrapped at spaces when possible.
            - If no suitable space is found, a '-' is inserted and the
              word is broken.
            - Automatically inserted newlines and hyphens are regenerated
              whenever the text changes.
            - Cursor position is maintained in logical text coordinates,
              so reflow does not move the user's cursor.
        """

        self.maxLineLength = maxLineLength
        self.wrapTolerance = wrapTolerance

        # Logical text entered by the user.
        #
        # This contains user-entered newlines, but NOT automatically
        # generated wrapping newlines or hyphens.
        self.buffer = ""

        # Cursor position in logical buffer coordinates.
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

            # Special keys
            36: '\n',
            76: '\n',
            48: '\t',
        }

        # Shift-modified characters
        self.shiftMap = {
            '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
            '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
            '-': '_', '=': '+', '[': '{', ']': '}', '\\': '|',
            ';': ':', "'": '"', ',': '<', '.': '>', '/': '?',
        }

        # Special function key codes
        self.deleteKey = 51
        self.forwardDeleteKey = 117
        self.leftArrowKey = 123
        self.rightArrowKey = 124
        self.upArrowKey = 126
        self.downArrowKey = 125
        self.homeKey = 115
        self.endKey = 119

    ##################################################################
    # Keyboard processing
    ##################################################################

    def processEvent(self, event):
        """
        Process a keyboard event.
        """
        modifiers = {
            'shift': event.shift,
            'command': event.cmd,
            'option': event.alt
        }

        return self.processKeyCode(event.keyCode, modifiers)

    def processKeyCode(self, keyCode, modifiers=None):
        """
        Process a key code and update the buffer.

        Returns:
            bool: True if buffer/cursor was modified.
        """

        if modifiers is None:
            modifiers = {
                'shift': False,
                'command': False,
                'option': False
            }

        # Editing keys
        if keyCode == self.deleteKey:
            return self.backspace()

        elif keyCode == self.forwardDeleteKey:
            return self.delete()

        elif keyCode == self.leftArrowKey:
            return self.moveCursorLeft(
                modifiers.get('command', False)
            )

        elif keyCode == self.rightArrowKey:
            return self.moveCursorRight(
                modifiers.get('command', False)
            )

        elif keyCode == self.homeKey:
            return self.moveCursorToStart()

        elif keyCode == self.endKey:
            return self.moveCursorToEnd()

        # Regular character
        elif keyCode in self.keyCodeMap:
            char = self.keyCodeMap[keyCode]

            # Apply shift
            if modifiers.get('shift', False):
                if char.isalpha():
                    char = char.upper()
                elif char in self.shiftMap:
                    char = self.shiftMap[char]

            return self.insertCharacter(char)

        return False

    ##################################################################
    # Text modification
    ##################################################################

    def insertCharacter(self, char):
        """
        Insert a character at the logical cursor position.

        Newline characters inserted here are considered USER newline
        characters and therefore become hard wrapping boundaries.
        """

        self.buffer = (
            self.buffer[:self.cursorPosition]
            + char
            + self.buffer[self.cursorPosition:]
        )

        self.cursorPosition += len(char)

        return True

    def backspace(self):
        """
        Delete the character before the logical cursor.
        """

        if self.cursorPosition > 0:

            self.buffer = (
                self.buffer[:self.cursorPosition - 1]
                + self.buffer[self.cursorPosition:]
            )

            self.cursorPosition -= 1

            return True

        return False

    def delete(self):
        """
        Delete the character after the logical cursor.
        """

        if self.cursorPosition < len(self.buffer):

            self.buffer = (
                self.buffer[:self.cursorPosition]
                + self.buffer[self.cursorPosition + 1:]
            )

            return True

        return False

    ##################################################################
    # Cursor movement
    ##################################################################

    def moveCursorLeft(self, jumpToStart=False):
        """
        Move cursor left by one logical character.

        If jumpToStart is True, move to beginning of buffer.
        """

        if jumpToStart:

            if self.cursorPosition != 0:
                self.cursorPosition = 0
                return True

            return False

        if self.cursorPosition > 0:
            self.cursorPosition -= 1
            return True

        return False

    def moveCursorRight(self, jumpToEnd=False):
        """
        Move cursor right by one logical character.

        If jumpToEnd is True, move to end of buffer.
        """

        if jumpToEnd:

            if self.cursorPosition != len(self.buffer):
                self.cursorPosition = len(self.buffer)
                return True

            return False

        if self.cursorPosition < len(self.buffer):
            self.cursorPosition += 1
            return True

        return False

    def moveCursorToStart(self):
        """
        Move cursor to start of logical buffer.
        """

        if self.cursorPosition != 0:
            self.cursorPosition = 0
            return True

        return False

    def moveCursorToEnd(self):
        """
        Move cursor to end of logical buffer.
        """

        if self.cursorPosition != len(self.buffer):
            self.cursorPosition = len(self.buffer)
            return True

        return False

    ##################################################################
    # Word wrapping
    ##################################################################

    def _wrapLine(self, line):
        """
        Wrap one logical line.

        This operates only on a line between USER-entered newlines.

        Returns:
            list[str]: Automatically wrapped lines.
        """

        if not line:
            return [""]

        if self.maxLineLength <= 0:
            return [line]

        wrapped = []
        remaining = line

        while len(remaining) > self.maxLineLength:

            # Preferred region around maxLineLength.
            minLength = max(
                1,
                self.maxLineLength - self.wrapTolerance
            )

            maxLength = min(
                len(remaining),
                self.maxLineLength + self.wrapTolerance
            )

            # Look for spaces in the preferred region.
            breakAt = None

            for i in range(maxLength, minLength - 1, -1):
                if remaining[i - 1] == ' ':
                    breakAt = i - 1
                    break

            if breakAt is not None:
                # Keep text before the space.
                wrapped.append(remaining[:breakAt])

                # Remove the wrapping space.
                remaining = remaining[breakAt + 1:]

                continue

            # No space found in preferred region.
            #
            # Look backward for ANY space before maxLineLength.
            for i in range(
                min(self.maxLineLength, len(remaining)),
                0,
                -1
            ):
                if remaining[i - 1] == ' ':
                    breakAt = i - 1
                    break

            if breakAt is not None:
                wrapped.append(remaining[:breakAt])
                remaining = remaining[breakAt + 1:]
                continue

            # No space available.
            #
            # Hard-break the word and add '-' to indicate that the
            # word was broken.
            breakLength = max(1, self.maxLineLength - 1)

            wrapped.append(
                remaining[:breakLength] + '-'
            )

            remaining = remaining[breakLength:]

        wrapped.append(remaining)

        return wrapped

    def getWrappedText(self):
        """
        Return the logical buffer with automatic wrapping applied.

        User-entered newlines are preserved exactly.
        Automatically generated newlines are inserted between wrapped
        lines.

        Returns:
            str: Display-ready wrapped text.
        """

        # Split ONLY on user-entered newlines.
        hardLines = self.buffer.split('\n')

        outputLines = []

        for line in hardLines:
            outputLines.extend(self._wrapLine(line))

        return '\n'.join(outputLines)

    ##################################################################
    # Text / display
    ##################################################################

    def getText(self):
        """
        Get the logical, unwrapped text.

        This is the text containing user-entered newlines only.
        """

        return self.buffer

    def getWrappedTextWithCursor(self, cursorChar='_'):
        """
        Get display-ready wrapped text with cursor.

        The cursor is inserted at the logical cursor position, then
        the entire text is wrapped. This keeps the cursor logically
        attached to the user's position even when reflow occurs.
        """

        before = self.buffer[:self.cursorPosition]
        after = self.buffer[self.cursorPosition:]

        # Use a sentinel that cannot normally occur in user text.
        sentinel = '\x00'

        text = before + sentinel + after

        hardLines = text.split('\n')
        outputLines = []

        for line in hardLines:

            # Find the cursor sentinel.
            if sentinel in line:

                cursorIndex = line.index(sentinel)

                beforeCursor = line[:cursorIndex]
                afterCursor = line[cursorIndex + 1:]

                # Wrap the text before the cursor and after it separately
                # so the cursor remains exactly at the logical position.
                #
                # Normally this means the cursor stays with the text
                # immediately surrounding it.
                combined = beforeCursor + sentinel + afterCursor

                wrapped = self._wrapLineWithCursor(
                    combined,
                    sentinel
                )

                outputLines.extend(wrapped)

            else:
                outputLines.extend(self._wrapLine(line))

        return '\n'.join(outputLines).replace(
            sentinel,
            cursorChar
        )

    def _wrapLineWithCursor(self, line, sentinel):
        """
        Wrap a line containing the cursor sentinel.

        The sentinel is treated as a zero-width character.
        """

        cursorIndex = line.index(sentinel)

        beforeCursor = line[:cursorIndex]
        afterCursor = line[cursorIndex + 1:]

        # Wrap the whole line while preserving the cursor.
        #
        # Since the cursor does not represent actual text, temporarily
        # replace it with a character that cannot affect word wrapping.
        marker = '\x01'

        temp = beforeCursor + marker + afterCursor

        wrapped = []
        remaining = temp

        while len(remaining.replace(marker, '')) > self.maxLineLength:

            # Number of REAL characters allowed on this line.
            target = self.maxLineLength

            # Search for a space near the target, counting the marker
            # as zero characters.
            realCount = 0
            candidateSpaces = []

            for i, char in enumerate(remaining):
                if char != marker:
                    realCount += 1

                if char == ' ':
                    candidateSpaces.append((realCount, i))

                if realCount >= target + self.wrapTolerance:
                    break

            breakIndex = None

            # Prefer the latest suitable space.
            for realCount, index in reversed(candidateSpaces):
                if (
                    target - self.wrapTolerance
                    <= realCount
                    <= target + self.wrapTolerance
                ):
                    breakIndex = index
                    break

            if breakIndex is None:
                # Look for any space before target.
                for realCount, index in reversed(candidateSpaces):
                    if realCount <= target:
                        breakIndex = index
                        break

            if breakIndex is not None:

                wrapped.append(
                    remaining[:breakIndex]
                )

                remaining = remaining[breakIndex + 1:]

            else:
                # Hard break.
                realCount = 0
                breakIndex = 0

                for i, char in enumerate(remaining):
                    if char != marker:
                        realCount += 1

                    if realCount >= max(1, target - 1):
                        breakIndex = i + 1
                        break

                chunk = remaining[:breakIndex]

                # Add hyphen only if the cursor isn't immediately
                # involved in the break.
                if marker not in chunk:
                    chunk += '-'

                wrapped.append(chunk)
                remaining = remaining[breakIndex:]

        wrapped.append(remaining)

        return wrapped

    def getTextWithCursor(self, cursorChar='|'):
        """
        Get wrapped text with cursor.

        This is intended for display.
        """

        return self.getWrappedTextWithCursor(cursorChar)

    ##################################################################
    # Buffer management
    ##################################################################

    def clear(self):
        """
        Clear the buffer and reset cursor.
        """

        self.buffer = ""
        self.cursorPosition = 0

    def setText(self, text):
        """
        Set the logical buffer text.

        Newlines in text are treated as USER-ENTERED hard breaks.
        """

        self.buffer = text
        self.cursorPosition = len(text)

    ##################################################################
    # Cursor position
    ##################################################################

    def getCursorPosition(self):
        """
        Get logical cursor position.

        This is an index into the unwrapped logical buffer.
        """

        return self.cursorPosition

    def setCursorPosition(self, position):
        """
        Set logical cursor position.

        Position is clamped to valid range.
        """

        self.cursorPosition = max(
            0,
            min(position, len(self.buffer))
        )

    ##################################################################
    # Wrapping configuration
    ##################################################################

    def setMaxLineLength(self, maxLineLength):
        """
        Change the preferred maximum line length.
        """

        if maxLineLength <= 0:
            raise ValueError(
                "maxLineLength must be greater than zero"
            )

        self.maxLineLength = maxLineLength

    def setWrapTolerance(self, wrapTolerance):
        """
        Change the allowed wrapping tolerance.
        """

        if wrapTolerance < 0:
            raise ValueError(
                "wrapTolerance must be zero or greater"
            )

        self.wrapTolerance = wrapTolerance
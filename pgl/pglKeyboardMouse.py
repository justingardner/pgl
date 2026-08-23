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

            # Normalize numeric-keypad digits to top-row digit key codes.
            keyCode = {
                82: 29, 83: 18, 84: 19, 85: 20, 86: 21,
                87: 23, 88: 22, 89: 26, 91: 28, 92: 25
            }.get(keyCode, keyCode)

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

        Numeric top-row and keypad keys are treated as equivalent.

        Args:
            keyCodes (list): List of macOS key codes to eat.
            keyChars (list): List of characters to eat, e.g. ["a", "1"].
        '''
        if not self.isRunning():
            return

        keypadNumericToTopRowKeyCode = {
            82: 29, 83: 18, 84: 19, 85: 20, 86: 21,
            87: 23, 88: 22, 89: 26, 91: 28, 92: 25,
        }

        # Start with either passed key codes or no codes.
        requestedKeyCodes = set(keyCodes or [])

        # Convert requested characters before expanding keypad equivalents.
        if keyChars is not None:
            requestedKeyCodes.update(
                charToKeyCode(keyChar)
                for keyChar in keyChars
            )

        # Normalize keypad digits to their corresponding top-row codes.
        normalizedKeyCodes = {
            keypadNumericToTopRowKeyCode.get(keyCode, keyCode)
            for keyCode in requestedKeyCodes
        }

        # Retain nonnumeric keys; include both physical forms of requested digits.
        finalKeyCodes = [
            rawKeyCode
            for rawKeyCode in (
                requestedKeyCodes | set(keypadNumericToTopRowKeyCode)
            )
            if keypadNumericToTopRowKeyCode.get(rawKeyCode, rawKeyCode)
            in normalizedKeyCodes
        ]

        self.eatKeyCodes = self.listener.setEatKeys(
            keyCodes=finalKeyCodes,
            keyChars=None
        )
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
    def __init__(
        self,
        maxLineLength=80,
        wrapTolerance=3
    ):
        """
        Keyboard input buffer with word wrapping.

        Args:
            maxLineLength (int):
                Target maximum number of characters per displayed line.

            wrapTolerance (int):
                How far a line may be under/over maxLineLength when
                choosing a word-wrap point.

        Behavior:
            - User-entered newlines are preserved.
            - Automatic wrapping occurs at spaces when possible.
            - If a word is too long, a '-' is inserted and the word
              is broken.
            - Automatic line breaks are sticky during normal typing.
            - Editing existing text causes local reflow.
            - Cursor position always refers to the underlying buffer,
              not the wrapped display representation.
        """

        self.buffer = ""
        self.cursorPosition = 0

        self.maxLineLength = maxLineLength
        self.wrapTolerance = wrapTolerance

        # Positions of user-entered newline characters.
        # These are positions in self.buffer.
        self.hardBreaks = set()

        # Positions in the underlying buffer where automatic visual
        # line breaks occur.
        #
        # A break at position N means:
        #
        #     buffer[:N]
        #     buffer[N:]
        #
        # are displayed on separate lines.
        self.wrapBreaks = set()

        # MacOS key codes
        self.keyCodeMap = {
            # Letters
            0: 'a', 1: 's', 2: 'd', 3: 'f', 4: 'h', 5: 'g', 6: 'z', 7: 'x',
            8: 'c', 9: 'v', 11: 'b', 12: 'q', 13: 'w', 14: 'e', 15: 'r',
            16: 'y', 17: 't', 31: 'o', 32: 'u', 34: 'i', 35: 'p', 37: 'l',
            38: 'j', 40: 'k', 45: 'n', 46: 'm',

            # Numbers
            18: '1', 19: '2', 20: '3', 21: '4', 23: '5', 22: '6',
            26: '7', 28: '8', 25: '9', 29: '0',

            # Special characters
            27: '-', 24: '=', 33: '[', 30: ']', 41: ';', 39: "'",
            42: '\\', 43: ',', 47: '.', 44: '/',

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

    ############################################################
    # Event processing
    ############################################################

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
            True if the buffer/cursor was modified.
        """

        if modifiers is None:
            modifiers = {
                'shift': False,
                'command': False,
                'option': False
            }

        # Backspace
        if keyCode == self.deleteKey:
            return self.backspace()

        # Forward delete
        elif keyCode == self.forwardDeleteKey:
            return self.delete()

        # Cursor movement
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

        # Up/down are intentionally not implemented here.
        # They require mapping the cursor into wrapped display lines.

        # Regular character
        elif keyCode in self.keyCodeMap:

            char = self.keyCodeMap[keyCode]

            if modifiers.get('shift', False):

                if char.isalpha():
                    char = char.upper()

                elif char in self.shiftMap:
                    char = self.shiftMap[char]

            return self.insertCharacter(char)

        return False

    ############################################################
    # Text insertion
    ############################################################

    def insertCharacter(self, char):
        """
        Insert a character at the cursor.

        Newline creates a hard/user line break.
        """

        oldPosition = self.cursorPosition

        self.buffer = (
            self.buffer[:self.cursorPosition]
            + char
            + self.buffer[self.cursorPosition:]
        )

        self.cursorPosition += len(char)

        # Adjust existing break positions because text was inserted.
        self._shiftBreaksAfterInsertion(
            oldPosition,
            len(char)
        )

        if char == '\n':
            # This is a real user-entered newline.
            self.hardBreaks.add(oldPosition)

            # Remove automatic breaks immediately around it.
            self.wrapBreaks.discard(oldPosition)
            self.wrapBreaks.discard(oldPosition + 1)

            # Reflow only the affected paragraph.
            self._reflowAroundPosition(self.cursorPosition)

        else:
            # Normal typing.
            #
            # Reflow locally only if the current line has become too long.
            self._wrapIfNecessary(self.cursorPosition)

        return True

    ############################################################
    # Backspace
    ############################################################

    def backspace(self):
        """
        Delete the character before the cursor.

        Deleting causes local reflow because text may now fit
        differently on the surrounding lines.
        """

        if self.cursorPosition <= 0:
            return False

        deletePosition = self.cursorPosition - 1

        # Was this a user-entered newline?
        wasHardBreak = deletePosition in self.hardBreaks

        # Remove character
        self.buffer = (
            self.buffer[:deletePosition]
            + self.buffer[deletePosition + 1:]
        )

        self.cursorPosition -= 1

        # Update break positions
        self._shiftBreaksAfterDeletion(deletePosition)

        if wasHardBreak:
            self.hardBreaks.remove(deletePosition)

        # Reflow affected paragraph/region.
        self._reflowAroundPosition(self.cursorPosition)

        return True

    ############################################################
    # Forward delete
    ############################################################

    def delete(self):
        """
        Delete the character at the cursor and locally reflow.
        """

        if self.cursorPosition >= len(self.buffer):
            return False

        deletePosition = self.cursorPosition

        wasHardBreak = deletePosition in self.hardBreaks

        self.buffer = (
            self.buffer[:deletePosition]
            + self.buffer[deletePosition + 1:]
        )

        self._shiftBreaksAfterDeletion(deletePosition)

        if wasHardBreak:
            self.hardBreaks.remove(deletePosition)

        self._reflowAroundPosition(self.cursorPosition)

        return True

    ############################################################
    # Break bookkeeping
    ############################################################

    def _shiftBreaksAfterInsertion(self, position, amount):
        """
        Move break positions after inserted text.
        """

        self.hardBreaks = {
            p + amount if p >= position else p
            for p in self.hardBreaks
        }

        self.wrapBreaks = {
            p + amount if p >= position else p
            for p in self.wrapBreaks
        }

    def _shiftBreaksAfterDeletion(self, position):
        """
        Move/remove break positions after deleted text.
        """

        self.hardBreaks = {
            p - 1 if p > position else p
            for p in self.hardBreaks
            if p != position
        }

        self.wrapBreaks = {
            p - 1 if p > position else p
            for p in self.wrapBreaks
            if p != position
        }

    ############################################################
    # Wrapping
    ############################################################

    def _wrapIfNecessary(self, cursorPosition):
        """
        Check the current visual line.

        If it has exceeded maxLineLength + tolerance, create
        a sticky wrap.

        Existing lines before the current line are NOT reconsidered.
        """

        lineStart = self._getLineStart(cursorPosition)
        lineEnd = self._getLineEnd(cursorPosition)

        lineLength = lineEnd - lineStart

        if lineLength <= self.maxLineLength + self.wrapTolerance:
            return

        # Find the best break point.
        breakPosition = self._findBreakPosition(
            lineStart,
            lineEnd
        )

        if breakPosition is not None:
            self.wrapBreaks.add(breakPosition)

    def _findBreakPosition(self, lineStart, lineEnd):
        """
        Find a good place to wrap.

        Prefer a space near maxLineLength.

        If no suitable space exists, break the word and insert
        a hyphen.
        """

        target = lineStart + self.maxLineLength

        # Search for spaces around target.
        lowerBound = max(
            lineStart + 1,
            target - self.wrapTolerance
        )

        upperBound = min(
            lineEnd - 1,
            target + self.wrapTolerance
        )

        candidates = []

        for position in range(lowerBound, upperBound + 1):

            if position < len(self.buffer) and self.buffer[position] == ' ':
                candidates.append(position)

        if candidates:

            # Pick the closest space to target.
            return min(
                candidates,
                key=lambda p: abs(p - target)
            )

        # No nearby space.
        #
        # Find the nearest space anywhere in the line.
        spaces = [
            p for p in range(lineStart + 1, lineEnd)
            if self.buffer[p] == ' '
        ]

        if spaces:
            return min(
                spaces,
                key=lambda p: abs(p - target)
            )

        # No space at all: this is one long word.
        #
        # We don't physically insert '-' here because that would
        # alter the user's underlying text. Instead, the display
        # representation inserts it at the wrap point.
        return target

    ############################################################
    # Local reflow
    ############################################################

    def _reflowAroundPosition(self, position):
        """
        Reflow the paragraph containing position.

        Explicit user-entered newlines remain permanent.
        """

        paragraphStart = self._getParagraphStart(position)
        paragraphEnd = self._getParagraphEnd(position)

        # Remove automatic breaks within this paragraph.
        self.wrapBreaks = {
            p for p in self.wrapBreaks
            if not (
                paragraphStart < p < paragraphEnd
            )
        }

        # Recalculate wrapping for this paragraph.
        lineStart = paragraphStart

        while lineStart < paragraphEnd:

            lineLength = paragraphEnd - lineStart

            if lineLength <= self.maxLineLength:
                break

            breakPosition = self._findBreakPosition(
                lineStart,
                paragraphEnd
            )

            if breakPosition is None:
                break

            self.wrapBreaks.add(breakPosition)

            # Continue after the break.
            lineStart = breakPosition

            # Skip the space that caused the wrap.
            if (
                lineStart < len(self.buffer)
                and self.buffer[lineStart] == ' '
            ):
                lineStart += 1

    ############################################################
    # Line / paragraph helpers
    ############################################################

    def _getParagraphStart(self, position):
        """
        Find the start of the hard-newline-delimited paragraph.
        """

        hardBreaksBefore = [
            p for p in self.hardBreaks
            if p < position
        ]

        if not hardBreaksBefore:
            return 0

        return max(hardBreaksBefore) + 1

    def _getParagraphEnd(self, position):
        """
        Find the end of the hard-newline-delimited paragraph.
        """

        hardBreaksAfter = [
            p for p in self.hardBreaks
            if p >= position
        ]

        if not hardBreaksAfter:
            return len(self.buffer)

        return min(hardBreaksAfter)

    def _getLineStart(self, position):
        """
        Find the start of the current visual line.
        """

        breaks = [
            p for p in self.wrapBreaks
            if p < position
        ]

        hardBreaks = [
            p for p in self.hardBreaks
            if p < position
        ]

        candidates = breaks + hardBreaks

        if not candidates:
            return 0

        lastBreak = max(candidates)

        # Hard newline means next line starts after newline.
        if lastBreak in self.hardBreaks:
            return lastBreak + 1

        return lastBreak

    def _getLineEnd(self, position):
        """
        Find the end of the current visual line.
        """

        breaks = [
            p for p in self.wrapBreaks
            if p > position
        ]

        hardBreaks = [
            p for p in self.hardBreaks
            if p >= position
        ]

        candidates = breaks + hardBreaks

        if not candidates:
            return len(self.buffer)

        return min(candidates)

    ############################################################
    # Cursor movement
    ############################################################

    def moveCursorLeft(self, jumpToStart=False):
        """
        Move cursor left by one position, or to beginning.
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
        Move cursor right by one position, or to end.
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
        Move cursor to beginning of buffer.
        """

        if self.cursorPosition != 0:
            self.cursorPosition = 0
            return True

        return False

    def moveCursorToEnd(self):
        """
        Move cursor to end of buffer.
        """

        if self.cursorPosition != len(self.buffer):
            self.cursorPosition = len(self.buffer)
            return True

        return False

    ############################################################
    # Text access
    ############################################################

    def getText(self):
        """
        Return the underlying user-entered text.

        Automatic wrapping is NOT included.
        """

        return self.buffer

    def getWrappedText(self):
        """
        Return text formatted for display.

        Automatic wrapping is included.
        User-entered newlines are preserved.

        Long words that must be broken receive a visual '-'.
        The '-' is NOT added to self.buffer.
        """

        if not self.buffer:
            return ""

        lines = []

        # All visual breaks.
        breaks = sorted(
            self.hardBreaks | self.wrapBreaks
        )

        start = 0

        for breakPosition in breaks:

            if breakPosition < start:
                continue

            # Hard newline.
            if breakPosition in self.hardBreaks:

                lines.append(
                    self.buffer[start:breakPosition]
                )

                start = breakPosition + 1

            else:
                # Automatic word wrap.
                line = self.buffer[start:breakPosition]

                # If the break isn't at a space, this is a
                # long-word break, so display a hyphen.
                if (
                    breakPosition < len(self.buffer)
                    and self.buffer[breakPosition] != ' '
                ):
                    line += '-'

                lines.append(line)

                # Don't display the wrapping space.
                if (
                    breakPosition < len(self.buffer)
                    and self.buffer[breakPosition] == ' '
                ):
                    start = breakPosition + 1
                else:
                    start = breakPosition

        # Remaining text
        lines.append(self.buffer[start:])

        return '\n'.join(lines)

    def getTextWithCursor(self, cursorChar='|'):
        """
        Return the wrapped display text with cursor.

        NOTE:
            The cursor is based on the underlying buffer position.
        """

        wrapped = self.getWrappedText()

        # Build display text while mapping underlying positions
        # to display positions.
        display = []
        displayPosition = 0

        for i in range(len(self.buffer) + 1):

            if i == self.cursorPosition:
                display.append(cursorChar)

            if i < len(self.buffer):
                display.append(self.buffer[i])

                # Add automatic visual newline.
                if i + 1 in self.wrapBreaks:
                    display.append('\n')

                # Replace user newline with actual newline.
                elif i in self.hardBreaks:
                    display.append('\n')

        return ''.join(display)

    ############################################################
    # Buffer management
    ############################################################

    def clear(self):
        """
        Clear buffer and all wrapping state.
        """

        self.buffer = ""
        self.cursorPosition = 0
        self.hardBreaks.clear()
        self.wrapBreaks.clear()

    def setText(self, text):
        """
        Set buffer to specific text.

        Existing newlines are treated as user-entered hard breaks.

        Automatic wrapping is recalculated.
        """

        self.buffer = text
        self.cursorPosition = len(text)

        # Every newline in supplied text is considered explicit.
        self.hardBreaks = {
            i for i, char in enumerate(text)
            if char == '\n'
        }

        self.wrapBreaks.clear()

        # Wrap each paragraph.
        start = 0

        for hardBreak in sorted(self.hardBreaks):
            self._reflowParagraph(start, hardBreak)
            start = hardBreak + 1

        self._reflowParagraph(start, len(self.buffer))

    def _reflowParagraph(self, start, end):
        """
        Wrap one paragraph.
        """

        lineStart = start

        while lineStart < end:

            if end - lineStart <= self.maxLineLength:
                break

            breakPosition = self._findBreakPosition(
                lineStart,
                end
            )

            if breakPosition is None:
                break

            self.wrapBreaks.add(breakPosition)

            lineStart = breakPosition

            if (
                lineStart < len(self.buffer)
                and self.buffer[lineStart] == ' '
            ):
                lineStart += 1

    def getCursorPosition(self):
        """
        Return cursor position in underlying buffer.
        """

        return self.cursorPosition

    def setCursorPosition(self, position):
        """
        Set cursor position, clamped to valid range.
        """

        self.cursorPosition = max(
            0,
            min(position, len(self.buffer))
        )
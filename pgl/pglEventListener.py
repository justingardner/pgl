################################################################
#   filename: pglEventListener.py
#    purpose: Python interface for macOS event listener capturing keyboard and mouse events
#         by: JLG
#       date: Feb 16, 2026
################################################################

#############
# Import modules
#############
from . import _pglEventListener
from collections import deque
from typing import Callable, Dict, List, Optional, Set
import threading
import atexit
import time

#############
# EventListener
#############
class pglEventListener:
    """
    Captures keyboard and mouse events from macOS with microsecond precision.
    
    Example:
        listener = pglEventListener()
        listener.start()
        
        # Wait for some events
        time.sleep(1)
        
        events = listener.getKeyboardEvents()
        for event in events:
            print(f"Key {event['keyCode']} at {event['timestamp']}")
        
        listener.stop()
    """
    
    def __init__(self):
        self._keyboardQueue = deque()
        self._mouseQueue = deque()
        self._keyStatus: Dict[int, float] = {}  # keyCode -> last press time
        self._lock = threading.Lock()
        self._running = False
        
        self._lastEscTime = 0.0
        self.DOUBLE_PRESS_WINDOW = 0.5
        
        # set this to prevent all keys from passing through to other applications
        self.eatAllKeys = False

        # Register cleanup
        atexit.register(self.stop)
    
    def start(self) -> None:
        """
        Start capturing events.
        
        Raises:
            PermissionError: If accessibility permissions not granted
            RuntimeError: If listener already running or thread creation failed
        """
        if self._running:
            print("(pglEventListener) Listener already running")
            return
        
        if _pglEventListener.isRunning():
            print("(pglEventListener) ❌ Another listener is already running in this process")
            self._running = False
            return
        
        _pglEventListener.start(self._eventCallback)
        self._running = True
    
    def stop(self) -> None:
        """Stop capturing events and cleanup resources."""
        if not self._running:
            return
        
        _pglEventListener.stop()
        self._running = False
        
        with self._lock:
            self._keyboardQueue.clear()
            self._mouseQueue.clear()
            self._keyStatus.clear()
            
    def isRunning(self) -> bool:
        """Check if listener is currently running."""
        return _pglEventListener.isRunning()
    
    def _eventCallback(self, event: Dict) -> bool:
        """
        Internal callback called from C extension.
        Runs in separate thread - must be thread-safe!
        """
        eventType = event.get('eventType', '')
        
        with self._lock:
            returnValue = False
            # handle ESC key for failsafe shutdown
            if eventType == 'keydown' and event.get('keyCode') == 53:
                self._handleEscPress()
                return returnValue
        
            if eventType in ('keydown', 'keyup'):
                keyCode = event['keyCode']
                
                # Update key status
                if eventType == 'keydown':
                    self._keyStatus[keyCode] = event['timestamp']
                else:  # keyup
                    self._keyStatus.pop(keyCode, None)
                
                # add to keyboard queue
                self._keyboardQueue.append(event)
                returnValue = self.eatAllKeys
                
            elif 'mouse' in eventType:
                self._mouseQueue.append(event)
        # returning False indicates that the event should be passed through
        # unless it is an eat key (which is handled in the c-code)
        return returnValue
    
    def _handleEscPress(self) -> None:
        """
        Handle ESC key press for failsafe shutdown 
        
        Single press: Turn off eat keys
        Double press (within 500ms): Shutdown listener
        """
        timestamp = time.time()
        
        if 0 < (timestamp - self._lastEscTime) <= self.DOUBLE_PRESS_WINDOW:
            # Double ESC press, call stop
            self.stop()
            self._lastEscTime = 0
        else:
            # Single ESC press, turn off eat keys
            _pglEventListener.setEatKeys([])
            self._lastEscTime = timestamp
 
    def getKeyboardEvent(self) -> Optional[Dict]:
        """
        Get the oldest keyboard event and remove it from queue.
        
        Returns:
            Event dict with keys: timestamp, eventType, keyCode, keyboardType,
            shift, control, alt, command, capsLock
            Returns None if no events available.
        """
        with self._lock:
            if self._keyboardQueue:
                return self._keyboardQueue.popleft()
            return None
    
    def getAllKeyboardEvents(self) -> List[Dict]:
        """
        Get all pending keyboard events and clear the queue.
        
        Returns:
            List of event dicts (may be empty)
        """
        with self._lock:
            events = list(self._keyboardQueue)
            self._keyboardQueue.clear()
            return events
    
    def clearQueues(self) -> None:
        """Clear any pending events in both keyboard and mouse queues."""
        with self._lock:
            self._keyboardQueue.clear()
            self._mouseQueue.clear()
            
    def getMouseEvent(self) -> Optional[Dict]:
        """
        Get the oldest mouse event and remove it from queue.
        
        Returns:
            Event dict with keys depending on eventType:
            - Mouse clicks: timestamp, eventType, button, clickState, x, y
            - Mouse moves: timestamp, eventType, x, y
            Returns None if no events available.
        """
        with self._lock:
            if self._mouseQueue:
                return self._mouseQueue.popleft()
            return None
    
    def getAllMouseEvents(self) -> List[Dict]:
        """
        Get all pending mouse events and clear the queue.
        
        Returns:
            List of event dicts (may be empty)
        """
        with self._lock:
            events = list(self._mouseQueue)
            self._mouseQueue.clear()
            return events
    
    def getKeyStatus(self) -> Dict[int, float]:
        """
        Get currently pressed keys.
        
        Returns:
            Dict mapping keyCode -> timestamp of key press
            Empty dict if no keys currently pressed
        """
        with self._lock:
            return self._keyStatus.copy()
    
    def isKeyPressed(self, keyCode: int) -> bool:
        """
        Check if a specific key is currently pressed.
        
        Args:
            keyCode: The key code to check
            
        Returns:
            True if key is currently pressed
        """
        with self._lock:
            return keyCode in self._keyStatus
    
    def setEatKeys(self, keyCodes: Optional[List[int]] = None, keyChars: Optional[List[str]] = None) -> None:
        """
        Set which key events to suppress from reaching other applications.
    
        Args:
            keyCodes: List of keyCodes to suppress. Can be None or empty list.
            keyChars: List of characters to suppress (e.g., ["a", "b", "c", "space"]). 
                      Each character is converted to its keycode.
                      
        Returns:
            List of all keyCodes that are now being suppressed.
        
        Note:
            - If only one argument provided (and not named, will interpret as keyChars if it contains strings, otherwise as keyCodes)
            - If both keyCodes and keyChars are provided, they are combined
            - If both are None/empty, clears all eaten keys
            - Suppressed keys are still captured and queued in Python, but are
            prevented from being passed to other applications
          
        Example:
            listener.setEatKeys(keyChars=["a", "b", "c", "space"])  # Eat a, b, c keys
            listener.setEatKeys(keyCodes=[49])    # Eat spacebar (keycode 49)
            listener.setEatKeys(keyCodes=[49], keyChars=["1", "2", "3"])  # Eat both
            listener.setEatKeys()  # Stop eating all keys
        """
        # Build combined list of keycodes
        allKeyCodes = []
        
        # if we only have one argument, allow flexibility of whether it is a list of keyCodes or keyChars
        if keyCodes is not None and keyChars is None:
            if all(isinstance(k, str) for k in keyCodes):
                # treat as keyStrings
                keyChars = keyCodes
                keyCodes = None
            
        # Add explicitly provided keycodes
        if keyCodes:
            allKeyCodes.extend(keyCodes)
    
        # Convert characters to keycodes
        if keyChars:
            for char in keyChars:
                keyCode = charToKeyCode(char)
                if keyCode is not None:
                    allKeyCodes.append(keyCode)
                else:
                    print(f"(pglEventListener) Warning: Could not convert '{char}' to keycode")
    
        # Remove duplicates
        print(allKeyCodes)
        allKeyCodes = list(set(allKeyCodes))
    
        # Call C extension to set which keys to suppress at OS level
        _pglEventListener.setEatKeys(allKeyCodes)
    
        if allKeyCodes:
            print(f"(pglEventListener) Eating {len(allKeyCodes)} keys: {sorted(allKeyCodes)}")
        else:
            print("(pglEventListener) Not eating any keys")
            
        return allKeyCodes
 
    def clearQueues(self) -> None:
        """Clear all pending events from both queues."""
        with self._lock:
            self._keyboardQueue.clear()
            self._mouseQueue.clear()
    
    def getQueueSizes(self) -> Dict[str, int]:
        """
        Get current queue sizes.
        
        Returns:
            Dict with keys 'keyboard' and 'mouse' containing queue lengths
        """
        with self._lock:
            return {
                'keyboard': len(self._keyboardQueue),
                'mouse': len(self._mouseQueue)
            }


# Convenience function for simple use cases
def waitForKey(timeout: Optional[float] = None) -> Optional[Dict]:
    """
    Simple blocking function to wait for a single key press.
    
    Args:
        timeout: Maximum time to wait in seconds (None = wait forever)
        
    Returns:
        Event dict or None if timeout
        
    Example:
        event = waitForKey(timeout=5.0)
        if event:
            print(f"Got key {event['keyCode']}")
    """
    import time
    
    listener = EventListener()
    listener.start()
    
    startTime = time.time()
    try:
        while True:
            event = listener.getKeyboardEvent()
            if event and event['eventType'] == 'keydown':
                return event
            
            if timeout and (time.time() - startTime) > timeout:
                return None
            
            time.sleep(0.001)  # 1ms polling
    finally:
        listener.stop()

def keyCodeToChar(keyCode: int, shift: bool = False) -> Optional[str]:
    """
    Convert macOS keycode to character representation.
    
    Args:
        keyCode: The macOS keycode
        shift: Whether shift is pressed
        
    Returns:
        Character string or special key name
    """
    # Handle list input recursively
    if isinstance(keyCode, list):
        return [keyCodeToChar(k) for k in keyCode]

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

def charToKeyCode(char: str) -> Optional[int]:
    """
    Convert character representation to macOS keycode.
    
    Args:
        char: Character string or special key name. If list, runs recursively on each element of list
        
    Returns:
        The macOS keycode, or None if not found
    """
    # Handle list input recursively
    if isinstance(char, list):
        return [self.charToKeyCode(c) for c in char]

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

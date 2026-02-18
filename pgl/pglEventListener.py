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
        self._eatKeys: Set[int] = set()
        
        self._lastEscTime = 0.0
        self.DOUBLE_PRESS_WINDOW = 0.5

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
    
    def _eventCallback(self, event: Dict) -> None:
        """
        Internal callback called from C extension.
        Runs in separate thread - must be thread-safe!
        """
        eventType = event.get('eventType', '')
        
        # handle ESC key for failsafe shutdown
        if eventType == 'keydown' and event.get('keyCode') == 53:
            self._handleEscPress()
            return
        
        with self._lock:
            if eventType in ('keydown', 'keyup'):
                keyCode = event['keyCode']
                
                # Update key status
                if eventType == 'keydown':
                    self._keyStatus[keyCode] = event['timestamp']
                else:  # keyup
                    self._keyStatus.pop(keyCode, None)
                
                # Check if we should eat this key
                if keyCode not in self._eatKeys:
                    self._keyboardQueue.append(event)
                
            elif 'mouse' in eventType:
                self._mouseQueue.append(event)
    
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
    
    def setEatKeys(self, keyCodes: Optional[List[int]] = None, keyString: Optional[str] = None) -> None:
        """
        Set which key events to suppress from reaching other applications.
    
        Args:
            keyCodes: List of keyCodes to suppress. Can be None or empty list.
            keyString: String of characters to suppress (e.g., "abc123"). 
                      Each character is converted to its keycode.
        
        Note:
            - If both keyCodes and keyString are provided, they are combined
            - If both are None/empty, clears all eaten keys
            - Suppressed keys are still captured and queued in Python, but are
            prevented from being passed to other applications
          
        Example:
            listener.setEatKeys(keyString="abc")  # Eat a, b, c keys
            listener.setEatKeys(keyCodes=[49])    # Eat spacebar (keycode 49)
            listener.setEatKeys(keyCodes=[49], keyString="123")  # Eat both
            listener.setEatKeys()  # Stop eating all keys
        """
        # Build combined list of keycodes
        allKeyCodes = []
    
        # Add explicitly provided keycodes
        if keyCodes:
            allKeyCodes.extend(keyCodes)
    
        # Convert string to keycodes
        if keyString:
            for char in keyString:
                keyCode = charToKeyCode(char)
                if keyCode is not None:
                    allKeyCodes.append(keyCode)
                else:
                    print(f"(pglEventListener) Warning: Could not convert '{char}' to keycode")
    
        # Remove duplicates
        allKeyCodes = list(set(allKeyCodes))
    
        # Call C extension to set which keys to suppress at OS level
        _pglEventListener.setEatKeys(allKeyCodes)
    
        if allKeyCodes:
            print(f"(pglEventListener) Eating {len(allKeyCodes)} keys: {sorted(allKeyCodes)}")
        else:
            print("(pglEventListener) Not eating any keys")
 
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

def charToKeyCode(char: str) -> Optional[int]:
    """
    Convert a single character to its macOS keycode.
    
    Args:
        char: Single character to convert
        
    Returns:
        Keycode (int) or None if character cannot be mapped
        
    Note:
        This uses the ANSI US keyboard layout mapping.
    """
    # macOS keycode mapping (ANSI keyboard layout)
    charMap = {
        'a': 0, 'b': 11, 'c': 8, 'd': 2, 'e': 14, 'f': 3, 'g': 5, 'h': 4,
        'i': 34, 'j': 38, 'k': 40, 'l': 37, 'm': 46, 'n': 45, 'o': 31,
        'p': 35, 'q': 12, 'r': 15, 's': 1, 't': 17, 'u': 32, 'v': 9,
        'w': 13, 'x': 7, 'y': 16, 'z': 6,
        
        'A': 0, 'B': 11, 'C': 8, 'D': 2, 'E': 14, 'F': 3, 'G': 5, 'H': 4,
        'I': 34, 'J': 38, 'K': 40, 'L': 37, 'M': 46, 'N': 45, 'O': 31,
        'P': 35, 'Q': 12, 'R': 15, 'S': 1, 'T': 17, 'U': 32, 'V': 9,
        'W': 13, 'X': 7, 'Y': 16, 'Z': 6,
        
        '0': 29, '1': 18, '2': 19, '3': 20, '4': 21, '5': 23,
        '6': 22, '7': 26, '8': 28, '9': 25,
        
        '-': 27, '=': 24, '[': 33, ']': 30, '\\': 42, ';': 41,
        "'": 39, ',': 43, '.': 47, '/': 44, '`': 50,
        
        ' ': 49,  # Space
        '\t': 48, # Tab
        '\n': 36, # Return/Enter
        '\r': 36, # Return/Enter
    }
    
    return charMap.get(char)
  
################################################################
#   filename: pglEvent.py
#    purpose: Events for recording timestamps of keyboard, mouse
#             digio and other events
#         by: JLG
#       date: July 28, 2025
################################################################

#############
# Import modules
#############
from .pglSerialize import pglSerialize
from dataclasses import dataclass, field
from typing import List, Optional

#################################################################
# Parent class for events
#################################################################
class pglEvent(pglSerialize):
    """
    Parent class for all pglEvent types
    """
    def __init__(self, type="pglEvent", **kwargs):
        """
        Initialize the pglEvent instance.
        
        Args:
            type (str): The type/category of this event
        """
        self.type = type
        
        # if the subclass event has set annotations and there
        # are passed i n kwargs, then set those fields
        fields = vars(self.__class__).get('__annotations__', {})
        # for every field in annotations
        for f in fields:
            # if it ia an input field
            if f in kwargs:
                # set the attribute
                setattr(self, f, kwargs[f])
    
    # keep a registry of pglEvent subclasses             
    _registry = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Automatically register every subclass
        pglEvent._registry[cls.__name__] = cls
        
    @classmethod
    def getClass(cls, className):
        '''
        get sub classes from classNames as created by registry above
        '''
        if className not in cls._registry:
            raise ValueError(
                f"(pglEvent:getClass) Unknown event class '{className}'"
            )
        return cls._registry[className]
    
    def __repr__(self):
        if hasattr(self, 'type'):
            return f"<pglEvent type={self.type}>"
        else:
            return f"<pglEvent type={self.__class__.__name__}>"

    def print(self):
        """
        Print the details of the pglEvent instance.
        """
        if hasattr(self, 'type'):
            print(f"(pglEvent) Type: {self.type}")
        else:
            print(f"<pglEvent type={self.__class__.__name__}>")
        for name, value in vars(self).items():
            print(f"  {name}: {value}")
#############
# pglEvents - Event container using dataclass
#############
@dataclass
class pglEvents():
    """
    Container for managing multiple pglEvent instances.

    This class provides methods for handling events related to keyboard, 
    mouse, and other device interactions.
    
    Attributes:
        events (List[pglEvent]): FIFO queue of events
    """
    events: List[pglEvent] = field(default_factory=list)
    
    ################################################################
    # Add events to the event list
    ################################################################
    def eventsAdd(self, events):
        """
        Add events to the event list.

        Args:
            events (pglEvent or list): Event(s) to add.

        Returns:
            None
        """
        if isinstance(events, list):
            self.events.extend(events)
        else:
            self.events.append(events)

    ################################################################
    # Get an event
    ################################################################
    def eventsGet(self) -> Optional[pglEvent]:
        """
        Get and remove the next event from the queue (FIFO).

        Returns:
            pglEvent or None: The next event, or None if queue is empty.
        """ 
        if self.events:
            return self.events.pop(0)
        return None
    
    ################################################################
    # eventsWaitFor
    ################################################################
    def eventsWaitFor(self, waitForList):
        """
        Wait for specific events to occur.

        Args:
            waitForList (list): A list of event types/IDs to wait for.

        Returns:
            pglEvent: The first matching event.
        """
        while True:
            events = self.poll()
            if events:
                for event in events:
                    if hasattr(event, 'id') and event.id in waitForList:
                        return event
            self.waitSecs(0.01)
            
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


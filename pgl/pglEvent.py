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
    
    def __init__(self, type="pglEvent"):
        """
        Initialize the pglEvent instance.
        
        Args:
            type (str): The type/category of this event
        """
        self.type = type

    def __repr__(self):
        return f"<pglEvent type={self.type}>"

    def print(self):
        """
        Print the details of the pglEvent instance.
        """
        print(f"(pglEvent) Type: {self.type}")

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
            events = self.devicesPoll()
            if events:
                for event in events:
                    if hasattr(event, 'id') and event.id in waitForList:
                        return event
            self.waitSecs(0.01)
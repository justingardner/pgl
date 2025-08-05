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

#############
# pglEvents
#############
class pglEvents:
    """
    Initialize the pglEvents class.

    This class provides methods for handling events related to keyboard, mouse, and other device interactions.

    Args:
        None

    Returns:
        None
    """
    def __init__(self):
        """
        Initialize the pglEvents instance.
        
        Args:
            None
        
        Returns:
            None
        """
        self.events = []

    ################################################################
    # Add events to the event list
    ################################################################
    def eventsAdd(self,events):
        """
        Add events to the event list.

        Args:
            events (list): A list of events to add.

        Returns:
            None
        """
        if isinstance(events, list):
            self.events.extend(events)
        else:
            self.events.append(events)

    ################################################################
    # Gets an event
    ################################################################
    def eventsGet(self):
        '''
        Get events from the event list (which will be populated by polling devices)

        Args:
            None

        Returns:
            pglEvent: An instance of pglEvent containing the event data.
        ''' 
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
            waitForList (list): A list of event types to wait for.

        Returns:
            pglEvent: An instance of pglEvent containing the event data.
        """
        while True:
            # Poll for events
            events = self.devicesPoll()
            if events:
                for event in events:
                    if event.id in waitForList:
                        return event
            self.waitSecs(0.01)

#################################################################
# Parent classes for events
#################################################################
class pglEvent:
    """
    Parent class for all pglEvent types
    """
    
    def __init__(self, deviceType="pglEvent"):
        """
        Initialize the pglEvent instance.
        """
        self.deviceType = deviceType

    def __repr__(self):
        return f"<pglEvent type={self.deviceType}>"

    def __del__(self):
        """
        Clean up the pglEvent instance.
        """
        # Perform any necessary cleanup here
        pass
    
    def print(self):
        """
        Print the details of the pglEvent instance.
        """
        print(f"(pglEvent) Device Type: {self.deviceType}: {self.__repr__()}")

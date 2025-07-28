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

    ################################################################
    # create an event type
    ################################################################
    def eventCreate(self,type):
        '''
        Create a new event of the specified type.

        Args:
            type (str): The type of the event to create.

        Returns:
            pglEvent: An instance of the pglEvent class representing the created event.
        ''' 
        pass

    ################################################################
    # create an event type
    ################################################################
    def eventDelete(self,event):
        '''
        Delete an existing event.

        Args:
            event (pglEvent): The event to delete.

        Returns:
            None
        ''' 
        pass

#################################################################
# Parent classes for events
#################################################################
class pglEvent:
    """
    Parent class for all pglEvent types
    """
    
    def __init__(self,pgl,type):
        """
        Initialize the pglEvent instance.
        """
        self.type = type
        self.pgl = pgl

    def __repr__(self):
        return f"<pglEvent type={self.type}>"
    
    def __del__(self):
        """
        Clean up the pglEvent instance.
        """
        # Perform any necessary cleanup here
        pass


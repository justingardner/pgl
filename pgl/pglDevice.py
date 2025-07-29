################################################################
#   filename: pglDevice.py
#    purpose: Device class
#         by: JLG
#       date: July 28, 2025
################################################################

###########
# # Import
##########
from pgl import pglTimestamp

#################################################################
# Parent class for devices
#################################################################
class pglDevice:
    """
    Parent class for all pglDevice types
    """
    def __init__(self, deviceType):  
        '''
        Initialize the _pglDevice instance.
        
        Args:
            pgl (object): The pgl instance.
            type (str): The type of the device.
                
        Returns:
            None
        '''
        # set the device type
        self.deviceType = deviceType
        # set the initialization time
        self.pglTimestamp = pglTimestamp()
        self.startTime = self.pglTimestamp.getDateAndTime()
        # set the device status
        self.currentStatus = 0
        # some fields about the device that will be set by subclasses
        self.device = None
        self.deviceAttributes = {}
        # set verbosity
        self.verbose = 1


    def __repr__(self):
        return f"<pglDevice type={self.deviceType}>"
    
    def __del__(self):
        """
        Clean up the _pglDevice instance.
        """
        # Perform any necessary cleanup here
        print(f"(pglDevice) Cleaning up device of type {self.deviceType}")
        pass

    def poll(self):
        """
        Poll the event.

        This method is used to poll the event for any updates or changes.
        Should be implemented in subclasses.

        """
        # Implement polling logic here
        return "(pglDevice) Device poll not implemented"
    
    def status(self):
        """
        Get the status of the device.

        This method retrieves the current status of the device.
        Should be implemented in subclasses.

        Returns:
            str: A string representing the current status of the device.
        """
        # Implement status retrieval logic here
        return "(pglDevice) Device status not implemented"
    

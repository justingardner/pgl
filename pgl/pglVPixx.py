################################################################
#   filename: pglVPIxx.py
#    purpose: Code for working with VPIxx devices in pgl
#         by: JLG
#       date: July 28, 2025
################################################################

#############å
# Import modules
#############
#from pgl import pglEvent
from pgl import pglDevice
from pgl import pglEvent

###################################
# DataPixx device
###################################
class pglDataPixx(pglDevice):
    """
    Represents a DataPixx device.
    """    
    def __init__(self):
        '''
        Initialize the pglDataPixx instance.
        
        Args:
            pgl (object): The pgl instance.
                
        Returns:
            None
        '''
        # set to not initialized
        self.currentStatus = -1

        # call parent constructor
        super().__init__("DataPixx")
        
        # get library
        try:
            from pypixxlib.datapixx import DATAPixx3
        except ImportError: 
            print("(pglDataPixx) pypixxlib is not installed. Please install it to use DataPixx.")
            return
        
        # Initialize the DATAPixx3 instance
        try:
            self.device = DATAPixx3()
        except Exception as e:
            print(f"(pglDataPixx) Failed to initialize DataPixx: {e}")
            self.device = None
            return  
        
        # run status to get status
        self.currentStatus = self.status()
    
    def status(self):
        """
        Get the status of the DataPixx device.

        Returns:
            int: Status code of the DataPixx device.
        """
        if self.currentStatus == -1:
            print("(pglDataPixx) DataPixx not initialized properly.")
            return self.currentStatus

        try:
            # Names of methods that are supported by the DataPixx device
            methodNames = {
                'getAssemblyRevision': 'assemblyRevision',
                'getFirmwareRevision': 'firmwareRevision',
                'getName': 'name',
                'getSerialNumber': 'serialNumber',
                'getTime': 'deviceTime'
            }
            # Get current status using the method names
            for method, attributeName in methodNames.items():
                # Use getattr to call the method dynamically on the device object
                self.deviceAttributes[attributeName] = getattr(self.device, method)()

            # get cpu time
            self.deviceAttributes['cpuTime'] = self.pglTimestamp.getSecs()

            # print current status
            if self.verbose > 0:
                print(f"(pglDataPixx) {self.deviceAttributes.get('name','Unknown DataPixx name')}")
                print(f"              serial #: {self.deviceAttributes.get('serialNumber','Unknown')}, assembly revision: {self.deviceAttributes.get('assemblyRevision','Unknown')}, firmware revision: {self.deviceAttributes.get('firmwareRevision','Unknown')}")
                print(f"              device time: {self.deviceAttributes.get('deviceTime','Unknown')}, cpu time: {self.deviceAttributes.get('cpuTime','Unknown')}")

            self.currentStatus = 1
        except Exception as e:
            print(f"(pglDataPixx) Could not get current status: {e}")
            self.currentStatus = 0
            return self.currentStatus

###################################
# ProPixx device
###################################
class pglProPixx(pglDevice):
    """
    Represents a ProPixx device.
    """    
    def __init__(self):
        '''
        Initialize the pglProPixx instance.
        
        Args:
            pgl (object): The pgl instance.
                
        Returns:
            None
        '''
        # set to not initialized
        self.currentStatus = -1

        # call parent constructor
        super().__init__("ProPixx")
        
        # get library
        try:
            from pypixxlib.propixx import PROPixx
        except ImportError: 
            print("(pglProPixx) pypixxlib is not installed. Please install it to use ProPixx.")
            return
        
        # Initialize the DATAPixx3 instance
        try:
            self.device = PROPixx()
        except Exception as e:
            print(f"(pglProPixx) Failed to initialize ProPixx: {e}")
            return
        
        # run status to get status
        self.currentStatus = self.status()
    
    def status(self):
        """
        Get the status of the ProPixx device.
        
        Returns:
            int: Status code of the ProPixx device.
        """
        if self.currentStatus == -1:
            print("(pglProPixx) ProPixx not initialized properly.")
            return self.currentStatus
        try:
            # Names of methods that are supported by the ProPixx device
            # These methods will be used to get the current status of the device
            methodNames = {
                'getAssemblyRevision': 'assemblyRevision',
                'getCoreTemperature': 'coreTemperature',
                'getDisplayResolution': 'displayResolution',
                'getDlpSequencerProgram': 'dlpSequencerProgram',
                'getFanPwm': 'fanPwm',
                'getFirmwareRevision': 'firmwareRevision',
                'getLedIntensity': 'ledIntensity',
                'getName': 'name',
                'getRamSize': 'ramSize',
                'getRasterLinePixelSync': 'rasterLinePixelSync',
                'getSerialNumber': 'serialNumber',
                'getVideoSource': 'videoSource',
                'getVideoVerticalFrameFrequency': 'videoVerticalFrameFrequency',
                'getVideoVerticalFramePeriod': 'videoVerticalFramePeriod',
                'getVideoVerticalTotal': 'videoVerticalTotal',
                'getVisibleLinePerVerticalFrame': 'visibleLinePerVerticalFrame',
                'getVisiblePixelsPerHorizontalLine': 'visiblePixelsPerHorizontalLine',
                'isActive': 'isActive',
                'isQuietMode': 'isQuietMode',
                'isReady': 'isReady',
                'isRearProjection': 'isRearProjection',
                'getTime': 'deviceTime'
            }

            # Get current status using the method names
            for method, attributeName in methodNames.items():
                # Use getattr to call the method dynamically on the device object
                self.deviceAttributes[attributeName] = getattr(self.device, method)()

            # Get CPU time
            self.deviceAttributes['cpuTime'] = self.pglTimestamp.getSecs()

            # Print current status
            if self.verbose > 0:
                print(f"(pglProPixx) {self.deviceAttributes.get('name', 'Unknown ProPixx name')}: {self.deviceAttributes.get('displayResolution', 'Unknown')} {self.deviceAttributes.get('dlpSequencerProgram', 'Unknown')}")
                print(f"             isActive: {self.deviceAttributes.get('isActive', 'Unknown')}, isQuietMode: {self.deviceAttributes.get('isQuietMode', 'Unknown')}, isReady: {self.deviceAttributes.get('isReady', 'Unknown')}, isRearProjection: {self.deviceAttributes.get('isRearProjection', 'Unknown')}")
                print(f"             core temperature: {self.deviceAttributes.get('coreTemperature', 'Unknown')}C, fan PWM: {self.deviceAttributes.get('fanPwm', 'Unknown')}, LED intensity: {self.deviceAttributes.get('ledIntensity', 'Unknown')}, video source: {self.deviceAttributes.get('videoSource', 'Unknown')}")
                print(f"             serial #: {self.deviceAttributes.get('serialNumber', 'Unknown')} assembly revision: {self.deviceAttributes.get('assemblyRevision', 'Unknown')}, firmware revision: {self.deviceAttributes.get('firmwareRevision', 'Unknown')}, ram: {self.deviceAttributes.get('ramSize', 'Unknown')}")
                print(f"             device time: {self.deviceAttributes.get('deviceTime', 'Unknown')}, cpu time: {self.deviceAttributes.get('cpuTime', 'Unknown')}")

            if self.verbose > 1:
                print(f"             video vertical frame frequency: {self.deviceAttributes.get('videoVerticalFrameFrequency', 'Unknown')}Hz, video vertical frame period: {self.deviceAttributes.get('videoVerticalFramePeriod', 'Unknown')}ms, video vertical total: {self.deviceAttributes.get('videoVerticalTotal', 'Unknown')}")
                print(f"             visible pixels per horizontal line: {self.deviceAttributes.get('visiblePixelsPerHorizontalLine', 'Unknown')}, visible lines per vertical frame: {self.deviceAttributes.get('visibleLinePerVerticalFrame', 'Unknown')}, raster line pixel sync: {self.deviceAttributes.get('rasterLinePixelSync', 'Unknown')}")
                self.currentStatus = 1
        except Exception as e:
            print(f"(pglProPixx) Could not get current status: {e}")
            self.currentStatus = 0
            return self.currentStatus
        
    def setRearProjection(self, rearProjection=True):
        '''
        Set the rear projection mode for the ProPixx device.
        
        Args:
            rearProjection (bool): True to enable rear projection, False to disable.
        '''
        if self.currentStatus == -1:
            print("(pglProPixx) ProPixx not initialized properly.")
            return

        try:
            self.device.setRearProjectionMode(rearProjection)
            self.isRearProjection = self.device.isRearProjection()
        except Exception as e:
            print(f"(pglProPixx) Could not set rear projection: {e}")
            self.currentStatus = 0

        return self.currentStatus


###################################
# ResponsePixx events (buttons)
###################################
class pglEventResponsePixx(pglEvent):
    """
    Represents a response event for the Pixx device.

    Args:
        pgl (object): The pgl instance.
        type (str): The type of the event.

    Returns:
        None
    """
    
    def __init__(self, pgl, type):
        '''
        Initialize the pglEventResponsePixx instance.
        Args:
            pgl (object): The pgl instance.
            type (str): The type of the event.
        Returns:
            None
        '''
        super().__init__(pgl, type)

        try:
            from pypixxlib.datapix import DATAPixx3
        except ImportError:
            raise ImportError("(pglEventResponsePixx) pypixxlib is not installed. Please install it to use Pixx events.")
        # Initialize the DATAPixx3 instance
        try:
            self.datapixx = DATAPixx3()
        except Exception as e:
            raise RuntimeError(f"(pglEventResponsePixx) Failed to initialize DATAPixx3: {e}")   

    def poll(self):
        """
        Poll the Pixx event.

        This method polls the Pixx device for any updates or changes.
        """
        # Implement Pixx polling logic here
        pass

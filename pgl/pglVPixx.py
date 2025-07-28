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
    def __init__(self, pgl, deviceType=None):
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
        super().__init__(pgl, deviceType or "DataPixx")
        
        # get library
        try:
            from pypixxlib.datapixx import DATAPixx3
        except ImportError: 
            print("(pglDataPixx) pypixxlib is not installed. Please install it to use DataPixx.")
            return
        
        # Initialize the DATAPixx3 instance
        try:
            self.datapixx = DATAPixx3()
        except Exception as e:
            print(f"(pglDataPixx) Failed to initialize DataPixx: {e}")
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
            # Get current status
            self.assemblyRevision = self.datapixx.getAssemblyRevision()
            self.firmwareRevision = self.datapixx.getFirmwareRevision()
            self.name = self.datapixx.getName()
            self.serialNumber = self.datapixx.getSerialNumber()

            # get device and cpu time
            self.deviceTime = self.datapixx.getTime()
            self.cpuTime = self.pgl.getSecs()

            # print current status
            if self.pgl.verbose > 0:
                print(f"(pglDataPixx) {self.name}")
                print(f"              serial #: {self.serialNumber}, assembly revision: {self.assemblyRevision}, firmware revision: {self.firmwareRevision}")
                print(f"              device time: {self.deviceTime}, cpu time: {self.cpuTime}")

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
    def __init__(self, pgl, deviceType=None):
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
        super().__init__(pgl, deviceType or "ProPixx")
        
        # get library
        try:
            from pypixxlib.propixx import PROPixx
        except ImportError: 
            print("(pglProPixx) pypixxlib is not installed. Please install it to use ProPixx.")
            return
        
        # Initialize the DATAPixx3 instance
        try:
            self.propixx = PROPixx()
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
            # Get current status
            self.assemblyRevision = self.propixx.getAssemblyRevision()
            self.coreTemperature = self.propixx.getCoreTemperature()
            self.displayResolution = self.propixx.getDisplayResolution()
            self.dlpSequencerProgram = self.propixx.getDlpSequencerProgram()
            self.fanPwm = self.propixx.getFanPwm()
            self.firmwareRevision = self.propixx.getFirmwareRevision()
            self.ledIntensity = self.propixx.getLedIntensity()
            self.name = self.propixx.getName()
            self.ramSize = self.propixx.getRamSize()
            self.rasterLinePixelSync = self.propixx.getRasterLinePixelSync()
            self.serialNumber = self.propixx.getSerialNumber()
            self.videoSource = self.propixx.getVideoSource()
            self.videoVerticalFrameFrequency = self.propixx.getVideoVerticalFrameFrequency()
            self.videoVerticalFramePeriod = self.propixx.getVideoVerticalFramePeriod()
            self.videoVerticalTotal = self.propixx.getVideoVerticalTotal()
            self.visibleLinePerVerticalFrame = self.propixx.getVisibleLinePerVerticalFrame()
            self.visiblePixelsPerHorizontalLine = self.propixx.getVisiblePixelsPerHorizontalLine()
            self.isActive = self.propixx.isActive()
            self.isQuietMode = self.propixx.isQuietMode()
            self.isReady = self.propixx.isReady()
            self.isRearProjection = self.propixx.isRearProjection()

            # get device and cpu time
            self.deviceTime = self.propixx.getTime()
            self.cpuTime = self.pgl.getSecs()

            # print current status
            if self.pgl.verbose > 0:
                print(f"(pglProPixx) {self.name}: {self.displayResolution} {self.dlpSequencerProgram}")
                print(f"             isActive: {self.isActive}, isQuietMode: {self.isQuietMode}, isReady: {self.isReady}, isRearProjection: {self.isRearProjection}")
                print(f"             core temperature: {self.coreTemperature}C, fan PWM: {self.fanPwm}, LED intensity: {self.ledIntensity}, video source: {self.videoSource}")
                print(f"             serial #: {self.serialNumber} assembly revision: {self.assemblyRevision}, firmware revision: {self.firmwareRevision}, ram: {self.ramSize}")
                print(f"             device time: {self.deviceTime}, cpu time: {self.cpuTime}")
            if self.pgl.verbose > 1:
                print(f"             video vertical frame frequency: {self.videoVerticalFrameFrequency}Hz, video vertical frame period: {self.videoVerticalFramePeriod}ms, video vertical total: {self.videoVerticalTotal}")
                print(f"             visible pixels per horizontal line: {self.visiblePixelsPerHorizontalLine}, visible lines per vertical frame: {self.visibleLinePerVerticalFrame}, raster line pixel sync: {self.rasterLinePixelSync}")
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
            self.propixx.setRearProjectionMode(rearProjection)
            self.isRearProjection = self.propixx.isRearProjection()
            self.currentStatus = self.isRearProjection
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

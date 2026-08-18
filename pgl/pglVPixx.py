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
from .pglDevice import pglDevice, pglDigitalIODevice
from .pglEvent import pglEvent
from .pglMessages import pglMessages
import numpy as np
from .pglTimestamp import pglTimestamp


###################################
# Base DataPixx device
###################################
class pglDataPixxBase(pglDevice):
    '''
    Base class for dataPixx clases which has helper functions to open close and stop schedules
    as well as initialize the library    
    '''
    def __init__(self, deviceType=None):
        '''
        init function

        Args:
            deviceType (str): String descriptive name of device
        '''
        if not deviceType: deviceType = "DataPixx"

       # call parent constructor
        super().__init__(deviceType=deviceType, deviceDescription="VPixx dataPixx")

        self.dp = None        
        # get library
        try:
            from pypixxlib.datapixx import DATAPixx3 as DATAPixx3
            from pypixxlib import _libdpx as dp

            # keep reference to libraries
            self.dp = dp        
            self.DATAPixx3 = DATAPixx3
        except ImportError: 
            pglMessages.warning("pypixxlib is not installed. Please install it to use DataPixx.")
            return

    ################################################################
    # open device for low0level DPx ibrary
    ################################################################
    def openDPx(self):
        '''
        Open datapixx device using DPx
        '''
        self.dp.DPxOpen()
        if not self.dp.DPxIsReady():
            self.getError()
            return

    
        # DPxUpdateRegCache() writes to the device and reads the registers back.
        try:
            self.dp.DPxUpdateRegCache()
            if self.getError():
                pglMessages.warning("Is DataPixx3 connected and turned on (green power light)")
                return

        except Exception as e:
            pglMessages.warning(f"DataPixx: device opened but hardware communication failed: {e}")
            return False

        # print status
        try:
            pglMessages.message(f"Opened DataPixx with firmware version: {self.dp.DPxGetFirmwareRev()}")
            if self.getError(): 
                pglMessages.warning("Is DataPixx3 connected and turned on (green power light)")
                return
            pglMessages.message(f"Current pixel mode: {self.dp.DPxIsDoutPixelMode()}")
            if self.getError():
                pglMessages.warning("Is DataPixx3 connected and turned on (green power light)")
                return
            pglMessages.message(f"DAC schedule: {self.dp.DPxIsDacSchedRunning()}")
            if self.getError():
                pglMessages.warning("Is DataPixx3 connected and turned on (green power light)")
                return
        except Exception as e:
            pglMessages.warning(f"DataPixx: device opened but hardware communication failed: {e}")
            return False
        
        
    
    ################################################################
    # close device for low-level DPX library
    ################################################################
    def closeDPx(self):
        '''
        Close datapixx device using DPx
        '''
        try:
            # close
            self.dp.DPxClose()

            pglMessages.message("Closed DataPixx DPx")
        except Exception as e:
            self.getError()

    ################################################################
    # clear schedules
    ################################################################
    def stopDPxSchedules(self):
        # close schedules
        self.dp.DPxDisableDoutPixelMode()
        self.dp.DPxDisableDoutPixelModeB()
        self.dp.DPxDisableDoutPixelModeGB()
        
        # stop all schedules
        self.dp.DPxStopAllScheds()
        
        # write changes 
        self.dp.DPxWriteRegCache()

    ################################################################
    # getError: Modified from VPIxx example code
    ################################################################
    def getError(self):
        """
        Gets any error from the DataPixx device and prints out the message
        
        Will return 0 if no error or the error number if there is an error

        """
        try:
            hasError = False
            # get error
            error = self.dp.DPxGetError()
            if error != "DPX_SUCCESS":
                errorStr = self.dp.DPxGetErrorString()
                pglMessages.warning(f"(pglDataPixx:getError) DataPixx error {error}: {errorStr}")
                hasError = True
            # clear error
            self.dp.DPxClearError()

            # detect 5V fault            
            if self.dp.DPxIs5VFault():
                pglMessages.message("5V fault detected.")
                hasError = True
            
            
        except Exception as e:
            pglMessages.warning(f"(pglDataPixx:getError) Could not get error state: {e}")
        
        return hasError
        
    ################################################################
###################################
# DataPixx device
###################################
class pglDataPixx(pglDataPixxBase):
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
        super().__init__()
        
        # Initialize the DATAPixx3 instance
        try:
            self.device = self.DATAPixx3()
        except Exception as e:
            print(f"(pglDataPixx) Failed to initialize DataPixx: {e}")
            self.device = None
            return  
        
        # button codes (hardcoded, note that these maybe different for different responsePixx devices)
        self.buttonCodes = {64528:'white left', 64513:'red left', 64514:'yellow left', 64516:'green left', 64520:'blue left', 
                            65024:'white right', 64544:'red right', 64576:'yellow right', 64640:'green right', 64768:'blue right',
                            64512:'button release'}

        
        # run status to get status
        self.currentStatus = self.status()

        # get the device start time
        self.deviceStartTime = self.deviceAttributes.get('deviceTime', 0)

        # start device log
        self.deviceLog = self.device.din.setDinLog(12e6, 1000)

        # start logging
        self.device.din.startDinLog()
        self.device.updateRegisterCache()

        # open the device
        self.openDPx()
    
    def __del__(self):
        """
        Destructor for the pglDataPixx class.
        """
        if self.device is not None and self.currentStatus != -1:
            try:
                self.device.din.stopDinLog()
                self.device.updateRegisterCache()
            except Exception as e:
                print(f"(pglDataPixx) Error during cleanup: {e}")

        # set variables to initial state
        self.device = None
        self.currentStatus = -1

        # close the device
        self.closeDPx()

    ################################################################
    # Get the status of the DataPixx device
    ################################################################
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
            self.deviceAttributes['cpuTime'] = pglTimestamp.getSecs()

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

    ################################################################
    # Poll for events
    ################################################################
    def poll(self):
        """
        Poll the DataPixx device for events.

        This method polls the DataPixx device for digital input events.
        """
        # Update register cache to get latest hardware state
        self.dp.DPxUpdateRegCache()
        
        # Get digital input log status
        # Initialize status dict with required field
        if not hasattr(self, 'dinStatus'):
            self.dinStatus = {'currentReadFrame': 0}
        
        self.dp.DPxGetDinStatus(self.dinStatus)
        newEvents = self.dinStatus.get('newLogFrames', 0)
        
        if newEvents > 0:
            # Read the digital input log data
            eventData = self.dp.DPxReadDinLog(self.dinStatus, newEvents)
            
            # Set up a list for holding events
            events = []

            for x in eventData:
                # Get the time of the event
                time = round(x[0] - self.deviceStartTime, 2)
                code = x[1]  # Digital input value
                id = self.buttonCodes.get(code, 'Unknown')
                events.append(pglEventResponsePixx(code, id, time))
            
            return events
        
        return []  # No new events
    def pollold(self):
        """
        Poll the DataPixx device for events.

        This method polls the DataPixx device for any keypad or other device events
        """
        # read device status
        self.device.din.startDinLog()
        self.device.updateRegisterCache()
        self.device.din.getDinLogStatus(self.deviceLog)
        self.device.updateRegisterCache()
        newEvents = self.deviceLog["newLogFrames"]
        
        if newEvents > 0:
            # get the eventList
            eventList = self.device.din.readDinLog(self.deviceLog, newEvents)
            # set up a list for holding events
            events = []

            for x in eventList:
                #get the time of the press, since we started logging
                time = round(x[0] - self.deviceStartTime, 2)
                code = x[1]
                id = self.buttonCodes.get(code, 'Unknown')
                events.append(pglEventResponsePixx(code, id, time))
            
            # return all the events
            return(events)

   ################################################################
    # setup digital output
    ################################################################
    def setupDigitalOutput(self,):
        """
        Setup the digital output for the DataPixx device.
        """
        # Check that we have a device
        if self.device is None or self.currentStatus==-1:
            print("(pglDataPixx:enableButtonSchedules) DataPixx device is not initialized.")
            return
        

        # Initialize the DataPixx device
        #self.openDPx()

        #self.enableVsyncTrigger()

        # enable button scheduless
        self.enableButtonSchedules()

        # enable pixel mode
        #self.enablePixelMode()

        #self.closeDPx()

        print("(pglDataPixx:enableButtonSchedules) DataPixx digital output setup complete.")

    # enableButtonSchedules: Modified from VPIxx example code
    ################################################################
    def enableButtonSchedules(self, buttonMap = None, pulseWidth=50):
        """
            Enable button schedules for the DataPixx device. Button schedules convert button press events into digital output waveforms.

            Args:
                buttonMap (string or dict): For a string, sets a default mapping of buttons to digital outputs can be: 'all', 'left', or 'right' which map
                            button presses to digital output values 0001, 0010, 0011, etc (for left, right or all buttons in order of red, yellow, green, blue, white)
                            or can be 'allPressRelease', 'leftPressRelease', 'rightPressRelease' which map button press and release events (see code for mapping)
                            For a dict, it should be a dictionary mapping button names to their corresponding digital output values. The digital output
                                  values will be converted into lines, so for example 1 = 0001, 4 = 0100, 7 = 0111 etc
                                  The button names are: redLeft, yellowLeft, greenLeft, blueLeft, whiteLeft, redRight, yellowRight, greenRight, blueRight, whiteRight
                                  For release events: redLeftRelease, yellowLeftRelease, greenLeftRelease, blueLeftRelease, whiteLeftRelease, etc.
                pulseWidth (int): The width of the pulse in milliseconds. Default is 50 ms. When tested on oscilliscope, pulses still look good even down to a microsecond in width
            Returns:
                None
        """

        allPressRelease =  {
                'redLeft': 1, 'yellowLeft': 2, 'greenLeft': 3, 'blueLeft': 4, 'whiteLeft': 5,
                'redLeftRelease': 6, 'yellowLeftRelease': 7, 'greenLeftRelease': 8, 'blueLeftRelease': 9, 'whiteLeftRelease': 10,
                'redRight': 11, 'yellowRight': 12, 'greenRight': 13, 'blueRight': 14, 'whiteRight': 15,
                'redRightRelease': 16, 'yellowRightRelease': 17, 'greenRightRelease': 18, 'blueRightRelease': 19, 'whiteRightRelease': 20
            }
        allPress = {
                'redLeft': 1, 'yellowLeft': 2, 'greenLeft': 3, 'blueLeft': 4, 'whiteLeft': 5,
                'redRight': 6, 'yellowRight': 7, 'greenRight': 8, 'blueRight': 9, 'whiteRight': 10
            }
        leftPress = {
                'redLeft': 1, 'yellowLeft': 2, 'greenLeft': 3, 'blueLeft': 4, 'whiteLeft': 5,
            }
        leftPressRelease = {
                'redLeft': 1, 'yellowLeft': 2, 'greenLeft': 3, 'blueLeft': 4, 'whiteLeft': 5,
                'redLeftRelease': 6, 'yellowLeftRelease': 7, 'greenLeftRelease': 8, 'blueLeftRelease': 9, 'whiteLeftRelease': 10
            }
        rightPressRelease = {
                'redRight': 1, 'yellowRight': 2, 'greenRight': 3, 'blueRight': 4, 'whiteRight': 5,
                'redRightRelease': 6, 'yellowRightRelease': 7, 'greenRightRelease': 8, 'blueRightRelease': 9, 'whiteRightRelease': 10
            }
        if buttonMap is None:
            # default button map
            buttonMap = allPress
        elif isinstance(buttonMap, str):
            if buttonMap.lower() in ['allpressrelease']:
                buttonMap = allPressRelease
            elif buttonMap.lower() in ['all', 'allpress']:
                buttonMap = allPress
            elif buttonMap.lower() in ['left', 'leftpress']:
                buttonMap = leftPress
            elif buttonMap.lower() in ['leftpressrelease']:
                buttonMap = leftPressRelease
            elif buttonMap.lower() in ['right', 'rightpress']:
                buttonMap = rightPress
            elif buttonMap.lower() in ['rightpressrelease']:
                buttonMap = rightPressRelease
            else:
                print(f"(pglDataPixx:enableButtonSchedules) Unknown buttonMap type: {buttonMap} (defaulting to all).")
                buttonMap = allPress
        elif not isinstance(buttonMap, dict):
            print(f"(pglDataPixx:enableButtonSchedules) buttonMap should be a string or a dictionary, got {type(buttonMap)}. Defaulting to all.")
            buttonMap = allPress

        #Create our digital output waveforms. Each button press (rising edge) triggers a
        #1 msec trig on the corresponding dout pin, followed by 2 msec on low.

        # JG: THis statement is also incorrect. the number encodes the digital word that you want to send, not the pin number.
        #     Noting the cable, then the bits to cable pin are: 17 4 16 3 15 2 14 1
        #     So, to get the pins to go high independetnly: 128=17, 64=4, 32=16, 16=3, 8=15, 4=2, 2=14, 1=1
        #
        #     Next set of bits to cable pin are: 21 8 20 7 19 6 18 5
        
        # JG: This is not what actually happens. Looks like what it actually does is related to what
        #.    Hz you set below in the DPxSetDoutSched. If it is 10 Hz, then each entry is 100ms for example.

        #JG Response Pix maps buttons to DB25 PIns as follows
        
        # LEFT
        # Red = DB25 Pin 1 (input) -> DevicePixx Digital in 0
        # Yellow = DB25 Pin 14 (input) -> DevicePixx Digital in 1
        # Green = DB25 Pin 2 (input) -> DevicePixx Digital in 2
        # Blue = DB25 Pin 15 (input) -> DevicePixx Digital in 3
        # White = DB25 Pin 3 (input) -> DevicePixx Digital in 4 (not implemented ion all contol pads, according to the manual
        
        # RIGHT
        # Red = DB25 Pin 16 (input) -> DevicePixx Digital in 5
        # Yellow = DB25 Pin 4 (input) -> DevicePixx Digital in 6
        # Green = DB25 Pin 17 (input) -> DevicePixx Digital in 7
        # Blue = DB25 Pin 5 (input) -> DevicePixx Digital in 8
        # White = DB25 Pin 18 (input) -> DevicePixx Digital in 9 (not implemented ion all contol pads, according to the manual

        #We'll use the dual /MRI as our example. DinChannels will depend on your button box type, you can use the PyPixx Digital I/O demo to verify your channel mappings.
        #Note that if PixelModeGB is enabled it will control dout 8-23, dout waveforms which try to alter these will have no effect


        redLeftWaveform = [buttonMap.get('redLeft', 0)]
        redLeftReleaseWaveform = [buttonMap.get('redLeftRelease', 0)]
        yellowLeftWaveform = [buttonMap.get('yellowLeft', 0)]
        yellowLeftReleaseWaveform = [buttonMap.get('yellowLeftRelease', 0)]        
        greenLeftWaveform = [buttonMap.get('greenLeft', 0)]
        greenLeftReleaseWaveform = [buttonMap.get('greenLeftRelease', 0)]
        blueLeftWaveform = [buttonMap.get('blueLeft', 0)]
        blueLeftReleaseWaveform = [buttonMap.get('blueLeftRelease', 0)]
        whiteLeftWaveform = [buttonMap.get('whiteLeft', 0)]
        whiteLeftReleaseWaveform = [buttonMap.get('whiteLeftRelease', 0)]

        redRightWaveform = [buttonMap.get('redRight', 0)]
        redRightReleaseWaveform = [buttonMap.get('redRightRelease', 0)]
        yellowRightWaveform = [buttonMap.get('yellowRight', 0)]
        yellowRightReleaseWaveform = [buttonMap.get('yellowRightRelease', 0)]
        greenRightWaveform = [buttonMap.get('greenRight', 0)]
        greenRightReleaseWaveform = [buttonMap.get('greenRightRelease', 0)]
        blueRightWaveform = [buttonMap.get('blueRight', 0)]
        blueRightReleaseWaveform = [buttonMap.get('blueRightRelease', 0)]
        whiteRightWaveform = [buttonMap.get('whiteRight', 0)]
        whiteRightReleaseWaveform = [buttonMap.get('whiteRightRelease', 0)]

        # 1 shows up on PIN 1 of DB25
        # 4, 6 shows up on PIN 2 of DB25 (so, pin 2 is 3rd bit)
        # 16 shows up on PIN 3 of DB25 (10000)
        #

        #Let's write the waveforms into the DPx memory. The address is set by 0 + 4096*channel_of_desired_digital_in_trigger
        buttonAddressOffset = 4096
        releaseOffset = 2048
        redLeftAddress = buttonAddressOffset*0
        redLeftReleaseAddress = buttonAddressOffset*0 + releaseOffset
        yellowLeftAddress = buttonAddressOffset*1
        yellowLeftReleaseAddress = buttonAddressOffset*1 + releaseOffset
        greenLeftAddress = buttonAddressOffset*2
        greenLeftReleaseAddress = buttonAddressOffset*2 + releaseOffset
        blueLeftAddress = buttonAddressOffset*3
        blueLeftReleaseAddress = buttonAddressOffset*3 + releaseOffset
        whiteLeftAddress = buttonAddressOffset*4
        whiteLeftReleaseAddress = buttonAddressOffset*4 + releaseOffset

        redRightAddress = buttonAddressOffset*5
        redRightReleaseAddress = buttonAddressOffset*5 + releaseOffset
        yellowRightAddress = buttonAddressOffset*6
        yellowRightReleaseAddress = buttonAddressOffset*6 + releaseOffset
        greenRightAddress = buttonAddressOffset*7
        greenRightReleaseAddress = buttonAddressOffset*7 + releaseOffset
        blueRightAddress = buttonAddressOffset*8
        blueRightReleaseAddress = buttonAddressOffset*8 + releaseOffset
        whiteRightAddress = buttonAddressOffset*9
        whiteRightReleaseAddress = buttonAddressOffset*9 + releaseOffset

        #write schedules into ram
        self.dp.DPxWriteRam(redLeftAddress, redLeftWaveform)
        self.dp.DPxWriteRam(redLeftReleaseAddress, redLeftReleaseWaveform)
        self.dp.DPxWriteRam(yellowLeftAddress, yellowLeftWaveform)
        self.dp.DPxWriteRam(yellowLeftReleaseAddress, yellowLeftReleaseWaveform)
        self.dp.DPxWriteRam(greenLeftAddress, greenLeftWaveform)
        self.dp.DPxWriteRam(greenLeftReleaseAddress, greenLeftReleaseWaveform)
        self.dp.DPxWriteRam(blueLeftAddress, blueLeftWaveform)
        self.dp.DPxWriteRam(blueLeftReleaseAddress, blueLeftReleaseWaveform)
        self.dp.DPxWriteRam(whiteLeftAddress, whiteLeftWaveform)
        self.dp.DPxWriteRam(whiteLeftReleaseAddress, whiteLeftReleaseWaveform)
        
        self.dp.DPxWriteRam(redRightAddress, redRightWaveform)
        self.dp.DPxWriteRam(redRightReleaseAddress, redRightReleaseWaveform)
        self.dp.DPxWriteRam(yellowRightAddress, yellowRightWaveform)
        self.dp.DPxWriteRam(yellowRightReleaseAddress, yellowRightReleaseWaveform)
        self.dp.DPxWriteRam(greenRightAddress, greenRightWaveform)
        self.dp.DPxWriteRam(greenRightReleaseAddress, greenRightReleaseWaveform)
        self.dp.DPxWriteRam(blueRightAddress, blueRightWaveform)
        self.dp.DPxWriteRam(blueRightReleaseAddress, blueRightReleaseWaveform)
        self.dp.DPxWriteRam(whiteRightAddress, whiteRightWaveform)
        self.dp.DPxWriteRam(whiteRightReleaseAddress, whiteRightReleaseWaveform)

        #configure buffer-- only need to configure the first one, rest will follow the same format
        self.dp.DPxSetDoutBuff(redLeftAddress, len(redLeftWaveform)*2)
        self.dp.DPxSetDoutSched(0, np.round(1000/pulseWidth).astype(int), 'hz', len(redLeftWaveform)+1)
        self.dp.DPxUpdateRegCache()

        #turn on debounce so button jitter is suppressed
        self.dp.DPxEnableDinDebounce()

        # Enable button schedules
        self.dp.DPxEnableDoutButtonSchedules()
        # Set the button schedules mode to 2 for button push and release events (1 for push only)
        self.dp.DPxSetDoutButtonSchedulesMode(2)
        self.dp.DPxWriteRegCache()

   ################################################################
    # test function, can be removed once working
    ################################################################
    def test(self):


        # Initialize the device
        #self.openDPx()

        self.stopDPxSchedules()
        # Set digital output bit(s) high
        #self.dp.DPxSetDoutValue(0xFFFFFFFF, 0xFFFFFFFF)
        #self.dp.DPxSetDoutValue(0x0, 0xFFFFFFFF)
        #self.dp.DPxUpdateRegCache()

        #value = self.dp.DPxGetDoutValue()
        #print(hex(value))



        base_address = self.dp.DPxGetDoutBuffBaseAddr()
        buffer_dout = [0xFFFF, 0]
        self.dp.DPxSetDoutBuff(base_address, 4)
        self.dp.DPxWriteRam(base_address, buffer_dout)
        self.dp.DPxSetDoutSched(0, 2, 'video', 0) 

        self.dp.DPxUpdateRegCache()

        self.dp.DPxStartDoutSched()
        self.dp.DPxUpdateRegCache()

        # Close when done
        #self.closeDPx()

###################################
# ProPixx device
###################################
class pglProPixx(pglDataPixxBase):
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
            self.deviceAttributes['cpuTime'] = pglTimestamp.getSecs()

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
    Represents a response event for ResponsePixx

    """
    
    def __init__(self, code, id, deviceTime):
        '''
        Initialize the pglEventResponsePixx instance.
        Args:
            code (int): The event code.
            id (str): The event ID.
            deviceTime (float): The device time.
        Returns:
            None
        '''
        super().__init__("ResponsePixx")
        self.code = code
        self.id = id
        self.deviceTime = deviceTime
    
    def __repr__(self):
        '''
        Return a string representation of the pglEventResponsePixx instance.
        Returns:
            str: String representation of the instance.
        '''
        return f"(pglEventResponsePixx) Code: {self.code}, ID: {self.id}, Device Time: {self.deviceTime}"

###################################
# Use datapixx as Digital IO
###################################
class pglDataPixxDigitalIODevice(pglDigitalIODevice, pglDataPixxBase):
    '''
    send digital pulses with DataPixx
    '''
    def __init__(self):
        '''
        Init
        '''
        # initialize super
        super().__init__(deviceType="DataPixxDigitalIO")

        # no configured digital channels
        self.digitalChannels = {}

        # note the hardcoded address here - this is in the sample code - eeks
        self.currentAddress = int(8e6)

        # open as DPx
        self.openDPx()

    @property
    def isActive(self):
        '''
        Make sure that dataPixx is active and ready
        '''
        if self.dp is None: 
            pglMessages.message("No DP library found")
            return False

        try:
            # Check if device is ready
            if not self.dp.DPxIsReady():
                pglMessages.warning("Device not ready")
                return False
            
            return True
            
        except Exception as e:
            pglMessages.warning(f"Error checking device: {e}")
            return False

    ################################################################
    # enablePixelMode: Modified from VPIxx example code
    ################################################################
    def enablePixelMode(self):
        """
        Enable pixel mode for the DataPixx device.
        
        This method enables the pixel mode for the DataPixx device, allowing it to control pixel-level output.
        """
        # Check that we have a device
        if self.device is None or self.currentStatus==-1:
            pglMessages.warning("DataPixx device is not initialized.",level=1)
            return
        

        # Enable pixel mode
        self.dp.DPxEnableDoutPixelModeB()
        #self.dp.DPxEnableDoutPixelModeGB()
        #self.dp.DPxEnableDoutPixelMode()
        self.dp.DPxWriteRegCache()

        pglMessages.message("Pixel mode enabled.")

 
    ################################################################
    # enableVsyncTrigger: Modified from VPIxx example code
    ################################################################
    def enableVsyncTrigger(self):
        """
        Enable vsync trigger for ProPixx
        
        This method enables the vsync trigger for the ProPixx device
        """
        pglMessages.message("(pglDataPixx:enableVsyncTrigger) vsync trigger enabled.")

        # stop currently running schedules
        self.stopDPxSchedules()

        # setup schedule
        base_address = self.dp.DPxGetDoutBuffBaseAddr()
        buffer_dout = [0xFFFF, 0]
        self.dp.DPxSetDoutBuff(base_address, 4)
        self.dp.DPxWriteRam(base_address, buffer_dout)
        self.dp.DPxSetDoutSched(0, 2, 'video', 0) 

        # updae cache
        self.dp.DPxUpdateRegCache()

        # start schedule
        self.dp.DPxStartDoutSched()
        self.dp.DPxUpdateRegCache()

    ################################################################
    # configureDigitialOuptut, adapted form vpixx documentation
    ################################################################
    def _configureDigitalOutputs(self, triggers):
        """
        Configures digital outputs based on the provided trigger dictionary and
        assigns a memory address. Each signal's bits are shifted according to the
        specified output channel, and then written to the VPixx hardware memory.
        
        Args:
            triggers (dict): Dictionary with entries in the format:
                {
                    'eventName': {'signal': [0, 1, 0, ...], 'channel': int}
                }
                Where the eventName is an aribtrary string to use to trigger the event
                the signal is the shape of the digitial pulse (e.g. 0, 1, 0 starts at 0
                rises to 1 and falls back to 0 when triggered)
                channel is the digitial channel that will be written. 

        e.g.:
            triggers = {
                "stimulusOn": {"signal": [1, 0], "channel": 8},
                "stimulusOff": {"signal": [1, 0], "channel": 1},
            }

        
        this is called by setupDigitalOutput, which allows you to just set channels in a simpler interface
        
        """
        # Loop through each event in the configuration dictionary
        for event, details in triggers.items():
            
            # Ensure the currentAddress is even; if it's odd, increment by 1
            if self.currentAddress % 2 != 0:
                self.currentAddress += 1
                
            # get details of trigger
            channel = details.get('channel')
            signal = details.get('signal', [])

            # set some new details
            details['address'] = self.currentAddress
            details['signalLength'] = len(signal)
            
            # Shift each bit in the signal to the left by the value of the channel.
            # This positions the bit correctly for the digital output channel.
            toggledSignal = [(bit << channel) for bit in signal]
            
            # Write the modified signal (toggledSignal) into the VPixx hardware memory
            # at the specified address.
            self.dp.DPxWriteRam(self.currentAddress, toggledSignal)
            
            print(f"Configured: {event} is {signal} on DOut channel {channel}")
            
            # Update the current memory address by adding the length of the signal.
            # Important to multiply by 2 to reserve enough space.
            self.currentAddress += details['signalLength'] * 2
        
        # After configuring all events, commit changes to the register cache of the
        # hardware
        self.dp.DPxWriteRegCache()

        # and update our trigger dictionary
        self.digitalChannels |= triggers

    def setupDigitalOutput(self, channel=0, pulseLen = 1, **kwargs):
        '''

        Setus up digital output for a channel. Calls _cnofigure DigitalOutputs
        
        '''
        if channel >= 0 and channel < 16:
            trigger = {channel: {"signal": [1, 0], "channel": channel}}
            self._configureDigitalOutputs(triggers=trigger)
        else:
            pglMessages.warning("Channel must be between 0 and 16: {channel}")

    def digitalOutputPulse(self, channel=0):
        '''
        Send a digital output pulse on channel

        Implementations should check self.digitalOutputConfigured first.

        Args:
            channel (int or str): Digitial channel number or name, needs to be configured

        Returns:
            timestamp (float or None): Timestamp when output was set,
                                       or None on error.
        '''
        if not self.digitalChannels:
            pglMessages.warning("Triggers are not configured, run setupDigitalOutput")
            return

        # send the specified trigger
        self._sendTrigger(channel)
    
    ################################################################
    # send a digital trigger, 
    ################################################################
    def _sendTrigger(self, channel, delay=0.0, samplingRate=1000):
        """
        Sends a digital trigger signal based on the provided dictionary entry.
        
        Parameters:
            channel (int): The key for the digitalChannels entry 
            delay (float): Delay (in seconds) before the trigger signal starts
                        (default is 0.0).
            samplingRate (int): Sampling rate in Hz for the digital output (default
                                is 10).
        """
        #import cProfile
        #import pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        # get the entry
        entry = self.digitalChannels.get(channel, None)
        if entry is None:
            pglMessages.warning(f"Could not find {channel} in configured triggers.")
            return
        
        # Determine the length of the signal (number of bits) for scheduling purposes
        signalLength = entry.get('signalLength')
        
        # Retrieve the memory address for this signal from the entry
        address = entry.get('address')
        
        # Schedule the digital output signal on the hardware:
        # - delay: when to start the signal,
        # - samplingRate: how often to sample the signal,
        # - signal_length: the duration of the signal,
        # - address: the location in memory where the signal is stored.
        self.dp.DPxSetDoutSchedule(delay, samplingRate, signalLength, address)

        #self.dp.DPxSetDoutSchedRate(1,'video')
        self.dp.DPxSetDoutSchedRate(1000,'hz')

        # Start the digital output schedule to send the trigger signal
        self.dp.DPxStartDoutSched()
        self.dp.DPxWriteRegCache()

        #profiler.disable()
        #pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)

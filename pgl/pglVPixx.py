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
import numpy as np

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
        
        # buttonCodes
        self.buttonCodes = {65528:'blue', 65522:'yellow', 65521:'red', 65524:'green', 65521:'white', 65520:'button release'}

        
        # run status to get status
        self.currentStatus = self.status()

        # set the device start time
        self.deviceStartTime = self.deviceAttributes.get('deviceTime', 0)

        # start device log
        self.deviceLog = self.device.din.setDinLog(12e6, 1000)
        # start logging
        self.device.din.startDinLog()
        self.device.updateRegisterCache()

    
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

        # Call the superclass's destructor (FIX, why can't we call this?)
        #super().__del__()

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
    ################################################################
    # Poll for events
    ################################################################
    def poll(self):
        """
        Poll the DataPixx device for events.

        This method polls the DataPixx device for any keypad or other device events
        """
        # read device status
        self.device.din.startDinLog()
        self.device.updateRegisterCache()
        self.device.din.getDinLogStatus(self.deviceLog)
        newEvents = self.deviceLog["newLogFrames"]
        
        if newEvents > 0:
            eventList = self.device.din.readDinLog(self.deviceLog, newEvents)

            for x in eventList:
               if x[1] in self.buttonCodes:
                   #look up the name of the button
                   buttonID = self.buttonCodes[x[1]]

                   #get the time of the press, since we started logging
                   time = round(x[0] - self.deviceStartTime, 2)
                   print(f'(pglDataPixx:poll) Button code: {str(x[1])}  Button ID: {buttonID}, Time: {time}')


    ################################################################
    # setup digital output
    ################################################################
    def setupDigitalOutput(self):
        """
        Setup the digital output for the DataPixx device.
        """
        # Check that we have a device
        if self.device is None or self.currentStatus==-1:
            print("(pglDataPixx:enableButtonSchedules) DataPixx device is not initialized.")
            return
        
        # load the dp library
        try:
            from pypixxlib import _libdpx as dp
        except ImportError:
            print("(pglDataPixx:enableButtonSchedules) pypixxlib is not installed. Please install it to use DataPixx button schedules.")
            return
    
        # Initialize the DataPixx device
        dp.DPxOpen()
        
        # enable button scheduless
        self.enableButtonSchedules()

        # enable pixel mode
        self.enablePixelMode()


        print("(pglDataPixx:enableButtonSchedules) DataPixx digital output setup complete.")

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
            print("(pglDataPixx:enablePixelMode) DataPixx device is not initialized.")
            return
        
        # load the dp library
        try:
            from pypixxlib import _libdpx as dp
        except ImportError:
            print("(pglDataPixx:enablePixelMode) pypixxlib is not installed. Please install it to use DataPixx pixel mode.")
            return

        # Enable pixel mode
        #dp.DPxOpen()
        dp.DPxEnableDoutPixelModeB()
        #dp.DPxEnableDoutPixelModeGB()
        #dp.DPxEnableDoutPixelMode()
        dp.DPxWriteRegCache()

        print("(pglDataPixx:enablePixelMode) Pixel mode enabled.")

    ################################################################
    # getError: Modified from VPIxx example code
    ################################################################
    def getError(self):
        """
        Gets any error from the DataPixx device and prints out the message
        
        Will return 0 if no error or the error number if there is an error

        """
        if self.device is None or self.currentStatus == -1:
            print("(pglDataPixx:getError) DataPixx device is not initialized.")
            return
        try:
            from pypixxlib import _libdpx as dp
        except ImportError:
            print("(pglDataPixx:getError) pypixxlib is not installed. Please install it to use DataPixx error handling.")
            return

        try:
            # get error
            errorNum = dp.DPxGetError()
            if errorNum != 0:
                errorStr = dp.DPxGetErrorString()
                print(f"(pglDataPixx:getError) DataPixx error {errorNum}: {errorStr}")

            # clear error
            dp.DPxClearError()
            print("(pglDataPixx:getError) Error state cleared.")
            
            if dp.DPxIs5VFault(): print("(pglDataPixx:getError) 5V fault detected.")
            
            
        except Exception as e:
            print(f"(pglDataPixx:getError) Could not get error state: {e}")
    ################################################################
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

        # load the dp library
        try:
            from pypixxlib import _libdpx as dp
        except ImportError:
            print("(pglDataPixx:enableButtonSchedules) pypixxlib is not installed. Please install it to use DataPixx button schedules.")
            return
        
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
        dp.DPxWriteRam(redLeftAddress, redLeftWaveform)
        dp.DPxWriteRam(redLeftReleaseAddress, redLeftReleaseWaveform)
        dp.DPxWriteRam(yellowLeftAddress, yellowLeftWaveform)
        dp.DPxWriteRam(yellowLeftReleaseAddress, yellowLeftReleaseWaveform)
        dp.DPxWriteRam(greenLeftAddress, greenLeftWaveform)
        dp.DPxWriteRam(greenLeftReleaseAddress, greenLeftReleaseWaveform)
        dp.DPxWriteRam(blueLeftAddress, blueLeftWaveform)
        dp.DPxWriteRam(blueLeftReleaseAddress, blueLeftReleaseWaveform)
        dp.DPxWriteRam(whiteLeftAddress, whiteLeftWaveform)
        dp.DPxWriteRam(whiteLeftReleaseAddress, whiteLeftReleaseWaveform)
        
        dp.DPxWriteRam(redRightAddress, redRightWaveform)
        dp.DPxWriteRam(redRightReleaseAddress, redRightReleaseWaveform)
        dp.DPxWriteRam(yellowRightAddress, yellowRightWaveform)
        dp.DPxWriteRam(yellowRightReleaseAddress, yellowRightReleaseWaveform)
        dp.DPxWriteRam(greenRightAddress, greenRightWaveform)
        dp.DPxWriteRam(greenRightReleaseAddress, greenRightReleaseWaveform)
        dp.DPxWriteRam(blueRightAddress, blueRightWaveform)
        dp.DPxWriteRam(blueRightReleaseAddress, blueRightReleaseWaveform)
        dp.DPxWriteRam(whiteRightAddress, whiteRightWaveform)
        dp.DPxWriteRam(whiteRightReleaseAddress, whiteRightReleaseWaveform)

        #configure buffer-- only need to configure the first one, rest will follow the same format
        dp.DPxSetDoutBuff(redLeftAddress, len(redLeftWaveform)*2)
        dp.DPxSetDoutSched(0, np.round(1000/pulseWidth).astype(int), 'hz', len(redLeftWaveform)+1)
        dp.DPxUpdateRegCache()

        #turn on debounce so button jitter is suppressed
        dp.DPxEnableDinDebounce()

        # Enable button schedules
        dp.DPxEnableDoutButtonSchedules()
        # Set the button schedules mode to 2 for button push and release events (1 for push only)
        dp.DPxSetDoutButtonSchedulesMode(2)
        dp.DPxWriteRegCache()

        
        

    ################################################################
    # test function, can be removed once working
    ################################################################
    def test(self):
        exitButton = 'blue'
        #self.deviceLog = self.device.din.setDinLog(12e6, 1000)
        self.device.din.startDinLog()
        self.device.updateRegisterCache()
        finished = False


        #let's create a loop which checks the schedule at 0.25 s intervals for button presses.
        #Any time a button press is found, we print the timestamp and button pressed.
        #If a designated exit button is pressed, we disconnect.
        while finished == False:
            #read device status
            self.device.updateRegisterCache()
            self.device.din.getDinLogStatus(self.deviceLog)
            newEvents = self.deviceLog["newLogFrames"]

            if newEvents > 0:
                eventList = self.device.din.readDinLog(self.deviceLog, newEvents)

                for x in eventList:
                    if x[1] in self.buttonCodes:
                        #look up the name of the button
                        buttonID = self.buttonCodes[x[1]]

                        #get the time of the press, since we started logging
                        time = round(x[0] - self.deviceStartTime, 2)
                        printStr = 'Button pressed! Button code: ' + str(x[1]) + ', Button ID: ' + buttonID + ', Time:' + str(time)
                        print(printStr)
                        if buttonID == exitButton:
                            finished = True
            #wait for 0.25 seconds
            self.pglTimestamp.waitSecs(0.25)
            #Finished=True

        #Stop logging
        #self.device.din.stopDinLog()
        #self.device.updateRegisterCache()


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

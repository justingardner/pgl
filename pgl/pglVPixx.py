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
    def enableButtonSchedules(self):

        # load the dp library
        try:
            from pypixxlib import _libdpx as dp
        except ImportError:
            print("(pglDataPixx:enableButtonSchedules) pypixxlib is not installed. Please install it to use DataPixx button schedules.")
            return
        
        #Create our digital output waveforms. Each button press (rising edge) triggers a
        #1 msec trig on the corresponding dout pin, followed by 2 msec on low.
        
        # JG: This is not what actually happens. Looks like what it actually does is related to what
        #.    Hz you set below in the DPxSetDoutSched. If it is 10 Hz, then each entry is 100ms for example.

        #We'll use the dual /MRI as our example. DinChannels will depend on your button box type, you can use the PyPixx Digital I/O demo to verify your channel mappings.
        #Note that if PixelModeGB is enabled it will control dout 8-23, dout waveforms which try to alter these will have no effect

        #Din0 - Red
        redWaveform = [1, 0, 0]

        #Din1 - Yellow
        yellowWaveform = [1, 0, 0]

        #Din2 - Green
        greenWaveform = [4, 0, 0]

        #Din3 - Blue
        blueWaveform = [8, 0, 0]

        #Let's write the waveforms into the DPx memory. The address is set by 0 + 4096*channel_of_desired_digital_in_trigger
        redAddress = 0+4096*0
        yellowAddress = 0+4096*1
        greenAddress = 0+4096*2
        blueAddress = 0+4096*3

        #write schedules into ram
        dp.DPxWriteRam(redAddress, redWaveform)
        dp.DPxWriteRam(yellowAddress, yellowWaveform)
        dp.DPxWriteRam(greenAddress, greenWaveform)
        dp.DPxWriteRam(blueAddress, blueWaveform)

        #configure buffer-- only need to configure the first one, rest will follow the same format
        dp.DPxSetDoutBuff(redAddress, len(redWaveform)*2)
        dp.DPxSetDoutSched(0, 20, 'hz', len(redWaveform)+1)
        dp.DPxUpdateRegCache()

        #turn on debounce so button jitter is suppressed
        dp.DPxEnableDinDebounce()

        #Enable button schedules, set mode to 1 for RPx /MRI setup
        dp.DPxEnableDoutButtonSchedules()
        dp.DPxSetDoutButtonSchedulesMode(1)
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

################################################################
#   filename: pglCalibration.py
#    purpose: Handles calibration of display
#         by: JLG
#       date: September 18, 2025
################################################################

#############
# Import modules
#############
import numpy as np
import matplotlib.pyplot as plt
from ._pglComm import pglSerial
from .pglBase import printHeader
from .pglSettings import filename, pglSettingsEditable, pglSettings, pglSettingsManager
from traitlets import Unicode, Int, Instance, Dict, Tuple, Float
from datetime import datetime
from .pglExperiment import pglExperiment
from tqdm.notebook import tqdm
from traitlets import HasTraits
from .pglSerialize import pglSerialize
from scipy.interpolate import interp1d
from .pglDevice import pglDigitalIODevice, pglAnalogTraceData
from scipy.io import loadmat

##########################
# Calibration device class
##########################
class pglLuminanceCalibrationDevice():
    '''
    Class representing a calibration device, this 
    should be subclassed for specific devices
    '''
    def __init__(self, description="", verbose=True):
        '''
        Initialize the calibration device
        '''
        self.description = description
        self.verbose = verbose

    def __del__(self):
        '''
        Destructor for the luminance calibration device.
        '''
        pass

    def measure(self):
        '''
        Should return a luminance measurement from the device
        '''
        pass
    
    def units(self):
        '''
        units that the device measures in
        '''
        return 'cd/m2'
        

# for testing
class pglLuminanceCalibrationDeviceDebug(pglLuminanceCalibrationDevice):
    '''
    Class representing a debug calibration device
    '''
    def measure(self):
        '''
        Measure the display characteristics using debug device.
        '''
        self.currentMeasurement = np.random.rand()
        print(f"(pglLuminanceCalibrationDeviceDebug) Measurement: {self.currentMeasurement}")
        return self.currentMeasurement
    
####################################################
# Minolta CS-100A calibration device class
####################################################
class pglLuminanceCalibrationDeviceMinolta(pglLuminanceCalibrationDevice):
    '''
    Class representing a Minolta CS-100A luminance / chrominance meter
    '''
    def __init__(self, description="Minolta CS-100A", verbose=True):
        '''
        Initialize the Minolta calibration.
        '''
        super().__init__(description, verbose)
        
        # tell user to connect device and turn on
        printHeader("Connect Minolta CS-100A")
        print("(pglCalibrationDeviceMinolta) Connect the Minolta CS-100A to the serial port")
        print("(pglCalibrationDeviceMinolta) hold down F key and turn it on.")
        print("(pglCalibrationDeviceMinolta) You should see the letter C on the Minolta display.")
        print("(pglCalibrationDeviceMinolta) Press Enter to continue...")
        input()
        
        # init serial port
        printHeader("Choose Serial Port")
        print("(pglCalibrationDeviceMinolta) Select the serial port that the Minolta CS-100A is connected to.")        
        print("(pglCalibrationDeviceMinolta) This should appear as something like:")
        print("(pglCalibrationDeviceMinolta)   /dev/cu.usbserial-110 - USB-Serial Controller Device")
        self.serial = pglSerial(dataLen=7, parity='e', stopBits=2, baudrate=4800, timeout=5.0)

        if self.serial.isOpen() is False:
            printHeader("Cancelled")
        else:
            printHeader("Minolta CS-100A Connected")
    def __del__(self):
        '''
        Destructor for the Minolta calibration device.
        '''
        self.serial.close()
    
    def __repr__(self):
        return f"<pglCalibrationDeviceMinolta: {self.description}>"

    def measure(self):
        '''
        Measure the display characteristics using Minolta device.
        '''
        # send measurement command
        self.serial.write("MES\r\n")
        
        # read response
        r = self.serial.read()
        
        errorCodes = {
            '00': 'Offending command',
            '01': 'Setting error',
            '11': 'Memory value error',
            '10': 'Measuring range over',
            '19': 'Display range over',
            '20': 'EEPROM error',
            '30': 'Battery exhausted'
        }
    
        try:
            # Convert bytes to string and strip whitespace/newlines
            s = r.decode('ascii').strip()
        
            # Check for error response
            if s.startswith('ER'):
                errorCode = s[2:4]
                errorMessage = errorCodes.get(errorCode, 'Unknown error')
                print(f"(pglCalibrationDeviceMinolta) ❌ Error from device: {errorMessage} (Code: {errorCode})")
                return None
        
            # Split at first comma to separate 'OK13' from the rest
            firstComma = s.index(',')
            statusAndMode = s[:firstComma]   # e.g., 'OK13'
            rest = s[firstComma+1:]            # e.g., '  10.73, .4795, .4287'
        
            # Extract status ('OK') and mode (numeric)
            status = statusAndMode[:2]
            mode = int(statusAndMode[2:])
        
            # Split the rest of the values and convert to float
            values = [float(v.strip()) for v in rest.split(',')]
            luminance, x, y = values

            if self.verbose:
                print(f"(pglCalibrationDeviceMinolta) Luminance={luminance}, x={x}, y={y} (mode={mode})")
            return luminance
        
        except Exception as e:
            print(f"(pglCalibrationDeviceMinolta) ❌ Error parsing response ({r}): {e}")
            return None
    

##########################
# Calibration class
##########################
class pglDisplayCalibration():
    '''
    Runs calibration for a display
    '''
    def __init__(self, pgl, luminanceCalibrationDevice : pglLuminanceCalibrationDevice = None, digitalIODevice : pglDigitalIODevice = None):
        '''
        Initialize the calibration.
        '''
        # set the devices
        self.addLuminanceCalibrationDevice(luminanceCalibrationDevice)
        self.addDigitalIODevice(digitalIODevice)

        # keep pgl reference
        self.pgl = pgl

        # make a dict for where to save calibration data
        self.calibrationModeSaveLocation = {
            "minsearch": "init",
            "minmax": "init",
            "full": "calibration",
            "done": "None",
        }
        
        # value for how close to min and max a value can be
        # to just be considered min or max when doing the
        # search for min and max values
        self.epsilon = 0.001
    
    def addLuminanceCalibrationDevice(self, luminanceCalibrationDevice : pglLuminanceCalibrationDevice = None):
        ''''
        add a luminanceCalibrationDevice to run luminanceCalibrations
        
        Args:
            luminanceCalibrationDevice: pglLuminanceCalibrationDevice for measuring luminance
        '''
        self.luminanceCalibrationDevice = None
        if luminanceCalibrationDevice is not None:
            # check luminance calibration device
            if not isinstance(luminanceCalibrationDevice, pglLuminanceCalibrationDevice):
                print("(pglDisplayCalibration) ❌ luminanceCalibrationDevice must be of type pglCalibrationDevice.")
                return
            self.luminanceCalibrationDevice = luminanceCalibrationDevice        
            self.luminanceCalibrationDevice.verbose = False

    def addDigitalIODevice(self, digitalIODevice : pglDigitalIODevice = None):
        ''''
        add a digitalIODevice
        
        Args:
            digitalIODevice: pglDigitalIODevice for digital and analog IO
        '''
        self.digitalIODevice = None
        # check digitalIO device
        if not isinstance(digitalIODevice, pglDigitalIODevice):
            print("(pglDisplayCalibration) ❌ digitalIODevice must be of type pglDigitalIODevice.")
        self.digitalIODevice = digitalIODevice

    @property
    def calibrationMode(self):
        '''
        Get the current calibration mode.
        '''
        return self._calibrationMode
    @calibrationMode.setter
    def calibrationMode(self, mode):
        '''
        Set the current calibration mode.
        '''
        validModes = ["minmax", "minsearch", "full", "done"]
        if mode not in validModes:
            print(f"(pglCalibration) Invalid calibration mode: {mode}. Valid modes are: {validModes}")
            return
        self._calibrationMode = mode
        # get save location for current mode
        self.saveLocation = self.calibrationModeSaveLocation.get(mode, "None")
    def __del__(self):
        '''
        Destructor for the calibration class.
        '''
        pass

    def calibrateTiming(self, settingsName):
        '''
        User faceing function which runs the timing calibration process. 
        
        You need to make sure that you have initialized with a digitalIODevice or run addDigitalIODevice
        
        Args:
            settingsName (Str): Name of the screen settings to calibrate
        '''
        if self.digitalIODevice is None:
            print("(pglDisplayCalibration) No digitalIO device specified. Please specify on initialization of pglDisplayCalibration or use addDigitalIODevice")
            return None

        self.timingCalibrationData = None
        
        # open the display with specified settings
        e = pglExperiment(self.pgl, settingsName)
        e.settings.closeScreenOnEnd = True
        e.settings.backgroundColor = (0, 0, 0)
        e.initScreen()
        if self.pgl.isOpen() is False:
            print(f"(pglCalibration:calibrate) Display {settingsName} did not open, cannot calibrate.")
            return None
        
        # initialize the data structure for saving data
        self.timingCalibrationData = pglDisplayTimingCalibrationData()

        # set information
        self.timingCalibrationData.settingsName = settingsName
        self.timingCalibrationData.settings = e.getSettings(settingsName)
        self.timingCalibrationData.deviceDescription = self.digitalIODevice.deviceDescription
        
        # get calibration date and time
        self.timingCalibrationData.creationDateTime = datetime.now()

        try:
            # get display info if available
            gpu = next(iter(self.pgl.gpuInfo.values()))
            displays = gpu.get('Displays', [])
            self.timingCalibrationData.displayInfo = displays[self.timingCalibrationData.settings.displayNumber-1]
            
            # get info from pgl
            self.timingCalibrationData.metalInfo = self.pgl.info()
            
        except Exception as ex:
            print(f"(pglDisplayCalibration:_calibrateTIming) Warning: Could not get display info: {ex}")

        # run internal function to calibrate the timing
        self.timingCalibrationData.analogTraceData = self._calibrateTiming()
        
        # and save
        self.timingCalibrationData.save()
        
        # plot data
        self.timingCalibrationData.display()

        # close screen
        e.endScreen()
    
    def _calibrateTiming(self):
        
        # set stimulus duration in frames
        # first number is pre-stimulus black
        # second number is the actual stimulus duration
        # third number is post-stimulus black
        # fourth number is inter-trial interval 
        stimulusDurationFrames = [2, 3, 5, 0]

        # size of patch to draw in the center of the screen (in degrees)
        patchWidth = self.pgl.screenWidth.deg
        patchHeight = self.pgl.screenHeight.deg

        # number of repeats
        numRepeats = 30

        # get total number of frames
        trialFrames = sum(stimulusDurationFrames)
        totalFrames = numRepeats * trialFrames

        # calculate stimulus time in seconds based on frame rate
        frameRate = self.pgl.getFrameRate()
        trialDurationSecs = trialFrames / frameRate
        totalDurationSecs = totalFrames / frameRate

        # open as full screen
        self.pgl.fullScreen(True)
        self.pgl.waitSecs(0.75)
        
        # set to black
        self.pgl.rect(0,0,patchWidth,patchHeight,0);
        self.pgl.flush()
        self.pgl.waitSecs(0.25)

        # setup digital output for sync pulse 
        self.digitalIODevice.setupDigitalOutput(channel=0)
        self.digitalIODevice.digitalOutput(0)

        # read analog input for a little bit beyond stimulus duration
        startTime = self.pgl.getSecs()
        analogReadDurationSecs = totalDurationSecs + 0.5
        self.digitalIODevice.startAnalogRead(duration=analogReadDurationSecs, channels=['AIN0','AIN1'], range=1.0)

        # display
        print(f"Total read time: {analogReadDurationSecs:.2f} seconds, (n={numRepeats}, trialLen={trialDurationSecs:.2f} s, totalDuration={totalDurationSecs:.2f} s)")

        # length of digital output pulse in ms
        pulseLenMilliseconds = 2 

        # initialize arrays of timestamps
        flushTime = []
        for iRepeat in range(numRepeats):
            trialStart = True
            currentFrame = 0
            self.pgl.rect(0,0,patchWidth,patchHeight,0);
            
            # set the digital pulse to be timelocked to the start of stimuls presentation
            #startFrameTime = pgl.getTargetPresentationTimestamp()
            #pglLabJack.digitalOutputAtTime(startFrameTime+((1/frameRate)*stimulusDurationFrames[0])/1000, 1, pulseLen=pulseLenMilliseconds)
            
            for iFrame in range(sum(stimulusDurationFrames)):
                # compute which stimulus phase we are in based on the current frame number and the stimulusDurationFrames
                stimulusPhase = np.searchsorted(np.cumsum(stimulusDurationFrames), iFrame, side='right')

                if trialStart and stimulusPhase == 1:
                    self.digitalIODevice.digitalOutput(1,pulseLen=pulseLenMilliseconds)
                    trialStart = False

                # set what color to draw the screen
                if stimulusPhase == 1:
                    self.pgl.rect(0,0,patchWidth,patchHeight,1);
                else:
                    self.pgl.rect(0,0,patchWidth,patchHeight,0);
                
                # flush the screen, recording the time of each flush for later analysis
                flushTime.append(self.pgl.flush())

        # get analog data
        analogTraceData = self.digitalIODevice.stopAnalogRead(waitToFinish=True)
        print(f"Analog read complete (duration={self.pgl.getSecs() - startTime:.2f} s). Read {analogTraceData.nSamples} samples.")

        frameTime = np.median(np.diff(flushTime))*1000
        print(f"Median frame time: {frameTime:.4f} ms (frame rate: {1000/frameTime:.2f} Hz)")
        # close full screen
        #pgl.fullScreen(False)

        # return data
        return(analogTraceData)
    
    def computeFrameOnsetDelay(self):
        # try to compute onset time
        cycleData = f['cycleData']
        # calculate standard deviation around 0 for end of data
        baseline = np.mean(data[1,-2500:])
        baselineSTD = np.std(data[1,-2500:])
        # now see when the median trace goes 2 std above the baseline
        onset = np.argmax(cycleData['median']>baseline+baselineSTD*2)

        # let's refine the search for the onset, by fitting a linear
        # finction and finding the intersection with baseline

        # let's refine the search for the onset, by fitting a linear
        # finction and finding the intersection with baseline
        print(f"baseline: {baseline} baselineSTD: {baselineSTD}")
        cycle = cycleData['median'][0]

        # set the thresholds for where we will do the linear fit.
        leftThreshold = baseline + baselineSTD * 1.5
        rightThreshold = baseline + baselineSTD * 20

        # Search backwards from onset
        leftIndex = onset
        while leftIndex > 0 and cycle[leftIndex] > leftThreshold:
            leftIndex -= 1

        # Find right point
        rightIndex = onset
        while rightIndex < len(cycle) - 1 and cycle[rightIndex] < rightThreshold:
            rightIndex += 1

        leftIndex = onset-2
        rightIndex = onset+2
        # Make sure we have enough points to fit
        if rightIndex - leftIndex < 3:
            print("Warning: fitting window too small, using fallback")
        else:
            # Fit linear function over this adaptive window
            xFit = np.arange(leftIndex, rightIndex + 1)
            yFit = cycle[leftIndex:rightIndex + 1]
            
            coeffs = np.polyfit(xFit, yFit, 1)
            m, c = coeffs
            
            # Find intercept with baseline
            if m != 0:
                onset = (baseline - c) / m
            
        print(onset)
        times = [onset + 1000*i/frameRate for i in range(stimulusDurationFrames[1]+1)]
        [plt.axvline(x=t, color='red', linestyle='--') for t in times]

        plt.ion()
        plt.figure(figsize=(14,7))
        plt.plot(cycle, 'k-', label='Median trace', linewidth=1.5)
        plt.axvline(x=58, color='r', linestyle='--', linewidth=2)
        plt.axvline(x=onset, color='g', linestyle='--', linewidth=2)
        xLineExtended = np.array([onset, rightIndex])
        yLineExtended = m * xLineExtended + c
        plt.plot(xLineExtended, yLineExtended, 'b-', linewidth=2.5, label='Fitted line', alpha=0.8)
        plt.xlim(45, 275)
        
    def calibrateLuminance(self, settingsName, nRepeats=4, nSteps=256, runValidation=True, runGammaValidation=True):
        '''
        User faceing function which runs the luminance calibration process. It will get a luminance calibration
        and then validate that you get a linear gamma (gamma=1.0) if runValidation is set and a 2.2 gamma
        for display natural images and movies if runGammaValidation is set.
        
        You need to make sure that you have initialized with a luminnaceCalibrationDevice or run addLuminanceCalibrationDevice
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
            runValidation (bool): If True, run validation after calibration.
            runGammaValidation (bool): If True, run gamma validation after calibration. This can validate
                for making a gamma of 2.2 for videos
        '''
        if self.luminanceCalibrationDevice is None:
            print("(pglDisplayCalibration) No luminance calibration device specified. Please specify on initialization of pglDisplayCalibration or use addLuminanceCalibrationDevice")
            return None

        # initialize calibration data
        self.luminanceCalibrationData = None
        self.luminanceValidationData = None
        self.luminanceGammaValidationData = None
        
        # open the display with specified settings
        e = pglExperiment(self.pgl, settingsName)
        e.settings.closeScreenOnEnd = True
        e.initScreen()
        if self.pgl.isOpen() is False:
            print(f"(pglCalibration:calibrate) Display {settingsName} did not open, cannot calibrate.")
            return None
        
        # run calibraiton
        self.luminanceCalibrationData = self._calibrateLuminance(e, settingsName, nRepeats, nSteps, validate=False)
        self.luminanceCalibrationData.display()
        
        # run validation
        if runValidation:
            self.luminanceValidationData = self.validateLuminance(e, gamma=1.0)
            self.luminanceValidationData.display(gamma=1.0)
    
        if runGammaValidation:
            self.luminanceGammaValidationData = self.validateLuminance(e, gamma=2.2)
            self.luminanceGammaValidationData.display(gamma=2.2)

        # and save
        self.saveLuminanceCalibration()
        
        # close screen
        e.endScreen()

    def validateLuminance(self, e=None, nRepeats=None, nSteps=None, gamma=1.0):
        '''
        Validate the display calibration by applying inverse gamma and re-measuring. Usually
        run automatically by calibrate() after calibration is complete.
        
        Args:
            e: The experiment variable which contains the display settings (None will start a new one)
            nRepeats (int): Number of times to repeat each measurement (None will use the same as calibration)
            nSteps (int): Number of steps in the validation. (None will use the same as calibration)
        '''
        if self.luminanceCalibrationDevice is None:
            print("(pglDisplayCalibration) No luminance calibration device specified. Please specify on initialization of pglDisplayCalibration")
            return None

        if not self.checkLuminanceCalibrationData(self.luminanceCalibrationData):
            print("(pglDisplayCalibration:validate) Calibration data is not valid. Cannot run validation.")
            return None
        print("(pglCalibration) Starting validation with inverse gamma correction...")
        
        closeScreenOnEnd = False
        if e is None:
            # open the display with specified settings
            e = pglExperiment(self.pgl, self.luminanceCalibrationData.settingsName)
            e.initScreen()
            closeScreenOnEnd = True
            if self.pgl.isOpen() is False:
                print(f"(pglCalibration:validate) Display {self.luminanceCalibrationData.settingsName} did not open, cannot validate.")
                return None
        
        # defaults for nRepeats and nSteps
        if nRepeats is None:
            nRepeats = self.luminanceCalibrationData.nRepeats
        if nSteps is None:
            nSteps = self.luminanceCalibrationData.nSteps
            
        # Calculate inverse gamma table from calibration measurements
        inverseGammaTable = self.luminanceCalibrationData.calculateInverseGamma(gamma=gamma)
        
        # Run calibration with the inverse gamma table
        luminanceValidationData = self._calibrateLuminance(
            e,
            self.luminanceCalibrationData.settingsName,
            nRepeats=nRepeats, 
            nSteps=nSteps,
            validate=True,
            inverseGammaTable=inverseGammaTable
        )
        
        # save the gamma that we were trying to achieve
        luminanceValidationData.gamma = gamma
        
        if closeScreenOnEnd:
            self.luminanceValidationData = luminanceValidationData
            e.endScreen()
            self.save()
            
        return luminanceValidationData
 
    def _calibrateLuminance(self, e, settingsName, nRepeats=4, nSteps=256, validate=False, inverseGammaTable=None):
        '''
        Measure the display characteristics.
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
            validate (bool): If True, use inverseGammaTable instead of linear gamma.
            inverseGammaTable: Tuple of (R, G, B) gamma tables to apply for validation.
        '''
        
        # initialize calibration data
        self.currentLuminanceCalibrationData = pglDisplayLuminanceCalibrationData()
        self.currentLuminanceCalibrationData.deviceDescription = self.luminanceCalibrationDevice.description
        self.currentLuminanceCalibrationData.units = self.luminanceCalibrationDevice.units()
        self.calibrationIndex = 0
        self.fullCalibrationIndex = None
        self.findMinIndex = None
        
        # no progress bar at start
        self.progressBar = None
        
        # start with trying to measure min and max
        self.calibrationMode = "minmax"
        
        # get calibration date and time
        self.currentLuminanceCalibrationData.creationDateTime = datetime.now()

        # save settings name
        self.currentLuminanceCalibrationData.settingsName = settingsName
        self.currentLuminanceCalibrationData.settings = e.getSettings(settingsName)
        
        # save the current gamma table so we can replace it
        displayNumber = self.currentLuminanceCalibrationData.settings.displayNumber
        if displayNumber > 0: displayNumber -= 1
        self.currentGammaTable = self.pgl.getGammaTable(displayNumber)
        
        # Set gamma table - either linear or inverse for validation
        if validate and inverseGammaTable is not None:
            self.pgl.setGammaTable(displayNumber, inverseGammaTable[0], inverseGammaTable[1], inverseGammaTable[2])
        else:
            self.pgl.setGammaTableLinear(displayNumber)
        
        # get the gamma table and save it
        self.currentLuminanceCalibrationData.gammaTableSize = self.pgl.getGammaTableSize(displayNumber)
        gammaTable = self.pgl.getGammaTable(displayNumber)
        self.currentLuminanceCalibrationData.gammaTable = tuple(np.array(table) for table in gammaTable)

        # set number of repeats and steps
        self.currentLuminanceCalibrationData.nRepeats = nRepeats
        self.currentLuminanceCalibrationData.nSteps = nSteps

        # get the display info
        try:
            # get display info if available
            gpu = next(iter(self.pgl.gpuInfo.values()))
            displays = gpu.get('Displays', [])
            self.currentLuminanceCalibrationData.displayInfo = displays[displayNumber]
            
            # get info from pgl
            self.currentLuminanceCalibrationData.metalInfo = self.pgl.info()
            
        except Exception as ex:
            print(f"(pglCalibration) Warning: Could not get display info: {ex}")
        
        # set the current display
        self.setDisplay()

        # now, tell operator to make sure everything is setup before continuing
        if validate:
            print("(pglCalibration) Starting validation measurement...")
            printHeader("Validating calibration with inverse gamma")
        else:
            print("(pglCalibration) Please ensure the calibration device is properly positioned and ready.")
            print("(pglCalibration) Press Enter to continue...")
            #input("")
            printHeader("Establishing min and max values")
        
        # loop to set display and make measurements
        while (self.setDisplay() != -1):
            # make a measurement
            self.measure()
            # and display
            self.displayProgress()

        # restore the original gamma table
        self.pgl.setGammaTable(displayNumber, self.currentGammaTable[0], self.currentGammaTable[1], self.currentGammaTable[2])
        
        # collect min and max measurements
        _, measurements, _, _ = self.currentLuminanceCalibrationData.getMedianMeasurements()
        self.currentLuminanceCalibrationData.minLuminance = measurements[0]
        self.currentLuminanceCalibrationData.maxLuminance = measurements[-1]
        
        # return the calibration data
        calibrationData = self.currentLuminanceCalibrationData
        self.currentLuminanceCalibrationData = None
        return(calibrationData)
    
    def checkLuminanceCalibrationData(self, luminanceCalibrationData):
        '''
        Check if the calibration data is valid.
        
        Args:
            luminanceCalibrationData: The calibration data to check.
        
        Returns:
            True if valid, False otherwise.
        '''
        
        if luminanceCalibrationData is None:
            print("(pglDisplayCalibration) No calibration data available. Please run calibrate() first.")
            return False
        
        # Check that calibration data is complete
        if not hasattr(luminanceCalibrationData, 'calibrationValues') or \
        not hasattr(luminanceCalibrationData, 'calibrationMeasurements'):
            print("(pglDisplayCalibration:checkLuminanceCalibrationData) Calibration data is incomplete. Please run calibrate() first.")
            return False
        
        # Verify the calibration data matches expected size
        expectedSize = luminanceCalibrationData.nRepeats * luminanceCalibrationData.nSteps
        if len(luminanceCalibrationData.calibrationValues) != expectedSize or \
        len(luminanceCalibrationData.calibrationMeasurements) != expectedSize:
            print(f"(pglDisplayCalibration:checkLuminanceCalibrationData) Calibration data size mismatch. Expected {expectedSize}, got {len(self.luminanceCalibrationData.calibrationValues)}")
            return False
        return True 

    def display(self):
        '''
        Display results of calibration and validation
        '''
        if self.checkLLuminanceCalibrationData(self.luminanceCalibrationData):
            self.luminanceCalibrationData.display()
        if self.checkLuminanceCalibrationData(self.luminanceValidationData):
            self.luminanceValidationData.display(gamma=1.0)
        if self.checkLuminanceCalibrationData(self.luminanceGammaValidationData):
            self.luminanceGammaValidationData.display(gamma=2.2)

    def displayProgress(self, startProgress=False):
        '''
        Display the progress of the calibration.
        '''
        if startProgress:
            self.progressBar = tqdm(total=self.currentLuminanceCalibrationData.nSteps*self.currentLuminanceCalibrationData.nRepeats, desc="Calibrating", unit="measurements")
        else:
            if self.calibrationMode == "minmax":
                print(f"(pglCalibration) {self.calibrationValueGet(-1)}: {self.calibrationMeasurementGet(-1)}")
            elif self.calibrationMode == "minsearch":
                print(f"(pglCalibration) {self.calibrationValueGet(-1)}: {self.calibrationMeasurementGet(-1)}")
            elif self.calibrationMode == "full":
                print(f"{self.calibrationMeasurementGet(-1):<6} ", end="")
                if self.progressBar is not None:
                    self.progressBar.update(1)
        
    def getNextCalibrationValue(self):
        '''
        Get the calibration values. This chooses which value to test
        based on what values have been measured so far. Sometimes
        calibration devices have a problem with low luminance measurements
        so will need to skip values if measurents fail
        '''
        # if we already have a calibration value that hasn't been measured
        # then return that
        if self.currentLuminanceCalibrationData.hasLastMeasurement(self.saveLocation) is False:
            return self.currentLuminanceCalibrationData.getLastValue(self.saveLocation)
        # check for minmax mode to get min and max values
        if self.calibrationMode == "minmax":
            return self.getMinMaxCalibrationValue()
        elif self.calibrationMode == "minsearch":
            # this mode is to try to find a measurable min value
            return self.getFindMinCalibrationValue()
        elif self.calibrationMode == "full":
            # this mode is to do a full calibration
            return self.getFullCalibrationValue()
        else:
            print("(pglCalibration) Calibration is complete.")
            return None
    
    def getMinMaxCalibrationValue(self):
        '''
        Get the min and max calibration values. This mode is to establish
        the min and max measurable values
        '''
        # first show the max value (this should always be measurable
        if self.calibrationIndex < self.currentLuminanceCalibrationData.nRepeats:
            self.calibrationValueAppend(1.0)
        # next show the min value (this may not be measurable)
        elif self.calibrationIndex < 2*self.currentLuminanceCalibrationData.nRepeats:
            self.calibrationValueAppend(0.0)
        # Check if we have valid min and max values
        else:
            # get the measured min and max
            self.maxCalibrationVal = self.calibrationValueGet(0)
            # see how many measurement failures we had for min
            nMeasurementFailures = sum(x is None for x in self.calibrationMeasurementGet(self.currentLuminanceCalibrationData.nRepeats, 2*self.currentLuminanceCalibrationData.nRepeats))
            if nMeasurementFailures == 0:
                # get the min value
                self.minCalibrationVal = self.calibrationValueGet(-1)
                # and set our mode to full calibration
                self.calibrationMode = "full"
                return self.getFullCalibrationValue()
            else:
                # this calibration mode is to try to find the min value
                self.calibrationMode = "minsearch"
                print("(pglCalibration) Minimum value was not measurable, trying to find a measureable minimum")
                return self.getFindMinCalibrationValue()
                
        # return the value that was set
        return self.calibrationValueGet(-1)

    def getFindMinCalibrationValue(self):
        '''
        Returns values that do a bisection search to find the minimum
        measurable value
        '''
        
        # if we have no measurements yet, start at 0.5
        if self.findMinIndex is None:
            # remember where we started the min search
            self.findMinIndex = self.calibrationIndex
            # set the interval to search between
            self.findMinUnmeasurableVal = 0.0
            self.findMinMeasurableVal = self.maxCalibrationVal
        else:
            # see if the last measurement was valid
            if self.calibrationMeasurementGet(-1) is not None:
                if self.calibrationValueGet(-1) < self.findMinMeasurableVal:
                    self.findMinMeasurableVal = self.calibrationValueGet(-1)
            else:
                # Unmeasurable, so replace unmeasurable value
                if self.calibrationValueGet(-1) > self.findMinUnmeasurableVal:
                    self.findMinUnmeasurableVal = self.calibrationValueGet(-1)
            # if we are within epsilon of the unmeasurable value, then accept
            if (self.findMinMeasurableVal-self.findMinUnmeasurableVal) < self.epsilon:
                print(f"(pglCalibration) Minimum measurable value found: {self.findMinMeasurableVal}")
                self.minCalibrationVal = self.findMinMeasurableVal
                self.calibrationMode = "full"
                return self.getFullCalibrationValue()
        # Check the halfway point between the current unmesurable and measurable
        newVal = self.findMinUnmeasurableVal + (self.findMinMeasurableVal - self.findMinUnmeasurableVal) / 2.0
        self.calibrationValueAppend(newVal)
        
        return self.calibrationValueGet(-1)
    
    def getFullCalibrationValue(self):
        '''
        Get the next calibration value for a full calibration
        '''

        # starting fullcalibration
        if self.fullCalibrationIndex is None:
            self.fullCalibrationIndex = self.calibrationIndex
            # display the min and max value
            printHeader(f"Starting Full Calibration between {self.minCalibrationVal} and {self.maxCalibrationVal}")
            self.displayProgress(startProgress=True)

        if ((self.calibrationIndex-self.fullCalibrationIndex) % self.currentLuminanceCalibrationData.nRepeats) == 0:
            # set the next value to measure
            self.currentStep = (self.calibrationIndex-self.fullCalibrationIndex) // self.currentLuminanceCalibrationData.nRepeats
            if self.currentStep == self.currentLuminanceCalibrationData.nSteps:
                print(f"")
                printHeader("Full calibration complete.")
                self.calibrationMode = "done"
                return -1
            value = self.minCalibrationVal + (self.maxCalibrationVal - self.minCalibrationVal) * (self.currentStep / (self.currentLuminanceCalibrationData.nSteps - 1))
            print(f"\n(pglCalibration) Measuring step {self.currentStep+1:>3d}/{self.currentLuminanceCalibrationData.nSteps}: {value:.4f} = ", end="")
            self.calibrationValueAppend(value)
        else:
            # repeat last value
            self.calibrationValueAppend(self.calibrationValueGet(-1))

        return self.calibrationValueGet(-1)

    def calibrationValueAppend(self, value):
        '''
        Append a calibration value to calibrationData
        '''
        self.currentLuminanceCalibrationData.appendValue(value, self.saveLocation)
    
    def calibrationValueGet(self, startIndex, endIndex=None):
        '''
        Get calibration value(s) from calibrationData
        '''
        if endIndex is None:
            return self.currentLuminanceCalibrationData.getValues(startIndex, None, self.saveLocation)
        else:
            return self.currentLuminanceCalibrationData.getValues(startIndex, endIndex, self.saveLocation)
    
    def calibrationMeasurementAppend(self, measurement):
        '''
        Append a calibration measurement to calibrationData
        '''
        self.currentLuminanceCalibrationData.appendMeasurement(measurement, self.saveLocation)
    
    def calibrationMeasurementGet(self, startIndex, endIndex=None):
        '''
        Get calibration measurement(s) from calibrationData
        '''
        if endIndex is None:
            return self.currentLuminanceCalibrationData.getMeasurements(startIndex, None, self.saveLocation)
        else:
            return self.currentLuminanceCalibrationData.getMeasurements(startIndex, endIndex, self.saveLocation)
    
    def setDisplay(self):
        '''
        Set the display to the calibration value
        '''
        # get the next calibration value
        value = self.getNextCalibrationValue()
        if value == -1: return -1
        
        # set the display to that value
        self.pgl.clearScreen(value)
        self.pgl.flush()
        self.pgl.clearScreen(value)
        self.pgl.flush()
        return value
    
    def measure(self): 
        '''
        Measure the display using the calibration device
        '''
        if self.luminanceCalibrationDevice is None:
            print("(pglDisplayCalibration) No calibration device specified.")

        # make measurement
        self.calibrationMeasurementAppend(self.luminanceCalibrationDevice.measure())
        
        # increment index
        self.calibrationIndex += 1
        return self.calibrationMeasurementGet(-1)

    def saveLuminanceCalibration(self):
        '''
        Save calibration data to a file.
        
        Args:
            filename (str, optional): The filename to save the calibration data to.
            If None, a default filename based on the current date will be used.
        '''
        # check calibration data
        if not self.checkLuminanceCalibrationData(self.luminanceCalibrationData):
            print("(pglCalibration) Calibration data is not valid. Cannot save.")
            return None
        
        # get the filepath
        displayName = self.luminanceCalibrationData.getDisplayName()
        filepath = self.getCalibrationFilepath(displayName, makePath=True)
        
        # save the calibration data
        self.luminanceCalibrationData.save(filename="calibration", filepath=filepath)    
        if self.checkLuminanceCalibrationData(self.luminanceValidationData):
            self.luminanceValidationData.save(filename="validation", filepath=filepath)
        if self.checkLuminanceCalibrationData(self.luminanceGammaValidationData):
            self.luminanceGammaValidationData.save(filename="gammaValidation", filepath=filepath)
        
    @staticmethod
    def chooseCalibrationFilepath(displayName, calibrationType=None, date=None):
        '''
        List the available calibration directories for a display and let the
        user choose one. Returns the chosen directory Path, or None.

        Args:
            displayName: the display name whose calibrations to list
            date: optional filter (str "YYYYMMDD", datetime, or date)
        '''
        
        filepath = pglDisplayCalibration.getCalibrationFilepath(displayName, makePath=False, calibrationType=calibrationType)
        if filepath is None:
            print(f"(pglDisplayCalibration:chooseCalibrationFilepath) Could not get filepath for calibrations")
            return
        
        # get all directories
        dirList = [d for d in filepath.iterdir() if d.is_dir()]
        
        # filter by date if requested
        if date is not None:
            if isinstance(date, str):
                dateStr = date
            elif isinstance(date, datetime):
                dateStr = date.strftime("%Y%m%d")
            elif isinstance(date, Date):
                dateStr = date.strftime("%Y%m%d")
            else:
                raise TypeError("date must be a string, datetime.date, datetime.datetime, or None")
            dirList = [d for d in dirList if d.name.startswith(dateStr)]

        # sort the directory list
        dirList = sorted(dirList, key=lambda d: d.name)

        # nothing to show
        if len(dirList) == 0:
            print(f"(pglDisplayCalibration:chooseCalibrationFilepath) ❌ No calibration directories found in: {filepath}")
            return None

        # print header
        print(f"(pglDisplayCalibration:chooseCalibrationFilepath) Calibration directory: {filepath}")

        # print each directory
        for i, d in enumerate(dirList, start=1):
            try:
                # try to parse the typical timestamp name
                dt = datetime.strptime(d.name, "%Y%m%d_%H%M%S")
                printName = dt.strftime("%A %B %-d, %Y %-I:%M%p")

                # try to read a calibration file for extra info
                calibrationFilename = calibrationDir / d.name / "calibration.json"
                if calibrationFilename.exists():
                    try:
                        calibrationData = pglSerialize.load(calibrationFilename)
                        # add some useful summary info
                        printName += f" | nSteps: {calibrationData.nSteps}"
                        printName += f" | nRepeats: {calibrationData.nRepeats}"
                        printName += f" | maxLum: {calibrationData.maxLuminance:.2f}"
                    except:
                        pass

                # add the raw folder name
                printName = f"{printName} ({d.name})"
            except:
                # not a typical name, just show it
                printName = d.name

            print(f"{i}. {printName}", flush=True)

        # ask the user to choose
        print("\nSelect a directory number: ", flush=True)
        choice = int(input())
        if choice < 1 or choice > len(dirList):
            print(f"(pglCalibration:chooseCalibrationFilepath) ❌ Invalid choice: {choice}")
            return None

        # return the chosen directory
        chosenDir = dirList[choice - 1]
        return chosenDir
    
    @staticmethod
    def getCalibrationFilepath(displayName, makePath=False, calibrationType="luminance"):
        '''
        Get default filepath for displays (usually ~/.pgl/calibrations)
        
        Args:
            displayName: the name of the display to get the filepath for
            makePath: Set to true when saving and it will append the data (and time if necessary)
                and will make the path if it does not exist
        '''
        from pgl import pgl
        
        # make the path be a directory
        settingsPath = pglSettingsManager().getCalibrationsDir()
        settingsPath = settingsPath / pgl.makeValidFilename(displayName)
        settingsPath = settingsPath / calibrationType
        
        # Check if directory exists, if so add time
        if makePath:
            # add the data
            dateStem = datetime.now().strftime("%Y%m%d")
            filepath = settingsPath / dateStem
            if filepath.exists():
                dateStem = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = settingsPath / dateStem
        else:
            filepath = settingsPath

        # Create the directory
        if makePath:
            filepath.mkdir(parents=True, exist_ok=True)
        else:
            # check that the path exists
            if not filepath.exists() or not filepath.is_dir():
                print(f"(pglDisplayCalibration:getCalibrationFilepath) ❌ Could not find calibration directory: {filepath}")
                return None
        
        return(filepath)

# Calibration settings, subclass of pglSettings to inherit load/save functionality
class pglDisplayTimingCalibrationData(HasTraits, pglSerialize):
    
    settingsName = Unicode("Default", help="Settings name used to open display")
    settings = Instance(pglSettings, allow_none=True, help="Settings used during calibration") 
    displayInfo = Dict(help="Display information at time of calibration")
    metalInfo = Dict(help="PGL info including display info such as UUID, serial number, and other information at time of calibration")   
    creationDateTime = Instance(datetime, default_value=datetime.now(), help="Date and time of calibration creation")
    deviceDescription = Unicode("Unknown", help="Desciption of measurement device")
    analogTraceData = Instance(pglAnalogTraceData, allow_none=True, default_value=None, help="analong measurement from photodiode")
    syncChannel = Int(1, help="Channel with sync pulse on it")     
    syncChannelThreshold = Float(0.2, help="Threshold for considering sync to be active")
    
    def display(self):
        '''
        display the timing calibration data
        '''
        if self.analogTraceData is None:
            print(f"(pglDisplayTimingCalibrationData:No data to display)")
            return
        
        # display the analog traces        
        self.analogTraceData.display(digitalSyncChannel=self.syncChannel, digitalSyncThreshold=self.syncChannelThreshold, ignoreInitial=0)

    def save(self, filename=None, filepath=None):
        '''
        save
        
        Args:
            filename: If none defaults to calibration.json
            filepath: If none defaults to calibrationDir / display name / data
        '''
        # get a filepath
        if filepath is None:
            filepath = pglDisplayCalibration.getCalibrationFilepath(self.getDisplayName(), makePath=True, calibrationType="timing")
        
        # get the filename
        if filename is None:
            filename = "timing"
                           
        # call parent to save
        super().save(filepath / filename)
    
    def getDisplayName(self):
        '''
        Get the display name associated with this data
        '''
        return self.displayInfo.get("DisplayName",self.settingsName)
    
    def getUUID(self):
        '''
        Get the UUID associated with this data
        '''
        return self.metalInfo.get("display.uuid","Unknown")

        
# Calibration settings, subclass of pglSettings to inherit load/save functionality
class pglDisplayLuminanceCalibrationData(HasTraits, pglSerialize):
    
    settingsName = Unicode("Default", help="Settings name used to open display")
    displayInfo = Dict(help="Display information at time of calibration")
    settings = Instance(pglSettings, allow_none=True, help="Settings used during calibration") 
    metalInfo = Dict(help="PGL info including display info such as UUID, serial number, and other information at time of calibration")   
    gammaTableSize = Int(-1, help="Size of the gamma table at time of calibration")
    gammaTable = Tuple(Instance(np.ndarray), Instance(np.ndarray), Instance(np.ndarray), allow_none=True, help="Gamma table at time of calibration")
    creationDateTime = Instance(datetime, default_value=datetime.now(), help="Date and time of calibration creation")
    nRepeats = Int(4, help="Number of repeats per calibration value")
    nSteps = Int(256, help="Number of steps in the calibration")
    initValues = Instance(np.ndarray, allow_none=True, help="Initial display values used in calibration, if any, used for finding min and max values")
    initMeasurements = Instance(np.ndarray, allow_none=True, help="Measured luminance values corresponding to initValues calibration")
    calibrationValues = Instance(np.ndarray, allow_none=True, help="Display values used in calibration")
    calibrationMeasurements = Instance(np.ndarray, allow_none=True, help="Measured luminance values from calibration")
    minLuminance = Float(-1.0, help="Minimum measured luminance from calibration")
    maxLuminance = Float(-1.0, help="Maximum measured luminance from calibration")
    gamma = Float(0.0, help="If not 0 then this is a validation run trying to achieve this gamma - i.e. 1.0 = linear")
    units = Unicode("cd/m2", help="Units that were used for measurement")
    deviceDescription = Unicode("Unknown", help="Desciption of measurement device")
    
    def getDisplayName(self):
        '''
        Get the display name associated with this data
        '''
        return self.displayInfo.get("DisplayName",self.settingsName)
    
    def getUUID(self):
        '''
        Get the UUID associated with this data
        '''
        return self.metalInfo.get("display.uuid","Unknown")
        
        
    def save(self, filename=None, filepath=None):
        '''
        save
        
        Args:
            filename: If none defaults to calibration.json
            filepath: If none defaults to calibrationDir / display name / data
        '''
        # get a filepath
        if filepath is None:
            filepath = pglDisplayCalibration.getCalibrationFilepath(self.getDisplayName(), makePath=True, calibrationType="luminance")
        
        # get the filename
        if filename is None:
            filename = "calibration"
                           
        # call parent to save
        super().save(filepath / filename)
    
    @classmethod
    def load(cls, displayName, filename=None, filepath=None, date=None):
        '''
        load
        
        Args:
            displayName: displayName to load
            filename: If none defaults to calibration.json
            filepath: If none defaults to calibrationDir / display name / data
        '''
        # get a filepath
        if filepath is None:
            filepath = pglDisplayCalibration.chooseCalibrationFilepath(displayName, date=date, calibrationType="luminance")
            if filepath is None: return
        # get the filename
        if filename is None:
            filename = "calibration"
        
        print(f"(pglDisplayLuminanceCalibrationData:load) Loading {filepath / filename}")
        
        # call super load to instantiate the object and load it                
        return super(pglDisplayLuminanceCalibrationData, cls).load(filepath / filename)
    
    @classmethod
    def loadMatlab(cls, filename):
        '''
        load a calibration from mgl Matlab file
        
        Args:
            filename: The filename of the .mat calibration file
        '''
        try:
            matData = loadmat(filename, squeeze_me=True, struct_as_record=False)
        except Exception as loadError:
            print(f"Failed to load {filename}: {loadError}")
            return None

        # check for calib strcuture
        calib = matData.get('calib', None)
        if calib is None:
            print(f"(pglDisplayLuminanceCalibrationData:loadMatlab) Could not find calib structure in: {filename}")
            return None
        
        # instantiate the class
        c = cls()
        
        # get creation time
        try:
            c.creationDateTime = datetime.strptime(calib.date, '%d-%b-%Y %H:%M:%S')
        except (AttributeError, ValueError) as dateError:
            print(f"(pglDisplayLuminanceCalibrationData:loadMatlab) Could not parse date: {dateError}")
            c.creationDateTime = None
            
        # set num repeats, because we only have the stored median value, always set to 1
        c.nRepeats = 1
        
        try:
            c.calibrationValues = np.array(calib.uncorrected.outputValues).T
            c.calibrationMeasurements = np.array(calib.uncorrected.luminance).T
            c.nSteps = len(c.calibrationValues)
        except (AttributeError, ValueError) as e:
            print(f"(pglDisplayLuminanceCalibrationData:loadMatlab) ❌❌❌ Could not parse calibration❌❌❌: {e}")
            return None
        
        # get min and max luminance
        c.minLuminance = np.min(c.calibrationMeasurements)
        c.maxLuminance = np.max(c.calibrationMeasurements)

        # get the monitor ID and set as DisplayName  
        try:
            c.displayInfo["DisplayName"] = calib.monitor.ID
        except (AttributeError, ValueError) as e:
            print(f"(pglDisplayLuminanceCalibrationData:loadMatlab) Could not parse monitor ID: {e}")
            
        c.deviceDescription = "Import from matlab"
        return(c,calib) 
    
    def attachSettings(self, pgl, settingsName):
        '''
        attach settings to the data (useful for import from matlab)
        
        Args:
            pgl: pgl instance
            settingsName: name of settings to attach to these data
        '''
        
        e = pglExperiment(pgl, settingsName)
        e.settings.closeScreenOnEnd = True
        e.settings.backgroundColor = (0, 0, 0)
        e.initScreen()
        if pgl.isOpen() is False:
            print(f"(pglDisplayLuminanceCalibrationData:attachSettings) Display {settingsName} did not open, cannot attach settings.")
            return None
        
        # set information
        self.settingsName = settingsName
        self.settings = e.getSettings(settingsName)
        
        try:
            # get display info if available
            gpu = next(iter(pgl.gpuInfo.values()))
            displays = gpu.get('Displays', [])
            self.displayInfo = displays[self.settings.displayNumber-1]
            
            # get info from pgl
            self.metalInfo = pgl.info()
            
        except Exception as ex:
            print(f"(pglDisplayLuminanceCalibrationData:attachSettings) Warning: Could not get display info: {ex}")    
    
        e.endScreen()
        
    def print(self, verbose=False):
        '''
        print the calibration data in a readable format.
        
        Args:
            verbose (bool): If True, print detailed information.
        '''   
        print("="*80)
        print (f"Calibration Data for settings: {self.settingsName}")
        if self.gammaTableSize > 0:
            print (f"Gamma Table Size: {self.gammaTableSize}")
        print (f"Creation Date and Time: {self.creationDateTime}")
        print (f"Number of Repeats: {self.nRepeats} number of Steps: {self.nSteps}")
        if verbose:
            values, measurements, minMeasurements, maxMeasurements = self.getMedianMeasurements()
            for iStep in range(self.nSteps):
                print(f"Step {iStep:3d}/{self.nSteps}: {values[iStep]:<5.3f} = median: {measurements[iStep]:<7.3f} (min: {minMeasurements[iStep]:<7.3f}, max: {maxMeasurements[iStep]:<7.3f}, percent difference: {((maxMeasurements[iStep]-minMeasurements[iStep])/measurements[iStep]*100):<5.3f}%)")
            # compute max difference between max and min in percentage
            maxDiff = np.max(np.abs(maxMeasurements - minMeasurements)/measurements * 100)
            print(f"Maximum difference between max and min measurements: {maxDiff:.3f}%")
        
    def display(self, gamma=None, ax=None):
        '''
        display graph of calibration data
        Args:
            gamma (float, optional): If provided, display the ideal gamma curve (1.0 or 2.2 or whatever) for comparison.
            ax (matplotlib.axes.Axes, optional): If provided, plot into this axis instead of creating a new figure.
        '''
        if self.calibrationValues is None or self.calibrationMeasurements is None:
            print("(pglDisplayLuminanceCalibrationData) No calibration data to display.")
            return

        # set the ideal gamma if we have one saved
        if gamma is None and self.gamma != 0:
            gamma = self.gamma

        # use the passed-in axis, or create a new figure/axis if none given
        createdOwnFigure = ax is None
        if createdOwnFigure:
            fig, ax = plt.subplots(figsize=(10, 6))

        # plot raw data points
        ax.plot(self.calibrationValues, self.calibrationMeasurements, '.')

        # plot median
        values, measurements, minMeasurements, maxMeasurements = self.getMedianMeasurements()
        if gamma is not None:
            # plot points without line
            ax.plot(values, measurements, 'o', label='Median', color='black', markeredgecolor='white', markersize=8)
            # plot ideal gamma curve
            idealMeasurements = np.power(values, gamma)
            idealMeasurements = idealMeasurements * (self.maxLuminance - self.minLuminance) + self.minLuminance
            ax.plot(values, idealMeasurements, 'r--', label=f'Ideal Gamma {gamma}')
        else:
            ax.plot(values, measurements, 'o-', label='Median', color='black', markeredgecolor='white', markersize=8)

        ax.legend()
        ax.set_xlabel("Display Value (normalized 0-1)")
        ax.set_ylabel(f"Measured Luminance ({self.units})")

        if gamma is None:
            ax.set_title(f"{self.getDisplayName()} UUID: {self.getUUID()}\n{self.deviceDescription}: {self.creationDateTime}\nnRepeats: {self.nRepeats} nSteps: {self.nSteps} min = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nCalibration data")
        else:
            ax.set_title(f"{self.getDisplayName()} UUID: {self.getUUID()}\n{self.deviceDescription}: {self.creationDateTime}\nnRepeats: {self.nRepeats} nSteps: {self.nSteps} min = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nValidation for gamma: {gamma if gamma is not None else 'N/A'}")

        ax.grid(True)

        if createdOwnFigure:
            plt.show()        
    def appendValue(self, value, mode="calibration"):
        '''
        Append a calibration value.
        
        Args:
            value (float): The display value to append.
            mode (str): "init" to append to initial values, "calibration" to append to calibration values.
        '''
        if mode == "init":
            if self.initValues is None:
                self.initValues = np.array([value])
            else:
                self.initValues = np.append(self.initValues, value)
        else:
            if self.calibrationValues is None:
                self.calibrationValues = np.array([value])
            else:
                self.calibrationValues = np.append(self.calibrationValues, value)

    def getValues(self, startIndex, endIndex=None, mode="calibration"):
        '''
        Get calibration value(s).
        
        Args:
            index (int or slice): Index or slice of values to get.
            mode (str): "init" to get from initial values, "calibration" to get from calibration values.
            
        Returns:
            float or np.ndarray: The requested calibration value(s).
        '''
        if mode == "init":
            if self.initValues is None:
                return None
            if endIndex is None:
                return self.initValues[startIndex]
            else:                
                return self.initValues[startIndex:endIndex]
        else:
            if self.calibrationValues is None:
                return None
            if endIndex is None:
                return self.calibrationValues[startIndex]
            else:
                return self.calibrationValues[startIndex:endIndex]
            
    def getMedianMeasurements(self):
        '''
        Get the median of the calibration measurements for each step.
        
        Returns:
            tuple: A tuple containing four numpy arrays: (medianValues, medianMeasurements, min_measurements, max_measurements)
        '''
        
        if self.calibrationMeasurements is None:
            return None
        
        values = np.zeros(self.nSteps)
        measurements = np.zeros(self.nSteps)
        minMeasurement = np.zeros(self.nSteps)
        maxMeasurement = np.zeros(self.nSteps)
        for iStep in range(self.nSteps):
            # get value and measurements
            value = self.calibrationValues[iStep*self.nRepeats:(iStep+1)*self.nRepeats] if self.calibrationValues is not None else None
            measurement = self.calibrationMeasurements[iStep*self.nRepeats:(iStep+1)*self.nRepeats] if self.calibrationMeasurements is not None else None
            # compute median if both value and measurement are available
            if value is not None and measurement is not None:
                values[iStep] = np.median(value)
                measurements[iStep] = np.median(measurement)
                # compute min and max
                minMeasurement[iStep] = np.min(measurement)
                maxMeasurement[iStep] = np.max(measurement)
        
        return (values, measurements, minMeasurement, maxMeasurement)

    def getMeasurements(self, startIndex, endIndex=None,mode="calibration"):
        '''
        Get calibration measurement(s).
        
        Args:
            index (int or slice): Index or slice of measurements to get.
            mode (str): "init" to get from initial measurements, "calibration" to get from calibration measurements.
            
        Returns:
            float or np.ndarray: The requested calibration measurement(s).
        '''
        if mode == "init":
            if self.initMeasurements is None:
                return None
            if endIndex is None:
                return self.initMeasurements[startIndex]
            else:                
                return self.initMeasurements[startIndex:endIndex]
        else:
            if self.calibrationMeasurements is None:
                return None
            if endIndex is None:
                return self.calibrationMeasurements[startIndex]
            else:
                return self.calibrationMeasurements[startIndex:endIndex]
 
    def appendMeasurement(self, measurement, mode="calibration"):
        '''
        Append a calibration measurement.
        
        Args:
            measurement (float): The measured luminance value to append.
            mode (str): "init" to append to initial measurements, "calibration" to append to calibration measurements.
        '''
        if mode == "init":
            if self.initMeasurements is None:
                self.initMeasurements = np.array([measurement])
            else:
                self.initMeasurements = np.append(self.initMeasurements, measurement)
        else:
            if self.calibrationMeasurements is None:
                self.calibrationMeasurements = np.array([measurement])
            else:
                self.calibrationMeasurements = np.append(self.calibrationMeasurements, measurement)
                
    def hasLastMeasurement(self, mode="calibration"):
        '''
        Check if there is a last measurement.
        
        Args:
            mode (str): "init" to check initial measurements, "calibration" to check calibration measurements.
            
        Returns:
            bool: True if there is a measurement for each value, False otherwise.
        '''
        if mode == "init":
            # If no values exist yet, return True (nothing missing)
            if self.initValues is None:
                return True
            # If values exist but no measurements, return False
            if self.initMeasurements is None:
                return False
            # Check if counts match
            return len(self.initMeasurements) == len(self.initValues)
        else:
            if self.calibrationValues is None:
                return True
            if self.calibrationMeasurements is None:
                return False
            return len(self.calibrationMeasurements) == len(self.calibrationValues)       

    def getLastValue(self, mode="calibration"):
        '''
        Get the last calibration value.
        
        Args:
            mode (str): "init" to get from initial values, "calibration" to get from calibration values.
            
        Returns:
            float: The last calibration value.
        '''
        if mode == "init":
            if self.initValues is None or len(self.initValues) == 0:
                return None
            return self.initValues[-1]
        else:
            if self.calibrationValues is None or len(self.calibrationValues) == 0:
                return None
            return self.calibrationValues[-1]
        
    def calculateInverseGamma(self, gamma = 1.0):
        '''
        Calculate inverse gamma table from calibration measurements.
        
        Args:
            gamma: The target gamma value for inverse (default is 1.0 = linear table).
        
        Returns:
            Tuple of three numpy arrays (R, G, B) for the inverse gamma table.
        '''
        # Average the repeated measurements for each step
        nSteps = self.nSteps
        nRepeats = self.nRepeats
        
        # Reshape and average
        calValues = np.array(self.calibrationValues).reshape(nSteps, nRepeats)
        calMeasurements = np.array(self.calibrationMeasurements).reshape(nSteps, nRepeats)
        
        medianValues = np.median(calValues, axis=1)
        medianMeasurements = np.median(calMeasurements, axis=1)
        
        # Normalize measurements to 0-1 range
        minLum = np.min(medianMeasurements)
        maxLum = np.max(medianMeasurements)
        normalizedMeasurements = (medianMeasurements - minLum) / (maxLum - minLum)
        
        # Create interpolation function: maps desired linear output to required input
        # We want: given a desired output level, what input do we need?
        interpFunc = interp1d(normalizedMeasurements, medianValues, 
                            kind='cubic', 
                            bounds_error=False, 
                            fill_value='extrapolate')
        
        # Create inverse gamma table
        gammaTableSize = self.gammaTableSize
        linearOutput = np.linspace(0, 1, gammaTableSize)
        gammaOutput = np.power(linearOutput, gamma)
        inverseGamma = interpFunc(gammaOutput)
        
        # Clip to valid range [0, 1] and convert to float32
        inverseGamma = np.clip(inverseGamma, 0, 1).astype(np.float32)
        
        # For RGB, use the same correction for all channels (can be modified for per-channel)
        return (inverseGamma, inverseGamma.copy(), inverseGamma.copy())
    

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
        


##########################
# Calibration device class
##########################
class pglCalibrationDevice():
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
        Destructor for the calibration device.
        '''
        pass

    def measure(self):
        '''
        Should return a measurement from the device
        '''
        pass

# for testing
class pglCalibrationDeviceDebug(pglCalibrationDevice):
    '''
    Class representing a debug calibration device
    '''
    def measure(self):
        '''
        Measure the display characteristics using debug device.
        '''
        self.currentMeasurement = np.random.rand()
        print(f"(pglCalibrationDeviceDebug) Measurement: {self.currentMeasurement}")
        return self.currentMeasurement
    
####################################################
# Minolta CS-100A calibration device class
####################################################
class pglCalibrationDeviceMinolta(pglCalibrationDevice):
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
class pglCalibration():
    '''
    Runs calibration for a display
    '''
    def __init__(self, pgl, calibrationDevice : pglCalibrationDevice):
        '''
        Initialize the calibration.
        '''
        self.device = None
        print(calibrationDevice)
        if not isinstance(calibrationDevice, pglCalibrationDevice):
            print("(pglCalibration) calibrationDevice must be of type pglCalibrationDevice.")
            return
        self.device = calibrationDevice
        self.device.verbose = False
        
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

    def calibrate(self, settingsName, nRepeats=4, nSteps=256, runValidation=True, runGammaValidation=True):
        '''
        User facing function which runs the calibration process.
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
            runValidation (bool): If True, run validation after calibration.
            runGammaValidation (bool): If True, run gamma validation after calibration. This can validate
                for making a gamma of 2.2 for videos
        '''
        
        # initialize calibration data
        self.calibrationData = None
        self.validationData = None
        self.gammaValidationData = None
        
        # open the display with specified settings
        e = pglExperiment(self.pgl, settingsName)
        e.initScreen()
        if self.pgl.isOpen() is False:
            print(f"(pglCalibration:calibrate) Display {settingsName} did not open, cannot calibrate.")
            return None
        
        # run calibraiton
        self.calibrationData = self._calibrate(e, settingsName, nRepeats, nSteps, validate=False)
        self.calibrationData.display()
        
        # run validation
        if runValidation:
            self.validationData = self.validate(e, gamma=1.0)
            self.validationData.display(gamma=1.0)
    
        if runGammaValidation:
            self.gammaValidationData = self.validate(e, gamma=2.2)
            self.gammaValidationData.display(gamma=2.2)

        # and save
        self.save()
        
        # close screen
        e.endScreen()

    def validate(self, e=None, nRepeats=None, nSteps=None, gamma=1.0):
        '''
        Validate the display calibration by applying inverse gamma and re-measuring. Usually
        run automatically by calibrate() after calibration is complete.
        
        Args:
            e: The experiment variable which contains the display settings (None will start a new one)
            nRepeats (int): Number of times to repeat each measurement (None will use the same as calibration)
            nSteps (int): Number of steps in the validation. (None will use the same as calibration)
        '''
        if not self.checkCalibrationData(self.calibrationData):
            print("(pglCalibration:validate) Calibration data is not valid. Cannot run validation.")
            return None
        print("(pglCalibration) Starting validation with inverse gamma correction...")
        
        closeScreenOnEnd = False
        if e is None:
            # open the display with specified settings
            e = pglExperiment(self.pgl, self.calibrationData.settingsName)
            e.initScreen()
            closeScreenOnEnd = True
            if self.pgl.isOpen() is False:
                print(f"(pglCalibration:validate) Display {self.calibrationData.settingsName} did not open, cannot validate.")
                return None
        
        # defaults for nRepeats and nSteps
        if nRepeats is None:
            nRepeats = self.calibrationData.nRepeats
        if nSteps is None:
            nSteps = self.calibrationData.nSteps
            
        # Calculate inverse gamma table from calibration measurements
        inverseGammaTable = self.calibrationData.calculateInverseGamma(gamma=gamma)
        
        # Run calibration with the inverse gamma table
        validationData = self._calibrate(
            e,
            self.calibrationData.settingsName,
            nRepeats=nRepeats, 
            nSteps=nSteps,
            validate=True,
            inverseGammaTable=inverseGammaTable
        )
        
        if closeScreenOnEnd:
            self.validationData = validationData
            e.endScreen()
            self.save()
            
        return validationData
 
    def _calibrate(self, e, settingsName, nRepeats=4, nSteps=256, validate=False, inverseGammaTable=None):
        '''
        Measure the display characteristics.
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
            validate (bool): If True, use inverseGammaTable instead of linear gamma.
            inverseGammaTable: Tuple of (R, G, B) gamma tables to apply for validation.
        '''
        if self.device is None:
            print("(pglCalibration) No calibration device specified.")
            return None
        
        # initialize calibration data
        self.currentCalibrationData = pglCalibrationData()
        self.calibrationIndex = 0
        self.fullCalibrationIndex = None
        self.findMinIndex = None
        
        # no progress bar at start
        self.progressBar = None
        
        # start with trying to measure min and max
        self.calibrationMode = "minmax"
        
        # get calibration date and time
        self.currentCalibrationData.creationDateTime = datetime.now()

        # save settings name
        self.currentCalibrationData.settingsName = settingsName
        self.currentCalibrationData.settings = e.getSettings(settingsName)
        
        # save the current gamma table so we can replace it
        displayNumber = self.currentCalibrationData.settings.displayNumber
        if displayNumber > 0: displayNumber -= 1
        self.currentGammaTable = self.pgl.getGammaTable(displayNumber)
        
        # Set gamma table - either linear or inverse for validation
        if validate and inverseGammaTable is not None:
            self.pgl.setGammaTable(displayNumber, inverseGammaTable[0], inverseGammaTable[1], inverseGammaTable[2])
        else:
            self.pgl.setGammaTableLinear(displayNumber)
        
        # get the gamma table and save it
        self.currentCalibrationData.gammaTableSize = self.pgl.getGammaTableSize(displayNumber)
        gammaTable = self.pgl.getGammaTable(displayNumber)
        self.currentCalibrationData.gammaTable = tuple(np.array(table) for table in gammaTable)

        # set number of repeats and steps
        self.currentCalibrationData.nRepeats = nRepeats
        self.currentCalibrationData.nSteps = nSteps

        # get the display info
        try:
            # get display info if available
            gpu = next(iter(self.pgl.gpuInfo.values()))
            displays = gpu.get('Displays', [])
            self.currentCalibrationData.displayInfo = displays[displayNumber]
            
            # get info from pgl
            self.currentCalibrationData.metalInfo = self.pgl.info()
            
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
        _, measurements, _, _ = self.currentCalibrationData.getMedianMeasurements()
        self.currentCalibrationData.minLuminance = measurements[0]
        self.currentCalibrationData.maxLuminance = measurements[-1]
        
        # return the calibration data
        calibrationData = self.currentCalibrationData
        self.currentCalibrationData = None
        return(calibrationData)
    
    def checkCalibrationData(self, calibrationData):
        '''
        Check if the calibration data is valid.
        
        Args:
            calibrationData: The calibration data to check.
        
        Returns:
            True if valid, False otherwise.
        '''
        
        if calibrationData is None:
            print("(pglCalibration) No calibration data available. Please run calibrate() first.")
            return False
        
        # Check that calibration data is complete
        if not hasattr(self.calibrationData, 'calibrationValues') or \
        not hasattr(self.calibrationData, 'calibrationMeasurements'):
            print("(pglCalibration) Calibration data is incomplete. Please run calibrate() first.")
            return False
        
        # Verify the calibration data matches expected size
        expectedSize = self.calibrationData.nRepeats * self.calibrationData.nSteps
        if len(self.calibrationData.calibrationValues) != expectedSize or \
        len(self.calibrationData.calibrationMeasurements) != expectedSize:
            print(f"(pglCalibration) Calibration data size mismatch. Expected {expectedSize}, got {len(self.calibrationData.calibrationValues)}")
            return False
        return True 

    def display(self):
        '''
        Display results of calibration and validation
        '''
        if self.checkCalibrationData(self.calibrationData):
            self.calibrationData.display()
        if self.checkCalibrationData(self.validationData):
            self.validationData.display(gamma=1.0)
        if self.checkCalibrationData(self.gammaValidationData):
            self.gammaValidationData.display(gamma=2.2)

    def displayProgress(self, startProgress=False):
        '''
        Display the progress of the calibration.
        '''
        if startProgress:
            self.progressBar = tqdm(total=self.currentCalibrationData.nSteps*self.currentCalibrationData.nRepeats, desc="Calibrating", unit="measurements")
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
        if self.currentCalibrationData.hasLastMeasurement(self.saveLocation) is False:
            return self.currentCalibrationData.getLastValue(self.saveLocation)
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
        if self.calibrationIndex < self.currentCalibrationData.nRepeats:
            self.calibrationValueAppend(1.0)
        # next show the min value (this may not be measurable)
        elif self.calibrationIndex < 2*self.currentCalibrationData.nRepeats:
            self.calibrationValueAppend(0.0)
        # Check if we have valid min and max values
        else:
            # get the measured min and max
            self.maxCalibrationVal = self.calibrationValueGet(0)
            # see how many measurement failures we had for min
            nMeasurementFailures = sum(x is None for x in self.calibrationMeasurementGet(self.currentCalibrationData.nRepeats, 2*self.currentCalibrationData.nRepeats))
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

        if ((self.calibrationIndex-self.fullCalibrationIndex) % self.currentCalibrationData.nRepeats) == 0:
            # set the next value to measure
            self.currentStep = (self.calibrationIndex-self.fullCalibrationIndex) // self.currentCalibrationData.nRepeats
            if self.currentStep == self.currentCalibrationData.nSteps:
                print(f"")
                printHeader("Full calibration complete.")
                self.calibrationMode = "done"
                return -1
            value = self.minCalibrationVal + (self.maxCalibrationVal - self.minCalibrationVal) * (self.currentStep / (self.currentCalibrationData.nSteps - 1))
            print(f"\n(pglCalibration) Measuring step {self.currentStep+1:>3d}/{self.currentCalibrationData.nSteps}: {value:.4f} = ", end="")
            self.calibrationValueAppend(value)
        else:
            # repeat last value
            self.calibrationValueAppend(self.calibrationValueGet(-1))

        return self.calibrationValueGet(-1)

    def calibrationValueAppend(self, value):
        '''
        Append a calibration value to calibrationData
        '''
        self.currentCalibrationData.appendValue(value, self.saveLocation)
    
    def calibrationValueGet(self, startIndex, endIndex=None):
        '''
        Get calibration value(s) from calibrationData
        '''
        if endIndex is None:
            return self.currentCalibrationData.getValues(startIndex, None, self.saveLocation)
        else:
            return self.currentCalibrationData.getValues(startIndex, endIndex, self.saveLocation)
    
    def calibrationMeasurementAppend(self, measurement):
        '''
        Append a calibration measurement to calibrationData
        '''
        self.currentCalibrationData.appendMeasurement(measurement, self.saveLocation)
    
    def calibrationMeasurementGet(self, startIndex, endIndex=None):
        '''
        Get calibration measurement(s) from calibrationData
        '''
        if endIndex is None:
            return self.currentCalibrationData.getMeasurements(startIndex, None, self.saveLocation)
        else:
            return self.currentCalibrationData.getMeasurements(startIndex, endIndex, self.saveLocation)
    
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
        if self.device is None:
            print("(pglCalibration) No calibration device specified.")

        # make measurement
        self.calibrationMeasurementAppend(self.device.measure())
        
        # increment index
        self.calibrationIndex += 1
        return self.calibrationMeasurementGet(-1)

    def load(self, filepath):
        '''
        Load calibration data from a file.
        '''
        self.calibrationData = pglCalibrationData()
        self.calibrationData.load(filepath)
        print(f"(pglCalibration) Calibration data loaded from {filepath}")

    def save(self, filename=None):
        '''
        Save calibration data to a file.
        
        Args:
            filename (str, optional): The filename to save the calibration data to.
            If None, a default filename based on the current date will be used.
        '''
        # check calibration data
        if not self.checkCalibrationData(self.calibrationData):
            print("(pglCalibration) Calibration data is not valid. Cannot save.")
            return None
        
        # save the calibration data
        self.calibrationData.save(filePath / "calibration")    
        if self.checkCalibrationData(self.validationData):
            self.validationData.save(filePath / "validation")
        if self.checkCalibrationData(self.gammaValidationData):
            self.gammaValidationData.save(filePath / "gammaValidation")
        print(f"(pglCalibration) Calibration data saved to {filePath / dateStem}")
        
    def linearizeDisplay(self, value):
        '''
        Apply the calibration to a display to achieve a linear luminance
        '''
        if self.calibrationData is None:
            print("(pglCalibration) No calibration data available. Please run calibrate() first.")
            return None
        
        # get the inverse calibration table
    
# Calibration settings, subclass of pglSettings to inherit load/save functionality
class pglCalibrationData(HasTraits, pglSerialize):
    
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
    
    def save(self, filename=None, filepath=None):
        '''
        save
        
        Args:
            filename: If none defaults to calibration.json
            filepath: If none defaults to calibrationDir / display name / data
        '''
        # get the displayName
        displayName = self.displayInfo.get("DisplayName",self.settingsName)

        # get a filepath
        filepath = self.getFilepath(filepath, displayName, makePath=True)
        
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
        filepath = cls.getFilepath(filepath, displayName, makePath=False)
        if filepath is None:
            return
        filepath = cls.chooseCalibrationDir(filepath, date)
        
        # get the filename
        if filename is None:
            filename = "calibration"
           
        # call super load to instantiate the object and load it                
        return super(pglCalibrationData, cls).load(filepath / filename)
    
    @staticmethod
    def chooseCalibrationDir(filepath, date=None):
        '''
        List the available calibration directories for a display and let the
        user choose one. Returns the chosen directory Path, or None.

        Args:
            displayName: the display name whose calibrations to list
            filepath: base calibration path; if None uses the default
            date: optional filter (str "YYYYMMDD", datetime, or date)
        '''
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
            print(f"(pglCalibrationData:chooseCalibrationDir) ❌ No calibration directories found in: {filepath}")
            return None

        # print header
        print(f"(pglCalibration:chooseCalibrationDir) Calibration directory: {filepath}")

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
            print(f"(pglCalibration:chooseCalibrationDir) ❌ Invalid choice: {choice}")
            return None

        # return the chosen directory
        chosenDir = dirList[choice - 1]
        print(f"(pglCalibration:chooseCalibrationDir) Selected: {chosenDir}")
        return chosenDir
    
    @staticmethod
    def getFilepath(filepath, displayName, makePath=True):
        '''
        Gets default filepath
        '''
        if filepath is None:
            from pgl import pgl
            # make the path be a directory
            settingsPath = pglSettingsManager().getCalibrationsDir()
            settingsPath = settingsPath / pgl.makeValidFilename(displayName)
        
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
                print(f"(pglCalibrationData:getFilepath) ❌ Could not find calibration directory: {filepath}")
                return None
        
        return(filepath)
        
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
        
    def display(self, gamma=None):
        '''
        display graph of calibration data
        
        Args:
            gamma (float, optional): If provided, display the ideal gamma curve (1.0 or 2.2 or whatever) for comparison.
        '''
        if self.calibrationValues is None or self.calibrationMeasurements is None:
            print("(pglCalibrationData) No calibration data to display.")
            return
        plt.figure(figsize=(10, 6))
        # plot raw data points
        plt.plot(self.calibrationValues, self.calibrationMeasurements, '.')
        # plot median
        values, measurements, minMeasurements, maxMeasurements = self.getMedianMeasurements()
        if gamma is not None:
            # plot points without line
            plt.plot(values, measurements, 'o', label='Median', color='black', markeredgecolor='white', markersize=8)
            # plot ideal gamma curve
            idealMeasurements = np.power(values, gamma)
            idealMeasurements = idealMeasurements * (self.maxLuminance - self.minLuminance) + self.minLuminance
            plt.plot(values, idealMeasurements, 'r--', label=f'Ideal Gamma {gamma}')        
        else:
            plt.plot(values, measurements, 'o-', label='Median', color='black', markeredgecolor='white', markersize=8)
        plt.legend()
        plt.xlabel("Display Value")
        plt.ylabel("Measured Luminance")
        if gamma is None:
            plt.title(f"Settings: {self.settingsName} nRepeats: {self.nRepeats} nSteps: {self.nSteps}\nmin = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nCalibration data")
        else:
            plt.title(f"Settings: {self.settingsName} nRepeats: {self.nRepeats} nSteps: {self.nSteps}\nmin = {self.minLuminance:.2f}, max = {self.maxLuminance:.2f}\nValidation for gamma: {gamma if gamma is not None else 'N/A'}")
            
        plt.grid(True)
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
    

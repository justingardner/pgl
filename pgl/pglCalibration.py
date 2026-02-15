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
from .pglSettings import filename, _pglSettings, pglSettings, pglSettingsManager
from traitlets import Unicode, Int, Instance, Dict, Tuple
from datetime import datetime
from .pglExperiment import pglExperiment

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
    Runs calibrtion for a display
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
        
        # initialize calibration data
        self.calibrationData = pglCalibrationData()
        self.calibrationIndex = 0
        self.fullCalibrationIndex = None
        self.findMinIndex = None
        
        # start with trying to measure min and max
        self.calibrationMode = "minmax"
        
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
    
    def calibrate(self, settingsName, nRepeats=4, nSteps=256):
        '''
        Measure the display characteristics.
        
        Args:
            settingsName (str): Name of the screen settings to calibrate.
            nRepeats (int): Number of times to repeat each measurement.
            nSteps (int): Number of steps in the calibration.
        '''
        if self.device is None:
            print("(pglCalibration) No calibration device specified.")
            return None
        
        # open the display with specified settings
        e = pglExperiment(self.pgl, settingsName)
        if self.pgl.isOpen() is False:
            print(f"(pglCalibration) Display {settingsName} did not open, cannot calibrate.")
            return None
        
        # get calibration date and time
        self.calibrationData.creationDateTime = datetime.now()

        # save settings name
        self.calibrationData.settingsName = settingsName
        self.calibrationData.screenSettings = e.getScreenSettings(settingsName)
        
        # linearlize gamma, save the current gamma table so we can replace it
        screenNumber = self.calibrationData.screenSettings.screenNumber
        self.currentGammaTable = self.pgl.getGammaTable(screenNumber)
        self.pgl.setGammaTableLinear(screenNumber)
        
        # get the linearized gamma table and save it (for reference as validation will have an inverse table)
        self.calibrationData.gammaTableSize = self.pgl.getGammaTableSize(screenNumber)
        gammaTable = self.pgl.getGammaTable(screenNumber)
        self.calibrationData.gammaTable = tuple(np.array(table) for table in gammaTable)

        # set number of repeats and steps
        self.calibrationData.nRepeats = nRepeats
        self.calibrationData.nSteps = nSteps

        # get the display info
        try:
            # get display info if available
            gpu = next(iter(self.pgl.gpuInfo.values()))
            displays = gpu.get('Displays', [])
            self.calibrationData.displayInfo = displays[screenNumber]
        except Exception as ex:
            print(f"(pglCalibration) Warning: Could not get display info: {ex}")
        
        # set the current display
        self.setDisplay()

        # now, tell operator to make sure everything is setup before continuing
        print("(pglCalibration) Please ensure the calibration device is properly positioned and ready.")
        print("(pglCalibration) Press Enter to continue...")
        input("")
        printHeader("Establishing min and max values")
        
        # loop to set display and make measurements
        while (self.setDisplay() != -1):
            # make a measurement
            self.measure()
            # and display
            print(f"(pglCalibration) {self.calibrationValueGet(-1)}: {self.calibrationMeasurementGet(-1)}")

        # restore the original gamma table
        self.pgl.setGammaTable(screenNumber, self.currentGammaTable[0], self.currentGammaTable[1], self.currentGammaTable[2])
        
        # close the screen
        self.pgl.close()
        
        # display results when done
        self.calibrationData.display()
        
        # and save
        self.save()
        
    def getNextCalibrationValue(self):
        '''
        Get the calibration values. This chooses which value to test
        based on what values have been measured so far. Sometimes
        calibration devices have a problem with low luminance measurements
        so will need to skip values if measurents fail
        '''
        # if we already have a calibration value that hasn't been measured
        # then return that
        if self.calibrationData.hasLastMeasurement(self.saveLocation) is False:
            return self.calibrationData.getLastValue(self.saveLocation)
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
        if self.calibrationIndex < self.calibrationData.nRepeats:
            self.calibrationValueAppend(1.0)
        # next show the min value (this may not be measurable)
        elif self.calibrationIndex < 2*self.calibrationData.nRepeats:
            self.calibrationValueAppend(0.0)
        # Check if we have valid min and max values
        else:
            # get the measured min and max
            self.maxCalibrationVal = self.calibrationValueGet(0)
            # see how many measurement failures we had for min
            nMeasurementFailures = sum(x is None for x in self.calibrationMeasurementGet(self.calibrationData.nRepeats, 2*self.calibrationData.nRepeats))
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

        if ((self.calibrationIndex-self.fullCalibrationIndex) % self.calibrationData.nRepeats) == 0:
            # set the next value to measure
            step = (self.calibrationIndex-self.fullCalibrationIndex) // self.calibrationData.nRepeats
            if step == self.calibrationData.nSteps:
                printHeader("Full calibration complete.")
                self.calibrationMode = "done"
                return -1
            value = self.minCalibrationVal + (self.maxCalibrationVal - self.minCalibrationVal) * (step / (self.calibrationData.nSteps - 1))
            print(f"(pglCalibration) Measuring step {step+1} of {self.calibrationData.nSteps}: {value}")
            self.calibrationValueAppend(value)
        else:
            # repeat last value
            self.calibrationValueAppend(self.calibrationValueGet(-1))

        return self.calibrationValueGet(-1)

    def calibrationValueAppend(self, value):
        '''
        Append a calibration value to calibrationData
        '''
        self.calibrationData.appendValue(value, self.saveLocation)
    def calibrationValueGet(self, startIndex, endIndex=None):
        '''
        Get calibration value(s) from calibrationData
        '''
        if endIndex is None:
            return self.calibrationData.getValues(startIndex, None, self.saveLocation)
        else:
            return self.calibrationData.getValues(startIndex, endIndex, self.saveLocation)
    def calibrationMeasurementAppend(self, measurement):
        '''
        Append a calibration measurement to calibrationData
        '''
        self.calibrationData.appendMeasurement(measurement, self.saveLocation)
    def calibrationMeasurementGet(self, startIndex, endIndex=None):
        '''
        Get calibration measurement(s) from calibrationData
        '''
        if endIndex is None:
            return self.calibrationData.getMeasurements(startIndex, None, self.saveLocation)
        else:
            return self.calibrationData.getMeasurements(startIndex, endIndex, self.saveLocation)
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
        pass

    def save(self, filename=None):
        '''
        Save calibration data to a file.
        
        Args:
            filename (str, optional): The filename to save the calibration data to.
            If None, a default filename based on the current date will be used.
        '''
        if filename is None:
            # make the path be a directory
            settingsPath = pglSettingsManager().getCalibrationsDir()
            fileStem = datetime.now().strftime("%Y%m%d")
            filePath = settingsPath / fileStem
        
            # Check if directory exists, if so add time
            if filePath.exists():
                fileStem = datetime.now().strftime("%Y%m%d_%H%M%S")
                filePath = settingsPath / fileStem
        else:
            filePath = filename
            fileStem = filename
            
        # Create the directory
        filePath.mkdir(parents=True, exist_ok=True)
        
        # save the calibration data
        self.calibrationData.save(filePath / fileStem)    

    def apply(self, value):
        '''
        Apply the calibration to a given value.
        '''
        pass
    

# Calibration settings, subclass of pglSettings to inherit load/save functionality
class pglCalibrationData(_pglSettings):
    
    settingsName = Unicode("Default", help="Settings name used to open display")
    displayInfo = Dict(help="Display information at time of calibration")
    screenSettings = Instance(pglSettings, allow_none=True, help="Settings used during calibration")    
    gammaTableSize = Int(-1, help="Size of the gamma table at time of calibration")
    gammaTable = Tuple(Instance(np.ndarray), Instance(np.ndarray), Instance(np.ndarray), allow_none=True, help="Gamma table at time of calibration")
    creationDateTime = Instance(datetime, default_value=datetime.now(), help="Date and time of calibration creation")
    nRepeats = Int(4, help="Number of repeats per calibration value")
    nSteps = Int(256, help="Number of steps in the calibration")
    initValues = Instance(np.ndarray, allow_none=True, help="Initial display values used in calibration, if any, used for finding min and max values")
    initMeasurements = Instance(np.ndarray, allow_none=True, help="Measured luminance values corresponding to initValues calibration")
    calibrationValues = Instance(np.ndarray, allow_none=True, help="Display values used in calibration")
    calibrationMeasurements = Instance(np.ndarray, allow_none=True, help="Measured luminance values from calibration")

    def display(self):
        # the calibration values should be at the end of the measurement, so just grab those
        plt.figure(figsize=(8,6))
        plt.plot(self.calibrationValues, self.calibrationMeasurements, 'o-')
        plt.xlabel("Display Value")
        plt.ylabel("Measured Luminance")
        plt.title("Display Calibration")
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
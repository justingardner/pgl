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
    Class representing a display calibration.
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

        # make a dict for calibrationModes
        self.calibrationModes = {
            "minsearch": -2,
            "minmax": -1,
            "full": 0,
            "done": 10,
        }
        
        # initialize calibration data
        self.calibrationValue = []
        self.calibrationMeasurement = []        
        self.calibrationIndex = 0
        self.fullCalibrationIndex = None
        self.findMinIndex = None
        
        # start with trying to measure min and max
        self.calibrationMode = self.calibrationModes.get("minmax")
        
        # value for how close to min and max a value can be
        # to just be considered min or max when doing the
        # search for min and max values
        self.epsilon = 0.001

    def __del__(self):
        '''
        Destructor for the calibration class.
        '''
        pass
    
    def calibrate(self, nRepeats=4, nSteps=256):
        '''
        Measure the display characteristics.
        '''
        if self.device is None:
            print("(pglCalibration) No calibration device specified.")
            return None
        
        if self.pgl.isOpen() is False:
            print("(pglCalibration) PGL display is not open, cannot calibrate.")
            return None

        # set number of repeats and steps
        self.nRepeats = nRepeats
        self.nSteps = nSteps

        # set the current dispaly
        self.setDisplay()

        # now, tell subject to make sure everything is setup before continuing
        print("(pglCalibration) Please ensure the calibration device is properly positioned and ready.")
        print("(pglCalibration) Press Enter to continue...")
        input("")
        printHeader("Establishing min and max values")
        
        # loop to set display and make measurements
        while (self.setDisplay() != -1):
            # make a measurement
            self.measure()
            # and display
            print(f"(pglCalibration) {self.calibrationValue[-1]}: {self.calibrationMeasurement[-1]}")

        self.displayCalibration()
        
    def displayCalibration(self):
        '''
        Display the calibration results.
        '''
        # the calibration values should be at the end of the measurement, so just grab those
        nCalibrationValues = self.nSteps*self.nRepeats
        values = self.calibrationValue[-nCalibrationValues:]
        measurements = self.calibrationMeasurement[-nCalibrationValues:]
        plt.figure(figsize=(8,6))
        plt.plot(values, measurements, 'o-')
        plt.xlabel("Display Value")
        plt.ylabel("Measured Luminance")
        plt.title("Display Calibration")
        plt.grid(True)
        plt.show()
        
    def getCalibrationValue(self):
        '''
        Get the calibration values. This chooses which value to test
        based on what values have been measured so far. Sometimes
        calibration devices have a problem with low luminance measurements
        so will need to skip values if measurents fail
        '''
        # if we alread have a calibration value that hasn't been measured
        # then return that
        if len(self.calibrationMeasurement) < len(self.calibrationValue):
            return self.calibrationValue[-1]
        # check for minmax mode to get min and max values
        if self.calibrationMode == self.calibrationModes.get("minmax"):
            return self.getMinMaxCalibrationValue()
        elif self.calibrationMode == self.calibrationModes.get("minsearch"):
            # this mode is to try to find a measurable min value
            return self.getFindMinCalibrationValue()
        elif self.calibrationMode == self.calibrationModes.get("full"):
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
        if self.calibrationIndex < self.nRepeats:
            self.calibrationValue.append(1.0)
        # next show the min value (this may not be measurable)
        elif self.calibrationIndex < 2*self.nRepeats:
            self.calibrationValue.append(0.0)
        # Check if we have valid min and max values
        else:
            # get the measured min and max
            self.maxCalibrationVal = self.calibrationValue[0]
            # see how many measurement failures we had for min
            nMeasurementFailures = sum(x is None for x in self.calibrationMeasurement[self.nRepeats:2*self.nRepeats])
            if nMeasurementFailures == 0:
                # get the min value
                self.minCalibrationVal = self.calibrationValue[-1]
                # and set our mode to full calibration
                self.calibrationMode = self.calibrationModes.get("full")
                return self.getFullCalibrationValue()
            else:
                # this calibration mode is to try to find the min value
                self.calibrationMode = self.calibrationModes.get("minsearch")
                print("(pglCalibration) Minimum value was not measurable, trying to find a measureable minimum")
                return self.getFindMinCalibrationValue()
                
        # return the value that was set
        return self.calibrationValue[-1]

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
            if self.calibrationMeasurement[-1] is not None:
                if self.calibrationValue[-1] < self.findMinMeasurableVal:
                    self.findMinMeasurableVal = self.calibrationValue[-1]
            else:
                # Unmeasurable, so replace unmeasurable value
                if self.calibrationValue[-1] > self.findMinUnmeasurableVal:
                    self.findMinUnmeasurableVal = self.calibrationValue[-1]
            # if we are within epsilon of the unmeasurable value, then accept
            if (self.findMinMeasurableVal-self.findMinUnmeasurableVal) < self.epsilon:
                print(f"(pglCalibration) Minimum measurable value found: {self.findMinMeasurableVal}")
                self.minCalibrationVal = self.findMinMeasurableVal
                self.calibrationMode = self.calibrationModes.get("full")
                return self.getFullCalibrationValue()
        # Check the halfway point between the current unmesurable and measurable
        newVal = self.findMinUnmeasurableVal + (self.findMinMeasurableVal - self.findMinUnmeasurableVal) / 2.0
        self.calibrationValue.append(newVal)
        
        return self.calibrationValue[-1]
    def getFullCalibrationValue(self):
        '''
        Get the next calibration value for a full calibration
        '''

        # starting fullcalibration
        if self.fullCalibrationIndex is None:
            self.fullCalibrationIndex = self.calibrationIndex
            # display the min and max value
            printHeader(f"Starting Full Calibration between {self.minCalibrationVal} and {self.maxCalibrationVal}")

        if ((self.calibrationIndex-self.fullCalibrationIndex) % self.nRepeats) == 0:
            # set the next value to measure
            step = (self.calibrationIndex-self.fullCalibrationIndex) // self.nRepeats
            if step == self.nSteps:
                printHeader("Full calibration complete.")
                self.calibrationMode = self.calibrationModes.get("done")
                return -1
            value = self.minCalibrationVal + (self.maxCalibrationVal - self.minCalibrationVal) * (step / (self.nSteps - 1))
            print(f"(pglCalibration) Measuring step {step+1} of {self.nSteps}: {value}")
            self.calibrationValue.append(value)
        else:
            # repeat last value
            self.calibrationValue.append(self.calibrationValue[-1])

        return self.calibrationValue[-1]

    def setDisplay(self):
        '''
        Set the display to the calibration value
        '''
        # get the next calibration value
        value = self.getCalibrationValue()
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
        self.calibrationMeasurement.append(self.device.measure())
        
        # increment index
        self.calibrationIndex += 1
        return self.calibrationMeasurement[-1]

    def load(self, filepath):
        '''
        Load calibration data from a file.
        '''
        pass

    def save(self, filepath):
        '''
        Save calibration data to a file.
        '''
        pass

    def apply(self, value):
        '''
        Apply the calibration to a given value.
        '''
        pass
    

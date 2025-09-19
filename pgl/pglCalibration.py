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
from ._pglComm import pglSerial

#############
# Parameter class
#############
class pglCalibration():
    '''
    Class representing a display calibration.
    '''
    def __init__(self, description=""):
        '''
        Initialize the calibration.
        '''
        self.description = description

        
    def __del__(self):
        '''
        Destructor for the calibration class.
        '''
        pass
    
    def measure(self):
        '''
        Measure the display characteristics.
        '''
        pass

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
    
class pglCalibrationDevice():
    '''
    Class representing a calibration device, this 
    should be subclassed for specific devices
    '''
    def __init__(self, description=""):
        '''
        Initialize the calibration device
        '''
        self.description = description

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


class pglCalibrationDeviceMinolta(pglCalibrationDevice):
    '''
    Class representing a Minolta CS-100A luminance / chrominance meter
    '''
    def __init__(self, description="Minolta CS-100A"):
        '''
        Initialize the Minolta calibration.
        '''
        super().__init__(description)
        
        # init serial port
        self.serial = pglSerial(dataLen=7, parity='e', stopBits=2, baudrate=4800, timeout=5.0)

    def __del__(self):
        '''
        Destructor for the Minolta calibration device.
        '''
        self.serial.close()

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
            
            print(f"(pglCalibrationDeviceMinolta) Luminance={luminance}, x={x}, y={y} (mode={mode})")
            return luminance
        
        except Exception as e:
            print(f"(pglCalibrationDeviceMinolta) ❌ Error parsing response ({r}): {e}")
            return None
    

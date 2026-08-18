################################################################
#   filename: pglLabJack.py
#    purpose: Device class for interfacing with LabJack T7
#             for analog and digital IO
#         by: JLG
#       date: Jan 27, 2026
################################################################

###########
# Import
##########
import io
import threading
import time
import numpy as np
from .pglTimestamp import pglTimestamp
from .pglDevice import pglDigitalIODevice, pglAnalogInputDevice, pglAnalogTraceData
import matplotlib.pyplot as plt
from .pglMessages import pglMessages
import time

class pglLabJack(pglDigitalIODevice, pglAnalogInputDevice):
    def __init__(self):
        self.digitalOutputConfigured = False
        self.analogInputConfigured = False
        super().__init__(deviceType="LabJack")
        
        # import library, checking for errors
        try:
            from labjack import ljm
            # keep ljm reference
            self.ljm = ljm
        except ImportError: 
            pglMessages.warning("Labjack library is not installed. Please install LJM Library to use LabJack.\n Installation is available from: https://support.labjack.com/docs/ljm-software-installer-macos-x64\nAfter downloading, install into pgl pip environment: python -m pip install labjack-ljm")
            return
        
        try:
            # open LabJack device
            self.h = ljm.openS("ANY", "USB", "ANY")
        except Exception as e:
            if e.errorCode == 1227:
                pglMessages.warning(f"(pglLabJack) No LabJack device found: {e}")
            else:
                pglMessages.warning(f"(pglLabJack) Error opening LabJack device: {e}")
            self.h = None
            return
        
        if self.h is not None:
            # get handle info
            (deviceType, connectionType, self.serialNumber, self.ipAddress, self.port, self.maxBytesPerMB)= ljm.getHandleInfo(self.h)
            
            # get device type as a string
            deviceTypeStrings = {
                ljm.constants.dtT4: "T4",
                ljm.constants.dtT7: "T7",
                ljm.constants.dtT8: "T8"
            }
            self.type = deviceTypeStrings.get(deviceType, "Unknown")
            
            # get connection types as a string
            connectionTypeStrings = {
                ljm.constants.ctUSB: "USB",
                ljm.constants.ctETHERNET: "Ethernet",
                ljm.constants.ctWIFI: "WiFi",
                ljm.constants.ctANY: "Any"
            }
            self.connectionType = connectionTypeStrings.get(connectionType, "Unknown")
            print(f"(pglLabJack) Opened {self.type} LabJack device via {self.connectionType} connection.")
            print(f"             serialNumber: {self.serialNumber} ipAddress: {self.ipAddress} port: {self.port} maxBytesPerMB: {self.maxBytesPerMB}")
    
            # set description
            self.deviceDescription = f"{self.type} LabJack via {self.connectionType}"
               
    @property
    def isActive(self):
        return True if self.h is not None else False
            
    def __repr__(self):
        if self.h is None:
            return "<pglLabJack device not connected>"
        else:
            return f"<pglLabJack deviceType={self.type} connectionType={self.connectionType} serialNumber={self.serialNumber}>"
    
    def setupDigitalOutput(self, channel=0, pulseLen=1, channelGroup="FIO", **kwargs):
        '''
        Setup a digital output channel.
    
        Args:
            channel (int): Digital channel number (e.g., 0 for FIO0)
            pulseLen (int): Time in ms for digital pulse to last (for digitalOutputPulse)
            channelGroup (str): String for the channel group, can be "FIO", "EIO", "CIO", "MIO" or "DIO" Defaults to FIO
        '''
        if self.h is None:
            pglMessages.warning("(pglLabJack:setupDigitalOutput) LabJack device not connected.", level=1)
            self.digitalOutputConfigured = False
            return
        
        # call super to store pulseLen for this chanel
        super().setupDigitalOutput(channel, pulseLen)
    
        validChannelGroups = set(["FIO", "EIO", "CIO", "MIO", "DIO"])
        if channelGroup not in validChannelGroups:
            pglMessages.warning(f"Input channel group: {channelGroup} is not in valid group: {validChannelGroups}")
            return
        
        # Convert to FIO name if needed
        if isinstance(channel, int):
            self.digitalChannels[channel]["name"] = f"{channelGroup}{channel}"
        else:
            pglMessages.warning(f"channel must be an integer", level=2)
            return
            
        try:    
            # Set as digital output (direction = 1 for output)
            self.ljm.eWriteName(self.h, f"{self.digitalChannels[channel]["name"]}", 1)
            # Set initial state to LOW
            self.ljm.eWriteName(self.h, self.digitalChannels[channel]["name"], 0)
            print(f"(pglLabJack:setupDigitalOutput) {self.digitalChannels[channel]["name"]} configured as output, set to LOW")
        except Exception as e:
            print(f"(pglLabJack:setupDigitalOutput) Error setting up {self.digitalChannels[channel]["name"]}: {e}")
            self.digitalOutputConfigured = False
        
        # configured
        self.digitalOutputConfigured = True

    def digitalOutput(self, channel, state):
        '''
        Set the digital output state. Call setupDigitalOutput() first to configure the channel.

        Args:
            state (bool): True for HIGH, False for LOW
        Returns:
            timestamp (float): Timestamp of when the digital output was set,
                               or None if there was an error.
        '''

        if not self.digitalOutputConfigured:
            pglMessages.warning(f"Digital output channel not configured. Call setupDigitalOutput() first.")
            return None

        # set state
        try:
            self.ljm.eWriteName(self.h, self.digitalChannels[channel]["name"], 1 if state else 0)
        except Exception as e:
            pglMessages.warning(f"Error writing {self.digitalChannels[channel]["name"]}: {e}")
            return None
        
        return pglTimestamp.getSecs()
          
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000, voltageRange=10.0):
        '''
        Start analog input reading from specified channels.
        
        Args:
            duration (float): Duration of recording in seconds
            channels (list): List of channel numbers or names
            scanRate (int): Sampling rate in Hz
            scansPerRead (int): Number of scans per read operation
            voltageRange (float): Voltage range for analog inputs. Options: 10.0V, 1.0V, 0.1V, 0.01V

        '''
        if self.h is None:
            print("(pglLabJack:startAnalogRead) LabJack device not connected.")
            return

        # Convert channel numbers to AIN names if needed
        channelAddresses = []
        for ch in channels:
            if isinstance(ch, int):
                channelAddresses.append(f"AIN{ch}")
            else:
                channelAddresses.append(ch)  # Already a string like "AIN0"
        
        # validate range 
        validRanges = [10.0, 1.0, 0.1, 0.01]
        if voltageRange not in validRanges:
            print(f"(pglLabJack:startAnalogRead) Invalid range {voltageRange}V. Valid options: {validRanges}")
            return
        try:
            # set each channel to the specified range
            for channel in channelAddresses:
                self.ljm.eWriteName(self.h, f"{channel}_RANGE", voltageRange)
        except Exception as e:
            print(f"(pglLabJack:startAnalogRead) Error setting range: {e}")
            return

        # save parameters
        self.channels = channelAddresses
        self.scanRate = scanRate
        self.scansPerRead = scansPerRead
        self.range = voltageRange
        self.analogStreamDuration = duration

        # derived parameters
        self.numChannels = len(channels)
        self.totalScans = int(duration * scanRate)
        self.totalReads = int(np.ceil(self.totalScans / scansPerRead))

        if self.totalScans % scansPerRead != 0:
            print(f"(pglLabJack:startAnalogRead) totalScans ({self.totalScans}) is not an integer multiple of scansPerRead ({scansPerRead}). Will collect {self.totalReads * scansPerRead} samples instead of {self.totalScans} and throw out extra samples.")
            
        # buffer and synchronization
        self.analogBuffer = []
        self.bufferLock = threading.Lock()
        self.stopEvent = threading.Event()

        # state flag
        self.isReading = True

        # start acquisition thread
        self.acquisitionThread = threading.Thread(
            target=self._analogReadThread,
            daemon=True
        )
        self.acquisitionThread.start()
           
    def _analogReadThread(self):
        """
        Thread function to read analog data from LabJack
        """
        
        # record the start time of the stream
        self.analogStartTimestamp = pglTimestamp.getSecs()
        
        # Convert channel names to addresses
        try:
            channelAddresses = self.ljm.namesToAddresses(self.numChannels, self.channels)[0]
        except Exception as e:
            print(f"(pglLabJack:analogReadThread) Error converting channel names: {e}")
            self.isReading = False
            return

        # start stream
        try:
            self.scanRate = self.ljm.eStreamStart(
                self.h,
                self.scansPerRead,
                self.numChannels,
                channelAddresses,
                self.scanRate
            )
        except Exception as e:
            print(f"(pglLabJack:analogReadThread) Error starting stream: {e}")
            self.isReading = False
            return

        # keep getting data until duration is reached or stop event is set
        try:
            while not self.stopEvent.is_set():
                if (pglTimestamp.getSecs() - self.analogStartTimestamp) >= self.analogStreamDuration:
                    break

                # read the data from labJack stream
                dataArray, deviceBacklog, ljmBacklog = self.ljm.eStreamRead(self.h)

                # copy over the data that was received
                with self.bufferLock:
                    self.analogBuffer.extend(dataArray)

        finally:
            try:
                # stop the stream
                self.ljm.eStreamStop(self.h)
            except Exception:
                pass

            self.isReading = False

    def stopAnalogRead(self, waitToFinish=False, doNotTruncate=False):
        """
        Stop the analog reading and return time and data arrays.
        
        Args:
            waitToFinish (bool): If True, waits for the acquisition thread to finish before returning data.
                                 If False, signals the thread to stop and returns immediately with whatever data has been collected so far.
            doNotTruncate (bool): If True, do not truncate the data to the exact number of samples.
                                 If False (default), truncates the data to the expected number of samples based on duration and scan rate.  
        Returns:
            data: pglAnalogTraceData which holds time and data
        """
        if self.h is None:
            pglMessages.warning("Device not initialized")
            return None

        # If acquisition is active, request stop when not waiting
        if not waitToFinish and self.isReading:
            self.stopEvent.set()

        # If acquisition thread exists, wait for it to finish
        if self.acquisitionThread is not None and self.acquisitionThread.is_alive():
            pglMessages.message("waiting for analog acquisition to end")
            self.acquisitionThread.join()

        # copy data safely
        with self.bufferLock:
            data = np.array(self.analogBuffer)

        if data.size == 0:
            pglMessages.warning("No data read")
            return None

        # Reshape data to separate channels
        # data shape will be (numSamples, numChannels)
        numSamples = len(data) // self.numChannels
        data = data[:numSamples * self.numChannels] 
        data = data.reshape(numSamples, self.numChannels)

        # truncate to exact number of samples
        if not doNotTruncate and numSamples > self.totalScans:
            data = data[:self.totalScans, :]
            numSamples = self.totalScans
            
        # create time array (one timestamp per sample, not per data point)
        time = np.linspace(
            0,
            (numSamples / self.scanRate),
            numSamples
        )

        return pglAnalogTraceData(time=time, data=data, channelNames=self.channels)
              
    def __del__(self):
        """
        Clean up the labJack instance
        """
        # Perform any necessary cleanup here
        if self.h is not None:
            self.ljm.close(self.h)
            self.h = None
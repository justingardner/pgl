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
from pgl import pglTimestamp
from .pglDevice import pglDigitalIODevice, pglAnalogTraceData
import matplotlib.pyplot as plt

class pglLabJack(pglDigitalIODevice):
    def __init__(self):
        self.digitalOutputConfigured = False
        super().__init__(deviceType="LabJack")
        
        # import library, checking for errors
        try:
            from labjack import ljm
        except ImportError: 
            print("(pglLabJack) labjack library is not installed. Please install it to use LabJack.")
            return
        
        try:
            # open LabJack device
            self.h = ljm.openS("ANY", "USB", "ANY")
        except Exception as e:
            if e.errorCode == 1227:
                print(f"(pglLabJack) No LabJack device found: {e}")
            else:
                print(f"(pglLabJack) Error opening LabJack device: {e}")
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
            
            # ljm library reference
            self.ljm = ljm
            
            # timestamp utility
            self.pglTimestamp = pglTimestamp()
                        
    def __repr__(self):
        if self.h is None:
            return "<pglLabJack device not connected>"
        else:
            return f"<pglLabJack deviceType={self.type} connectionType={self.connectionType} serialNumber={self.serialNumber}>"
    
    def setupDigitalOutput(self, channel=0):
        '''
        Setup a digital output channel.
    
        Args:
            channel (int or str): Digital channel number (e.g., 0 for FIO0) or name (e.g., 'FIO0')
        '''
        if self.h is None:
            print("(pglLabJack:setupDigitalOutput) LabJack device not connected.")
            self.digitalOutputConfigured = False
            return
    
        # Convert to FIO name if needed
        if isinstance(channel, int):
            self.channel = f"FIO{channel}"
        else:
            self.channel = channel
    
        try:    
            # Set as digital output (direction = 1 for output)
            self.ljm.eWriteName(self.h, f"{self.channel}", 1)
            # Set initial state to LOW
            self.ljm.eWriteName(self.h, self.channel, 0)
            print(f"(pglLabJack:setupDigitalOutput) {self.channel} configured as output, set to LOW")
        except Exception as e:
            print(f"(pglLabJack:setupDigitalOutput) Error setting up {self.channel}: {e}")
            self.digitalOutputConfigured = False
        
        # configured
        self.digitalOutputConfigured = True


    def digitalOutput(self, state, pulseLen=None):
        '''
        Set the digital output state. Call setupDigitalOutput() first to configure the channel.

        Args:
            state (bool): True for HIGH, False for LOW
            pulseLen (float or None): Pulse length in milliseconds. If set, the output
                                      will return back to the opposite state after this time

        Returns:
            timestamp (float): Timestamp of when the digital output was set,
                               or None if there was an error.
        '''

        if not self.digitalOutputConfigured:
            print("(pglLabJack:setDigitalOutput) Digital output channel not configured. Call setupDigitalOutput() first.")
            return None

        # set state
        try:
            self.ljm.eWriteName(self.h, self.channel, 1 if state else 0)
        except Exception as e:
            print(f"(pglLabJack:setDigitalOutput) Error reading {self.channel}: {e}")
            return None

        if pulseLen is not None:

            def restoreState():
                try:
                    # wait for pluseLen (in ms)
                    time.sleep(pulseLen / 1000.0)
                    # reset the state
                    self.ljm.eWriteName(self.h, self.channel, 0 if state else 1)
                except Exception as e:
                    print(f"(pglLabJack:setDigitalOutput) Error restoring {self.channel}: {e}")

            # start thread to reset state
            thread = threading.Thread(target=restoreState, daemon=True)
            thread.start()

        return self.pglTimestamp.getSecs()

    def digitalOutputAtTime(self, targetTime, state, pulseLen=None):
        '''
        Set the digital output state at a specified future time. Call setupDigitalOutput() first to configure the channel.

        WARNING: If calling mulitple times, ensure pulses don't overlap in time as this code
            does not currently handle multiple overlapping pulses and may produce unexpected results if pulses overlap.

        Args:
            targetTime (float): Timestamp (in seconds) when the pulse should be delivered.
                                Must be in the future relative to pglTimestamp.getSecs().
            state (bool): True for HIGH, False for LOW
            pulseLen (float or None): Pulse length in milliseconds. If set, the output
                                      will return back to the opposite state after this time

        Returns:
            bool: True if the pulse was successfully scheduled, False otherwise
        '''

        if not self.digitalOutputConfigured:
            print("(pglLabJack:digitalOutputAtTime) Digital output channel not configured. Call setupDigitalOutput() first.")
            return False

        # Validate that targetTime is in the future
        currentTime = self.pglTimestamp.getSecs()
        if targetTime <= currentTime:
            print(f"(pglLabJack:digitalOutputAtTime) Target time {targetTime:.6f} is not in the future (current time: {currentTime:.6f}).")
            return False

        def waitAndPulse():
            try:
                # Busy wait until target time
                while self.pglTimestamp.getSecs() < targetTime:
                    pass  # Busy wait for precise timing
                
                # Set the digital output state
                self.ljm.eWriteName(self.h, self.channel, 1 if state else 0)
                
                # If pulseLen is specified, restore state after delay
                if pulseLen is not None:
                    time.sleep(pulseLen / 1000.0)
                    self.ljm.eWriteName(self.h, self.channel, 0 if state else 1)
                    
            except Exception as e:
                print(f"(pglLabJack:digitalOutputAtTime) Error in scheduled pulse: {e}")

        # Start thread to wait and deliver pulse
        thread = threading.Thread(target=waitAndPulse, daemon=True)
        thread.start()

        return True
            
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000, range=10.0):
        '''
        Start analog input reading from specified channels.
        
        Args:
            duration (float): Duration of recording in seconds
            channels (list): List of channel numbers or names
            scanRate (int): Sampling rate in Hz
            scansPerRead (int): Number of scans per read operation
            range (float): Voltage range for analog inputs. Options: 10.0V, 1.0V, 0.1V, 0.01V

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
        if range not in validRanges:
            print(f"(pglLabJack:startAnalogRead) Invalid range {range}V. Valid options: {validRanges}")
            return
        try:
            # set each channel to the specified range
            for channel in channelAddresses:
                self.ljm.eWriteName(self.h, f"{channel}_RANGE", range)
        except Exception as e:
            print(f"(pglLabJack:startAnalogRead) Error setting range: {e}")
            return

        # save parameters
        self.channels = channelAddresses
        self.scanRate = scanRate
        self.scansPerRead = scansPerRead
        self.range = range
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
            target=self.analogReadThread,
            daemon=True
        )
        self.acquisitionThread.start()
           
    def analogReadThread(self):
        """
        Thread function to read analog data from LabJack
        """
        
        # record the start time of the stream
        self.analogStartTimestamp = self.pglTimestamp.getSecs()
        
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
                if (self.pglTimestamp.getSecs() - self.analogStartTimestamp) >= self.analogStreamDuration:
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
        if self.h is None or not self.isReading:
            return None, None

        # waitToFinish
        if not waitToFinish:
            # stop thread
            self.stopEvent.set()

        # wait for acquisition thread
        if self.acquisitionThread.is_alive():
            self.acquisitionThread.join()

        # copy data safely
        with self.bufferLock:
            data = np.array(self.analogBuffer)

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
              
    def start(self):
        '''
        Start the LabJack device
        '''
        if self.isRunning(): return
        print(f"(pglLabJack:start) Starting LabJack device.")
        
    
    def stop(self):
        '''
        Stop the LabJack Device
        '''
        if self.isRunning(): self.stopListener()
        print(f"(pglLabJack:stop) Stopping LabJack device.")

 
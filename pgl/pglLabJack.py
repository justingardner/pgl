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
from pgl import pglDevice
import matplotlib.pyplot as plt

class pglLabJack(pglDevice):
    def __init__(self):
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
    
            # ljm library reference
            self.ljm = ljm
            
            # timestamp utility
            self.pglTimestamp = pglTimestamp()
            
    def __repr__(self):
        if self.h is None:
            return "<pglLabJack device not connected>"
        else:
            return f"<pglLabJack deviceType={self.type} connectionType={self.connectionType} serialNumber={self.serialNumber}>"
        
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000):
        '''
        Start analog input reading from specified channels.

        Args:
            duration (float): Duration in seconds to read data.
            channels (list): List of channel numbers to read from. Defualt [0] which is A0
            scanRate (int): Scans per second.
            scansPerRead (int): Number of scans to read per read operation.
        '''
        if self.h is None:
            print("(pglLabJack:startAnalogRead) LabJack device not connected.")
            return
        
        # save parameters
        self.channels = channels
        self.scanRate = scanRate
        self.scansPerRead = scansPerRead
        self.analogStreamDuration = duration
        
        # derived parameters
        self.numChannels = len(channels)
        self.totalScans = int(duration * scanRate)
        self.totalReads = self.totalScans // scansPerRead

        # Buffer for the analog samples
        self.analogBuffer = []
        # lock for thread safety
        self.bufferLock = threading.Lock()
        
        # start the acquisition thread
        self.acquisitionThread = threading.Thread(target=self.analogReadThread)
        self.acquisitionThread.start()

        self.isReading = True
    
    def analogReadThread(self):
        '''
        Thread function to read analog data from LabJack
        '''
                # start the stream
        try:
            self.scanRate = self.ljm.eStreamStart(self.h, self.scansPerRead, self.numChannels, self.channels, self.scanRate)
        except Exception as e:
            print(f"(pglLabJack:startAnalogRead) Error starting stream: {e}")
            return
        
        # get the start time of the buffer
        self.analogStartTimestamp = self.pglTimestamp.getSecs()
        
        while (self.pglTimestamp.getSecs() - self.analogStartTimestamp) < self.analogStreamDuration:
            dataArray, deviceBacklog, ljmBacklog = self.ljm.eStreamRead(self.h)
            with self.bufferLock:
                self.analogBuffer.extend(dataArray)

        self.ljm.eStreamStop(self.h)
        self.ljm.close(self.h)

    def stopAnalogRead(self):
        '''
        Stop the analog reading and plot the data
        '''
        if not self.isReading: return
        self.isReading = False
        
        # Wait for acquisition to finish
        self.acquisitionThread.join()

        # copy data safely
        with self.bufferLock:
            data = np.array(self.analogBuffer)

        # Plot result
        plt.figure()
        plt.plot(np.linspace(0, self.analogStreamDuration, len(data)), data)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("AIN0 Acquisition")
        plt.show()


    
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

 
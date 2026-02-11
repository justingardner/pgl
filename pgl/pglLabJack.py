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
                                      will return to its original state after this time.

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
    
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000):
        '''
        Start analog input reading from specified channels.
        
        Args:
            duration (float): Duration of recording in seconds
            channels (list): List of channel numbers or names
            scanRate (int): Sampling rate in Hz
            scansPerRead (int): Number of scans per read operation
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
        
        # save parameters
        self.channels = channelAddresses
        self.scanRate = scanRate
        self.scansPerRead = scansPerRead
        self.analogStreamDuration = duration

        # derived parameters
        self.numChannels = len(channels)
        self.totalScans = int(duration * scanRate)
        self.totalReads = self.totalScans // scansPerRead

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

    def stopAnalogRead(self):
        """
        Stop the analog reading and return time and data arrays.
        """
        if self.h is None or not self.isReading:
            return None, None

        # signal thread to stop
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
        data = data[:numSamples * self.numChannels]  # trim incomplete scans
        data = data.reshape(numSamples, self.numChannels)

        # create time array (one timestamp per sample, not per data point)
        time = np.linspace(
            0,
            (numSamples / self.scanRate),
            numSamples
        )

        return time, data

    def getCycles(self, time, data, cycleLen=None, digitalSyncChannel=None, digitalSyncThreshold=None, ignoreInitial=None):
        '''
        Extract cycles from analog data based on fixed cycle length or digital sync triggers.
        
        Args:
            time (array): Time array
            data (array): Data array
            cycleLen (float): Fixed cycle length in seconds (used if digitalSyncChannel is None)
            digitalSyncChannel (int): Channel index to use for digital sync detection
            digitalSyncThreshold (float): Voltage threshold for detecting digital pulse rising edge
            ignoreInitial (float): Number of seconds to ignore from the beginning of data.
                                  If None (default), no data is ignored. Must be non-negative.
            
        Returns:
            dict: Dictionary containing:
                - 'cycles': list of arrays, one per channel, each array is (numCycles, samplesPerCycle)
                - 'cycleTime': time array for one cycle
                - 'mean': list of mean cycles per channel
                - 'std': list of std cycles per channel
                - 'median': list of median cycles per channel
                - 'numCycles': number of cycles detected
                - 'cycleLen': actual cycle length used
                - 'ignoredSamples': number of samples ignored from the beginning
        '''
        if time is None or data is None:
            print("(pglLabJack:getCycles) No data provided.")
            return None
        
        # Validate ignoreInitial parameter
        if ignoreInitial is not None:
            if not isinstance(ignoreInitial, (int, float)):
                print(f"(pglLabJack:getCycles) Error: ignoreInitial must be a number or None, got {type(ignoreInitial).__name__}")
                return None
            if ignoreInitial < 0:
                print(f"(pglLabJack:getCycles) Error: ignoreInitial must be non-negative, got {ignoreInitial}")
                return None
            if ignoreInitial >= time[-1] - time[0]:
                print(f"(pglLabJack:getCycles) Error: ignoreInitial ({ignoreInitial}s) is greater than or equal to total data duration ({time[-1] - time[0]:.3f}s)")
                return None
        
        # Filter data if ignoreInitial is specified
        ignoredSamples = 0
        if ignoreInitial is not None and ignoreInitial > 0:
            # Find the index where time exceeds ignoreInitial seconds from start
            startTime = time[0] + ignoreInitial
            maskIndices = np.where(time >= startTime)[0]
            
            if len(maskIndices) == 0:
                print(f"(pglLabJack:getCycles) Error: No data remains after ignoring initial {ignoreInitial}s")
                return None
            
            startIdx = maskIndices[0]
            ignoredSamples = startIdx
            
            # Slice the data
            time = time[startIdx:]
            if data.ndim == 1:
                data = data[startIdx:]
            else:
                data = data[startIdx:, :]
            
            print(f"(pglLabJack:getCycles) Ignoring first {ignoreInitial}s ({ignoredSamples} samples)")
        
        # Handle single or multi-channel data
        if data.ndim == 1:
            dataToProcess = data.reshape(-1, 1)
            numChannels = 1
        else:
            dataToProcess = data
            numChannels = data.shape[1]
        
        # Determine cycle start indices and samples per cycle
        if digitalSyncChannel is not None and digitalSyncThreshold is not None:
            # Use digital sync channel to detect cycle starts
            syncData = dataToProcess[:, digitalSyncChannel]
            
            # Detect rising edges (when signal crosses threshold from below)
            aboveThreshold = syncData > digitalSyncThreshold
            risingEdges = np.where(np.diff(aboveThreshold.astype(int)) > 0)[0] + 1
            
            if len(risingEdges) < 2:
                print(f"(pglLabJack:getCycles) Warning: Found {len(risingEdges)} rising edges. Need at least 2 for cycle analysis.")
                return None
            
            # Calculate cycle length from detected triggers
            cycleLengths = np.diff(risingEdges)
            samplesPerCycle = int(np.median(cycleLengths))
            cycleLen = samplesPerCycle * np.mean(np.diff(time))
            
            # Use detected trigger indices as cycle starts
            cycleStarts = risingEdges[:-1]  # Exclude last one to ensure complete cycles
            
        else:
            # Use fixed cycleLen
            if cycleLen is None:
                print("(pglLabJack:getCycles) Must provide either cycleLen or digitalSyncChannel/digitalSyncThreshold.")
                return None
                
            dt = np.mean(np.diff(time))
            samplesPerCycle = int(cycleLen / dt)
            
            # Check if data is long enough for at least one cycle
            if len(time) < samplesPerCycle:
                print(f"(pglLabJack:getCycles) Warning: Data length ({len(time)} samples) is shorter than one cycle ({samplesPerCycle} samples).")
                return None
            
            # Generate regular cycle starts
            numCycles = len(dataToProcess) // samplesPerCycle
            cycleStarts = np.arange(numCycles) * samplesPerCycle
        
        # Create cycle time array
        cycleTime = np.linspace(0, cycleLen, samplesPerCycle)
        
        # Extract cycles for each channel
        allCycles = []
        allMeans = []
        allStds = []
        allMedians = []
        
        for ch in range(numChannels):
            channelData = dataToProcess[:, ch]
            cycles = []
            
            for startIdx in cycleStarts:
                endIdx = startIdx + samplesPerCycle
                
                # Skip if cycle extends beyond data
                if endIdx > len(channelData):
                    continue
                
                cycle = channelData[startIdx:endIdx]
                
                # Pad or trim to exact samplesPerCycle length (for digital sync with varying lengths)
                if len(cycle) < samplesPerCycle:
                    cycle = np.pad(cycle, (0, samplesPerCycle - len(cycle)), mode='edge')
                elif len(cycle) > samplesPerCycle:
                    cycle = cycle[:samplesPerCycle]
                    
                cycles.append(cycle)
            
            if len(cycles) == 0:
                print(f"(pglLabJack:getCycles) No complete cycles found for channel {ch}.")
                return None
            
            # Convert to array (numCycles, samplesPerCycle)
            cycles = np.array(cycles)
            
            # Calculate statistics
            meanCycle = np.mean(cycles, axis=0)
            stdCycle = np.std(cycles, axis=0)
            medianCycle = np.median(cycles, axis=0)
            
            allCycles.append(cycles)
            allMeans.append(meanCycle)
            allStds.append(stdCycle)
            allMedians.append(medianCycle)
        
        return {
            'cycles': allCycles,
            'cycleTime': cycleTime,
            'mean': allMeans,
            'std': allStds,
            'median': allMedians,
            'numCycles': len(cycles),
            'cycleLen': cycleLen,
            'ignoredSamples': ignoredSamples
        }
    
    
    def plotAnalogRead(self, time, data, cycleLen=None, digitalSyncChannel=None, digitalSyncThreshold=3, ignoreInitial=None):
        '''
        Plot the analog read data

        Args:
            time (array): Time array
            data (array): Data array
            cycleLen (float): If provided, creates a second subplot showing cycle-averaged data
            digitalSyncChannel (int): Channel index to use for digital sync detection
            digitalSyncThreshold (float): Voltage threshold for detecting digital pulse rising edge
            ignoreInitial (float): Time in seconds to ignore at the beginning of the recording for
                displaying cycles (e.g., to exclude initial transients). If None, no data is ignored. Must be non-negative.
        '''
        if time is None or data is None:
            print("(pglLabJack:plotAnalogRead) No data to plot.")
            return
        
        # Determine number of subplots
        if cycleLen is None and digitalSyncChannel is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # First subplot: Original data
        if data.ndim == 1:
            # Single channel
            axes[0].plot(time, data, label=self.channels[0])
        else:
            # Multiple channels
            for i in range(data.shape[1]):
                axes[0].plot(time, data[:, i], label=self.channels[i])
        
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Voltage (V)")
        axes[0].set_title("LabJack Analog Input")
        axes[0].legend()
        axes[0].grid(True)
        
        # Second subplot: Cycle-averaged data
        if cycleLen is not None or digitalSyncChannel is not None:
            # Get cycles using the getCycles function
            cycleData = self.getCycles(time, data, cycleLen, digitalSyncChannel, digitalSyncThreshold, ignoreInitial)
            
            if cycleData is None:
                axes[1].text(0.5, 0.5, 'Unable to extract cycles', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_xlabel("Time in Cycle (s)")
                axes[1].set_ylabel("Voltage (V)")
                axes[1].set_title("Cycle-Averaged Data")
            else:
                cycleTime = cycleData['cycleTime']
                numChannels = len(cycleData['cycles'])
                
                # convert time to ms, if cycle time is less than 1 second
                if max(cycleTime) < 1.0:
                    cycleTime = cycleTime * 1000
                    xAxisLabel = "Time in Cycle (ms)"
                else:
                    xAxisLabel = "Time in Cycle (s)"
                    
                for ch in range(numChannels):
                    cycles = cycleData['cycles'][ch]
                    meanCycle = cycleData['mean'][ch]
                    stdCycle = cycleData['std'][ch]
                    
                    # Plot individual trials as thin lines in background
                    for i in range(cycles.shape[0]):
                        axes[1].plot(cycleTime, cycles[i, :], color=f'C{ch}', alpha=0.2, linewidth=0.5)
                    
                    # Plot standard deviation as filled polygon
                    axes[1].fill_between(cycleTime, 
                                         meanCycle - stdCycle, 
                                         meanCycle + stdCycle,
                                         color=f'C{ch}', alpha=0.3)
                    
                    # Plot mean as solid line
                    axes[1].plot(cycleTime, meanCycle, color=f'C{ch}', 
                               linewidth=2, label=self.channels[ch])
                
                titleStr = "Trigger-Averaged Data" if digitalSyncChannel is not None else "Cycle-Averaged Data"
                axes[1].set_xlabel(xAxisLabel)
                axes[1].set_ylabel("Voltage (V)")
                axes[1].set_title(f"{titleStr} (n={cycleData['numCycles']} cycles)")
                axes[1].legend()
                axes[1].grid(True)
        
        plt.tight_layout()
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

 
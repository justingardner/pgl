################################################################
#   filename: pglDevice.py
#    purpose: Device class
#         by: JLG
#       date: July 28, 2025
################################################################

###########
# # Import
##########
from asyncio import subprocess
import io
import sys
from time import sleep
from typing import Optional
from pgl import pglTimestamp
from .pglEvent import List, pglEvent
from .pglEventListener import pglEventListener
from dataclasses import dataclass
from .pglSerialize import pglSerialize
import matplotlib.pyplot as plt
import numpy as np
from .pglMessages import pglMessages
from .pglTimestamp import pglTimestamp
import time
import threading

#################################################################
# Parent class for devices
#################################################################
class pglDevice:
    """
    Parent class for all pglDevice types
    """
    def __init__(self, deviceType, deviceDescription=None):  
        '''
        Initialize the _pglDevice instance.
        
        Args:
            pgl (object): The pgl instance.
            type (str): The type of the device.
                
        Returns:
            None
        '''
        # set the device type
        self.deviceType = deviceType
        if deviceDescription == None:
            self.deviceDescription = deviceType
        else:
            self.deviceDescription = deviceDescription
        # set the initialization time
        self.startTime = pglTimestamp.getDateAndTime()
        # set the device status
        self.currentStatus = 0
        # some fields about the device that will be set by subclasses
        self.device = None
        self.deviceAttributes = {}
        # set verbosity
        self.verbose = 1


    def __repr__(self):
        return f"<pglDevice type={self.deviceType}>"
    
    def __del__(self):
        """
        Clean up the _pglDevice instance.
        """
        # Perform any necessary cleanup here
        print(f"(pglDevice) Cleaning up device of type {self.deviceType}")
        pass

    def poll(self):
        """
        Poll the event.

        This method is used to poll the event for any updates or changes.
        Should be implemented in subclasses.

        """
        # Implement polling logic here
        return f"(pglDevice) Device {self.deviceType}: poll not implemented"
    
    def start(self):
        pass
    def stop(self):
        pass
    
    def status(self):
        """
        Get the status of the device.

        This method retrieves the current status of the device.
        Should be implemented in subclasses.

        Returns:
            str: A string representing the current status of the device.
        """
        # Implement status retrieval logic here
        return f"(pglDevice) Device {self.deviceType}: status not implemented"

    @property
    def isActive(self):
        '''
        True if initialized and running, False if not.
        
        Should be implemented in subclass, defaults to returning True
        '''
        return True
        
#################################################################
# pglDevices is mixed into pgl and handles multiple pglDevice instances
#################################################################
class pglDevices:
    """
    Class to manage multiple pglDevice instances.
    """
    
    def __init__(self):
        """
        Initialize the pglDevices instance.
        """
        self.devices = []

    def devicesAdd(self, device):
        """
        Add a pglDevice instance to the list of devices.

        Args:
            device (pglDevice): The device to add.
        """
        if isinstance(device, pglDevice):
            self.devices.append(device)
            print(f"(pglDevices) Added device: {device.deviceType}")
        else:
            print("(pglDevices) Error: Device must be an instance of pglDevice.")

    def devicesGet(self, deviceType):
        '''
        Get a pglDevice instance by its type.

        Args:
            deviceType (str): The type of the device to retrieve.

        Returns:
            pglDevice: The device instance if found, None otherwise.
        '''
        return [d for d in self.devices if isinstance(d, deviceType)]

    def devicesGetKeyboard(self):
        '''
        Get a pglKeyboardMouse device from pgl. This assumes that there is only
        one pglKeyboardMouse device in the devices list.

        Returns:
            pglDevice: The device instance if found, None otherwise.
        '''
        from .pglKeyboardMouse import pglKeyboardMouse
        d = self.devicesGet(pglKeyboardMouse)
        return d[0] if len(d) > 0 else None

    def setEatKeys(self, keyCodes=None, keyChars=None):
        '''
        Set eat keys for keyboard device. This calls the function setEatKeys on the pglKeyboardMouse device if it exists.

        Args:
            keyCodes (list, optional): List of key codes to eat. Defaults to None.
            keyChars (list, optional): List of key characters to eat. Defaults to None.
        '''
        keyboardDevice = self.devicesGetKeyboard()
        if keyboardDevice is not None:
            keyboardDevice.setEatKeys(keyCodes=keyCodes, keyChars=keyChars)

    def poll(self):
        """
        Poll all devices for updates.

        This method iterates through all devices and calls their poll method.
        """
        eventList = []
        for device in self.devices: 
            # poll each device for events
            eventList = device.poll()
            # add them to the events list
            self.eventsAdd(eventList)
        # return the eventList
        return eventList

################################################################
#  Abstract base class defining the interface for
#  digital IO devices. Concrete devices (e.g.,
#  pglLabJack) should inherit from this and implement
#  the stubbed methods.
################################################################
class pglDigitalIODevice(pglDevice):
    '''
    Abstract base class for digital IO devices.

    Any concrete digital IO device (e.g., a LabJack, an Arduino, a
    National Instruments DAQ, etc.) should inherit from this class and
    implement the stubbed methods below. This defines the common
    interface that the rest of pgl can rely on regardless of the
    underlying hardware.

    Subclasses are expected to:
        - Establish/close the hardware connection (in __init__ / stop)
        - Set self.digitalOutputConfigured appropriately
        - Implement all methods marked as NotImplementedError below
    '''

    def __init__(self, deviceType="DigitalIODevice"):
        # whether a digital output channel has been configured
        self.digitalOutputConfigured = False
        self.digitalChannels={}
        
        # for outputing words
        self.wordDigitalChanels = []
        self.wordBits = 0
        self.wordMaxValue = 0
        
        super().__init__(deviceType=deviceType)

    def __repr__(self):
        return f"<pglDigitalIODevice deviceType={getattr(self, 'deviceType', 'Unknown')}>"

    ################################################################
    # Digital output interface
    ################################################################
    def setupDigitalOutput(self, channel=0, pulseLen = 1):
        '''
        Configure a digital output channel.

        Implementations should convert the channel argument into the
        hardware-specific representation, set the channel as an output,
        initialize it to a known state (typically LOW), and set
        self.digitalOutputConfigured = True on success (False on failure).

        Args:
            channel (int or str): Digital channel number or name
            pulseLen (int): Time in ms for digital pulse to last (for digitalOutputPulse)
        '''
        # store pulseLen for this channel, which is used by digitalOutputPulse and digitalOutputPulseAtTime
        self.digitalChannels[channel]= {"pulseLen": pulseLen}

    def digitalOutput(self, state, pulseLen=None):
        '''
        Set the digital output state immediately.

        Implementations should check self.digitalOutputConfigured first.
        If pulseLen is provided, the output should return to the opposite
        state after pulseLen milliseconds.

        Args:
            state (bool): True for HIGH, False for LOW
            pulseLen (float or None): Pulse length in milliseconds

        Returns:
            timestamp (float or None): Timestamp when output was set,
                                       or None on error.
        '''
        raise NotImplementedError("(pglDigitalIODevice:digitalOutput) Subclass must implement digitalOutput().")

    def setupDigitalOutputWord(self, channels=None):
        '''
        sets up the channels (list of channel numbers) for outputing words, where the list goes
        from lowst order bit to highest order bit in order
        
        channels must first be initialized with setupDigitalOutput
        
        Args:
            channels (list of int): channels to use in word
        '''
        if channels is None:
            pglMessages.warning(f"digital output word has no channels")
            return
        
        # make sure each channel is valid
        channelsNotSetup = set(channels) - self.digitalChannels.keys()
        if channelsNotSetup:
            for channel in channelsNotSetup:
                pglMessages.warning(f"Channel {channel} has not been setup",level=1)
            return      
        
        # save the list
        self.wordDigitalChanels = channels
        self.wordBits = len(channels)
        self.wordMaxValue = 2**self.wordBits-1
              
        
    def digitalOutputWord(self, outputWord):
        '''
        Will put out a pulse on the digital channels setup with setupDigitalOutputWord
        representing the outputWord        
        
        Args:
            channels (list of int): channels to use in word
        '''
        if outputWord < 0 or outputWord > self.wordMaxValue:
            pglMessages.warning(f"outputWord must be between 0 and {self.wordMaxValue}: {outputWord}",level=1)
            return
        
        # write each bit out
        for iBit in range(self.wordBits):
            val = (outputWord >> iBit) & 0x1
            # send pulses for all positive ones
            if val:self.digitalOutputPulse(self.wordDigitalChanels[iBit])
            
    def digitalOutputPulse(self, channel):
        '''
        Send a digital output pulse. Call setupDigitalOutput() first to configure the channel.

        Args:
            channel: channel to place the digital output pulse on to
        Returns:
            timestamp (float): Timestamp of when the digital output was set,
                               or None if there was an error.
        '''
        # create a function to end pulse
        def restoreState():
            try:
                # wait for pluseLen (in ms)
                time.sleep(self.digitalChannels[channel]["pulseLen"] / 1000.0)
                # reset the state
                self.digitalOutput(channel, 0)
            except Exception as e:
                print(f"(pglLabJack:setDigitalOutput) Error restoring {self.digitalChannel[channel]["name"]}: {e}")

        # set high
        timestamp = self.digitalOutput(channel, 1)
        if timestamp is None: return
        
        # start thread to reset state
        thread = threading.Thread(target=restoreState, daemon=True)
        thread.start()
        
        return timestamp

    def digitalOutputPulseAtTime(self, targetTime, channel):
        '''
        Set the digital output state at a specified future time. Call setupDigitalOutput() first to configure the channel.

        WARNING: If calling mulitple times, ensure pulses don't overlap in time as this code
            does not currently handle multiple overlapping pulses and may produce unexpected results if pulses overlap.

        Args:
            targetTime (float): Timestamp (in seconds) when the pulse should be delivered.
                                Must be in the future relative to pglTimestamp.getSecs().
            channel: channel to put the pulse on

        Returns:
            bool: True if the pulse was successfully scheduled, False otherwise
        '''

        if not self.digitalOutputConfigured:
            pglMessages.warning("Digital output channel not configured. Call setupDigitalOutput() first.")
            return False

        # Validate that targetTime is in the future
        currentTime = pglTimestamp.getSecs()
        if targetTime <= currentTime:
            pglMessages.warning(f"Target time {targetTime:.6f} is not in the future (current time: {currentTime:.6f}).")
            return False

        def waitAndPulse():
            try:
                # Busy wait until target time
                while pglTimestamp.getSecs() < targetTime:
                    pass  # Busy wait for precise timing
                
                # send a pulse
                self.digitalOutputPulse(channel)
                    
            except Exception as e:
                pglMessages.warning(f"Error in scheduled pulse: {e}")

        # Start thread to wait and deliver pulse
        thread = threading.Thread(target=waitAndPulse, daemon=True)
        thread.start()

        return True
    
################################################################
#  Abstract base class defining the interface for
#  digital IO devices. Concrete devices (e.g.,
#  pglLabJack) should inherit from this and implement
#  the stubbed methods.
################################################################
class pglAnalogInputDevice(pglDevice):
    '''
    Abstract base class for analog input device.

    Any concrete ADC device (e.g., a LabJack, an Arduino)
    should inherit from this class and implement the
    stubbed methods below. This defines the common interface
    that the rest of pgl can rely on regardless of the
    underlying hardware.

    Subclasses are expected to:
        - Establish/close the hardware connection (in __init__ / stop)
        - Set self.analogInputConfigured appropriately
        - Implement all methods marked as NotImplementedError below
    '''

    def __init__(self, deviceType="analogIputDevice"):
        # whether a digital output channel has been configured
        self.analogInputConfigured = False
        super().__init__(deviceType=deviceType)

    def __repr__(self):
        return f"<pglAnalogInputDevice deviceType={getattr(self, 'deviceType', 'Unknown')}>"

    ################################################################
    # start analog input
    ################################################################
    def startAnalogRead(self, duration=2, channels=[0], scanRate=1000, scansPerRead=1000, range=10.0):
        '''
        Start analog input reading from specified channels.

        Args:
            duration (float): Duration of recording in seconds
            channels (list): List of channel numbers or names
            scanRate (int): Sampling rate in Hz
            scansPerRead (int): Number of scans per read operation
            range (float): Voltage range for analog inputs
        '''
        raise NotImplementedError("(pglDigitalIODevice:startAnalogRead) Subclass must implement startAnalogRead().")

    ################################################################
    # stop analog input, should return data
    ################################################################
    def stopAnalogRead(self, waitToFinish=False, doNotTruncate=False):
        '''
        Stop the analog reading and return time and data arrays.

        Args:
            waitToFinish (bool): If True, wait for acquisition to finish.
            doNotTruncate (bool): If True, do not truncate to exact sample count.

        Returns:
            tuple: (time, data) arrays, or (None, None) on error.
        '''
        raise NotImplementedError("(pglDigitalIODevice:stopAnalogRead) Subclass must implement stopAnalogRead().")

    
class pglAnalogTraceData(pglSerialize):
    # Stores short trains of analog data created by pglDigitalIODevice
    # and offers functions for display
    def __init__(self, time, data, channelNames=None):
        '''
        store the time and data
        '''
        self.time = time
        self.data = data
        
        # compute number of channels
        self.numChannels = 1 if data.ndim == 1 else data.shape[1]
        
        # Handle channel names
        if channelNames is None:
            channelNames = []
        else:
            channelNames = list(channelNames)

        # figure out zero-padding width based on channel count
        padWidth = max(3, len(str(self.numChannels - 1)))

        # build a normalized list of channelNames (if any are missing)
        # Will add channelNames like: CH001
        normalizedChannelNames = []
        for i in range(self.numChannels):
            if i < len(channelNames) and channelNames[i] is not None and str(channelNames[i]).strip() != "":
                # coerce existing entry to string
                normalizedChannelNames.append(str(channelNames[i]))
            else:
                # fill missing/blank entry with CHnnn
                normalizedChannelNames.append(f"CH{i:0{padWidth}d}")

        # warn if caller supplied more names than there are channels
        if len(channelNames) > self.numChannels:
            print(f"(pglAnalogTraceData) Warning: {len(channelNames)} names given but only {self.numChannels} channels. Extra names ignored.")
        
        # store the normalized channel names
        self.channelNames = normalizedChannelNames
        
    def __len__(self):
        if self.time is not None:
            return len(self.time)
        else:
            return 0
    
    @property
    def nSamples(self):
        return self.__len__()
    
    def getCycles(self, cycleLen=None, digitalSyncChannel=None, digitalSyncThreshold=None, ignoreInitial=None):
        '''
        Extract cycles from analog data based on fixed cycle length or digital sync triggers.
        
        Args:
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
        if self.time is None or self.data is None:
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
            if ignoreInitial >= self.time[-1] - self.time[0]:
                print(f"(pglLabJack:getCycles) Error: ignoreInitial ({ignoreInitial}s) is greater than or equal to total data duration ({self.time[-1] - self.time[0]:.3f}s)")
                return None
        
        # Filter data if ignoreInitial is specified
        ignoredSamples = 0
        time=self.time
        data=self.data
        if ignoreInitial is not None and ignoreInitial > 0:
            # Find the index where time exceeds ignoreInitial seconds from start
            startTime = self.time[0] + ignoreInitial
            maskIndices = np.where(self.time >= startTime)[0]
            
            if len(maskIndices) == 0:
                print(f"(pglLabJack:getCycles) Error: No data remains after ignoring initial {ignoreInitial}s")
                return None
            
            startIdx = maskIndices[0]
            ignoredSamples = startIdx
            
            # Slice the data
            time = self.time[startIdx:]
            if data.ndim == 1:
                data = self.data[startIdx:]
            else:
                data = self.data[startIdx:, :]
            
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
            cycleLen = samplesPerCycle * np.median(np.diff(time))
            
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
    
    def display(self, cycleLen=None, digitalSyncChannel=None, digitalSyncThreshold=3, ignoreInitial=None, displayStartEnd=None, fig=None):
        '''
        Plot the analog read data

        Args:
            cycleLen (float): If provided, creates a second subplot showing cycle-averaged data
            digitalSyncChannel (int): Channel index to use for digital sync detection
            digitalSyncThreshold (float): Voltage threshold for detecting digital pulse rising edge
            ignoreInitial (float): Time in seconds to ignore at the beginning of the recording for
                displaying cycles (e.g., to exclude initial transients). If None, no data is ignored. Must be non-negative.
            displayStartEnd (float): If not None, will display the first displayStartEnd seconds as separate graphs  
            fig (matplotlib fig): If not none, will plot into the supplied fig
            
        Returns:
            dict: Dictionary containing:
            - 'fig': Figure object
            - 'cycleData': dict from getCycles function (if cycleLen or digitalSyncChannel is provided), otherwise None
            - 'axes': Dictionary of axis objects with keys:
                - 'fullTrace': Full trace axis (always present)
                - 'start': Start segment axis (if displayStartEnd is set)
                - 'end': End segment axis (if displayStartEnd is set)
                - 'cycle': Cycle-averaged axis (if cycleLen or digitalSyncChannel is set)
   
        '''
        retval = {}

        if self.time is None or self.data is None:
            print("(pglLabJack:plotAnalogRead) No data to plot.")
            return
        
        # Determine number of rows needed
        numRows = 1  # Always have full trace row
        if displayStartEnd is not None:
            numRows += 1  # Add row for start/end segments
        if cycleLen is not None or digitalSyncChannel is not None:
            numRows += 1  # Add row for cycle-averaged data
        
        # Determine grid layout
        if displayStartEnd is not None or (cycleLen is not None or digitalSyncChannel is not None):
            if fig is not None:
                fig.clear()
            else:
                fig = plt.figure(figsize=(16, 6 * numRows / 2))
            gs = fig.add_gridspec(numRows, 2, hspace=0.3, wspace=0.3)
        else:
            if fig is not None:
                fig.clear()
            else:
                fig = plt.figure(figsize=(16, 6))
            gs = fig.add_gridspec(1, 1)
        retval['fig'] = fig
        currentRow = 0
        
        # First row: Full analog trace (spans both columns)
        axFullTrace = fig.add_subplot(gs[currentRow, :])
        retval['fullTrace'] = axFullTrace
        if self.data.ndim == 1:
            # Single channel
            axFullTrace.plot(self.time, self.data, label=self.channelNames[0])
        else:
            # Multiple channels
            for i in range(self.data.shape[1]):
                axFullTrace.plot(self.time, self.data[:, i], label=self.channelNames[i])
        
        axFullTrace.set_xlabel("Time (s)")
        axFullTrace.set_ylabel("Voltage (V)")
        axFullTrace.set_title("Analog Trace Data")
        axFullTrace.legend()
        axFullTrace.grid(True)
        currentRow += 1
        
        # Second row: Start and end segments (if requested)
        if displayStartEnd is not None:
            # Determine if we should use ms or s for display
            useMilliseconds = displayStartEnd < 1.0
            timeMultiplier = 1000 if useMilliseconds else 1
            timeUnit = "ms" if useMilliseconds else "s"
            
            # Left column: First displayStartEnd seconds
            axStart = fig.add_subplot(gs[currentRow, 0])
            retval['start'] = axStart
            startIdx = int(displayStartEnd * self.scanRate)
            timeStart = self.time[:startIdx] * timeMultiplier
            if self.data.ndim == 1:
                axStart.plot(timeStart, self.data[:startIdx], label=self.channelNames[0])
            else:
                for ch in range(self.data.shape[1]):
                    axStart.plot(timeStart, self.data[:startIdx, ch], label=self.channelNames[ch])
            
            axStart.set_xlabel(f"Time ({timeUnit})")
            axStart.set_ylabel("Voltage (V)")
            axStart.set_title(f"First {displayStartEnd} seconds")
            axStart.legend()
            axStart.grid(True)
            
            # Right column: Last displayStartEnd seconds
            axEnd = fig.add_subplot(gs[currentRow, 1])
            retval['end'] = axEnd
            endIdx = int(displayStartEnd * self.scanRate)
            timeEnd = self.time[-endIdx:] * timeMultiplier
            if data.ndim == 1:
                axEnd.plot(timeEnd, self.data[-endIdx:], label=self.channelNames[0])
            else:
                for ch in range(self.data.shape[1]):
                    axEnd.plot(timeEnd, self.data[-endIdx:, ch], label=self.channelNames[ch])
            
            axEnd.set_xlabel(f"Time ({timeUnit})")
            axEnd.set_ylabel("Voltage (V)")
            axEnd.set_title(f"Last {displayStartEnd} seconds")
            axEnd.legend()
            axEnd.grid(True)
            currentRow += 1
        
        # Next row: Cycle-averaged data (if requested, spans both columns)
        if cycleLen is not None or digitalSyncChannel is not None:
            axCycle = fig.add_subplot(gs[currentRow, :])
            retval['cycle'] = axCycle
            
            # Get cycles using the getCycles function
            cycleData = self.getCycles(cycleLen, digitalSyncChannel, digitalSyncThreshold, ignoreInitial)
            retval['cycleData'] = cycleData
            
            if cycleData is None:
                axCycle.text(0.5, 0.5, 'Unable to extract cycles', 
                           ha='center', va='center', transform=axCycle.transAxes)
                axCycle.set_xlabel("Time in Cycle (s)")
                axCycle.set_ylabel("Voltage (V)")
                axCycle.set_title("Cycle-Averaged Data")
            else:
                cycleTime = cycleData['cycleTime']
                numChannels = len(cycleData['cycles'])
                
                # Convert time to ms, if cycle time is less than 1 second
                if max(cycleTime) < 1.0:
                    cycleTime = cycleTime * 1000
                    xAxisLabel = "Time in Cycle (ms)"
                else:
                    xAxisLabel = "Time in Cycle (s)"
                    
                for ch in range(numChannels):
                    cycles = cycleData['cycles'][ch]
                    medianCycle = cycleData['median'][ch]
                    
                    # Plot individual trials as thin lines in background
                    for i in range(cycles.shape[0]):
                        axCycle.plot(cycleTime, cycles[i, :], color=f'C{ch}', alpha=0.2, linewidth=0.5)
                                        
                    # Plot median as solid line
                    axCycle.plot(cycleTime, medianCycle, color=f'C{ch}', 
                               linewidth=2, label=self.channelNames[ch])
                
                titleStr = "Trigger-Averaged Data" if digitalSyncChannel is not None else "Cycle-Averaged Data"
                axCycle.set_xlabel(xAxisLabel)
                axCycle.set_ylabel("Voltage (V)")
                axCycle.set_title(f"{titleStr} (n={cycleData['numCycles']} cycles)")
                axCycle.legend()
                axCycle.grid(True)
        
        #plt.tight_layout()
        #plt.show(block=False)    
        
        # Return figure and axes dictionary
        return retval
 
        
        
        
    


    
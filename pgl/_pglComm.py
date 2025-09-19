################################################################
# socket.py: manages the unix socket which is used to 
#            communicate with the standalone mgl engine for
#            pgl psychophysics and experiment library
################################################################
import sys, time, struct, subprocess, os, re
from socket import socket, AF_UNIX, SOCK_STREAM
import numpy as np
import time


class _pglComm:
    # init variables
    s = None
    socketName = None
    verbose = 1

    # Init Function
    def __init__(self, socketName, pgl=None, timeout=10):
        # keep pgl reference
        self.pgl = pgl

        # initialize start time
        startTime = time.time()
        attempt = 0

        # display what we are doing
        sys.stdout.write("(pgl:_pglComm) ")
        sys.stdout.flush()
        while True:
            try:
                self.s = socket(AF_UNIX, SOCK_STREAM)
                self.s.connect(socketName)
                self.socketName = socketName
                print("Connected to:", socketName)
                return
            except (FileNotFoundError, ConnectionRefusedError):
                # Keep trying until timeout
                elapsed = time.time() - startTime
                if elapsed > timeout:
                    print("\n(pgl:_pglComm) ❌ Timeout: Could not connect to socket:", socketName)
                    self.s = None
                    return None
                # Print a dot for feedback
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(0.5)  # Wait before retrying

    def isOpen(self):
        """
        Check if the socket is open.
        """
        return self.s is not None

    def close(self):
        """
          Close the socket connection. 
        """
        if os.path.exists(self.socketName):
            try:
                os.remove(self.socketName)
                print("(pgl:_pglComm) Closed socket:", self.socketName)
            except Exception as e:
                print("(pgl:_pglComm) ❌ Error closing socket:", e)
            finally:
                self.s = None

    def write(self, message):
        """
        Write a message to the socket.
        """
        # check socket
        if not self.s:
            print("(pgl:_pglComm) ❌ Not connected to socket")
            return
        # Pack data for sending
        if type(message) == np.uint16:
            packed = struct.pack('@H', message)
        # single int value
        elif type(message) == np.uint32:
            # Pack as 4-byte unsigned int
            packed = struct.pack('@I', message)        
        # single float value
        elif type(message) == float or type(message) == np.float32:
            # Pack as 4-byte double float
            packed = struct.pack('@f', message)
        elif type(message) == np.double or type(message) == np.float64:
            # Pack as 8-byte double float
            packed = struct.pack('@d', message)
        # array of floats
        elif isinstance(message, np.ndarray) and np.issubdtype(message.dtype, np.float32):
            # pack as 4 byte floats
            packed = message.astype(np.float32).tobytes()
        # array of doubles
        elif isinstance(message, np.ndarray) and np.issubdtype(message.dtype, np.float64):
            # pack as 8 byte doubles
            packed = message.astype(np.float64).tobytes()
        else:
            raise TypeError("Unsupported data type")

        try:
            if self.verbose>=3:print(f"(pgl:_pglComm) Sending message with length {len(packed)} bytes")
             # if we are logging commands for replay, log the command values
            if self.pgl.commandRecording: self.pgl.logCommandData(packed)
            # send the packed data
            self.s.sendall(packed)
            if self.verbose > 1: print("(pgl:_pglComm) Message sent:", message)
        except Exception as e:
            print("(pgl:_pglComm) ❌ Error sending message:", e)
    
    def writeCommand(self, commandName):
        """
        Write a command to the socket by its name.

        Args:
            commandName (str): The name of the command to send.
        
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.s:
            print("(pgl:_pglComm) ❌ Not connected to socket")
            return False
        
        # get the command value from the string
        commandValue = self.getCommandValue(commandName)
        if commandValue is None:
            print(f"(pgl:_pglComm) ❌ Command '{commandName}' not found")
            return False
        
        # display command if high verbosity is set
        if self.verbose>1:
            print(f"(pgl:_pglComm) Sending command: {commandName} (value: {commandValue})")

        # if we are logging commands for replay, log the command
        if self.pgl.commandRecording: self.pgl.logCommandValue(commandValue)

        # write the command value to the socket
        self.write(commandValue)
        return True
    def replayCommand(self, commandData):
        """
        Replay a command to the socket.
        
        Args:
            commandValue (int): The value of the command to send.
            commandData (bytes, optional): Additional data associated with the command.
        
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.s:
            print("(pgl:_pglComm) ❌ Not connected to socket")
            return False
        
        for chunk in commandData: self.s.sendall(chunk)
 
        # read the command results
        commandResults = self.readCommandResults()
        
        return True
    
    def getCommandValue(self,commandName):
        """
        Get the value of a command by its name.
        """
        return(self.commandValues.get(commandName))
    def getCommandName(self,commandValue):
        """
        Get the name of a command by its value.
        """
        return(self.commandNames.get(commandValue))
    
    def read(self, dataType, numRows=1, numCols=1, numSlices=1):
        """
        Read a message from the socket.
        """
        if not self.s:
            print("(pgl:_pglComm) ❌ Not connected to socket")
            return None

        try:
            # Read the appropriate number of bytes based on the data type
            numBytes = np.dtype(dataType).itemsize*numRows*numCols*numSlices
            packed = self.recvBlocking(numBytes)
            # check length of packed data
            if len(packed) != numBytes:
                print(f"(pgl:_pglComm:read) ❌ Expected {numBytes} bytes ({numRows}x{numCols}x{numSlices} of {np.dtype(dataType).itemsize}), but received {len(packed)} bytes")
                return None
            else:
                # unpack the data and reshape it
                return np.squeeze(np.frombuffer(packed, dtype=dataType).reshape((numRows, numCols, numSlices)))
        except Exception as e:
            print("(pgl:_pglComm:read) ❌ Error reading message:", e)
            return None

    def recvBlocking(self, numBytes):
        '''
        Receive exactly numBytes from the socket. Will block until all bytes are received.
        '''
        bytesReceived = 0
        packed = bytearray()  # Use a bytearray to collect the bytes

        while bytesReceived < numBytes:
            chunk = self.s.recv(numBytes - bytesReceived)  # Receive remaining bytes
            if not chunk:  # If chunk is empty, the connection has been closed
                raise ConnectionError("(pgl:_pglComm:recvBlocking) Connection closed unexpectedly")
            packed.extend(chunk)  # Append the received chunk to the packed bytearray
            bytesReceived += len(chunk)  # Update the count of received bytes

        return packed

    def readAck(self):
        """
        Read an acknowledgment from the socket.
        
        Returns:
            float: The acknowledgment time from the socket, or None if an error occurs.
        """
        if not self.s:
            print("(pgl:_pglComm:readAck) ❌ Not connected to socket")
            return None
        
        try:
            ack = self.read(np.double)
            if ack is None:
                print("(pgl:_pglComm:readAck) ❌ Error reading acknowledgment")
                return None
            return ack
        except Exception as e:
            print("(pgl:_pglComm:readAck) ❌ Error reading acknowledgment:", e)
            return None
    def readCommandResults(self, ack=None, nCommands=1):
        """
        Read the results from the socket after sending a command.
        
        Returns:
            np.ndarray: The results read from the socket, or None if an error occurs.
        """
        if not self.s:
            print("(pgl:_pglComm:readCommandResults) ❌ Not connected to socket")
            return None
        
        try:
            commandResults = {}
            # Read ack if not passed in
            if ack is None:
                commandResults['ack'] = self.read(np.double)
            else:
                commandResults['ack'] = ack
            # Read the rest of the command results
            commandResults['commandCode'] = self.read(np.uint16, nCommands)
            commandResults['success'] = self.read(np.uint32, nCommands)
            commandResults['processedTime'] = self.read(np.double, nCommands)
            commandResults['vertexStart'] = self.read(np.double, nCommands)
            commandResults['vertexEnd'] = self.read(np.double, nCommands)
            commandResults['fragmentStart'] = self.read(np.double, nCommands)
            commandResults['fragmentEnd'] = self.read(np.double, nCommands)
            commandResults['drawableAcquired'] = self.read(np.double, nCommands)
            commandResults['drawablePresented'] = self.read(np.double, nCommands)
            return(commandResults)
        
        except Exception as e:
            print("(pgl:_pglComm:readCommandResults) ❌ Error reading results:", e)
            return None



    def parseCommandValues(self, filename="mglCommandTypes.h"):
        """
        Parse the command values from the mglCommandTypes.h file.

        This function reads the mglCommandTypes.h file, extracts command names and their values,
        and returns a dictionary mapping command names to their corresponding values.

        """
        self.commandValues = {}

        # check for file
        if not os.path.isfile(filename):
            print(f"(pgl:_pglComm:parseCommandValues) ❌ Error: File not found: {filename}")
            # close connection, since we are now screwed
            self.close()
            return
        
        with open(filename, "r") as f:
            lines = f.readlines()

        inEnum = False

        for line in lines:
            line = line.strip()

            # Start of the enum
            if line.startswith("typedef enum mglCommandCode"):
                inEnum = True
                continue

            if inEnum:
                # End of the enum
                if line.startswith("}"): 
                    inEnum = False
                    break

                # Match lines like: mglPing = 0,
                match = re.match(r"(mgl\w+)\s*=\s*([0-9]+|UINT16_MAX)", line)
                if match:
                    name, valueStr = match.groups()
                    if valueStr == "UINT16_MAX":
                        value = np.uint16(0xFFFF)
                    else:
                        value = np.uint16(int(valueStr))
                    self.commandValues[name] = value
        # now let's make a reverse lookup dictionary
        self.commandNames = {v: k for k, v in self.commandValues.items()}
    def getPID(self):
        """
        Find the PID of the process who made the socket
        """
        try:
            output = subprocess.check_output(['lsof', '-U', self.socketName])
            for line in output.decode().splitlines():
                if 'pgl' in line:
                    return int(line.split()[1])
        except Exception as e:
            print("(pgl:_pglComm) ❌ Error finding PID:", e)
        return None
    
   
class pglSerial:
    '''
    Class representing a serial communication interface.    
    '''
    def __init__(self, port=None, baudrate=9600, dataLen=8, parity='n', stopBits=1, timeout=1):
        '''
        Initialize the serial communication interface.
        Args:
            port (str, optional): The serial port to connect to. If None, will prompt user to select a port.
            baudrate (int, optional): The baud rate for the serial communication. Default is 9600.
            dataLen (int, optional): The number of data bits. Default is 8.
            parity (str, optional): The parity bit ('e','o','n'). Default is 'n' (none).
            stopBits (int, optional): The number of stop bits. Default is 1.
            timeout (float, optional): The read timeout in seconds. Default is 1 second.
        '''
        try:
            import serial
        except ImportError:
            print("(pglSerialComm) ❌ Error: pyserial library is not installed. Please install it to use serial communication.")
            self.serial = None
            return

       # parse arguments
        parityMap = {
              'e': serial.PARITY_EVEN,
              'o': serial.PARITY_ODD,
              'n': serial.PARITY_NONE}
        parity = parityMap.get(parity.lower(), serial.PARITY_NONE)
        
        # get the dataLen
        dataLenMap = {
            5: serial.FIVEBITS,
            6: serial.SIXBITS,
            7: serial.SEVENBITS,
            8: serial.EIGHTBITS}    
        dataLen = dataLenMap.get(dataLen, serial.EIGHTBITS)
        
        # get the stopBits
        stopBitsMap = {
            1: serial.STOPBITS_ONE,
            1.5: serial.STOPBITS_ONE_POINT_FIVE,
            2: serial.STOPBITS_TWO}
        stopBits = stopBitsMap.get(stopBits, serial.STOPBITS_ONE)
        
        # get port if not provided
        if port is None:
            
            # get a list of ports
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            
            # if there are no ports, then return displaying error
            if len(ports) == 0:
                print("(pglSerialComm) ❌ No serial ports found.")
                self.serial = None
                return
            else:
                # display info about each port
                for iPort, port in enumerate(ports):
                    print(f"(pglSerialComm) {iPort}: {port.device} - {port.description}")
                
                # ask user to select port
                print(f"(pglSerialComm) Select port [0-{len(ports)-1}]: ")
                
                # wait for user to select port
                portIndex = int(input(""))
                if portIndex < 0 or portIndex >= len(ports):
                    print("(pglSerialComm) ❌ Invalid port index.")
                    self.serial = None
                    return
                
                # port has been selected
                port = ports[portIndex].device
        try:
            # open port
            self.serial = serial.Serial(port, baudrate, timeout=timeout, bytesize=dataLen, parity=parity, stopbits=stopBits)
            print(f"(pglSerialComm) Connected to serial port: {port} at {baudrate} baud.")
        
        except Exception as e:
            # check for error on opening port
            print(f"(pglSerialComm) ❌ Error connecting to serial port {port}: {e}")
            self.serial = None
    
    def write(self, data):
        '''
        Write data to the serial port.
        '''
        if self.serial is None:
            print("(pglSerialComm) ❌ Serial port not initialized.")
            return
        try:
            # flush anything in the input buffer
            self.flush()
            # convert string data to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            # write data
            self.serial.write(data)
            
        except Exception as e:
            print(f"(pglSerialComm) ❌ Error writing to serial port: {e}")
    
    def read(self, timeout=10.0):
        """
        Read all available data from the serial port, waiting up to `timeout` seconds
        for a response.
        """
        if self.serial is None:
            print("(pglSerialComm) ❌ Serial port not initialized.")
            return None

        start_time = time.time()
        received_data = bytearray()

        try:
            while True:
                # Check how many bytes are waiting
                bytes_waiting = self.serial.in_waiting
                if bytes_waiting:
                    # Read all available bytes
                    received_data += self.serial.read(bytes_waiting)
                    time.sleep(0.1)  # Short delay to allow more data to arrive
            
                # Break if we have received some data and no new bytes arrived
                if received_data and self.serial.in_waiting == 0:
                    break

                # Break if timeout expired
                if (time.time() - start_time) > timeout:
                    break

                # wait before checking again
                time.sleep(0.2)

            if received_data:
                return bytes(received_data)  # return as immutable bytes
            else:
                print("(pglSerialComm) ❌ No data received before timeout.")
                return None

        except Exception as e:
            print(f"(pglSerialComm) ❌ Error reading from serial port: {e}")
            return None
        
    def flush(self):
        '''
        Flush the serial port input and output buffers.
        '''
        if self.serial is None:
            print("(pglSerialComm) ❌ Serial port not initialized.")
            return
        try:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except Exception as e:
            print(f"(pglSerialComm) ❌ Error flushing serial port: {e}")
            
    def close(self):
        '''
        Close the serial port.
        '''
        if self.serial is None:
            return
        try:
            self.serial.close()
        except Exception as e:
            print(f"(pglSerialComm) ❌ Error closing serial port: {e}") 

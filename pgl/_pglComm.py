################################################################
# socket.py: manages the unix socket which is used to 
#            communicate with the standalone mgl engine for
#            pgl psychophysics and experiment library
################################################################
import sys, time, struct, subprocess, os, re
from socket import socket, AF_UNIX, SOCK_STREAM
import numpy as np

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
    
   
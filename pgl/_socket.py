################################################################
# socket.py: manages the unix socket which is used to 
#            communicate with the standalone mgl engine for
#            pgl psychophysics and experiment library
################################################################
import socket, sys, time, struct, subprocess, os, re
import numpy as np

class _socket:
    # init variables
    s = None
    socketName = None
    verbose = 1

    # Init Function
    def __init__(self,socketName,timeout=10):
        # initialize start time
        startTime = time.time()
        attempt = 0

        # display what we are doing
        sys.stdout.write("(pgl:_socket) ")
        sys.stdout.flush()
        while True:
            try:
                self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.s.connect(socketName)
                self.socketName = socketName
                print("Connected to:", socketName)
                return
            except (FileNotFoundError, ConnectionRefusedError):
                # Keep trying until timeout
                elapsed = time.time() - startTime
                if elapsed > timeout:
                    print("\n(pgl:_socket) ❌ Timeout: Could not connect to socket:", socketPath)
                    return None
                # Print a dot for feedback
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(0.5)  # Wait before retrying

    def close(self):
        """
          Close the socket connection. 
        """
        if os.path.exists(self.socketName):
            try:
                os.remove(self.socketName)
                print("(pgl:_socket) Closed socket:", self.socketName)
            except Exception as e:
                print("(pgl:_socket) ❌ Error closing socket:", e)
            finally:
                self.s = None

    def write(self, message):
        """
        Write a message to the socket.
        """
        # check socket
        if not self.s:
            print("(pgl:_socket) ❌ Not connected to socket")
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
        elif type(message) == np.double:
            # Pack as 8-byte double float
            packed = struct.pack('@d', message)
        # array of floats
        elif isinstance(message, np.ndarray) and np.issubdtype(message.dtype, np.floating):
            # pack as 4 byte floats
            packed = message.astype(np.float32).tobytes()
        else:
            raise TypeError("Unsupported data type")

        try:
            self.s.sendall(packed)
            if self.verbose > 1: print("(pgl:_socket) Message sent:", message)
        except Exception as e:
            print("(pgl:_socket) ❌ Error sending message:", e)
    
    def writeCommand(self, commandName):
        """
        Write a command to the socket by its name.

        Args:
            commandName (str): The name of the command to send.
        
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.s:
            print("(pgl:_socket) ❌ Not connected to socket")
            return False
        
        commandValue = self.commandTypes.get(commandName)
        if commandValue is None:
            print(f"(pgl:_socket) ❌ Command '{commandName}' not found")
            return False
        
        self.write(np.uint16(commandValue))
        return True
    def read(self, dataType, numRows=1, numCols=1, numSlices=1):
        """
        Read a message from the socket.
        """
        if not self.s:
            print("(pgl:_socket) ❌ Not connected to socket")
            return None

        try:
            # Read the appropriate number of bytes based on the data type
            numBytes = np.dtype(dataType).itemsize*numRows*numCols*numSlices
            packed = self.s.recv(numBytes)
            # check length of packed data
            if len(packed) != numBytes:
                print("(pgl:_socket) ❌ Expected ", numBytes, "bytes, but received", len(packed), "bytes")
                return None
            else:
                # unpack the data and reshape it
                return np.squeeze(np.frombuffer(packed, dtype=dataType).reshape((numRows, numCols, numSlices)))
        except Exception as e:
            print("(pgl:_socket) ❌ Error reading message:", e)
            return None
        
    def readAck(self):
        """
        Read an acknowledgment from the socket.
        
        Returns:
            float: The acknowledgment time from the socket, or None if an error occurs.
        """
        if not self.s:
            print("(pgl:_socket:readAck) ❌ Not connected to socket")
            return None
        
        try:
            ack = self.read(np.double)
            if ack is None:
                print("(pgl:_socket:readAck) ❌ Error reading acknowledgment")
                return None
            return ack
        except Exception as e:
            print("(pgl:_socket:readAck) ❌ Error reading acknowledgment:", e)
            return None
    def readCommandResults(self, ack=None):
        """
        Read the results from the socket after sending a command.
        
        Returns:
            np.ndarray: The results read from the socket, or None if an error occurs.
        """
        if not self.s:
            print("(pgl:_socket:readCommandResults) ❌ Not connected to socket")
            return None
        
        try:
            commandResults = {}
            # Read ack if not passed in
            if ack is None:
                commandResults['ack'] = self.read(np.double)
            else:
                commandResults['ack'] = ack
            # Read the rest of the command results
            commandResults['commandCode'] = self.read(np.uint16)
            commandResults['success'] = self.read(np.uint32)
            commandResults['processedTime'] = self.read(np.double)
            commandResults['vertexStart'] = self.read(np.double)
            commandResults['vertexEnd'] = self.read(np.double)
            commandResults['fragmentStart'] = self.read(np.double)
            commandResults['fragmentEnd'] = self.read(np.double)
            commandResults['drawableAcquired'] = self.read(np.double)
            commandResults['drawablePresented'] = self.read(np.double)
            return(commandResults)
        
        except Exception as e:
            print("(pgl:_socket:readCommandResults) ❌ Error reading results:", e)
            return None



    def parseCommandTypes(self, filename="mglCommandTypes.h"):
        """
        Parse the command types from the mglCommandTypes.h file.

        This function reads the mglCommandTypes.h file, extracts command names and their values,
        and returns a dictionary mapping command names to their corresponding values.

        """
        self.commandTypes = {}

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
                        value = 0xFFFF
                    else:
                        value = int(valueStr)
                    self.commandTypes[name] = value
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
            print("(pgl:_socket) ❌ Error finding PID:", e)
        return None
    
   
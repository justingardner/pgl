################################################################
# socket.py: manages the unix socket which is used to 
#            communicate with the standalone mgl engine for
#            pgl psychophysics and experiment library
################################################################
import socket, sys, time, struct
import numpy as np

class _socket:
    # Init Function
    def __init__(self,socketPath,timeout=10):
        # initialize start time
        startTime = time.time()
        attempt = 0

        # display what we are doing
        sys.stdout.write("(pgl:_socket) ")
        sys.stdout.flush()
        while True:
            try:
                self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.s.connect(socketPath)
                print("✅ Connected!")
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

    def write(self, message):
        # check socket
        if not self.s:
            print("(pgl:_socket) ❌ Not connected to socket")
            return
        # Pack data for sending
        if isinstance(message, np.uint16):
            packed = struct.pack('@H', message)
        # single float value
        elif isinstance(message, float):
            # Pack as 4-byte double float
            packed = struct.pack('@f', message)
        # array of floats
        elif isinstance(message, np.ndarray) and np.issubdtype(message.dtype, np.floating):
            # pack as 4 byte floats
            packed = message.astype(np.float32).tobytes()
        else:
            raise TypeError("Unsupported data type")


        try:
            self.s.sendall(packed)
            print("(pgl:_socket) ✅ Message sent:", message)
        except Exception as e:
            print("(pgl:_socket) ❌ Error sending message:", e)

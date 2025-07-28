################################################################
#   filename: pglTimestamp.py
#    purpose: Timestamps for getting precise system time
#         by: JLG
#       date: July 28, 2025
################################################################

##############
# import
##############
import time

# FIX, FIX, FIX: add mglGetSecs / mglWaitSecs codes

#################################################################
# pglTimestamp
#################################################################
class pglTimestamp:
    '''
    A class to handle timestamps for precise system time retrieval.
    '''
    def getSecs(self):
        '''
        Get the current system time in seconds.
        
        Returns:
            float: Current system time in seconds.
        '''
        return time.time()
    
    def waitSecs(self, secs):
        '''
        Wait for a specified number of seconds.
        
        Args:
            secs (float): The number of seconds to wait.
        
        Returns:
            None
        '''
        time.sleep(secs)

    def getDateAndTime(self):
        '''
        Get the current system date and time.
        
        Returns:
            str: Current system date and time as a formatted string.
        '''
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

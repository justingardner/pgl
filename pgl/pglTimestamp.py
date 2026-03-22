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
from . import _pglTimestamp

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
        return _pglTimestamp.getSecs()
    
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
    
    def formatDuration(self, seconds):
        """
        Convert a number of seconds into a human-readable string.
        """
        seconds = int(seconds)  # make sure it's an integer
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        parts = []
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} min{'s' if minutes != 1 else ''}")
        if secs or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

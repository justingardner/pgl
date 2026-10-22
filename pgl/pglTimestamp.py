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
import sys

#################################################################
# pglTimestamp
#################################################################
class pglTimestamp:
    '''
    A class to handle timestamps for precise system time retrieval.
    '''
    @staticmethod
    def getSecs():
        """Select the clock on first use; subsequent calls use it directly."""
        if sys.platform == "darwin":
            from . import _pglTimestamp
            clock = _pglTimestamp.getSecs
        else:
            clock = time.perf_counter

        pglTimestamp.getSecs = staticmethod(clock)
        return clock()
    
    @staticmethod
    def waitSecs(secs):
        '''
        Wait for a specified number of seconds.
        
        Args:
            secs (float): The number of seconds to wait.
        
        Returns:
            None
        '''
        time.sleep(secs)

    @staticmethod
    def getDateAndTime():
        '''
        Get the current system date and time.
        
        Returns:
            str: Current system date and time as a formatted string.
        '''
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    @staticmethod
    def formatDuration(seconds):
        """
        Convert a number of seconds into a human-readable string.
        """
        ms = int(seconds * 1000)  # convert to milliseconds for precision
        hours = ms // 3600000
        ms %= 3600000
        minutes = ms // 60000
        ms %= 60000
        secs = ms // 1000
        ms %= 1000

        parts = []
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} min{'s' if minutes != 1 else ''}")
        if secs:
            parts.append(f"{secs}s")
        # only print ms, if nothing else printed
        if not parts:
            parts.append(f"{ms}ms")

        return " ".join(parts)
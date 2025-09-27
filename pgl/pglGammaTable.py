################################################################
#   filename: pglGammaTable.py
#    purpose: Implements gamma table functions which are wrappers
#             to cocoa code
#         by: JLG
#       date: September 25, 2025
################################################################

#############
# Import modules
#############
from . import _pglGammaTable

#############
# Main class
#############
class pglGammaTable:
    '''
    pglGammaTable class wrapper class for getting and setting
    gamma table on macOS using cocoa calls in _pglGammaTable
    '''
    def getGammaTable(self, whichScreen = None):
        '''
        Get the current gamma table for a given screen.

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays. If ommitted, defaults to the screen on which
                pgl is open and running or, if not running, the primary display.

        Returns:
            tuple: A tuple of three numpy arrays (red, green, blue) each containing the gamma table values.
        '''
        return _pglGammaTable.getGammaTable(whichScreen)

    def setGammaTable(self, whichScreen, red, green, blue):
        '''
        Set the gamma table for a given screen.

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays. If set to None, defaults to the screen on which
                pgl is open and running or, if not running, the primary display.
            red (numpy.ndarray): Array containing the red gamma values.
            green (numpy.ndarray): Array containing the green gamma values.
            blue (numpy.ndarray): Array containing the blue gamma values.
        '''
        whichScreen = self.validateWhichScreen(whichScreen)
        if (whichScreen is None): return
        
        return _pglGammaTable.setGammaTable(whichScreen, red, green, blue)

    def getGammaTableSize(self, whichScreen = None):
        '''
        Get the size of the gamma table for a given screen.

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays. If ommitted, defaults to the screen on which
                pgl is open and running or, if not running, the primary display.

        Returns:
            int: The size of the gamma table.
        '''
        # validate whichScreen
        whichScreen = self.validateWhichScreen(whichScreen)
        if (whichScreen is None): return -1
        
        # call objective-c function
        return _pglGammaTable.getGammaTableSize(whichScreen)

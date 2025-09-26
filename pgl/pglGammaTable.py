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

    def setGammaTable(self, red, green, blue, whichScreen = None):
        '''
        Set the gamma table for a given screen.

        Args:
            red (numpy.ndarray): Array containing the red gamma values.
            green (numpy.ndarray): Array containing the green gamma values.
            blue (numpy.ndarray): Array containing the blue gamma values.
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays. If ommitted, defaults to the screen on which
                pgl is open and running or, if not running, the primary display.
        '''
        return _pglGammaTable.setGammaTable(red, green, blue, whichScreen)

    def getGammaTableBitDepth(self, whichScreen = None):
        '''
        Get the bit depth of the gamma table for a given screen.

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays. If ommitted, defaults to the screen on which
                pgl is open and running or, if not running, the primary display.

        Returns:
            int: The bit depth of the gamma table.
        '''
        return _pglGammaTable.getGammaTableBitDepth(whichScreen)

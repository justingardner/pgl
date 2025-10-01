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


#############
# Main class
#############
class pglGammaTable:
    def __init__(self):
        # check for _pglGammaTable
        try:
            from . import _pglGammaTable
            self._pglGammaTable = _pglGammaTable
        except ImportError as e:
            self._pglGammaTable = None
            print("(pglGammaTable) ❌ Could not import _pglGammaTable: You may need to compile by going to pgl in terminal and running 'make force'")
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
        if self._pglGammaTable is None:
            print("(pglGammaTable) ❌ _pglGammaTable not available, cannot getGammaTable.")
            return (None, None, None)
        else:
            return self._pglGammaTable.getGammaTable(whichScreen)

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

        if self._pglGammaTable is None:
            print("(pglGammaTable) ❌ _pglGammaTable not available, cannot setGammaTable.")
            return False
        else:
            return self._pglGammaTable.setGammaTable(whichScreen, red, green, blue)

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
        if self._pglGammaTable is None:
            print("(pglGammaTable) ❌ _pglGammaTable not available, cannot getGammaTableSize.")
            return -1
        else:
            return self._pglGammaTable.getGammaTableSize(whichScreen)

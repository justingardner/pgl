################################################################
#   filename: pglResolution.py
#    purpose: Implements resolution functions which are wrappers
#             to cocoa code
#         by: JLG
#       date: July 9, 2025
################################################################

#############
# Import modules
#############
from . import _displayInfo

#############
# Main class
#############
class pglResolution:
    """
    Initialize the pglResolution class.

    This class provides methods to get and set display resolutions
    and other display-related information using the underlying
    `_displayInfo` compiled extension.

    Args:
        None

    Returns:
        None
    """

    ################################################################
    # Get the display resolution
    ################################################################
    def getResolution(self, whichScreen):
        """
        Get the resolution and display settings for a given screen.

        This function retrieves the width, height, refresh rate, and bit depth of the specified
        display using the underlying `_displayInfo` compiled extension. 

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays.

        Returns:
            tuple[int, int, int, int]: A 4-tuple containing:
                - width (int): Screen width in pixels.
                - height (int): Screen height in pixels.
                - refresh_rate (int): Refresh rate in Hz.
                - bit_depth (int): Color depth in bits per pixel.

        Raises:
            None. Errors are signaled by a return value of (-1, -1, -1, -1)

        Verbose Mode:
            Module-level 'verbose' (pgl.verbose) can be set to display:
                - 1 Screen resolution, refresh rate and bit depth
                - 2 all available modes for the display
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:getResolution) Getting resolution for screen {whichScreen}")

        # Call the C function to get the display info
        return _displayInfo.getResolution(whichScreen)
    
    ################################################################
    # Set the display resolution
    ################################################################
    def setResolution(self, whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
        """
        Set the resolution and display settings for a given screen.

        This function sets the width, height, refresh rate, and bit depth of the specified
        display using the underlying `_displayInfo` compiled extension. 

        Args:
            whichScreen (int): Index of the display to query (0 = primary). Must be >= 0 and less
                than the number of active displays.
            screenWidth (int): Desired screen width in pixels.
            screenHeight (int): Desired screen height in pixels.
            screenRefreshRate (int): Desired refresh rate in Hz.
            screenColorDepth (int): Desired color depth in bits per pixel (e.g., 32 for 32-bit color).

        Returns:
            None: The function does not return a value, but it will print the new resolution if successful.

        Author:
            JLG

        Date:
            July 9, 2025
        """
        # Print what we are doing
        if self.verbose > 1: print(f"(pgl:setResolution) Setting resolution for screen {whichScreen} to {screenWidth}x{screenHeight}, refresh rate {screenRefreshRate}Hz, color depth {screenColorDepth}-bit")

        # Call the C function to set the display info
        if _displayInfo.setResolution(whichScreen, screenWidth, screenHeight, screenRefreshRate, screenColorDepth):
            # print what resolution the display was set to
            self.getResolution(whichScreen)
    
    ################################################################
    # Get the number of displays and the default display
    ################################################################
    def getNumDisplaysAndDefault(self):
        """
        Get the number of displays and the default display index.

        This function retrieves the total number of active displays and identifies the
        default display (usually the primary display) using the underlying `_displayInfo`
        compiled extension.

        Args:
            None

        Returns:
            tuple[int, int]: A 2-tuple containing:
                - numDisplays (int): The total number of active displays.
                - defaultDisplay (int): The index of the default display (0 = primary).

        Author:
            JLG

        Date:
            July 9, 2025
        """
        # Print what we are doing
        if self.verbose > 1: print("(pgl:getNumDisplaysAndDefault) Getting number of displays and default display")

        # Call the C function to get the number of displays and the default display
        return _displayInfo.getNumDisplaysAndDefault()


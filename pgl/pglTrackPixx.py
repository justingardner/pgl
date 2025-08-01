################################################################
#   filename: pglTrackPixx.py
#    purpose: Code for working with TrackPixx3 eye tracker
#         by: JLG
#       date: August 1, 2025
################################################################

#############å
# Import modules
#############
from pgl import pglEyeTracker
from pgl import pglDevice
import numpy as np

###################################
# TrackPixx3 device
###################################
class pglTrackPixx3(pglEyeTracker):
    """
    Class for interfacing with the TrackPixx3 eye tracker.
    Inherits from pglEyeTracker.
    """
    def __init__(self, deviceType="TrackPixx3"):
        """
        Initialize the TrackPixx3 device.
        """
        # call superclass constructor
        super().__init__(deviceType)
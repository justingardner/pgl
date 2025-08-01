from .pglBase import pglBase
from .pglResolution import pglResolution
from .pglDraw import pglDraw
from .pglTransform import pglTransform
from .pglProfile import pglProfile
from .pglBatch import pglBatch
from .pglImage import pglImage
from .pglStimuli import pglStimuli
from .pglTimestamp import pglTimestamp
from .pglDevice import pglDevice
from .pglEvent import pglEvent
from .pglVPixx import pglProPixx, pglDataPixx
from .pglEyeTracker import pglEyeTracker
from .pglTrackPixx import pglTrackPixx3


#from .screen import screen
#from .task import task

class pgl(pglBase, pglResolution, pglDraw, pglTransform, pglProfile, pglBatch, pglImage, pglStimuli, pglTimestamp):
    """
    purpose: psychophysics and experiment library for Python.
    License: MIT License — see LICENSE file for details.
         by: JLG
       date: July 9, 2025
    """
    pass

__version__ = "1.0.0"
__author__ = "JLG"
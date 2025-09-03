from .pglBase import pglBase
from .pglResolution import pglResolution
from .pglDraw import pglDraw
from .pglTransform import pglTransform
from .pglProfile import pglProfile
from .pglBatch import pglBatch
from .pglImage import pglImage
from .pglStimuli import pglStimuli
from .pglTimestamp import pglTimestamp
from .pglDevice import pglDevice, pglDevices, pglKeyboard, pglEventKeyboard
from .pglEvent import pglEvent, pglEvents
from .pglCommandReplayer import pglCommandReplayer
from .pglFrameGrab import pglFrameGrab
from .pglExperiment import pglExperiment, pglTask 

# Device specific imports (eye trackers, etc.)
from .pglVPixx import pglProPixx, pglDataPixx
from .pglEyeTracker import pglEyeTracker
from .pglTrackPixx import pglTrackPixx3
from .pglEyelink import pglEyelinkCustomDisplay, pglEyelink

class pgl(pglBase, pglResolution, pglDraw, pglTransform, pglProfile, pglBatch, pglImage, pglStimuli, pglTimestamp, pglDevices, pglEvents, pglCommandReplayer, pglFrameGrab):
    """
    purpose: psychophysics and experiment library for Python.
    License: MIT License — see LICENSE file for details.
         by: JLG
       date: July 9, 2025
    """
    def __init__(self, *args, **kwargs):
      # Explicitly initialize each parent class
      pglBase.__init__(self, *args, **kwargs)
      pglResolution.__init__(self, *args, **kwargs)
      pglDraw.__init__(self, *args, **kwargs)
      pglTransform.__init__(self, *args, **kwargs)
      pglProfile.__init__(self, *args, **kwargs)
      pglBatch.__init__(self, *args, **kwargs)
      pglImage.__init__(self, *args, **kwargs)
      pglStimuli.__init__(self, *args, **kwargs)
      pglTimestamp.__init__(self, *args, **kwargs)
      pglDevices.__init__(self, *args, **kwargs)
      pglEvents.__init__(self, *args, **kwargs)
      pglCommandReplayer.__init__(self, *args, **kwargs)
      pglFrameGrab.__init__(self, *args, **kwargs)

__version__ = "1.0.0"
__author__ = "JLG"
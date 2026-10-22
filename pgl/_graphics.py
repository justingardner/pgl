"""Composite graphics controller; imported only when graphics are requested."""

from .pglBase import pglBase
from .pglResolution import pglResolution
from .pglDraw import pglDraw
from .pglTransform import pglTransform
from .pglProfile import pglProfile
from .pglBatch import pglBatch
from .pglImage import pglImage
from .pglStimuli import pglStimuli
from .pglTimestamp import pglTimestamp
from .pglDevice import pglDevices
from .pglEvent import pglEvents
from .pglCommandReplayer import pglCommandReplayer
from .pglFrameGrab import pglFrameGrab
from .pglGammaTable import pglGammaTable
from .pglSettings import pglSettingsManager
from .pglDialog import pglDialogs
from .pglMessages import pglMessages


class pgl(pglBase, pglResolution, pglDraw, pglTransform, pglProfile, pglBatch, pglImage, pglStimuli, pglTimestamp, pglDevices, pglEvents, pglCommandReplayer, pglFrameGrab, pglGammaTable, pglSettingsManager, pglMessages, pglDialogs):
    """
    purpose: psychophysics and experiment library for Python.
    License: MIT License — see LICENSE file for details.
         by: JLG
       date: July 9, 2025
    """
    def __init__(self, *args, **kwargs):
      # Explicitly initialize each parent class
      pglGammaTable.__init__(self, *args, **kwargs)
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
      pglSettingsManager.__init__(self, *args, **kwargs)
      pglDialogs.__init__(self, *args, **kwargs)

# Preserve the existing public class identity for pickle and introspection.
pgl.__module__ = "pgl"

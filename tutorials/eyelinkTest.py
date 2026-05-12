# library imports
from matplotlib import pyplot as plt

# import PGL Eyelink interface
from pgl import pglEyelink

# Load PGL libraries and start a PGL window
from pgl import pgl, pglExperiment
pgl = pgl()
e = pglExperiment(pgl, settingsName="ViewPixx")

# close any existing windows
pgl.cleanUp()

# start screen
e.initScreen()

#init eye tracker
eyelink = pglEyelink(pgl)

# set custom calibration points
eyelink.setCustomCalibrationPoints(margin=0.7, numPoints=9)

#calibrate
eyelink.calibrate()

# close
e.endScreen()


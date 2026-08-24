################################################################
#   filename: pglDigitalBrain
#    purpose: Code for digital brain
#         by: JLG
#       date: Aug 23, 2026
################################################################

# import
from .pglSettings import pglTraitSettings
from traitlets import Unicode, Int, List, Tuple

# class for choosing parameters for each block
class pglChooseBlock(pglTraitSettings):
    subjectID = Unicode(default_value="s000", help="Subject ID")
    descriptionLength = Int(12, help="description length in seconds", min=0)
    displayWidth = Int(30, help="Size of movie width in deg", min=0)
    nCalibrationPoints = List(default_value=[5, 9, 13, 17], help="number of calibration points")
    calibrationSize = Tuple(Int, Int, default_value=[15,15],labels=("width","height"), help="Width and height of area to calibrate")

    def __init__(self, subjectMax=5, dayMax=2, blockMax=5, **kwargs):
        super().__init__(**kwargs)

        self.add_traits(
            subjectNum=Int(
                min=0,
                max=subjectMax,
                default_value=0,
                help="Subject number for memory pilot"
            ),
            dayNum=Int(
                min=0,
                max=dayMax,
                default_value=0,
                help="Day of experiment"
            ),
            blockNum=Int(
                min=0,
                max=blockMax,
                default_value=0,
                help="Which block"
            ),
        )

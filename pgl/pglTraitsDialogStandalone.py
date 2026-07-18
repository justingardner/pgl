################################################################
#   filename: pglTraitsDialogStandalone.py
#    purpose: Standalone process which just runs _pglTraitsDialog
#             outside of the jupyter notebook to avoid crashy-conflicty behavior
#         by: JLG
#       date: Jul 18, 2026
################################################################

################################################################
# inport
################################################################
import sys
from PySide6.QtWidgets import QApplication, QDialog
from pgl.pglDialog import _pglTraitsDialog
from pgl.pglSerialize import pglSerialize

################################################################
# main function
################################################################
def main():
    # takes as input the serialized settings filename
    # and the desired output - these will be tmp
    # files and facilitate the transfer of settings
    # to and from this function from jupyter
    inFile = sys.argv[1]      # settings to edit
    outFile = sys.argv[2]     # where to write result (only if OK)

    # load the settings
    settings = pglSerialize.load(inFile)

    # setup dialog
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = _pglTraitsDialog(settings)

    # start dialog, and handle returns
    if dlg.exec() == QDialog.Accepted:
        dlg.settings.save(outFile)   # user hit OK -> write result
        sys.exit(0)
    else:
        sys.exit(1)                  # user hit Cancel -> no output file

if __name__ == "__main__":
    main()
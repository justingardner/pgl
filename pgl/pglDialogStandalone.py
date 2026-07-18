################################################################
#   filename: pglEditSettings.py
#    purpose: Standalone process to edit pglSettings via PySide6
################################################################
import sys
from PySide6.QtWidgets import QApplication, QDialog
from pgl.pglDialog import pglTraitsDialog
from pgl.pglSettings import pglSettings

def main():
    inFile = sys.argv[1]      # settings to edit
    outFile = sys.argv[2]     # where to write result (only if OK)

    settings = pglSettings(filename=inFile)

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = pglTraitsDialog(settings)

    if dlg.exec() == QDialog.Accepted:
        dlg.settings.save(outFile)   # user hit OK -> write result
        sys.exit(0)
    else:
        sys.exit(1)                  # user hit Cancel -> no output file

if __name__ == "__main__":
    main()
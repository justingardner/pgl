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
    requestedWidth = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if requestedWidth is not None and requestedWidth <= 0:
        raise ValueError("Dialog width must be positive.")

    # load the settings
    settings = pglSerialize.load(inFile)

    # setup dialog
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = _pglTraitsDialog(settings)

    # Apply the optional width after the normal interface has been built.
    # Preserve the existing height and keep the window within the screen.
    if requestedWidth is not None:
        screen = dlg.screen() or app.primaryScreen()

        if screen is not None:
            available = screen.availableGeometry()
            frameWidth = max(0, dlg.frameGeometry().width() - dlg.width())
            maxWidth = max(1, available.width() - frameWidth - 40)
            dlg.setMinimumWidth(min(dlg.minimumWidth(), maxWidth))
            targetWidth = min(max(requestedWidth, dlg.minimumWidth()), maxWidth)
            dlg.resize(targetWidth, dlg.height())

            dialogRect = dlg.frameGeometry()
            dialogRect.moveCenter(available.center())
            dlg.move(dialogRect.topLeft())
        else:
            dlg.resize(max(requestedWidth, dlg.minimumWidth()), dlg.height())

    # start dialog, and handle returns
    if dlg.exec() == QDialog.Accepted:
        dlg.settings.save(outFile)   # user hit OK -> write result
        sys.exit(0)
    else:
        sys.exit(1)                  # user hit Cancel -> no output file

if __name__ == "__main__":
    main()
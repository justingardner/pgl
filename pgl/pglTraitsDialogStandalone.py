################################################################
#   filename: pglTraitsDialogStandalone.py
#    purpose: Standalone process which just runs _pglTraitsDialog
#             outside of the jupyter notebook to avoid crashy-conflicty behavior
#         by: JLG
#       date: Jul 18, 2026
################################################################
import sys

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from pgl.pglDialog import _pglTraitsDialog, pglToolDialog
from pgl.pglSerialize import pglSerialize


def main():
    """Open either a serialized settings dialog or a portable tool editor."""
    import json
    from pathlib import Path

    toolMode = len(sys.argv) > 1 and sys.argv[1] == "--tool-schema"
    arguments = sys.argv[2:] if toolMode else sys.argv[1:]

    if len(arguments) not in (2, 3):
        raise ValueError("Expected input filename, output filename, and optional width.")

    inFile, outFile = arguments[:2]
    requestedWidth = int(arguments[2]) if len(arguments) == 3 else None
    if requestedWidth is not None and requestedWidth <= 0:
        raise ValueError("Dialog width must be positive.")

    document = None

    if toolMode:
        

        document = pglToolDialog.validate(
            json.loads(Path(inFile).read_text(encoding="utf-8"))
        )
        settings = pglToolDialog.build(document)
        title = f"{document['label']} — installation settings"
    else:
        settings = pglSerialize.load(inFile)
        title = "Settings"

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = _pglTraitsDialog(settings, title=title)

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

    while True:
        if dlg.exec() != QDialog.Accepted:
            return 1

        if not toolMode:
            dlg.settings.save(outFile)
            return 0

        try:
            # Duplicate names or invalid defaults must not be silently saved.
            edited = pglToolDialog.extract(document, dlg.settings)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(dlg, "Invalid tool configuration", str(error))
            continue

        Path(outFile).write_text(
            json.dumps(edited, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        historyPath = Path(outFile).parent / "validation-history.json"
        with historyPath.open("x", encoding="utf-8") as target:
            json.dump(
                {
                    "schemaVersion": 1,
                    "records": getattr(dlg.settings, "_toolValidationRecords", []),
                },
                target, ensure_ascii=False, indent=2, allow_nan=False,
            )
        return 0


if __name__ == "__main__":
    try:
        exitCode = main()
    except Exception:
        import traceback

        traceback.print_exc()
        exitCode = 2

    sys.exit(exitCode)
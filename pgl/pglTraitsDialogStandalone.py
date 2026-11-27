################################################################
#   filename: pglTraitsDialogStandalone.py
#    purpose: Standalone process which just runs _pglTraitsDialog
#             outside of the jupyter notebook to avoid crashy-conflicty behavior
#         by: JLG
#       date: Jul 18, 2026
################################################################
import base64
import json
import os
from pathlib import Path
import sys

from PySide6.QtCore import QByteArray, QBuffer, QEvent, QIODevice, QPoint, QPointF, QRect, QTimer, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from pgl.pglDialog import _pglTraitsDialog, pglToolDialog
from pgl.pglSerialize import pglSerialize


################################################################
# Mirror the live Qt windows and dispatch desktop input on the SSH host
################################################################
class _TraitsDisplayBridge:
    # ========================================================================
    def __init__(self, app, dialog, directory):
        from PySide6.QtTest import QTest

        self.app = app
        self.dialog = dialog
        self.directory = directory
        self.qtTest = QTest
        self.ack = 0
        self.previousFrame = None
        self.previousKey = None
        self.clipboardText = None
        self.failed = False
        self.app.clipboard().dataChanged.connect(self._recordClipboard)
        self.timer = QTimer(dialog)
        self.timer.timeout.connect(self._poll)
        self.timer.start(65)

    # ========================================================================
    def _recordClipboard(self):
        self.clipboardText = self.app.clipboard().text()[:65536]

    # ========================================================================
    def _poll(self):
        try:
            self._queueInput()
            self._capture()
        except Exception:
            self._fail()

    # ========================================================================
    def _fail(self):
        import traceback

        self.failed = True
        self.timer.stop()
        traceback.print_exc()
        self.app.exit(2)

    # ========================================================================
    def _queueInput(self):
        try:
            with (self.directory / "input.json").open("rb") as source:
                content = source.read(1024 * 1024 + 1)
        except FileNotFoundError:
            return
        if len(content) > 1024 * 1024: raise ValueError("Desktop input is too large.")
        for index, event in json.loads(content):
            if index <= self.ack: continue
            if index != self.ack + 1: raise ValueError("Invalid desktop input sequence.")
            self.ack = index
            # Input can open a nested modal dialog. Keep the capture timer free
            # to service the new window while that callback is waiting.
            QTimer.singleShot(0, lambda event=event: self._dispatch(event))

    # ========================================================================
    def _dispatch(self, event):
        try:
            kind, *args = event
            if kind in ("keyPress", "keyRelease"):
                key, modifiers, text, repeat, clipboard = args
                if clipboard is not None: self.app.clipboard().setText(clipboard)
                focus = self.app.focusWidget() or self.app.activeModalWidget() or self.dialog
                eventType = QEvent.Type.KeyPress if kind == "keyPress" else QEvent.Type.KeyRelease
                self.app.sendEvent(focus, QKeyEvent(eventType, key, Qt.KeyboardModifier(modifiers), text, repeat))
                return

            x, y, button, buttons, modifiers, dx, dy = args
            point = QPoint(x, y)
            hit = self.app.widgetAt(point)
            window = self.app.activePopupWidget() or self.app.activeModalWidget() or (hit.window() if hit is not None else self.dialog)
            handle = window.windowHandle()
            if handle is None: return
            local = window.mapFromGlobal(point)
            modifiers = Qt.KeyboardModifier(modifiers)
            if kind == "wheel":
                target = self.app.widgetAt(point)
                if target is None or target.window() is not window: target = window
                wheel = QWheelEvent(
                    QPointF(target.mapFromGlobal(point)), QPointF(point), QPoint(), QPoint(dx, dy),
                    Qt.MouseButton(buttons), modifiers, Qt.ScrollPhase.NoScrollPhase, False,
                )
                self.app.sendEvent(target, wheel)
            elif kind == "move":
                self.qtTest.mouseMove(handle, local)
            else:
                mouse = {"press": self.qtTest.mousePress, "release": self.qtTest.mouseRelease, "doubleClick": self.qtTest.mouseDClick}[kind]
                mouse(handle, Qt.MouseButton(button), modifiers, local)
        except Exception:
            self._fail()

    # ========================================================================
    def _capture(self):
        screen = self.dialog.screen()
        bounds = QRect()
        for widget in self.app.topLevelWidgets():
            if widget.isVisible(): bounds = bounds.united(QRect(widget.mapToGlobal(QPoint()), widget.size()))
        bounds = bounds.intersected(screen.geometry())
        if bounds.isEmpty(): return
        frame = screen.grabWindow(0, bounds.x(), bounds.y(), bounds.width(), bounds.height()).toImage()
        if frame.isNull(): raise RuntimeError("Qt could not capture the settings dialog.")
        key = (bounds.x(), bounds.y(), self.ack, self.clipboardText)
        if key == self.previousKey and frame == self.previousFrame: return
        self.previousKey, self.previousFrame = key, frame

        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not frame.save(buffer, "PNG"): raise RuntimeError("Qt could not save the settings dialog frame.")
        pending = self.directory / "frame.tmp"
        pending.write_text(json.dumps({
            "origin": [bounds.x(), bounds.y()], "ack": self.ack,
            "image": base64.b64encode(data.data()).decode("ascii"), "clipboard": self.clipboardText,
        }), encoding="utf-8")
        pending.replace(self.directory / "frame.json")


# ========================================================================
def main():
    """Open either a serialized settings dialog or a portable tool editor."""

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

    bridgeDirectory = os.environ.get("PGL_DIALOG_FRAME_DIRECTORY")
    bridge = _TraitsDisplayBridge(app, dlg, Path(bridgeDirectory)) if bridgeDirectory else None

    while True:
        result = dlg.exec()
        if bridge is not None and bridge.failed:
            return 2
        if result != QDialog.Accepted:
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

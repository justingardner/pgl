################################################################
#   filename: pglTraitsDialog.py
#    purpose: PySide6 dialog for editing pglSettings traits
#         by: JLG
#       date: Jul 17, 2026
################################################################

#############
# Import
#############
import copy
from unicodedata import name
import uuid
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QSlider, QPushButton, QWidget, QScrollArea, QDialogButtonBox, QAbstractSpinBox,
    QStylePainter, QStyleOptionComboBox, QStyle, QMessageBox, QSizePolicy, QListView,
    QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QCoreApplication, QTimer, Signal
from PySide6.QtGui import QColor, QStandardItemModel
from PySide6.QtCore import QPoint, QPropertyAnimation, QEasingCurve

from traitlets import (
    HasTraits, Float, Int, List, Unicode, Bool, Tuple, TraitType, Enum
)

from .pglSerialize import pglSerialize
import sys, subprocess, tempfile
from pathlib import Path
from IPython.display import HTML, display
from collections import OrderedDict
import ipywidgets as widgets
from traitlets import HasTraits, Float, Int, List, TraitError, Unicode, Dict, default, link, Bool, TraitType, Instance
from functools import partial
import math
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from .pglMessages import pglMessages
import traceback
import re
from copy import deepcopy

#####################################################################
# Wrapping layout for compact trait controls
#####################################################################
from PySide6.QtWidgets import QLayout
from PySide6.QtCore import QRect, QSize


class _pglTraitFlowLayout(QLayout):
    """Wrap compact trait controls with centered alignment and accurate height."""

    def __init__(self, parent=None, spacing=12):
        super().__init__(parent)
        self._items = []

        # Small vertical allowance prevents controls touching adjacent rows.
        self.setContentsMargins(0, 3, 0, 3)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)
        self.invalidate()

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        if not 0 <= index < len(self._items): return None
        item = self._items.pop(index)
        self.invalidate()
        return item

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), measureOnly=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, measureOnly=False)

    def _itemMetrics(self, item):
        """Return the allocated control size and font-aware separation."""
        widget = item.widget()
        size = item.sizeHint().expandedTo(item.minimumSize())

        characterGap = (
            max(1, widget.fontMetrics().horizontalAdvance("  "))
            if widget is not None else 8
        )
        gap = max(0, self.spacing(), characterGap)

        if isinstance(widget, QCheckBox):
            # Reserve explicit space after the checkbox description.
            labelPadding = max(1, widget.fontMetrics().horizontalAdvance(" "))
            size = QSize(size.width() + labelPadding, size.height())

        return size, gap

    def minimumSize(self):
        """
        Permit wrapping while reserving enough width for an indivisible item.

        The inline text/controller pair remains one indivisible item.
        """
        size = QSize(0, 0)

        for item in self._items:
            if item.isEmpty(): continue
            itemSize, _ = self._itemMetrics(item)
            size = size.expandedTo(itemSize)

        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )

    def sizeHint(self):
        """
        Report at least the natural height of the tallest control.

        Width remains compact; heightForWidth supplies the wrapped height
        once the form assigns the actual available width.
        """
        return self.minimumSize()

    def _arrange(self, rect, measureOnly):
        """
        Measure and place exactly the same wrapped lines.

        Controls on each line are vertically centered relative to the tallest
        item, including an inline text/checkbox container.
        """
        margins = self.contentsMargins()
        availableWidth = max(
            0, rect.width() - margins.left() - margins.right()
        )

        lines = []
        line = []
        lineWidth = 0
        lineHeight = 0
        lineGap = 0
        previousGap = 0

        for item in self._items:
            if item.isEmpty(): continue

            size, gap = self._itemMetrics(item)
            beforeGap = max(previousGap, gap) if line else 0
            proposedWidth = lineWidth + beforeGap + size.width()

            if line and proposedWidth > availableWidth:
                lines.append((line, lineHeight, lineGap))
                line = []
                lineWidth = 0
                lineHeight = 0
                lineGap = 0
                previousGap = 0
                beforeGap = 0

            offsetX = lineWidth + beforeGap
            line.append((item, size, offsetX))
            lineWidth = offsetX + size.width()
            lineHeight = max(lineHeight, size.height())
            lineGap = max(lineGap, gap)
            previousGap = gap

        if line:
            lines.append((line, lineHeight, lineGap))

        y = rect.y() + margins.top()
        previousLineGap = 0

        for lineIndex, (line, lineHeight, lineGap) in enumerate(lines):
            if lineIndex:
                y += max(previousLineGap, lineGap)

            if not measureOnly:
                for item, size, offsetX in line:
                    # Center short checkboxes beside the taller inline pair.
                    itemY = y + (lineHeight - size.height()) // 2
                    item.setGeometry(QRect(
                        rect.x() + margins.left() + offsetX,
                        itemY,
                        size.width(),
                        size.height(),
                    ))

            y += lineHeight
            previousLineGap = lineGap

        return y - rect.y() + margins.bottom()


#####################################################################
# Preserve per-trait visibility after several controls share a form row
#####################################################################
class _pglCompactTraitGroup:
    """Manage individual controls without hiding their neighboring traits."""

    def __init__(self, formLayout, container, entries, pair=None):
        self.formLayout = formLayout
        self.container = container
        self.entries = entries
        self.pair = pair
        self.visibility = {
            id(entry["widget"]): bool(entry["trait"].metadata.get("visible", True))
            for entry in entries
        }

    def setVisible(self, widget, visible):
        self.visibility[id(widget)] = bool(visible)
        self.refresh()

    def refresh(self):
        for entry in self.entries:
            widget = entry["widget"]
            widget.setVisible(self.visibility[id(widget)])

        if self.pair is not None:
            pairWidget, pairEntries = self.pair
            pairWidget.setVisible(any(
                self.visibility[id(entry["widget"])]
                for entry in pairEntries
            ))

        self.formLayout.setRowVisible(
            self.container, any(self.visibility.values())
        )
        self.container.layout().invalidate()
        self.container.updateGeometry()


class _pglCompactRowLayout:
    """
    Adapt existing setRowVisible(widget, visible) calls for grouped controls.

    Other layout operations continue to use the original QFormLayout.
    """

    def __init__(self, formLayout, group):
        self.formLayout = formLayout
        self.group = group

    def setRowVisible(self, widget, visible):
        if id(widget) in self.group.visibility:
            self.group.setVisible(widget, visible)
        else:
            self.formLayout.setRowVisible(widget, visible)

    def __getattr__(self, name):
        return getattr(self.formLayout, name)

#######################################
# _pglTraitsDialog
# Actual code for the pglTraitsDialog, but this 
# gets run by pglTraitsDialogStandalone so that it avoids
# crashy-conflicty behavior with jupyter notebooks
#######################################
class _pglTraitsDialog(QDialog):
    """
    Puts up a PySide6 dialog to edit the traits of a settings class.

    Usage:
        newSettings = pglTraitsDialog(settings).run()
        if newSettings is not None:
            # user hit OK
        else:
            # user hit Cancel

    The settings passed in are copied. The copy has a field _dialog set to
    this dialog. When traits on the copy change (either from the dialog or
    programmatically), the settings can check if _dialog is not None and
    call the small trait API exposed here:

        _dialog.enable(traitName, isEnabled)
        _dialog.visible(traitName, isVisible)
        _dialog.set(traitName, value)
    """
    def __init__(self, settings, parent=None, title="Settings"):
        super().__init__(parent)

        try:
            # Original settings remain untouched until the caller accepts edits.
            self.settings = copy.deepcopy(settings)
            self.settings._dialog = self

            self.traitWidgets = {}
            self._contingentFields = []
            self._contingentWarnings = set()
            self._compactGroups = []

            self._updatingWidget = False
            self.accepted_ = False

            self.setWindowTitle(title)
            self.setStyleSheet(self._darkStyle())
            self._selectedSettings = {}

            self._buildUI()
            self._compactTraitRows()
            self._refreshContingencies()

            # Scope this change to consolidated checkbox descriptions.
            # Indicator styling continues to come from the dialog stylesheet.
            for group in self._compactGroups:
                for entry in group.entries:
                    checkbox = entry["widget"]
                    if isinstance(checkbox, QCheckBox):
                        checkbox.setStyleSheet(
                            "QCheckBox { color: #000000; }"
                            "QCheckBox:disabled { color: #6b7078; }"
                        )

            # First fit immediately, then repeat once Qt has resolved widget
            # geometry and the standalone launcher has applied any width.
            self._fitDialogToContents()
            QTimer.singleShot(0, self._fitDialogToContents)

        except Exception as error:
            pglMessages.warning(f"Error creating traitsDialog: {error}")
            raise



    #########################################
    # Public entry point
    #########################################
    def run(self):
        """
        Show the dialog modally. Returns the (copied) settings with any
        edits if the user hit OK, or None if they hit Cancel.
        """
        # make sure a QApplication exists
        app = QApplication.instance()
        ownApp = False
        if app is None:
            app = QApplication([])
            ownApp = True

        result = self.exec()

        if result == QDialog.Accepted:
            return self.settings
        return None

    #########################################
    # Small trait API for the settings object
    #########################################
    def enable(self, traitName, isEnabled=True, settingsKey=None):
        """Enable one control without disabling its grouped neighbors."""
        key = ("root",) if settingsKey is None else settingsKey
        entry = self.traitWidgets.get(key, {}).get(traitName)
        if entry is None: return
        entry["row"].setEnabled(bool(isEnabled))

    def visible(self, traitName, isVisible=True, settingsKey=None):
        """Show/hide one control, respecting compact shared rows."""
        key = ("root",) if settingsKey is None else settingsKey
        entry = self.traitWidgets.get(key, {}).get(traitName)
        if entry is None: return
        entry["layout"].setRowVisible(entry["widget"], bool(isVisible))

    def set(self, traitName, value, settingsKey=None):
        """Update a displayed control without committing another model edit."""
        key = ("root",) if settingsKey is None else settingsKey
        entry = self.traitWidgets.get(key, {}).get(traitName)
        if entry is None: return

        previousUpdating = self._updatingWidget
        self._updatingWidget = True
        try:
            entry["setter"](value)
        finally:
            self._updatingWidget = previousUpdating
            
    def _onPlotWindowClosed(self):
        """Reset plot controls when the separate window is closed."""

        self.plotCanvas.setVisible(False)
        self.plotButtonState = False
        self.settingsListPlotButtonState = False
        self.multiSelectPlotButtonState = False

        for state in self._selectedSettings.values():
            if "plotVisible" in state:
                state["plotVisible"] = False

        button = self._activePlotButton
        if button is not None:
            wasBlocked = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(wasBlocked)

        self._activePlotButton = None
    #########################################
    # UI construction
    #########################################
    def _buildUI(self):
        formWidget = QWidget()
        self.formLayout = QFormLayout(formWidget)

        # give the form real breathing room
        self.formLayout.setContentsMargins(24, 16, 24, 16)
        self.formLayout.setHorizontalSpacing(20)
        self.formLayout.setVerticalSpacing(6)
        self.formLayout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QFormLayout.DontWrapRows)

        for traitName, trait in self._getOrderedTraits().items():
            if traitName.startswith('_') and not trait.metadata.get("property", None) and trait.metadata.get("visible") is not True:
                continue
            self._addTraitWidget(traitName, trait)

        # Shared matplotlib figure, displayed in a separate window.
        self.plotWindow = _PlotWindow(self)
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.plotAxis = self.figure.add_subplot(111)
        self.plotCanvas = _WindowFigureCanvas(self.figure, self.plotWindow)

        plotLayout = QVBoxLayout(self.plotWindow)
        plotLayout.setContentsMargins(0, 0, 0, 0)
        plotLayout.addWidget(self.plotCanvas)

        self.plotCanvas.setVisible(False)
        self.finished.connect(lambda result: self._onPlotWindowClosed())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(formWidget)

        # Standard OK/Cancel buttons (right side)
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttonBox.accepted.connect(self._onOk)
        buttonBox.rejected.connect(self._onCancel)

        # Left-side custom actions
        customButtonBox = QDialogButtonBox()

        # create extra buttons that have callbacks
        for label, callbackName in getattr(self.settings, "buttons", []):
            button = QPushButton(label)
            callback = getattr(self.settings, callbackName)
            def makeWrappedCallback(callback):
                def wrapped():
                    callback()
                    # raise the gui back up after we have run the callback
                    QTimer.singleShot(750, self.raiseAndActivate)
                return wrapped

            button.clicked.connect(makeWrappedCallback(callback))
            customButtonBox.addButton(button, QDialogButtonBox.ActionRole)
            
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.setSpacing(0)
        mainLayout.addWidget(scroll)

        buttonBar = QWidget()
        buttonBar.setObjectName("buttonBar")

        bl = QHBoxLayout(buttonBar)
        bl.setContentsMargins(24, 8, 24, 8)

        # Left side
        bl.addWidget(customButtonBox)

        # Push OK/Cancel to the right
        bl.addStretch(1)

        # Right side
        bl.addWidget(buttonBox)

        mainLayout.addWidget(buttonBar)

        # Set overall dimensions
        self.setMinimumWidth(560)
        maxDialogHeight = 900

        # Height needed for the form contents
        formHeight = formWidget.sizeHint().height()

        # Add the button bar and layout margins
        buttonHeight = buttonBar.sizeHint().height()
        extra = mainLayout.contentsMargins().top() + mainLayout.contentsMargins().bottom()

        desiredHeight = formHeight + buttonHeight + extra

        self.resize(720, min(desiredHeight, maxDialogHeight))
        
        # give hint for window to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()

        # Position half a dialog-width to the right of its normal centered position.
        screenRect = self.screen().availableGeometry()
        dialogRect = self.frameGeometry()

        #x = screenRect.left() + screenRect.width() // 2
        x = screenRect.left() + (screenRect.width() - dialogRect.width()) // 2
        y = screenRect.top() + (screenRect.height() - dialogRect.height()) // 2

        # Keep the dialog within the available screen area.
        x = max(screenRect.left(), min(x, screenRect.right() - dialogRect.width() + 1))
        y = max(screenRect.top(), min(y, screenRect.bottom() - dialogRect.height() + 1))

        self.move(x, y)        
    def _getOrderedTraits(self, obj=None):
        """Return traits in class definition order (like getOrderedTraits)."""
        if obj is None:
            obj = self.settings
        from collections import OrderedDict
        ordered = OrderedDict()
        # walk the MRO so subclass traits keep their definition order
        for cls in reversed(type(obj).__mro__):
            for name, o in cls.__dict__.items():
                if isinstance(o, TraitType):
                    ordered[name] = o
        return ordered

    def _helpText(self, traitName, trait):
        return getattr(trait, 'help', "") or ""

    #########################################
    # Widget factory per trait type
    #########################################
    def _addTraitWidget(self, traitName, trait, settingsObject=None, layout=None, settingsKey=None):
        
        # settingsObject is the object that owns this trait.
        # Defaults to the root dialog settings object.
        if settingsObject is None:
            settingsObject = self.settings
        helpText = self._helpText(traitName, trait)
        
        # get the property to read (usualy just the traitName, but if property)
        # is set there is another property that is used for the trait (like for lazy-loaded properties)
        readName = trait.metadata.get("property", traitName)
        current = getattr(settingsObject, readName)

        # a tuple
        if isinstance(trait, Tuple):
            self._addTuple(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # a multi-select list (checkbox rows)
        elif isinstance(trait, List) and "settingsListKey" in trait.metadata and trait.metadata.get("multiSelect", False):
            if not current:
                # if empty list just move on
                return
            else:
                if trait.metadata.get("style") == "dropdown":
                    self._addMultiSelectDropdown(
                        traitName, trait, current, helpText,
                        settingsObject, layout, settingsKey
                    )
                else:
                    self._addMultiSelectList(
                        traitName, trait, current, helpText,
                        settingsObject, layout, settingsKey
                    )
                
        # a settings list
        elif isinstance(trait, List) and "settingsListKey" in trait.metadata:
            if not current:
                # if empty list just move on
                return
            else:
                self._addSettingsList(traitName, trait, current, helpText, settingsObject, layout, settingsKey)        
        
        # Float with min and max -> slider + spinbox
        elif isinstance(trait, Float) and trait.min is not None and not math.isinf(trait.max) and not math.isinf(trait.min):
            self._addFloatRange(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # Float (min only or unbounded)
        elif isinstance(trait, Float):
            self._addFloat(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # Int
        elif isinstance(trait, Int):
            self._addInt(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # Bool
        elif isinstance(trait, Bool):
            self._addBool(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # RGB list
        elif isinstance(trait, List) and trait.metadata.get("isRGB", False):
            self._addRGB(traitName, trait, current, helpText, settingsObject, layout, settingsKey)
            
        # Unicode with button, including path fields with an action
        elif isinstance(trait, Unicode) and trait.metadata.get("hasSetButton", False):
            self._addTextWithSetButton(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # Unicode / Path
        elif isinstance(trait, Unicode):
            self._addText(traitName, trait, current, helpText, settingsObject, layout, settingsKey)
            
        # List with a plot button
        elif isinstance(trait, List) and trait.metadata.get("hasPlotButton", False):
            self._addListWithPlotButton(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # List -> dropdown
        elif isinstance(trait, List):
            self._addList(traitName, trait, current, helpText, settingsObject, layout, settingsKey)        

        # Enum
        elif isinstance(trait, Enum):
            self._addEnum(traitName, trait, current, helpText, settingsObject, layout, settingsKey)    
            
        # Register optional Boolean dependencies after the widget exists.
        if trait.metadata.get("contingentOn"):
            key = settingsKey if settingsKey is not None else ("root",)
            entry = self.traitWidgets.get(key, {}).get(traitName)

            # Some composite controls register their outer row at root.
            if entry is None:
                entry = self.traitWidgets.get(("root",), {}).get(traitName)

            if entry is not None:
                self._contingentFields.append({
                    "owner": settingsObject,
                    "traitName": traitName,
                    "widget": entry["row"],
                })
                self._refreshContingencies()    
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Multi-select list -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addMultiSelectList(self, traitName, trait, current, helpText,
                             settingsObject, layout=None, settingsKey=None):
        # get metadata settings
        keyTraitName = trait.metadata["settingsListKey"]
        hideKey = trait.metadata.get("hideKey", False)
        hideAll = trait.metadata.get("hideAll", False)
        maxRowsVisible = trait.metadata.get("maxRowsVisible", 5)
        plotButtonFunction = trait.metadata.get("buttonFunction", None)
        hasPlotButton = trait.metadata.get("hasPlotButton", None)

        if layout is None:
            layout = self.formLayout

        objectName = settingsObject.__class__.__name__ if not isinstance(settingsObject, _RetargetableProxy) else settingsObject.getClassName()
        settingsKey = (objectName, traitName)
        self.traitWidgets.setdefault(settingsKey, {})

        state = {"list": current, "focused": current[0], "key": settingsKey, "rows": {}, "plotVisible": False}
        self._selectedSettings[settingsKey] = state

        # Build the detail rows for whichever object currently has focus
        #--------------------
        def buildRows():
            for name, childTrait in self._getOrderedTraits(current[0]).items():
                if name.startswith("_") and not childTrait.metadata.get("property", None) and childTrait.metadata.get("visible") is not True:
                    continue
                if hideKey and name == keyTraitName:
                    continue
                if hideAll:
                    continue
                self._addTraitWidget(name, childTrait, proxy, layout, settingsKey)
                childNames.append(name)
                childTraits[name] = childTrait
        # Update the detail widgets from one object
        #------------------------------
        def updateFields(obj):
            self._updatingWidget = True
            try:
                for name in childNames:
                    # read what the underlying property is (usually the trait, but
                    # could be a property for lazy-loading)
                    childTrait = childTraits.get(name)
                    readName = name
                    if childTrait is not None: 
                        readName = childTrait.metadata.get("property", name)
                    value = getattr(obj, readName)
                    entry = self.traitWidgets[state["key"]].get(name)

                    if entry is not None and "setter" in entry:
                        entry["setter"](value)

                    if entry is not None:
                        visible = obj.trait_metadata(name, "visible", True)
                        entry["layout"].setRowVisible(entry["widget"], bool(visible))

                    nestedKey = (builtClassName, name)
                    nested = self._selectedSettings.get(nestedKey)
                    if nested is not None:
                        nested["retargetList"](value)
                        visible = obj.trait_metadata(name, "visible", True)
                        nested["setVisible"](bool(visible))
            except Exception as e:
                print(f"Error updating fields for {obj}: {e}")
                
            finally:
                self._updatingWidget = False

                # Proxies now point at the newly displayed settings objects.
                self._refreshContingencies()

        # Give a row the highlighted / non-highlighted look
        #------------------------------
        def styleRow(row, focused):
            row.setProperty("focused", focused)
            row.style().unpolish(row)
            row.style().polish(row)
            row.update()

        def updatePlot():
            obj = state["focused"]

            try:
                buttonFunction = getattr(obj, plotButtonFunction, None) if plotButtonFunction else None
                if not callable(buttonFunction):
                    pglMessages.warning(f"{obj.name} does not have function: {plotButtonFunction}")
                    state["plotVisible"] = False
                    self.plotCanvas.setVisible(False)
                    return

                self.figure.clear()
                buttonFunction(self.figure)
                self.plotCanvas.setVisible(True)
                self.plotCanvas.draw_idle()
                state["plotVisible"] = True

            except Exception as e:
                state["plotVisible"] = False
                self.plotCanvas.setVisible(False)
                pglMessages.warning(f"Error calling plotButton function {plotButtonFunction}: {e}")
        
        # Set which object's details are shown below the scroll area
        #------------------------------
        def setFocus(obj):
            prev = state["focused"]
            prevEntry = state["rows"].get(id(prev))
            if prevEntry is not None:
                styleRow(prevEntry["row"], False)

            state["focused"] = obj
            entry = state["rows"].get(id(obj))
            if entry is not None:
                styleRow(entry["row"], True)

            proxy.retarget(obj)
            updateFields(obj)

            if hasPlotButton and state["plotVisible"] and hasattr(self, "plotCanvas"):
                updatePlot()
                
        # Keep a checkbox synced if isSelected changes from elsewhere
        # (e.g. select all / select none, or programmatic changes)
        #------------------------------
        def makeSelectedObserver(obj, checkbox):
            def onSelectedChanged(change):
                if self._updatingWidget:
                    return
                checkbox.blockSignals(True)
                checkbox.setChecked(change["new"])
                checkbox.blockSignals(False)
            return onSelectedChanged

        # Build one row widget for an object
        #------------------------------
        def buildRow(obj):
            row = _ClickableRow()
            row.setObjectName("multiSelectRow")
            row.setProperty("focused", False)

            h = QHBoxLayout(row)
            h.setContentsMargins(6, 4, 6, 4)

            label = QLabel(str(getattr(obj, keyTraitName)))
            #label.setObjectName("rowInternalLabel")
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            h.addWidget(label, 1)

            checkbox = QCheckBox()
            checkbox.setChecked(bool(getattr(obj, "isSelected", False)))
            h.addWidget(checkbox)

            def onCheckboxToggled(checked, obj=obj):
                self._updatingWidget = True
                try:
                    obj.isSelected = checked
                finally:
                    self._updatingWidget = False
            checkbox.toggled.connect(onCheckboxToggled)

            observer = makeSelectedObserver(obj, checkbox)
            obj.observe(observer, names="isSelected")

            row.clicked.connect(lambda obj=obj: setFocus(obj))

            state["rows"][id(obj)] = {"row": row, "checkbox": checkbox,
                                       "label": label, "observer": observer,
                                       "object": obj}
            return row

        # (Re)build the full stack of rows for the given list
        #------------------------------
        def rebuildRows(objList):
            for key, entry in list(state["rows"].items()):
                entry["object"].unobserve(entry["observer"], names="isSelected")
                entry["row"].setParent(None)
                entry["row"].deleteLater()
            state["rows"] = {}

            for obj in objList:
                addRowToLayout(buildRow(obj))
            applyScrollHeight()
        # Retarget this multi-select list to another backing list
        # (used when nested inside a settings list that changes)
        #----------------------------
        def retargetList(newList):
            state["list"] = newList
            rebuildRows(newList)
            setFocus(newList[0])

        # Show/hide everything, respecting nested visibility too
        #------------------------------
        def setVisible(visible):
            layout.setRowVisible(container, visible)

            obj = state["focused"]
            for name in childNames:
                childEntry = self.traitWidgets[settingsKey].get(name)
                if childEntry is not None:
                    ownVisible = obj.trait_metadata(name, "visible", True)
                    childEntry["layout"].setRowVisible(childEntry["widget"], visible and bool(ownVisible))

                nestedKey = (builtClassName, name)
                nested = self._selectedSettings.get(nestedKey)
                if nested is not None:
                    nested["setVisible"](visible)

        state["setVisible"] = setVisible
        state["retargetList"] = retargetList

        # --- build the widget tree ---
        container = QWidget()
        containerLayout = QVBoxLayout(container)
        containerLayout.setContentsMargins(10, 10, 10, 10)

        buttonRow = QWidget()
        buttonLayout = QHBoxLayout(buttonRow)
        buttonLayout.setContentsMargins(0, 0, 0, 0)


        def onSelectAll():
            # Match exact names using the list's display-key trait.
            ignoreNames = trait.metadata.get("selectAllIgnoreNames", [])
            if not isinstance(ignoreNames, (list, tuple)) or not all(isinstance(name, str) for name in ignoreNames):
                raise ValueError(f"{traitName}: selectAllIgnoreNames must be a list or tuple of strings.")
            ignoreNames = set(ignoreNames)

            # Use the current list after retargeting and clear ignored selections.
            for obj in state["list"]:
                obj.isSelected = str(getattr(obj, keyTraitName)) not in ignoreNames        
        selectAllButton = QPushButton("select all")
        selectAllButton.clicked.connect(onSelectAll)

        selectNoneButton = QPushButton("select none")
        def onSelectNone():
            for obj in state["list"]:
                obj.isSelected = False
        selectNoneButton.clicked.connect(onSelectNone)
        
        if hasPlotButton:
            plotButton = QPushButton(trait.metadata.get("buttonLabel", "display"))

            def onPlotButton():
                if state["plotVisible"]:
                    state["plotVisible"] = False
                    self.plotCanvas.setVisible(False)
                else:
                    updatePlot()

            buttonLayout.addWidget(plotButton)
            plotButton.clicked.connect(onPlotButton)

        buttonLayout.addWidget(selectNoneButton)
        buttonLayout.addWidget(selectAllButton)
        containerLayout.addWidget(buttonRow)

        scrollArea = QScrollArea()
        scrollArea.setObjectName("multiSelect")
        scrollArea.setWidgetResizable(True)

        rowsContainer = QWidget()
        rowsLayout = QVBoxLayout(rowsContainer)
        rowsLayout.setContentsMargins(0, 0, 0, 0)
        rowsLayout.setSpacing(2)
        rowsLayout.addStretch()  # keep rows packed at top as list grows/shrinks

        def addRowToLayout(row):
            rowsLayout.insertWidget(rowsLayout.count() - 1, row)

        # Size the scroll area to show ~maxRowsVisible rows, measured
        # from an actual built row rather than a guessed constant
        #------------------------------
        def applyScrollHeight():
            if not state["rows"]:
                return

            sampleRow = next(iter(state["rows"].values()))["row"]

            rowHeight = sampleRow.sizeHint().height()
            spacing = rowsLayout.spacing()

            margins = rowsLayout.contentsMargins()
            marginHeight = margins.top() + margins.bottom()

            total = (
                maxRowsVisible * rowHeight
                + (maxRowsVisible - 1) * spacing
                + marginHeight
            )

            scrollArea.setFixedHeight(total + 8)

        scrollArea.setWidget(rowsContainer)
        containerLayout.addWidget(scrollArea)

        self._register(traitName, trait, container, lambda v: None, layout)

        # _register() assumes single-row widgets (combo boxes, spin boxes) and
        # pins vertical size policy to Fixed with a 30px minimum. This container
        # is a multi-row scroll area + button row, so give it room to size to
        # its real content instead of being squeezed to the registration-time
        # sizeHint (which was computed before any rows existed).
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        container.setMinimumHeight(0)

        # set what the detail widgets are updating
        proxy = _RetargetableProxy(current[0])
        childNames = []
        childTraits = {}
        builtClassName = current[0].__class__.__name__

        if not childNames:
            buildRows()

        for obj in current:
            addRowToLayout(buildRow(obj))
        applyScrollHeight()

        setFocus(current[0])

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Setting list -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addSettingsList(self, traitName, trait, current, helpText,
                        settingsObject, layout=None, settingsKey=None):
        # get metadata settings
        keyTraitName = trait.metadata["settingsListKey"]
        hideKey = trait.metadata.get("hideKey", False)
        hideAll = trait.metadata.get("hideAll", False)
        highlightSelector = trait.metadata.get("highlightSelector", True)
        buttons = trait.metadata.get("buttons", False)
        setDefault = trait.metadata.get("setDefault", False)
        hasPlotButton = trait.metadata.get("hasPlotButton", False)
        plotButtonFunction = trait.metadata.get("buttonFunction", None)
        self.settingsListPlotButtonState = False 

        # Build widgets once
        #--------------------
        def buildRows():
            for name, childTrait in self._getOrderedTraits(current[0]).items():
                if name.startswith("_") and not childTrait.metadata.get("property", None) and childTrait.metadata.get("visible") is not True:
                    continue

                if hideKey and name == keyTraitName:
                    continue
                
                if hideAll:
                    continue

                self._addTraitWidget(name, childTrait, proxy, layout, settingsKey)
                childNames.append(name)
                childTraits[name]=childTrait

        # Commit current selection
        # this function will reorder the list so that the selection is at top
        #--------------------------
        def commitSelection(updateCombo=True):
            lst = state["list"]
            obj = state["object"]

            if len(lst) <= 1:
                return

            # make a sort key which sorts by numbers if they are in the name
            def naturalSortKey(s):
                return [int(chunk) if chunk.isdigit() else chunk.lower()
                for chunk in re.split(r'(\d+)', s)]

            # sort the remaining values
            remaining = [x for x in lst if x is not obj]
            remaining.sort(
                key=lambda x: naturalSortKey(str(getattr(x, keyTraitName)))
            )

            # make list with selected item at top with 
            # the remaining items alphabetically sorted afterwards
            lst[:] = [obj] + remaining

            # update selector order in combo
            if updateCombo:
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(
                    [str(getattr(x, keyTraitName)) for x in lst]
                )
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
            
        # Retarget this settings list to another backing list
        # used for recursive calls such that if we change a setting list
        # and there is a subfield that is also a settings list,
        # that settings list will get retargeted to the appropriate
        # new settings from the parent settings list
        #----------------------------
        def retargetList(newList):

            # Save current selection before leaving
            commitSelection()
            state["list"] = newList
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(
                [str(getattr(x, keyTraitName)) for x in newList]
            )
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
            state["object"] = newList[0]
            proxy.retarget(state["object"])
            updateFields(state["object"])

        # Update widgets from one object
        #------------------------------
        def updateFields(obj):
            self._updatingWidget = True

            try:
                for name in childNames:
                    # read what the underlying property is (usually the trait, but
                    # could be a property for lazy-loading)
                    childTrait = childTraits.get(name)
                    readName = name
                    if childTrait is not None:
                        readName = childTrait.metadata.get("property", name)
                    value = getattr(obj, readName)
                    entry = self.traitWidgets[state["key"]].get(name)

                    # ordinary widget
                    if entry is not None and "setter" in entry:
                        entry["setter"](value)

                    # dynamic visibility based on obj's trait metadata
                    if entry is not None:
                        visible = obj.trait_metadata(name, "visible", True)
                        entry["layout"].setRowVisible(entry["widget"], bool(visible))
                    
                    # nested settings list
                    nestedKey = (builtClassName,name)
                    nested = self._selectedSettings.get(nestedKey)

                    if nested is not None:
                        nested["retargetList"](value)
                        visible = obj.trait_metadata(name, "visible", True)
                        nested["setVisible"](bool(visible))
            except Exception as e:
                print(f"Error updating fields for {obj}: {e}")

            finally:
                self._updatingWidget = False

                # Proxies now point at the newly displayed settings objects.
                self._refreshContingencies()

        # set visible turns on off the settings list and all its children and buttons
        #------------------------------
        def setVisible(visible):
            # hide/show the combo row itself
            layout.setRowVisible(combo, visible)

            # hide/show the button row
            if buttons:
                newButton.setVisible(visible)
                copyButton.setVisible(visible)
                deleteButton.setVisible(visible)

            # hide/show all currently-built child rows, respecting their own metadata too
            obj = state["object"]
            for name in childNames:
                childEntry = self.traitWidgets[settingsKey].get(name)
                if childEntry is not None:
                    ownVisible = obj.trait_metadata(name, "visible", True)
                    childEntry["layout"].setRowVisible(childEntry["widget"], visible and bool(ownVisible))

                # recurse into any further-nested settings lists
                nestedKey = (builtClassName, name)
                nested = self._selectedSettings.get(nestedKey)
                if nested is not None:
                    nested["setVisible"](visible)

        # User changed combo selection
        #------------------------------
        def showSelection(index):

            # first commit selection order
            state["object"] = state["list"][index]
            commitSelection()
            obj = state["list"][0]
            state["object"] = obj
            
            # retarget and update all the fields
            proxy.retarget(obj)
            updateFields(obj)
            # refresh plot if it's currently shown 
            if hasPlotButton and self.settingsListPlotButtonState:
                try:
                    buttonFunction = getattr(obj, plotButtonFunction, None)
                    if buttonFunction is not None:
                        self.figure.clear()
                        buttonFunction(self.figure)
                        self.plotCanvas.draw()
                except Exception as e:
                    pglMessages.warning(f"Error refreshing plot: {e}")
            #elif hasattr(self, "plotCanvas"):
            #    self.plotCanvas.setVisible(False)
            #    self.plotCanvas.draw()
                
            if setDefault:                
                # update the default checkbox state
                check.blockSignals(True)
                check.setChecked(obj.isDefault)
                check.blockSignals(False)
                     
        # Updates the selection combo when the keyTrait changes
        #------------------------------
        def keyChanged(change):
            if self._updatingWidget:
                return

            combo.setItemText(
                state["list"].index(change["owner"]),
                str(change["new"])
            )
        if layout is None:
            layout = self.formLayout

        # keep the state, with a key which is used by updateFields to
        # find settings list object which need to be recursed on
        # key is the class name and the traitname. Note that
        # we use _RetargetableProxy helper class because recursed
        # settings list will need to have what they are updating retargeted
        # if there is a parent change.
        objectName = settingsObject.__class__.__name__ if not isinstance(settingsObject, _RetargetableProxy) else settingsObject.getClassName()
        settingsKey = (objectName, traitName)
        self.traitWidgets.setdefault(settingsKey, {})
        state = {"list": current, "object": current[0], "key":settingsKey}
        self._selectedSettings[settingsKey] = state

        state["setVisible"] = setVisible

        # make sure the order of the list is selected on top
        # and the rest alphabetical
        commitSelection(updateCombo=False)
        
        # make the combo box which selects the list
        combo = CenteredComboBox()
        if highlightSelector: combo.setObjectName("settingsSelector")
        combo.addItems([str(getattr(x, keyTraitName)) for x in current])

         # add plot button
        if hasPlotButton:
            plotButton = QPushButton(trait.metadata.get("buttonLabel", "display"))

            def onPlotButton():
                try:
                    # act on the currently-selected object
                    obj = state["object"]
                    buttonFunction = getattr(obj, plotButtonFunction, None)
                    if buttonFunction is None:
                        print(f"{obj} does not have function: {plotButtonFunction}")
                        return
                    if not self.settingsListPlotButtonState:
                        self.plotCanvas.setVisible(True)
                        self.figure.clear()
                        buttonFunction(self.figure)
                        self.plotCanvas.draw()
                        self.settingsListPlotButtonState = True
                    else:
                        self.figure.clear()
                        self.plotCanvas.setVisible(False)
                        self.plotCanvas.draw()
                        self.settingsListPlotButtonState = False
                except Exception as e:
                    pglMessages.warning(f"Error calling plotButton function {plotButtonFunction}: {e}")

            plotButton.clicked.connect(onPlotButton)
            # combo + plot button in one row
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(combo, 1)
            h.addWidget(plotButton)
            self._register(traitName, trait, row, lambda v: None, layout)

        # add defaults checkbox if requested
        elif setDefault:
            # add a checkbox for setting defaults
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(combo, 1)
            
            label = QLabel("default:")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setObjectName("rowInternalLabel")
            h.addWidget(label)
            
            check = QCheckBox()
            check.setChecked(True) # fix, temp
            h.addWidget(check)

            def onSelectDefault(checked):
                check.blockSignals(True)
                if checked:
                    # set the current object as the default
                    for obj in current:
                        if obj is state["object"]:
                            obj.isDefault = True
                        else:
                            obj.isDefault = False
                else:
                    # if the unser unchecks the default, we will revert it back to checked, because there must always be a default
                    check.setChecked(True)
                check.blockSignals(False)
                
            # check whether there is only one default, and if so, check the box
            nSelected = sum(1 for obj in current if getattr(obj, "isDefault", False))
            if nSelected == 0:
                pglMessages.message("No default selected in settings list. Setting the first item as default.")
                # no selections, so set the current one to be selected
                state["object"].isDefault = True
            elif nSelected > 1:
                pglMessages.message("Multiple defaults selected in settings list. Setting the first item as default.")
                for obj in current:
                    if obj is state["object"]:
                        obj.isDefault = True
                    else:
                        obj.isDefault = False
            check.stateChanged.connect(onSelectDefault)     
                
            self._register(traitName, trait, row, lambda v: None, layout)
        else:
            # register the combo
            self._register(traitName, trait, combo, lambda v: None, layout)
        
        # if we have buttons add them here
        if buttons:
            # create the buttons
            # make a new settings
            newButton = QPushButton("new")
            def onNew():
                # make a new object of the same type as the first in the list
                newObj = current[0].__class__()
                # create a new UUID for it
                newObj.uuid = str(uuid.uuid4())
                current.append(newObj)
                newObj.observe(keyChanged, names=keyTraitName)
                retargetList(current)
                # show the new object in the list
                showSelection(current.index(newObj))
                updateDeleteButtonState()
            newButton.clicked.connect(onNew)
            
            # make a copy of current settings
            copyButton = QPushButton("copy")
            def onCopy():
                # make a copy of the current settings
                newObj = copy.deepcopy(current[combo.currentIndex()])
                # create a new UUID for it
                newObj.uuid = str(uuid.uuid4())
                # rename it
                newObjName = getattr(newObj, keyTraitName)
                newObjName += " copy"
                setattr(newObj, keyTraitName, newObjName)
                # Only one can be isDefault, so fix that if it is set
                if newObj.isDefault: newObj.isDefault = False
                # link it to changes in the combo
                newObj.observe(keyChanged, names=keyTraitName)
                # add it to the list
                current.append(newObj)
                retargetList(current)
                # show the new object in the list
                showSelection(current.index(newObj))
                updateDeleteButtonState()
            copyButton.clicked.connect(onCopy)
            
            # delete the current settings
            def updateDeleteButtonState():
                # delete button is only enabled if there is more than one item in the list
                deleteButton.setEnabled(len(current) > 1)
            deleteButton = QPushButton("delete")
            def onDelete():
                # do not allow deleting the last item in the list
                if len(current) <= 1: return
                # confirm delete
                itemName = getattr(current[combo.currentIndex()], keyTraitName)
                reply = QMessageBox.question(
                    self,
                    "Confirm Delete",
                    f"Are you sure you want to delete '{itemName}'?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No  # default button
                )
                if reply != QMessageBox.Yes:
                    return
                # if this is the default display, we will need to reassign
                wasDefault = getattr(current[combo.currentIndex()], "isDefault")
                # pop off the current item and retarget the list
                current.pop(combo.currentIndex())
                if current:
                    state["object"] = current[0]
                    # set the top object to default if we just deleted the last default
                    if wasDefault: current[0].isDefault = True
                retargetList(current)
                showSelection(0)
                updateDeleteButtonState()

            deleteButton.clicked.connect(onDelete)
            self.addButtonRow(None, [newButton, copyButton, deleteButton], displayName="")
                # if we have a plot button, add it
        
        # set what the widget is updating
        proxy = _RetargetableProxy(current[0])
        childNames = []
        childTraits={}
        
        state["retargetList"] = retargetList

        # get the build class
        builtClassName = current[0].__class__.__name__
        
        # build the widgets
        if not childNames: buildRows()
        
        # observe any changes to keyTraitName, so that we can update the combo
        for obj in current:
            obj.observe(keyChanged, names=keyTraitName)

        # connect showObject to changes in the index
        combo.currentIndexChanged.connect(showSelection)

        # and show the first item in the list
        showSelection(0)
        
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- add multi select dropdown -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addMultiSelectDropdown(self, traitName, trait, current, helpText,
                            settingsObject, layout=None, settingsKey=None):
        if layout is None:
            layout = self.formLayout

        keyTraitName = trait.metadata["settingsListKey"]

        combo = CheckableComboBox()
        combo.setToolTip(helpText)

        # These are the objects currently represented by the dropdown,
        # not necessarily the objects present when it was first built.
        state = {
            "objects": [],
            "updating": False,
        }

        def setter(newList):
            """Retarget the dropdown and load each object's selection state."""
            wasUpdating = state["updating"]
            signalsBlocked = combo.blockSignals(True)
            state["updating"] = True

            try:
                # Copy the container, but retain the actual settings objects.
                state["objects"] = list(newList) if newList is not None else []

                combo.clear()

                for obj in state["objects"]:
                    combo.addItem(
                        str(getattr(obj, keyTraitName)),
                        checked=bool(getattr(obj, "isSelected", False)),
                    )
            finally:
                state["updating"] = wasUpdating
                combo.blockSignals(signalsBlocked)
                combo.update()

        def onSelectionChanged(_):
            if self._updatingWidget or state["updating"]:
                return

            # Snapshot before writing: trait observers may refresh widgets
            # synchronously while these assignments are taking place.
            changes = [
                (obj, combo.isItemChecked(row))
                for row, obj in enumerate(state["objects"])
            ]

            state["updating"] = True
            try:
                for obj, checked in changes:
                    if bool(getattr(obj, "isSelected", False)) != checked:
                        self._commit(obj, "isSelected", checked)
            finally:
                state["updating"] = False

        setter(current)
        combo.selectionChanged.connect(onSelectionChanged)

        self._register(
            traitName,
            trait,
            combo,
            setter,
            layout,
            settingsKey,
        )
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Float with min/max -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addFloatRange(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        step = getattr(trait, 'step', (trait.max - trait.min) / 100.0)

        spin = QDoubleSpinBox()
        spin.setMinimum(trait.min)
        spin.setMaximum(trait.max)
        spin.setSingleStep(step)
        spin.setValue(float(current))
        spin.setToolTip(helpText)

        slider = QSlider(Qt.Horizontal)
        # slider works in integer steps -> scale
        scale = max(1, int(round((trait.max - trait.min) / step)))
        slider.setMinimum(0)
        slider.setMaximum(scale)
        slider.setToolTip(helpText)

        def toSlider(v):
            return int(round((v - trait.min) / (trait.max - trait.min) * scale))

        def fromSlider(v):
            return trait.min + (v / scale) * (trait.max - trait.min)

        slider.setValue(toSlider(float(current)))

        def onSpin(v):
            if self._updatingWidget:
                return
            self._updatingWidget = True
            slider.setValue(toSlider(v))
            self._updatingWidget = False
            self._commit(settingsObject, traitName, v)

        def onSlider(v):
            if self._updatingWidget:
                return
            fv = fromSlider(v)
            self._updatingWidget = True
            spin.setValue(fv)
            self._updatingWidget = False
            self._commit(settingsObject, traitName, fv)

        spin.valueChanged.connect(onSpin)
        slider.valueChanged.connect(onSlider)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)

        def setter(value):
            spin.setValue(float(value))
            slider.setValue(toSlider(float(value)))

        self._register(traitName, trait, row, setter, layout, settingsKey)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Float -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addFloat(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        spin = QDoubleSpinBox()
        spin.setAlignment(Qt.AlignCenter) 
        spin.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        spin.setDecimals(1)
        if trait.min is not None:
            spin.setMinimum(trait.min)
        else:
            spin.setMinimum(-1e12)
        spin.setMaximum(1e12)
        spin.setSingleStep(getattr(trait, 'step', 0.1) or 0.1)
        spin.setValue(float(current))
        spin.setToolTip(helpText)

        def onChange(v):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, v)

        spin.valueChanged.connect(onChange)
        
        # wrap with - and + buttons
        row = self._wrapSpinDoubleStep(spin, bigStep = 1.0)
        self._register(traitName, trait, row, lambda v: spin.setValue(float(v)), layout, settingsKey)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Int -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addInt(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        spin = QDoubleSpinBox()
        spin.setAlignment(Qt.AlignCenter) 
        spin.setDecimals(0)
        spin.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        spin.setMinimum(trait.min if trait.min is not None else -2**53)
        spin.setMaximum(trait.max if trait.max is not None else 2**53)
        spin.setSingleStep(getattr(trait, 'step', 1) or 1)
        spin.setValue(int(current))
        spin.setToolTip(helpText)

        def onChange(v):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, int(v))

        spin.valueChanged.connect(onChange)
        
        # wrap with - and + buttons
        row = self._wrapSpinSingleStep(spin)
        self._register(traitName, trait, row, lambda v: spin.setValue(int(v)), layout, settingsKey)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Tuple -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addTuple(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        spins = []

        # get tuple Labels if it exits
        tupleLabels = trait.metadata.get("labels", None)
        
        # get the elementTraits, so we can get the type of each element
        tupleTraits = getattr(trait, "_traits", None)
        
        for i, value in enumerate(current):
            if tupleLabels is not None:
                label = QLabel(f"{tupleLabels[i]}:")
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                label.setObjectName("rowInternalLabel")
                h.addWidget(label)
            
            # Get the trait for this tuple element
            elementTrait = tupleTraits[i] if tupleTraits else None           

            if isinstance(elementTrait, Int):
                spin = QSpinBox()
                # set min and max
                spin.setMinimum(elementTrait.min if elementTrait.min is not None else -2**31)
                spin.setMaximum(elementTrait.max if elementTrait.max is not None else 2**31 - 1)
                # set value
                spin.setValue(int(value))

            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(1)
                # set min and max
                spin.setMinimum(elementTrait.min if elementTrait.min is not None else -1e12)
                spin.setMaximum(elementTrait.max if elementTrait.max is not None else 1e12)
                # set value
                spin.setValue(float(value))

            spin.setAlignment(Qt.AlignCenter)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

            spin.setSingleStep(getattr(trait, "step", 1) or 1)
            spin.setToolTip(helpText)

            spins.append(spin)
            h.addWidget(spin)

        def onChange(_):
            if not self._updatingWidget:
                if tupleTraits:
                    values = tuple(
                        int(spin.value()) if isinstance(elementTrait, Int)
                        else float(spin.value())
                        for spin, elementTrait in zip(spins, tupleTraits)
                    )
                else:
                    values = tuple(spin.value() for spin in spins)

                self._commit(settingsObject, traitName, values)

        for spin in spins:
            spin.valueChanged.connect(onChange)

        def setValue(values):
            self._updatingWidget = True
            try:
                for spin, value in zip(spins, values):
                    spin.setValue(value)
            finally:
                self._updatingWidget = False

        self._register(traitName, trait, row, setValue, layout, settingsKey)
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Bool -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addBool(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        check = QCheckBox()
        check.setChecked(bool(current))
        check.setToolTip(helpText)

        def onChange(state):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, check.isChecked())

        check.stateChanged.connect(onChange)
        self._register(traitName, trait, check, lambda v: check.setChecked(bool(v)), layout, settingsKey)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Text / Path -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addText(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        """Add text, with the shared Browse control for isPath=True traits.

        Accepted paths use the ordinary text-change commit mechanism.
        Picker cancellation leaves the editor and settings copy unchanged.
        The enclosing dialog still controls whether edits are saved.
        """
        edit = QLineEdit(str(current) if current is not None else "")
        edit.setAlignment(Qt.AlignCenter)
        edit.setToolTip(helpText)

        def onChange(text):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, text)

        edit.textChanged.connect(onChange)

        def setter(value):
            # Retargeting a profile must not generate another edit/commit.
            wasBlocked = edit.blockSignals(True)
            try:
                edit.setText(str(value) if value is not None else "")
            finally:
                edit.blockSignals(wasBlocked)

        if not trait.metadata.get("isPath", False):
            self._register(traitName, trait, edit, setter, layout, settingsKey)
            return

        browse = self._makePathBrowseButton(edit, traitName, trait)

        row = QWidget()
        rowLayout = QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(6)
        rowLayout.addWidget(edit, 1)
        rowLayout.addWidget(browse)

        # Enabled/visible metadata applies to the entire row.
        self._register(traitName, trait, row, setter, layout, settingsKey)

    def _makePathBrowseButton(self, edit, traitName, trait):
        """Create a Browse button using existing path trait metadata.

        Shared by ordinary path fields and text fields with an action button.

        Metadata:
            pathType: directory, file, or saveFile.
            pathFilter: optional file-filter string.
            selectApplicationBundle: select macOS .app directories.

        The picker only sets editor text. Existing textChanged handlers decide
        how to commit the value. Cancellation leaves the text unchanged.
        """
        import os
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        pathType = trait.metadata.get("pathType", "directory")
        if pathType not in ("directory", "file", "saveFile"):
            raise ValueError(f"{traitName}: pathType must be directory, file, or saveFile.")

        pathFilter = trait.metadata.get("pathFilter", "All files (*)")
        if not isinstance(pathFilter, str):
            raise TypeError(f"{traitName}: pathFilter must be a string.")

        browse = QPushButton("…")
        browse.setToolTip("Browse for a directory" if pathType == "directory" else "Browse for a file")
        browse.setAutoDefault(False)
        browse.setDefault(False)

        # Override the dialog-wide button minimum width for a compact picker.
        browse.setStyleSheet(
            "QPushButton { min-width: 28px; max-width: 28px; padding: 2px 4px; }"
        )
        browse.setFixedWidth(40)

        fieldLabel = trait.metadata.get("traitDisplayName", traitName)

        def initialLocation():
            """Start at the entered local path or its nearest existing parent."""
            text = edit.text().strip()

            # Filesystem pickers cannot browse fsspec URLs. Users can still
            # enter those manually in the ordinary text editor.
            if not text or "://" in text or "::" in text or "\x00" in text:
                return str(Path.home())

            path = Path(os.path.expandvars(text)).expanduser().absolute()

            if pathType == "saveFile" and path.parent.is_dir():
                return str(path)
            if pathType == "file" and path.is_file():
                return str(path)
            if path.is_dir():
                return str(path)

            parent = path.parent
            while not parent.is_dir():
                if parent == parent.parent:
                    return str(Path.home())
                parent = parent.parent

            return str(parent)

        def onBrowse(checked=False):
            if self._updatingWidget:
                return

            try:
                start = initialLocation()

                if pathType == "directory":
                    import sys

                    options = QFileDialog.Option.ShowDirsOnly
                    if (
                        sys.platform == "darwin"
                        and trait.metadata.get("selectApplicationBundle", False)
                    ):
                        # A MATLAB .app is an installation directory. The
                        # non-native directory picker allows selecting the
                        # bundle itself rather than launching the application.
                        options |= QFileDialog.Option.DontUseNativeDialog

                    selected = QFileDialog.getExistingDirectory(
                        self,
                        f"Select {fieldLabel}",
                        start,
                        options=options,
                    )
                elif pathType == "file":
                    selected, _ = QFileDialog.getOpenFileName(
                        self,
                        f"Select {fieldLabel}",
                        start,
                        pathFilter,
                    )
                else:
                    selected, _ = QFileDialog.getSaveFileName(
                        self,
                        f"Select {fieldLabel}",
                        start,
                        pathFilter,
                    )

                # Empty means Cancel: do not set text or commit anything.
                if selected:
                    edit.setText(selected)

            except Exception as error:
                message = QMessageBox(self)
                message.setIcon(QMessageBox.Icon.Warning)
                message.setWindowTitle("Path selection failed")
                message.setTextFormat(Qt.TextFormat.PlainText)
                message.setText(str(error))
                message.exec()

        browse.clicked.connect(onBrowse)

        return browse

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Enum -> dropdown -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addEnum(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        combo = CenteredComboBox()

        enumValues = list(trait.values)   # fixed allowed values, preserve original types

        combo.addItems([str(o) for o in enumValues])

        if current in enumValues:
            combo.setCurrentIndex(enumValues.index(current))
        elif enumValues:
            combo.setCurrentIndex(0)

        combo.setToolTip(helpText)

        combo.setMinimumWidth(280)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.view().setMinimumWidth(combo.sizeHint().width())
        combo.setMaxVisibleItems(12)

        def onChange(index):
            if self._updatingWidget:
                return

            selected = enumValues[index]

            self._commit(settingsObject, traitName, selected)

        combo.currentIndexChanged.connect(onChange)

        def setter(value):
            combo.blockSignals(True)

            if value in enumValues:
                combo.setCurrentIndex(enumValues.index(value))

            combo.blockSignals(False)

        self._register(traitName, trait, combo, setter, layout, settingsKey)
        
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- List -> dropdown -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addList(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        combo = CenteredComboBox()

        options = list(current) if current else []
        comboValues = options.copy()   # preserve original types

        combo.addItems([str(o) for o in comboValues])

        if comboValues:
            combo.setCurrentIndex(0)

        combo.setToolTip(helpText)

        combo.setMinimumWidth(280)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        combo.view().setMinimumWidth(combo.sizeHint().width())
        combo.setMaxVisibleItems(12)

        def onChange(index):
            if self._updatingWidget:
                return

            selected = comboValues[index]

            # move selected to top while preserving types
            newList = [selected] + [x for x in comboValues if x != selected]

            self._commit(settingsObject, traitName, newList)

            # update local ordering
            comboValues[:] = newList

        combo.currentIndexChanged.connect(onChange)

        def setter(value):
            nonlocal comboValues

            combo.blockSignals(True)

            comboValues = list(value) if value else []

            combo.clear()
            combo.addItems([str(o) for o in comboValues])

            if comboValues:
                combo.setCurrentIndex(0)

            combo.blockSignals(False)

        self._register(traitName, trait, combo, setter, layout, settingsKey)
        
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- List with toggle plot button -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    plotButtonState = False
    _activePlotButton = None
    def _addListWithPlotButton(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        plotFunc = trait.metadata.get("buttonFunction", None)

        combo = CenteredComboBox()
        combo.addItems([str(item) for item in current])
        combo.setToolTip(helpText)

        button = QPushButton(trait.metadata.get("buttonLabel", "Display"))
        button.setToolTip(helpText)
        button.setCheckable(True)

        def updatePlot():
            if plotFunc is None:
                return
            method = getattr(settingsObject, plotFunc, None)
            if method is None:
                return

            selected = combo.currentText()
            self.figure.clear()
            if method(self.figure, selected):
                self.plotCanvas.setVisible(True)
                self.plotButtonState = True
            else:
                self.plotCanvas.setVisible(False)
                self.plotButtonState = False
            self.plotCanvas.draw()

        def onButtonToggled(checked):
            if checked:
                # Un-check the previously active button (if it's a different one)
                if self._activePlotButton is not None and self._activePlotButton is not button:
                    prev = self._activePlotButton
                    prev.blockSignals(True)     # avoid triggering its toggled handler
                    prev.setChecked(False)
                    prev.blockSignals(False)

                self._activePlotButton = button
                updatePlot()
            else:
                # Only clear state if THIS button is the active one
                if self._activePlotButton is button:
                    self._activePlotButton = None
                    self.plotButtonState = False
                    self.figure.clear()
                    self.plotCanvas.setVisible(False)
                    self.plotCanvas.draw()

        def onSelectionChanged(index):
            if self._updatingWidget or index < 0:
                return

            # Only refresh if this control already owns the displayed plot.
            refreshPlot = self._activePlotButton is button and button.isChecked() and self.plotWindow.isVisible()

            # Preserve the existing selected-first settings convention.
            selected = combo.itemText(index)
            opts = [combo.itemText(i) for i in range(combo.count())]
            newList = [selected] + [x for x in opts if x != selected]

            wasBlocked = combo.blockSignals(True)
            try:
                self._commit(settingsObject, traitName, newList)
            finally:
                combo.blockSignals(wasBlocked)

            if refreshPlot:
                updatePlot()     
        
        button.toggled.connect(onButtonToggled)
        combo.currentIndexChanged.connect(onSelectionChanged)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(combo, 1)
        h.addWidget(button)

        def setter(value):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(item) for item in value])
            combo.blockSignals(False)

        self._register(traitName, trait, row, setter, layout, settingsKey)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- Text with a set button -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addTextWithSetButton(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        """
        Add a Unicode editor with an action button.

        Normal input:
            Edits update the settings object.
            buttonFunction() returns the replacement string.

        Transient input:
            Edits remain exclusively in the widget.
            buttonFunction(text) returns the replacement string.
            Refreshing or changing profiles discards pending input.

        isPassword masks input but does not itself prevent persistence.

        isPath=True additionally displays the shared Browse button beside
        the action button. Both use the same text editor. A Browse selection
        follows the ordinary/transient text-change behavior above.
        """
        buttonFunc = trait.metadata.get("buttonFunction")
        transientInput = bool(trait.metadata.get("transientInput", False))
        isPassword = bool(trait.metadata.get("isPassword", False))
        writeName = trait.metadata.get("property", traitName)

        edit = QLineEdit()
        edit.setAlignment(Qt.AlignCenter)
        edit.setToolTip(helpText)

        if isPassword:
            edit.setEchoMode(QLineEdit.EchoMode.Password)

        # Transient fields never populate from a stored settings value.
        if not transientInput:
            edit.setText(str(current) if current is not None else "")

        button = QPushButton(trait.metadata.get("buttonLabel", "Set"))
        button.setToolTip(helpText)

        # This is an action, not a toggle or an implicit Enter-key action.
        button.setCheckable(False)
        button.setAutoDefault(False)
        button.setDefault(False)

        def currentOwner():
            """Resolve the currently selected profile at action time."""
            if isinstance(settingsObject, _RetargetableProxy):
                return object.__getattribute__(settingsObject, "_target")
            return settingsObject

        def setEditorText(value):
            """Update the editor without triggering a settings assignment."""
            wasBlocked = edit.blockSignals(True)
            try:
                edit.setText(value)
            finally:
                edit.blockSignals(wasBlocked)

        def showError(error):
            """Display useful error details with the entered secret redacted."""
            details = str(error)

            if isPassword:
                password = edit.text()

                if password:
                    details = details.replace(password, "*********")
                else:
                    # Without the input, we cannot reliably redact it.
                    details = "Error details are unavailable for this password operation."

            message = QMessageBox(self)
            message.setIcon(QMessageBox.Icon.Warning)
            message.setWindowTitle("Setting action failed")
            message.setTextFormat(Qt.TextFormat.PlainText)
            message.setText(f"The operation could not be completed ({type(error).__name__}).")
            message.setInformativeText(details)
            message.setStandardButtons(QMessageBox.StandardButton.Ok)
            message.exec()
            
        def onChange(text):
            if self._updatingWidget or transientInput:
                return

            try:
                # Assign directly so failures do not pass through _commit(),
                # which prints exception messages and might expose input.
                setattr(currentOwner(), writeName, text)
            except Exception as error:
                showError(error)

        def onButtonClicked(checked=False):
            if self._updatingWidget:
                return

            owner = currentOwner()
            button.setEnabled(False)

            try:
                if not isinstance(buttonFunc, str) or not buttonFunc:
                    raise ValueError("Missing buttonFunction metadata.")

                method = getattr(owner, buttonFunc, None)

                if not callable(method):
                    raise TypeError("buttonFunction must name a callable method.")

                replacement = method(edit.text()) if transientInput else method()

                if not isinstance(replacement, str):
                    raise TypeError("The button callback must return a string.")

                if not transientInput:
                    setattr(owner, writeName, replacement)

                if currentOwner() is owner:
                    setEditorText(replacement)

            except Exception as error:
                # Redact using the input before clearing it.
                showError(error)

            finally:
                if isPassword:
                    setEditorText("")

                button.setEnabled(True)

        def setter(value):
            # Used when refreshing fields or switching profiles.
            # Never carry pending transient input into another profile.
            text = "" if transientInput else (str(value) if value is not None else "")
            setEditorText(text)

        edit.textChanged.connect(onChange)
        button.clicked.connect(onButtonClicked)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        if trait.metadata.get("isPath", False):
            h.addWidget(self._makePathBrowseButton(edit, traitName, trait))
        h.addWidget(button)

        self._register(traitName, trait, row, setter, layout, settingsKey)


    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    # ----- RGB -----
    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def _addRGB(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        rgb = list(current) if current else [0.0, 0.0, 0.0]
        boxes = []
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        for i, name in enumerate(("R", "G", "B")):
            h.addWidget(QLabel(name))
            spin = QDoubleSpinBox()
            spin.setMinimum(0.0)
            spin.setMaximum(1.0)
            spin.setSingleStep(0.01)
            spin.setValue(float(rgb[i]) if i < len(rgb) else 0.0)
            spin.setToolTip(f"{helpText} - {name}")
            h.addWidget(spin)
            boxes.append(spin)

        def onChange(_=None):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, [b.value() for b in boxes])

        for b in boxes:
            b.valueChanged.connect(onChange)

        def setter(value):
            for i, b in enumerate(boxes):
                if i < len(value):
                    b.setValue(float(value[i]))

        self._register(traitName, trait, row, setter, layout, settingsKey)
            
    #########################################
    # Helpers
    #########################################
    def _register(self, traitName, trait, widget, setter, layout=None, settingsKey=None):
        if layout is None:
            layout = self.formLayout

        traitLabel = traitName
        if trait is not None and "traitDisplayName" in trait.metadata:
            traitLabel = trait.metadata["traitDisplayName"]

        label = QLabel(traitLabel)
        label.setObjectName("traitLabel")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        if trait is not None:
            label.setToolTip(self._helpText(traitName, trait))

        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        widget.setMinimumHeight(30)

        if settingsKey is None:
            settingsKey = ("root",)

        self.traitWidgets.setdefault(settingsKey, {})
        self.traitWidgets[settingsKey][traitName] = {
            "widget": widget,
            "row": widget,
            "label": label,
            "setter": setter,
            "layout": layout,
            "trait": trait,
            "traitName": traitName,
            "settingsKey": settingsKey,
        }

        layout.addRow(label, widget)

        # Keep hidden controls registered for later visibility changes.
        if trait is not None and not trait.metadata.get("visible", True):
            layout.setRowVisible(widget, False)

        if trait is not None:
            widget.setEnabled(bool(trait.metadata.get("enabled", True)))

    def addButtonRow(self, name, buttons, displayName=None, layout=None, settingsKey=None):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        for b in buttons:
            h.addWidget(b)

        # fake a trait just to carry a display label, or pass None and rely on `name`
        self._register(displayName or name, None, row, lambda v: None, layout, settingsKey)
        
    def _commit(self, settingsObject, traitName, value):
        """Push a widget change into the settings copy."""
        try:
            writeName = traitName

            # Look up property metadata if the object exposes traits.
            try:
                trait = settingsObject.trait(traitName)
                writeName = trait.metadata.get("property", traitName)
            except Exception:
                pass

            setattr(settingsObject, writeName, value)

        except Exception as error:
            # Keep the dialog alive on a bad value.
            print(f"(pglTraitsDialog:_commit) Could not set {traitName}: {error}")

        finally:
            # Checkbox changes immediately update dependent fields.
            self._refreshContingencies()
            
    def _onOk(self):
        """Validate edits, reset invalid values to defaults, and accept."""
        validator = getattr(self.settings, "validateSettings", None)
        if callable(validator): validator()

        self.accepted_ = True
        self.accept()
        
    def _onCancel(self):
        self.accepted_ = False
        self.reject()

    #########################################
    # Style
    #########################################
    def _darkStyle(self):
        return """
        QDialog {
            background-color: #1e1f22;
        }

        #buttonBar {
            background-color: #26282c;
            border-top: 1px solid #3a3d42;
        }

        QLabel {
            color: #d6d9de;
            font-size: 13px;
        }
        #traitLabel, #rowInternalLabel {
            color: #000000;
            font-weight: 600;
        }

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #2b2d31;
            color: #eaecef;
            border: 1px solid #3a3d42;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 13px;
            selection-background-color: #3d6fd1;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #4a8cff;
            background-color: #303338;
        }
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
        QComboBox:disabled  {
            color: #6b7078;
            background-color: #232427;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 24px;
            border-left: 1px solid #3a3d42;
            border-top-right-radius: 6px;
            background-color: #34373c;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 24px;
            border-left: 1px solid #3a3d42;
            border-bottom-right-radius: 6px;
            background-color: #34373c;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #4a8cff;
        }
        /* Combo box */
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 26px;
            border-left: 0px solid #3a3d42;
        }
        QComboBox::down-arrow {
            width: 0px; height: 0px;
            image: none;
            border-left: 0px solid transparent;
            border-right: 0px solid transparent;
            border-top: 0px solid #d6d9de;
        }
        QComboBox QAbstractItemView {
            background-color: #2b2d31;
            color: #eaecef;
            border: 1px solid #3a3d42;
            border-radius: 6px;
            padding: 4px;
            outline: none;
            selection-background-color: #3d6fd1;
            selection-color: #ffffff;
        }
        QComboBox QAbstractItemView::item {
            min-height: 26px;
            padding: 2px 8px;
        }

        QComboBox#settingsSelector {
            background-color: #263b4d;
            color: #ffffff;
            font-weight: bold;
            border: 2px solid #4fa3d1;
            border-radius: 4px;
            padding: 4px 8px;
        }

        QComboBox#settingsSelector:hover {
            border: 2px solid #88c0d0;
            background-color: #434c5e;
        }

        QComboBox#settingsSelector:focus {
            border: 2px solid #8fbcbb;
        }
        
        /* Checkboxes */
        QCheckBox {
            color: #d6d9de;
            spacing: 8px;
            font-size: 13px;
        }
        QCheckBox::indicator {
            width: 18px; height: 18px;
            border: 1px solid #3a3d42;
            border-radius: 4px;
            background-color: #2b2d31;
        }
        QCheckBox::indicator:checked {
            background-color: #4a8cff;
            border: 1px solid #4a8cff;
        }

        /* Sliders */
        QSlider::groove:horizontal {
            height: 6px;
            background: #3a3d42;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #4a8cff;
            width: 18px;
            height: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #6aa0ff;
        }
        QSlider::sub-page:horizontal {
            background: #3d6fd1;
            border-radius: 3px;
        }

        /* Scroll area */
        QScrollArea { background-color: #1e1f22; border: none; }
        QScrollBar:vertical {
            background: #1e1f22; width: 12px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #3a3d42; border-radius: 6px; min-height: 30px;
        }
        QScrollBar::handle:vertical:hover { background: #4a4d53; }
        QScrollBar::add-line, QScrollBar::sub-line { height: 0; }

        /* Buttons */
        QPushButton {
            background-color: #34373c;
            color: #eaecef;
            border: 1px solid #3a3d42;
            border-radius: 6px;
            padding: 7px 20px;
            font-size: 13px;
            min-width: 84px;
        }
        QPushButton:hover { background-color: #3f4247; }
        QPushButton:default {
            background-color: #4a8cff;
            border: 1px solid #4a8cff;
            color: #ffffff;
        }
        QPushButton:default:hover { background-color: #5a97ff; }

        /* Spin box adjustment buttons */
        QPushButton#halfSizeButton {
            min-width: 36px;
            padding: 7px 20px;
        }
        QPushButton:disabled {
            background-color: #2a2c30;
            color: #6b6e73;
            border: 1px solid #2f3136;
        }
        
        /* Multi Select Rows */
        QScrollArea#multiSelect {
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            background: white;            
        }

        QWidget#multiSelectRow {
            border-radius: 4px;
            background-color: #1e1f22;
        }
        QWidget#multiSelectRow:hover {
           background-color: #292b2f;
        }
        QWidget#multiSelectRow[focused="true"] {
            background-color: #3a5f8f;
        }
        QWidget#multiSelectRow[focused="true"]:hover {
            background-color: #4a6fa0;
        }        """
    def _wrapSpinSingleStep(self, spin):
        """Wrap a spinbox with a large - on the left and + on the right."""
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spin.setAlignment(Qt.AlignCenter)          # center the number between buttons

        minus = QPushButton("\u2212")              # real minus sign −
        plus  = QPushButton("+")
        for b in (minus, plus):
            b.setObjectName("stepButton")
            b.setFixedSize(40, 32)                 # large, square-ish
            b.setAutoRepeat(True)                  # hold to keep stepping
            b.setAutoRepeatDelay(300)
            b.setAutoRepeatInterval(60)

        minus.clicked.connect(spin.stepDown)
        plus.clicked.connect(spin.stepUp)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(minus)                         # left
        h.addWidget(spin, 1)                       # middle, expands
        h.addWidget(plus)                          # right
        return row
    
    def _wrapSpinDoubleStep(self, spin, bigStep=None):
        spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        if bigStep is None:
            bigStep = spin.singleStep() * 10

        minusBig = QPushButton("--")
        minus = QPushButton("-")
        plus = QPushButton("+")
        plusBig = QPushButton("++")
        
        minusBig.setObjectName("halfSizeButton")
        plusBig.setObjectName("halfSizeButton")
        minus.setObjectName("halfSizeButton")
        plus.setObjectName("halfSizeButton")

        # get natural size of a normal button
        normalWidth = minus.sizeHint().width()
        smallWidth = normalWidth // 2

        for button in (minusBig, minus, plus, plusBig):
            button.setFixedWidth(smallWidth)

        minus.clicked.connect(lambda: spin.setValue(spin.value() - spin.singleStep()))
        minusBig.clicked.connect(lambda: spin.setValue(spin.value() - bigStep))
        plus.clicked.connect(lambda: spin.setValue(spin.value() + spin.singleStep()))
        plusBig.clicked.connect(lambda: spin.setValue(spin.value() + bigStep))

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        h.addWidget(minusBig)
        h.addWidget(minus)
        h.addWidget(spin, 1)
        h.addWidget(plus)
        h.addWidget(plusBig)

        return row

    def raiseAndActivate(self):
        self.show()
        self.raise_()
        self.activateWindow()


    ################################################################
    # Refresh Boolean-dependent fields
    ################################################################
    def _refreshContingencies(self):
        """
        Apply contingentOn metadata to registered fields.

        The controlling trait must be a Boolean on the same settings object.
        Missing or invalid dependencies warn once per object/field and are
        ignored. Explicit enabled=False metadata always takes precedence.
        """
        for field in self._contingentFields:
            owner = field["owner"]

            # Resolve the actual preset currently displayed by this proxy.
            if isinstance(owner, _RetargetableProxy):
                owner = object.__getattribute__(owner, "_target")

            traitName = field["traitName"]
            trait = owner.traits().get(traitName)
            if trait is None: continue

            # Start from the field's ordinary enabled setting.
            enabled = bool(trait.metadata.get("enabled", True))
            controllerName = trait.metadata.get("contingentOn")
            problem = None

            if controllerName:
                if not isinstance(controllerName, str):
                    problem = "contingentOn must name a Boolean trait."
                else:
                    controller = owner.traits().get(controllerName)

                    if controller is None:
                        problem = f"controlling trait '{controllerName}' does not exist."
                    elif not isinstance(controller, Bool):
                        problem = f"controlling trait '{controllerName}' is not Boolean."
                    else:
                        enabled = enabled and bool(getattr(owner, controllerName))

            if problem is not None:
                # Ignore the invalid dependency, preserving ordinary behavior.
                warningKey = (id(owner), traitName, problem)

                if warningKey not in self._contingentWarnings:
                    self._contingentWarnings.add(warningKey)
                    settingsName = getattr(owner, "name", type(owner).__name__)
                    pglMessages.warning(
                        f"{settingsName}.{traitName}: {problem} "
                        "Ignoring contingentOn."
                    )

            field["widget"].setEnabled(enabled)
            
    ################################################################
    # Compact rows explicitly started by existing layout metadata
    ################################################################
    def _compactTraitRows(self):
        """
        Leave ordinary controls on separate rows by default.

        Explicit group starters:
            rowLabel on a Boolean:
                Start a consolidated row containing this Boolean and
                subsequent consecutive Booleans in the same settings scope.

            inlineContingent=True on a plain text field:
                Pair it with its immediately following controlling Boolean.
                Subsequent consecutive Booleans may join the row.

        Group boundaries:
            A non-Boolean, a different settings scope, another explicit
            rowLabel, rowBreakBefore, or rowBreakAfter.

        Reuses existing controls, setters, and contingency registrations.
        No consolidate metadata is used.
        """
        form = self.formLayout

        # Match actual form rows to their existing registrations.
        entriesByWidget = {}
        for fields in self.traitWidgets.values():
            for entry in fields.values():
                if entry["layout"] is form:
                    entriesByWidget[id(entry["widget"])] = entry

        rows = []
        for rowIndex in range(form.rowCount()):
            fieldItem = form.itemAt(rowIndex, QFormLayout.ItemRole.FieldRole)
            widget = fieldItem.widget() if fieldItem is not None else None
            rows.append(
                entriesByWidget.get(id(widget))
                if widget is not None else None
            )

        def metadata(entry):
            trait = entry.get("trait") if entry else None
            return trait.metadata if trait is not None else {}

        def visible(entry):
            return entry is not None and bool(metadata(entry).get("visible", True))

        def boolean(entry):
            return visible(entry) and isinstance(entry.get("trait"), Bool)

        def sameScope(left, right):
            return (
                left is not None and right is not None
                and left["settingsKey"] == right["settingsKey"]
            )

        plans = []
        index = 0

        while index < len(rows):
            first = rows[index]
            if not visible(first):
                index += 1
                continue

            firstMetadata = metadata(first)
            pairCount = 0

            # Only plain text editors participate in inline pairing.
            # Composite path/action editors retain their existing layout.
            inlineText = (
                isinstance(first.get("trait"), Unicode)
                and isinstance(first["widget"], QLineEdit)
                and firstMetadata.get("inlineContingent") is True
            )

            if inlineText:
                following = rows[index + 1] if index + 1 < len(rows) else None
                controllerName = firstMetadata.get("contingentOn")

                if (
                    boolean(following)
                    and sameScope(first, following)
                    and following["traitName"] == controllerName
                    and not firstMetadata.get("rowBreakAfter", False)
                    and not metadata(following).get("rowBreakBefore", False)
                    and "rowLabel" not in metadata(following)
                ):
                    members = [first, following]
                    pairCount = 2
                else:
                    pglMessages.warning(
                        f"{first['traitName']}: inlineContingent requires an "
                        "immediately following visible controlling Boolean "
                        "without an intervening row break. Keeping ordinary layout."
                    )
                    index += 1
                    continue

            elif boolean(first) and "rowLabel" in firstMetadata:
                members = [first]

            else:
                # Neither existing opt-in mechanism starts a group here.
                index += 1
                continue

            nextIndex = index + len(members)

            while nextIndex < len(rows):
                candidate = rows[nextIndex]

                if metadata(members[-1]).get("rowBreakAfter", False): break
                if not boolean(candidate): break
                if not sameScope(first, candidate): break
                if metadata(candidate).get("rowBreakBefore", False): break
                if "rowLabel" in metadata(candidate): break

                members.append(candidate)
                nextIndex += 1

            plans.append((members, pairCount))
            index = nextIndex

        for members, pairCount in plans:
            self._makeCompactTraitRow(members, pairCount)

    ################################################################
    # Build one grouped row from already-registered controls
    ################################################################
    def _makeCompactTraitRow(self, entries, pairCount=0):
        """Move controls without replacing their signals or backing objects."""
        form = self.formLayout
        first = entries[0]
        trait = first["trait"]

        firstRow, role = form.getWidgetPosition(first["widget"])
        if firstRow < 0:
            raise RuntimeError("Could not locate a trait row for compact layout")

        # takeRow() removes layout items without deleting their widgets.
        for entry in entries:
            removed = form.takeRow(entry["widget"])
            if removed.fieldItem is None:
                raise RuntimeError(f"Could not remove row for {entry['traitName']}")

            oldLabel = entry.get("label")
            if oldLabel is not None:
                oldLabel.hide()
                oldLabel.deleteLater()

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        flow = _pglTraitFlowLayout(container)

        pair = None
        if pairCount:
            pairWidget = QWidget()
            pairLayout = QHBoxLayout(pairWidget)
            pairLayout.setContentsMargins(0, 0, 0, 0)
            pairLayout.setSpacing(8)

            textEntry, controllerEntry = entries[:2]
            textWidget = textEntry["widget"]
            textWidget.setMinimumWidth(160)
            textWidget.setMaximumWidth(240)
            textWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

            controller = controllerEntry["widget"]
            controller.setText(str(controllerEntry["trait"].metadata.get(
                "traitDisplayName", controllerEntry["traitName"]
            )))
            controller.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

            pairLayout.addWidget(textWidget)
            pairLayout.addWidget(controller)
            flow.addWidget(pairWidget)
            pair = (pairWidget, entries[:2])

        for entry in entries[pairCount:]:
            checkbox = entry["widget"]
            checkbox.setText(str(entry["trait"].metadata.get(
                "traitDisplayName", entry["traitName"]
            )))
            checkbox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            flow.addWidget(checkbox)

        # Existing naming remains the fallback when no row label is supplied.
        rowLabelText = trait.metadata.get(
            "rowLabel", trait.metadata.get("traitDisplayName", first["traitName"])
        )
        rowLabel = QLabel(str(rowLabelText))
        rowLabel.setObjectName("traitLabel")
        rowLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rowLabel.setToolTip(self._helpText(first["traitName"], trait))

        form.insertRow(firstRow, rowLabel, container)

        group = _pglCompactTraitGroup(form, container, entries, pair=pair)
        adapter = _pglCompactRowLayout(form, group)

        for entry in entries:
            # "row" remains the individual field. Contingencies must never
            # disable the shared container or the controlling checkbox.
            entry["row"] = entry["widget"]
            entry["label"] = None
            entry["layout"] = adapter
            entry["compactGroup"] = group

        self._compactGroups.append(group)
        group.refresh()
        
    ################################################################
    # Show all form rows unless limited by the available screen height
    ################################################################
    def _fitDialogToContents(self):
        """
        Fit the dialog to its current form contents and current width.

        No arbitrary height cap. The screen's available geometry limits the
        window, leaving a small margin around its frame.

        Called during initial construction and once after Qt resolves layout.
        Does not repeatedly override subsequent user resizing.
        """
        import math

        formWidget = self.formLayout.parentWidget()
        if formWidget is None: return

        # Find this form's scroll area, not a nested multi-select scroll area.
        scroll = formWidget.parentWidget()
        while scroll is not None and not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        if scroll is None: return

        mainLayout = self.layout()
        screen = self.screen() or QApplication.primaryScreen()
        if mainLayout is None or screen is None: return

        available = screen.availableGeometry()
        screenMargin = 12

        frameWidth = max(0, self.frameGeometry().width() - self.width())
        frameHeight = max(0, self.frameGeometry().height() - self.height())

        maxClientWidth = max(1, available.width() - 2 * screenMargin - frameWidth)
        maxClientHeight = max(1, available.height() - 2 * screenMargin - frameHeight)

        # Respect the existing/requested width unless it exceeds the screen.
        if self.minimumWidth() > maxClientWidth:
            self.setMinimumWidth(maxClientWidth)
        if self.minimumHeight() > maxClientHeight:
            self.setMinimumHeight(maxClientHeight)

        targetWidth = min(self.width(), maxClientWidth)
        self.resize(targetWidth, min(self.height(), maxClientHeight))

        mainLayout.invalidate()
        mainLayout.activate()

        # Determine the available form width without a vertical scrollbar.
        # If scrolling is needed, remeasure with space reserved for it.
        noScrollWidth = max(
            1,
            scroll.width() - 2 * scroll.frameWidth(),
        )

        def formHeightForWidth(width):
            self.formLayout.invalidate()
            if self.formLayout.hasHeightForWidth():
                height = self.formLayout.totalHeightForWidth(width)
            else:
                height = self.formLayout.totalSizeHint().height()

            if height < 0:
                height = self.formLayout.totalSizeHint().height()

            return max(formWidget.minimumHeight(), int(math.ceil(height)))

        # Account for everything outside the form's scroll viewport:
        # button bar, main-layout margins, spacing, and scroll-area frame.
        margins = mainLayout.contentsMargins()
        outsideHeight = margins.top() + margins.bottom()
        visibleItems = []

        for index in range(mainLayout.count()):
            item = mainLayout.itemAt(index)
            if item.isEmpty(): continue
            visibleItems.append(item)

            if item.widget() is not scroll:
                outsideHeight += max(
                    item.sizeHint().height(),
                    item.minimumSize().height(),
                )

        outsideHeight += max(0, len(visibleItems) - 1) * max(0, mainLayout.spacing())
        outsideHeight += 2 * scroll.frameWidth()

        desiredHeight = formHeightForWidth(noScrollWidth) + outsideHeight

        if desiredHeight > maxClientHeight:
            scrollbarWidth = scroll.verticalScrollBar().sizeHint().width()
            desiredHeight = (
                formHeightForWidth(max(1, noScrollWidth - scrollbarWidth))
                + outsideHeight
            )

        targetHeight = min(
            maxClientHeight,
            max(self.minimumHeight(), desiredHeight),
        )
        self.resize(targetWidth, targetHeight)
        mainLayout.activate()

        # Keep the resized frame within the available desktop.
        # Center vertically so growing the window does not push its bottom
        # offscreen from the earlier, shorter window position.
        frame = self.frameGeometry()
        minX = available.left() + screenMargin
        maxX = available.right() - screenMargin - frame.width() + 1

        x = max(minX, min(frame.left(), maxX))
        y = available.top() + max(0, (available.height() - frame.height()) // 2)
        self.move(x, y)
#####################################################################
# Helper class uses by settingsList which retargets fields
#####################################################################
class _RetargetableProxy:
    """Stands in for settingsObject; forwards get/set to whatever object it currently targets."""
    def __init__(self, target):
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_target"), name, value)

    def retarget(self, newTarget):
        object.__setattr__(self, "_target", newTarget)
        
    def getClassName(self):
        return self._target.__class__.__name__
        
#####################################################################
# subclassed UI elements for customization
#####################################################################
class CenteredComboBox(QComboBox):
    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)
        opt.currentText = ""  # suppress default left-aligned text
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)
        painter.drawText(self.rect(), Qt.AlignCenter, self.currentText())
        
class ScrollableFigureCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event):
        # Allow Ctrl+wheel for matplotlib zoom if desired
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return

        # Otherwise let the scroll area handle scrolling
        p = self.parent()
        while p is not None and not isinstance(p, QScrollArea):
            p = p.parent()

        if p is not None:
            QCoreApplication.sendEvent(p.verticalScrollBar(), event)
        else:
            super().wheelEvent(event)
            
class _ClickableRow(QWidget):
    clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_StyledBackground, True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QListView,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
)


class CheckableComboBox(QComboBox):
    selectionChanged = Signal(list)
    selectionClosed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._popupOpen = False
        self._pressedIndex = None

        self.setEditable(False)
        self.setView(QListView(self))
        self.setModel(QStandardItemModel(self))

        # Install these after setView()/setModel(), so our filters
        # receive events before the combo box's default handlers.
        self.view().installEventFilter(self)
        self.view().viewport().installEventFilter(self)

        self.model().itemChanged.connect(self._onItemChanged)

    def eventFilter(self, obj, event):
        eventType = event.type()

        if obj is self.view().viewport():
            if eventType in (
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonDblClick,
            ):
                if event.button() == Qt.MouseButton.LeftButton:
                    index = self.view().indexAt(
                        event.position().toPoint()
                    )

                    self._pressedIndex = (
                        index if index.isValid() else None
                    )

                    if index.isValid():
                        self.view().setCurrentIndex(index)

                    # Prevent Qt from treating this as a normal
                    # single-selection combo box click.
                    return True

            elif eventType == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    index = self.view().indexAt(
                        event.position().toPoint()
                    )

                    if (
                        self._pressedIndex is not None
                        and index.isValid()
                        and index == self._pressedIndex
                    ):
                        self._toggleItem(index.row())

                    self._pressedIndex = None

                    # Consume release too, so the popup stays open.
                    return True

        if obj is self.view() or obj is self.view().viewport():
            if eventType == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Space:
                    index = self.view().currentIndex()

                    if index.isValid() and not event.isAutoRepeat():
                        self._toggleItem(index.row())

                    return True

                if event.key() in (
                    Qt.Key.Key_Return,
                    Qt.Key.Key_Enter,
                    Qt.Key.Key_Escape,
                ):
                    self.hidePopup()
                    return True

        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        """Draw the normal combo box with our selection summary."""
        option = QStyleOptionComboBox()
        self.initStyleOption(option)

        count = len(self.checkedItems())
        option.currentText = (
            "None selected" if count == 0 else f"{count} selected"
        )

        # Do not display the current item's icon in the summary.
        option.currentIcon = type(option.currentIcon)()

        painter = QStylePainter(self)
        painter.drawComplexControl(
            QStyle.ComplexControl.CC_ComboBox,
            option,
        )
        painter.drawControl(
            QStyle.ControlElement.CE_ComboBoxLabel,
            option,
        )

    def showPopup(self):
        self._pressedIndex = None
        self.view().setMinimumWidth(self.width())

        super().showPopup()
        self._popupOpen = self.view().isVisible()

    def hidePopup(self):
        wasOpen = self._popupOpen
        self._popupOpen = False
        self._pressedIndex = None

        super().hidePopup()
        self.update()

        if wasOpen:
            self.selectionClosed.emit(len(self.checkedItems()))

    def addItem(self, text, userData=None, checked=False):
        item = QStandardItem(text)

        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )

        if userData is not None:
            item.setData(userData, Qt.ItemDataRole.UserRole)

        item.setCheckState(
            Qt.CheckState.Checked
            if checked
            else Qt.CheckState.Unchecked
        )

        self.model().appendRow(item)
        self.update()

    def addItems(self, texts, checked=False):
        for text in texts:
            self.addItem(text, checked=checked)

    def checkedItems(self):
        return [
            self.model().item(row).text()
            for row in range(self.model().rowCount())
            if self.isItemChecked(row)
        ]

    def checkedData(self):
        return [
            self.model().item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.model().rowCount())
            if self.isItemChecked(row)
        ]

    def setItemChecked(self, index, checked=True):
        if not 0 <= index < self.model().rowCount():
            return

        item = self.model().item(index)
        item.setCheckState(
            Qt.CheckState.Checked
            if checked
            else Qt.CheckState.Unchecked
        )

        # itemChanged handles updating and emitting the signal.

    def isItemChecked(self, index):
        if not 0 <= index < self.model().rowCount():
            return False

        return (
            self.model().item(index).checkState()
            == Qt.CheckState.Checked
        )

    def _toggleItem(self, row):
        item = self.model().item(row)

        if item is not None and item.isEnabled() and item.isCheckable():
            self.setItemChecked(row, not self.isItemChecked(row))

    def _onItemChanged(self, item):
        self.update()
        self.selectionChanged.emit(self.checkedItems())

class _PlotWindow(QDialog):
    """Move the main dialog first, then reveal the plot beside it."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self._savedGeometry = None
        self._opening = False
        self._moveAnimation = None
        self.setWindowTitle("Plot")
        self.setWindowModality(Qt.NonModal)

    def setVisible(self, visible):
        if not visible:
            # Cancel a pending first opening.
            self._opening = False
            if self._moveAnimation is not None:
                self._moveAnimation.stop()

            if self.isVisible():
                self._savedGeometry = self.saveGeometry()

            super().setVisible(False)
            return

        if self.isVisible() or self._opening:
            return

        # Subsequent openings reuse the saved geometry without animation.
        if self._savedGeometry is not None:
            super().setVisible(True)
            self.restoreGeometry(self._savedGeometry)
            return

        self._opening = True

        screenRect = self.owner.screen().availableGeometry()
        ownerRect = self.owner.frameGeometry()

        targetX = screenRect.left() + screenRect.width() // 2
        targetX = max(screenRect.left(), min(targetX, screenRect.right() - ownerRect.width() + 1))
        shiftX = targetX - ownerRect.left()

        if not shiftX:
            self._showInitially()
            return

        startPos = self.owner.pos()
        self._moveAnimation = QPropertyAnimation(self.owner, b"pos", self)
        self._moveAnimation.setDuration(250)
        self._moveAnimation.setStartValue(startPos)
        self._moveAnimation.setEndValue(startPos + QPoint(shiftX, 0))
        self._moveAnimation.setEasingCurve(QEasingCurve.InOutCubic)
        self._moveAnimation.finished.connect(self._showInitially)
        self._moveAnimation.start()

    def _showInitially(self):
        if not self._opening:
            return

        self._opening = False

        # The movement has finished. Show and position before the next paint.
        super().setVisible(True)
        self._placeInitially()
        self.raise_()
        self.activateWindow()

    def _placeInitially(self):
        gap = 8
        ownerRect = self.owner.frameGeometry()
        screenRect = self.owner.screen().availableGeometry()

        availableWidth = max(0, ownerRect.left() - screenRect.left() - gap)
        targetWidth = min(ownerRect.width(), availableWidth)
        targetHeight = min(ownerRect.height(), screenRect.height())

        frameWidth = self.frameGeometry().width() - self.width()
        frameHeight = self.frameGeometry().height() - self.height()

        minSize = self.minimumSizeHint().expandedTo(self.minimumSize())
        clientWidth = max(1, minSize.width(), targetWidth - frameWidth)
        clientHeight = max(1, minSize.height(), targetHeight - frameHeight)
        self.resize(clientWidth, clientHeight)

        plotRect = self.frameGeometry()
        x = max(screenRect.left(), ownerRect.left() - gap - plotRect.width())
        y = max(screenRect.top(), min(ownerRect.top(), screenRect.bottom() - plotRect.height() + 1))
        self.move(x, y)

    def closeEvent(self, event):
        self.owner._onPlotWindowClosed()
        event.accept()

    def reject(self):
        self.close()

class _WindowFigureCanvas(FigureCanvasQTAgg):
    """Make existing canvas visibility calls control its separate window."""

    def __init__(self, figure, plotWindow):
        self.plotWindow = None
        super().__init__(figure)
        self.plotWindow = plotWindow

    def setVisible(self, visible):
        super().setVisible(visible)

        if self.plotWindow is None:
            return

        self.plotWindow.setVisible(visible)

        if visible:
            self.plotWindow.raise_()
            self.plotWindow.activateWindow()
            
#####################################################################
# pglTraitsDialog: what gets called by the user. This rund
# pglTraitsDialogStandalone which runs outside the jupyter notebook
# to avoid crashy-conflicty behavior.
#####################################################################
class pglDialogs:
    @staticmethod
    def traitsDialog(settings, width=None):
        """
        Pops up a PySide6 dialog in a separate process, blocks until closed,
        and returns edited settings (OK) or None (Cancel).

        width optionally requests a dialog width in Qt logical pixels.
        None preserves the normal dialog width.
        The standalone process limits the requested width to the screen.
        """
        if width is not None and (type(width) is not int or width <= 0):
            raise ValueError("width must be a positive integer or None.")

        try:
            with tempfile.TemporaryDirectory() as tmpDirStr:
                tmpDir = Path(tmpDirStr)
                inFile = tmpDir / "in.json"
                outFile = tmpDir / "out.json"

                settings.save(inFile)

                # run, redirecting stderr because it produces meaningless messages from text handling
                scriptPath = Path(__file__).parent / "pglTraitsDialogStandalone.py"
                command = [sys.executable, str(scriptPath), str(inFile), str(outFile)]
                if width is not None: command.append(str(width))
                result = subprocess.run(command, stderr=subprocess.DEVNULL)

                if result.returncode == 0 and outFile.exists():
                    # OK
                    return pglSerialize.load(outFile)
                else:
                    # Cancel
                    return None
        except Exception as e:
            pglMessages.warning(f"Error running traitsDialog: {e}")
            return None
        
    @staticmethod
    def toolDialog(document, width=None, *, returnValidation=False):
        """Edit a tool schema using the existing standalone traits dialog.

        Validate uses settings.buttons and checks only the displayed profile.
        Opening the dialog does not probe.

        Default return: edited document, or None on Cancel.
        returnValidation=True: (edited document, validation records), or None.

        All dialog files/logs are retained. configure() publishes matching
        observations only after saving accepted configuration.
        """
        import json
        import os
        import time
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        if width is not None and (type(width) is not int or width <= 0):
            raise ValueError("width must be positive or None.")
        if type(returnValidation) is not bool:
            raise TypeError("returnValidation must be Boolean.")

        document = pglToolDialog.validate(document)
        root = Path(tempfile.mkdtemp(prefix="pgl-tool-dialog-"))
        inputPath = root / "in.json"
        outputPath = root / "out.json"
        historyPath = root / "validation-history.json"
        stdoutPath = root / "stdout.log"
        stderrPath = root / "stderr.log"

        inputPath.write_text(
            json.dumps(document, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        pglMessages.message(f"Tool dialog files retained at: {root}")

        script = Path(__file__).parent / "pglTraitsDialogStandalone.py"
        command = [
            sys.executable, str(script), "--tool-schema",
            str(inputPath), str(outputPath),
        ]
        if width is not None: command.append(str(width))

        progressPath = root / "progress.jsonl"
        progressPath.touch(exist_ok=False)
        environment = dict(os.environ)
        environment["PGL_TOOL_DIALOG_PROGRESS_FILE"] = str(progressPath)

        pending = b""

        def forwardProgress(reader, final=False):
            nonlocal pending
            # Bound each live pass so noisy output cannot monopolize polling.
            remaining = (
                max(0, progressPath.stat().st_size - reader.tell())
                if final else 65536
            )
            while remaining > 0:
                chunk = reader.read(min(65536, remaining))
                if not chunk: break
                remaining -= len(chunk)
                pending += chunk

                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line: continue
                    try:
                        event = json.loads(line)
                        if (
                            type(event) is not dict or set(event) != {"message"}
                            or not isinstance(event["message"], str)
                        ):
                            raise ValueError("Invalid progress event.")
                        pglMessages.print(event["message"], emphasize=True)
                    except Exception:
                        # UI reporting must not kill a dialog or validation.
                        # Original event bytes remain available for inspection.
                        pass

                if len(pending) > 1024 * 1024:
                    pending = b""

        with stdoutPath.open("xb") as stdout, stderrPath.open("xb") as stderr:
            with progressPath.open("rb", buffering=0) as progressReader:
                process = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, env=environment,
                )
                try:
                    while process.poll() is None:
                        forwardProgress(progressReader)
                        time.sleep(0.1)
                    forwardProgress(progressReader, final=True)
                finally:
                    # As before, interruption stops/reaps the dialog child.
                    # A separately running validation descendant is not
                    # guaranteed to stop; its diagnostic files are retained.
                    if process.poll() is None: process.kill()
                    process.wait()

        if process.returncode == 1: return None
        if process.returncode != 0:
            with stderrPath.open("rb") as source:
                source.seek(max(0, stderrPath.stat().st_size - 12000))
                details = source.read(12000).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Tool dialog failed. Retained files: {root}\n"
                + (details.strip() or "No child diagnostic.")
            )

        def readResponse(path):
            with path.open("rb") as source:
                content = source.read(16 * 1024 * 1024 + 1)
            if len(content) > 16 * 1024 * 1024:
                raise ValueError(f"Dialog response exceeds 16 MiB: {path}")
            return json.loads(content)

        if not outputPath.is_file():
            raise RuntimeError(f"Dialog accepted without returning settings: {root}")

        edited = pglToolDialog.validate(readResponse(outputPath))
        for field in ("schemaVersion", "toolName", "label", "multipleInstallations", "fields"):
            if edited[field] != document[field]:
                raise ValueError(f"Tool dialog changed protected field {field!r}.")

        records = []
        if historyPath.exists():
            history = readResponse(historyPath)
            if (
                type(history) is not dict
                or set(history) != {"schemaVersion", "records"}
                or type(history["schemaVersion"]) is not int
                or history["schemaVersion"] != 1
                or type(history["records"]) is not list
            ):
                raise ValueError("Invalid dialog validation history.")
            records = history["records"]

        return (edited, records) if returnValidation else edited

    ################################################################
    # Display information and request explicit confirmation
    ################################################################
    @staticmethod
    def confirm(summary, title="Confirm", *, details=None, confirmButtonText="Ok", copyText=None):
        """Request confirmation in a separate PySide6 process.

        summary is required nonblank string.
        title is optional and defaults to "Confirm".
        details optionally adds a scrollable, read-only text area.
        confirmButtonText defaults to "Ok".

        copyText optionally enables a copy button with the supplied label.
        It copies details when supplied, otherwise summary.
        Text remains selectable without a copy button.

        Returns True only after explicit confirmation.
        Cancel, Escape, window closure, or dialog failure returns False.
        KeyboardInterrupt propagates without granting approval.

        All displayed content is plain text. This dialog does not execute
        commands or perform the operation being confirmed.

        Uses temporary JSON files for communication, matching traitsDialog's
        process isolation without creating Qt widgets in the caller.

        Examples:
            approved = pglDialogs.confirm(
                "Remove intermediate image attachments from this session?",
            )

            approved = pglDialogs.confirm(
                "Install AFNI",
                "Review the proposed changes before installing.",
                details=installationPlan,
                confirmButtonText="Install",
                copyText="Copy instructions",
            )
        """
        import hashlib
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        from .pglMessages import pglMessages

        for name, value in (("title", title), ("summary", summary), ("confirmButtonText", confirmButtonText)):
            if not isinstance(value, str): raise TypeError(f"{name} must be a string.")
            if not value.strip(): raise ValueError(f"{name} must not be blank.")
            if "\x00" in value: raise ValueError(f"{name} must not contain NUL.")

        for name, value in (("details", details), ("copyText", copyText)):
            if value is None: continue
            if not isinstance(value, str): raise TypeError(f"{name} must be a string or None.")
            if not value.strip(): raise ValueError(f"{name} must be nonblank when supplied.")
            if "\x00" in value: raise ValueError(f"{name} must not contain NUL.")

        request = {
            "title": title,
            "summary": summary,
            "details": details,
            "confirmButtonText": confirmButtonText,
            "copyText": copyText,
        }

        # Bind the response to the complete request, including button labels.
        # Successful process exit alone is never interpreted as confirmation.
        requestText = json.dumps(request, ensure_ascii=False, sort_keys=True)
        requestHash = hashlib.sha256(requestText.encode("utf-8")).hexdigest()

        # Qt lives entirely in the child process. User-supplied text enters
        # through JSON, never through Python source or shell interpolation.
        dialogScript = r'''
import hashlib
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout,
)

requestBytes = Path(sys.argv[1]).read_bytes()
requestHash = hashlib.sha256(requestBytes).hexdigest()
request = json.loads(requestBytes)

# Create the application before constructing any widgets.
app = QApplication([])
dialog = QDialog()
dialog.setWindowTitle(request["title"])
dialog.setSizeGripEnabled(True)
dialog.setStyleSheet("""
    QDialog { background-color: #1e1f22; }
    QLabel { color: #d6d9de; font-size: 13px; }
    QPlainTextEdit {
        background-color: #2b2d31;
        color: #eaecef;
        border: 1px solid #3a3d42;
        border-radius: 6px;
        padding: 8px;
        selection-background-color: #3d6fd1;
    }
    QPushButton {
        background-color: #34373c;
        color: #eaecef;
        border: 1px solid #3a3d42;
        border-radius: 6px;
        padding: 7px 20px;
        min-width: 84px;
    }
    QPushButton:hover { background-color: #3f4247; }
    QPushButton:focus { border: 1px solid #4a8cff; }
    QPushButton:default { border: 2px solid #4a8cff; }
""")

layout = QVBoxLayout(dialog)
layout.setContentsMargins(20, 20, 20, 16)
layout.setSpacing(12)

summaryLabel = QLabel()
summaryLabel.setTextFormat(Qt.TextFormat.PlainText)
summaryLabel.setText(request["summary"])
summaryLabel.setWordWrap(True)
summaryLabel.setTextInteractionFlags(
    Qt.TextInteractionFlag.TextSelectableByMouse
    | Qt.TextInteractionFlag.TextSelectableByKeyboard
)
layout.addWidget(summaryLabel)

# Detailed content is optional; ordinary confirmations remain compact.
hasDetails = request["details"] is not None

if hasDetails:
    detailsEditor = QPlainTextEdit()
    detailsEditor.setReadOnly(True)
    detailsEditor.setUndoRedoEnabled(False)
    detailsEditor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    detailsEditor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    detailsEditor.setPlainText(request["details"])
    layout.addWidget(detailsEditor, 1)

buttons = QHBoxLayout()
cancelButton = QPushButton("Cancel")
confirmButton = QPushButton(request["confirmButtonText"])

# Focusing another button must not silently make it the Enter-key default.
for button in (cancelButton, confirmButton):
    button.setAutoDefault(False)
    button.setDefault(False)

cancelButton.setDefault(True)

# Only show a copy button when the caller explicitly supplies its label.
if request["copyText"] is not None:
    copyButton = QPushButton(request["copyText"])
    copyButton.setAutoDefault(False)
    copyButton.setDefault(False)
    copyContent = request["details"] if hasDetails else request["summary"]
    copyButton.clicked.connect(lambda: app.clipboard().setText(copyContent))
    buttons.addWidget(copyButton)

cancelButton.clicked.connect(dialog.reject)
confirmButton.clicked.connect(dialog.accept)

buttons.addStretch(1)
buttons.addWidget(cancelButton)
buttons.addWidget(confirmButton)
layout.addLayout(buttons)

dialog.resize(900, 650) if hasDetails else dialog.resize(560, 200)

# Keep the initial dialog within the available desktop area.
screen = app.primaryScreen()
if screen is not None:
    available = screen.availableGeometry()
    dialog.resize(
        min(dialog.width(), max(1, available.width() - 40)),
        min(dialog.height(), max(1, available.height() - 60)),
    )
    dialog.move(available.center() - dialog.rect().center())

cancelButton.setFocus()
confirmed = dialog.exec() == QDialog.DialogCode.Accepted

# Return explicit confirmation bound to the request that was displayed.
response = {"confirmed": confirmed, "requestHash": requestHash}
Path(sys.argv[2]).write_text(json.dumps(response), encoding="utf-8")
'''

        try:
            with tempfile.TemporaryDirectory(prefix="pgl-confirm-") as directory:
                root = Path(directory)
                inputFile = root / "request.json"
                outputFile = root / "response.json"
                inputFile.write_bytes(requestText.encode("utf-8"))

                # There is no interaction timeout: users may need time to
                # review or copy details. subprocess.run cleans up its
                # child if execution is interrupted.
                result = subprocess.run(
                    [sys.executable, "-c", dialogScript, str(inputFile), str(outputFile)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )

                if result.returncode != 0:
                    errorDetails = result.stderr.strip()[-4000:] or "No diagnostic output."
                    raise RuntimeError(f"Dialog process exited with code {result.returncode}:\n{errorDetails}")

                if not outputFile.is_file():
                    raise RuntimeError("Dialog closed without a confirmation response.")

                response = json.loads(outputFile.read_text(encoding="utf-8"))

                if (
                    not isinstance(response, dict)
                    or set(response) != {"confirmed", "requestHash"}
                    or type(response["confirmed"]) is not bool
                    or response["requestHash"] != requestHash
                ):
                    raise ValueError("Invalid or mismatched confirmation response.")

                return response["confirmed"]

        except Exception as error:
            # Fail closed. Never interpret a launch or decoding failure as
            # consent, and do not silently switch confirmation interfaces.
            pglMessages.warning(f"Confirmation was not obtained: {type(error).__name__}: {error}", level=1)
            return False
    
    ################################################################
    # Select from a list of strings
    ################################################################
    @staticmethod
    def selectItems(items, title="Select items"):
        """Display checkboxes and return zero-based selected positions.

        items is a nonempty list of strings. Duplicate labels are allowed.
        Nothing is selected initially; at least one selection is required.

        Returns selected indices in list order, or None on Cancel/failure.
        Qt runs in a separate process to avoid notebook conflicts.
        """
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        if not isinstance(items, list) or not items or not all(isinstance(item, str) for item in items):
            raise ValueError("items must be a nonempty list of strings.")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a nonblank string.")

        # Pass display strings through JSON, not Python source interpolation.
        dialogScript = r'''
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

app = QApplication([])
dialog = QDialog()
dialog.setWindowTitle(request["title"])
dialog.setSizeGripEnabled(True)
dialog.setStyleSheet("""
    QDialog { background-color: #1e1f22; }
    QListWidget {
        background-color: #2b2d31;
        color: #eaecef;
        border: 1px solid #3a3d42;
        border-radius: 6px;
        padding: 6px;
    }
    QListWidget::item { padding: 6px; }
    QPushButton {
        background-color: #34373c;
        color: #eaecef;
        border: 1px solid #3a3d42;
        border-radius: 6px;
        padding: 7px 16px;
    }
    QPushButton:hover { background-color: #3f4247; }
    QPushButton:disabled { color: #777777; }
""")

layout = QVBoxLayout(dialog)
itemList = QListWidget()
layout.addWidget(itemList, 1)

for label in request["items"]:
    item = QListWidgetItem(label)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(Qt.CheckState.Unchecked)
    itemList.addItem(item)

buttonRow = QHBoxLayout()
selectAllButton = QPushButton("Select all")
selectNoneButton = QPushButton("Select none")
buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
okButton = buttonBox.button(QDialogButtonBox.StandardButton.Ok)

for button in [selectAllButton, selectNoneButton] + buttonBox.buttons():
    button.setAutoDefault(False)

def selectedIndices():
    return [
        index for index in range(itemList.count())
        if itemList.item(index).checkState() == Qt.CheckState.Checked
    ]

def setSelection(checked):
    state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
    for index in range(itemList.count()):
        itemList.item(index).setCheckState(state)

selectAllButton.clicked.connect(lambda: setSelection(True))
selectNoneButton.clicked.connect(lambda: setSelection(False))
itemList.itemChanged.connect(lambda item: okButton.setEnabled(bool(selectedIndices())))
buttonBox.accepted.connect(dialog.accept)
buttonBox.rejected.connect(dialog.reject)
okButton.setEnabled(False)

buttonRow.addWidget(selectAllButton)
buttonRow.addWidget(selectNoneButton)
buttonRow.addStretch(1)
buttonRow.addWidget(buttonBox)
layout.addLayout(buttonRow)

dialog.resize(720, 450)
screen = app.primaryScreen()
if screen is not None:
    available = screen.availableGeometry()
    dialog.resize(min(dialog.width(), max(1, available.width() - 40)), min(dialog.height(), max(1, available.height() - 60)))
    dialog.move(available.center() - dialog.rect().center())

selection = selectedIndices() if dialog.exec() == QDialog.DialogCode.Accepted else None
Path(sys.argv[2]).write_text(json.dumps(selection), encoding="utf-8")
'''

        try:
            with tempfile.TemporaryDirectory(prefix="pgl-select-") as directory:
                root = Path(directory)
                inputFile = root / "request.json"
                outputFile = root / "response.json"
                inputFile.write_text(json.dumps({"title": title, "items": items}, ensure_ascii=False), encoding="utf-8")

                # No interaction timeout: the user may need time to choose.
                result = subprocess.run(
                    [sys.executable, "-c", dialogScript, str(inputFile), str(outputFile)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                if result.returncode != 0:
                    details = result.stderr.strip()[-4000:] or "No diagnostic output."
                    raise RuntimeError(f"Selection dialog exited with code {result.returncode}:\n{details}")
                if not outputFile.is_file(): raise RuntimeError("Selection dialog returned no response.")

                selected = json.loads(outputFile.read_text(encoding="utf-8"))
                if selected is None: return None
                if (
                    not isinstance(selected, list) or not selected
                    or any(type(index) is not int or not 0 <= index < len(items) for index in selected)
                    or len(set(selected)) != len(selected)
                ):
                    raise ValueError("Invalid selection response.")

                return sorted(selected)

        except Exception as error:
            pglMessages.warning(f"Selection was not obtained: {type(error).__name__}: {error}", level=1)
            return None
        
################################################################
#   pglToolDialog.py
#    purpose: Build existing PGL traits dialogs from tool field schemas
################################################################
class pglToolDialog:
    """Portable tool-editor schema backed by the existing traits GUI.

    JSON contains field descriptions and values, not executable class names.
    Only allowlisted field types are converted to traits.

    Runtime editor classes exist only within the current process. They never
    pass through pglSerialize or require public export registration.

    One editor handles one tool. Installation profiles share that tool's
    field layout, as required by the existing settings-list widget.
    """

    @staticmethod
    def validate(document):
        """Validate and copy a tool-editor document without opening a GUI."""
        import re
        from .pglSettings import pglTraitSettings

        if type(document) is not dict or set(document) != {
            "schemaVersion", "toolName", "label", "multipleInstallations",
            "fields", "profiles", "defaultInstallation",
        }:
            raise ValueError("Invalid tool-editor document fields.")

        if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1:
            raise ValueError("Unsupported tool-editor schema.")
        if type(document["multipleInstallations"]) is not bool:
            raise TypeError("multipleInstallations must be Boolean.")

        def requireText(value, label):
            if (
                not isinstance(value, str) or not value.strip()
                or any(character in value for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError(f"{label} must be a nonblank string without NUL or newlines.")

        requireText(document["toolName"], "toolName")
        requireText(document["label"], "label")

        fields = document["fields"]
        if type(fields) is not dict or not fields:
            raise ValueError("Tool fields must be a nonempty dictionary.")

        reserved = set(pglTraitSettings.class_traits()) | set(dir(pglTraitSettings))
        reserved |= {"profiles", "configNames"}

        def validateValue(value, definition, label):
            if definition["type"] == "bool":
                if type(value) is not bool:
                    raise TypeError(f"{label} must be Boolean.")
            else:
                if not isinstance(value, str):
                    raise TypeError(f"{label} must be a string.")
                if any(character in value for character in ("\x00", "\n", "\r")):
                    raise ValueError(f"{label} must not contain NUL or newlines.")

        for name, definition in fields.items():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
                or name in reserved
            ):
                raise ValueError(f"Invalid or reserved tool field name: {name!r}")

            requiredKeys = {"type", "default", "label", "help"}
            actionKeys = {"hasSetButton", "buttonLabel", "buttonFunction"}
            if (
                type(definition) is not dict
                or not requiredKeys <= set(definition)
                or set(definition) - requiredKeys - actionKeys
            ):
                raise ValueError(f"Invalid definition for tool field {name!r}.")
            suppliedActionKeys = set(definition) & actionKeys
            if suppliedActionKeys:
                if (
                    suppliedActionKeys != actionKeys
                    or definition["hasSetButton"] is not True
                    or definition["type"] not in ("text", "path")
                ):
                    raise ValueError(f"Invalid action-button metadata for {name!r}.")
                requireText(definition["buttonLabel"], "Button label")
                requireText(definition["buttonFunction"], "Button function")
                # Portable schemas select an explicitly supported action;
                # they cannot name an arbitrary method to execute.
                if (
                    document["toolName"], name, definition["buttonFunction"]
                ) != ("matlab", "executable", "discoverInstallation"):
                    raise ValueError("Unsupported tool-field action.")
            if definition["type"] not in ("text", "path", "bool"):
                raise ValueError(f"Unsupported tool field type: {definition['type']!r}")

            requireText(definition["label"], f"{name} label")
            if not isinstance(definition["help"], str):
                raise TypeError("Field help must be a string.")

            validateValue(definition["default"], definition, f"{name} default")

        profiles = document["profiles"]
        if type(profiles) is not list or not profiles:
            raise ValueError("Supply at least one installation profile.")
        if not document["multipleInstallations"] and len(profiles) != 1:
            raise ValueError("This tool supports only one configuration.")

        names = []
        for profile in profiles:
            if type(profile) is not dict or set(profile) != {"name", "values"}:
                raise ValueError("A profile requires name and values.")

            requireText(profile["name"], "Installation name")
            names.append(profile["name"])

            values = profile["values"]
            if type(values) is not dict or set(values) != set(fields):
                raise ValueError("Profile values must exactly match declared fields.")

            for name, definition in fields.items():
                validateValue(values[name], definition, name)

        if len(set(names)) != len(names):
            raise ValueError("Installation names must be unique.")

        defaultName = document["defaultInstallation"]
        if not isinstance(defaultName, str) or defaultName not in names:
            raise ValueError("defaultInstallation must identify one profile.")

        return deepcopy(document)

    @classmethod
    def build(cls, document):
        """Build the existing traits editor from a validated tool schema.

        Multiple installations:
            Show the installation selector, editable names, default checkbox,
            and New/Copy/Delete controls.

        Single configuration:
            Show configuration fields directly, without a profile selector,
            name, default checkbox, or profile-management controls.

        Runtime editor classes remain local to this process. Only the plain
        document returned by extract() crosses the process boundary.
        """
        from traitlets import Bool, Instance, List, Unicode

        from .pglSettings import pglTraitSettings

        document = cls.validate(document)
        multiple = document["multipleInstallations"]

        traits = {
            "__module__": __name__,
            "name": Unicode(
                "New installation",
                visible=multiple,
                enabled=multiple,
                help="Unique name for this installation profile.",
            ),
        }

        for name, definition in document["fields"].items():
            options = {
                "help": definition["help"],
                "traitDisplayName": definition["label"],
            }

            if definition["type"] == "bool":
                trait = Bool(definition["default"], **options)
            else:
                trait = Unicode(definition["default"], **options)
                if definition["type"] == "path":
                    matlabInstallation = (
                        document["toolName"] == "matlab" and name == "executable"
                    )
                    trait = trait.tag(
                        isPath=True,
                        pathType=(
                            "file" if name == "executable" and not matlabInstallation
                            else "directory"
                        ),
                        selectApplicationBundle=matlabInstallation,
                    )

            if definition.get("hasSetButton", False):
                trait = trait.tag(
                    hasSetButton=True,
                    buttonLabel=definition["buttonLabel"],
                    buttonFunction=definition["buttonFunction"],
                )

            # Advanced installation preference: retain the value without displaying it.
            if document["toolName"] == "glmsingle" and name == "editableInstall":
                trait = trait.tag(visible=False)

            traits[name] = trait

        # Build from allowlisted traits, not executable source or class names
        # supplied by the document.
        if document["toolName"] == "matlab":
            # A known callback, not a dynamic import/eval from schema data.
            def discoverInstallation(self):
                return cls._discoverMatlabInstallation(self)

            traits["discoverInstallation"] = discoverInstallation

        profileClass = type("pglToolProfileEditor", (pglTraitSettings,), traits)

        profiles = []
        for record in document["profiles"]:
            profile = profileClass()
            profile.name = record["name"]
            profile.isDefault = record["name"] == document["defaultInstallation"]

            for name, value in record["values"].items():
                setattr(profile, name, value)

            profiles.append(profile)

        if not multiple:
            # No List trait means no configuration-name selector is rendered.
            # Keep the internal name solely for document round-trip identity.
            return cls._attachValidationButton(document, profiles[0])

        profileTrait = List(
            Instance(profileClass),
            settingsListKey="name",
            buttons=True,
            setDefault=True,
            hideKey=False,
            traitDisplayName="Installation",
            help="Select the installation to edit. The default is explicitly marked.",
        )
        editorClass = type(
            "pglToolEditor",
            (pglTraitSettings,),
            {
                "__module__": __name__,
                "profiles": profileTrait,
            },
        )

        # Display the default first. Selecting a row for editing does not itself
        # change the explicitly marked default installation.
        profiles.sort(key=lambda profile: not profile.isDefault)
        editor = editorClass()
        editor.profiles = profiles
        return cls._attachValidationButton(document, editor)

    @classmethod
    def extract(cls, document, editor):
        """Return edited values without serializing runtime editor classes.

        Single-configuration tools retain their internal profile identity
        without displaying or editing that identity in the GUI.

        Multiple-installation tools return the edited profile names and require
        exactly one explicitly selected default.
        """
        document = cls.validate(document)

        if not document["multipleInstallations"]:
            originalName = document["profiles"][0]["name"]

            edited = {
                **document,
                "profiles": [{
                    "name": originalName,
                    "values": {
                        name: getattr(editor, name)
                        for name in document["fields"]
                    },
                }],
                "defaultInstallation": originalName,
            }
            return cls.validate(edited)

        profiles = list(editor.profiles)
        defaults = [profile.name for profile in profiles if profile.isDefault]

        if len(defaults) != 1:
            raise ValueError("Select exactly one default installation.")

        edited = {
            **document,
            "profiles": [
                {
                    "name": profile.name,
                    "values": {
                        name: getattr(profile, name)
                        for name in document["fields"]
                    },
                }
                for profile in profiles
            ],
            "defaultInstallation": defaults[0],
        }
        return cls.validate(edited)
    @classmethod
    def _selectedValidationDocument(cls, document, editor):
        """Snapshot only the currently displayed profile's edited values.

        The existing settings-list widget moves the displayed profile to
        profiles[0]. isDefault does not select what this button validates.

        Other profiles are neither read nor validated here.
        """
        if document["multipleInstallations"]:
            if not editor.profiles:
                raise ValueError("No displayed installation profile.")
            profile = editor.profiles[0]
            profileName = profile.name
        else:
            profile = editor
            profileName = document["profiles"][0]["name"]

        snapshot = {
            **document,
            "profiles": [{
                "name": profileName,
                "values": {
                    name: getattr(profile, name) for name in document["fields"]
                },
            }],
            # This is only the worker's one-profile snapshot. It does not
            # change the real editor's or saved configuration's default.
            "defaultInstallation": profileName,
        }
        return cls.validate(snapshot)

    @classmethod
    def _attachValidationButton(cls, document, editor):
        """Attach the ordinary settings.buttons callback; no GUI code here."""
        document = cls.validate(document)
        if document["toolName"] not in ("afni", "matlab", "glmsingle"):
            return editor

        def validateConfiguration(self):
            cls._validateEditor(document, self)

        type(editor).buttons = [("Validate", "validateConfiguration")]
        type(editor).validateConfiguration = validateConfiguration
        return editor

    @classmethod
    def _validateEditor(cls, document, editor):
        """Check only the displayed profile using a separate worker process.

        Follows the existing synchronous settings-button convention: the
        configuration dialog waits until completion. Worker logs are retained.

        Each probe has a 120-second subprocess deadline, not a hard deadline
        for all environment capture/hashing. MATLAB may execute startup code
        and check out a license. No dependency installation occurs.

        Records stay on the copied root editor. They are published only after
        OK saves matching profile names/values. Cancel publishes nothing.
        """
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        from .pglMessages import pglMessages

        workingDirectory = None
        owner = getattr(editor, "_dialog", None)

        try:
            snapshot = cls._selectedValidationDocument(document, editor)
            selected = snapshot["profiles"][0]
            workingDirectory = Path(tempfile.mkdtemp(prefix="pgl-tool-validation-"))
            requestPath = workingDirectory / "request.json"
            responsePath = workingDirectory / "response.json"

            requestPath.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            pglMessages.print(
                f"Validating {snapshot['label']} / {selected['name']} only.\n"
                f"Retained diagnostics: {workingDirectory}"
            )

            workerCode = r"""
import importlib
import json
import sys
from pathlib import Path

packageDirectory = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(packageDirectory.parent))
module = importlib.import_module("pgl.pglExternalTools")
if Path(module.__file__).resolve() != packageDirectory / "pglExternalTools.py":
    raise RuntimeError("Validation worker imported a different PGL checkout.")

base = module.pglExternalTool
requestPath = Path(sys.argv[2])
responsePath = Path(sys.argv[3])
with requestPath.open("rb") as source:
    content = source.read(1024 * 1024 + 1)
if len(content) > 1024 * 1024:
    raise ValueError("Validation request exceeds 1 MiB.")
document = module._copyProvenance(json.loads(content))

if type(document) is not dict:
    raise ValueError("Invalid tool document.")
toolName = document.get("toolName")
if toolName not in ("afni", "matlab", "glmsingle"):
    raise ValueError("Unsupported validation adapter.")
adapter = base.registeredTools()[toolName]

profiles = document.get("profiles")
if type(profiles) is not list or len(profiles) != 1:
    raise ValueError("Validate must receive exactly one displayed profile.")
profile = profiles[0]
if type(profile) is not dict or set(profile) != {"name", "values"}:
    raise ValueError("Invalid validation profile.")
name = profile["name"]
if (
    not isinstance(name, str) or not name.strip()
    or any(character in name for character in ("\x00", "\n", "\r"))
):
    raise ValueError("Invalid profile name.")
values = profile["values"]
if type(values) is not dict or set(values) != set(adapter.configurationFields):
    raise ValueError("Invalid validation fields.")
adapter._settingsFromValues(values)
if document.get("defaultInstallation") != name:
    raise ValueError("Invalid one-profile validation snapshot.")

configPath = requestPath.parent / "worker-tools.json"
base._writeToolSettings(configPath, {
    "schemaVersion": 2,
    "tools": {
        toolName: {
            "defaultInstallation": name,
            "installations": {name: values},
        },
    },
})

# Even temporary adapter constructors use this explicit snapshot, never the
# ordinary tools.json. A blank field retains its requested discovery policy.
base._toolSettingsPath = staticmethod(
    lambda configFile=None: configPath if configFile is None
    else Path(configFile).expanduser().absolute()
)

if toolName == "matlab":
    requested = values["executable"]
    if requested.strip():
        base._printConfigurationProgress(
            "Validating MATLAB installation: " + requested
        )
    else:
        base._printConfigurationProgress("Discovering MATLAB for validation…")

    try:
        resolved = adapter._findExecutable(requested if requested.strip() else None)
    except Exception as error:
        base._printConfigurationProgress(
            "MATLAB executable could not be resolved: "
            + type(error).__name__ + ": " + str(error)
        )
    else:
        base._printConfigurationProgress(
            "Validating MATLAB executable: " + str(resolved)
        )
else:
    base._printConfigurationProgress(
        "Validating " + (adapter.toolLabel or toolName)
        + " configuration: " + name
        + "\\n" + json.dumps(values, ensure_ascii=False)
    )

adapter.validate(
    timeout=120.0,
    cacheDirectory=requestPath.parent / "probe-logs",
    configFile=configPath,
    printOutput=False,
)

cachePath = configPath.with_name("worker-tools.status.json")
cache = json.loads(cachePath.read_text(encoding="utf-8"))
record = {
    "toolName": toolName,
    "installation": name,
    **cache["tools"][toolName][name],
}
with responsePath.open("x", encoding="utf-8") as target:
    json.dump(
        {"schemaVersion": 1, "records": [record]},
        target, ensure_ascii=False, indent=2, allow_nan=False,
    )
"""
            stdoutPath = workingDirectory / "stdout.log"
            stderrPath = workingDirectory / "stderr.log"
            with stdoutPath.open("xb") as stdout, stderrPath.open("xb") as stderr:
                result = subprocess.run(
                    [
                        sys.executable, "-c", workerCode,
                        str(Path(__file__).resolve().parent),
                        str(requestPath), str(responsePath),
                    ],
                    cwd=str(workingDirectory),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )

            if result.returncode != 0:
                with stderrPath.open("rb") as source:
                    source.seek(max(0, stderrPath.stat().st_size - 8000))
                    details = source.read(8000).decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Validation worker exited with code {result.returncode}.\n"
                    + (details.strip() or "See retained worker logs.")
                )

            with responsePath.open("rb") as source:
                content = source.read(16 * 1024 * 1024 + 1)
            if len(content) > 16 * 1024 * 1024:
                raise ValueError("Validation response exceeds 16 MiB.")
            response = json.loads(content)
            if (
                type(response) is not dict
                or set(response) != {"schemaVersion", "records"}
                or type(response["schemaVersion"]) is not int
                or response["schemaVersion"] != 1
                or type(response["records"]) is not list
                or len(response["records"]) != 1
            ):
                raise ValueError("Expected exactly one validation result.")

            record = response["records"][0]
            if (
                record["toolName"] != snapshot["toolName"]
                or record["installation"] != selected["name"]
                or record["context"]["configuration"] != selected["values"]
            ):
                raise ValueError("Validation result does not match the displayed snapshot.")

            report = record["report"]
            lines = [f"{record['installation']}: {report['status']}"]
            if report["version"]: lines.append(f"Version: {report['version']}")
            if report["location"]: lines.append(f"Location: {report['location']}")
            if report["diagnostic"]: lines.append(report["diagnostic"])

            editor._toolValidationRecords = [
                *getattr(editor, "_toolValidationRecords", []), record,
            ]

            headline = (
                "Displayed configuration passed validation."
                if report["available"] is True else
                "Displayed configuration did not pass validation."
            )
            details = (
                "\n".join(lines)
                + f"\n\nRetained diagnostics: {workingDirectory}\n\n"
                "Other profiles were not checked.\n"
                "OK publishes this result only if the saved name and values match.\n"
                "Cancel leaves saved settings/status unchanged."
            )
            pglMessages.print(headline + "\n" + details)

            message = QMessageBox(owner)
            message.setWindowTitle("Tool validation")
            message.setTextFormat(Qt.TextFormat.PlainText)
            message.setIcon(
                QMessageBox.Icon.Information if report["available"] is True
                else QMessageBox.Icon.Warning
            )
            message.setText(headline)
            message.setInformativeText(details)
            message.exec()

        except Exception as error:
            location = str(workingDirectory) if workingDirectory is not None else "(none created)"
            details = (
                f"{type(error).__name__}: {error}\n\n"
                f"Retained diagnostics: {location}"
            )
            pglMessages.warning(f"Tool validation could not be completed:\n{details}")
            message = QMessageBox(owner)
            message.setWindowTitle("Tool validation failed")
            message.setTextFormat(Qt.TextFormat.PlainText)
            message.setIcon(QMessageBox.Icon.Warning)
            message.setText("Tool validation could not be completed.")
            message.setInformativeText(details)
            message.exec()

    @staticmethod
    def _matlabInstallationSuggestion(location):
        """Convert a resolved macOS bundle executable to its installation.

        Preserve other executable locations, including unusual layouts.
        This is a display suggestion, not a runtime availability check.
        """
        from pathlib import Path

        if not isinstance(location, str) or not location.strip():
            return None

        path = Path(location)
        if not path.is_absolute():
            return None

        if (
            path.name == "matlab"
            and path.parent.name == "bin"
            and path.parent.parent.name.lower().endswith(".app")
        ):
            return str(path.parent.parent)

        return str(path)

    @classmethod
    def _fillMatlabInstallation(cls, document, editor, record):
        """Fill only the displayed blank MATLAB field after successful discovery.

        Returns True when the editor and visible widget were updated.
        The caller must then validate the explicit selection. The automatic
        discovery report is not relabeled as an explicit-configuration check.

        Does not save settings, change defaults, or access other profiles.
        """
        if document["toolName"] != "matlab":
            return False
        if record["toolName"] != "matlab":
            return False
        if record["report"]["available"] is not True:
            return False

        checkedValues = record["context"]["configuration"]
        checkedExecutable = checkedValues.get("executable")
        if not isinstance(checkedExecutable, str) or checkedExecutable.strip():
            return False

        if document["multipleInstallations"]:
            if not editor.profiles: return False
            profile = editor.profiles[0]
            profileName = profile.name
        else:
            profile = editor
            profileName = document["profiles"][0]["name"]

        if profileName != record["installation"]:
            return False

        currentValues = {
            name: getattr(profile, name) for name in document["fields"]
        }
        if currentValues != checkedValues:
            return False

        suggestion = cls._matlabInstallationSuggestion(record["report"]["location"])
        if suggestion is None: return False

        # The existing text widget setter blocks edit signals. Locate the
        # displayed field before changing the model so failures cannot quietly
        # leave the text showing a different value from the actual settings.
        setter = None
        owner = getattr(editor, "_dialog", None)
        if owner is not None:
            if document["multipleInstallations"]:
                matchingStates = [
                    state for state in owner._selectedSettings.values()
                    if state.get("object") is profile
                ]
                if len(matchingStates) != 1:
                    raise RuntimeError("Could not identify the displayed MATLAB profile widget.")
                widgetKey = matchingStates[0]["key"]
            else:
                widgetKey = ("root",)

            entry = owner.traitWidgets.get(widgetKey, {}).get("executable")
            if entry is None or not callable(entry.get("setter")):
                raise RuntimeError("Could not identify the MATLAB installation editor.")
            setter = entry["setter"]

        previous = profile.executable
        profile.executable = suggestion
        try:
            if setter is not None: setter(suggestion)
        except Exception:
            profile.executable = previous
            raise

        return True

    @staticmethod
    def _discoverMatlabInstallation(profile):
        """Return a discovered installation for the existing text-action setter.

        No MATLAB process is started. Cancel/no candidates/error preserves the
        previous value. Multiple candidates require an explicit single choice.
        """
        from .pglExternalTools import pglMATLAB

        previous = profile.executable
        pglMATLAB._printConfigurationProgress("Discovering MATLAB installations…")

        try:
            candidates = pglMATLAB._discoverInstallations()
            if not candidates:
                pglMATLAB._printConfigurationProgress(
                    "No MATLAB installation found. The field was left unchanged."
                )
                pglMessages.warning(
                    "No MATLAB installation found on PATH or in /Applications. "
                    "Use Browse or enter the installation path."
                )
                return previous

            if len(candidates) == 1:
                selected = candidates[0]
            else:
                indices = pglDialogs.selectItems(
                    candidates, title="Choose one MATLAB installation",
                )
                if indices is None:
                    pglMATLAB._printConfigurationProgress(
                        "MATLAB discovery selection cancelled; field unchanged."
                    )
                    return previous
                if len(indices) != 1:
                    pglMATLAB._printConfigurationProgress(
                        "Select exactly one MATLAB installation; field unchanged."
                    )
                    pglMessages.warning("Please click Discover again and select exactly one installation.")
                    return previous
                selected = candidates[indices[0]]

            pglMATLAB._printConfigurationProgress(
                f"Discovered MATLAB installation: {selected}\n"
                "The displayed field was selected; runtime validation has not run."
            )
            return selected

        except Exception as error:
            pglMATLAB._printConfigurationProgress(
                f"MATLAB discovery failed: {type(error).__name__}: {error}\n"
                "The installation field was left unchanged."
            )
            return previous
    
    
    

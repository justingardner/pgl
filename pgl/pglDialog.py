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
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QSlider, QPushButton, QWidget, QScrollArea, QDialogButtonBox, QAbstractSpinBox,
    QStylePainter, QStyleOptionComboBox, QStyle
)
from PySide6.QtCore import Qt, QCoreApplication, QTimer
from traitlets import (
    HasTraits, Float, Int, List, Unicode, Bool, Tuple, TraitType
)
from .pglSerialize import pglSerialize
import sys, subprocess, tempfile
from pathlib import Path
from IPython.display import HTML, display
from collections import OrderedDict
import ipywidgets as widgets
from traitlets import HasTraits, Float, Int, List, TraitError, Unicode, Dict, default, link, Bool, TraitType
from functools import partial
import math
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

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

        # copy the settings so the original is untouched until OK
        self.settings = copy.deepcopy(settings)

        # give the copy a back-reference to this dialog
        self.settings._dialog = self

        # maps traitName -> {'widget', 'row', 'label'} for the trait API
        self.traitWidgets = {}

        # keep track of whether we are pushing values into widgets so that
        # we do not create feedback loops when the trait observer fires
        self._updatingWidget = False

        # dialog result flag
        self.accepted_ = False

        # window setup
        self.setWindowTitle(title)
        self.setStyleSheet(self._darkStyle())

        # build the interface
        self._buildUI()

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
    def enable(self, traitName, isEnabled=True):
        """Enable or disable the widget(s) for a trait."""
        entry = self.traitWidgets.get(traitName)
        if entry is None:
            return
        entry['row'].setEnabled(bool(isEnabled))

    def visible(self, traitName, isVisible=True):
        """Show or hide the widget(s) for a trait."""
        entry = self.traitWidgets.get(traitName)
        if entry is None:
            return
        entry['row'].setVisible(bool(isVisible))
        if entry.get('label') is not None:
            entry['label'].setVisible(bool(isVisible))

    def set(self, traitName, value):
        """
        Set the widget for a trait to a value without re-triggering the
        settings->dialog callback (guards against feedback loops).
        """
        entry = self.traitWidgets.get(traitName)
        if entry is None:
            return

        self._updatingWidget = True
        try:
            entry['setter'](value)
        finally:
            self._updatingWidget = False

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
            if traitName.startswith('_'):
                continue
            self._addTraitWidget(traitName, trait)

        # Shared matplotlib axis for any plot-button traits
        self.figure = Figure(figsize=(5, 3))
        self.plotAxis = self.figure.add_subplot(111)
        self.plotCanvas = ScrollableFigureCanvas(self.figure)
        self.plotCanvas.setMinimumHeight(680)
        self.plotCanvas.setVisible(False)
        self.formLayout.addRow(self.plotCanvas)

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
        maxDialogHeight = 760

        # Height needed for the form contents
        formHeight = formWidget.sizeHint().height()

        # Add the button bar and layout margins
        buttonHeight = buttonBar.sizeHint().height()
        extra = mainLayout.contentsMargins().top() + mainLayout.contentsMargins().bottom()

        desiredHeight = formHeight + buttonHeight + extra

        self.resize(680, min(desiredHeight, maxDialogHeight))
        
        # give hint for window to stay on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        
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
        current = getattr(settingsObject, traitName)

        # a tuple
        if not trait.metadata.get('visible',True):
            return
        
        if isinstance(trait, Tuple):
            self._addTuple(traitName, trait, current, helpText, settingsObject, layout, settingsKey)
        
        # a settings list
        elif isinstance(trait, List) and "settingsListKey" in trait.metadata:
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
            
        # Path
        elif isinstance(trait, Unicode) and trait.metadata.get("isPath", False):
            self._addText(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # Unicode with button
        elif isinstance(trait, Unicode) and trait.metadata.get("hasSetButton", False):
            self._addTextWithSetButton(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # Unicode
        elif isinstance(trait, Unicode):
            self._addText(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # List with a plot button
        elif isinstance(trait, List) and trait.metadata.get("hasPlotButton", False):
            self._addListWithPlotButton(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

        # List -> dropdown
        elif isinstance(trait, List):
            self._addList(traitName, trait, current, helpText, settingsObject, layout, settingsKey)

    # ----- Setting list -----
    _selectedSettings = {}
    def _addSettingsList(self, traitName, trait, current, helpText,
                        settingsObject, layout=None, settingsKey=None):

        # Build widgets once
        #--------------------
        def buildRows():
            for name, childTrait in self._getOrderedTraits(current[0]).items():
                if name.startswith("_"):
                    continue

                if hideKey and name == keyTraitName:
                    continue
                
                if hideAll:
                    continue

                self._addTraitWidget(name, childTrait, proxy, layout, settingsKey)
                childNames.append(name)

        # Commit current selection
        # this function will reorder the list so that the selection is at top
        #--------------------------
        def commitSelection(updateCombo=True):
            lst = state["list"]
            obj = state["object"]

            if len(lst) <= 1:
                return

            # sort the remaining values
            remaining = [x for x in lst if x is not obj]
            remaining.sort(
                key=lambda x: str(getattr(x, keyTraitName))
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
                    value = getattr(obj, name)
                    entry = self.traitWidgets[state["key"]].get(name)

                    # ordinary widget
                    if entry is not None and "setter" in entry:
                        entry["setter"](value)

                    # nested settings list
                    nestedKey = (obj.__class__.__name__,name)
                    nested = self._selectedSettings.get(nestedKey)

                    if nested is not None:
                        nested["retargetList"](value)

            finally:
                self._updatingWidget = False

        # User changed combo selection
        #------------------------------
        def showObject(index):

            # first commit selection order
            state["object"] = state["list"][index]
            commitSelection()
            obj = state["list"][0]
            state["object"] = obj
            
            # retarget and update all the fields
            proxy.retarget(obj)
            updateFields(obj)
            if hasattr(self, "plotCanvas"):
                self.plotCanvas.setVisible(False)
                self.plotCanvas.draw()
        
        # Updates the selection combo when th keyTrait changes
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

        # get metadata settings
        keyTraitName = trait.metadata["settingsListKey"]
        hideKey = trait.metadata.get("hideKey", False)
        hideAll = trait.metadata.get("hideAll", False)
        highlightSelector = trait.metadata.get("highlightSelector", True)

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

        # make sure the order of the list is selected on top
        # and the rest alphabetical
        commitSelection(updateCombo=False)
        
        # make the combo box which selects the list
        combo = CenteredComboBox()
        if highlightSelector: combo.setObjectName("settingsSelector")
        combo.addItems([str(getattr(x, keyTraitName)) for x in current])

        # register the combo
        self._register(traitName, trait, combo, lambda v: None, layout)
        
        # set what the widget is updating
        proxy = _RetargetableProxy(current[0])
        childNames = []
        
        state["retargetList"] = retargetList

        # build the widgets
        if not childNames: buildRows()
        
        # observe any changes to keyTraitName, so that we can update the combo
        for obj in current:
            obj.observe(keyChanged, names=keyTraitName)

        # connect showObject to changes in the index
        combo.currentIndexChanged.connect(showObject)

        # and show the first item in the list
        showObject(0)
        
    # ----- Float with min/max -----
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

    # ----- Float -----
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

    # ----- Int -----
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

    # ----- Tuple -----
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
                label.setObjectName("tupleLabel")
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
    # ----- Bool -----
    def _addBool(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        check = QCheckBox()
        check.setChecked(bool(current))
        check.setToolTip(helpText)

        def onChange(state):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, check.isChecked())

        check.stateChanged.connect(onChange)
        self._register(traitName, trait, check, lambda v: check.setChecked(bool(v)), layout, settingsKey)

    # ----- Text / Path -----
    def _addText(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):
        edit = QLineEdit(str(current) if current is not None else "")
        edit.setAlignment(Qt.AlignCenter) 
        edit.setToolTip(helpText)

        def onChange(text):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, text)

        edit.textChanged.connect(onChange)
        self._register(traitName, trait, edit, lambda v: edit.setText(str(v) if v is not None else ""), layout, settingsKey)

    # ----- List -> dropdown -----
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
        
    # ----- List with toggle plot button -----
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
            # display plot
            if self._activePlotButton is not None:
                self._activePlotButton.setChecked(False)
            self._activePlotButton = button
            self.plotButtonState = True
            button.setChecked(True)
            updatePlot()
            
            # move selected to top, like onListSelect did
            selected = combo.itemText(index)
            opts = [combo.itemText(i) for i in range(combo.count())]
            newList = [selected] + [x for x in opts if x != selected]
            self._commit(settingsObject, traitName, newList)

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

    # ----- Text with a set button -----
    def _addTextWithSetButton(self, traitName, trait, current, helpText, settingsObject, layout=None, settingsKey=None):

        buttonFunc = trait.metadata.get("buttonFunction", None)

        edit = QLineEdit(str(current) if current is not None else "")
        edit.setAlignment(Qt.AlignCenter) 
        edit.setToolTip(helpText)

        button = QPushButton(trait.metadata.get("buttonLabel", "Set"))
        button.setToolTip(helpText)
        button.setCheckable(True)
        
        def onButtonClicked():
            if buttonFunc is None:
                return
            method = getattr(settingsObject, buttonFunc, None)
            if method is None:
                return
            edit.setText(method())


        def onChange(text):
            if not self._updatingWidget:
                self._commit(settingsObject, traitName, text)

        edit.textChanged.connect(onChange)
        button.clicked.connect(onButtonClicked)
        
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(button)

        self._register(traitName, trait, row, lambda v: edit.setText(str(v) if v is not None else ""), layout, settingsKey)


    # ----- RGB -----
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
    def _register(self, traitName, trait, widget, setter, layout = None, settingsKey = None):
        if layout is None:
            layout = self.formLayout

        traitLabel = trait.metadata["traitDisplayName"] if "traitDisplayName" in trait.metadata else traitName
        label = QLabel(traitLabel)
        label.setObjectName("traitLabel")
        #label.setMinimumWidth(180)          # consistent label column
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # let fields expand to fill the row
        from PySide6.QtWidgets import QSizePolicy
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        widget.setMinimumHeight(30)
        
        if settingsKey is None:
            settingsKey = ("root",)

        self.traitWidgets.setdefault(settingsKey, {})

        self.traitWidgets[settingsKey][traitName] = {
            'widget': widget,
            'row': widget,
            'label': label,
            'setter': setter,
        }

        layout.addRow(label, widget)
                
        # honor default-enabled metadata
        if trait is not None:
            isEnabled = trait.metadata.get('enabled', True)
            widget.setEnabled(bool(isEnabled))

    def _commit(self, settingsObject, traitName, value):
        """Push a widget change into the settings copy."""
        try:
            setattr(settingsObject, traitName, value)
            print(f"_commit: {traitName} -> {value} on {getattr(settingsObject, '_target', settingsObject)}")
        except Exception as e:
            # keep the dialog alive on a bad value
            print(f"(pglTraitsDialog:_commit) Could not set {traitName}: {e}")
            
    def _onOk(self):       
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
        #traitLabel, #tupleLabel {
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
        """
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
#####################################################################
# pglTraitsDialog: what gets called by the user. This rund
# pglTraitsDialogStandalone which runs outside the jupyter notebook
# to avoid crashy-conflicty behavior.
#####################################################################
class pglDialogs:
    @staticmethod
    def traitsDialog(settings):
        """
        Pops up a PySide6 dialog in a separate process, blocks until closed,
        and returns edited settings (OK) or None (Cancel).
        """
        tmpDir  = Path(tempfile.mkdtemp())
        inFile  = tmpDir / "in.json"
        outFile = tmpDir / "out.json"

        settings.save(inFile)

        # run, redirecting stderr because it produces meaningless messages from text handling
        scriptPath = Path(__file__).parent / "pglTraitsDialogStandalone.py"
        result = subprocess.run(
            [sys.executable, str(scriptPath), str(inFile), str(outFile)],
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0 and outFile.exists():
            # OK
            return pglSerialize.load(outFile)
        else:
            # Cancel
            return None                               

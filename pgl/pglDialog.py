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
    QSlider, QPushButton, QWidget, QScrollArea, QDialogButtonBox, QAbstractSpinBox
)
from PySide6.QtCore import Qt
from traitlets import (
    HasTraits, Float, Int, List, Unicode, Bool, TraitType
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
        self.plotCanvas = FigureCanvasQTAgg(self.figure)
        self.plotCanvas.setMinimumHeight(680)
        self.plotCanvas.setVisible(False)
        self.formLayout.addRow(self.plotCanvas)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(formWidget)

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttonBox.accepted.connect(self._onOk)
        buttonBox.rejected.connect(self._onCancel)

        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.setSpacing(0)
        mainLayout.addWidget(scroll)
        
        buttonBar = QWidget()
        buttonBar.setObjectName("buttonBar")
        bl = QHBoxLayout(buttonBar)
        bl.setContentsMargins(24, 8, 24, 8)
        bl.addStretch(1)
        bl.addWidget(buttonBox)
        mainLayout.addWidget(buttonBar)

        # bigger default window
        self.setMinimumWidth(560)
        self.adjustSize()                       # size to content
        h = min(self.sizeHint().height(), 760)  # cap tall forms
        self.resize(680, h)
    def _getOrderedTraits(self):
        """Return traits in class definition order (like getOrderedTraits)."""
        from collections import OrderedDict
        ordered = OrderedDict()
        # walk the MRO so subclass traits keep their definition order
        for cls in reversed(type(self.settings).__mro__):
            for name, obj in cls.__dict__.items():
                if isinstance(obj, TraitType):
                    ordered[name] = obj
        return ordered

    def _helpText(self, traitName, trait):
        return getattr(trait, 'help', "") or ""

    #########################################
    # Widget factory per trait type
    #########################################
    def _addTraitWidget(self, traitName, trait):
        helpText = self._helpText(traitName, trait)
        current = getattr(self.settings, traitName)

        # Float with min and max -> slider + spinbox
        if isinstance(trait, Float) and trait.min is not None and not math.isinf(trait.max) and not math.isinf(trait.min):
            self._addFloatRange(traitName, trait, current, helpText)

        # Float (min only or unbounded)
        elif isinstance(trait, Float):
            self._addFloat(traitName, trait, current, helpText)

        # Int
        elif isinstance(trait, Int):
            self._addInt(traitName, trait, current, helpText)

        # Bool
        elif isinstance(trait, Bool):
            self._addBool(traitName, trait, current, helpText)

        # RGB list
        elif isinstance(trait, List) and trait.metadata.get("isRGB", False):
            self._addRGB(traitName, trait, current, helpText)
            
        # Path
        elif isinstance(trait, Unicode) and trait.metadata.get("isPath", False):
            self._addText(traitName, trait, current, helpText)

        # Unicode
        elif isinstance(trait, Unicode):
            self._addText(traitName, trait, current, helpText)

        # List with a plot button
        elif isinstance(trait, List) and trait.metadata.get("hasPlotButton", False):
            self._addListWithPlotButton(traitName, trait, current, helpText)

        # List -> dropdown
        elif isinstance(trait, List):
            self._addList(traitName, trait, current, helpText)

    # ----- Float with min/max -----
    def _addFloatRange(self, traitName, trait, current, helpText):
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
            self._commit(traitName, v)

        def onSlider(v):
            if self._updatingWidget:
                return
            fv = fromSlider(v)
            self._updatingWidget = True
            spin.setValue(fv)
            self._updatingWidget = False
            self._commit(traitName, fv)

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

        self._register(traitName, trait, row, setter)

    # ----- Float -----
    def _addFloat(self, traitName, trait, current, helpText):
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
                self._commit(traitName, v)

        spin.valueChanged.connect(onChange)
        self._register(traitName, trait, spin, lambda v: spin.setValue(float(v)))

    # ----- Int -----
    def _addInt(self, traitName, trait, current, helpText):
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
                self._commit(traitName, int(v))

        spin.valueChanged.connect(onChange)
        row = self._wrapSpin(spin)
        self._register(traitName, trait, row, lambda v: spin.setValue(int(v)))

    # ----- Bool -----
    def _addBool(self, traitName, trait, current, helpText):
        check = QCheckBox()
        check.setChecked(bool(current))
        check.setToolTip(helpText)

        def onChange(state):
            if not self._updatingWidget:
                self._commit(traitName, check.isChecked())

        check.stateChanged.connect(onChange)
        self._register(traitName, trait, check, lambda v: check.setChecked(bool(v)))

    # ----- Text / Path -----
    def _addText(self, traitName, trait, current, helpText):
        edit = QLineEdit(str(current) if current is not None else "")
        edit.setAlignment(Qt.AlignCenter) 
        edit.setToolTip(helpText)

        def onChange(text):
            if not self._updatingWidget:
                self._commit(traitName, text)

        edit.textChanged.connect(onChange)
        self._register(traitName, trait, edit, lambda v: edit.setText(str(v) if v is not None else ""))

    # ----- List -> dropdown -----
    def _addList(self, traitName, trait, current, helpText):
        combo = QComboBox()
        options = list(current) if current else []
        combo.addItems([str(o) for o in options])
        if options:
            combo.setCurrentIndex(0)
        combo.setToolTip(helpText)

        combo.setMinimumWidth(280)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        # make the popup as wide as its widest item, and tall enough to read
        combo.view().setMinimumWidth(combo.sizeHint().width())
        combo.setMaxVisibleItems(12)
        
        def onChange(index):
            if self._updatingWidget:
                return
            selected = combo.itemText(index)
            # move selected to top, like onListSelect did
            opts = [combo.itemText(i) for i in range(combo.count())]
            newList = [selected] + [x for x in opts if x != selected]
            self._commit(traitName, newList)

        combo.currentIndexChanged.connect(onChange)

        def setter(value):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([str(o) for o in value])
            if value:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

        self._register(traitName, trait, combo, setter)

    # ----- RGB -----
    def _addRGB(self, traitName, trait, current, helpText):
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
                self._commit(traitName, [b.value() for b in boxes])

        for b in boxes:
            b.valueChanged.connect(onChange)

        def setter(value):
            for i, b in enumerate(boxes):
                if i < len(value):
                    b.setValue(float(value[i]))

        self._register(traitName, trait, row, setter)
        
    # ----- List with dropdown + plot button -----
    def _addListWithPlotButton(self, traitName, trait, current, helpText):
        plotFunc = trait.metadata.get("buttonFunction", None)

        combo = QComboBox()
        combo.addItems([str(item) for item in current])
        combo.setToolTip(helpText)

        button = QPushButton(trait.metadata.get("buttonLabel", "Plot"))
        button.setToolTip(helpText)

        def onClick():
            if plotFunc is None:
                return
            selected = combo.currentText()
            # get the plotFunc (the metadata is a string which we need to bind to the class function)
            method = getattr(self.settings, plotFunc, None)
            if method is None:
                return
            self.figure.clear()
            method(self.figure, selected)
            self.plotCanvas.draw()
            self.plotCanvas.setVisible(True)

        button.clicked.connect(onClick)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(combo, 1)
        h.addWidget(button)

        def setter(value):
            combo.clear()
            combo.addItems([str(item) for item in value])

        self._register(traitName, trait, row, setter)

    #########################################
    # Helpers
    #########################################
    def _register(self, traitName, trait, widget, setter):
        label = QLabel(traitName)
        label.setObjectName("traitLabel")
        #label.setMinimumWidth(180)          # consistent label column
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # let fields expand to fill the row
        from PySide6.QtWidgets import QSizePolicy
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        widget.setMinimumHeight(30)

        self.formLayout.addRow(label, widget)
        self.traitWidgets[traitName] = {
            'widget': widget,
            'row': widget,
            'label': label,
            'setter': setter,
        }
        
        # honor default-enabled metadata
        if trait is not None:
            isEnabled = trait.metadata.get('enabled', True)
            widget.setEnabled(bool(isEnabled))

    def _commit(self, traitName, value):
        """Push a widget change into the settings copy."""
        try:
            setattr(self.settings, traitName, value)
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
        #traitLabel {
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
        """
    def _wrapSpin(self, spin):
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

#####################################################################
# pglTraitsDialog: what gets called by the user. This rund
# pglTraitsDialogStandalone which runs outside the jupyter notebook
# to avoid crashy-conflicty behavior.
#####################################################################
def pglTraitsDialog(settings):
    """
    Pops up a PySide6 dialog in a separate process, blocks until closed,
    and returns edited settings (OK) or None (Cancel).
    """
    tmpDir  = Path(tempfile.mkdtemp())
    inFile  = tmpDir / "in.json"
    outFile = tmpDir / "out.json"

    settings.save(inFile)

    scriptPath = Path(__file__).parent / "pglTraitsDialogStandalone.py"  # adjust path
    result = subprocess.run(
        [sys.executable, str(scriptPath), str(inFile), str(outFile)]
    )

    if result.returncode == 0 and outFile.exists():
        return pglSerialize.load(outFile)   # OK
    return None                                # Cancel

#############
# Main class which should be subclassed for specific settings,
# provides methods for loading/saving from JSON and displaying widgets
# to edit the settings
#############
class pglSettingsEditable(HasTraits, pglSerialize):
    def __init__(self, filename=None):
        # Initialize HasTraits
        super().__init__()
        # Load from file if provided
        if filename:
            #print(f"(pglSettingsEditable:init) Loading settings from '{filename}'.")
            self.updateFromFile(filename)
    
   # display parameters
    def __repr__(self):
        traitValues = ", ".join(f"{key}={getattr(self, key)!r}" for key in self.trait_names())
        return f"{self.__class__.__name__}({traitValues})"
    
    # setup CSS
    def setupDisplayStyle(self):
        # --- Dark widget CSS ---
        display(HTML("""
        <style>
        .dark-widget-card {
            background-color: #000 !important;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 16px;
        }

        .dark-widget-card .widget-label {
            color: #eaeaea !important;
        }

        .dark-widget-card input,
        .dark-widget-card textarea,
        .dark-widget-card select {
            background-color: #111 !important;
            color: #eaeaea !important;
            border: 1px solid #444 !important;
        }

        .dark-widget-card .slider {
            background-color: #222 !important;
        }

        .dark-widget-card select option {
            background-color: #111 !important;
            color: #eaeaea !important;
        }

        .help-text, .options-panel {
            background-color: #111;
            color: #eaeaea;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 8px;
        }
        </style>
        """))

    # gets the traits in the order that they are defined
    @classmethod
    def getOrderedTraits(cls):
        """Return traits defined in this class (not inherited), in definition order."""
        ordered = OrderedDict()
        for name, obj in cls.__dict__.items():
            if isinstance(obj, TraitType):
                ordered[name] = obj
        return ordered

    def makeWidgets(self):
        """Automatically create widgets for all traits in self."""
        style = {'description_width': '120px'}
        widgetRows = []
        allHelpText = ""

        # Initialize widget map
        self.widgetMap = {}

        for traitName, trait in self.getOrderedTraits().items():
            if traitName.startswith('_'):
                continue  # skip private traits
            
            helpText = getattr(trait, 'help', f"{traitName}: float value")
            if helpText:
                allHelpText += f"<b>{traitName}:</b> {helpText}<br>"
            
            # Float with min/max
            if isinstance(trait, Float) and trait.min is not None and trait.max is not None:
                traitStep = getattr(trait, 'step', (trait.max - trait.min) / 100)
                slider = widgets.FloatSlider(
                    min=trait.min, max=trait.max, step=traitStep,
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='calc(100% - 100px)'),
                    tooltip=helpText
                )
                text = widgets.BoundedFloatText(
                    min=trait.min, max=trait.max, step=traitStep,
                    layout=widgets.Layout(width='100px'),
                    tooltip=helpText
                )
                link((self, traitName), (slider, 'value'))
                link((self, traitName), (text, 'value'))
                row = widgets.HBox([slider, text])
                widgetRows.append(row)
                self.widgetMap[traitName] = row

            # Float with min only
            elif isinstance(trait, Float) and trait.min is not None:
                wFloat = widgets.BoundedFloatText(
                    description=traitName,
                    min=trait.min,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wFloat, 'value'))
                widgetRows.append(wFloat)
                self.widgetMap[traitName] = wFloat

            # Float without min/max
            elif isinstance(trait, Float):
                wFloat = widgets.FloatText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wFloat, 'value'))
                widgetRows.append(wFloat)
                self.widgetMap[traitName] = wFloat

            # Int
            elif isinstance(trait, Int):
                wInt = widgets.IntText(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wInt, 'value'))
                wInt.observe(partial(self.onIntSelect, traitName), names='value')
                widgetRows.append(wInt)
                self.widgetMap[traitName] = wInt
            # Bool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    value=getattr(self, traitName, False),
                    tooltip=helpText,
                    indent=False
                )
                wLabel = widgets.Label(value=traitName, style=style)
                wLabel.layout.width = '125px' 
                wBox = widgets.HBox([wLabel, wBool], layout=widgets.Layout(width='100%'))
                
                link((self, traitName), (wBool, 'value'))
                wBool.observe(partial(self.onBoolSelect, traitName), names='value')
                widgetRows.append(wBox)
                self.widgetMap[traitName] = wBool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText,
                    indent=False
                )
                link((self, traitName), (wBool, 'value'))
                wBool.observe(partial(self.onBoolSelect, traitName), names='value')
                widgetRows.append(wBool)
                self.widgetMap[traitName] = wBool
            # Path
            elif isinstance(trait, Unicode) and trait.metadata.get("isPath", False):
                wPath = widgets.Text(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wPath, 'value'))
                wPath.on_submit(partial(self.onPathSubmit, traitName=traitName))
                widgetRows.append(wPath)
                self.widgetMap[traitName] = wPath

            # Unicode
            elif isinstance(trait, Unicode):
                wText = widgets.Text(
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wText, 'value'))
                widgetRows.append(wText)
                self.widgetMap[traitName] = wText

            # Bool
            elif isinstance(trait, Bool):
                wBool = widgets.Checkbox(
                    description=traitName,
                    value=getattr(self, traitName),
                    tooltip=helpText
                )
                link((self, traitName), (wBool, 'value'))
                widgetRows.append(wBool)
                self.widgetMap[traitName] = wBool
            elif (isinstance(trait, List) and trait.metadata.get("isRGB", False)):
                # Create three float inputs for R, G, B
                r_input = widgets.BoundedFloatText(
                    value=getattr(self, traitName)[0] if getattr(self, traitName) else 0.0,
                    min=0.0, max=1.0, step=0.01,
                    description='R:',
                    style={'description_width': '20px'},
                    layout=widgets.Layout(width='120px'),
                    tooltip=f"{helpText} - Red channel"
                )
                g_input = widgets.BoundedFloatText(
                    value=getattr(self, traitName)[1] if getattr(self, traitName) else 0.0,
                    min=0.0, max=1.0, step=0.01,
                    description='G:',
                    style={'description_width': '20px'},
                    layout=widgets.Layout(width='120px'),
                    tooltip=f"{helpText} - Green channel"
                )
                b_input = widgets.BoundedFloatText(
                    value=getattr(self, traitName)[2] if getattr(self, traitName) else 0.0,
                    min=0.0, max=1.0, step=0.01,
                    description='B:',
                    style={'description_width': '20px'},
                    layout=widgets.Layout(width='120px'),
                    tooltip=f"{helpText} - Blue channel"
                )
                
                # Label for the RGB group
                label = widgets.Label(value=traitName, style=style)
                label.layout.width = '125px'
                
                # Combine in HBox
                rgb_box = widgets.HBox([label, r_input, g_input, b_input])
                
                # Update the trait when any input changes
                def update_rgb(change, name=traitName, inputs=(r_input, g_input, b_input)):
                    setattr(self, name, [inputs[0].value, inputs[1].value, inputs[2].value])
                
                r_input.observe(update_rgb, names='value')
                g_input.observe(update_rgb, names='value')
                b_input.observe(update_rgb, names='value')
                
                # Optional: Update inputs when trait changes externally
                def update_inputs(change, inputs=(r_input, g_input, b_input)):
                    if change['new'] and len(change['new']) == 3:
                        inputs[0].value = change['new'][0]
                        inputs[1].value = change['new'][1]
                        inputs[2].value = change['new'][2]
                
                self.observe(update_inputs, names=traitName)
                
                widgetRows.append(rgb_box)
                self.widgetMap[traitName] = rgb_box
    
            # List
            elif isinstance(trait, List):
                currentList = getattr(self, traitName)
                wDropdown = widgets.Dropdown(
                    options=currentList,
                    value=currentList[0] if currentList else None,
                    description=traitName,
                    style=style,
                    layout=widgets.Layout(width='100%'),
                    tooltip=helpText
                )
                link((self, traitName), (wDropdown, 'options'))
                wDropdown.observe(partial(self.onListSelect, traitName), names='value')
                widgetRows.append(wDropdown)
                self.widgetMap[traitName] = wDropdown

        # Help widget
        helpWidget = widgets.HTML(allHelpText)
        helpWidget.layout.display = 'none'
        helpWidget.add_class("help-text")

        helpButton = widgets.Button(
            description="Show Help",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        helpButton.on_click(partial(self.toggleHelp, helpWidget=helpWidget))

        saveButton = widgets.Button(
            description="Save settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onSave'):
            saveButton.layout.display = 'none'
        else:
            saveButton.on_click(partial(self.onSave))

        testButton = widgets.Button(
            description="Test settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onTest'):
            testButton.layout.display = 'none'
        else:
            testButton.on_click(partial(self.onTest))

        deleteButton = widgets.Button(
            description="Delete settings",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        if not hasattr(self, 'onDelete'):
            deleteButton.layout.display = 'none'
        else:
            deleteButton.on_click(partial(self.onDelete))
        
        spacer = widgets.Box(layout=widgets.Layout(width="120px"))
        
        cancelButton = widgets.Button(
            description="Cancel",
            button_style='info',
            layout=widgets.Layout(width='120px')
        )
        cancelButton.on_click(partial(self.onCancel))

        # Pack all widgets
        widgetDisplay = widgetRows + [
            widgets.HBox([
                cancelButton,
                deleteButton,
                saveButton,
                spacer,
                testButton,
                widgets.Box(layout=widgets.Layout(flex='1')),
                helpButton
            ]),
            helpWidget
        ]

        return widgetDisplay
    def onCancel(self, cancelButton):
        self.hide()
        pass
    
    def hide(self):
        """
        Hide the settings widget.
        """
        if hasattr(self, 'wrapper'):
            self.wrapper.layout.display = 'none'
        
    def toggleHelp(self, helpButton, helpWidget):
        helpWidget.layout.display = 'block' if helpWidget.layout.display == 'none' else 'none'

    def onIntSelect(self, traitName, change):
        pass
        
    def onBoolSelect(self, traitName, change):
        pass
        
    def onListSelect(self, traitName, change):
        # get the selected and currentList
        selected = change['new']
        currentList = list(getattr(self, traitName))

        # not in current list
        if selected not in currentList:
            return

        # Move selected item to top of list
        newList = [selected] + [x for x in currentList if x != selected]

        # now set to this newList
        setattr(self, traitName, newList)

    def onPathSubmit(self, textWidget, traitName):
        raw = textWidget.value
        try:
            path = Path(raw).expanduser()

            if not path.exists():
                # set the border to indicate no change
                textWidget.layout.border = '2px solid red'
            else:
                #self.dataPath = str(path)            
                textWidget.layout.border = '2px solid green'
        except Exception:
            textWidget.layout.border = "2px solid red"
            
            
    # ----- Put up edit dialog ---- #
    def edit(self):
        # setup css styles
        self.setupDisplayStyle()
        
        # make widgets for each parameter
        widgetDisplay = self.makeWidgets()
        
        # --- Container for widgets display ---
        widgetsBox = widgets.Box(
            widgetDisplay,
            layout=widgets.Layout(
                display='flex',
                flex_flow='column',
                gap='10px',
                width='100%'
            )
        )
        widgetsBox.add_class("dark-widget-card")

        # --- Centering wrapper ---
        self.wrapper = widgets.Box(
            [widgetsBox],
            layout=widgets.Layout(
                display='flex',
                justify_content='center',
                width='95%',
                margin='0 auto'
            )   
        )

        # display
        display(self.wrapper)

class confirmationPanel:
    def __init__(self, confirmMessage="Confirm?", onConfirm=None, onCancel=None):
        """
        onConfirm: function called if user clicks Yes
        onCancel: function called if user clicks No
        """
        # Message
        self.label = widgets.HTML(f"<b>{confirmMessage}</b>")

        # Yes button (green)
        self.yesButton = widgets.Button(
            description="Yes",
            button_style="success",
            layout=widgets.Layout(width="80px")
        )
        self.yesButton.on_click(self._yes_clicked)

        # No button (red)
        self.noButton = widgets.Button(
            description="No",
            button_style="danger",
            layout=widgets.Layout(width="80px")
        )
        self.noButton.on_click(self._no_clicked)

        # Store callbacks
        self.onConfirm = onConfirm
        self.onCancel = onCancel

        # Pack the panel
        self.panel = widgets.VBox([
            self.label,
            widgets.HBox([self.yesButton, self.noButton])
        ])
        
        # Output to display result
        self.output = widgets.Output()

    def _yes_clicked(self, b):
        with self.output:
            self.output.clear_output()
        if self.onConfirm:
            self.onConfirm()
        self._hide_panel()

    def _no_clicked(self, b):
        with self.output:
            self.output.clear_output()
        if self.onCancel:
            self.onCancel()
        self._hide_panel()

    def _hide_panel(self):
        # Hide the panel widgets
        self.panel.layout.display = 'none'

    def display(self):
        display(self.panel, self.output)
      

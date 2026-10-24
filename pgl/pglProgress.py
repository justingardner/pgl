################################################################
#   filename: pglProgress.py
#    purpose: Progress bar for PGL
#         by: JLG
#       date: October 24, 2026
################################################################
################################################################
#   filename: pglSaveProgress.py
#    purpose: Shared progress display for checkpoint publication
################################################################

import math
import time

from .pglMessages import pglMessages


class pglProgress:
    """One tqdm display with weighted child progress sections.

    Fractions are between zero and one.
    Child sections share the root display and never create additional bars.

    Rendering errors disable progress reporting without failing publication.
    Completion must be reported explicitly after commit and resource cleanup.

    Updates run synchronously in the caller's thread. No background timer or
    network activity is introduced.
    """

    def __init__(self, description="Progress"):
        """Create one fixed-width bar followed by single-line status text.

        Notebook:
            One graphical bar at the left.
            Percentage, timing, and description at the right.
            Long text is clipped with an ellipsis rather than wrapping.

        Terminal:
            One fixed-width text bar with output constrained to terminal width.

        Rendering setup failures do not prevent saving.
        """
        self._root = self
        self._start = 0.0
        self._end = 1.0
        self._label = ""
        self._fraction = 0.0
        self._description = description
        self._lastRendered = 0.0
        self._closed = False
        self._finished = False
        self._bar = None

        try:
            from tqdm.auto import tqdm
            from tqdm.notebook import tqdm_notebook

            notebookDisplay = issubclass(tqdm, tqdm_notebook)

            if notebookDisplay:
                # A bare {bar} becomes the notebook's graphical widget.
                # Do not add character-bar width or decorative pipe symbols.
                barFormat = "{bar} {percentage:3.0f}% [{elapsed:>8}<{remaining:>8}] {desc}"
                self._bar = tqdm(total=100, desc=description, unit="%", mininterval=0.2, leave=True, bar_format=barFormat)
            else:
                import shutil

                terminalWidth = max(40, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)
                barFormat = "{bar:20} {percentage:3.0f}% [{elapsed}<{remaining}] {desc}"
                self._bar = tqdm(total=100, desc=description, unit="%", mininterval=0.2, leave=True, dynamic_ncols=False, ncols=terminalWidth, bar_format=barFormat)

        except Exception as error:
            pglMessages.warning(f"Progress display unavailable ({type(error).__name__}); saving will continue without a waitbar.", level=0)
            return

        if not notebookDisplay:
            return

        try:
            from IPython.display import HTML, display

            container = self._bar.container
            leftText, barWidget, rightText = container.children

            container.add_class("pgl-progress-single-line")
            container.layout.display = "flex"
            container.layout.flex_flow = "row nowrap"
            container.layout.align_items = "center"
            container.layout.width = "100%"
            container.layout.max_width = "100%"
            container.layout.overflow = "hidden"

            # Nothing precedes {bar}, so this widget is unnecessary.
            leftText.layout.display = "none"

            barWidget.layout.flex = "0 0 240px"
            barWidget.layout.width = "240px"
            barWidget.layout.min_width = "240px"
            barWidget.layout.max_width = "240px"
            barWidget.layout.margin = "0 8px 0 0"

            rightText.add_class("pgl-progress-status")
            rightText.layout.flex = "1 1 0%"
            rightText.layout.min_width = "0px"
            rightText.layout.overflow = "hidden"
            rightText.layout.margin = "0"

            # Apply clipping to both the widget and its inner HTML content.
            # Scoped classes avoid changing other notebook widgets.
            display(HTML("""
                <style>
                    .pgl-progress-single-line {
                        flex-wrap: nowrap !important;
                        overflow: hidden !important;
                    }

                    .pgl-progress-single-line .pgl-progress-status,
                    .pgl-progress-single-line .pgl-progress-status .widget-html-content {
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                        min-width: 0 !important;
                    }

                    .pgl-progress-single-line .pgl-progress-status .widget-html-content {
                        display: block !important;
                        width: 100% !important;
                        font-variant-numeric: tabular-nums;
                    }
                </style>
            """))

            self._bar.refresh()

        except Exception as error:
            pglMessages.warning(f"Could not customize notebook progress layout ({type(error).__name__}); using tqdm's default display.", level=0)
            
    def section(self, start, end, label=None):
        """Return a weighted subsection of this progress range."""
        if not 0 <= start <= end <= 1:
            raise ValueError("Progress section must satisfy 0 <= start <= end <= 1.")

        child = object.__new__(type(self))
        child._root = self._root
        child._start = self._start + (self._end - self._start) * start
        child._end = self._start + (self._end - self._start) * end
        child._label = self._label if label is None else label
        return child

    def __call__(self, fraction=None, description=None):
        """Update progress; None leaves the percentage unchanged.

        Phase changes render immediately. Repeated byte updates are throttled.
        Progress never moves backward and remains below 100% until finish().
        """
        root = self._root

        if root._closed:
            return

        if fraction is not None:
            if not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
                raise ValueError("Progress fraction must be finite.")

            fraction = min(1.0, max(0.0, fraction))
            overall = self._start + (self._end - self._start) * fraction
            root._fraction = max(root._fraction, min(overall, 0.99))

        if description is not None:
            text = str(description)
            phase = text.split(" — ", 1)[0]
            text = f"{self._label}: {text}" if self._label else text
            phaseKey = (self._label, phase)
        else:
            text = root._description
            phaseKey = getattr(root, "_phaseKey", None)

        phaseChanged = phaseKey != getattr(root, "_phaseKey", None)
        root._phaseKey = phaseKey
        root._description = text
        now = time.monotonic()

        if root._bar is None or (not phaseChanged and now - root._lastRendered < 0.2):
            return

        try:
            root._bar.set_description_str(text, refresh=False)
            root._bar.n = root._fraction * 100
            root._bar.refresh()
            root._lastRendered = now
        except Exception as error:
            root._disableDisplay(error)

    def _disableDisplay(self, error):
        """Stop rendering without replacing a scientific or SQL outcome."""
        bar = self._bar
        self._bar = None

        if bar is not None:
            try:
                bar.close()
            except Exception:
                pass

        pglMessages.warning(f"Progress display failed ({type(error).__name__}); the operation will continue.", level=0)

    @staticmethod
    def fileSections(records, progress=None):
        """Yield each record with a byte-weighted child section.

        Zero-byte files receive one unit of weight.
        Successful completion of an iteration advances its entire section,
        including work skipped through verified reuse.

        Exceptions do not mark the current file complete.
        """
        records = list(records)
        weights = [max(1, record["size_bytes"]) for record in records]
        total = sum(weights)
        completed = 0

        for index, (record, weight) in enumerate(zip(records, weights)):
            child = None

            if progress is not None:
                label = f"File {index + 1}/{len(records)} {record['name']}"
                child = progress.section(completed / total, (completed + weight) / total, label=label)
                child(0, "Preparing file")

            yield record, child

            if child is not None:
                child(1, "File processed")

            completed += weight

        if progress is not None and not records:
            progress(1, "No files to process")

    def finish(self, description):
        """Report success only after the caller confirms commit and cleanup."""
        root = self._root
        root._finished = True
        root._fraction = 1.0
        root._description = description

        if root._bar is not None:
            try:
                root._bar.set_description_str(description, refresh=False)
                root._bar.n = 100
                root._bar.refresh()
            except Exception as error:
                root._disableDisplay(error)

    def close(self, outcome=None):
        """Close the display without inventing successful completion."""
        root = self._root

        if root._closed:
            return

        root._closed = True
        bar = root._bar
        root._bar = None

        if bar is None:
            return

        try:
            if outcome is not None and not root._finished:
                bar.set_description_str(f"{outcome} — {root._description}", refresh=False)

            bar.n = root._fraction * 100
            bar.refresh()
            bar.close()
        except Exception:
            pass

    def __enter__(self): return self

    def __exit__(self, exceptionType, exception, traceback):
        if exceptionType is None:
            outcome = None if self._finished else "Not completed"
        elif issubclass(exceptionType, KeyboardInterrupt):
            outcome = "Interrupted"
        else:
            outcome = "Failed"

        self.close(outcome)
        return False
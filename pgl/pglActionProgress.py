################################################################
#   filename: pglActionProgress.py
#    purpose: Shared action progress without duplicate displays
################################################################

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps


_activeProgress = ContextVar("pglActiveProgress", default=None)


def currentProgress():
    """Return the current explicitly assigned progress scope, or None."""
    return _activeProgress.get()


def _statusSection(progress):
    """Borrow a zero-width section for a helper without assigned work weight."""
    child = progress.section(0, 0)
    position = min(progress._end, max(progress._start, progress._root._fraction))
    child._start = position
    child._end = position
    return child


def borrowProgress(instance, description):
    """Initialize a nested pglProgress as a borrowed status-only section.

    Explicit weighted scopes advance percentages. Incidental helper bars only
    update status, so a completed file cannot finish the enclosing action.
    Returns False when there is no active action/operation display.
    """
    active = currentProgress()
    if active is None:
        return False

    child = _statusSection(active)
    instance._root = child._root
    instance._start = child._start
    instance._end = child._end
    instance._label = child._label
    instance(None, description)
    return True


def statusOutput(*values, sep=" ", end="\n", flush=False, **kwargs):
    """Route routine processing descriptions to progress, not the console.

    Used only by explicitly migrated processing helpers. It is not a global
    stdout filter: AFNI output, warnings, and tracebacks remain ordinary output.
    """
    text = " ".join(sep.join(str(value) for value in values).split())
    progress = currentProgress()

    if progress is not None:
        progress(None, text)


def noProgressBar(*args, **kwargs):
    """Compatibility context replacing removed Dask display contexts."""
    return nullcontext()


def progressCall(progress, start, end, description, function, *args, **kwargs):
    """Execute a helper in an explicit weighted section of one shared display."""
    child = progress.section(start, end)
    token = _activeProgress.set(child)

    try:
        child(0, description)
        result = function(*args, **kwargs)
        child(1, description)
        return result
    finally:
        _activeProgress.reset(token)


def operationProgress(description):
    """Give an independently called operation a bar, or borrow its caller's.

    Within an action, reserve the last 5% of the assigned scope for the owning
    caller's finalization. An operation never declares overall action success.
    """
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            from .pglProgress import pglProgress

            parent = currentProgress()
            owned = parent is None
            root = pglProgress(description) if owned else None
            scope = root.section(0, 0.95) if owned else parent.section(0, 0.95)
            token = _activeProgress.set(scope)
            outcome = "Failed"

            try:
                scope(0, description)
                result = function(*args, **kwargs)
                if owned:
                    root.finish(f"{description} complete")
                outcome = None
                return result
            except KeyboardInterrupt:
                outcome = "Interrupted"
                raise
            finally:
                _activeProgress.reset(token)
                if owned:
                    root.close(outcome)

        return wrapped
    return decorate


def actionProgress(function):
    """Own one display around the complete existing action runner.

    Includes history commitment, retention, presentation, and console-log
    attachment because those remain inside the wrapped run() implementation.

    Nested actions borrow the display without claiming their parent's work
    fraction. They do not close it or declare overall completion.

    Does not change action outcomes, recorder contexts, or scientific code.
    """
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        from .pglPipeline import pglActionStatus
        from .pglProgress import pglProgress

        # Preserve existing refusals without opening a misleading new bar.
        if getattr(self, "_executionInProgress", False) or not self.isConfigured():
            return function(self, *args, **kwargs)

        parent = currentProgress()
        owned = parent is None
        root = pglProgress(self.name) if owned else None
        scope = root.section(0, 0.99) if owned else _statusSection(parent)
        token = _activeProgress.set(scope)
        previousCount = len(self.actionHistory)
        outcome = "Failed"

        try:
            scope(0, f"{self.name} — starting")
            result = function(self, *args, **kwargs)

            if len(self.actionHistory) == previousCount:
                outcome = "Not started"
            elif self.status == pglActionStatus.SUCCESS:
                if owned:
                    root.finish(f"{self.name} — complete")
                outcome = None
            elif self.status == pglActionStatus.SKIPPED:
                outcome = "Skipped"
            else:
                outcome = "Failed"

            return result

        except KeyboardInterrupt:
            outcome = "Interrupted"
            raise

        finally:
            _activeProgress.reset(token)
            if owned:
                root.close(outcome)

    return wrapped


@contextmanager
def daskProgress(progress, description):
    """Update the existing display from Dask task completions.

    Creates no Dask/tqdm display. Fractions describe tasks, not bytes or elapsed
    time. The motion publisher uses the synchronous scheduler, so these updates
    run on the computation's calling thread.

    Graph completion reserves the final fraction for writer return/cleanup.
    """
    from dask.callbacks import Callback

    class ReportTasks(Callback):
        def _start(self, graph):
            self.total = max(len(graph), 1)
            self.completed = 0
            progress(0, f"{description} — computing and writing")

        def _posttask(self, key, result, graph, state, workerID):
            self.completed += 1
            progress(
                min(0.98, self.completed / self.total * 0.98),
                f"{description} — {self.completed}/{self.total} tasks completed",
            )

    with ReportTasks():
        yield

    progress(1, f"{description} — write completed")

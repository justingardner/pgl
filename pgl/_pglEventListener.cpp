/*
 * macOS Event Tap Listener
 * Captures keyboard and mouse events via CGEventTap
 * Calls Python callback with event data
 * author: Justin Gardner (modified from mglEventListener with help from Claude)
 * date: 2026-02-16
 * 
 * Shutdown:
 * - Never join the event thread while holding Python's GIL.
 * - A stop requested from the callback only requests shutdown.
 * - Event-tap resources are owned and released by the event thread.
 * - Startup waits until the tap is ready or startup has failed.
 *
 * Escape releases native keyboard suppression before acquiring the GIL.
 * Python remains responsible for queuing Escape and requesting emergency
 * shutdown on a second press.
 */

#include <Python.h>
#include <pthread.h>
#include <ApplicationServices/ApplicationServices.h>

#define MAX_EAT_KEYS 1024

// These mutexes remain valid for the module's lifetime.
static pthread_mutex_t stateMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t startupCondition = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t eatKeysMutex = PTHREAD_MUTEX_INITIALIZER;

static CFMachPortRef eventTap = NULL;
static CFRunLoopRef runLoop = NULL;
static pthread_t eventThread;

// Accessed only while holding the GIL. Retained until the thread is joined.
static PyObject *callbackObj = NULL;
static int lifecycleBusy = 0;

// Protected by stateMutex.
static int threadExists = 0;
static int listenerRunning = 0;
static int stopRequested = 0;
static int startupStatus = 0;  // 0=pending, 1=ready, -1=failed

// Protected by eatKeysMutex.
static int eatKeys[MAX_EAT_KEYS];
static int numEatKeys = 0;
static int eatAllKeys = 0;

static void* eventLoopThread(void* arg);
static CGEventRef eventCallback(
    CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon
);

static void clearSuppression(void) {
    pthread_mutex_lock(&eatKeysMutex);
    numEatKeys = 0;
    eatAllKeys = 0;
    pthread_mutex_unlock(&eatKeysMutex);
}

static int shouldEatKey(CGKeyCode keyCode) {
    int shouldEat = 0;

    pthread_mutex_lock(&eatKeysMutex);

    if (eatAllKeys) {
        shouldEat = 1;
    } else {
        for (int index = 0; index < numEatKeys; index++) {
            if (eatKeys[index] == (int)keyCode) {
                shouldEat = 1;
                break;
            }
        }
    }

    pthread_mutex_unlock(&eatKeysMutex);
    return shouldEat;
}

static void requestStop(void) {
    clearSuppression();

    pthread_mutex_lock(&stateMutex);
    stopRequested = 1;

    // The event thread clears these references under the same mutex before
    // releasing its resources, so they cannot be released during these calls.
    if (eventTap) CGEventTapEnable(eventTap, false);
    if (runLoop) {
        CFRunLoopStop(runLoop);
        CFRunLoopWakeUp(runLoop);
    }

    pthread_mutex_unlock(&stateMutex);
}

/*
 * Called with the GIL held and lifecycleBusy set.
 * The callback reference is released only after a successful join.
 */
static int joinEventThread(void) {
    pthread_mutex_lock(&stateMutex);
    int exists = threadExists;
    pthread_t thread = eventThread;
    pthread_mutex_unlock(&stateMutex);

    if (!exists) return 0;

    int error;
    Py_BEGIN_ALLOW_THREADS
    error = pthread_join(thread, NULL);
    Py_END_ALLOW_THREADS

    if (error != 0) {
        PyErr_Format(
            PyExc_RuntimeError,
            "Could not join event-listener thread (pthread error %d)", error
        );
        return -1;
    }

    pthread_mutex_lock(&stateMutex);
    threadExists = 0;
    listenerRunning = 0;
    pthread_mutex_unlock(&stateMutex);

    Py_CLEAR(callbackObj);
    return 0;
}

static PyObject* listenerStart(PyObject* self, PyObject* args) {
    PyObject *callback;

    if (!PyArg_ParseTuple(args, "O", &callback)) return NULL;
    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable");
        return NULL;
    }

    if (lifecycleBusy) {
        PyErr_SetString(PyExc_RuntimeError, "Listener startup/shutdown is in progress");
        return NULL;
    }

    pthread_mutex_lock(&stateMutex);
    int active = listenerRunning && !stopRequested;
    int calledFromEventThread =
        threadExists && pthread_equal(pthread_self(), eventThread);
    pthread_mutex_unlock(&stateMutex);

    if (active || calledFromEventThread) {
        PyErr_SetString(PyExc_RuntimeError, "Listener is already running or still inside its callback");
        return NULL;
    }

    if (!AXIsProcessTrusted()) {
        PyErr_SetString(
            PyExc_PermissionError,
            "Accessibility permission required. Enable your application in "
            "System Settings > Privacy & Security > Accessibility."
        );
        return NULL;
    }

    lifecycleBusy = 1;

    // Reap an earlier callback-requested shutdown before restarting.
    requestStop();
    if (joinEventThread() < 0) {
        lifecycleBusy = 0;
        return NULL;
    }

    clearSuppression();
    Py_INCREF(callback);
    callbackObj = callback;

    pthread_mutex_lock(&stateMutex);
    stopRequested = 0;
    startupStatus = 0;
    listenerRunning = 0;

    int error = pthread_create(&eventThread, NULL, eventLoopThread, NULL);
    if (error == 0) threadExists = 1;
    pthread_mutex_unlock(&stateMutex);

    if (error != 0) {
        Py_CLEAR(callbackObj);
        lifecycleBusy = 0;
        PyErr_Format(
            PyExc_RuntimeError,
            "Failed to create event-listener thread (pthread error %d)", error
        );
        return NULL;
    }

    int ready;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&stateMutex);
    while (startupStatus == 0) {
        pthread_cond_wait(&startupCondition, &stateMutex);
    }
    ready = startupStatus;
    pthread_mutex_unlock(&stateMutex);
    Py_END_ALLOW_THREADS

    if (ready != 1) {
        requestStop();
        if (joinEventThread() < 0) {
            lifecycleBusy = 0;
            return NULL;
        }

        lifecycleBusy = 0;
        PyErr_SetString(PyExc_RuntimeError, "Could not create the macOS event tap/run-loop source");
        return NULL;
    }

    lifecycleBusy = 0;
    Py_RETURN_NONE;
}

static PyObject* listenerStop(PyObject* self, PyObject* args) {
    pthread_mutex_lock(&stateMutex);
    int calledFromEventThread =
        threadExists && pthread_equal(pthread_self(), eventThread);
    pthread_mutex_unlock(&stateMutex);

    // Never join ourselves or release callbackObj during its invocation.
    if (calledFromEventThread) {
        requestStop();
        Py_RETURN_NONE;
    }

    if (lifecycleBusy) {
        PyErr_SetString(PyExc_RuntimeError, "Listener startup/shutdown is already in progress");
        return NULL;
    }

    lifecycleBusy = 1;
    requestStop();
    int result = joinEventThread();
    lifecycleBusy = 0;

    if (result < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject* listenerIsRunning(PyObject* self, PyObject* args) {
    pthread_mutex_lock(&stateMutex);
    int running = listenerRunning && !stopRequested;
    pthread_mutex_unlock(&stateMutex);
    return PyBool_FromLong(running);
}

static PyObject* listenerSetEatKeys(PyObject* self, PyObject* args) {
    PyObject *keyList;

    if (!PyArg_ParseTuple(args, "O", &keyList)) return NULL;
    if (!PyList_Check(keyList)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }

    Py_ssize_t listSize = PyList_Size(keyList);
    if (listSize > MAX_EAT_KEYS) {
        PyErr_Format(PyExc_ValueError, "Too many keys to eat (max %d)", MAX_EAT_KEYS);
        return NULL;
    }

    // Validate the entire request before changing the active configuration.
    int requestedKeys[MAX_EAT_KEYS];
    for (Py_ssize_t index = 0; index < listSize; index++) {
        PyObject *item = PyList_GetItem(keyList, index);
        if (!PyLong_Check(item) || PyBool_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Key codes must be integers, not Booleans");
            return NULL;
        }

        long value = PyLong_AsLong(item);
        if (PyErr_Occurred()) return NULL;
        if (value < 0 || value > 65535) {
            PyErr_SetString(PyExc_ValueError, "Key codes must be between 0 and 65535");
            return NULL;
        }
        requestedKeys[index] = (int)value;
    }

    pthread_mutex_lock(&eatKeysMutex);
    numEatKeys = (int)listSize;
    for (int index = 0; index < numEatKeys; index++) {
        eatKeys[index] = requestedKeys[index];
    }
    pthread_mutex_unlock(&eatKeysMutex);

    Py_RETURN_NONE;
}

static PyObject* listenerSetEatAllKeys(PyObject* self, PyObject* args) {
    int eatAll;
    if (!PyArg_ParseTuple(args, "p", &eatAll)) return NULL;

    pthread_mutex_lock(&eatKeysMutex);
    eatAllKeys = eatAll;
    pthread_mutex_unlock(&eatKeysMutex);

    Py_RETURN_NONE;
}

static void* eventLoopThread(void* arg) {
    CGEventMask eventMask =
        CGEventMaskBit(kCGEventKeyDown) |
        CGEventMaskBit(kCGEventKeyUp) |
        CGEventMaskBit(kCGEventLeftMouseDown) |
        CGEventMaskBit(kCGEventLeftMouseUp) |
        CGEventMaskBit(kCGEventRightMouseDown) |
        CGEventMaskBit(kCGEventRightMouseUp) |
        CGEventMaskBit(kCGEventOtherMouseDown) |
        CGEventMaskBit(kCGEventOtherMouseUp) |
        CGEventMaskBit(kCGEventMouseMoved) |
        CGEventMaskBit(kCGEventLeftMouseDragged) |
        CGEventMaskBit(kCGEventRightMouseDragged);

    CFMachPortRef localTap = CGEventTapCreate(
        kCGSessionEventTap, kCGHeadInsertEventTap,
        kCGEventTapOptionDefault, eventMask, eventCallback, NULL
    );

    CFRunLoopSourceRef source = localTap
        ? CFMachPortCreateRunLoopSource(kCFAllocatorDefault, localTap, 0)
        : NULL;

    if (!localTap || !source) {
        if (source) CFRelease(source);
        if (localTap) CFRelease(localTap);

        pthread_mutex_lock(&stateMutex);
        startupStatus = -1;
        listenerRunning = 0;
        pthread_cond_broadcast(&startupCondition);
        pthread_mutex_unlock(&stateMutex);
        return NULL;
    }

    CFRunLoopRef localLoop = CFRunLoopGetCurrent();
    CFRetain(localLoop);
    CFRunLoopAddSource(localLoop, source, kCFRunLoopCommonModes);

    pthread_mutex_lock(&stateMutex);
    eventTap = localTap;
    runLoop = localLoop;
    listenerRunning = 1;
    startupStatus = 1;
    CGEventTapEnable(localTap, true);
    pthread_cond_broadcast(&startupCondition);
    pthread_mutex_unlock(&stateMutex);

    // A bounded run-loop iteration also handles a stop request arriving
    // immediately before the run loop begins.
    for (;;) {
        pthread_mutex_lock(&stateMutex);
        int shouldStop = stopRequested;
        pthread_mutex_unlock(&stateMutex);
        if (shouldStop) break;

        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, false);
    }

    pthread_mutex_lock(&stateMutex);
    eventTap = NULL;
    runLoop = NULL;
    listenerRunning = 0;
    pthread_mutex_unlock(&stateMutex);

    CGEventTapEnable(localTap, false);
    CFRunLoopRemoveSource(localLoop, source, kCFRunLoopCommonModes);
    CFMachPortInvalidate(localTap);
    CFRelease(source);
    CFRelease(localTap);
    CFRelease(localLoop);
    clearSuppression();

    return NULL;
}

// PyDict_SetItemString does not steal its value reference.
static int setOwnedItem(PyObject *dictionary, const char *key, PyObject *value) {
    if (!value) return -1;
    int result = PyDict_SetItemString(dictionary, key, value);
    Py_DECREF(value);
    return result;
}

static CGEventRef eventCallback(
    CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon
) {
    if (type == kCGEventTapDisabledByTimeout ||
        type == kCGEventTapDisabledByUserInput) {
        // Fail open on a disabled tap. Do not silently restore suppression.
        clearSuppression();
        pthread_mutex_lock(&stateMutex);
        if (!stopRequested && eventTap) CGEventTapEnable(eventTap, true);
        pthread_mutex_unlock(&stateMutex);
        return event;
    }

    if (!event) return event;

    int keyboardEvent = type == kCGEventKeyDown || type == kCGEventKeyUp;
    CGKeyCode keyCode = keyboardEvent
        ? (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        : 0;

    // Escape is always passed through. Release suppression before waiting
    // for Python, so this recovery step does not depend on acquiring the GIL.
    if (keyboardEvent && keyCode == 53) clearSuppression();

    CGEventRef returnEvent = event;
    if (keyboardEvent && keyCode != 53 && shouldEatKey(keyCode)) {
        returnEvent = NULL;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    if (!callbackObj) {
        PyGILState_Release(gstate);
        return event;
    }

    PyObject *eventDict = PyDict_New();
    int success = eventDict != NULL;

    if (success) {
        success = setOwnedItem(
            eventDict, "timestamp",
            PyFloat_FromDouble((double)CGEventGetTimestamp(event) / 1e9)
        ) == 0;
    }

    if (success && keyboardEvent) {
        CGEventFlags flags = CGEventGetFlags(event);

        success =
            setOwnedItem(eventDict, "eventType", PyUnicode_FromString(
                type == kCGEventKeyDown ? "keydown" : "keyup")) == 0 &&
            setOwnedItem(eventDict, "keyCode", PyLong_FromLong(keyCode)) == 0 &&
            setOwnedItem(eventDict, "keyboardType", PyLong_FromLongLong(
                CGEventGetIntegerValueField(event, kCGKeyboardEventKeyboardType))) == 0 &&
            setOwnedItem(eventDict, "isRepeat", PyBool_FromLong(
                CGEventGetIntegerValueField(event, kCGKeyboardEventAutorepeat) != 0)) == 0 &&
            setOwnedItem(eventDict, "shift", PyBool_FromLong(
                (flags & kCGEventFlagMaskShift) != 0)) == 0 &&
            setOwnedItem(eventDict, "control", PyBool_FromLong(
                (flags & kCGEventFlagMaskControl) != 0)) == 0 &&
            setOwnedItem(eventDict, "alt", PyBool_FromLong(
                (flags & kCGEventFlagMaskAlternate) != 0)) == 0 &&
            setOwnedItem(eventDict, "command", PyBool_FromLong(
                (flags & kCGEventFlagMaskCommand) != 0)) == 0 &&
            setOwnedItem(eventDict, "capsLock", PyBool_FromLong(
                (flags & kCGEventFlagMaskAlphaShift) != 0)) == 0;
    } else if (success) {
        const char *typeName = NULL;
        int buttonEvent = 0;

        switch (type) {
            case kCGEventLeftMouseDown: typeName = "leftMouseDown"; buttonEvent = 1; break;
            case kCGEventLeftMouseUp: typeName = "leftMouseUp"; buttonEvent = 1; break;
            case kCGEventRightMouseDown: typeName = "rightMouseDown"; buttonEvent = 1; break;
            case kCGEventRightMouseUp: typeName = "rightMouseUp"; buttonEvent = 1; break;
            case kCGEventOtherMouseDown: typeName = "otherMouseDown"; buttonEvent = 1; break;
            case kCGEventOtherMouseUp: typeName = "otherMouseUp"; buttonEvent = 1; break;
            case kCGEventMouseMoved: typeName = "mouseMoved"; break;
            case kCGEventLeftMouseDragged: typeName = "leftMouseDragged"; break;
            case kCGEventRightMouseDragged: typeName = "rightMouseDragged"; break;
            default: break;
        }

        if (!typeName) {
            Py_DECREF(eventDict);
            PyGILState_Release(gstate);
            return returnEvent;
        }

        CGPoint location = CGEventGetLocation(event);
        success =
            setOwnedItem(eventDict, "eventType", PyUnicode_FromString(typeName)) == 0 &&
            setOwnedItem(eventDict, "x", PyFloat_FromDouble(location.x)) == 0 &&
            setOwnedItem(eventDict, "y", PyFloat_FromDouble(location.y)) == 0;

        if (success && buttonEvent) {
            success =
                setOwnedItem(eventDict, "button", PyLong_FromLongLong(
                    CGEventGetIntegerValueField(event, kCGMouseEventButtonNumber))) == 0 &&
                setOwnedItem(eventDict, "clickState", PyLong_FromLongLong(
                    CGEventGetIntegerValueField(event, kCGMouseEventClickState))) == 0;
        }
    }

    if (success) {
        PyObject *result = PyObject_CallFunctionObjArgs(callbackObj, eventDict, NULL);
        if (!result) {
            PyErr_Print();
        } else {
            // Never let a Python suppression decision override Escape recovery.
            if (result == Py_True && !(keyboardEvent && keyCode == 53)) {
                returnEvent = NULL;
            }
            Py_DECREF(result);
        }
    } else {
        PyErr_Print();
    }

    Py_XDECREF(eventDict);
    PyGILState_Release(gstate);
    return returnEvent;
}

static PyMethodDef listenerMethods[] = {
    {"start", listenerStart, METH_VARARGS, "Start the event listener"},
    {"stop", listenerStop, METH_NOARGS, "Stop the event listener"},
    {"isRunning", listenerIsRunning, METH_NOARGS, "Check listener status"},
    {"setEatKeys", listenerSetEatKeys, METH_VARARGS, "Set selectively suppressed keys"},
    {"setEatAllKeys", listenerSetEatAllKeys, METH_VARARGS, "Set all-key suppression"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef listenerModule = {
    PyModuleDef_HEAD_INIT,
    "_pglEventListener",
    "macOS event tap listener",
    -1,
    listenerMethods
};

PyMODINIT_FUNC PyInit__pglEventListener(void) {
    return PyModule_Create(&listenerModule);
}
 
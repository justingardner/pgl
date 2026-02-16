/*
 * macOS Event Tap Listener
 * Captures keyboard and mouse events via CGEventTap
 * Calls Python callback with event data
 * author: Justin Gardner (modified from mglEventListener with help from Claude)
 * date: 2026-02-16
 */

#include <Python.h>
#include <pthread.h>
#include <ApplicationServices/ApplicationServices.h>

// Constants
#define MAX_EAT_KEYS 1024

// Globals
static CFMachPortRef eventTap = NULL;
static CFRunLoopRef runLoop = NULL;
static pthread_t eventThread;
static PyObject *callbackObj = NULL;
static int listenerRunning = 0;
static pthread_mutex_t eatKeysMutex; 
static int eatKeys[MAX_EAT_KEYS];
static int numEatKeys = 0;

// Forward declarations
static void* eventLoopThread(void* arg);
static CGEventRef eventCallback(CGEventTapProxy proxy, CGEventType type, 
                                CGEventRef event, void *refcon);
static int shouldEatKey(CGKeyCode keyCode);

/*
 * Initialize and start the event tap
 */
static PyObject* listenerStart(PyObject* self, PyObject* args) {
    PyObject *callback;
    
    if (!PyArg_ParseTuple(args, "O", &callback)) {
        return NULL;
    }
    
    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "Callback must be callable");
        return NULL;
    }
    
    if (listenerRunning) {
        PyErr_SetString(PyExc_RuntimeError, "Listener already running");
        return NULL;
    }
    
    // Check accessibility permissions
    if (!AXIsProcessTrusted()) {
        PyErr_SetString(PyExc_PermissionError, 
            "Accessibility permission required. Go to System Preferences > "
            "Security & Privacy > Privacy > Accessibility and add Python/Terminal");
        return NULL;
    }
    
    // Initialize mutex for eatKeys array
    pthread_mutex_init(&eatKeysMutex, NULL);
    numEatKeys = 0;

    // Store callback reference
    Py_INCREF(callback);
    Py_XDECREF(callbackObj);
    callbackObj = callback;
    
    // Start event loop in separate thread
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    int err = pthread_create(&eventThread, &attr, eventLoopThread, NULL);
    pthread_attr_destroy(&attr);
    
    if (err != 0) {
        Py_DECREF(callbackObj);
        callbackObj = NULL;
        pthread_mutex_destroy(&eatKeysMutex); 
        PyErr_SetString(PyExc_RuntimeError, "Failed to create thread");
        return NULL;
    }
    
    listenerRunning = 1;

    Py_RETURN_NONE;
}

/*
 * Stop the event tap
 */
static PyObject* listenerStop(PyObject* self, PyObject* args) {
    if (!listenerRunning) {
        Py_RETURN_NONE;
    }
    
    // Stop the run loop
    if (runLoop) {
        CFRunLoopStop(runLoop);
    }
    
    // Wait for thread to finish
    pthread_join(eventThread, NULL);
    
    // Cleanup
    if (eventTap) {
        CGEventTapEnable(eventTap, false);
        CFRelease(eventTap);
        eventTap = NULL;
    }
    
    Py_XDECREF(callbackObj);
    callbackObj = NULL;
    listenerRunning = 0;
    runLoop = NULL;
    
    // Destroy mutex
    pthread_mutex_destroy(&eatKeysMutex);
    
    Py_RETURN_NONE;
}

/*
 * Check if listener is running
 */
static PyObject* listenerIsRunning(PyObject* self, PyObject* args) {
    return PyBool_FromLong(listenerRunning);
}

/*
 * Set which keys to eat (suppress from OS)
 */
static PyObject* listenerSetEatKeys(PyObject* self, PyObject* args) {
    PyObject *keyList;
    
    if (!PyArg_ParseTuple(args, "O", &keyList)) {
        return NULL;
    }
    
    if (!PyList_Check(keyList)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }
    
    Py_ssize_t listSize = PyList_Size(keyList);
    
    if (listSize > MAX_EAT_KEYS) {
        PyErr_Format(PyExc_ValueError, "Too many keys to eat (max %d)", MAX_EAT_KEYS);
        return NULL;
    }
    
    pthread_mutex_lock(&eatKeysMutex);
    
    numEatKeys = 0;
    for (Py_ssize_t i = 0; i < listSize; i++) {
        PyObject *item = PyList_GetItem(keyList, i);
        if (!PyLong_Check(item)) {
            pthread_mutex_unlock(&eatKeysMutex);
            PyErr_SetString(PyExc_TypeError, "All items must be integers");
            return NULL;
        }
        eatKeys[numEatKeys++] = (int)PyLong_AsLong(item);
    }
    
    pthread_mutex_unlock(&eatKeysMutex);
    
    Py_RETURN_NONE;
}

/*
 * Check if a keycode should be eaten (suppressed)
 */
static int shouldEatKey(CGKeyCode keyCode) {
    int shouldEat = 0;
    
    pthread_mutex_lock(&eatKeysMutex);
    
    for (int i = 0; i < numEatKeys; i++) {
        if (eatKeys[i] == (int)keyCode) {
            shouldEat = 1;
            break;
        }
    }
    
    pthread_mutex_unlock(&eatKeysMutex);
    
    return shouldEat;
}

/*
 * Event loop thread
 */
static void* eventLoopThread(void* arg) {
    CGEventMask eventMask = 
        (1 << kCGEventKeyDown) | 
        (1 << kCGEventKeyUp) |
        (1 << kCGEventLeftMouseDown) | 
        (1 << kCGEventLeftMouseUp) |
        (1 << kCGEventRightMouseDown) | 
        (1 << kCGEventRightMouseUp) |
        (1 << kCGEventOtherMouseDown) |
        (1 << kCGEventOtherMouseUp) |
        (1 << kCGEventMouseMoved) |
        (1 << kCGEventLeftMouseDragged) |
        (1 << kCGEventRightMouseDragged);
    
    eventTap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionDefault,
        eventMask,
        eventCallback,
        NULL
    );
    
    if (!eventTap) {
        fprintf(stderr, "(_pglEventListener) Failed to create event tap\n");
        return NULL;
    }
    
    CFRunLoopSourceRef runLoopSource = 
        CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0);
    
    runLoop = CFRunLoopGetCurrent();
    CFRunLoopAddSource(runLoop, runLoopSource, kCFRunLoopCommonModes);
    CGEventTapEnable(eventTap, true);
    
    CFRelease(runLoopSource);
    
    // Run the event loop
    CFRunLoopRun();
    
    return NULL;
}

/*
 * Event callback - called by OS on each event
 */
static CGEventRef eventCallback(CGEventTapProxy proxy, CGEventType type,
                                CGEventRef event, void *refcon) {
    if (!callbackObj) {
        return event;
    }

    // Check if we should eat this key event BEFORE processing
    CGEventRef returnEvent = event;  // Default: pass event through
    
    if (type == kCGEventKeyDown || type == kCGEventKeyUp) {
        CGKeyCode keyCode = (CGKeyCode)CGEventGetIntegerValueField(
            event, kCGKeyboardEventKeycode);
        
        if (shouldEatKey(keyCode)) {
            // Suppress this event - don't pass to other applications
            returnEvent = NULL;
        }
    }

    // Acquire GIL for Python callback
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    // Create event dictionary
    PyObject *eventDict = PyDict_New();
    
    // Common fields
    double timestamp = (double)CGEventGetTimestamp(event) / 1e9;
    PyDict_SetItemString(eventDict, "timestamp", PyFloat_FromDouble(timestamp));
    
    // Type-specific fields
    if (type == kCGEventKeyDown || type == kCGEventKeyUp) {

        PyDict_SetItemString(eventDict, "eventType", 
            PyUnicode_FromString(type == kCGEventKeyDown ? "keydown" : "keyup"));
        
        CGKeyCode keyCode = (CGKeyCode)CGEventGetIntegerValueField(
            event, kCGKeyboardEventKeycode);
        PyDict_SetItemString(eventDict, "keyCode", PyLong_FromLong(keyCode));
        
        int64_t keyboardType = CGEventGetIntegerValueField(
            event, kCGKeyboardEventKeyboardType);
        PyDict_SetItemString(eventDict, "keyboardType", PyLong_FromLong(keyboardType));
        
        CGEventFlags flags = CGEventGetFlags(event);
        PyDict_SetItemString(eventDict, "shift", PyBool_FromLong(flags & kCGEventFlagMaskShift));
        PyDict_SetItemString(eventDict, "control", PyBool_FromLong(flags & kCGEventFlagMaskControl));
        PyDict_SetItemString(eventDict, "alt", PyBool_FromLong(flags & kCGEventFlagMaskAlternate));
        PyDict_SetItemString(eventDict, "command", PyBool_FromLong(flags & kCGEventFlagMaskCommand));
        PyDict_SetItemString(eventDict, "capsLock", PyBool_FromLong(flags & kCGEventFlagMaskAlphaShift));
    }
    else if (type == kCGEventLeftMouseDown || type == kCGEventLeftMouseUp ||
             type == kCGEventRightMouseDown || type == kCGEventRightMouseUp ||
             type == kCGEventOtherMouseDown || type == kCGEventOtherMouseUp) {
        
        const char *typeStr = NULL;
        if (type == kCGEventLeftMouseDown) typeStr = "leftMouseDown";
        else if (type == kCGEventLeftMouseUp) typeStr = "leftMouseUp";
        else if (type == kCGEventRightMouseDown) typeStr = "rightMouseDown";
        else if (type == kCGEventRightMouseUp) typeStr = "rightMouseUp";
        else if (type == kCGEventOtherMouseDown) typeStr = "otherMouseDown";
        else if (type == kCGEventOtherMouseUp) typeStr = "otherMouseUp";
        
        PyDict_SetItemString(eventDict, "eventType", PyUnicode_FromString(typeStr));
        
        int64_t button = CGEventGetIntegerValueField(event, kCGMouseEventButtonNumber);
        PyDict_SetItemString(eventDict, "button", PyLong_FromLong(button));
        
        int64_t clickState = CGEventGetIntegerValueField(event, kCGMouseEventClickState);
        PyDict_SetItemString(eventDict, "clickState", PyLong_FromLong(clickState));
        
        CGPoint location = CGEventGetLocation(event);
        PyDict_SetItemString(eventDict, "x", PyFloat_FromDouble(location.x));
        PyDict_SetItemString(eventDict, "y", PyFloat_FromDouble(location.y));
    }
    else if (type == kCGEventMouseMoved || 
             type == kCGEventLeftMouseDragged ||
             type == kCGEventRightMouseDragged) {
        
        const char *typeStr = NULL;
        if (type == kCGEventMouseMoved) typeStr = "mouseMoved";
        else if (type == kCGEventLeftMouseDragged) typeStr = "leftMouseDragged";
        else if (type == kCGEventRightMouseDragged) typeStr = "rightMouseDragged";
        
        PyDict_SetItemString(eventDict, "eventType", PyUnicode_FromString(typeStr));
        
        CGPoint location = CGEventGetLocation(event);
        PyDict_SetItemString(eventDict, "x", PyFloat_FromDouble(location.x));
        PyDict_SetItemString(eventDict, "y", PyFloat_FromDouble(location.y));
    }
    
    // Call Python callback
    PyObject *result = PyObject_CallFunctionObjArgs(callbackObj, eventDict, NULL);
    
    // Check for errors
    if (result == NULL) {
        PyErr_Print();
    } else {
        Py_DECREF(result);
    }
    
    Py_DECREF(eventDict);
    
    // Release GIL
    PyGILState_Release(gstate);
    
    // Return NULL to suppress event, or event to pass it through
    return returnEvent;
}

/*
 * Module methods
 */
static PyMethodDef listenerMethods[] = {
    {"start", listenerStart, METH_VARARGS, "Start the event listener with a callback function"},
    {"stop", listenerStop, METH_NOARGS, "Stop the event listener"},
    {"isRunning", listenerIsRunning, METH_NOARGS, "Check if listener is running"},
    {"setEatKeys", listenerSetEatKeys, METH_VARARGS, "Set which keys to suppress from OS"},
     {NULL, NULL, 0, NULL}
};

/*
 * Module definition
 */
static struct PyModuleDef listenerModule = {
    PyModuleDef_HEAD_INIT,
    "_pglEventListener",
    "macOS event tap listener (C extension)",
    -1,
    listenerMethods
};

/*
 * Module initialization
 */
PyMODINIT_FUNC PyInit__pglEventListener(void) {
    // Initialize threading support (needed for Python < 3.9, harmless in 3.9+)
    #if PY_VERSION_HEX < 0x03090000
    PyEval_InitThreads();
    #endif
    
    return PyModule_Create(&listenerModule);
}
/////////////////////////
//   include section   //
/////////////////////////
#include <Python.h>
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

////////////////////////
//   define section   //
////////////////////////
#define kMaxDisplays 16

///////////////////////////////
//   function declarations   //
///////////////////////////////
static PyObject* setVerbose(PyObject* self, PyObject* args);
static PyObject* setGammaTable(PyObject* self, PyObject* args);
static PyObject* getGammaTable(PyObject* self, PyObject* args);
static PyObject* getGammaTableSize(PyObject* self, PyObject* args);

//////////////////////////
//   helper functions   //
//////////////////////////
CGDirectDisplayID getDisplayID(int whichScreen);

//////////////////////
// global variables //
//////////////////////
int verbose = 1;

///////////////////////////////
//   Python Object Defs      //
///////////////////////////////
// Method table
static PyMethodDef GammaTableMethods[] = {
    {"setGammaTable", setGammaTable, METH_VARARGS, "Set gamma table for a display"},
    {"getGammaTable", getGammaTable, METH_VARARGS, "Get gamma table for a display"},
    {"getGammaTableSize", getGammaTableSize, METH_VARARGS, "Get gamma table size for a display"},
    {"setVerbose", setVerbose, METH_VARARGS, "Set verbose level"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef GammaTableModule = {
    PyModuleDef_HEAD_INIT,
    "_pglGammaTable",
    "Gamma Table Module",
    -1,
    GammaTableMethods
};

PyMODINIT_FUNC PyInit__pglGammaTable(void) {
    return PyModule_Create(&GammaTableModule);
}

////////////////////////////
//   setVerbose function  //
////////////////////////////
static PyObject* setVerbose(PyObject* self, PyObject* args) 
{
  // get the verbose level from the arguments
  int newVerbose;
  if (!PyArg_ParseTuple(args, "i", &newVerbose)) return NULL;

  // set the global verbose variable
  verbose = newVerbose;

  // print a message if verbose is set to a high level
  if (verbose > 1)
    printf("(pgl:resolution:setVerbose) Verbose level set to %d\n", verbose);

  // return success
  Py_INCREF(Py_True); return Py_True;
}

///////////////////////////////
//   setGammaTable function  //
///////////////////////////////
static PyObject* setGammaTable(PyObject* self, PyObject* args) 
{
    PyObject *pyRed, *pyGreen, *pyBlue;
    int displayNumber;

    // Parse arguments: display index + 3 arrays
    if (!PyArg_ParseTuple(args, "iOOO", &displayNumber, &pyRed, &pyGreen, &pyBlue)) {
        return NULL;
    }

    // Get display ID
    CGDirectDisplayID whichDisplay = getDisplayID(displayNumber);
    if (whichDisplay == kCGNullDirectDisplay) {
        PyErr_SetString(PyExc_ValueError, "(_pglGammaTable:setGammaTable)Invalid display index");
        return NULL;
    }


    // Initialize NumPy C API
    import_array(); 

    // Convert to contiguous float32 arrays (CGGammaValue)
    PyArrayObject *redArray = (PyArrayObject*)PyArray_FROM_OTF(pyRed, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *greenArray = (PyArrayObject*)PyArray_FROM_OTF(pyGreen, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *blueArray = (PyArrayObject*)PyArray_FROM_OTF(pyBlue, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!redArray || !greenArray || !blueArray) {
        Py_XDECREF(redArray);
        Py_XDECREF(greenArray);
        Py_XDECREF(blueArray);
        PyErr_SetString(PyExc_ValueError, "(_pglGammaTable:setGammaTable) Failed to convert arrays to float32");
        return NULL;
    }

    // Get gamma table capacity
    npy_intp numEntries = (npy_intp)CGDisplayGammaTableCapacity(whichDisplay);

    // Ensure all arrays have the correct number of entries
    if (PyArray_SIZE(greenArray) != numEntries || PyArray_SIZE(blueArray) != numEntries) {
        Py_DECREF(redArray); Py_DECREF(greenArray); Py_DECREF(blueArray);
        PyErr_SetString(PyExc_ValueError, "(_pglGammaTable:setGammaTable) Red, green, and blue arrays must have the same length");
        return NULL;
    }

    // Get raw pointers to CGGammaValue
    CGGammaValue *redPtr = (CGGammaValue*)PyArray_DATA(redArray);
    CGGammaValue *greenPtr = (CGGammaValue*)PyArray_DATA(greenArray);
    CGGammaValue *bluePtr = (CGGammaValue*)PyArray_DATA(blueArray);

    // Apply the gamma table
    CGError err = CGSetDisplayTransferByTable(whichDisplay, (uint32_t)numEntries, redPtr, greenPtr, bluePtr);

    Py_DECREF(redArray); Py_DECREF(greenArray); Py_DECREF(blueArray);

    if (err != kCGErrorSuccess) {
        PyErr_Format(PyExc_RuntimeError, "Failed to set gamma table (error=%d)", err);
        return NULL;
    }

    Py_RETURN_NONE;
}

///////////////////////////////
//   getGammaTable function  //
///////////////////////////////
static PyObject* getGammaTable(PyObject* self, PyObject* args) 
{
    // parse the arguments
    int displayNumber = 0;
    if (!PyArg_ParseTuple(args, "i", &displayNumber)) return NULL;

    // Get the display
    CGDirectDisplayID whichDisplay = getDisplayID(displayNumber);
    if (whichDisplay == kCGNullDirectDisplay) {
        PyErr_SetString(PyExc_ValueError, "(_pglGammaTable:getGammaTable) Invalid display index");
        return NULL;
    }

    // Get gamma table capacity
    size_t gammaTableSize = CGDisplayGammaTableCapacity(whichDisplay);

    // allocate tables
    CGGammaValue *redTable = (CGGammaValue *)malloc(sizeof(CGGammaValue) * gammaTableSize);
    CGGammaValue *greenTable = (CGGammaValue *)malloc(sizeof(CGGammaValue) * gammaTableSize);
    CGGammaValue *blueTable = (CGGammaValue *)malloc(sizeof(CGGammaValue) * gammaTableSize);
    if (!redTable || !greenTable || !blueTable) {
        free(redTable); free(greenTable); free(blueTable);
        PyErr_SetString(PyExc_MemoryError, "(_pglGammaTable:getGammaTable)Failed to allocate gamma tables");
        return NULL;
    }

    uint32_t sampleCount = 0;
    CGError err = CGGetDisplayTransferByTable(
        whichDisplay,
        (uint32_t)gammaTableSize,
        redTable,
        greenTable,
        blueTable,
        &sampleCount
    );

    // check for error
    if (err != kCGErrorSuccess) {
        free(redTable); free(greenTable); free(blueTable);
        PyErr_Format(PyExc_RuntimeError, "(_pglGammaTable:getGammaTable)Error getting gamma table (error=%d)", err);
        return NULL;
    }

    // convert to Python lists
    PyObject *pyRed = PyList_New(sampleCount);
    PyObject *pyGreen = PyList_New(sampleCount);
    PyObject *pyBlue = PyList_New(sampleCount);
    for (uint32_t i = 0; i < sampleCount; i++) {
        printf("Index %d: R=%f, G=%f, B=%f\n", i, redTable[i], greenTable[i], blueTable[i]);
        PyList_SET_ITEM(pyRed, i, PyFloat_FromDouble(redTable[i]));
        PyList_SET_ITEM(pyGreen, i, PyFloat_FromDouble(greenTable[i]));
        PyList_SET_ITEM(pyBlue, i, PyFloat_FromDouble(blueTable[i]));
    }
    
    // deallocate tables
    free(redTable); free(greenTable); free(blueTable);

    // return as tuple (red, green, blue)
    PyObject *result = PyTuple_Pack(3, pyRed, pyGreen, pyBlue);
    Py_DECREF(pyRed);
    Py_DECREF(pyGreen);
    Py_DECREF(pyBlue);

    return result;
}

///////////////////////////////////
//   getGammaTableSize function  //
///////////////////////////////////
static PyObject* getGammaTableSize(PyObject* self, PyObject* args) 
{
    // parse the arguments
    int displayNumber = 0;
    if (!PyArg_ParseTuple(args, "i", &displayNumber)) return NULL;

    // Get the display
    CGDirectDisplayID whichDisplay = getDisplayID(displayNumber);
    if (whichDisplay == kCGNullDirectDisplay) {
        PyErr_SetString(PyExc_ValueError, "(_pglGammaTable:getGammaTable) Invalid display index");
        return NULL;
    }

    // Get gamma table capacity
    size_t gammaTableSize = CGDisplayGammaTableCapacity(whichDisplay);

    return PyLong_FromSize_t(gammaTableSize);}

//////////////////////////////
//   getDisplayID function  //
//////////////////////////////
CGDirectDisplayID getDisplayID(int displayNumber) 
{
    // Get list of active displays
    CGDirectDisplayID displays[kMaxDisplays];
    uint32_t numDisplays = 0;

    CGError err = CGGetActiveDisplayList(kMaxDisplays, displays, &numDisplays);
    if (err != kCGErrorSuccess || numDisplays == 0) {
        PyErr_SetString(PyExc_RuntimeError, "(_pglGammaTable:getDisplayID) Unable to get active displays");
        return  kCGNullDirectDisplay;
    }

    // Select which display
    CGDirectDisplayID whichDisplay;
    if (displayNumber < numDisplays) {
        whichDisplay = displays[displayNumber];
    } else {
        PyErr_Format(PyExc_ValueError,
                     "(_pglGammaTable:getDisplayID) Display index %d out of range (0-%d)",
                     displayNumber, numDisplays);
        return kCGNullDirectDisplay;
    }
    return whichDisplay;
}

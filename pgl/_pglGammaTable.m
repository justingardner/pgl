/////////////////////////
//   include section   //
/////////////////////////
#include <Python.h>
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>

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
    printf("setGammaTable called\n");
    Py_INCREF(Py_True); return Py_True;
}

///////////////////////////////
//   getGammaTable function  //
///////////////////////////////
static PyObject* getGammaTable(PyObject* self, PyObject* args) 
{
    printf("getGammaTable called\n");
    Py_INCREF(Py_True); return Py_True;
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
    if (whichDisplay ==  kCGNullDirectDisplay) return PyLong_FromLong(0);

    // Use size_t for the gamma table capacity
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
        PyErr_SetString(PyExc_RuntimeError, "Unable to get active displays");
        return  kCGNullDirectDisplay;
    }

    // Select which display
    CGDirectDisplayID whichDisplay;
    if (displayNumber == 0) {
        whichDisplay = kCGDirectMainDisplay;
    } else if (displayNumber <= numDisplays) {
        whichDisplay = displays[displayNumber - 1];
    } else {
        PyErr_Format(PyExc_ValueError,
                     "Display index %d out of range (0-%d)",
                     displayNumber, numDisplays);
        return kCGNullDirectDisplay;
    }
    return whichDisplay;
}

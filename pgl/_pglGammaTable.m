/////////////////////////
//   include section   //
/////////////////////////
#include <Python.h>
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>

////////////////////////
//   define section   //
////////////////////////
#define kMaxDisplays 8

///////////////////////////////
//   function declarations   //
///////////////////////////////
static PyObject* setVerbose(PyObject* self, PyObject* args);
static PyObject* setGammaTable(PyObject* self, PyObject* args);
static PyObject* getGammaTable(PyObject* self, PyObject* args);
static PyObject* getGammaTableBitDepth(PyObject* self, PyObject* args);

//////////////////////////
//   helper functions   //
//////////////////////////

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
    {"getGammaTableBitDepth", getGammaTableBitDepth, METH_NOARGS, "Get gamma table bit depth"},
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
}

///////////////////////////////
//   getGammaTable function  //
///////////////////////////////
static PyObject* getGammaTable(PyObject* self, PyObject* args) 
{
}

///////////////////////////////////////
//   getGammaTableBitDepth function  //
///////////////////////////////////////
static PyObject* getGammaTableBitDepth(PyObject* self, PyObject* args) 
{
}


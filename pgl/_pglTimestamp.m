///////////////////////
//   include section   //
///////////////////////
#include <Python.h>
#include <stdint.h>
#include <mach/mach_time.h>

///////////////////////////////
//   function declarations   //
///////////////////////////////
static PyObject* getSecs(PyObject* self, PyObject* args);

///////////////////////////////
//   Python Object Defs      //
///////////////////////////////
// Method table
static PyMethodDef MetalTimeMethods[] = {
    {"getSecs", getSecs, METH_VARARGS, "Get current time in seconds (Metal timebase)."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef MetalTimeModule = {
    PyModuleDef_HEAD_INIT,
    "_pglTimestamp",           // module name
    "Metal Time Module",       // docstring
    -1,
    MetalTimeMethods
};

PyMODINIT_FUNC PyInit__pglTimestamp(void) {
    return PyModule_Create(&MetalTimeModule);
}

////////////////////////////
//   getSecs function    //
////////////////////////////
static PyObject* getSecs(PyObject* self, PyObject* args) 
{
    static mach_timebase_info_data_t timebaseInfo = {0,0};

    // initialize timebase info once
    if (timebaseInfo.denom == 0) {
        mach_timebase_info(&timebaseInfo);
    }

    uint64_t absTime = mach_absolute_time();
    double nanoseconds = (double)absTime * timebaseInfo.numer / timebaseInfo.denom;
    double seconds = nanoseconds / 1e9;

    return Py_BuildValue("d", seconds);
}

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
int getBitDepth(CGDisplayModeRef displayMode);

// Stub: return fake resolution for a display
static PyObject* setResolution(PyObject* self, PyObject* args) {
    int displayNumber;
    if (!PyArg_ParseTuple(args, "i", &displayNumber))
        return NULL;

    // Replace this with real logic
    int screenWidth = 1920 + displayNumber;
    int screenHeight = 1080 + displayNumber;
    int frameRate = 60;
    int bitDepth = 24;

    return Py_BuildValue("(iiii)", screenWidth, screenHeight, frameRate, bitDepth);
}

static PyObject* getResolution(PyObject* self, PyObject* args) {
    int displayNumber = 1;
    if (!PyArg_ParseTuple(args, "i", &displayNumber))
        return NULL;

   // start auto release pool
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  // get all screen array
  NSArray *screens = [NSScreen screens];
  // grab the screen in question
  NSScreen *thisDisplay = [screens objectAtIndex:(displayNumber-1)];
  // get the display description
  NSDictionary *thisDisplayDescription = [thisDisplay deviceDescription];
  //NSLog(@"test %@",[thisDisplayDescription objectForKey:@"NSDeviceResolution"]);
  // get the display size
  NSSize thisDisplaySize = [[thisDisplayDescription objectForKey:@"NSDeviceSize"] sizeValue];
  int screenWidth = thisDisplaySize.width;
  int screenHeight = thisDisplaySize.height;
  // get the bit depth
  int bitDepth = [[thisDisplayDescription objectForKey:@"NSDeviceBitsPerSample"] integerValue];

  // FIX FIX FIX ok, giving up carbon code to follow here, when I can figure out how to 
  // write cocoa code above to get bitDepth and frameRate then the code
  // below can be removed
  CGDisplayErr displayErrorNum;
  CGDirectDisplayID displays[kMaxDisplays];
  CGDirectDisplayID whichDisplay;
  CGDisplayCount numDisplays;
  CFDictionaryRef modeInfo;

  // get status of global variable that sets wether to display
  // verbose information
  int verbose = 1;

  // check number of displays
  displayErrorNum = CGGetActiveDisplayList(kMaxDisplays,displays,&numDisplays);
  if (displayErrorNum) {
    printf("(mglResolution) Cannot get displays (%d)\n", displayErrorNum);
    return NULL;
  }

  if (verbose)
    printf("(mglResolution) Found %i displays\n",numDisplays);

  // get the display
  whichDisplay = displays[displayNumber-1];
  // get the display settings
  CGDisplayModeRef displayMode;
  displayMode = CGDisplayCopyDisplayMode(whichDisplay);
  
  // get bit rate
  bitDepth = getBitDepth(displayMode);
  // get frame rate
  int frameRate = (int)CGDisplayModeGetRefreshRate(displayMode);

  // release the display settings
  CGDisplayModeRelease(displayMode);

  // release the autorelease pool
  [pool release];
  return Py_BuildValue("(iiii)", screenWidth, screenHeight, frameRate, bitDepth);

}

static PyObject* getNumDisplaysAndDefault(PyObject* self, PyObject* args) {
    // Replace this with real logic
    int numDisplays = 2;
    int defaultDisplayNum = 2;
    // start auto release pool
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    // return num displays and default display number
    numDisplays = [[NSScreen screens] count];
    defaultDisplayNum = numDisplays;

    [pool release]; 

    return Py_BuildValue("(ii)", numDisplays, defaultDisplayNum);
}

// Method table
static PyMethodDef DisplayInfoMethods[] = {
    {"setResolution", setResolution, METH_VARARGS, "Get resolution info for a display"},
    {"getResolution", getResolution, METH_VARARGS, "Same as setResolution"},
    {"getNumDisplaysAndDefault", getNumDisplaysAndDefault, METH_NOARGS, "Get number of displays and default"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef displayInfoModule = {
    PyModuleDef_HEAD_INIT,
    "displayInfo",
    "Display Info Extension Module",
    -1,
    DisplayInfoMethods
};

PyMODINIT_FUNC PyInit_displayInfo(void) {
    return PyModule_Create(&displayInfoModule);
}

/////////////////////
//   getBitDepth   //
/////////////////////
int getBitDepth(CGDisplayModeRef displayMode)
{
  int bitDepth = 0;

  // get bit depth
  CFStringRef pixelEncoding;
  pixelEncoding = CGDisplayModeCopyPixelEncoding(displayMode);
  // return an appropriate bit depth for each one of these strings
  // defined in IOGraphicsTypes.h
  if (CFStringCompare(pixelEncoding,CFSTR(IO32BitDirectPixels),0)==kCFCompareEqualTo)
    bitDepth = 32;
  else if (CFStringCompare(pixelEncoding,CFSTR(IO16BitDirectPixels),0)==kCFCompareEqualTo)
    bitDepth = 16;
  else if (CFStringCompare(pixelEncoding,CFSTR(IO8BitIndexedPixels),0)==kCFCompareEqualTo)
    bitDepth = 8;
  else if (CFStringCompare(pixelEncoding,CFSTR(kIO30BitDirectPixels),0)==kCFCompareEqualTo)
    bitDepth = 30;
  else if (CFStringCompare(pixelEncoding,CFSTR(kIO64BitDirectPixels),0)==kCFCompareEqualTo)
    bitDepth = 64;
  // release the pixel encoding
  CFRelease(pixelEncoding);
  return(bitDepth);
}

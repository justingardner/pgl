/////////////////////////
//   include section   //
/////////////////////////
#include <Python.h>
#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CoreGraphics.h>

///////////////////////////////
//   function declarations   //
///////////////////////////////
static PyObject* setVerbose(PyObject* self, PyObject* args);
static PyObject* setResolution(PyObject* self, PyObject* args);
static PyObject* getResolution(PyObject* self, PyObject* args);
static PyObject* getNumDisplaysAndDefault(PyObject* self, PyObject* args);

//////////////////////////
//   helper functions   //
//////////////////////////
static CGDirectDisplayID getScreenDisplayID(NSScreen *screen);
int getBitDepth(CGDisplayModeRef displayMode);
boolean_t setBestMode(CGDirectDisplayID whichDisplay,int screenWidth,int screenHeight,int frameRate,int bitDepth);
void printDisplayModes(CGDirectDisplayID whichDisplay);

//////////////////////
// global variables //
//////////////////////
int verbose = 1;

///////////////////////////////
//   Python Object Defs      //
///////////////////////////////
// Method table
static PyMethodDef ResolutionMethods[] = {
    {"setResolution", setResolution, METH_VARARGS, "Get resolution info for a display"},
    {"getResolution", getResolution, METH_VARARGS, "Set resolution for a display"},
    {"getNumDisplaysAndDefault", getNumDisplaysAndDefault, METH_NOARGS, "Get number of displays and default"},
    {"setVerbose", setVerbose, METH_VARARGS, "Set verbose level"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef resolutionModule = {
    PyModuleDef_HEAD_INIT,
    "_resolution",
    "Display Info Extension Module",
    -1,
    ResolutionMethods
};

PyMODINIT_FUNC PyInit__resolution(void) {
    return PyModule_Create(&resolutionModule);
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
//   setResolution function  //
///////////////////////////////
static PyObject* setResolution(PyObject* self, PyObject* args) 
{
  // get arguments
  // displayNumber is the index of the display to set resolution for
  // screenWidth, screenHeight, frameRate, and bitDepth are the desired
  // parameters to set the display to
  // note that the displayNumber is zero indexed, so 0 is the first display
  int displayNumber = 0;
  int screenWidth, screenHeight, frameRate, bitDepth;
  if (!PyArg_ParseTuple(args, "iiiii", &displayNumber, &screenWidth, &screenHeight, &frameRate, &bitDepth)) return NULL;

  // start auto release pool
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  // Resolve the AppKit screen index used by the renderer to a Core Graphics ID.
  NSArray *screens = [NSScreen screens];
  if (displayNumber < 0 || (NSUInteger)displayNumber >= [screens count]) {
    [pool release];
    Py_RETURN_FALSE;
  }

  // Switch the display mode
  CGDirectDisplayID whichDisplay = getScreenDisplayID([screens objectAtIndex:displayNumber]);
  boolean_t success = whichDisplay != kCGNullDirectDisplay && setBestMode(whichDisplay, screenWidth, screenHeight, frameRate, bitDepth);
  [pool release];
  return PyBool_FromLong(success);
}

///////////////////////////////
//   getResolution function  //
///////////////////////////////
static PyObject* getResolution(PyObject* self, PyObject* args) 
{
  // get displayNumber for which display to return resolution info
  int displayNumber = 0;
  if (!PyArg_ParseTuple(args, "i", &displayNumber)) return NULL;

  // start auto release pool
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  // get all screen array
  NSArray *screens = [NSScreen screens];

  // Check the index against the current screen snapshot.
  NSUInteger numDisplays = [screens count];
  if (displayNumber < 0 || (NSUInteger)displayNumber >= numDisplays) {
    [pool release];
    return Py_BuildValue("(iiii)", -1,-1,-1,-1);
  }

  // grab the screen in question
  NSScreen *thisDisplay = [screens objectAtIndex:displayNumber];
  CGDirectDisplayID whichDisplay = getScreenDisplayID(thisDisplay);
  if (whichDisplay == kCGNullDirectDisplay) {
    [pool release];
    return Py_BuildValue("(iiii)", -1,-1,-1,-1);
  }

  // The display can disappear after AppKit returns it.
  CGDisplayModeRef displayMode = CGDisplayCopyDisplayMode(whichDisplay);
  if (displayMode == NULL) {
    [pool release];
    return Py_BuildValue("(iiii)", -1,-1,-1,-1);
  }

  // Keep AppKit dimensions and read the refresh rate and bit depth for the same display.
  NSSize thisDisplaySize = [[[thisDisplay deviceDescription] objectForKey:@"NSDeviceSize"] sizeValue];
  int screenWidth = thisDisplaySize.width;
  int screenHeight = thisDisplaySize.height;
  int bitDepth = getBitDepth(displayMode);
  int frameRate = (int)CGDisplayModeGetRefreshRate(displayMode);

  // release the display settings
  CGDisplayModeRelease(displayMode);

  // release the autorelease pool
  [pool release];

  // display information depending on verbose level
  if (verbose>0) printf("(pgl:_resolution:getResolution) Display %i/%lu: %ix%i %iHz %ibits\n",displayNumber,(unsigned long)numDisplays,screenWidth,screenHeight,frameRate,bitDepth);
  if (verbose>1) printDisplayModes(whichDisplay);

  // return the screen resolution, frame rate, and bit depth
  return Py_BuildValue("(iiii)", screenWidth, screenHeight, frameRate, bitDepth);
}

//////////////////////////////////////////
//   getNumDisplaysAndDefault function  //
//////////////////////////////////////////
static PyObject* getNumDisplaysAndDefault(PyObject* self, PyObject* args)
{
  // start auto release pool
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

  // return num displays and default display number
  int numDisplays = [[NSScreen screens] count];
  int defaultDisplayNum = numDisplays-1;

  [pool release]; 

  return Py_BuildValue("(ii)", numDisplays, defaultDisplayNum);
}

////////////////////////////
//   getScreenDisplayID   //
////////////////////////////
static CGDirectDisplayID getScreenDisplayID(NSScreen *screen)
{
  NSNumber *screenNumber = [[screen deviceDescription] objectForKey:@"NSScreenNumber"];
  return [screenNumber unsignedIntValue];
}

/////////////////////
//   getBitDepth   //
/////////////////////
int getBitDepth(CGDisplayModeRef displayMode)
{
  int bitDepth = 0;
  // This call is deprecated, but there does not seem to be a good replacement yet (7/8/2025)
  // get bit depth
  CFStringRef pixelEncoding;
  pixelEncoding = CGDisplayModeCopyPixelEncoding(displayMode);
  if (pixelEncoding == NULL) return bitDepth;
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

/////////////////////
//   setBestMode   //
/////////////////////
boolean_t setBestMode(CGDirectDisplayID whichDisplay,int screenWidth,int screenHeight,int frameRate,int bitDepth)
{
  CGDisplayModeRef mode;
  CFArrayRef modeList;
  CFIndex index, count;
  int bestWidth, bestHeight, bestBitDepth, thisBitDepth, bestFrameRate, thisFrameRate;
  boolean_t retval = false;

  // get all available display modes
  modeList = CGDisplayCopyAllDisplayModes(whichDisplay, NULL);
  if (modeList == NULL) return false;
  count = CFArrayGetCount(modeList);
  if (count == 0) {
    CFRelease(modeList);
    return false;
  }

  // check for closest match in width, height
  double minDifference = DBL_MAX,thisDifference;
  for (index = 0; index < count; index++) {
    // get the mode
    mode = (CGDisplayModeRef)CFArrayGetValueAtIndex(modeList, index);
    // check how close the pixel match is
    thisDifference = pow(((double)CGDisplayModeGetWidth(mode)-(double)screenWidth),2)+pow(((double)CGDisplayModeGetHeight(mode)-(double)screenHeight),2);
    if (thisDifference<minDifference) {
      bestWidth = (int)CGDisplayModeGetWidth(mode);
      bestHeight = (int)CGDisplayModeGetHeight(mode);
      minDifference = thisDifference;
    }
  }
  
  // now that we found the mode with the closest width/height match
  // check for best match in number of bits
  minDifference = DBL_MAX;
  for (index = 0; index < count; index++) {
    // get the mode
    mode = (CGDisplayModeRef)CFArrayGetValueAtIndex(modeList, index);
    // check that the width/height are matched to the best
    if ((bestWidth == (int)CGDisplayModeGetWidth(mode)) && (bestHeight == (int)CGDisplayModeGetHeight(mode))) {
      thisBitDepth = getBitDepth(mode);
      if (fabs((double)bitDepth-(double)thisBitDepth) < minDifference) {
	      minDifference = fabs((double)bitDepth-(double)thisBitDepth);
	      bestBitDepth = thisBitDepth;
      }
    }
  }

  // now that we found the mode with the closest width/height match
  // and the best number of bits, choose the best refresh rate
  minDifference = DBL_MAX;
  for (index = 0; index < count; index++) {
    // get the mode
    mode = (CGDisplayModeRef)CFArrayGetValueAtIndex(modeList, index);
    // check that the width/height and bitDepth are matched to the best
    if ((bestWidth == (int)CGDisplayModeGetWidth(mode)) && (bestHeight == (int)CGDisplayModeGetHeight(mode)) && (bestBitDepth == getBitDepth(mode))) {
      thisFrameRate = (int)CGDisplayModeGetRefreshRate(mode);
      if (thisFrameRate == 0) thisFrameRate = 60;
      if (fabs((double)frameRate-(double)thisFrameRate) < minDifference) {
	      minDifference = fabs((double)frameRate-(double)thisFrameRate);
	      bestFrameRate = thisFrameRate;
      }
    }
  }

  // now go set the best matching mode
  for (index = 0; index < count; index++) {
    // get the mode
    mode = (CGDisplayModeRef)CFArrayGetValueAtIndex(modeList, index);
    thisFrameRate = (int)CGDisplayModeGetRefreshRate(mode);
    if (thisFrameRate == 0) thisFrameRate = bestFrameRate;
    // check that the width/height and bitDepth are matched to the best
    if ((bestWidth == (int)CGDisplayModeGetWidth(mode)) && (bestHeight == (int)CGDisplayModeGetHeight(mode)) && (bestBitDepth == getBitDepth(mode)) && (bestFrameRate == thisFrameRate))  {
      // print the mode that is being set
      if (verbose > 0)
        printf("(pgl:_resolution:setBestMode) Setting display %i to %ix%i %iHz %i bits\n",
               whichDisplay, bestWidth, bestHeight, bestFrameRate, bestBitDepth);
      // capture the appropriate display
      CGError captureError = CGDisplayCapture(whichDisplay);
      // set the video mode
      CGError modeError = CGDisplaySetDisplayMode(whichDisplay,mode,NULL);
      // release the appropriate display
      if (captureError == kCGErrorSuccess) CGDisplayRelease(whichDisplay);
      if (modeError == kCGErrorSuccess) retval = true;
    }
  }
  // release the mode list
  CFRelease(modeList);

  if (retval && ((bestWidth != screenWidth) || (bestHeight != screenHeight) || (bestBitDepth != bitDepth) || (bestFrameRate != frameRate))) {
    printDisplayModes(whichDisplay);
    printf("(mglResolution:setBestMode) No exact mode match found (see avaliable modes printed above). Using closest match: %ix%i %i bits %iHz\n",bestWidth,bestHeight,bestBitDepth,bestFrameRate);
  }

  return(retval);
}
//////////////////////////
//   printDisplayModes  //
//////////////////////////
void printDisplayModes(CGDirectDisplayID whichDisplay)
{
  CGDisplayModeRef mode;
  CFArrayRef modeList;
  CFIndex index, count;

  printf("(pgl:_resolution:printDisplayModes) Available video modes for display %i\n", whichDisplay);

  // get all available display modes
  modeList = CGDisplayCopyAllDisplayModes(whichDisplay, NULL);
  if (modeList == NULL) return;
  count = CFArrayGetCount(modeList);

  // cycle through each available mode
  for (index = 0; index < count; index++) {
    // display info about each mode
    mode = (CGDisplayModeRef)CFArrayGetValueAtIndex(modeList, index);
    // get pixel encoding string
    CFStringRef pixelEncoding;
    pixelEncoding = CGDisplayModeCopyPixelEncoding(mode);
    char encodingStr[1024];
    if (pixelEncoding == NULL || !CFStringGetCString(pixelEncoding, encodingStr, sizeof(encodingStr), kCFStringEncodingUTF8)) {
      strcpy(encodingStr, "Unknown");
    }
    if (pixelEncoding != NULL) CFRelease(pixelEncoding);

    printf("%2d: %4dx%4d %3dHz %2d bits\t%s\n",
       (int)index,
       (int)CGDisplayModeGetWidth(mode),
       (int)CGDisplayModeGetHeight(mode),
       (int)CGDisplayModeGetRefreshRate(mode),
       (int)getBitDepth(mode),
       encodingStr);
  }
  CFRelease(modeList);
}

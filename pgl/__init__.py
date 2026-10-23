"""Public PGL exports, loaded on demand."""

import importlib as _importlib
import sys as _sys
import types as _types

# export table (lists all the classes and files that are publicly exported from the PGL package)
# this is done so that the class import can be lazy and avoid loading dependencies that do not
# work on non macOS systems
_EXPORTS = {
 'pglBase': ('.pglBase', 'pglBase'),
 'pglResolution': ('.pglResolution', 'pglResolution'),
 'pglDraw': ('.pglDraw', 'pglDraw'),
 'pglTransform': ('.pglTransform', 'pglTransform'),
 'pglProfile': ('.pglProfile', 'pglProfile'),
 'pglBatch': ('.pglBatch', 'pglBatch'),
 'pglImage': ('.pglImage', 'pglImage'),
 'pglImageDatabase': ('.pglImage', 'pglImageDatabase'),
 'pglMovieDatabase': ('.pglImage', 'pglMovieDatabase'),
 'pglStimuli': ('.pglStimuli', 'pglStimuli'),
 'pglTimestamp': ('.pglTimestamp', 'pglTimestamp'),
 'pglDevice': ('.pglDevice', 'pglDevice'),
 'pglDevices': ('.pglDevice', 'pglDevices'),
 'pglDigitalIODevice': ('.pglDevice', 'pglDigitalIODevice'),
 'pglAnalogTraceData': ('.pglDevice', 'pglAnalogTraceData'),
 'pglKeyboardMouse': ('.pglKeyboardMouse', 'pglKeyboardMouse'),
 'pglEventKeyboard': ('.pglEvent', 'pglEventKeyboard'),
 'pglKeyBuffer': ('.pglKeyboardMouse', 'pglKeyBuffer'),
 'pglEvent': ('.pglEvent', 'pglEvent'),
 'pglEvents': ('.pglEvent', 'pglEvents'),
 'pglCommandReplayer': ('.pglCommandReplayer', 'pglCommandReplayer'),
 'pglFrameGrab': ('.pglFrameGrab', 'pglFrameGrab'),
 'pglExperiment': ('.pglExperiment', 'pglExperiment'),
 'pglTask': ('.pglExperiment', 'pglTask'),
 'pglParameter': ('.pglParameter', 'pglParameter'),
 'pglParameterBlock': ('.pglParameter', 'pglParameterBlock'),
 'pglParameterNestedBlock': ('.pglParameter', 'pglParameterNestedBlock'),
 'pglParameterBatch': ('.pglParameter', 'pglParameterBatch'),
 'pglStaircase': ('.pglStaircase', 'pglStaircase'),
 'pglStaircaseUpDown': ('.pglStaircase', 'pglStaircaseUpDown'),
 'pglFixationTaskLeftRight': ('.pglTasks', 'pglFixationTaskLeftRight'),
 'pglBarTask': ('.pglTasks', 'pglBarTask'),
 'pglTestTask': ('.pglTasks', 'pglTestTask'),
 'pglEyeTrackingCalibrationTask': ('.pglTasks',
                                   'pglEyeTrackingCalibrationTask'),
 'pglMessageAckTask': ('.pglTasks', 'pglMessageAckTask'),
 'pglSerial': ('._pglComm', 'pglSerial'),
 'pglDisplayCalibration': ('.pglCalibration', 'pglDisplayCalibration'),
 'pglLuminanceCalibrationDeviceMinolta': ('.pglCalibration',
                                          'pglLuminanceCalibrationDeviceMinolta'),
 'pglDisplayLuminanceCalibrationData': ('.pglCalibration',
                                        'pglDisplayLuminanceCalibrationData'),
 'pglLuminanceCalibrationDeviceDebug': ('.pglCalibration',
                                        'pglLuminanceCalibrationDeviceDebug'),
 'pglGammaTable': ('.pglGammaTable', 'pglGammaTable'),
 'pglSettingsManager': ('.pglSettings', 'pglSettingsManager'),
 'pglDisplaySettings': ('.pglSettings', 'pglDisplaySettings'),
 'pglDisplaySettingsList': ('.pglSettings', 'pglDisplaySettingsList'),
 'pglDisplayModeSettings': ('.pglSettings', 'pglDisplayModeSettings'),
 'pglSettings': ('.pglSettings', 'pglSettings'),
 'pglTraitSettings': ('.pglSettings', 'pglTraitSettings'),
 'pglStateDataSettings': ('.pglSettings', 'pglStateDataSettings'),
 'pglEventListener': ('.pglEventListener', 'pglEventListener'),
 'pglEyeTracker': ('.pglEyeTracker', 'pglEyeTracker'),
 'pglEyePositionSample': ('.pglEyeTracker', 'pglEyePositionSample'),
 'pglDialogs': ('.pglDialog', 'pglDialogs'),
 'pglMessages': ('.pglMessages', 'pglMessages'),
 'pglActions': ('.pglActions', 'pglActions'),
 'pglFlywheel': ('.pglFlywheel', 'pglFlywheel'),
 'pglChooseBlock': ('.pglDigitalBrain', 'pglChooseBlock'),
 'pglDigitalBrainMemoryTask': ('.pglDigitalBrain', 'pglDigitalBrainMemoryTask'),
 'pglDigitalBrainConfigure': ('.pglDigitalBrain', 'pglDigitalBrainConfigure'),
 'pglChoose': ('.pglChoose', 'pglChoose'),
 'pglItem': ('.pglChoose', 'pglItem'),
 'pglSession': ('.pglSession', 'pglSession'),
 'pglStorage': ('.pglStorage', 'pglStorage'),
 'pglDatabase': ('.pglDatabase', 'pglDatabase'),
 'pglPostgres': ('.pglPostgres', 'pglPostgres'),
 'pglActionsFixes': ('.pglActionsFixes', 'pglActionsFixes'),
 'pglActionsMNE': ('.pglActionsMNE', 'pglActionsMNE'),
 'pglDataset': ('.pglDataset', 'pglDataset'),
 'pglDataMatrix': ('.pglData', 'pglDataMatrix'),
 'pglDataMatrixReference': ('.pglData', 'pglDataMatrixReference'),
 'pglBehavior': ('.pglBehavior', 'pglBehavior'),
 'pglProPixx': ('.pglVPixx', 'pglProPixx'),
 'pglDataPixx': ('.pglVPixx', 'pglDataPixx'),
 'pglTrackPixx3': ('.pglTrackPixx', 'pglTrackPixx3'),
 'pglLabJack': ('.pglLabJack', 'pglLabJack'),
 'pglEyelinkData': ('.pglEyelink', 'pglEyelinkData'),
 'pglVWFATask': ('.pglVWFA', 'pglVWFATask'),
 'pglEyelink': ('.pglEyelink', 'pglEyelink'),
 'pgl': ('._graphics', 'pgl'),
 'pglInfo': ('.pglInfo', 'pglInfo')}

_resolved_exports = {}
__all__ = list(_EXPORTS)

__version__ = '1.0.0'
__author__ = 'JLG'


class _LazyPackage(_types.ModuleType):
    """Keep public class exports stable even after submodule imports."""

    def __getattribute__(self, name):
        namespace = _types.ModuleType.__getattribute__(self, "__dict__")
        exports = namespace.get("_EXPORTS", {})

        if name in exports:
            cache = namespace["_resolved_exports"]

            if name not in cache:
                module_name, attribute_name = exports[name]
                module = _importlib.import_module(module_name, namespace["__name__"])
                value = getattr(module, attribute_name)
                cache[name] = value

            return cache[name]

        return _types.ModuleType.__getattribute__(self, name)

    def __dir__(self):
        namespace = _types.ModuleType.__getattribute__(self, "__dict__")
        return sorted(set(namespace) | set(namespace["_EXPORTS"]))


_sys.modules[__name__].__class__ = _LazyPackage

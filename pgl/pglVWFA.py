################################################################
#   filename: pglVWFA.py
#    purpose: VWFA task (for C-ShARP project)
#         by: Qiyuan Feng (adapted from mgl version mglVWFA.m)
#       date: April 6, 2026 (modified June 1, 2026)
################################################################

#############
# Import modules
#############
from .pglExperiment import pglTask
from .pglStaircase import pglStaircaseUpDown
import numpy as np
from .pglParameter import pglParameter, pglParameterBlock
import os
from PIL import Image as PILImage
import scipy.io as sio
import h5py

#############
# Helper Functions
#############

def _load_mat(filepath, old_ver=False):
    """
    Load .mat file; Return the contents of a .mat file as a dict.
    """
    # for current version, use scipy.io.loadmat
    if not old_ver:
        return sio.loadmat(filepath)
    # for v7.3 .mat files need to use h5py instead (HDF5-based)
    else:
        out = {}
        with h5py.File(filepath, "r") as f:
            for k in f.keys():
                out[k] = f[k][()]
        return out
    
#############
def _gray_to_rgba_pil(img_uint8: np.ndarray) -> PILImage.Image:
    """ convert a raw 2-D uint8 grayscale image array -> RGBA PIL Image
    Pixels with value 128 are treated as background and become transparent
    (alpha = 0).  All other pixels are opaque (alpha = 255).
    In mglVWFA: 
    - alpha(img == 128) = 0;
    - rgba = cat(3, img, img, img, alpha);
    """
    h, w = img_uint8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = img_uint8   # R
    rgba[:, :, 1] = img_uint8   # G
    rgba[:, :, 2] = img_uint8   # B
    rgba[:, :, 3] = 255         # fully opaque by default
    rgba[img_uint8 == 128, 3] = 0  # transparent background
    return PILImage.fromarray(rgba, mode="RGBA")

#############
# Main Task
#############

class pglVWFATask(pglTask):

    # Condition index (stimType, fontType)
    CONDITION_NAMES = ["highFreqWords", "pseudowords", "consonants", "falseFonts"]
    FONT_NAMES = ["Sloan", "Courier"]
 
    NUM_CONDITIONS = 9    # 8 visual + 1 blank
    NUM_INSTANCES = 12   # 12 stimulus files per condition per run (36 total, 3 runs)
 
    # Segment durations (seconds): 4 × (700ms stim + 300ms blank), no ITI
    # Segments: 1=stim, 2=blank, 3=stim, 4=blank, 5=stim, 6=blank, 7=stim, 8=blank
    SEG_DURATIONS = [0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3]

    # Which segments accept a response (press = same/repeat)
    RESPONSE_SEGS = {2, 3, 4, 5, 6, 7, 8}  # 1-indexed segment numbers
 
    def __init__(self, pgl, stimulusFolder: str, runNum: int = 1, TR: float = 1.0, repeatForever: bool = False):
        super().__init__()

        if runNum not in (1, 2, 3):
            raise ValueError(f"runNum must be 1, 2, or 3 (got {runNum})")

        self.runNum = runNum
        # Instances for this run: run 1 → 1-12, run 2 → 13-24, run 3 → 25-36
        self._instStart = (runNum - 1) * self.NUM_INSTANCES + 1
        self._instEnd   = runNum * self.NUM_INSTANCES

        self.settings.taskName = "VWFA Task"
        self.settings.fixedParameters = {
            "stimulusFolder": stimulusFolder,
            "numConditions": self.NUM_CONDITIONS,
            "numInstances": self.NUM_INSTANCES,
            "runNum": runNum,
            "segDurations": self.SEG_DURATIONS,
        }

        # Shorten last blank so trigger-wait catches the TR boundary cleanly
        seg_durations = list(self.SEG_DURATIONS)
        seg_durations[-1] = max(seg_durations[-1] - TR / 2, TR / 10)
        self.settings.seglen = seg_durations
        self.settings.nTrials = np.inf if repeatForever else self.NUM_CONDITIONS * self.NUM_INSTANCES
        self.settings.waitUntilVolumeTrigger[-1] = True

        # every (condition, instance) pair appears exactly once per block
        self.addParameter(pglParameterNestedBlock([
            pglParameter("condition", list(range(1, self.NUM_CONDITIONS + 1))),
            pglParameter("instance",  list(range(self._instStart, self._instEnd + 1))),
        ]))

        # Preload file paths (textures are loaded per-trial for memory)
        self.stimulusFolder = stimulusFolder
        self.filePaths = self._build_file_paths()
 
        # Per-trial state (populated in startTrial / startSegment)
        self._currentImages: list = []   # list of 4 PIL RGBA images
        self._currentNames: list = []    # list of 4 name strings
        self._currentSegImage = None     # PIL image currently on screen (or None)
        self._isRepeat = False           # whether all 4 names in trial are identical
        self._gotResponse = False
        self._trialStartTime = None
 
    # File path pre-compute: Return a dict mapping (condIdx, instIdx) -> filepath (1-indexed)
    def _build_file_paths(self) -> dict:
        paths = {}
        for condIdx in range(1, 9):  # conditions 1-8 (skip blank=9)
            stim_type_idx = (condIdx - 1) // 2        # 0-3
            font_idx      = (condIdx - 1) %  2        # 0-1
            stim_type = self.CONDITION_NAMES[stim_type_idx]
            font_type = self.FONT_NAMES[font_idx]
            for inst in range(self._instStart, self._instEnd + 1):
                fname = f"{stim_type}{font_type}_single_{inst:03d}.mat"
                fpath = os.path.join(self.stimulusFolder, fname)
                if not os.path.isfile(fpath):
                    print(f"(pglVWFATask) Warning: missing stimulus file: {fpath}")
                paths[(condIdx, inst)] = fpath
        return paths
 
    # Load 4 RGBA PIL images and their names from the .mat file.
    def _load_images(self, condIdx: int, instIdx: int):
        fpath = self.filePaths.get((condIdx, instIdx))
        if fpath is None or not os.path.isfile(fpath):
            print(f"(pglVWFATask) Missing file for condition={condIdx} instance={instIdx}")
            return [], []
 
        mat = _load_mat(fpath)
 
        # --- img array ---
        # Expected shape from MATLAB: (900, 1440, 1, 4) uint8
        # scipy.io preserves MATLAB column-major order; we need (H, W, 1, 4)
        raw = mat["img"]
 
        # Squeeze out the singleton 3rd dim → (H, W, 4)
        if raw.ndim == 4:
            raw = raw.squeeze(axis=2)  # → (H, W, 4)
 
        # names
        raw_names = mat.get("names", None)
        names = self._parse_names(raw_names, n=raw.shape[2] if raw.ndim == 3 else 4)
 
        # Build PIL images
        images = []
        n_imgs = raw.shape[2] if raw.ndim == 3 else 0
        for k in range(n_imgs):
            img_slice = raw[:, :, k]
            # Ensure uint8
            if img_slice.dtype != np.uint8:
                img_slice = np.clip(img_slice, 0, 255).astype(np.uint8)
            images.append(_gray_to_rgba_pil(img_slice))
 
        return images, names
 
    # Parse MATLAB name arrays (cell arrays become object arrays via scipy).
    @staticmethod
    def _parse_names(raw_names, n=4) -> list:
        if raw_names is None:
            return [f"stim_{i}" for i in range(n)]
 
        # scipy.io returns MATLAB cell arrays as numpy object arrays
        if isinstance(raw_names, np.ndarray):
            flat = raw_names.flatten()
            result = []
            for item in flat:
                if isinstance(item, np.ndarray):
                    result.append(str(item.flat[0]))
                else:
                    result.append(str(item))
            return result
 
        # h5py path: byte strings
        if isinstance(raw_names, (list, tuple)):
            return [s.decode() if isinstance(s, bytes) else str(s) for s in raw_names]
 
        return [str(raw_names)] * n
 
    ########################
    def startTrial(self, startTime):
        """Called once at the beginning of each trial"""
        super().startTrial(startTime)
 
        self._gotResponse = False
        self._currentImages = []
        self._currentNames  = []
        self._currentSegImage = None
        self._trialStartTime = startTime
 
        cond = self.currentParams["condition"]
        inst = self.currentParams["instance"]
 
        if cond < 9:
            imgs, names = self._load_images(cond, inst)
            self._currentImages = imgs
            self._currentNames  = names
 
            # 1-back: repeat if any consecutive pair of stimuli is identical
            if names:
                self._isRepeat = any(names[i] == names[i-1] for i in range(1, len(names)))
            else:
                self._isRepeat = False
 
            print(f"(pglVWFATask)   names={names} | isRepeat={self._isRepeat}")
        else:
            # Blank condition
            self._isRepeat = False
 
    ########################
    def startSegment(self, startTime):
        '''
        Start a segment.
        '''
        super().startSegment(startTime)
 
        seg = self.state.currentSegment + 1  # convert to 1-indexed
        cond = self.currentParams["condition"]
 
        # Stimulus segments: 1, 3, 5, 7
        STIM_SEGS = {1, 3, 5, 7}
        seg_image_idx = {1: 0, 3: 1, 5: 2, 7: 3}
 
        if seg in STIM_SEGS and cond < 9 and self._currentImages:
            k = seg_image_idx[seg]
            if k < len(self._currentImages):
                self._currentSegImage = self._currentImages[k]
            else:
                self._currentSegImage = None
        else:
            self._currentSegImage = None  # blank segments show nothing
    
    ########################
    def updateScreen(self):
        """Called every frame to draw the screen."""
        self.pgl.clearScreen(0.5)

        if self._currentSegImage is not None:
            self.pgl.imageDisplay(self._currentSegImage, x=0, y=0)
    
    ########################
    def handleSubjectResponse(self, _response, updateTime):
        """Called when the subject presses any button (means: same/repeat).
        No response during trial means: different.
        """
        seg = self.state.currentSegment + 1  # 1-indexed

        if seg not in self.RESPONSE_SEGS:
            return None

        # Accept only the first press per trial
        if self._gotResponse:
            return None
        self._gotResponse = True

        # Any press = subject says SAME (repeat)
        correct = self._isRepeat

        rt = updateTime - self._trialStartTime
        print(f"(pglVWFATask) Press (SAME) | RT: {rt:.3f}s | correct: {correct}")

        return correct
    
    ########################
    def endTrial(self, endTime):
        """Clean up at the end of each trial."""
        super().endTrial(endTime)
        self._currentImages = []
        self._currentNames  = []
        self._currentSegImage = None


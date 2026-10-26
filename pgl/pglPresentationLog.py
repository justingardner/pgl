################################################################
#   filename: pglPresentationLog.py
#    purpose: Low-overhead in-memory capture of flush results.
#             Packing, analysis, and saving happen after capture.
#         by: JLG
#       date: Oct 25, 2026
################################################################

#############
# Imports
#############
import math
import numpy as np


class pglPresentationLog:
    """Preallocated storage for existing renderer flush results.

    Host timestamps are assumed to use the getSecs() clock in seconds.
    Missing result fields become NaN; renderer sentinel values are preserved.
    commandIndices contains -1 when command recording was not active.
    """

    fieldNames = (
        "drawablePresented",
        "processedTime",
        "drawableAcquired",
        "success",
        "commandCode",
    )

    def __init__(self, estimatedDuration, frameRate):
        """Allocate using estimated duration in seconds and display frame rate."""
        if isinstance(estimatedDuration, (bool, np.bool_)): raise ValueError("estimatedDuration must be a positive number of seconds.")
        if isinstance(frameRate, (bool, np.bool_)): raise ValueError("frameRate must be a positive number.")

        estimatedDuration = float(estimatedDuration)
        frameRate = float(frameRate)

        if not math.isfinite(estimatedDuration) or estimatedDuration <= 0: raise ValueError("estimatedDuration must be finite and positive.")
        if not math.isfinite(frameRate) or frameRate <= 0: raise ValueError("frameRate must be finite and positive.")

        estimatedFrames = estimatedDuration * frameRate
        if not math.isfinite(estimatedFrames): raise ValueError("Estimated frame capacity is too large.")

        self.estimatedDuration = estimatedDuration
        self.frameRate = frameRate
        self.capacity = max(1, math.ceil(estimatedFrames))
        self.count = 0
        self.growthCount = 0

        # Only populated rows are exported; every appended row is assigned.
        self._values = np.empty((self.capacity, len(self.fieldNames)), dtype=np.float64)
        self._commandIndices = np.empty(self.capacity, dtype=np.int64)

    def append(self, commandResults, commandIndex=-1):
        """Retain one flush result without socket operations or clock calls."""
        if self.count >= self.capacity:
            # Count growth rather than printing during presentation.
            capacity = self.capacity * 2
            values = np.empty((capacity, len(self.fieldNames)), dtype=np.float64)
            commandIndices = np.empty(capacity, dtype=np.int64)

            values[:self.count] = self._values[:self.count]
            commandIndices[:self.count] = self._commandIndices[:self.count]

            self._values = values
            self._commandIndices = commandIndices
            self.capacity = capacity
            self.growthCount += 1

        row = self._values[self.count]
        if commandResults is None:
            row.fill(np.nan)
        else:
            for index, name in enumerate(self.fieldNames):
                value = commandResults.get(name)
                row[index] = np.nan if value is None else value

        self._commandIndices[self.count] = commandIndex
        self.count += 1

    def getData(self):
        """Return populated arrays detached from the recording buffers."""
        data = {
            name: self._values[:self.count, index].copy()
            for index, name in enumerate(self.fieldNames)
        }
        data.update({
            "commandIndices": self._commandIndices[:self.count].copy(),
            "count": self.count,
            "capacity": self.capacity,
            "growthCount": self.growthCount,
            "estimatedDuration": self.estimatedDuration,
            "allocationFrameRate": self.frameRate,
            "timestampUnits": "seconds",
            "clockBasis": "assumedSameAsGetSecs",
        })
        return data
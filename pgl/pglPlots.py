################################################################
#   filename: pglPlots.py
#    purpose: Plotting classes which handle visualization of experimental data
#         by: JLG
#       date: November 9, 2026
################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

##############################################
# pglScanTimelinePlot
##############################################
class pglScanTimelinePlot:
    """Compare behavioral and MRI volume timelines.

    proposedMatch accepts one (scanIndex, runIndex) tuple or a list of pairs.
    Each scan and run may occur in at most one pair.

    Alignment:
        clock: Use recorded UTC times.
        first: Independently align each matched scan's first volume to its
            matched run's first behavioral trigger.
        bestMatch: Independently fit each matched scan's translation using
            least-squares differences between ordinal volume pairs.

    Independent alignment preserves each scan's TR and duration, but does
    not preserve the original spacing between scans. Unmatched scans retain
    their recorded UTC times.

    MRI volume times are inferred from acquisition start, volume count, and
    TR. Acquisition start is assumed to represent the first stored volume.

    showTrialSegments displays recorded segment starts from run.getTask(),
    using zero-based segment numbers and consistent colors across runs.
    Extraction uses a temporary behavioral matrix, not a session attachment.

    showBehaviorVolumes controls timeline markers only. Alignment and scatter
    comparisons still use behavioral triggers.

    Input timestamps are not modified.
    """

    ################################################################
    # Read timing, align matched scans, and construct the figure
    ################################################################
    def __init__(self, pglRuns, pglScans, proposedMatch,
                 alignmentMode='clock', volumeMarkerStyle='tick',
                 readHeaders=True, figsize=(16, 12), zoom=False,
                 showTrialSegments=False, showBehaviorVolumes=True):
        import math
        import warnings
        import numpy as np
        import matplotlib.pyplot as plt
        from numbers import Integral

        if alignmentMode not in ('clock', 'first', 'bestMatch'):
            raise ValueError("alignmentMode must be 'clock', 'first', or 'bestMatch'.")
        if volumeMarkerStyle not in ('tick', 'triangle'):
            raise ValueError("volumeMarkerStyle must be 'tick' or 'triangle'.")

        for name, value in (
            ('zoom', zoom),
            ('showTrialSegments', showTrialSegments),
            ('showBehaviorVolumes', showBehaviorVolumes),
        ):
            if type(value) is not bool: raise TypeError(f"{name} must be Boolean.")

        self.runs = list(pglRuns)
        self.scans = list(pglScans)
        self.alignmentMode = alignmentMode
        self.volumeMarkerStyle = volumeMarkerStyle
        self.zoom = zoom
        self.showTrialSegments = showTrialSegments
        self.showBehaviorVolumes = showBehaviorVolumes

        # Normalize one pair without confusing it with a tuple of pairs.
        def isIndex(value):
            return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))

        if isinstance(proposedMatch, tuple) and len(proposedMatch) == 2 and all(isIndex(index) for index in proposedMatch):
            proposedMatch = [proposedMatch]

        self.proposedMatch = []
        usedScans, usedRuns = set(), set()

        for pair in proposedMatch:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each match must be (scanIndex, runIndex).")
            if not all(isIndex(index) for index in pair):
                raise TypeError("Match indices must be integers.")

            scanIndex, runIndex = map(int, pair)
            if not 0 <= scanIndex < len(self.scans): raise IndexError(f"Invalid scan index: {scanIndex}")
            if not 0 <= runIndex < len(self.runs): raise IndexError(f"Invalid run index: {runIndex}")
            if scanIndex in usedScans or runIndex in usedRuns:
                raise ValueError("Each scan and run may appear in only one match.")

            usedScans.add(scanIndex)
            usedRuns.add(runIndex)
            self.proposedMatch.append((scanIndex, runIndex))

        if not self.runs and not self.scans:
            raise ValueError("Provide at least one run or scan.")
        if (zoom or alignmentMode != 'clock') and not self.proposedMatch:
            raise ValueError("Zoom and fitted alignment require at least one proposed match.")

        ################################################################
        # Behavioral times: recorded run clock converted to UTC
        ################################################################
        self.runBlocks = []

        for runIndex, run in enumerate(self.runs):
            data = run.data
            if data is None: raise ValueError(f"Run {runIndex} has no experiment data.")

            start = self.getUtcSeconds(data.startDateTime, f"Run {runIndex} startDateTime")
            duration = data.endTime - data.startTime
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"Run {runIndex} has an invalid duration.")

            volumeTimes = [
                start + self.getEventOffset(event, data.startTime, runIndex)
                for event in data.events if event.type == 'volumeTrigger'
            ]
            if not np.isfinite(volumeTimes).all():
                raise ValueError(f"Run {runIndex} has nonfinite volume times.")
            if len(volumeTimes) > 1 and np.any(np.diff(volumeTimes) <= 0):
                raise ValueError(f"Run {runIndex} volume times must be strictly increasing.")

            self.runBlocks.append({
                'start': start,
                'end': start + duration,
                'volumeTimes': volumeTimes,
                'tr': float(np.median(np.diff(volumeTimes))) if len(volumeTimes) > 1 else None,
                'segmentTimes': {},
                'label': ''
            })

        ################################################################
        # MRI times: acquisition start plus volume index times TR
        ################################################################
        self.scanBlocks = []

        for scanIndex, scan in enumerate(self.scans):
            acquisition = scan.sourceMetadata.get('acquisition', {})
            timestamp = acquisition.get('timestamp') if isinstance(acquisition, dict) else None
            start = self.getUtcSeconds(timestamp, f"Scan {scanIndex} acquisition timestamp")

            summary = scan.getScanSummary(readHeaders=readHeaders)
            dims = summary['scanDims']
            nVolumes = dims[3] if dims is not None and len(dims) > 3 else None
            trSeconds = summary['trSeconds']

            if nVolumes is not None:
                if not isIndex(nVolumes) or nVolumes < 0:
                    raise ValueError(f"Scan {scanIndex} has an invalid volume count.")
                nVolumes = int(nVolumes)

            if trSeconds is not None:
                if isinstance(trSeconds, (bool, np.bool_)) or not math.isfinite(trSeconds) or trSeconds <= 0:
                    raise ValueError(f"Scan {scanIndex} has an invalid TR.")
                trSeconds = float(trSeconds)

            hasTiming = nVolumes is not None and trSeconds is not None
            end = start + nVolumes * trSeconds if hasTiming else None
            volumeTimes = [start + index * trSeconds for index in range(nVolumes)] if hasTiming else []

            if end is not None and not math.isfinite(end):
                raise ValueError(f"Scan {scanIndex} has a nonfinite acquisition end.")

            self.scanBlocks.append({
                'start': start,
                'end': end,
                'volumeTimes': volumeTimes,
                'name': scan.name or f'Scan {scanIndex}',
                'dims': dims,
                'nVolumes': nVolumes,
                'trSeconds': trSeconds,
                'label': ''
            })

        ################################################################
        # Independent scan alignment and shared pair colors
        ################################################################
        self.runColors = {}
        self.scanColors = {}
        self.scanOffsets = {index: 0.0 for index in range(len(self.scans))}
        self.alignmentReports = []
        count = len(self.proposedMatch)
        colorMap = plt.get_cmap('tab20' if count <= 20 else 'hsv')

        print(f"Timeline alignment: {alignmentMode}. Behavioral timestamps remain unchanged.")
        if alignmentMode != 'clock':
            print("  Each matched MRI scan is shifted independently; inter-scan spacing may change.")
        print("  Volume correspondence is ordinal: first ↔ first, second ↔ second, etc.")

        for pairIndex, (scanIndex, runIndex) in enumerate(self.proposedMatch):
            scanBlock = self.scanBlocks[scanIndex]
            runBlock = self.runBlocks[runIndex]
            if scanBlock['end'] is None:
                raise ValueError(f"Matched scan {scanIndex} needs a valid TR and volume count.")

            baseColor = colorMap(pairIndex) if count <= 20 else colorMap(pairIndex / count)
            color = tuple(0.65 * channel + 0.35 for channel in baseColor[:3])
            self.scanColors[scanIndex] = color
            self.runColors[runIndex] = color

            scanTimes = np.asarray(scanBlock['volumeTimes'], dtype=float)
            runTimes = np.asarray(runBlock['volumeTimes'], dtype=float)
            nPairs = min(len(scanTimes), len(runTimes))

            if alignmentMode != 'clock' and not nPairs:
                raise ValueError(f"Scan {scanIndex} / Run {runIndex}: alignment requires at least one volume in each.")

            if len(scanTimes) != len(runTimes):
                warnings.warn(
                    f"Scan {scanIndex} / Run {runIndex}: {len(scanTimes)} MRI volumes "
                    f"versus {len(runTimes)} behavioral triggers. Scatter and bestMatch "
                    f"use the first {nPairs} ordinal pairs. Internal missing/extra "
                    "triggers can invalidate subsequent pairings.",
                    RuntimeWarning, stacklevel=2
                )

            differences = runTimes[:nPairs] - scanTimes[:nPairs]
            offset = 0.0
            if alignmentMode == 'first': offset = float(runTimes[0] - scanTimes[0])
            elif alignmentMode == 'bestMatch': offset = float(np.mean(differences))
            self.scanOffsets[scanIndex] = offset

            residuals = differences - offset
            rmsError = float(np.sqrt(np.mean(residuals ** 2))) if nPairs else None
            maxError = float(np.max(np.abs(residuals))) if nPairs else None

            self.alignmentReports.append({
                'scanIndex': scanIndex,
                'runIndex': runIndex,
                'mode': alignmentMode,
                'pairing': 'ordinal',
                'nPairs': nPairs,
                'mriOffsetSeconds': offset,
                'rmsDifferenceSeconds': rmsError,
                'maxAbsoluteDifferenceSeconds': maxError,
            })

            print(f"  Scan {scanIndex} ↔ Run {runIndex}: {nPairs} pairs; MRI shift={offset:+.6f} s.")
            if alignmentMode == 'first':
                print(f"    Align MRI first volume {scanTimes[0]:.6f} UTC s → behavioral first trigger {runTimes[0]:.6f} UTC s.")
            elif alignmentMode == 'bestMatch':
                print("    Minimize squared differences between paired volume timestamps; do not change TR.")
            if nPairs:
                print(f"    After alignment: RMS={rmsError * 1000:.3f} ms; max absolute difference={maxError * 1000:.3f} ms.")

            behaviorTRText = f"{runBlock['tr']:.3f}s" if runBlock['tr'] is not None else 'unknown'
            runBlock['label'] = f"Run {runIndex}\nnVols: {len(runTimes)}\nTR: {behaviorTRText}"
            scanBlock['label'] = (
                f"Scan {scanIndex}\nnVols: {len(scanTimes)}\n"
                f"TR: {scanBlock['trSeconds']:.3f}s\nShift: {offset:+.3f}s"
            )

        # A single global value is not meaningful when matched shifts differ.
        matchedOffsets = [self.scanOffsets[index] for index, _ in self.proposedMatch]
        self.mriOffset = matchedOffsets[0] if matchedOffsets and len(set(matchedOffsets)) == 1 else None
        if not matchedOffsets: self.mriOffset = 0.0

        ################################################################
        # Optional main-task segment starts
        ################################################################
        self.segmentColors = {}

        if showTrialSegments:
            for runIndex, run in enumerate(self.runs):
                self.runBlocks[runIndex]['segmentTimes'] = self._getTrialSegmentTimes(run, runIndex)

            segmentNumbers = sorted({
                number for block in self.runBlocks for number in block['segmentTimes']
            })
            segmentMap = plt.get_cmap('tab10' if len(segmentNumbers) <= 10 else 'hsv')

            for position, number in enumerate(segmentNumbers):
                self.segmentColors[number] = (
                    segmentMap(position) if len(segmentNumbers) <= 10
                    else segmentMap(position / len(segmentNumbers))
                )

        ################################################################
        # Common origin and one continuous zoom range
        ################################################################
        displayTimes = []

        for block in self.runBlocks:
            displayTimes.extend([block['start'], block['end'], *block['volumeTimes']])
            for times in block['segmentTimes'].values(): displayTimes.extend(times)

        for scanIndex, block in enumerate(self.scanBlocks):
            offset = self.scanOffsets[scanIndex]
            displayTimes.append(block['start'] + offset)
            if block['end'] is not None: displayTimes.append(block['end'] + offset)
            displayTimes.extend(timestamp + offset for timestamp in block['volumeTimes'])

        self.timeOrigin = min(displayTimes)
        duration = max(displayTimes) - self.timeOrigin
        padding = max(duration * 0.02, 1.0)
        self.viewStart, self.viewEnd = -padding, duration + padding
        self.zoomWindows = []

        if zoom:
            firstTimes, lastTimes, scanTRs = [], [], []

            for scanIndex, runIndex in self.proposedMatch:
                block = self.scanBlocks[scanIndex]
                if not block['volumeTimes']:
                    raise ValueError(f"Matched scan {scanIndex} needs at least one volume for zoom.")

                offset = self.scanOffsets[scanIndex]
                firstTimes.append(block['volumeTimes'][0] + offset - self.timeOrigin)
                lastTimes.append(block['volumeTimes'][-1] + offset - self.timeOrigin)
                scanTRs.append(block['trSeconds'])

            padding = 2.0 * max(scanTRs)
            self.viewStart = min(firstTimes) - padding
            self.viewEnd = max(lastTimes) + padding
            self.zoomWindows = [{'start': self.viewStart, 'end': self.viewEnd}]

        self._drawFigure(figsize)

    ################################################################
    # Extract main-task segment starts through the behavioral table
    ################################################################
    def _getTrialSegmentTimes(self, run, runIndex):
        import warnings
        import numpy as np
        from types import SimpleNamespace
        from .pglBehavior import pglBehavior

        # Reuse the run's existing main-task selection rule.
        task = run.getTask()
        if task is None: raise ValueError(f"Run {runIndex}: could not identify the main task.")

        taskIndices = [index for index, candidate in enumerate(run.tasks) if candidate is task]
        if len(taskIndices) != 1:
            raise ValueError(f"Run {runIndex}: main task does not have a unique task index.")
        taskIndex = taskIndices[0]

        # Extract into a temporary container without attaching a matrix to the
        # caller's session. This reads tasks only when segments are requested.
        matrix = pglBehavior.extractSession(SimpleNamespace(runs=[run], sessionID=None))
        table = pglBehavior.getBehaviorTable(
            matrix,
            taskIndex=taskIndex,
            includeAllColumns=True,
            includeTriggers=False,
        )
        if table is None:
            raise ValueError(f"Run {runIndex}: could not extract the main-task behavioral table.")

        taskName = getattr(task.settings, 'taskSaveName', None) or getattr(task.settings, 'taskName', None)
        print(f"  Run {runIndex}: segment ticks from main task {taskIndex} ({taskName!r}).")

        # segmentStartNN is a latency relative to trialStartTime.
        # trialStartTime itself uses the run's recorded acquisition clock.
        runStart = self.getUtcSeconds(run.data.startDateTime, f"Run {runIndex} startDateTime")
        trialStarts = table['trialStartTime'].to_numpy(dtype=float, na_value=np.nan)
        segmentTimes = {}

        for column, metadata in table.attrs.get('columnMetadata', {}).items():
            if metadata.get('sourceGroup') != 'segmentStartProjection': continue

            segmentNumber = int(metadata['segmentNum'])
            latencies = table[column].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(trialStarts) & np.isfinite(latencies)
            times = runStart + (trialStarts[valid] - run.data.startTime) + latencies[valid]
            segmentTimes[segmentNumber] = times.tolist()
            print(f"    Segment {segmentNumber}: {len(times)} recorded starts.")

        diagnostics = table.attrs.get('segmentStartTimingDiagnostics', [])
        if diagnostics:
            warnings.warn(
                f"Run {runIndex}: {len(diagnostics)} segment-start diagnostics. "
                "Missing, ambiguous, or invalid starts are omitted according "
                "to the behavioral table's projection rules.",
                RuntimeWarning, stacklevel=2
            )
        if not segmentTimes:
            warnings.warn(f"Run {runIndex}: no main-task segment starts available.", RuntimeWarning, stacklevel=2)

        return segmentTimes

    ################################################################
    # Draw the three timelines and paired-volume scatter
    ################################################################
    def _drawFigure(self, figsize):
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime, timezone

        self.fig, axes = plt.subplots(4, 1, figsize=figsize)
        axes[1].sharex(axes[0])
        axes[2].sharex(axes[0])

        self.behaviorTimeline = pglTimelinePlot(startTime=self.viewStart, endTime=self.viewEnd, ax=axes[0])
        self.mriTimeline = pglTimelinePlot(startTime=self.viewStart, endTime=self.viewEnd, ax=axes[1])
        self.combinedTimeline = pglTimelinePlot(startTime=self.viewStart, endTime=self.viewEnd, ax=axes[2])
        self.scatterAx = axes[3]
        self.behaviorTimelines = [self.behaviorTimeline]
        self.mriTimelines = [self.mriTimeline]

        self.drawTimeline(
            self.behaviorTimeline, self.runBlocks, self.runColors, 0,
            'Behavioral runs — current run data',
            showVolumes=self.showBehaviorVolumes,
            showSegments=self.showTrialSegments
        )
        self.drawTimeline(
            self.mriTimeline, self.scanBlocks, self.scanColors, self.scanOffsets,
            'MRI scans — inferred volume starts'
        )

        ################################################################
        # Combined timeline: behavior above, MRI below
        ################################################################
        ax = self.combinedTimeline.ax
        ax.set_title('Combined timeline — behavior black / MRI red / segments colored', loc='left', fontsize=11, fontweight='bold')
        ax.set_ylim(-0.35, 0.75)
        ax.set_yticks([])
        ax.axhline(0, color='0.65', linewidth=0.7)

        if self.showBehaviorVolumes:
            times = [
                timestamp - self.timeOrigin
                for block in self.runBlocks for timestamp in block['volumeTimes']
            ]
            self._drawVolumeMarkers(ax, times, 'black', 'Behavior volumes', above=True)

        times = [
            timestamp + self.scanOffsets[index] - self.timeOrigin
            for index, block in enumerate(self.scanBlocks)
            for timestamp in block['volumeTimes']
        ]
        self._drawVolumeMarkers(ax, times, 'red', 'MRI volumes', above=False)

        if self.showTrialSegments:
            self._drawSegmentMarkers(ax, self.runBlocks, 0.32, 0.58)

        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=min(5, len(handles)))

        ################################################################
        # Scatter uses the same independently shifted scan timestamps
        ################################################################
        self.volumeComparisons = []
        allMriTimes, allBehaviorTimes = [], []

        for scanIndex, runIndex in self.proposedMatch:
            offset = self.scanOffsets[scanIndex]
            scanTimes = np.asarray(self.scanBlocks[scanIndex]['volumeTimes'], dtype=float) + offset - self.timeOrigin
            runTimes = np.asarray(self.runBlocks[runIndex]['volumeTimes'], dtype=float) - self.timeOrigin
            nPairs = min(len(scanTimes), len(runTimes))
            pairedMri, pairedBehavior = scanTimes[:nPairs], runTimes[:nPairs]

            self.volumeComparisons.append({
                'scanIndex': scanIndex,
                'runIndex': runIndex,
                'volumeNumbers': np.arange(1, nPairs + 1),
                'mriTimes': pairedMri.copy(),
                'behaviorTimes': pairedBehavior.copy(),
                'differenceSeconds': pairedBehavior - pairedMri,
                'unpairedMriTimes': scanTimes[nPairs:].copy(),
                'unpairedBehaviorTimes': runTimes[nPairs:].copy(),
                'timeOriginUtcSeconds': self.timeOrigin,
                'mriOffsetSeconds': offset,
            })

            if not nPairs: continue
            self.scatterAx.scatter(pairedMri, pairedBehavior, s=14, alpha=0.8, label=f"Scan {scanIndex} / Run {runIndex}")
            allMriTimes.extend(pairedMri)
            allBehaviorTimes.extend(pairedBehavior)

        ax = self.scatterAx
        ax.set_title('Paired volume timestamps — ordinal correspondence', loc='left', fontsize=11, fontweight='bold')
        ax.set_xlabel('MRI volume time (sec relative to display origin)')
        ax.set_ylabel('Behavioral trigger time\n(sec relative to display origin)')
        ax.grid(True, alpha=0.2)
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)

        if allMriTimes:
            lower = min(min(allMriTimes), min(allBehaviorTimes))
            upper = max(max(allMriTimes), max(allBehaviorTimes))
            padding = max((upper - lower) * 0.02, 1.0)
            limits = (lower - padding, upper + padding)
            ax.plot(limits, limits, '--', color='0.5', linewidth=1, label='Equal timestamps', zorder=0)
            ax.set_xlim(*limits)
            ax.set_ylim(*limits)
            ax.legend(loc='best', fontsize=8, ncol=min(3, len(self.proposedMatch) + 1))
        else:
            ax.text(0.5, 0.5, 'No paired volumes available.', transform=ax.transAxes, ha='center', va='center')

        ################################################################
        # Shared timeline limits and annotations
        ################################################################
        for timeline in (self.behaviorTimeline, self.mriTimeline, self.combinedTimeline):
            timeline.ax.set_xlim(self.viewStart, self.viewEnd)
            timeline.ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            if self.zoom:
                for text in timeline.ax.texts: text.set_clip_on(True)

        for timeline in (self.behaviorTimeline, self.mriTimeline):
            timeline.ax.set_xlabel('')
            timeline.ax.tick_params(axis='x', labelbottom=False)

        originText = datetime.fromtimestamp(self.timeOrigin, tz=timezone.utc).isoformat(timespec='milliseconds')
        self.combinedTimeline.ax.set_xlabel(f'Display time (sec) relative to {originText}')

        alignmentText = 'recorded UTC' if self.alignmentMode == 'clock' else 'independent shift per matched scan'
        zoomText = ' | Continuous matched-scan zoom' if self.zoom else ''
        self.fig.suptitle(
            'Behavioral / MRI correspondence\n'
            f'Alignment: {self.alignmentMode} — {alignmentText}{zoomText}',
            fontsize=13, fontweight='bold'
        )
        self.fig.text(
            0.01, 0.012,
            'Scan, run, and segment indices are zero-based. Gray blocks = unmatched. MRI volume times are inferred.\n'
            'Scatter pairs volumes by order. Segment ticks are recorded segment STARTs, not verified stimulus onsets.',
            fontsize=8, color='0.35'
        )
        self.fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    ################################################################
    # Draw block backgrounds with optional volume and segment markers
    ################################################################
    def drawTimeline(self, timeline, blocks, colors, offset, title,
                     showVolumes=True, showSegments=False):
        from matplotlib.patches import Rectangle

        ax = timeline.ax
        ax.set_ylim(-0.4, 1.45)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        acquisitionTimes = []

        for index, block in enumerate(blocks):
            blockOffset = offset[index] if isinstance(offset, dict) else offset
            start = block['start'] + blockOffset - self.timeOrigin

            if block['end'] is None:
                ax.vlines(start, -0.25, 1.05, color='0.55', linewidth=1.5, linestyles='dashed', zorder=1)
                continue

            end = block['end'] + blockOffset - self.timeOrigin
            ax.add_patch(Rectangle(
                (start, -0.28), end - start, 1.38,
                facecolor=colors.get(index, 'lightgray'),
                edgecolor='white', linewidth=1, alpha=0.65, zorder=0
            ))

            if index in colors:
                ax.text((start + end) / 2, 1.04, block['label'], ha='center', va='top', fontsize=9, zorder=3)

            if showVolumes:
                acquisitionTimes.extend(timestamp + blockOffset - self.timeOrigin for timestamp in block['volumeTimes'])

        if showVolumes:
            self._drawVolumeMarkers(ax, acquisitionTimes, 'black', None, above=False)

        if showSegments:
            self._drawSegmentMarkers(ax, blocks, 0.08, 0.32)
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=min(5, len(handles)))

    ################################################################
    # Draw efficient volume tick or triangle collections
    ################################################################
    def _drawVolumeMarkers(self, ax, times, color, label, above=False):
        import numpy as np

        if self.volumeMarkerStyle == 'tick':
            bottom, top = (0, 0.22) if above else (-0.22, 0)
            ax.vlines(times, bottom, top, color=color, linewidth=0.75, label=label, zorder=3)
        else:
            direction = 1 if above else -1
            marker = [(0, 0), (-0.5, direction), (0.5, direction), (0, 0)]
            ax.scatter(times, np.zeros(len(times)), marker=marker, s=24, color=color, edgecolors='none', label=label, zorder=3)

    ################################################################
    # Draw consistently colored segment starts without shifting behavior
    ################################################################
    def _drawSegmentMarkers(self, ax, blocks, bottom, top):
        for segmentNumber, color in self.segmentColors.items():
            times = [
                timestamp - self.timeOrigin
                for block in blocks
                for timestamp in block.get('segmentTimes', {}).get(segmentNumber, [])
            ]
            if times:
                ax.vlines(times, bottom, top, color=color, linewidth=1.1, label=f"Segment {segmentNumber}", zorder=4)

    ################################################################
    # Parse timezone-aware acquisition timestamps
    ################################################################
    @staticmethod
    def getUtcSeconds(value, description):
        from datetime import datetime, timezone

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{description} is missing.")

        try:
            parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError(f"{description} is not a valid ISO timestamp.") from error

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{description} needs an explicit timezone, such as Z or +00:00.")

        return parsed.astimezone(timezone.utc).timestamp()

    ################################################################
    # Convert one recorded behavioral timestamp to a run-relative offset
    ################################################################
    @staticmethod
    def getEventOffset(event, startTime, runIndex):
        import math
        import numpy as np
        from numbers import Real

        timestamp = event.timestamp
        for name, value in (('event timestamp', timestamp), ('startTime', startTime)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"Run {runIndex} has an invalid {name}.")

        return float(timestamp) - float(startTime)

    ################################################################
    # Return scan-specific offsets; there may be no single global shift
    ################################################################
    def getAlignmentOffset(self, scanIndex=None):
        """Return one scan's shift, or all scan shifts as an index-keyed dict."""
        if scanIndex is None: return dict(self.scanOffsets)
        if scanIndex not in self.scanOffsets: raise IndexError(f"Invalid scan index: {scanIndex}")
        return self.scanOffsets[scanIndex]

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()

    def save(self, filename, dpi=300):
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')

##############################################
# timelinePlot
##############################################
class pglTimelinePlot:
    """
    A timeline visualization with triangular event markers and vertical line markers.
    
    Usage:
        timeline = TimelinePlot(startTime=0, endTime=100)
        timeline.addTriangleMarker(time=10, color='red', label='Start')
        timeline.addVerticalMarker(time=50, color='blue', label='Checkpoint', labelSide='right')
        timeline.show()
    """
    
    def __init__(self, startTime=0, endTime=100, figsize=(12, 4), ax=None):
        """
        Initialize the timeline plot.
        
        Args:
            startTime (float): Start time for the timeline
            endTime (float): End time for the timeline
            figsize (tuple): Figure size (width, height)
        """
        self.startTime = startTime
        self.endTime = endTime
        
        # Create figure and axis
        if ax:
            self.fig = ax.figure
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Setup the timeline axis
        self.ax.set_xlim(startTime, endTime)
        self.ax.set_ylim(-1, 2)  # Room for markers above and below
        
        # Draw the main timeline (horizontal line)
        self.ax.axhline(y=0, color='black', linewidth=2)
        
        # Axis labels
        self.ax.set_xlabel('Time (sec)', fontsize=12)
        self.ax.set_yticks([])  # Hide y-axis ticks
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        
        # Storage for markers (for legend if needed)
        self.markers = []
 
    def addTriangleMarker(self, time, color='red', label='', labelOffset=0.3, 
                          markerSize=10, fontsize=10, direction='down'):
        """
        Add a triangle marker at a specific time with tip touching the timeline.
        
        Args:
            time (float): Time position for the marker
            color (str): Color of the triangle
            label (str): Text label for the triangle
            labelOffset (float): Vertical offset for the label from the triangle edge
            markerSize (float): Size of the triangle (height in data units)
            fontsize (int): Font size for the label
            direction (str): 'down' for downward-pointing (▼) or 'up' for upward-pointing (▲)
        """
        # Convert markerSize to data coordinates (approximate)
        height = markerSize * 0.02  # Scale factor for visual size
        width = height * 0.8  # Make it slightly narrower
        
        # Create triangle vertices based on direction
        if direction == 'down':
            # Downward triangle: tip at timeline, base above
            vertices = [
                [time, 0],                    # Tip at timeline
                [time - width/2, height],     # Top left
                [time + width/2, height],     # Top right
            ]
            labelY = height + labelOffset
            labelVa = 'bottom'
        else:  # direction == 'up'
            # Upward triangle: tip at timeline, base below
            vertices = [
                [time, 0],                    # Tip at timeline
                [time - width/2, -height],    # Bottom left
                [time + width/2, -height],    # Bottom right
            ]
            labelY = -height - labelOffset
            labelVa = 'top'
        
        # Draw triangle as polygon
        triangle = patches.Polygon(vertices, closed=True, 
                                  facecolor=color, edgecolor='black', 
                                  linewidth=0.5, clip_on=False)
        self.ax.add_patch(triangle)
        
        # Add label if provided
        if label:
            self.ax.text(time, labelY, label, 
                        ha='center', va=labelVa, fontsize=fontsize,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.8))
        
        self.markers.append({'type': 'triangle', 'time': time, 'label': label, 
                            'color': color, 'direction': direction})
    
    def addVerticalMarker(self, time, color='blue', label='', labelSide='right',
                          lineHeight=1.5, linewidth=2, fontsize=9, rotation=90):
        """
        Add a vertical line marker at a specific time.
        
        Args:
            time (float): Time position for the marker
            color (str): Color of the vertical line
            label (str): Text label for the marker
            labelSide (str): 'left' or 'right' - side for the label
            lineHeight (float): Height of the vertical line
            linewidth (float): Width of the vertical line
            fontsize (int): Font size for the label
            rotation (int): Text rotation (90 for vertical, 0 for horizontal)
        """
        # Draw vertical line
        yBottom = -lineHeight / 2
        yTop = lineHeight / 2
        self.ax.plot([time, time], [yBottom, yTop], 
                    color=color, linewidth=linewidth)
        
        # Add label if provided
        if label:
            # Position label on left or right
            if labelSide == 'right':
                xOffset = 0.02 * (self.endTime - self.startTime)  # 2% of range
                ha = 'left'
            else:  # left
                xOffset = -0.02 * (self.endTime - self.startTime)
                ha = 'right'
            
            self.ax.text(time + xOffset, 0, label,
                        ha=ha, va='center', fontsize=fontsize,
                        rotation=rotation, color=color)
        
        self.markers.append({'type': 'vertical', 'time': time, 'label': label, 'color': color})
    
    def addTimeRange(self, start, end, color='lightgray', alpha=0.3, label=''):
        """
        Add a shaded time range (useful for highlighting periods).
        
        Args:
            start (float): Start time of the range
            end (float): End time of the range
            color (str): Color of the shaded region
            alpha (float): Transparency (0-1)
            label (str): Optional label for the range
        """
        self.ax.axvspan(start, end, color=color, alpha=alpha, label=label)
    
    def setTitle(self, title, fontsize=14):
        """Set the plot title."""
        self.ax.set_title(title, fontsize=fontsize, fontweight='bold')
    
    def addLegend(self, items, location='upper right', fontsize=10):
        """
        Add a legend with colored text labels (no symbols).
        
        Args:
            items (list): List of dicts with 'label' and 'color' keys
                         Example: [{'label': 'Keypress', 'color': 'red'}, ...]
            location (str): Legend location ('upper right', 'upper left', 'lower right', 'lower left', etc.)
            fontsize (int): Font size for legend text
        """
        from matplotlib.lines import Line2D
        
        # Create dummy line objects with the colors
        handles = []
        labels = []
        
        for item in items:
            # Create invisible line with the desired color
            handle = Line2D([0], [0], marker='', linestyle='', 
                          markersize=0, color=item['color'])
            handles.append(handle)
            labels.append(item['label'])
        
        # Create legend
        legend = self.ax.legend(handles, labels, loc=location, 
                               fontsize=fontsize, framealpha=0.9,
                               handlelength=0, handletextpad=0.5,
                               labelcolor='linecolor')  # KEY: Use line color for labels
        
        # Make text bold (optional)
        for text in legend.get_texts():
            text.set_weight('bold') 
    
    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()
    
    def save(self, filename, dpi=300):
        """
        Save the plot to a file.
        
        Args:
            filename (str): Output filename
            dpi (int): Resolution in dots per inch
        """
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        pglMessages.message(f"Timeline saved to {filename}")
    
    def getMarkers(self):
        """Return list of all markers added."""
        return self.markers
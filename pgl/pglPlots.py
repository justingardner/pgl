################################################################
#   filename: pglPlots.py
#    purpose: Plotting classes which handle visualization of experimental data
#         by: JLG
#       date: November 9, 2026
################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches

##############################################
# pglScanTimelinePlot
##############################################
class pglScanTimelinePlot:
    """Plot behavioral runs and MRI scans using a proposed correspondence.

    proposedMatch contains zero-based (scanIndex, runIndex) pairs.
    Each scan and run may appear in at most one pair.

    alignmentMode:
        'clock': Place both timelines using their recorded UTC times.
        'bestMatch': Apply one global MRI shift that minimizes squared
            differences between corresponding block starts and ends.

    Behavioral ticks use recorded volumeTrigger events.
    MRI ticks are inferred from acquisition start, volume count, and TR.
    This assumes acquisition start corresponds to the first stored volume.

    Input objects and their timestamps are not modified.
    """

    def __init__(self, pglRuns, pglScans, proposedMatch,
                 alignmentMode='clock', volumeMarkerStyle='tick',
                 readHeaders=True, figsize=(16, 7)):
        import math
        from datetime import datetime, timezone

        if alignmentMode not in ('clock', 'bestMatch'):
            raise ValueError("alignmentMode must be 'clock' or 'bestMatch'.")

        if volumeMarkerStyle not in ('tick', 'triangle'):
            raise ValueError("volumeMarkerStyle must be 'tick' or 'triangle'.")

        self.runs = list(pglRuns)
        self.scans = list(pglScans)
        self.proposedMatch = [tuple(pair) for pair in proposedMatch]
        self.alignmentMode = alignmentMode
        self.volumeMarkerStyle = volumeMarkerStyle

        if not self.runs and not self.scans:
            raise ValueError("Provide at least one run or scan.")

        # Validate correspondence before reading any run or scan metadata.
        usedScans = set()
        usedRuns = set()

        for pair in self.proposedMatch:
            if len(pair) != 2:
                raise ValueError("Each match must be (scanIndex, runIndex).")

            scanIndex, runIndex = pair
            if type(scanIndex) is not int or type(runIndex) is not int:
                raise TypeError("Match indices must be integers.")

            if not 0 <= scanIndex < len(self.scans):
                raise IndexError(f"Invalid scan index: {scanIndex}")

            if not 0 <= runIndex < len(self.runs):
                raise IndexError(f"Invalid run index: {runIndex}")

            if scanIndex in usedScans or runIndex in usedRuns:
                raise ValueError("Each scan and run may appear in only one match.")

            usedScans.add(scanIndex)
            usedRuns.add(runIndex)

        # Read behavioral data, but not tasks or other unnecessary contents.
        self.runBlocks = []

        for runIndex, run in enumerate(self.runs):
            data = run.data
            if data is None:
                raise ValueError(f"Run {runIndex} has no experiment data.")

            start = self.getUtcSeconds(
                data.startDateTime, f"Run {runIndex} startDateTime"
            )
            duration = data.endTime - data.startTime

            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"Run {runIndex} has an invalid duration.")

            # getsecs values are meaningful relative to this run's startTime.
            volumeTimes = [
                start + self.getEventOffset(event, data.startTime, runIndex)
                for event in data.events
                if event.type == 'volumeTrigger'
            ]

            self.runBlocks.append({
                'start': start,
                'end': start + duration,
                'volumeTimes': volumeTimes,
                'label': ''
            })

        # Read acquisition metadata and geometry, never scan.data.
        self.scanBlocks = []

        for scanIndex, scan in enumerate(self.scans):
            acquisition = scan.sourceMetadata.get('acquisition', {})
            timestamp = (
                acquisition.get('timestamp')
                if isinstance(acquisition, dict) else None
            )
            start = self.getUtcSeconds(
                timestamp, f"Scan {scanIndex} acquisition timestamp"
            )

            summary = scan.getScanSummary(readHeaders=readHeaders)
            dims = summary['scanDims']
            nVolumes = dims[3] if dims is not None else None
            trSeconds = summary['trSeconds']

            # Without TR and a volume count, do not invent a duration.
            hasTiming = nVolumes is not None and trSeconds is not None
            end = start + nVolumes * trSeconds if hasTiming else None
            volumeTimes = (
                [start + index * trSeconds for index in range(nVolumes)]
                if hasTiming else []
            )

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

        # Assign the same muted background to each corresponding pair.
        self.runColors = {}
        self.scanColors = {}
        pairOffsets = []
        count = len(self.proposedMatch)
        colorMap = plt.get_cmap('tab20' if count <= 20 else 'hsv')

        for pairIndex, (scanIndex, runIndex) in enumerate(self.proposedMatch):
            runBlock = self.runBlocks[runIndex]
            scanBlock = self.scanBlocks[scanIndex]

            if scanBlock['end'] is None:
                raise ValueError(
                    f"Matched scan {scanIndex} needs a valid TR and volume "
                    "count to establish its acquisition duration."
                )

            baseColor = (
                colorMap(pairIndex) if count <= 20
                else colorMap(pairIndex / count)
            )
            color = tuple(0.65 * channel + 0.35 for channel in baseColor[:3])
            self.runColors[runIndex] = color
            self.scanColors[scanIndex] = color

            dimsText = ' × '.join(str(value) for value in scanBlock['dims'][:3])
            scanInfo = (
                f"{scanBlock['name']}\n"
                f"{dimsText} | {scanBlock['nVolumes']} volumes | "
                f"TR {scanBlock['trSeconds']:g} sec"
            )

            # Include MRI information in both corresponding backgrounds.
            # Counts remain explicit so mismatches are visible.
            scanBlock['label'] = f"Scan {scanIndex}\n{scanInfo}"
            runBlock['label'] = (
                f"Run {runIndex} ↔ Scan {scanIndex} | "
                f"{len(runBlock['volumeTimes'])} triggers\n{scanInfo}"
            )

            startOffset = runBlock['start'] - scanBlock['start']
            endOffset = runBlock['end'] - scanBlock['end']
            pairOffsets.append((startOffset + endOffset) / 2)

        # One global translation preserves spacing and duration differences.
        self.mriOffset = 0.0

        if alignmentMode == 'bestMatch':
            if not pairOffsets:
                raise ValueError("Best-match alignment requires a matched pair.")
            self.mriOffset = sum(pairOffsets) / len(pairOffsets)

        # Choose a common display origin after applying the global shift.
        # Include events outside run boundaries rather than clipping them.
        displayTimes = []

        for blocks, offset in (
                (self.runBlocks, 0),
                (self.scanBlocks, self.mriOffset)):
            for block in blocks:
                displayTimes.append(block['start'] + offset)
                if block['end'] is not None:
                    displayTimes.append(block['end'] + offset)
                displayTimes.extend(time + offset for time in block['volumeTimes'])

        self.timeOrigin = min(displayTimes)
        displayDuration = max(displayTimes) - self.timeOrigin
        padding = max(displayDuration * 0.02, 1.0)

        self.fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        self.behaviorTimeline = pglTimelinePlot(
            startTime=-padding,
            endTime=displayDuration + padding,
            ax=axes[0]
        )
        self.mriTimeline = pglTimelinePlot(
            startTime=-padding,
            endTime=displayDuration + padding,
            ax=axes[1]
        )

        self.drawTimeline(
            self.behaviorTimeline, self.runBlocks, self.runColors, 0,
            'Behavioral runs — recorded volume triggers'
        )
        self.drawTimeline(
            self.mriTimeline, self.scanBlocks, self.scanColors, self.mriOffset,
            'MRI scans — volume times inferred from acquisition start and TR'
        )

        originText = datetime.fromtimestamp(
            self.timeOrigin, tz=timezone.utc
        ).isoformat(timespec='milliseconds')

        self.behaviorTimeline.ax.set_xlabel('')
        self.mriTimeline.ax.set_xlabel(
            f'Display time (sec) relative to {originText}'
        )

        self.fig.suptitle(
            'Proposed behavioral / MRI correspondence\n'
            f'Alignment: {alignmentMode} | MRI shift: {self.mriOffset:+.3f} sec',
            fontsize=13,
            fontweight='bold'
        )

        self.fig.text(
            0.01, 0.015,
            'Indices are zero-based. Gray = unmatched. '
            'Dashed gray start marker = scan duration unknown.',
            fontsize=9,
            color='0.35'
        )
        self.fig.tight_layout(rect=(0, 0.06, 1, 0.90))

    @staticmethod
    def getUtcSeconds(value, description):
        """Parse an explicit timezone-aware ISO timestamp as UTC seconds."""
        from datetime import datetime, timezone

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{description} is missing.")

        try:
            parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError as error:
            raise ValueError(f"{description} is not a valid ISO timestamp.") from error

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(
                f"{description} needs an explicit timezone, such as Z or +00:00."
            )

        return parsed.astimezone(timezone.utc).timestamp()

    @staticmethod
    def getEventOffset(event, startTime, runIndex):
        """Validate and return one behavioral trigger's relative time."""
        import math

        timestamp = event.timestamp
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
        ):
            raise ValueError(f"Run {runIndex} has an invalid volume-trigger timestamp.")

        return timestamp - startTime

    def drawTimeline(self, timeline, blocks, colors, offset, title):
        """Draw backgrounds and batched acquisition markers on one panel."""
        ax = timeline.ax
        ax.set_ylim(-0.4, 1.25)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        acquisitionTimes = []

        for index, block in enumerate(blocks):
            start = block['start'] + offset - self.timeOrigin

            if block['end'] is None:
                # Unknown duration: show its known start without inventing
                # a block width or claiming a measured acquisition tick.
                ax.vlines(
                    start, -0.25, 1.05,
                    color='0.55', linewidth=1.5, linestyles='dashed',
                    zorder=1
                )
                continue

            end = block['end'] + offset - self.timeOrigin
            color = colors.get(index, 'lightgray')

            rectangle = patches.Rectangle(
                (start, -0.28), end - start, 1.38,
                facecolor=color,
                edgecolor='white',
                linewidth=1,
                alpha=0.65,
                zorder=0
            )
            ax.add_patch(rectangle)

            # Unmatched blocks intentionally have no information text.
            if index in colors:
                ax.text(
                    (start + end) / 2, 1.04, block['label'],
                    ha='center', va='top', fontsize=8,
                    zorder=3
                )

            acquisitionTimes.extend(
                time + offset - self.timeOrigin
                for time in block['volumeTimes']
            )

        if self.volumeMarkerStyle == 'tick':
            # One line collection keeps large numbers of triggers efficient.
            ax.vlines(
                acquisitionTimes, -0.18, 0,
                color='0.2', linewidth=0.65, zorder=3
            )
        else:
            # Point-sized triangles keep their visual size when zooming.
            # Tips touch the timeline, with bases underneath.
            triangleMarker = [(0, 0), (-0.5, -1), (0.5, -1), (0, 0)]
            ax.scatter(
                acquisitionTimes,
                [0] * len(acquisitionTimes),
                marker=triangleMarker,
                s=24,
                facecolors='0.2',
                edgecolors='none',
                zorder=3
            )

    def getAlignmentOffset(self):
        """Return the global offset added to MRI UTC times, in seconds."""
        return self.mriOffset

    def show(self):
        """Display the plot."""
        plt.show()

    def save(self, filename, dpi=300):
        """Save this figure without relying on the currently active figure."""
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
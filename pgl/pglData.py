################################################################
#   filename: pglData.py
#    purpose: Classes that handle serializing and deserializing data
#         by: JLG
#       date: Feb 18, 2026
################################################################

##############
# Imports
##############
import re
import numpy as np
import pandas as pd
import xarray as xr
from traitlets import Any, validate
from .pglMessages import pglMessages
from .pglPipeline import pglActionable

##################################
# pglDataMatrix
##################################
class pglDataMatrix(pglActionable):
    """
    Generic wrapper around an xarray DataArray or Dataset.

    Makes no assumptions about dimensions, coordinates, or data sources.
    Missing public attributes and indexing delegate to the underlying xarray.
    Delegated operations return native xarray objects.

    Use .data for arithmetic or functions requiring an actual xarray object.
    Attributes defined by pglActionable or this wrapper take precedence.
    """

    data = Any(default_value=None, allow_none=True, serialize=False, help="Underlying xarray DataArray or Dataset")

    def __init__(self, data=None):
        super().__init__()
        self.data = data

    @validate("data")
    def validateData(self, proposal):
        data = proposal["value"]
        if data is not None and not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise TypeError("data must be an xarray.DataArray, xarray.Dataset, or None")
        return data

    def requireData(self):
        """Return the underlying xarray object, or raise if empty."""
        if self.data is None:
            raise ValueError("No data have been assigned")
        return self.data

    def __getattr__(self, attrName):
        """Delegate missing public attributes to xarray."""
        if attrName.startswith("_"):
            raise AttributeError(f"{type(self).__name__!r} has no attribute {attrName!r}")

        try:
            data = object.__getattribute__(self, "data")
        except AttributeError:
            data = None

        if data is not None:
            try:
                return getattr(data, attrName)
            except AttributeError:
                pass

        raise AttributeError(f"{type(self).__name__!r} has no attribute {attrName!r}")

    def __getitem__(self, key):
        return self.requireData()[key]

    def __setitem__(self, key, value):
        self.requireData()[key] = value

    def __len__(self):
        return len(self.requireData())

    def __iter__(self):
        return iter(self.requireData())

    def __contains__(self, key):
        return key in self.requireData()

    def __repr__(self):
        if self.data is None:
            return f"{type(self).__name__}(empty)"
        return f"{type(self).__name__}\n{self.data!r}"


##################################
# pglEpochsDataMatrix
##################################
class pglEpochsDataMatrix(pglDataMatrix):
    """
    MNE-specific import of trial × sensor × time data.

    Quality coordinates:
        trialBad: Trial fails configured MNE rejection or contains non-finite
                  signal values in imported sensors.
        sensorBad: Sensor is marked bad in MNE Info.
        timeBad: Relative time point contains a non-finite value anywhere in
                 the returned matrix. This does not localize artifacts.

    dropBadData=True removes marked bad data sensors, then rejected or
    non-finite trials. dropBadData=False retains available data with labels.

    Only MNE data channels are imported; auxiliary channels are excluded.
    Requires preloaded epochs. Previously dropped trials cannot be restored.
    Annotation rejection depends on how the source epochs were constructed.

    Clean import guarantees finite values and respects configured rejection,
    but does not detect unmarked artifacts. Later mutation can invalidate
    these guarantees.
    """

    @classmethod
    def fromEpochs(cls, epochs, matrixName="epochs", verbose=True, dropBadData=True):
        """
        Import MNE epochs without modifying the source.

        Trial IDs come from epochs.selection. Metadata columns become
        trial-prefixed camelCap coordinates, such as trialCondition.

        Import counts describe currently available epochs, not trials
        already removed upstream.
        """
        import mne

        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError("epochs must be an mne.BaseEpochs instance")

        if not epochs.preload:
            raise ValueError("Epochs must be preloaded. To retain rejected trials, preload without rejection thresholds and configure those thresholds afterward.")

        inputTrialCount = len(epochs.events)
        if inputTrialCount == 0:
            raise ValueError("No epochs are available")

        if verbose:
            importMode = "dropping bad data" if dropBadData else "retaining data with bad labels"
            pglMessages.message(f"Importing MNE epochs: {importMode}.")

        # Evaluate configured rejection separately, with all source channels.
        checkedEpochs = epochs.copy()
        checkedEpochs.drop_bad(verbose=False)
        mneTrialBad = ~np.isin(epochs.selection, checkedEpochs.selection)
        del checkedEpochs

        # Preserve currently available, preloaded trials on this separate copy.
        importEpochs = epochs.copy()
        importEpochs.drop_bad(reject=None, flat=None, verbose=False)

        if not np.array_equal(importEpochs.selection, epochs.selection):
            raise ValueError("Some source epochs could not be retained; cannot safely align their quality labels")

        inputSensorCount = len(importEpochs.ch_names)
        markedBadNames = set(importEpochs.info["bads"])

        # Initially include bad data sensors so they can be counted and labeled.
        # MNE raises if no data channels are available.
        importEpochs.pick(picks="data", exclude=())

        sensorNames = np.asarray(importEpochs.ch_names, dtype=str)
        sensorTypes = np.asarray(importEpochs.get_channel_types(), dtype=str)
        sensorBad = np.isin(sensorNames, list(markedBadNames))
        auxiliarySensorCount = inputSensorCount - len(sensorNames)
        badSensorCount = int(sensorBad.sum())
        dataValues = importEpochs.get_data(copy=True, verbose=False)

        if not np.array_equal(importEpochs.selection, epochs.selection):
            raise ValueError("Reading data changed the available trials; cannot safely align their quality labels")

        if any(size == 0 for size in dataValues.shape):
            raise ValueError("No trial × sensor × time data are available")

        # Discarded sensors must not invalidate otherwise usable trials.
        sensorMask = ~sensorBad if dropBadData else np.ones(len(sensorNames), dtype=bool)

        if not sensorMask.any():
            raise ValueError("No good data sensors remain")

        if not sensorMask.all():
            dataValues = dataValues[:, sensorMask, :]

        sensorNames = sensorNames[sensorMask]
        sensorTypes = sensorTypes[sensorMask]
        sensorBad = sensorBad[sensorMask]

        nonFiniteTrialBad = ~np.isfinite(dataValues).all(axis=(1, 2))
        trialBad = mneTrialBad | nonFiniteTrialBad
        badTrialCount = int(trialBad.sum())
        mneBadTrialCount = int(mneTrialBad.sum())
        nonFiniteTrialCount = int(nonFiniteTrialBad.sum())
        trialMask = ~trialBad if dropBadData else np.ones(inputTrialCount, dtype=bool)

        if not trialMask.any():
            raise ValueError("No good trials remain")

        if not trialMask.all():
            dataValues = dataValues[trialMask]

        # Aggregate non-finite values over retained trials and sensors.
        timeBad = ~np.isfinite(dataValues).all(axis=(0, 1))

        coords = {
            "trial": ("trial", importEpochs.selection[trialMask].copy()),
            "sensor": ("sensor", sensorNames),
            "time": ("time", importEpochs.times.copy()),
            "trialBad": ("trial", trialBad[trialMask]),
            "sensorBad": ("sensor", sensorBad),
            "timeBad": ("time", timeBad),
            "eventSample": ("trial", importEpochs.events[trialMask, 0].copy()),
            "eventCode": ("trial", importEpochs.events[trialMask, 2].copy()),
            "sensorType": ("sensor", sensorTypes),
        }

        metadataColumnMap = {}

        if importEpochs.metadata is not None:
            trialMetadata = importEpochs.metadata.iloc[np.flatnonzero(trialMask)]

            if not trialMetadata.columns.is_unique:
                raise ValueError("Metadata column names must be unique")

            for columnName in trialMetadata.columns:
                coordName = cls.getMetadataCoordName(columnName)

                if coordName in coords:
                    raise ValueError(f"Metadata column {columnName!r} produces a duplicate coordinate: {coordName!r}")

                coords[coordName] = ("trial", trialMetadata[columnName].to_numpy(copy=True))
                metadataColumnMap[coordName] = str(columnName)

        attrs = {
            "source": "mne.Epochs",
            "samplingFrequency": float(importEpochs.info["sfreq"]),
            "eventId": dict(importEpochs.event_id),
            "metadataColumnMap": metadataColumnMap,
            "dropBadData": bool(dropBadData),
            "inputTrialCount": inputTrialCount,
            "markedBadSensorCount": len(markedBadNames),
            "badSensorCount": badSensorCount,
            "badTrialCount": badTrialCount,
            "mneBadTrialCount": mneBadTrialCount,
            "nonFiniteTrialCount": nonFiniteTrialCount,
            "excludedAuxiliarySensorCount": auxiliarySensorCount,
        }

        data = xr.DataArray(dataValues, dims=("trial", "sensor", "time"), coords=coords, name=matrixName, attrs=attrs)

        data.coords["time"].attrs.update({"units": "s", "description": "Time relative to the epoch event"})
        data.coords["trial"].attrs["description"] = "Original retained epoch index from MNE epochs.selection"
        data.coords["eventSample"].attrs["description"] = "Event sample number in MNE recording sample coordinates"
        data.coords["trialBad"].attrs["description"] = "True for MNE-rejected trials or trials with non-finite values"
        data.coords["sensorBad"].attrs["description"] = "True for sensors marked bad in MNE Info"
        data.coords["timeBad"].attrs["description"] = "True where any retained trial/sensor has a non-finite value"

        # MNE-specific validation stays outside the generic constructor.
        cls.validateEpochsData(data, requireClean=dropBadData)
        matrix = cls(data)

        if verbose:
            matrix.printImportSummary()

        return matrix

    @staticmethod
    def validateEpochsData(data, requireClean=False):
        """Validate MNE-specific dimensions and quality coordinates."""
        if not isinstance(data, xr.DataArray):
            raise TypeError("Epochs data must be an xarray.DataArray")

        if data.dims != ("trial", "sensor", "time"):
            raise ValueError("Epochs data must have dimensions ('trial', 'sensor', 'time')")

        if any(size == 0 for size in data.shape):
            raise ValueError("Epochs data cannot have empty dimensions")

        if not np.isfinite(data.coords["time"].values).all():
            raise ValueError("Time coordinates must be finite")

        for coordName, dimName in (("trialBad", "trial"), ("sensorBad", "sensor"), ("timeBad", "time")):
            if coordName not in data.coords:
                raise ValueError(f"Missing quality coordinate: {coordName}")

            coord = data.coords[coordName]

            if coord.dims != (dimName,):
                raise ValueError(f"{coordName} must use the {dimName!r} dimension")

            if coord.dtype != np.dtype(bool):
                raise TypeError(f"{coordName} must contain boolean values")

            if requireClean and coord.values.any():
                raise ValueError(f"Clean epochs data cannot contain True in {coordName}")

        if requireClean and not np.isfinite(data.values).all():
            raise ValueError("Clean epochs data must contain only finite values")

    @staticmethod
    def getMetadataCoordName(columnName):
        """Convert a metadata column name into a trial-prefixed camelCap name."""
        nameParts = re.findall(r"[A-Za-z0-9]+", str(columnName))

        if not nameParts:
            raise ValueError(f"Metadata column has no usable name: {columnName!r}")

        return "trial" + "".join(namePart[0].upper() + namePart[1:] for namePart in nameParts)

    @staticmethod
    def formatPreview(items, maxItems=4, maxLength=45):
        """Return a bounded, single-line preview."""
        itemList = list(items)
        labels = []

        for item in itemList[:maxItems]:
            label = " ".join(str(item).split())
            if len(label) > maxLength:
                label = label[:maxLength - 3] + "..."
            labels.append(label)

        if len(itemList) > maxItems:
            labels.append(f"... (+{len(itemList) - maxItems} more)")

        return ", ".join(labels) if labels else "none"

    def printImportSummary(self):
        """Print a compact import overview, including trial grouping labels."""
        data = self.requireData()
        attrs = data.attrs
        qualityAction = "Dropped" if attrs["dropBadData"] else "Retained and labeled"

        sensorCounts = pd.Series(data.sensorType.values).value_counts()
        sensorLabels = [f"{sensorType}: {int(typeCount)}" for sensorType, typeCount in sensorCounts.items()]

        eventCounts = pd.Series(data.eventCode.values).value_counts()
        eventLabels = []

        for eventCode, eventCount in eventCounts.items():
            eventNames = [eventName for eventName, mappedCode in attrs["eventId"].items() if mappedCode == eventCode]
            eventLabel = " / ".join(eventNames) or str(eventCode)
            eventLabels.append(f"{eventLabel} ({int(eventCount)})")

        metadataCoords = list(attrs["metadataColumnMap"])
        metadataLabels = []

        for coordName in metadataCoords[:4]:
            coordValues = pd.Series(data.coords[coordName].values)
            uniqueValues = coordValues.dropna().unique()
            missingCount = int(coordValues.isna().sum())
            coordSummary = f"{coordName}: {len(uniqueValues)} unique [{self.formatPreview(uniqueValues, maxItems=3)}]"

            if missingCount:
                coordSummary += f", {missingCount} missing"

            metadataLabels.append(coordSummary)

        if len(metadataCoords) > 4:
            metadataLabels.append(f"+{len(metadataCoords) - 4} more fields")

        trialCount = data.sizes["trial"]
        sensorCount = data.sizes["sensor"]
        timeCount = data.sizes["time"]
        timeStart = float(data.time.values[0])
        timeEnd = float(data.time.values[-1])
        badTimeCount = int(data.timeBad.values.sum())
        metadataSummary = "; ".join(metadataLabels) if metadataLabels else "none"

        summaryLines = [
            f"Loaded {data.name}: {trialCount} trials × {sensorCount} sensors × {timeCount} time points",
            f"{qualityAction} on import: {attrs['badSensorCount']} bad data sensors; {attrs['badTrialCount']} bad trials (MNE: {attrs['mneBadTrialCount']}; non-finite: {attrs['nonFiniteTrialCount']}; may overlap)",
            f"Sensors: {self.formatPreview(sensorLabels)}; {attrs['markedBadSensorCount']} marked bad in source; {attrs['excludedAuxiliarySensorCount']} auxiliary channels excluded",
            f"Time: {timeStart:.4f} to {timeEnd:.4f} s; {attrs['samplingFrequency']:g} Hz; {badTimeCount} time points labeled bad",
            f"Trial IDs: original epoch indices; event groups: {self.formatPreview(eventLabels)}",
            f"Metadata: {metadataSummary}",
        ]

        pglMessages.print("\n".join(summaryLines))
       
#######################
# # pglTimeSeries
#######################
class pglTimeSeries(pglDataMatrix):
    
    def _saveMetadata(self, h5file):
        ''' 
        save version
        '''
        h5file.attrs["timeSeriesVersion"] = 1.0

    def print(self):
        """Print a summary of the time series."""

        print("pglTimeSeries")
        print("-" * 40)
        print(f"nSamples    : {self.shape[0]}")
        print(f"nChannels   : {self.shape[1]}")
        if self.sampleRate is not None:
            print(f"Sample Rate : {self.sampleRate:g} Hz")

            if self.sampleRate > 0:
                print(f"Duration    : {self.shape[0] / self.sampleRate:.2f}s")

        print("\nChannels:")
        for i, channelName in enumerate(self.channelNames):
            unit = self.units[i] if i < len(self.units) else ""
            print(f"  {i:2d}: {channelName:<12} ({unit})")

    def timeSlice(self, startTime, endTime):

        if self.sampleRate is None:
            raise ValueError("sampleRate is not defined.")

        startIndex = int(startTime * self.sampleRate)
        endIndex = int(endTime * self.sampleRate)

        return self._dataset()[startIndex:endIndex, :]
    
#######################
# pglEventsData
#######################
class pglEventsData(pglDataMatrix):
    '''
    Wrapper for pglDataMatrix which provides member functions
    for data that has events (but is internally stored as a matrix
    '''
    # ---------------------------------------------------------
    # Construction
    # --------------------------------------------------------
    def __init__(self, eventClass):
        '''
        Init registers the eventClass that will be used for adding / retrieving data
        and providing required fields
        Args:
            eventClass: Either an instance or type of pglEvent
        '''
        self._registerEventClass(eventClass)
        
        # create memory backed storage for adding events to
        self._data = np.empty((0, len(self.channelNames)), dtype=float)
        
        # not used for memory backed storage
        self._h5 = None
        self.filePath = None

    def _registerEventClass(self, eventClass):
        '''
        Helper function that retrieves requiredFields and units from eventClass
        '''
        # accept either an instance or the class itself
        self.eventClass = eventClass if isinstance(eventClass, type) else type(eventClass)
        self.eventClassName = self.eventClass.__name__
        if not is_dataclass(self.eventClass):
            raise TypeError("(pglEventsData:_registerEventClass) eventClass needs to be a dataclass")
        
        # check the annotation of the eventClass for required fields
        self._eventFields = fields(self.eventClass)
        self.requiredFields = [f.name for f in self._eventFields]
        self.channelNames = self.requiredFields
        
        # get units from dataclass field metadata
        self.units = [f.metadata.get("units", "unknown") for f in self._eventFields]
        
        # no sample rate for events
        self.sampleRate = None
        
    def addEvent(self, event):
        # Check event type
        if not isinstance(event, self.eventClass):
            raise TypeError(f"Expected event of type {self.eventClass.__name__}, got {type(event).__name__}")

        # convert into a row of data
        row = np.array([getattr(event, field) for field in self.requiredFields], dtype=float)

        # Append to data matrix
        self.addRow(row)
    
    @classmethod
    def fromArray(cls, eventClass, data):

        obj = cls(eventClass)

        obj._data = np.asarray(data, dtype=float)

        if obj._data.ndim != 2:
            raise ValueError("data must be 2D")

        if obj._data.shape[1] != len(obj.channelNames):
            raise ValueError(
                "data columns do not match event fields"
            )

        return obj
    
    @classmethod
    def fromFile(cls, filePath, mode="r"):

        # First let pglDataMatrix do the normal HDF5 loading
        obj = super().fromFile(filePath, mode)

        try:
            # Retrieve saved event class name
            if "eventClassName" not in obj._h5.attrs:
                raise ValueError(
                    "(pglEventsData:fromFile) missing eventClassName attribute"
                )

            eventClassName = obj._h5.attrs["eventClassName"]

            # Convert bytes if needed
            if isinstance(eventClassName, bytes):
                eventClassName = eventClassName.decode()

            # Recover actual Python class
            eventClass = pglEvent.getClass(eventClassName)

            # Register it
            obj._registerEventClass(eventClass)
            
            # check that requiredFeidsl mach channelNames
            if obj.requiredFields != obj.channelNames:
                raise ValueError(
                    "(pglEventsData:fromFile) event fields do not match saved channel names"
                )

        except Exception:
            obj.close()
            raise

        return obj   
    
    def getEvents(self, fieldName, value=None, minVal=None, maxVal=None):
        '''
        Get events in which the fieldName matches the value or range of values
        
        Args:
            fieldName (string): The name of the field to search by
            value (float): If set, search for an exact match
            minVal (float): If set, returns events that are >= minVal
            maxVal (float): If set, returns event that are <= maxVal
        e.g.:
            getEvents("fieldName", 1)                 # exact match
            getEvents("fieldName", minVal=100, maxVal=200)  # range between and including 100-200
        '''
        # get the column (field) to check
        col = self[fieldName]
        if col is None:
            return(np.array([]))
        
        if value is not None:
            mask = col == value
        elif minVal is not None and maxVal is not None:
            mask = (col >= minVal) & (col <= maxVal)
        elif minVal is not None:
            mask = col >= minVal
        elif maxVal is not None:
            mask = col <= maxVal
        
        # get matching rows
        matchingRows =self._data[mask,:]
        
        # Create event instances from matching rows
        events = [
            self.eventClass(**dict(zip(self.channelNames, row)))
            for row in matchingRows
        ]
        return events
        
    def _saveMetadata(self, h5file):
        ''' 
        save eventClassname and version
        '''
        h5file.attrs["eventClassName"] = self.eventClassName
        h5file.attrs["eventsDataVersion"] = 1.0
    
    def print(self):
        """Print a summary of the time series."""

        print("pglEventsData")
        print("-" * 40)
        print(f"nEvents    : {self.shape[0]}")
        print(f"nFields    : {self.shape[1]}")

        print("\nFields:")
        for i, channelName in enumerate(self.channelNames):
            unit = self.units[i] if i < len(self.units) else ""
            print(f"  {i:2d}: {channelName:<12} ({unit})")

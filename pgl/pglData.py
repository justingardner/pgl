################################################################
#   filename: pglData.py
#    purpose: Classes that handle serializing and deserializing data
#         by: JLG
#       date: Feb 18, 2026
################################################################

##############
# Imports
##############
import numpy as np
import h5py
from dataclasses import fields, is_dataclass
from pathlib import Path
from .pglEvent import pglEvent

############################
# pglDataMatrix
############################
class pglDataMatrix:
    """
    Base class that wraps either:
        - an in-memory NumPy array
        - an HDF5 dataset
    """
    # ---------------------------------------------------------
    # Construction
    # --------------------------------------------------------
    @classmethod
    def fromArray(cls, data, channelNames, units, sampleRate=None):
        '''
        initialize the data matrix from array data - stored internally
        as a numpy array until save is called when it gets backed by an hdf5
        
        Args:
            data: matrix of data. Channels are in columns
            channelNames (list): string labels for each channel
            sampleRate (float): sampleRate of data
            units (list): string labels of units
        '''
        
        # initialize class 
        obj = cls.__new__(cls)

        # store the data
        obj._data = np.asarray(data)

        # validate channelNames length
        if len(channelNames) != obj._data.shape[1]:
            raise ValueError(
                f"(pglDataMatrix:fromArray) channelNames has {len(channelNames)} entries "
                f"but data has {obj._data.shape[1]} columns."
            )
        # validate channelNames has no duplicates
        if len(set(channelNames)) != len(channelNames):
            raise ValueError(
                "(pglDataMatrix:fromArray) channelNames must be unique."
            )
        # save channelNames
        obj.channelNames = list(channelNames)
        
        # validate units     and sve    
        if len(units) != obj._data.shape[1]:
            raise ValueError(
                f"(pglDataMatrix:fromArray) units has {len(units)} entries "
                f"but data has {obj._data.shape[1]} columns."
            )
        obj.units = list(units)
        
        # save sampleRate
        obj.sampleRate = sampleRate

        # not used for memory backed storage
        obj._h5 = None
        obj.filePath = None

        return obj

    @classmethod
    def fromFile(cls, filePath, mode="r"):

        # initialize class
        obj = cls.__new__(cls)

        # -----------------------------
        # filePath validation
        # -----------------------------
        if filePath is None:
            raise ValueError("(pglDataMatrix:fromFile) filePath is None")

        if not isinstance(filePath, (str, Path)):
            raise TypeError("(pglDataMatrix:fromFile) filePath must be a str or Path")

        path = Path(filePath)

        if not path.exists():
            raise FileNotFoundError(f"(pglDataMatrix:fromFile) file not found: {filePath}")

        # -----------------------------
        # open file
        # -----------------------------
        obj.filePath = str(path)
        obj._h5 = h5py.File(path, mode)
        obj._data = None

        try:
            # -----------------------------
            # required dataset checks
            # -----------------------------
            required = ["data", "channelNames", "units"]

            for key in required:
                if key not in obj._h5:
                    raise ValueError(f"(pglDataMatrix:fromFile) missing dataset '{key}'")

            data = obj._h5["data"]

            if data.ndim != 2:
                raise ValueError("(pglDataMatrix:fromFile) 'data' must be 2D (samples x channels)")

            numChannels = data.shape[1]

            # -----------------------------
            # channelNames
            # -----------------------------
            obj.channelNames = [
                c.decode() if isinstance(c, bytes) else c
                for c in obj._h5["channelNames"][()]
            ]

            if len(obj.channelNames) != numChannels:
                raise ValueError(
                    f"(pglDataMatrix:fromFile) channelNames has {len(obj.channelNames)} entries "
                    f"but data has {numChannels} columns."
                )

            if len(set(obj.channelNames)) != len(obj.channelNames):
                raise ValueError(
                    "(pglDataMatrix:fromFile) channelNames must be unique."
                )

            # -----------------------------
            # units
            # -----------------------------
            obj.units = [
                u.decode() if isinstance(u, bytes) else u
                for u in obj._h5["units"][()]
            ]

            if len(obj.units) != numChannels:
                raise ValueError(
                    f"(pglDataMatrix:fromFile) units has {len(obj.units)} entries "
                    f"but data has {numChannels} columns."
                )

            # -----------------------------
            # metadata
            # -----------------------------
            obj.sampleRate = obj._h5.attrs.get("sampleRate", None)
        
        except Exception:
            
            obj._h5.close()
            raise

        return obj
    
    # ---------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------
    def addRow(self, row):
        '''
        Append one or more rows of data to the matrix.

        Args:
            row: 1D array-like (single row) or 2D array-like (multiple rows).
                 Number of columns must match len(self.channelNames).
        '''
        row = np.asarray(row)

        # normalize a single row into shape (1, numChannels)
        if row.ndim == 1:
            row = row[np.newaxis, :]

        if row.ndim != 2:
            raise ValueError(
                f"(pglDataMatrix:addRow) row must be 1D or 2D, got {row.ndim}D"
            )

        numChannels = len(self.channelNames)
        if row.shape[1] != numChannels:
            raise ValueError(
                f"(pglDataMatrix:addRow) row has {row.shape[1]} columns, "
                f"expected {numChannels} to match channelNames"
            )

        if self._h5 is None:
            # in-memory: just stack onto the numpy array
            self._data = np.vstack([self._data, row])
        else:
            # make sure we can append to the file
            if self._h5.mode == "r":
                raise ValueError(
                    "(pglDataMatrix:addRow) file was opened read-only "
                    "(mode='r'); reopen with fromFile(path, mode='r+') to add rows"
                )
                
            # hdf5-backed: resize dataset and write new rows in place
            dataset = self._h5["data"]
            oldRows = dataset.shape[0]
            newRows = oldRows + row.shape[0]

            try:
                dataset.resize(newRows, axis=0)
            except TypeError as e:
                raise TypeError(
                    "(pglDataMatrix:addRow) underlying hdf5 dataset is not "
                    "resizable (was it created with maxshape=(None, numChannels))?"
                ) from e

            dataset[oldRows:newRows, :] = row
            
    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _dataset(self):
        """
        Returns either:
            NumPy array
        or
            h5py Dataset

        Neither caller nor derived classes need to know which.
        """
        if self._h5 is None:
            return self._data
        
        # returns the _h5 structure which implements lazy-loading
        return self._h5["data"]

    def _channelIndex(self, channelName):
        '''
        Returns the index of a named chanel
        '''
        try:
            return self.channelNames.index(channelName)

        except ValueError:
            raise KeyError(f"Unknown channel '{channelName}'")

    # ---------------------------------------------------------
    # Data access
    # ---------------------------------------------------------
    def __getitem__(self, key):
        '''
        Allows access for keys like 
        dataMatrix['key']
        '''
        dataset = self._dataset()

        if isinstance(key, str):
            index = self._channelIndex(key)
            return dataset[:, index]

        return dataset[key]

    #def __getattr__(self, name):
    #    '''
    #    ALlows access like
    #    dataMatrix.key
    #    '''
    #    # check first to make sure that channelNames is 
    #    # a field - otherwise this code we end up in 
    #    # ifinite recursion trying to find non-existient fields
    #    if "channelNames" in self.__dict__:
    #
    #        # if the name exist in channels, then return it
    #        if name in self.channelNames:
    #            return self[name]
    #    
    #    # throw error if not found
    #    raise AttributeError(name)

    def get(self, channelName):
        '''
        explicit call
        '''
        return self[channelName]

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------
    def save(self, filePath=None, overwrite=True):
        '''
        Save the data matrix to hdf5.

        If this object is backed by an in-memory NumPy array (created via
        fromArray), this creates a new hdf5 file at filePath.

        If this object is already file-backed (created via fromFile), any
        rows added via addRow are already live in the open hdf5 file, so
        this just flushes pending writes to disk. filePath is ignored in
        this case since the object is already tied to its own file.

        Args:
            filePath (str): Path to save to. Only used for in-memory-backed
              objects. Ignored for file-backed objects.
            overwrite (bool): whether to overwrite an existing file when
              creating a new hdf5 file. Only used for in-memory-backed objects.
        '''
        # already file-backed: nothing to create, just flush pending writes
        if self._h5 is not None:
            self._h5.flush()
            return

        # in-memory-backed: need a filePath to create a new file
        if filePath is None:
            raise ValueError("(pglDataMatrix:save) No file path specified.")

        if Path(filePath).exists() and not overwrite:
            raise FileExistsError(
                f"(pglDataMatrix:save) '{filePath}' already exists and overwrite=False"
            )

        # open file
        with h5py.File(filePath, "w") as f:

            # create the dataset, resizable so addRow can grow it later
            f.create_dataset(
                "data",
                data=self._dataset(),
                compression="gzip",
                chunks=True,
                maxshape=(None, len(self.channelNames)),
            )

            # create channel names
            f.create_dataset(
                "channelNames",
                data=np.array(self.channelNames, dtype="S")
            )

            # create units
            f.create_dataset(
                "units",
                data=np.array(self.units, dtype="S")
            )

            # create sampleRate
            if self.sampleRate is not None:
                f.attrs["sampleRate"] = self.sampleRate

            # class name and version number
            f.attrs["className"] = self.__class__.__name__
            f.attrs["dataMatrixVersion"] = 1.0
                
            # subclass hook to save metadata
            self._saveMetadata(f)

        # save filename
        self.filePath = filePath
        
    def _saveMetadata(self, h5file):
        """
        Save class-specific metadata.
        Subclasses can override.
        """
        pass
        
    def close(self):
        '''
        Close file if there is one open
        '''
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
    
    # ---------------------------------------------------------
    # for using as a with
    # ---------------------------------------------------------   
    def __enter__(self):
        return self

    def __exit__(self, excType, excValue, traceback):
        self.close()
    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------
    @property
    def shape(self):
        return self._dataset().shape

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
    
    def getEvents(self):
        '''
        '''
        pass

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

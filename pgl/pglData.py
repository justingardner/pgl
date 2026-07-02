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
    def fromArray(cls, data, channelNames, units, samplingRate=None):
        '''
        initialize the data matrix from array data - stored internally
        as a numpy array until save is called when it gets backed by an hdf5
        
        Args:
            data: matrix of data. Channels are in columns
            channelNames (list): string labels for each channel
            samplingRate (float): samplingRate of data
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
        
        # save samplingRate
        obj.samplingRate = samplingRate

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
        obj.samplingRate = obj._h5.attrs.get("samplingRate", None)

        return obj
    
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

    def __getattr__(self, name):
        '''
        ALlows access like
        dataMatrix.key
        '''
        # check first to make sure that channelNames is 
        # a field - otherwise this code we end up in 
        # ifinite recursion trying to find non-existient fields
        if "channelNames" in self.__dict__:

            # if the name exist in channels, then return it
            if name in self.channelNames:
                return self[name]
        
        # throw error if not found
        raise AttributeError(name)

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
        save the data matrix to hdf5
        
        Args:
            filePath (str): Path to save to. If ommitted overwrites existig file
            overwrite (bool): whether to overwrite files or not
        '''
        if filePath is None:
            filePath = self.filePath

        if filePath is None:
            raise ValueError("(pglDataMatrix:save) No file path specified.")

        # handle overwrite
        mode = "w" if overwrite else "a"

        # open file
        with h5py.File(filePath, mode) as f:
            
            # create the dataset 
            f.create_dataset(
                "data",
                data=self._dataset(),
                compression="gzip",
                chunks=True
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
            
            # create sampling rate
            if self.samplingRate is not None:
                f.attrs["samplingRate"] = self.samplingRate

        # save filename
        self.filePath = filePath

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


class pglTimeseriesData(pglDataMatrix):

    def timeSlice(self, startTime, endTime):

        if self.samplingRate is None:
            raise ValueError("samplingRate is not defined.")

        startIndex = int(startTime * self.samplingRate)
        endIndex = int(endTime * self.samplingRate)

        return self._dataset()[startIndex:endIndex, :]
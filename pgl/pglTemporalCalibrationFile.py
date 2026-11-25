
################################################################
# pglTemporalCalibrationFile.py
# Raw analog/frame storage for temporal display calibration.
################################################################

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from .pglDevice import pglAnalogTraceData
from .pglSerialize import pglSerialize


class pglTemporalCalibrationFile:
    """
    Local, versioned temporal-calibration payload storage.

    Does not analyze latency, infer missing samples, modify timestamps,
    or open hardware. Nonfinite raw samples/timestamps remain unchanged.

    New artifacts are published without overwriting an existing destination.
    Staging artifacts remain retained on success or failure.
    """

    formatName = "pglTemporalCalibration"
    formatVersion = 1
    fileName = "calibrationData.h5"

    @staticmethod
    def _fileDigest(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            while True:
                block = file.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _writeMetadata(group, value):
        text = pglSerialize.toJSON(value, strict=True)
        group.create_dataset(
            "metadata", data=text, dtype=h5py.string_dtype("utf-8")
        )

    @staticmethod
    def _readMetadata(group):
        value = pglSerialize.fromJSON(
            group["metadata"].asstr()[()], strict=True
        )
        if not isinstance(value, dict):
            raise ValueError("Temporal metadata must restore to a dictionary.")
        return value

    @staticmethod
    def _validateTrace(trace):
        if not isinstance(trace, pglAnalogTraceData):
            raise TypeError("trace must be pglAnalogTraceData.")

        samples = trace.data
        relativeTime = trace.time
        if (
            not isinstance(samples, np.ndarray)
            or samples.ndim not in (1, 2)
            or samples.dtype.kind not in "iuf"
        ):
            raise ValueError("Analog data must be a numeric 1D or samples×channels array.")
        if (
            not isinstance(relativeTime, np.ndarray)
            or relativeTime.ndim != 1
            or relativeTime.dtype.kind not in "iuf"
            or len(relativeTime) != len(samples)
        ):
            raise ValueError("Relative sample times must match the sample count.")

        # Do not reject raw NaNs, gap markers, or bad timestamp observations.
        # They are evidence; the analysis layer decides timing eligibility.
        channelCount = 1 if samples.ndim == 1 else samples.shape[1]
        if channelCount < 1:
            raise ValueError("At least one analog channel is required.")
        if trace.numChannels != channelCount:
            raise ValueError("Trace channel count disagrees with the array shape.")
        if (
            not isinstance(trace.channelNames, list)
            or len(trace.channelNames) != channelCount
            or any(not isinstance(name, str) for name in trace.channelNames)
        ):
            raise ValueError("Trace channel names must match the channel count.")

    @staticmethod
    def _validateFrames(frameArrays):
        """
        Preserve numeric frame columns in original acquisition order.

        Caller supplies frame identity columns such as repeatIndex/frameIndex,
        plus drawablePresented, success, and other returned fields.
        Missing scalar values should have an explicit presence column where
        distinguishing None from NaN is necessary.
        """
        if not isinstance(frameArrays, dict):
            raise TypeError("frameArrays must be a dictionary.")

        lengths = set()
        for name, array in frameArrays.items():
            if (
                not isinstance(name, str) or not name
                or not name.isidentifier() or name == "metadata"
            ):
                raise ValueError(f"Invalid frame dataset name: {name!r}")
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 1
                or array.dtype.kind not in "biuf"
            ):
                raise ValueError(f"Frame field {name!r} must be a numeric/bool vector.")
            lengths.add(len(array))

        if len(lengths) > 1:
            raise ValueError("Frame arrays have unequal lengths.")
        return next(iter(lengths), 0)

    @classmethod
    def saveRaw(cls, directory, trace, frameArrays, metadata):
        """
        Publish raw acquisition data once; return a small JSON-ready reference.

        Caller must stop acquisition before saving. metadata should retain
        presentation provenance, configuration and partial-acquisition status.

        New analysis results can later update calibration.json without
        rewriting this immutable raw artifact.
        """
        cls._validateTrace(trace)
        frameCount = cls._validateFrames(frameArrays)
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        # Preserve all public trace metadata, including timing/clock evidence,
        # without copying the large arrays into JSON.
        traceMetadata = {
            name: value for name, value in vars(trace).items()
            if not name.startswith("_") and name not in ("time", "data")
        }

        # Validate metadata before creating any HDF5 staging file.
        pglSerialize.toJSON(traceMetadata, strict=True)
        pglSerialize.toJSON(metadata, strict=True)

        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / cls.fileName
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Raw calibration artifact already exists; unchanged: {destination}"
            )

        payloadId = str(uuid4())
        staging = directory / (
            f".{cls.fileName}.{payloadId}.staging.h5"
        )

        try:
            with h5py.File(staging, "x") as file:
                file.attrs["formatName"] = cls.formatName
                file.attrs["formatVersion"] = cls.formatVersion
                file.attrs["payloadId"] = payloadId
                cls._writeMetadata(file, metadata)

                analog = file.create_group("analog")
                cls._writeMetadata(analog, traceMetadata)

                options = (
                    {"compression": "gzip", "compression_opts": 1, "shuffle": True}
                    if trace.data.size else {}
                )
                analog.create_dataset("data", data=trace.data, **options)
                analog.create_dataset("relativeTime", data=trace.time)

                frames = file.create_group("frames")
                frames.attrs["count"] = frameCount
                for name, array in frameArrays.items():
                    frames.create_dataset(name, data=array)

                file.flush()

            # Same-directory hard-link publication is atomic and refuses an
            # existing destination. Unlike os.replace(), it cannot overwrite it.
            # The staging link is deliberately retained, not deleted.
            checksum = cls._fileDigest(staging)
            os.link(staging, destination)

        except BaseException as error:
            error.calibrationDataStagingPath = str(staging)
            error.calibrationDataDestination = str(destination)
            raise

        return {
            "filename": cls.fileName,
            "formatName": cls.formatName,
            "formatVersion": cls.formatVersion,
            "payloadId": payloadId,
            "sha256": checksum,
            "sampleCount": len(trace.time),
            "channelCount": trace.numChannels,
            "frameCount": frameCount,
            "publicationStatus": "complete",
            "durabilityAfterPowerLoss": "notEstablished",
        }

    @classmethod
    def loadRaw(cls, directory, reference=None):
        """
        Restore raw arrays and metadata without hardware or timing analysis.

        A supplied JSON reference must match the immutable HDF5 artifact.
        Does not silently substitute current settings or clock calibrations.
        """
        directory = Path(directory).expanduser().resolve()
        filename = directory / cls.fileName

        if reference is not None:
            if not isinstance(reference, dict):
                raise TypeError("reference must be a dictionary or None.")
            for key, expected in (
                ("filename", cls.fileName),
                ("formatName", cls.formatName),
                ("formatVersion", cls.formatVersion),
            ):
                if reference.get(key) != expected:
                    raise ValueError(f"Calibration data reference mismatch: {key}")
            if cls._fileDigest(filename) != reference.get("sha256"):
                raise ValueError("Calibration HDF5 checksum does not match its reference.")

        with h5py.File(filename, "r") as file:
            if file.attrs.get("formatName") != cls.formatName:
                raise ValueError("Not a temporal-calibration HDF5 file.")
            if file.attrs.get("formatVersion") != cls.formatVersion:
                raise ValueError("Unsupported temporal-calibration HDF5 version.")

            payloadId = file.attrs["payloadId"]
            if reference is not None and payloadId != reference.get("payloadId"):
                raise ValueError("Calibration payload identity mismatch.")

            metadata = cls._readMetadata(file)
            analog = file["analog"]
            fields = cls._readMetadata(analog)
            if "time" in fields or "data" in fields:
                raise ValueError("Trace metadata duplicates reserved array fields.")

            fields["time"] = analog["relativeTime"][...]
            fields["data"] = analog["data"][...]

            # The supplied serializer's normal-class restoration does not call
            # __init__; preserve the saved attributes rather than recompute them.
            trace = pglAnalogTraceData.fromJSONdict(fields)
            cls._validateTrace(trace)

            frames = file["frames"]
            frameArrays = {
                name: frames[name][...] for name in frames.keys()
            }
            frameCount = cls._validateFrames(frameArrays)
            if frameCount != frames.attrs["count"]:
                raise ValueError("Frame dataset count mismatch.")

        if reference is not None:
            for key, actual in (
                ("sampleCount", len(trace.time)),
                ("channelCount", trace.numChannels),
                ("frameCount", frameCount),
            ):
                if reference.get(key) != actual:
                    raise ValueError(f"Calibration reference count mismatch: {key}")

        return {
            "trace": trace,
            "frameArrays": frameArrays,
            "metadata": metadata,
            "payloadId": payloadId,
        }

################################################################
#   filename: pglFlywheel
#    purpose: Code for browsing and downloading data from flywheel
#         by: JLG
#       date: Aug 20, 2026
################################################################

#############
# Import modules
#############
import os
from uuid import uuid4

from traitlets import Bool, Instance, List, Unicode

from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings
from .pglMRI import pglMRIFileBacking

################################################################
# Process-local clients
#
# Clients are not stored in settings or serialized. The dialog
# process creates its own client using an inherited environment
# variable whose name is stored in the settings.
################################################################
_flywheelClients = {}


def _getFlywheelClient(connectionEnv):
    """Get or create the Flywheel client for this process."""
    if connectionEnv in _flywheelClients:
        return _flywheelClients[connectionEnv]

    apikey = os.environ.get(connectionEnv)
    if not apikey:
        raise RuntimeError(
            "Flywheel credentials are unavailable in this process. "
            "Create a new pglFlywheel connection."
        )

    import flywheel

    # Do not include credentials or SDK exception details in this error.
    try:
        client = flywheel.Client(apikey)
    except Exception:
        raise RuntimeError("Could not create the Flywheel client.") from None

    _flywheelClients[connectionEnv] = client
    return client


################################################################
# Scan settings
################################################################
class pglFlywheelScan(pglTraitSettings):
    name = Unicode(help="Name / type of scan", visible=False)

    # Download identifiers survive serialization but are not displayed.
    acquisitionID = Unicode("", help="Flywheel acquisition ID", visible=False)
    fileName = Unicode("", help="Exact Flywheel filename", visible=False)

################################################################
# Session settings
################################################################
class pglFlywheelSession(pglTraitSettings):
    name = Unicode(help="Name of session", visible=False)
    subjectID = Unicode(help="subjectID", enabled=False)
    id = Unicode(help="Session ID", visible=False)

    scans = List(
        Instance(pglFlywheelScan),
        help="Scans in session",
        settingsListKey="name",
        multiSelect=True,
        maxRowsVisible=12,
        property="loadedScans",
    )

    # These traits survive serialization but are not displayed.
    _connectionEnv = Unicode("")
    _scansLoaded = Bool(False)

    @property
    def loadedScans(self):
        """Load NIfTI files once, ordered by acquisition time."""
        if self._scansLoaded:
            return self.scans

        pglMessages.message(f"Loading scans: {self.name}")

        fw = _getFlywheelClient(self._connectionEnv)
        flywheelSession = fw.get_session(self.id)

        # Sort acquisitions chronologically; missing timestamps go last.
        def acquisitionSortKey(acquisition):
            timestamp = getattr(acquisition, "timestamp", None)
            return timestamp.timestamp() if timestamp is not None else float("inf")

        acquisitions = sorted(
            flywheelSession.acquisitions(),
            key=acquisitionSortKey,
        )

        # Build locally so a failed request does not cache partial results.
        # Files belonging to the same acquisition remain grouped together.
        scans = []
        for flywheelScan in acquisitions:
            for flywheelFile in flywheelScan.files or []:
                if flywheelFile.type == "nifti":
                    scans.append(pglFlywheelScan(
                        name=f"{flywheelScan.label}: {flywheelFile.name}",
                        acquisitionID=flywheelScan.id,
                        fileName=flywheelFile.name,
                    ))

        self.scans = scans
        self._scansLoaded = True
        return self.scans


################################################################
# Project settings
################################################################
class pglFlywheelProject(pglTraitSettings):
    name = Unicode(help="Name of project", visible=False)
    id = Unicode(help="Identifier of project", visible=False)

    sessions = List(
        Instance(pglFlywheelSession),
        help="Sessions in project",
        settingsListKey="name",
        property="loadedSessions",
    )

    # Separate loaded state allows an empty result to be cached too.
    _connectionEnv = Unicode("")
    _sessionsLoaded = Bool(False)

    @property
    def loadedSessions(self):
        """Load this project's sessions once, without loading their scans."""
        if self._sessionsLoaded:
            return self.sessions

        pglMessages.message(f"Loading sessions: {self.name}")

        fw = _getFlywheelClient(self._connectionEnv)
        flywheelProject = fw.get_project(self.id)

        # Use the project ID rather than searching by a potentially
        # nonunique project label.
        sessions = []
        for flywheelSession in flywheelProject.sessions():
            timestamp = flywheelSession.timestamp
            sessionName = (
                timestamp.strftime("%Y-%m-%d %H:%M")
                if timestamp is not None
                else flywheelSession.label or flywheelSession.id
            )

            subject = flywheelSession.subject
            subjectID = getattr(subject, "label", "") or ""

            sessions.append(pglFlywheelSession(
                name=sessionName,
                subjectID=subjectID,
                id=flywheelSession.id,
                _connectionEnv=self._connectionEnv,
            ))

        self.sessions = sessions
        self._sessionsLoaded = True
        return self.sessions


################################################################
# Root selection settings
################################################################
class pglFlywheelChoose(pglTraitSettings):
    projects = List(
        Instance(pglFlywheelProject),
        settingsListKey="name",
        help="Flywheel projects to load data from",
    )


################################################################
# Deferred, version-pinned MRI source
################################################################
class pglFlywheelMRIBacking(pglMRIFileBacking):
    """Stream one pinned Flywheel file version without local staging.

    Inherits MRI materialization support. Complete reads verify the recorded
    size and, when available, Flywheel's source SHA-384 checksum.

    Missing source checksums are allowed and remain None in source metadata.
    Version pinning and size checks do not substitute for checksum verification.

    Construction performs no network access.
    The SDK client is runtime-only and must never enter serialized metadata.

    Header-only reads may close before full-file checks.
    Complete materialization/export must read through EOF.
    """

    def __init__(self, client, acquisitionID, fileName, *, version, sizeBytes, sourceHash):
        """Record a pinned source with an optional SHA-384 checksum."""
        import re

        # Require a runtime SDK client capable of streaming acquisition files.
        downloadMethod = getattr(getattr(client, "acquisitions_api", None), "download_file_from_acquisition_with_http_info", None)
        if not callable(downloadMethod):
            raise TypeError("A connected Flywheel SDK client is required.")

        # Validate the acquisition and filename before constructing a source.
        if not isinstance(acquisitionID, str) or re.fullmatch(r"[0-9a-fA-F]{24}", acquisitionID) is None:
            raise ValueError("Invalid Flywheel acquisition ID.")

        if (
            not isinstance(fileName, str)
            or not fileName.strip()
            or fileName in (".", "..")
            or any(character in fileName for character in ("/", "\\", "\x00", "\n", "\r"))
            or not fileName.lower().endswith((".nii", ".nii.gz"))
        ):
            raise ValueError("Flywheel MRI filename must be a single .nii or .nii.gz filename.")

        # Version and size remain mandatory even when the checksum is absent.
        if type(version) is not int or version < 1:
            raise ValueError("Flywheel file version must be a positive integer.")

        if type(sizeBytes) is not int or not 0 <= sizeBytes <= 9223372036854775807:
            raise ValueError("Invalid Flywheel file size.")

        # Normalize supported checksum representations for the stream verifier.
        # None means no checksum; other unsupported values are still rejected.
        normalizedHash = None
        if sourceHash is not None:
            if not isinstance(sourceHash, str):
                raise ValueError("Flywheel source hash must be a SHA-384 string or None.")

            if re.fullmatch(r"[0-9a-fA-F]{96}", sourceHash):
                normalizedHash = f"v0-sha384-{sourceHash.lower()}"
            elif re.fullmatch(r"v0-sha384-[0-9a-fA-F]{96}", sourceHash):
                normalizedHash = sourceHash.lower()
            else:
                raise ValueError("Expected a bare SHA-384 digest, a v0-sha384 source hash, or None.")

        # Keep the authenticated client runtime-only.
        # Original source metadata remains unchanged; normalization is internal.
        self._client = client
        self._acquisitionID = acquisitionID
        self._fileName = fileName
        self._version = version
        self._sourceSize = sizeBytes
        self._sourceHash = normalizedHash

        # Inherited materialization uses path to identify the format.
        # A Flywheel SHA-384 hash is not a PGL SHA-256 identity.
        self.path = fileName
        self._sha256 = None
        self._sizeBytes = None

    def open(self):
        """Return a context-managed stream for the recorded file version.

        Complete reads verify the recorded size and, when supplied,
        Flywheel's SHA-384 checksum before returning EOF.

        Without a source checksum, only version pinning and size checking
        are available; no source-checksum verification is claimed.

        Early close releases the response without draining the file.
        Header-only reads therefore do not perform full-file verification.

        SDK errors are reported without exposing authenticated URLs.
        The client remains caller-owned.
        """
        import hashlib
        import io
        from contextlib import contextmanager

        backing = self

        class VerifiedStream(io.RawIOBase):
            """Adapt SDK chunks to bounded-memory reads with EOF checks."""

            def __init__(self, response):
                super().__init__()

                # Hash only when a remote checksum is available for comparison.
                self._chunks = iter(response.iter_content(chunk_size=65536))
                self._pending = memoryview(b"")
                self._digest = hashlib.sha384() if backing._sourceHash is not None else None
                self._received = 0
                self._eofChecked = False

            def readable(self): return True

            def readinto(self, buffer):
                if self.closed: raise ValueError("Flywheel source stream is closed.")
                if not len(buffer): return 0
                if self._eofChecked: return 0

                # Request another chunk only after consuming the pending bytes.
                while not len(self._pending):
                    try:
                        chunk = next(self._chunks)
                    except StopIteration:
                        # Every complete read must match the recorded size.
                        if self._received != backing._sourceSize:
                            raise ValueError("Flywheel source size mismatch.")

                        # Compare a checksum only when Flywheel supplied one.
                        if self._digest is not None:
                            actualHash = f"v0-sha384-{self._digest.hexdigest()}"
                            if actualHash != backing._sourceHash:
                                raise ValueError("Flywheel source SHA-384 mismatch.")

                        self._eofChecked = True
                        return 0

                    except Exception:
                        raise RuntimeError("Flywheel source transfer failed while reading the pinned file version.") from None

                    if not chunk: continue

                    # Reject oversized responses immediately.
                    self._received += len(chunk)
                    if self._received > backing._sourceSize:
                        raise ValueError("Flywheel source exceeds its recorded size.")

                    if self._digest is not None: self._digest.update(chunk)
                    self._pending = memoryview(chunk)

                # Copy only what fits; retain the remainder for the next read.
                count = min(len(buffer), len(self._pending))
                buffer[:count] = self._pending[:count]
                self._pending = self._pending[count:]
                return count

        @contextmanager
        def openStream():
            response = None
            stream = None

            try:
                # Always request the pinned version, never the current latest.
                # Suppress SDK exception details that could expose credentials.
                try:
                    response = backing._client.acquisitions_api.download_file_from_acquisition_with_http_info(backing._acquisitionID, backing._fileName, version=backing._version, _return_http_data_only=True, _preload_content=False)
                except Exception:
                    raise RuntimeError("Could not open the pinned Flywheel file version.") from None

                if response is None:
                    raise RuntimeError("Flywheel returned no source response.")

                # Buffer reads without materializing the entire file.
                stream = io.BufferedReader(VerifiedStream(response), buffer_size=65536)
                yield stream

            finally:
                # Release both resources, including after an early header-only read.
                try:
                    if stream is not None: stream.close()
                finally:
                    if response is not None: response.close()

        return openStream()

################################################################
# Flywheel connection
################################################################
class pglFlywheel:
    def __init__(self, *, settingsName=None, settings=None):
        """Connect using an import profile's keychain credential.

        Uses the default profile when neither argument is supplied.
        Explicit settings take precedence over settingsName.
        API-key arguments, fallback credential sources, and API-key prompts
        are not supported.
        """
        from .pglSettings import pglSettingsManager

        # Start disconnected so failed initialization cannot be used for browsing.
        self.selection = pglFlywheelChoose()
        self.fw = None

        # Load the SDK before resolving credentials.
        try:
            import flywheel
        except ImportError:
            pglMessages.warning("Unable to import flywheel library. Install using: pip install flywheel-sdk")
            return

        # Resolve the requested import profile, or use the default profile.
        importSettings = pglSettingsManager.getImportSettings(settingsName=settingsName, settings=settings)
        if importSettings is None: return

        # Retrieve credentials only from this profile's OS keychain entry.
        try:
            apikey = importSettings._readSavedFlywheelApiKey()
        except (RuntimeError, ValueError) as error:
            # These helpers return credential-free error messages.
            pglMessages.warning(str(error), level=0)
            return

        try:
            # Connect and verify the saved key.
            # Never include SDK exception details in warnings.
            try:
                fw = flywheel.Client(apikey)
                user = fw.get_current_user()
                if user is None:
                    raise RuntimeError("No authenticated Flywheel user returned.")
            except Exception:
                pglMessages.warning(
                    "Could not connect or log in to Flywheel. "
                    "Check your network connection or update the saved API key "
                    "through importSettings()."
                )
                return

            pglMessages.message(f"Logged in as: {user.firstname} {user.lastname} ({user.email})")

            # Load projects only. Sessions and acquisitions remain lazy-loaded.
            try:
                flywheelProjects = list(fw.projects())
            except Exception:
                pglMessages.warning("Could not load Flywheel projects.")
                return

            # Preserve the existing runtime credential handoff to the dialog.
            # Its subprocess inherits the key through this environment variable.
            # Serialized selection settings contain only the variable's name.
            connectionEnv = f"PGL_FLYWHEEL_API_KEY_{uuid4().hex}"
            os.environ[connectionEnv] = apikey
            _flywheelClients[connectionEnv] = fw

            # Publish the connected client and the initial project selection.
            self.fw = fw
            self.selection.projects = [
                pglFlywheelProject(name=project.label, id=project.id, _connectionEnv=connectionEnv)
                for project in flywheelProjects
            ]

        finally:
            # The SDK and subprocess handoff still retain runtime credentials.
            # Clearing this local reference does not securely erase memory.
            apikey = None

    def choose(self):
        """Browse Flywheel and return selected-file records.

        Returns
        -------
        list of dict
            Checked files across all browsed projects and sessions.
            An empty list means OK was pressed with no files checked.
        None
            The dialog was cancelled or did not return settings.
            The previous selection remains unchanged.
        """
        if self.fw is None:
            raise RuntimeError("Create a working Flywheel connection first.")

        if not self.selection.projects:
            pglMessages.message("No Flywheel projects are available.")
            return []

        # Import only when interactive selection is requested.
        # Use the existing subprocess dialog, including its serialization.
        from .pglDialog import pglDialogs

        selection = pglDialogs.traitsDialog(self.selection)
        if selection is None:
            return None

        if not isinstance(selection, pglFlywheelChoose):
            raise TypeError("The Flywheel dialog returned unexpected settings.")

        self.selection = selection
        return self.getSelectedScans()

    def getSelectedScans(self):
        """Return checked files without opening a dialog or loading more data.

        Read the stored lists, not the lazy-loading properties. Unvisited
        projects and sessions therefore require no network requests.

        Records contain identifiers and labels, never credentials.
        """
        selectedScans = []

        for project in self.selection.projects:
            for session in project.sessions:
                for scan in session.scans:
                    if not scan.isSelected:
                        continue

                    selectedScans.append({
                        "projectID": project.id,
                        "projectName": project.name,
                        "sessionID": session.id,
                        "sessionName": session.name,
                        "subjectID": session.subjectID,
                        "acquisitionID": scan.acquisitionID,
                        "fileName": scan.fileName,
                        "name": scan.name,
                    })

        return selectedScans
    
    def download(self, dataPath):
        """Download selected files and source metadata with one progress bar.

        Layout:
            dataPath/
                originalFilename.nii.gz
                flywheel.json

        Preserves original filenames and rejects case-insensitive filename
        collisions before writing output. The destination must not exist.

        Records SDK metadata returned for selected projects, sessions,
        acquisitions, and files. Does not separately fetch subjects, groups,
        analyses, jobs, audit history, or unselected attachments/sidecars.

        Returns file records containing localPath, acquisitionTime, sha256,
        and size_bytes. Does not create or publish a PGL database session.

        Progress reports streamed download bytes and subsequent SHA-256
        verification. File sections are weighted by expected byte size.

        Size is checked against Flywheel metadata. SHA-256 identifies local
        downloaded bytes; it is not compared with a remote source checksum.

        Failures may leave earlier completed files. The current temporary
        download is cleaned up, and flywheel.json is published only after
        every file succeeds. No resume or automatic retry is implemented.

        Concurrent source/destination mutation is unsupported.
        """
        import hashlib
        import json
        import os
        import tempfile
        import time
        from datetime import date, datetime
        from pathlib import Path

        from .pglProgress import pglProgress

        if self.fw is None:
            raise RuntimeError("Create a working Flywheel connection first.")

        selectedScans = self.getSelectedScans()
        if not selectedScans:
            raise ValueError("No scans are selected. Call choose() first.")

        pathText = os.fspath(dataPath)
        if (
            not isinstance(pathText, str)
            or not pathText.strip()
            or "://" in pathText
            or "::" in pathText
            or any(character in pathText for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("dataPath must be an ordinary nonempty local path.")

        destination = Path(pathText).expanduser().absolute()
        if os.path.lexists(destination):
            pglMessages.warning(
                f"Download destination already exists: {destination}. "
                "Choose a new directory. Nothing was downloaded or overwritten."
            )
            return None

        # Validate all filenames before network requests or output creation.
        # Flywheel IDs remain metadata, never directory names.
        planned = []
        filenames = set()

        for original in selectedScans:
            scan = dict(original)

            for field in ("projectID", "sessionID", "acquisitionID"):
                value = scan[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing Flywheel {field}.")

            filename = scan["fileName"]
            if (
                not isinstance(filename, str)
                or not filename.strip()
                or filename in (".", "..")
                or any(
                    character in filename
                    for character in ("/", "\\", "\x00", "\n", "\r")
                )
            ):
                raise ValueError("Flywheel filename must be a single safe filename.")

            outputKey = filename.casefold()
            if outputKey == "flywheel.json":
                raise ValueError("flywheel.json is reserved for download metadata.")
            if outputKey in filenames:
                raise ValueError(
                    f"Duplicate selected filename: {filename!r}. "
                    "Download these files into separate directories."
                )

            filenames.add(outputKey)
            planned.append(scan)

        def encodeMetadataValue(value):
            """Preserve timestamps without guessing a missing timezone."""
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            raise TypeError(
                f"Unsupported Flywheel metadata type: {type(value).__name__}"
            )

        def snapshotMetadata(value):
            """Validate and copy SDK metadata before creating output."""
            converter = getattr(value, "to_dict", None)
            if not callable(converter):
                raise TypeError("Flywheel metadata object lacks to_dict().")

            document = converter()
            if not isinstance(document, dict):
                raise TypeError("Flywheel metadata must contain a dictionary.")

            return json.loads(json.dumps(
                document,
                default=encodeMetadataValue,
                allow_nan=False,
            ))

        def fetchMetadata(method, identifier, kind):
            """Do not expose SDK exception details or authenticated URLs."""
            try:
                return method(identifier)
            except Exception:
                raise RuntimeError(
                    f"Could not read Flywheel {kind} metadata."
                ) from None

        def formatBytes(size):
            """Format byte counts using binary units."""
            value = float(size)
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if value < 1024 or unit == "TiB":
                    return f"{value:.1f} {unit}"
                value /= 1024

        sourceMetadata = {
            "projects": {},
            "sessions": {},
            "acquisitions": {},
        }
        acquisitions = {}
        workRecords = []
        downloaded = []
        manifestRecords = []

        with pglProgress("Downloading Flywheel files") as progress:
            # Metadata requests are synchronous. Report the object being read;
            # do not invent byte progress for these SDK calls.
            progress(0, "Reading source metadata")

            for scan in planned:
                projectID = scan["projectID"]
                sessionID = scan["sessionID"]
                acquisitionID = scan["acquisitionID"]

                if projectID not in sourceMetadata["projects"]:
                    progress(None, f"Reading project metadata: {scan['projectName']}")
                    project = fetchMetadata(
                        self.fw.get_project, projectID, "project"
                    )
                    sourceMetadata["projects"][projectID] = snapshotMetadata(project)

                if sessionID not in sourceMetadata["sessions"]:
                    progress(None, f"Reading session metadata: {scan['sessionName']}")
                    session = fetchMetadata(
                        self.fw.get_session, sessionID, "session"
                    )
                    sourceMetadata["sessions"][sessionID] = snapshotMetadata(session)

                if acquisitionID not in acquisitions:
                    progress(None, f"Reading acquisition metadata: {scan['fileName']}")
                    acquisition = fetchMetadata(
                        self.fw.get_acquisition, acquisitionID, "acquisition"
                    )
                    acquisitions[acquisitionID] = acquisition
                    sourceMetadata["acquisitions"][acquisitionID] = (
                        snapshotMetadata(acquisition)
                    )

                acquisition = acquisitions[acquisitionID]
                matchingFiles = [
                    file
                    for file in acquisition.files or []
                    if file.name == scan["fileName"]
                ]
                if len(matchingFiles) != 1:
                    raise ValueError(
                        f"Selected file is missing or ambiguous: {scan['fileName']}"
                    )

                selectedFile = matchingFiles[0]
                expectedSize = selectedFile.size
                if type(expectedSize) is not int or expectedSize < 0:
                    raise ValueError(
                        f"Flywheel reported an invalid size: {scan['fileName']}"
                    )

                timestamp = getattr(acquisition, "timestamp", None)
                if timestamp is not None and not isinstance(timestamp, datetime):
                    raise TypeError(
                        "Flywheel acquisition timestamp must be a datetime."
                    )

                scan["acquisitionTime"] = (
                    timestamp.isoformat() if timestamp is not None else None
                )
                workRecords.append({
                    "name": scan["fileName"],
                    "size_bytes": expectedSize,
                    "scan": scan,
                    "metadata": snapshotMetadata(selectedFile),
                })

            destination.mkdir(parents=True, exist_ok=False)

            # Reserve the final portion for metadata publication and cleanup.
            # Within each file, download receives 95% and hashing receives 5%.
            filesProgress = progress.section(0, 0.98)

            for work, fileProgress in pglProgress.fileSections(
                workRecords, filesProgress
            ):
                scan = work["scan"]
                filename = work["name"]
                expectedSize = work["size_bytes"]
                target = destination / filename
                downloadProgress = fileProgress.section(0, 0.95)
                verifyProgress = fileProgress.section(0.95, 1)

                with tempfile.TemporaryDirectory(
                    prefix=".downloading-",
                    dir=str(destination),
                ) as temporary:
                    stagedFile = Path(temporary) / filename
                    response = None
                    received = 0
                    started = time.monotonic()

                    downloadProgress(
                        0,
                        f"Connecting — expected {formatBytes(expectedSize)}",
                    )

                    try:
                        # Match the installed SDK's own streaming implementation.
                        # Authentication, redirects, and HTTP handling stay in
                        # the SDK; no tickets or credential URLs are logged.
                        try:
                            response = (
                                self.fw.acquisitions_api
                                .download_file_from_acquisition_with_http_info(
                                    scan["acquisitionID"],
                                    filename,
                                    _return_http_data_only=True,
                                    _preload_content=False,
                                )
                            )
                        except Exception:
                            raise RuntimeError(
                                f"Could not start Flywheel download: {filename!r}."
                            ) from None

                        if response is None:
                            raise RuntimeError(
                                f"Flywheel returned no download response: {filename!r}."
                            )

                        downloadProgress(
                            0,
                            f"Downloading — 0 B / {formatBytes(expectedSize)}",
                        )

                        with stagedFile.open("xb") as output:
                            chunks = iter(response.iter_content(chunk_size=65536))

                            while True:
                                # Catch network iteration failures separately
                                # from local disk and integrity errors.
                                try:
                                    chunk = next(chunks)
                                except StopIteration:
                                    break
                                except Exception:
                                    raise RuntimeError(
                                        f"Flywheel transfer failed: {filename!r}. "
                                        "Earlier completed files remain in dataPath."
                                    ) from None

                                if not chunk:
                                    continue

                                received += len(chunk)
                                if received > expectedSize:
                                    raise ValueError(
                                        f"Download exceeds Flywheel's reported size: "
                                        f"{filename}"
                                    )

                                if output.write(chunk) != len(chunk):
                                    raise OSError(
                                        f"Incomplete local download write: {filename}"
                                    )

                                elapsed = max(time.monotonic() - started, 0.001)
                                rate = received / elapsed
                                downloadProgress(
                                    received / max(expectedSize, 1),
                                    f"Downloading — {formatBytes(received)} / "
                                    f"{formatBytes(expectedSize)} | "
                                    f"{formatBytes(rate)}/s",
                                )

                            output.flush()
                            os.fsync(output.fileno())

                    finally:
                        if response is not None:
                            response.close()

                    if received != expectedSize:
                        raise ValueError(
                            f"Downloaded size differs from Flywheel metadata: "
                            f"{filename}"
                        )

                    downloadProgress(1, "Download complete")
                    verifyProgress(0, "Verifying SHA-256")

                    # Hash the completed local file in bounded-memory chunks.
                    digest = hashlib.sha256()
                    size = 0

                    with stagedFile.open("rb") as source:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break

                            digest.update(block)
                            size += len(block)
                            verifyProgress(
                                size / max(expectedSize, 1),
                                f"Verifying SHA-256 — {formatBytes(size)} / "
                                f"{formatBytes(expectedSize)}",
                            )

                    if size != expectedSize:
                        raise ValueError(
                            f"Local file size changed during verification: {filename}"
                        )

                    if os.path.lexists(target):
                        raise FileExistsError(target)

                    # Final filenames refer only to completed downloads.
                    stagedFile.rename(target)

                identity = {
                    "sha256": digest.hexdigest(),
                    "size_bytes": size,
                }
                downloaded.append({
                    **scan,
                    **identity,
                    "localPath": str(target),
                })
                manifestRecords.append({
                    **scan,
                    **identity,
                    "path": filename,
                    "metadata": work["metadata"],
                })

            progress(0.98, "Saving Flywheel metadata")

            # This portable document contains source metadata, not local
            # absolute paths or SDK client/authentication configuration.
            # Source metadata may contain identifying subject information.
            manifest = {
                "contract": "pgl.flywheel.download.v2",
                "files": manifestRecords,
                "sourceMetadata": sourceMetadata,
            }
            content = json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")

            with tempfile.TemporaryDirectory(
                prefix=".manifest-",
                dir=str(destination),
            ) as temporary:
                stagedManifest = Path(temporary) / "flywheel.json"

                with stagedManifest.open("xb") as output:
                    if output.write(content) != len(content):
                        raise OSError("Incomplete Flywheel metadata write.")
                    output.flush()
                    os.fsync(output.fileno())

                if stagedManifest.read_bytes() != content:
                    raise ValueError("Flywheel metadata verification failed.")

                manifestPath = destination / "flywheel.json"
                if os.path.lexists(manifestPath):
                    raise FileExistsError(manifestPath)

                stagedManifest.rename(manifestPath)

            # All responses, files, and temporary directories are now closed.
            progress.finish(f"Downloaded {len(downloaded)} files to {destination}")

        return downloaded
    
    def toMRI(self, *, cacheDirectory):
        """Build deferred MRI scans from the currently accepted selection.

        Reads project/session/acquisition/file metadata, not NIfTI payloads.
        Pins the file versions observed during this call.

        Requires exactly one selected Flywheel session.
        Does not open a dialog, create cache directories, or publish records.

        Returns pglMRI, or None with a warning for expected refusals.
        Unsupported metadata serialization and unexpected errors propagate.

        Source metadata contains no SDK client or authentication configuration.
        Backings retain the caller's runtime SDK client.

        Metadata requests are separate, not an atomic Flywheel snapshot.
        Groups, standalone subject records, analyses, jobs, and sidecars
        are not separately retrieved in this implementation.
        """
        import json
        import os
        from datetime import date, datetime
        from pathlib import Path

        from .pglMRI import pglMRI, pglMRIScan
        from .pglProgress import pglProgress

        if self.fw is None:
            pglMessages.warning("Create a working Flywheel connection first.", level=0)
            return None

        selected = self.getSelectedScans()
        if not selected:
            pglMessages.warning("No scans are selected. Call flywheel.choose() first.", level=0)
            return None

        sessionKeys = {(scan["projectID"], scan["sessionID"]) for scan in selected}
        if len(sessionKeys) != 1:
            pglMessages.warning(
                "MRI scans were not attached: select files from one Flywheel "
                "session at a time. Different sessions were not combined.",
                level=0,
            )
            return None

        try:
            cacheText = os.fspath(cacheDirectory)
            if (
                not isinstance(cacheText, str)
                or not cacheText.strip()
                or "://" in cacheText
                or "::" in cacheText
                or any(character in cacheText for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("cacheDirectory must be an ordinary local path.")

            cachePath = str(Path(cacheText).expanduser().absolute())

        except (TypeError, ValueError) as error:
            pglMessages.warning(f"MRI scans were not attached: {error}", level=0)
            return None

        def encodeValue(value):
            if isinstance(value, (datetime, date)): return value.isoformat()
            raise TypeError(f"Unsupported Flywheel metadata type: {type(value).__name__}")

        def snapshot(value):
            """Copy SDK metadata into strict JSON-compatible values."""
            document = value.to_dict()
            if not isinstance(document, dict):
                raise TypeError("Flywheel metadata must be a dictionary.")
            return json.loads(json.dumps(document, default=encodeValue, allow_nan=False))

        def fetch(method, identifier, kind):
            """Keep SDK exception text and authenticated URLs out of warnings."""
            try:
                return method(identifier)
            except Exception:
                pglMessages.warning(
                    f"Could not read Flywheel {kind} metadata. "
                    "No MRI collection was returned.",
                    level=0,
                )
                return None

        projectID, sessionID = next(iter(sessionKeys))
        acquisitions = {}
        scans = []

        with pglProgress("Preparing deferred Flywheel MRI scans") as progress:
            progress(0, "Reading project metadata")
            project = fetch(self.fw.get_project, projectID, "project")
            if project is None: return None
            projectMetadata = snapshot(project)

            progress(None, "Reading session metadata")
            session = fetch(self.fw.get_session, sessionID, "session")
            if session is None: return None
            sessionMetadata = snapshot(session)

            for index, selectedScan in enumerate(selected):
                acquisitionID = selectedScan["acquisitionID"]
                filename = selectedScan["fileName"]
                progress(index / len(selected), f"Reading metadata: {filename}")

                if acquisitionID not in acquisitions:
                    acquisition = fetch(
                        self.fw.get_acquisition, acquisitionID, "acquisition"
                    )
                    if acquisition is None: return None
                    acquisitions[acquisitionID] = (acquisition, snapshot(acquisition))

                acquisition, acquisitionMetadata = acquisitions[acquisitionID]
                matches = [
                    file for file in acquisition.files or []
                    if file.name == filename
                ]

                if len(matches) != 1:
                    pglMessages.warning(
                        f"MRI scans were not attached: selected file "
                        f"{filename!r} is missing or ambiguous.",
                        level=0,
                    )
                    return None

                file = matches[0]
                fileMetadata = snapshot(file)

                # Pin the version observed now. Future reads request this
                # version explicitly and verify its recorded size.
                # SHA-384 is also checked when Flywheel supplies a checksum.
                try:
                    backing = pglFlywheelMRIBacking(
                        self.fw,
                        acquisitionID,
                        filename,
                        version=fileMetadata.get("version"),
                        sizeBytes=fileMetadata.get("size"),
                        sourceHash=fileMetadata.get("hash"),
                    )
                except (TypeError, ValueError) as error:
                    pglMessages.warning(
                        f"MRI scans were not attached: {filename!r}: {error}",
                        level=0,
                    )
                    return None

                provenance = pglMRIScan._copyMetadata({
                    "provider": "flywheel",
                    "selection": dict(selectedScan),
                    "versionPinnedAt": "toMRI",
                    "project": projectMetadata,
                    "session": sessionMetadata,
                    "acquisition": acquisitionMetadata,
                    "file": fileMetadata,
                })

                scan = pglMRIScan(
                    name=selectedScan["name"],
                    sourceMetadata=provenance,
                )
                if scan.attachBacking(backing, cacheDirectory=cachePath) is None:
                    return None

                scans.append(scan)

            # Install only after every selected source has been validated.
            mri = pglMRI()
            mri.scans = scans
            progress.finish(f"Prepared {len(scans)} MRI scans; no NIfTI payloads downloaded")

        return mri
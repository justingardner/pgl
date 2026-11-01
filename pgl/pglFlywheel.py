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
# Flywheel connection
################################################################
class pglFlywheel:
    def __init__(self, apikey):
        """Connect to Flywheel and load only the project list."""
        self.selection = pglFlywheelChoose()
        self.fw = None

        # Load library
        try:
            import flywheel
        except ImportError:
            pglMessages.warning(
                "Unable to import flywheel library. "
                "Install using: pip install flywheel-sdk"
            )
            return

        # Initiate connection and verify login.
        # Avoid logging SDK exception details that might contain credentials.
        try:
            fw = flywheel.Client(apikey)
            user = fw.get_current_user()
        except Exception:
            pglMessages.warning("Could not connect or log in to Flywheel.")
            return

        pglMessages.message(
            f"Logged in as: {user.firstname} {user.lastname} ({user.email})"
        )

        # Fetch projects only; do not walk their sessions or acquisitions.
        try:
            flywheelProjects = list(fw.projects())
        except Exception:
            pglMessages.warning("Could not load Flywheel projects.")
            return

        # A unique environment variable keeps different connections separate.
        # The subprocess inherits its value; JSON stores only its name.
        connectionEnv = f"PGL_FLYWHEEL_API_KEY_{uuid4().hex}"
        os.environ[connectionEnv] = apikey
        _flywheelClients[connectionEnv] = fw

        self.fw = fw
        self.selection.projects = [
            pglFlywheelProject(
                name=project.label,
                id=project.id,
                _connectionEnv=connectionEnv,
            )
            for project in flywheelProjects
        ]

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
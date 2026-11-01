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
        self.choose = pglFlywheelChoose()
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
        self.choose.projects = [
            pglFlywheelProject(
                name=project.label,
                id=project.id,
                _connectionEnv=connectionEnv,
            )
            for project in flywheelProjects
        ]
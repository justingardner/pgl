################################################################
#   filename: pglDatabase.py
#    purpose: Database-backed sessions, datasets, provenance,
#             checkpoint publication, and schema migrations
#         by: JLG
#       date: 10/20/2026
################################################################

import hashlib
import json
import posixpath
import re
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

import psycopg
from psycopg import sql
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

from .pglData import pglDataMatrix
from .pglMessages import pglMessages
from .pglStorage import pglStorage
from .pglInfo import pglInfo
from .pglSettings import pglTraitSettings
from traitlets import Instance, List, Unicode, default
from .pglSettings import pglSettingsManager
from .pglDialog import pglDialogs

##################################################
# Database settings
##################################################
class pglDatabaseSettings(pglTraitSettings):

    name = Unicode("default", help="Display name for these settings")
    databasePath = Unicode("~/data/pgl", isPath=True, help="Directory containing the PostgreSQL configuration, server log, and database files. Supports ~ for the home directory.")
    cacheDir = Unicode("~/data/pgl/cache", help="Local directory for files downloaded from database (e.g. raw mne files). Cached files are not automatically deleted.", isPath=True)
    username = Unicode(help="Creator username for data-store provenance. Defaults to the OS login username; editable to use a consistent identity across machines.")
    password = Unicode("", hasSetButton=True, buttonFunction="storePassword", buttonLabel="Store in keychain", isPassword=True, transientInput=True, help="Enter the existing PostgreSQL login password, then click Store in keychain to verify and save it securely. This does not change the server password. Unsubmitted input is discarded. Cancel does not undo a successful keychain write.")
    
    @default("username")
    def _default_username(self):
        import getpass
        return getpass.getuser()

    def save(self, filename=None, filesystem=None, filesystemPrefix=None):
        """
        Save database settings.

        If filename is omitted, save under the local database-settings
        directory using the settings name.
        """
        if filename is None:
            filename = self.saveDir()

        super().save(
            filename=filename,
            filesystem=filesystem,
            filesystemPrefix=filesystemPrefix,
        )
        pglMessages.message(f"Saved database settings to {filename}")

    def saveDir(self):
        return pglDatabase.getSettingsDir() / f"{pglStorage.makeValidFilename(self.name)}.json"
    
    def storePassword(self, password):
        """Test and securely store the selected profile's PostgreSQL password.

        Called by the dialog with transient widget input.
        Never assigns the password to a settings trait.
        Does not change the PostgreSQL server password.

        Returns an empty string only after successful connection cleanup
        and credential storage, instructing the dialog to clear its input.
        """
        from .pglPostgres import pglPostgres

        if not isinstance(password, str) or not password:
            raise ValueError("Enter a nonempty password.")

        if any(character in password for character in ("\n", "\r", "\x00")):
            raise ValueError("Passwords must contain no newline or NUL characters.")

        try:
            postgres = pglPostgres.fromSettings(settings=self)

            if postgres is None:
                raise RuntimeError("Could not resolve PostgreSQL configuration.")

            # Fail before attempting login if secure storage is unavailable.
            if postgres._passwordStore() is None:
                raise RuntimeError("Secure password storage is unavailable.")

            # Attempt only the supplied password. Do not fall back to the
            # keychain, prompt in the subprocess, or save during connect().
            with postgres.connect(
                password=password,
                rememberPassword=False,
                useSavedPassword=False,
            ) as connection:
                if connection is None:
                    raise RuntimeError("Password verification was cancelled.")

            # Connection and tunnel cleanup succeeded. Store using exactly
            # the identity used by ordinary client authentication.
            if not postgres._saveVerifiedPassword(postgres.databaseUser, password):
                raise RuntimeError("Could not save the password securely.")

            return ""

        finally:
            # Does not guarantee secure memory erasure.
            password = None


##################################################
# Database settings dialog wrapper
##################################################
class pglDatabaseSettingsList(pglTraitSettings):

    settingsList = List(
        Instance(pglDatabaseSettings),
        buttons=True,
        setDefault=True,
        settingsListKey="name",
        traitDisplayName="Choose database settings",
        help="List of database settings",
    )

    def __init__(self, settingsList=None):
        super().__init__()
        if settingsList is not None:
            self.settingsList = settingsList

########################################################
# pglDatabase
########################################################
class pglDatabase:
    """Coordinate PostgreSQL-backed scientific checkpoints.

    Main entry points:
        loadSession(), saveSession(), saveRawSession().
        loadDataset(), saveDataset().
        Session/dataset reporting and lifecycle operations.

    Responsibilities:
        Resolve database configuration and registered payload locations.
        Publish and restore session and dataset checkpoints.
        Validate lineage, history boundaries, manifests, and catalog links.
        Coordinate soft deletion and source dependency protection.
        Initialize and migrate the scientific schema.

    Dependencies:
        pglPostgres owns SQL connections, SSH tunnels, and local server setup.
        pglStorage owns standalone matrix persistence, hashing, verified file
        copies, and general scientific-save validation.

    Transaction ownership:
        Public save/lifecycle methods own their connections and transactions.
        Advanced APIs generally use caller-owned connections; their docstrings
        specify publication and commit responsibility.

    Filesystem writes are not rolled back by PostgreSQL. Failed saves may
    leave unregistered files, but publication must not be reported before
    successful commit and connection cleanup.

    Checkpoint manifests and pglCheckpointFilesystem belong here because
    they implement the database checkpoint representation, even when an
    individual helper performs no SQL.

    Existing method docstrings describe return conventions and limitations.
    Structural and provenance checks do not establish scientific validity.
    """

    # -------------------------------------------------------------------------
    # Shared constants
    # -------------------------------------------------------------------------
    MIGRATION_LOCK = 734_291_806_115

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    @classmethod
    def settings(cls):
        """
        Edit database settings using the standard settings dialog.
        """
        settingsList = cls.getSettings(returnSettingsList=True)

        original = pglDatabaseSettingsList(settingsList)

        modified = pglDialogs.traitsDialog(original)

        cls._saveModifiedSettings(modified, original)

    @classmethod
    def getSettingsDir(cls):
        """
        Return the directory containing local database-settings profiles.
        """
        pglDir = pglSettingsManager.getPGLSettingsDir()
        if pglDir is None:
            raise OSError("Could not resolve the PGL settings directory.")

        settingsDir = pglDir / "databaseSettings"
        settingsDir.mkdir(parents=True, exist_ok=True)
        return settingsDir

    @classmethod
    def getSettings(cls, settingsName=None, settings=None, returnSettingsList=False):
        """
        Load database settings.

        Explicit settings supersede settingsName.
        With neither, return the default settings.
        returnSettingsList=True ignores other arguments and returns all
        settings, with the default first.
        """
        if returnSettingsList:
            settings = None
            settingsName = None

        if settings is not None:
            if not isinstance(settings, pglDatabaseSettings):
                pglMessages.warning("Settings must be pglDatabaseSettings.")
                return None
            return settings

        settingsDir = cls.getSettingsDir()

        if settingsName is not None:
            filename = pglStorage.makeValidFilename(settingsName)
            settingsPath = settingsDir / f"{filename}.json"

            if not settingsPath.is_file():
                pglMessages.warning(f"Database settings file '{settingsPath}' not found.")
                return None

            return pglDatabaseSettings.load(filename=settingsPath)

        settingsList = []
        for filename in sorted(settingsDir.glob("*.json")):
            loaded = pglDatabaseSettings.load(filename=filename)
            if not isinstance(loaded, pglDatabaseSettings):
                raise ValueError(f"Invalid database settings file: {filename}")
            settingsList.append(loaded)

        if not settingsList:
            pglMessages.message("No saved database settings found, creating default settings.")
            settings = pglDatabaseSettings()
            settings.isDefault = True
            settings.save()
            settingsList = [settings]
        else:
            defaults = [s for s in settingsList if s.isDefault]

            if defaults:
                settings = defaults[0]

                if len(defaults) > 1:
                    pglMessages.warning(
                        f"Found {len(defaults)} default database settings, keeping only the first."
                    )
                    for other in defaults[1:]:
                        other.isDefault = False
                        other.save()
            else:
                settings = settingsList[0]
                settings.isDefault = True
                settings.save()
                pglMessages.message(
                    f"No default database settings found, using: {settings.name}."
                )

            # Move the default to the front.
            settingsList.remove(settings)
            settingsList.insert(0, settings)

        if returnSettingsList:
            return settingsList

        return settingsList[0]
    
    @classmethod
    def _saveModifiedSettings(cls, modifiedSettingsList, originalSettingsList):
        """
        Save new or modified database settings and trash removed files.

        Renaming a profile changes only its local settings filename.
        It does not rename or move databasePath or cacheDir.
        """
        if modifiedSettingsList is None:
            return

        modifiedList = modifiedSettingsList.settingsList
        originalList = originalSettingsList.settingsList

        # Validate before writing anything. Names must map to unique files.
        filenames = set()
        uuids = set()

        for settings in modifiedList:
            filename = pglStorage.makeValidFilename(settings.name)

            if not filename:
                pglMessages.warning("Database settings must have a nonempty valid name.")
                return

            if filename in filenames:
                pglMessages.warning(
                    f"Database settings names must produce unique filenames: {settings.name}"
                )
                return

            if not settings.uuid or settings.uuid in uuids:
                pglMessages.warning("Database settings must have unique, nonempty UUIDs.")
                return

            filenames.add(filename)
            uuids.add(settings.uuid)

        # Ensure exactly one default when profiles remain.
        if modifiedList:
            defaults = [s for s in modifiedList if s.isDefault]
            selectedDefault = defaults[0] if defaults else modifiedList[0]

            for settings in modifiedList:
                settings.isDefault = settings is selectedDefault

        originalByUUID = {s.uuid: s for s in originalList}

        # Write all new/changed profiles before removing obsolete files.
        for settings in modifiedList:
            original = originalByUUID.get(settings.uuid)

            if original is None:
                pglMessages.message(f"Saving new database settings {settings.name}")
                settings.save()
            elif not original.equals(settings):
                settings.save()

        # Preserve any filename still used by the resulting settings list,
        # including when profiles exchange names.
        retainedPaths = {s.saveDir() for s in modifiedList}

        for original in originalList:
            oldPath = original.saveDir()

            if oldPath not in retainedPaths and oldPath.exists():
                pglSettingsManager.moveToTrash(oldPath)

    # -------------------------------------------------------------------------
    # Database setup
    # -------------------------------------------------------------------------
    # User entry point for local server setup and schema initialization.

    # ++++++++++++++++++++++++++++++ initDatabase +++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def initDatabase(cls, postgres=None, *, settings=None, settingsName=None):
        """Set up the locally managed database and initialize its schema.

        Uses default PGL settings unless settings or settingsName is supplied.
        An explicit postgres object may be supplied instead.

        Loads configuration from databasePath. If missing, creates a local
        configuration using that directory as installDirectory.

        Ensures PostgreSQL is installed, initializes missing server data,
        starts the server, creates the database/application access, and
        applies pending schema migrations.

        Never deletes existing data or resets an existing database.
        Local server setup currently supports macOS/Homebrew only.
        Returns False on cancellation or an operational connection failure.
        """
        from .pglPostgres import pglPostgres

        if postgres is not None and (settings is not None or settingsName is not None):
            raise ValueError("Supply either postgres or PGL settings, not both.")

        if postgres is None:
            resolvedSettings = cls.getSettings(settings=settings, settingsName=settingsName)

            if resolvedSettings is None:
                pglMessages.warning("Could not resolve PGL settings.", level=0)
                return False

            directory = resolvedSettings.databasePath.strip()

            if not directory:
                pglMessages.warning("Set databasePath in PGL settings before initializing the database.", level=0)
                return False

            configurationFile = posixpath.join(directory.rstrip("/") + "/", pglPostgres.CONFIGURATION_FILENAME)

            try:
                postgres = pglPostgres.load(configurationFile)
            except FileNotFoundError:
                # Only bootstrap a local installation. Never interpret a
                # missing remote configuration as a request for local setup.
                _, localDirectory, isLocal = pglPostgres._resolveConfigurationFile(directory)

                if not isLocal:
                    pglMessages.warning("Remote PostgreSQL configuration was not found. Initialize the server on its installation machine.", level=0)
                    return False

                postgres = pglPostgres(installDirectory=localDirectory)
                pglMessages.message(f"Creating local PostgreSQL configuration in {localDirectory}.")

        if not isinstance(postgres, pglPostgres):
            raise TypeError("postgres must be a pglPostgres instance.")

        postgres._validateValues()

        # The existing administration and migration implementation manages
        # a local server only. Do not accidentally administer a local server
        # when the loaded configuration describes a remote connection.
        if (
            getattr(postgres, "_configurationLoadedRemotely", False)
            or postgres.connectionTransport != "direct"
            or postgres.connectionHost != postgres.postgresHost
            or postgres.connectionPort != postgres.postgresPort
        ):
            pglMessages.warning("Database initialization currently requires the locally managed server configuration. Run it on the installation machine.", level=0)
            return False

        try:
            postgres.install()

            if postgres.init() is False:
                return False

            postgres.start()

            if not postgres.createDatabase():
                return False

            return cls._initDatabaseSchema(postgres=postgres)

        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not initialize PostgreSQL: {error}", level=0)
            return False


    # -------------------------------------------------------------------------
    # Sessions: loading and publication
    # -------------------------------------------------------------------------
    # User entry points for loading saved sessions and publishing raw or derived checkpoints.

    # ++++++++++++++++++++++++++++++ loadSession ++++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    # Direct callers in this module: _loadPipelineInput, getDatasetBehaviorData, getDatasetSession.
    # +++++++++++++++++++++++++++++ loadSession +++++++++++++++++++++++++++++
    @classmethod
    def loadSession(cls, sessionID, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, includeDeleted=False, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load a completed checkpoint with retained caller-side payload access.

        User-facing entry point used by pglSession.loadSession(), dataset
        member loading, and pipeline input loading.

        Resolves configuration, selects the checkpoint-contract loader,
        restores annotations, and binds the saved history baseline.

        Supports legacy raw/derived contracts and derived v2 matrix products.
        Scientific fields remain deferred where supported. Matrix restoration
        reads descriptors and Zarr metadata without computing scientific arrays.

        Registered server-local roots resolve through the selected caller's
        local or SSH access context. Returned readers retain payload access,
        not the SQL connection or its tunnel.

        Returns:
            A restored pglSession.
            None for invalid arguments, unavailable configuration, connection
            cancellation, an unknown/hidden checkpoint, or handled SQL errors.

        Manifest, integrity, contract, SSH, and other unhandled errors propagate.

        On failure after reconstruction, closes attached matrices and the
        checkpoint filesystem without resolving deferred scientific fields.
        On success, those resources remain available to the caller.

        cacheDirectory overrides settings.cacheDir for MNE checkpoints.
        includeDeleted permits inspection without restoring lifecycle state.
        Does not publish or modify database records.
        """
        from .pglPostgres import pglPostgres

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            pglMessages.warning("sessionID must be a nonnegative PostgreSQL bigint.", level=1)
            return None

        if type(includeDeleted) is not bool:
            pglMessages.warning("includeDeleted must be a boolean.", level=1)
            return None

        if directLocalPolicy not in ("materialize", "immutable"):
            pglMessages.warning("directLocalPolicy must be 'materialize' or 'immutable'.", level=1)
            return None

        session = None
        checkpoint = None

        # +++++++++++++++++++++++ closeFailedLoad +++++++++++++++++++++++++++
        def closeFailedLoad():
            """Release privately loaded resources without opening lazy fields.

            Called only when this public load cannot return its constructed
            session. Cleanup errors must not replace the original failure.
            """
            if session is None:
                return

            for matrix in session.dataMatrices.values():
                try:
                    matrix.close()
                except Exception:
                    pass

            # Do not access session.mne.raw or other deferred fields to clean
            # them up. The loader has not intentionally resolved those fields.
            filesystem = session.filesystem

            if isinstance(filesystem, pglCheckpointFilesystem):
                try:
                    filesystem.close()
                except Exception:
                    pass

        try:
            try:
                resolvedSettings = cls.getSettings(settings=settings, settingsName=settingsName)

                if resolvedSettings is None:
                    pglMessages.warning("Cannot load session: could not resolve PGL settings.", level=0)
                    return None

                postgres = pglPostgres.fromSettings(settings=resolvedSettings)

            except (TypeError, ValueError) as error:
                pglMessages.warning(f"Cannot load session: invalid settings or configuration: {error}", level=1)
                return None

            if postgres is None:
                return None

            with postgres.connect(sslRootCert=sslRootCert) as connection:
                if connection is None:
                    return None

                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                    table = sql.Identifier(postgres.databaseSchema, "session")
                    statement = sql.SQL("SELECT origin, contract_version, description, checkpoint_name FROM {} WHERE session_id = %s AND (%s OR deleted_at IS NULL)").format(table)
                    checkpoint = connection.execute(statement, (sessionID, includeDeleted)).fetchone()

                    if checkpoint is not None:
                        origin, contractVersion, description, checkpointName = checkpoint
                        loadOptions = {
                            "storageOptionsByBackend": storageOptionsByBackend,
                            "includeDeleted": includeDeleted,
                        }

                        if (origin, contractVersion) == ("raw", "pgl.raw.behavior.v1"):
                            loader = cls.loadRawSession

                            # Legacy raw behavioral manifests declare no
                            # matrix products. Reject unexpected catalog links.
                            cls._verifySessionProducts(connection, postgres, sessionID, [])

                        elif (origin, contractVersion) == ("raw", "pgl.raw.session.v1"):
                            loader = cls.loadRawSessionBundle
                            resolvedCache = cls._resolveCacheDirectory(cacheDirectory, settings=resolvedSettings)
                            loadOptions.update(cacheDirectory=resolvedCache, directLocalPolicy=directLocalPolicy)

                        elif origin == "derived" and contractVersion in (
                            "pgl.derived.behavior.v1",
                            "pgl.derived.behavior.v2",
                            "pgl.derived.session.v1",
                            "pgl.derived.session.v2",
                        ):
                            loader = cls.loadDerivedSession
                            hasMNE = contractVersion in ("pgl.derived.session.v1", "pgl.derived.session.v2")
                            resolvedCache = cls._resolveCacheDirectory(cacheDirectory, settings=resolvedSettings) if hasMNE else cacheDirectory
                            loadOptions.update(cacheDirectory=resolvedCache, directLocalPolicy=directLocalPolicy)

                        else:
                            raise ValueError(f"Unsupported checkpoint origin/contract: {origin!r}, {contractVersion!r}")

                        session = loader(connection, postgres, sessionID, **loadOptions)

            # SQL and its tunnel have closed. Successful reconstruction
            # retains independently owned contexts for deferred payload reads.
            if checkpoint is None:
                reason = "Unknown checkpoint" if includeDeleted else "Unknown or deleted checkpoint"
                pglMessages.warning(f"{reason}: {sessionID}", level=0)
                return None

            session.sessionID = sessionID
            session.description = checkpoint[2]
            session.checkpointName = checkpoint[3]
            cls._bindCheckpointBaseline(session, sessionID, postgres)

        except psycopg.Error as error:
            closeFailedLoad()
            pglMessages.warning(f"Could not load session {sessionID}: {error}", level=2)
            return None

        except BaseException:
            closeFailedLoad()
            raise

        pglMessages.message(f"Loaded {checkpoint[0]} checkpoint {sessionID}; scientific payload values remain deferred.")
        return session

    # ++++++++++++++++++++++++++++++ saveSession ++++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def saveSession(cls, session, *, parentSessionID=None, settings=None, settingsName=None, backendID=None, checkpointName=None, description=None, ingestMethod="manual", storageOptions=None, sslRootCert=None):
        """Save current contents and history as a derived checkpoint.

        Defaults parentSessionID to session.sessionID.
        None annotation arguments carry forward the working session values.
        Explicit blank strings clear the corresponding annotations.

        Rejects failed working state and changes to an established baseline's
        historical prefix before opening a database connection.

        Returns the new or reused checkpoint ID, or None for missing-parent
        rejection, unavailable configuration, or connection cancellation.

        Prepares new baseline metadata before commit. Updates sessionID,
        checkpointName, description, and baseline only after commit and
        connection cleanup.

        Database changes are transactional; payload writes are not.
        Failed saves may leave unregistered files. Other errors propagate.
        Accumulated action history is retained.

        The working object must not be modified concurrently while saving.
        """
        from time import perf_counter
        from .pglPostgres import pglPostgres

        pglStorage.requireSafeSave(session)
        cls._getCheckpointHistoryBoundary(session)

        resolvedParentID = session.sessionID if parentSessionID is None else parentSessionID

        if resolvedParentID is None:
            pglMessages.warning("Cannot save session: this session has no loaded or saved source checkpoint. Use saveRawSession() to create a parentless root, or supply parentSessionID.", level=2)
            return None

        for label, value in (("parentSessionID", resolvedParentID), ("backendID", backendID)):
            if label == "backendID" and value is None:
                continue

            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        # ++++++++++++++++++++++++++ normalizeAnnotation ++++++++++++++++++++++++++
        # Local implementation within saveSession.
        def normalizeAnnotation(value, label):
            if value is None:
                return None

            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"{label} must be a string without NUL, or None.")

            # Preserve nonblank text verbatim, matching description semantics.
            return value if value.strip() else None

        resolvedCheckpointName = normalizeAnnotation(session.checkpointName if checkpointName is None else checkpointName, "checkpointName")
        resolvedDescription = normalizeAnnotation(session.description if description is None else description, "description")

        if not isinstance(ingestMethod, str) or ingestMethod not in {"manual", "automated"}:
            raise ValueError("ingestMethod must be manual or automated.")

        resolvedSettings = cls.getSettings(settings=settings, settingsName=settingsName)

        if resolvedSettings is None:
            pglMessages.warning("Cannot save session: could not resolve PGL settings.", level=0)
            return None

        username = resolvedSettings.username

        if not isinstance(username, str) or not username.strip() or "\x00" in username:
            pglMessages.warning("Cannot save session: set a valid username in PGL settings.", level=0)
            return None

        postgres = pglPostgres.fromSettings(settings=resolvedSettings)

        if postgres is None:
            return None

        if backendID is None and not postgres.storageLocations:
            pglMessages.warning("Cannot save session: no storageLocations are configured in the PostgreSQL configuration.", level=0)
            return None

        databasePath = resolvedSettings.databasePath.strip()
        saveStarted = perf_counter()
        pglMessages.message("Saving session: connecting to PostgreSQL...")

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.transaction():
                if backendID is None:
                    pglMessages.message("Saving session: resolving storage destination...")
                    backendID = cls._getConfiguredStorageBackend(connection, postgres, storageOptions=storageOptions)

                pglMessages.message("Saving session: resolving checkpoint owner...")
                createdBy = cls.getOrCreateUser(connection, postgres, username)

                pglMessages.message("Saving session: checking current contents and preparing checkpoint...")
                contentsStarted = perf_counter()
                sessionID = cls._saveDerivedSession(connection, postgres, session, createdBy, backendID, parentSessionID=resolvedParentID, checkpointName=resolvedCheckpointName, description=resolvedDescription, ingestMethod=ingestMethod, storageOptions=storageOptions)

                # Prepare baseline metadata while errors can still roll back
                # checkpoint publication.
                pglStorage.requireSafeSave(session)
                cls._getCheckpointHistoryBoundary(session)
                pendingBaseline = cls._prepareCheckpointBaseline(session, sessionID, postgres)

                pglMessages.message(f"Saving session: checkpoint ready ({perf_counter() - contentsStarted:.1f} s); committing transaction...")

        # Publication and connection cleanup have succeeded.
        session.sessionID = sessionID
        session.checkpointName = resolvedCheckpointName
        session.description = resolvedDescription
        session._checkpointBaseline = pendingBaseline

        pglMessages.message(f"Checkpoint {sessionID} is complete. Database configuration: {databasePath} (database={postgres.databaseName}, schema={postgres.databaseSchema}; {perf_counter() - saveStarted:.1f} s total).", wrapText=False)
        return sessionID

    # +++++++++++++++++++++++++++++ saveRawSession ++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def saveRawSession(cls, session, *, settings=None, settingsName=None, backendID=None, storageOptions=None, sslRootCert=None, allowMissingMetadata=False):
        """Save a parentless acquisition/root checkpoint.

        Behavioral labels/files come from saved acquisition sources.
        MNE-only labels come from session.experimentName and session.subjectID.

        Attached data matrices are derived products: warn and omit them from
        the raw checkpoint. Working attachments remain unchanged. Use
        saveSession() to persist those matrices.

        Missing labels warn and cancel before SQL or export.
        allowMissingMetadata=True explicitly permits unset labels.
        Malformed/unreadable settings are not bypassed by that override.

        Behavioral-only, MNE-only, and mixed sessions are supported.
        MNE objects are serialized from current contents; original vendor
        acquisition files are not automatically archived.

        Identical active raw checkpoints are reused with a warning.
        Matching deleted checkpoints are left deleted; a new checkpoint
        is created, subject to the current schema migration being installed.

        Returns the checkpoint ID, or None for missing-label rejection,
        unavailable configuration, or connection cancellation.

        Database changes are transactional; payload writes are not.
        Success is reported only after commit and connection cleanup.
        """
        from time import perf_counter

        from .pglPostgres import pglPostgres

        if not cls._checkRawSessionLabels(session, allowMissingMetadata=allowMissingMetadata):
            return None

        # Do not validate or materialize matrices that will not be saved.
        # Do not remove them, even temporarily, from the working session.
        if session.dataMatrices:
            productNames = list(session.dataMatrices)
            pglMessages.warning(f"saveRawSession() saves acquisition contents only. Attached data matrices {productNames!r} will not be included in this checkpoint. Working attachments are unchanged; use saveSession() to preserve them.", level=1)

        resolvedSettings = cls.getSettings(settings=settings, settingsName=settingsName)

        if resolvedSettings is None:
            pglMessages.warning("Cannot save raw session: could not resolve PGL settings.", level=0)
            return None

        username = resolvedSettings.username

        if not isinstance(username, str) or not username.strip() or "\x00" in username:
            pglMessages.warning("Cannot save raw session: set a valid username in PGL settings.", level=0)
            return None

        postgres = pglPostgres.fromSettings(settings=resolvedSettings)

        if postgres is None:
            return None

        # Numeric checkpoint IDs belong to one database. Until transfers
        # are implemented, do not reinterpret a loaded session's parent ID
        # against another database, even with an explicit parentSessionID.
        baseline = session._checkpointBaseline

        if baseline and baseline["databaseIdentity"] != cls._checkpointDatabaseIdentity(postgres):
            pglMessages.warning(
                "Cannot save this session to a different database. "
                "Cross-database checkpoint transfer is not implemented; "
                "session IDs are local to their database."
            )
            return None

        if backendID is None and not postgres.storageLocations:
            pglMessages.warning("Cannot save session: no storageLocations are configured in the PostgreSQL configuration.", level=0)
            return None

        databasePath = resolvedSettings.databasePath.strip()
        saveStarted = perf_counter()
        pglMessages.message("Saving raw session: connecting to PostgreSQL...")

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.transaction():
                if backendID is None:
                    pglMessages.message("Saving raw session: resolving storage destination...")
                    backendID = cls._getConfiguredStorageBackend(connection, postgres, storageOptions=storageOptions)

                pglMessages.message("Saving raw session: resolving checkpoint owner...")
                createdBy = cls.getOrCreateUser(connection, postgres, username)

                pglMessages.message("Saving raw session: checking for an existing checkpoint or saving new contents...")
                contentsStarted = perf_counter()
                sessionID = cls._saveRawSession(connection, postgres, session, createdBy, backendID, storageOptions=storageOptions)

                pglMessages.message(f"Saving raw session: checkpoint ready ({perf_counter() - contentsStarted:.1f} s); committing transaction...")

        pglMessages.message(f"Raw checkpoint {sessionID} is complete. Database configuration: {databasePath} (database={postgres.databaseName}, schema={postgres.databaseSchema}; {perf_counter() - saveStarted:.1f} s total).", wrapText=False)
        return sessionID


    # -------------------------------------------------------------------------
    # Sessions: annotations and lifecycle
    # -------------------------------------------------------------------------
    # User entry points for annotation changes and coordinated soft deletion or restoration.

    # +++++++++++++++++++++++++ setSessionDescription +++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def setSessionDescription(cls, sessionID, description, *, settings=None, settingsName=None, sslRootCert=None):
        """Set or clear an active saved checkpoint's description.

        Accepts raw and derived checkpoints.
        None or blank text clears the description.

        Updates only database metadata, not payloads, manifest identity,
        lineage, or saved action history. Previous descriptions are not
        retained by this method.

        Returns True for a committed change, False for an unchanged,
        unknown, or deleted checkpoint, and None for invalid arguments,
        unavailable configuration/connection, or database failure.
        """
        from .pglPostgres import pglPostgres

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            pglMessages.warning("sessionID must be a nonnegative PostgreSQL bigint.", level=1)
            return None

        if description is not None and (not isinstance(description, str) or "\x00" in description):
            pglMessages.warning("description must be a string without NUL, or None.", level=1)
            return None

        if description is not None:
            description = description.strip() or None

        try:
            try:
                postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)
            except (TypeError, ValueError) as error:
                pglMessages.warning(f"Cannot update session description: invalid settings or configuration: {error}", level=1)
                return None

            if postgres is None:
                return None

            table = sql.Identifier(postgres.databaseSchema, "session")

            with postgres.connect(sslRootCert=sslRootCert) as connection:
                if connection is None:
                    return None

                with connection.transaction():
                    statement = sql.SQL("SELECT description, deleted_at FROM {} WHERE session_id = %s FOR UPDATE").format(table)
                    record = connection.execute(statement, (sessionID,)).fetchone()

                    if record is None:
                        outcome = "unknown"
                    elif record[1] is not None:
                        outcome = "deleted"
                    elif record[0] == description:
                        outcome = "unchanged"
                    else:
                        statement = sql.SQL("UPDATE {} SET description = %s WHERE session_id = %s RETURNING description").format(table)
                        updated = connection.execute(statement, (description, sessionID)).fetchone()

                        if updated != (description,):
                            raise RuntimeError("The database did not apply the requested description.")

                        outcome = "updated"

        except psycopg.Error as error:
            pglMessages.warning(f"Could not update description for session {sessionID}: {error}. If the connection failed during commit, check the saved description before retrying.", level=2)
            return None

        if outcome == "unknown":
            pglMessages.warning(f"Unknown session: {sessionID}", level=0)
            return False

        if outcome == "deleted":
            pglMessages.warning(f"Cannot change the description of deleted session {sessionID}. Undelete it first.", level=2)
            return False

        if outcome == "unchanged":
            pglMessages.message(f"Session {sessionID} already has that description; nothing changed.")
            return False

        if description is None:
            pglMessages.message(f"Description cleared for session {sessionID}.")
        else:
            pglMessages.message(f"Description updated for session {sessionID}: {cls._printValue(description)}", wrapText=False)

        return True

    # +++++++++++++++++++++++++++++ deleteSession +++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def deleteSession(cls, sessionID, *, deleteDownstreamSessions=False, settings=None, settingsName=None, sslRootCert=None):
        """Soft-delete a session, optionally including active descendants.

        Without deleteDownstreamSessions, active descendants block deletion.

        With deleteDownstreamSessions=True, print every session to be newly
        deleted and require an explicit 'yes'. Recheck the exact deletion
        set under locks after confirmation.

        Every new deletion records a group, even for a single session.
        Previously deleted descendants are not enrolled in the new group.

        Returns:
            True: a change committed.
            False: unchanged, unknown, blocked, or confirmation cancelled.
            None: invalid arguments, unavailable configuration/connection,
                  or database failure.

        Unknown sessions warn at level 0, invalid arguments at level 1,
        and blocked operations/database failures at level 2.

        Never removes or rereads payloads.
        """
        return cls._changeSessionDeletionState(sessionID, deleted=True, deleteDownstreamSessions=deleteDownstreamSessions, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert)

    # ++++++++++++++++++++++++++++ undeleteSession ++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def undeleteSession(cls, sessionID, *, settings=None, settingsName=None, sslRootCert=None):
        """Restore an entire deletion group through its top-level session.

        Calls on other group members are refused. Required parents outside
        the group must already be active.

        Legacy deleted sessions without group records are restored alone,
        subject to the same parent requirement.

        Returns:
            True: restoration committed.
            False: unchanged, unknown, or blocked.
            None: invalid arguments, unavailable configuration/connection,
                  or database failure.

        Unknown sessions warn at level 0, invalid arguments at level 1,
        and blocked operations/database failures at level 2.

        Does not reread or rewrite payloads.
        """
        return cls._changeSessionDeletionState(sessionID, deleted=False, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert)


    # -------------------------------------------------------------------------
    # Datasets: loading and publication
    # -------------------------------------------------------------------------
    # User entry points for immutable dataset checkpoint publication and restoration.

    # ++++++++++++++++++++++++++++++ loadDataset ++++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def loadDataset(cls, datasetID, *, settings=None, settingsName=None, sslRootCert=None, includeDeleted=False, storageOptionsByBackend=None, cacheDirectory=None, directLocalPolicy="materialize"):
        """Restore metadata, membership, and unopened product references.

        Does not construct sessions, open Zarr stores, or rebuild products.
        includeDeleted permits inspection without restoring protection.
        """
        from psycopg.rows import tuple_row
        from .pglData import pglDataMatrixReference
        from .pglDataset import pglDataset
        from .pglSerialize import pglSerialize

        if type(datasetID) is not int or not 0 <= datasetID <= 9223372036854775807:
            raise ValueError("datasetID must be a nonnegative PostgreSQL bigint.")

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        if directLocalPolicy not in ("materialize", "immutable"):
            raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'.")

        shell = pglDataset(settings=settings, settingsName=settingsName)
        resolvedSettings, postgres = cls._resolveDatasetStorage(shell)

        if postgres is None:
            return None

        datasetsTable = sql.Identifier(postgres.databaseSchema, "dataset")
        membersTable = sql.Identifier(postgres.databaseSchema, "dataset_members")
        productsTable = sql.Identifier(postgres.databaseSchema, "dataset_products")
        checkpoint = None
        memberRows = []
        productRows = []

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    statement = sql.SQL("""
                        SELECT family_uuid, name, checkpoint_name, description,
                               status, contract_version, metadata
                        FROM {}
                        WHERE dataset_id = %s
                          AND (%s OR deleted_at IS NULL)
                    """).format(datasetsTable)

                    cursor.execute(statement, (datasetID, includeDeleted))
                    checkpoint = cursor.fetchone()

                    if checkpoint is not None:
                        statement = sql.SQL("SELECT member_index, session_id, entry_name, labels FROM {} WHERE dataset_id = %s ORDER BY member_index").format(membersTable)
                        cursor.execute(statement, (datasetID,))
                        memberRows = cursor.fetchall()

                        statement = sql.SQL("SELECT product_name, matrix_revision_id FROM {} WHERE dataset_id = %s ORDER BY product_name").format(productsTable)
                        cursor.execute(statement, (datasetID,))
                        productRows = cursor.fetchall()

        if checkpoint is None:
            pglMessages.warning(f"Unknown or hidden dataset checkpoint: {datasetID}", level=0)
            return None

        familyUUID, name, checkpointName, description, status, contractVersion, document = checkpoint

        if status != "complete":
            raise ValueError("Only completed dataset checkpoints can be loaded.")

        if contractVersion != "pgl.dataset.v1":
            raise ValueError(f"Unsupported dataset contract: {contractVersion!r}")

        if not isinstance(document, dict):
            raise ValueError("Dataset metadata must contain a serialized JSON object.")

        dataset = pglSerialize.fromJSON(json.dumps(document, allow_nan=False), strict=True)

        if type(dataset) is not pglDataset:
            raise TypeError("Saved dataset metadata did not restore as pglDataset.")

        if dataset.name != name or dataset.uuid != str(familyUUID):
            raise ValueError("Dataset metadata disagrees with its database identity or name.")

        if dataset.data:
            raise ValueError("Dataset metadata unexpectedly embedded products.")

        if [row[0] for row in memberRows] != list(range(len(memberRows))):
            raise ValueError("Dataset membership order is invalid.")

        records = [{"sessionID": sessionID, "entryName": entryName, "labels": labels} for _, sessionID, entryName, labels in memberRows]
        dataset._members = dataset._validateMembers(records)
        dataset.datasetID = datasetID
        dataset.checkpointName = checkpointName
        dataset.description = description

        cls._bindDatasetStorage(dataset, resolvedSettings, postgres)
        identityJSON = cls._datasetDatabaseIdentityJSON(dataset._datasetDatabaseIdentity)

        dataset.data = {productName: pglDataMatrixReference(str(revisionID), identityJSON) for productName, revisionID in productRows}
        cls._validateDatasetProducts(dataset)

        dataset._datasetLoadOptions = {
            "sslRootCert": sslRootCert,
            "storageOptionsByBackend": storageOptionsByBackend,
            "cacheDirectory": cacheDirectory,
            "directLocalPolicy": directLocalPolicy,
        }

        dataset._datasetBaseline = {
            "version": 1,
            "datasetID": datasetID,
            "familyUUID": dataset.uuid,
            "databaseIdentity": deepcopy(dataset._datasetDatabaseIdentity),
            "historyHashes": cls._datasetHistoryHashes(dataset),
        }

        pglMessages.message(f"Loaded dataset checkpoint {datasetID} with {len(records)} members and {len(productRows)} unopened products; sessions have not been loaded.")
        return dataset

    # +++++++++++++++++++++++++++++ saveDataset +++++++++++++++++++++++++++++
    @classmethod
    def saveDataset(cls, dataset, *, settings=None, settingsName=None, backendID=None, storageOptions=None, sslRootCert=None):
        """Publish a dataset checkpoint with named scientific products.

        Called by pglDataset.save().

        Existing unopened references reuse registered matrix revisions.
        Attached working matrices publish fresh standalone Zarr bundles.
        Cached working sessions are not saved by this operation.

        Resolves server-local destinations through the caller's local/SSH
        access context. Catalog roots and relative object keys remain unchanged;
        caller-specific SSH access URLs are not registered.

        Acquires the session dependency lock before dataset publication.
        Membership and required ancestors must be complete and active.

        Owns its SQL transaction and, when writing matrices, one destination
        access context. The access context closes before SQL commit. Working
        matrices' source readers are borrowed and remain caller-owned.

        Updates dataset identity, annotations, binding, and history baseline
        only after successful commit and connection cleanup.

        Returns:
            New checkpoint ID, or None for unavailable configuration, invalid
            creator configuration, or connection cancellation.
            Other errors propagate.

        Payload writes cannot be rolled back by PostgreSQL. Failed publication
        may leave orphaned matrix bundles but not a committed completed dataset.

        Does not rebuild products or rerun analyses.
        Concurrent working-object mutation is unsupported.
        """
        from contextlib import ExitStack

        from psycopg.types.json import Jsonb

        from .pglData import pglDataMatrix, pglDataMatrixReference

        metadata = cls._prepareDatasetMetadata(dataset)
        members = dataset._validateMembers(dataset.members)
        products = cls._validateDatasetProducts(dataset)
        checkpointName = cls._datasetAnnotation(dataset.checkpointName, "checkpointName")
        description = cls._datasetAnnotation(dataset.description, "description")
        historyHashes = cls._datasetHistoryHashes(dataset)

        if backendID is not None and (type(backendID) is not int or not 0 <= backendID <= 9223372036854775807):
            raise ValueError("backendID must be a nonnegative PostgreSQL bigint.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        # Reject structural errors before allocating permanent payload paths.
        for product in products.values():
            if isinstance(product, pglDataMatrix):
                pglStorage.requireSafeSave(product)
                pglStorage.validateDataMatrixEncoding(product.requireData())

        resolvedSettings, postgres = cls._resolveDatasetStorage(dataset, settings=settings, settingsName=settingsName)

        if postgres is None:
            return None

        username = resolvedSettings.username

        if not isinstance(username, str) or not username.strip() or "\x00" in username:
            pglMessages.warning("Cannot save dataset: set a valid username in PGL settings.", level=0)
            return None

        databaseIdentity = deepcopy(cls._checkpointDatabaseIdentity(postgres))
        identityJSON = cls._datasetDatabaseIdentityJSON(databaseIdentity)

        for name, product in products.items():
            if isinstance(product, pglDataMatrixReference) and product.databaseIdentityJSON != identityJSON:
                raise ValueError(f"Product {name!r} references a different configured database.")

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        datasetsTable = sql.Identifier(postgres.databaseSchema, "dataset")
        membersTable = sql.Identifier(postgres.databaseSchema, "dataset_members")
        productsTable = sql.Identifier(postgres.databaseSchema, "dataset_products")
        revisionsTable = sql.Identifier(postgres.databaseSchema, "matrix_revisions")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        hasWorkingProducts = any(isinstance(product, pglDataMatrix) for product in products.values())

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            # ExitStack closes destination access before the SQL transaction
            # commits. No destination context is needed for reference-only saves.
            with connection.transaction(), ExitStack() as resources:
                connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                connection.execute(sql.SQL("LOCK TABLE {} IN SHARE MODE").format(sessionsTable))

                createdBy = cls.getOrCreateUser(connection, postgres, username)
                destinationFS = None
                destinationRoot = None

                if hasWorkingProducts:
                    if backendID is None:
                        backendID = cls._getConfiguredStorageBackend(connection, postgres, storageOptions=storageOptions)

                    statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backendsTable)
                    backend = connection.execute(statement, (backendID,)).fetchone()

                    if backend is None or not backend[1]:
                        raise ValueError("Matrix destination backend is missing or inactive.")

                    access = resources.enter_context(pglDatabaseAccess(postgres, {backendID: storageOptions or {}}))
                    destinationFS, destinationRoot = access.resolve(backendID, backend[0])

                statement = sql.SQL("""
                    INSERT INTO {} (
                        family_uuid, name, checkpoint_name, description,
                        metadata, created_by
                    )
                    VALUES (%s::uuid, %s, %s, %s, %s, %s)
                    RETURNING dataset_id
                """).format(datasetsTable)

                datasetID = connection.execute(statement, (dataset.uuid, dataset.name, checkpointName, description, Jsonb(metadata), createdBy)).fetchone()[0]

                statement = sql.SQL("""
                    INSERT INTO {} (
                        dataset_id, session_id, member_index, entry_name, labels
                    )
                    VALUES (%s, %s, %s, %s, %s)
                """).format(membersTable)

                for index, member in enumerate(members):
                    connection.execute(statement, (datasetID, member["sessionID"], index, member["entryName"], Jsonb(member["labels"])))

                problems = cls._getDatasetSourceProblems(connection, postgres, datasetID)

                if problems:
                    problem = problems[0]
                    raise ValueError(f"Cannot save dataset: member session {problem['memberSessionID']} requires active completed source checkpoint {problem['sourceSessionID']}.")

                for productName, product in products.items():
                    if isinstance(product, pglDataMatrixReference):
                        revisionID = product.matrixRevisionID

                        # Preserve existing reference-reuse policy. This is a
                        # catalog check, not a fresh payload availability scan.
                        statement = sql.SQL("SELECT contract_version FROM {} WHERE matrix_revision_id = %s::uuid").format(revisionsTable)
                        registered = connection.execute(statement, (revisionID,)).fetchone()

                        if registered != ("pgl.matrix.zarr.v1",):
                            raise ValueError(f"Product {productName!r} references an unknown or unsupported matrix revision.")

                    else:
                        productRoot = posixpath.join(destinationRoot, "matrices")
                        savedURL = pglStorage.saveDataMatrix(product, productRoot, filesystem=destinationFS)
                        savedPath = destinationFS._strip_protocol(savedURL)

                        with destinationFS.open(posixpath.join(savedPath, "matrix.json"), "rb") as source:
                            descriptorBytes = source.read(16 * 1024 * 1024 + 1)

                        if len(descriptorBytes) > 16 * 1024 * 1024:
                            raise ValueError("Published matrix descriptor exceeds the size limit.")

                        descriptor = json.loads(descriptorBytes)
                        revisionID = descriptor["matrixRevisionID"]
                        objectKey = f"matrices/matrix-{revisionID}"
                        expectedPath = posixpath.join(destinationRoot, objectKey)

                        if posixpath.normpath(savedPath) != posixpath.normpath(expectedPath):
                            raise ValueError("Published matrix path differs from its expected backend-relative location.")

                        # Store only the backend ID and relative object key.
                        # The filesystem's caller-specific access path is not
                        # used as a new backend registration.
                        statement = sql.SQL("""
                            INSERT INTO {} (
                                matrix_revision_id, backend_id, object_key,
                                contract_version, descriptor, created_by
                            )
                            VALUES (%s::uuid, %s, %s, %s, %s, %s)
                        """).format(revisionsTable)

                        connection.execute(statement, (revisionID, backendID, objectKey, "pgl.matrix.zarr.v1", Jsonb(descriptor), createdBy))

                    statement = sql.SQL("INSERT INTO {} (dataset_id, product_name, matrix_revision_id) VALUES (%s, %s, %s::uuid)").format(productsTable)
                    connection.execute(statement, (datasetID, productName, revisionID))

                cls._requireSafeDatasetCheckpoint(dataset)
                cls._requireDatasetProductsUnchanged(dataset, products)

                if dataset.members != members or cls._prepareDatasetMetadata(dataset) != metadata:
                    raise ValueError("Dataset changed while saving. No checkpoint was published.")

                if cls._datasetAnnotation(dataset.checkpointName, "checkpointName") != checkpointName or cls._datasetAnnotation(dataset.description, "description") != description:
                    raise ValueError("Dataset annotations changed while saving. No checkpoint was published.")

                statement = sql.SQL("UPDATE {} SET status = 'complete', completed_at = statement_timestamp() WHERE dataset_id = %s RETURNING dataset_id").format(datasetsTable)

                if connection.execute(statement, (datasetID,)).fetchone() != (datasetID,):
                    raise RuntimeError("Dataset checkpoint completion failed.")

                pendingBaseline = {
                    "version": 1,
                    "datasetID": datasetID,
                    "familyUUID": dataset.uuid,
                    "databaseIdentity": databaseIdentity,
                    "historyHashes": historyHashes,
                }

        # Publication and connection cleanup succeeded. Keep editable working
        # matrices attached; saving does not relabel them as immutable references.
        dataset.datasetID = datasetID
        dataset.checkpointName = checkpointName
        dataset.description = description
        dataset._datasetDatabaseIdentity = databaseIdentity
        dataset._datasetSettings = resolvedSettings
        dataset._datasetSettingsName = None
        dataset._datasetBaseline = pendingBaseline

        pglMessages.message(f"Dataset checkpoint {datasetID} saved with {len(members)} members and {len(products)} products in {postgres.databaseName}.{postgres.databaseSchema}.")
        return datasetID


    # -------------------------------------------------------------------------
    # Datasets: members and scientific products
    # -------------------------------------------------------------------------
    # Resolve exact member checkpoints, inspect membership, or open and construct dataset products.

    # +++++++++++++++++++++++++++ getDatasetSession +++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def getDatasetSession(cls, dataset, sessionID, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy=None):
        """Resolve an exact member checkpoint and cache its working object.

        Explicit database overrides are checked even for a cache hit.
        Cached session mutation does not alter the membership record.
        """
        dataset._findMember(sessionID=sessionID)
        resolvedSettings, postgres = cls._resolveDatasetStorage(dataset, settings=settings, settingsName=settingsName)

        if postgres is None:
            return None

        if sessionID in dataset._sessionCache:
            return dataset._sessionCache[sessionID]

        options = dict(dataset._datasetLoadOptions)

        if storageOptionsByBackend is not None:
            options["storageOptionsByBackend"] = storageOptionsByBackend

        if sslRootCert is not None:
            options["sslRootCert"] = sslRootCert

        if cacheDirectory is not None:
            options["cacheDirectory"] = cacheDirectory

        if directLocalPolicy is not None:
            options["directLocalPolicy"] = directLocalPolicy

        session = cls.loadSession(sessionID, settings=resolvedSettings, **options)

        if session is None:
            return None

        cls._bindDatasetStorage(dataset, resolvedSettings, postgres)
        dataset._sessionCache[sessionID] = session
        return session

    # +++++++++++++++++++++++++++++ getDatasetData ++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    # Direct callers in this module: getDatasetBehaviorData.
    # +++++++++++++++++++++++++++++ getDatasetData ++++++++++++++++++++++++++
    @classmethod
    def getDatasetData(cls, dataset, productName, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None):
        """Resolve an attached dataset product and cache its working matrix.

        Called by pglDataset.getData() and getDatasetBehaviorData().

        Existing working matrix:
            Validate the dataset's configured database binding, then return
            the same attached object without reopening its payload.

        Saved matrix reference:
            Validate reference/database correspondence.
            Query the registered revision and backend in a read-only snapshot.
            Close SQL before opening the matrix through _openRegisteredMatrix().
            Resolve server-local roots using the caller's local/SSH context.
            Replace the reference only after successful loading and checks.

        Returns:
            The attached pglDataMatrix, or None for unavailable configuration
            or connection cancellation.

        Missing products raise KeyError. Invalid bindings, inactive backends,
        unsupported products, and catalog/payload errors propagate.

        Reads catalog and matrix/Zarr metadata. Scientific arrays remain lazy.
        Does not extract missing products, evaluate construction freshness,
        or publish records or files.

        An opened matrix owns its payload-access context through its xarray
        close callback. Call matrix.close() only after dependent readers are
        finished. A closed matrix remains attached until explicitly replaced
        or removed; this accessor does not automatically reopen it.

        Concurrent mutation during resolution is unsupported. Detectable
        replacement of the product prevents installing the loaded matrix.
        """
        from psycopg.rows import tuple_row

        from .pglData import pglDataMatrix, pglDataMatrixReference

        if not isinstance(productName, str) or not productName.strip():
            raise ValueError("productName must be a nonblank string.")

        if productName not in dataset.data:
            raise KeyError(f"No attached dataset product: {productName!r}")

        resolvedSettings, postgres = cls._resolveDatasetStorage(dataset, settings=settings, settingsName=settingsName)

        if postgres is None:
            return None

        product = dataset.data[productName]

        if isinstance(product, pglDataMatrix):
            return product

        if not isinstance(product, pglDataMatrixReference):
            raise TypeError(f"Unsupported dataset product: {productName!r}")

        identityJSON = cls._datasetDatabaseIdentityJSON(cls._checkpointDatabaseIdentity(postgres))

        if product.databaseIdentityJSON != identityJSON:
            raise ValueError("Matrix reference belongs to a different configured database.")

        options = dataset._datasetLoadOptions

        if sslRootCert is None:
            sslRootCert = options.get("sslRootCert")

        if storageOptionsByBackend is None:
            storageOptionsByBackend = options.get("storageOptionsByBackend")

        revisions = sql.Identifier(postgres.databaseSchema, "matrix_revisions")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    statement = sql.SQL("""
                        SELECT revision.backend_id,
                               revision.object_key,
                               revision.contract_version,
                               revision.descriptor,
                               backend.url_prefix,
                               backend.is_active
                        FROM {} AS revision
                        JOIN {} AS backend
                          ON backend.backend_id = revision.backend_id
                        WHERE revision.matrix_revision_id = %s::uuid
                    """).format(revisions, backends)

                    cursor.execute(statement, (product.matrixRevisionID,))
                    record = cursor.fetchone()

        if record is None:
            raise ValueError(f"Unknown registered matrix revision: {product.matrixRevisionID}")

        backendID, objectKey, contractVersion, descriptor, registeredRoot, isActive = record

        if not isActive:
            raise ValueError(f"Matrix backend {backendID} is inactive.")

        # SQL and its tunnel have closed. The matrix opener creates separate
        # payload access and retains it for subsequent lazy scientific reads.
        matrix = cls._openRegisteredMatrix(postgres, revisionID=product.matrixRevisionID, backendID=backendID, objectKey=objectKey, contractVersion=contractVersion, descriptor=descriptor, registeredRoot=registeredRoot, storageOptionsByBackend=storageOptionsByBackend)

        try:
            if dataset.data.get(productName) is not product:
                raise ValueError("Dataset product changed while it was being opened.")

            cls._bindDatasetStorage(dataset, resolvedSettings, postgres)
            dataset.data[productName] = matrix

        except BaseException:
            try:
                matrix.close()
            except Exception:
                pass

            raise

        return matrix

    # +++++++++++++++++++++++++ getDatasetBehaviorData ++++++++++++++++++++++
    @classmethod
    def getDatasetBehaviorData(cls, dataset, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy=None):
        """Return or construct the dataset-owned comprehensive behavior matrix.

        Called by pglDataset.getBehaviorData() and behavioral table access.

        Existing product:
            Resolve through getDatasetData(), validate its behavioral schema,
            and return it. Never silently replace an invalid product or
            automatically rebuild after membership/code changes.

        Missing product:
            Load exact active member checkpoints afresh in membership order.
            Bypass the editable session cache and saved matrix attachments.
            Extract behavioral sources and attach the combined matrix only
            after successful validation.

        Each privately loaded member's unused matrix attachments and checkpoint
        filesystem are closed after extraction. Cached/user-owned sessions are
        not touched. This relies on extractSession() returning scientific
        contents independent of the member's deferred filesystem.

        Reads behavioral payloads; general session loading may also inspect
        matrix/MNE metadata. Does not publish database records or payloads.

        Initial construction retains extracted matrices until assembly and is
        not bounded-memory streaming. Concurrent dataset mutation is unsupported;
        detectable membership, identity, or attachment changes prevent attachment.

        Returns a working pglDataMatrix, or None when configuration or existing
        product resolution returns no result. Other failures propagate.
        """
        from .pglBehavior import pglBehavior

        productName = "behavior"

        if productName in dataset.data:
            matrix = cls.getDatasetData(dataset, productName, settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert)

            if matrix is None:
                return None

            pglBehavior._requireBehaviorData(matrix)
            return matrix

        cls._requireSafeDatasetCheckpoint(dataset)
        members = dataset._validateMembers(dataset.members)

        if not members:
            raise ValueError("Cannot construct behavioral data from an empty dataset.")

        resolvedSettings, postgres = cls._resolveDatasetStorage(dataset, settings=settings, settingsName=settingsName)

        if postgres is None:
            return None

        databaseIdentity = deepcopy(cls._checkpointDatabaseIdentity(postgres))
        originalBinding = deepcopy(dataset._datasetDatabaseIdentity)
        originalDatasetID = dataset.datasetID
        options = dict(dataset._datasetLoadOptions)

        if storageOptionsByBackend is not None:
            options["storageOptionsByBackend"] = storageOptionsByBackend

        if sslRootCert is not None:
            options["sslRootCert"] = sslRootCert

        if cacheDirectory is not None:
            options["cacheDirectory"] = cacheDirectory

        if directLocalPolicy is not None:
            options["directLocalPolicy"] = directLocalPolicy

        # Historical dataset inspection does not authorize construction from
        # deleted source checkpoints.
        options["includeDeleted"] = False

        matrices = []
        combined = None

        try:
            for memberIndex, member in enumerate(members):
                sessionID = member["sessionID"]
                pglMessages.message(f"Constructing dataset behavior: member {memberIndex + 1}/{len(members)}, checkpoint {sessionID}...")

                # Load a private authoritative source, never an editable cached
                # working session.
                session = cls.loadSession(sessionID, settings=resolvedSettings, **options)

                if session is None:
                    raise RuntimeError(f"Cannot construct dataset behavior: member checkpoint {sessionID} could not be loaded.")

                try:
                    if session.sessionID != sessionID:
                        raise ValueError("Loaded session identity differs from the requested member.")

                    baseline = session._checkpointBaseline

                    if baseline.get("databaseIdentity") != databaseIdentity:
                        raise ValueError("Loaded member belongs to a different configured database.")

                    if not session.runs:
                        raise ValueError(f"Member checkpoint {sessionID} has no behavioral runs. Nonbehavioral members are not silently skipped.")

                    # Extract saved behavioral sources rather than reusing
                    # attachments that may describe another working state.
                    matrix = pglBehavior.extractSession(session, name=productName)
                    matrices.append(matrix)

                finally:
                    # These attachments belong to the private loaded session,
                    # not to the extracted matrices retained for assembly.
                    try:
                        for attachment in session.dataMatrices.values():
                            try:
                                attachment.close()
                            except Exception as error:
                                pglMessages.warning(f"Could not close an unused member matrix ({type(error).__name__}).", level=1)
                    finally:
                        # Do not resolve any MNE field merely to clean up.
                        # Behavioral extraction does not need MNE readers.
                        if isinstance(session.filesystem, pglCheckpointFilesystem):
                            try:
                                session.filesystem.close()
                            except Exception as error:
                                pglMessages.warning(f"Could not close a private member checkpoint filesystem ({type(error).__name__}).", level=1)

            combined = pglBehavior.combineSessionData(matrices, members=members, databaseIdentity=databaseIdentity, name=productName)

            # Supplement existing extraction/assembly provenance.
            combined.provenance = {
                **deepcopy(combined.provenance),
                "orchestration": "pglDatabase.getDatasetBehaviorData",
                "sourceState": "freshlyLoadedCheckpointBehavior",
                "datasetCheckpointIDAtConstruction": originalDatasetID,
                "sourceReadPolicy": "Exact member checkpoints loaded afresh; working session cache and saved session matrix attachments not reused",
                "membershipSnapshot": deepcopy(members),
            }

            cls._requireSafeDatasetCheckpoint(dataset)

            if dataset.members != members:
                raise ValueError("Dataset membership changed during behavioral construction; no product was attached.")

            if dataset.datasetID != originalDatasetID or dataset._datasetDatabaseIdentity != originalBinding:
                raise ValueError("Dataset identity/binding changed during behavioral construction; no product was attached.")

            if productName in dataset.data:
                raise ValueError("A behavior product was attached during construction; it was not replaced.")

            pglBehavior._requireBehaviorData(combined)
            pglStorage.validateDataMatrixEncoding(combined.requireData())

            cls._bindDatasetStorage(dataset, resolvedSettings, postgres)
            dataset.data = {**dataset.data, productName: combined}

            pglMessages.message(f"Constructed dataset behavior from {len(members)} member checkpoints. Attached as data['behavior']; not saved.")
            return combined

        except BaseException:
            if combined is not None:
                try:
                    combined.close()
                except Exception:
                    pass

            raise

        finally:
            # The assembler creates independent arrays, so its result does
            # not retain the temporary member matrices.
            for matrix in matrices:
                try:
                    matrix.close()
                except Exception as error:
                    pglMessages.warning(f"Could not close a temporary behavioral matrix ({type(error).__name__}).", level=1)

    # +++++++++++++++++++++++ getDatasetMemberSummaries ++++++++++++++++++++++
    @classmethod
    def getDatasetMemberSummaries(cls, dataset, *, sessionIDs=None, settings=None, settingsName=None, sslRootCert=None, includeBehaviorLabels=False):
        """Read member summaries without constructing sessions or matrices.

        Called by dataset membership validation and printDataset().

        Default:
            Read database metadata only. Acquisition time falls back to the
            root checkpoint. Labels use checkpoint metadata, then root metadata.

        includeBehaviorLabels=True:
            Read verified root experiment-settings files for acquisition
            sessionName values and fallback experiment/subject labels.
            Resolve payload access using the dataset's caller configuration
            and retained runtime backend options.

        SQL closes before payload reads. One short-lived access context reuses
        filesystem connections and closes on success or failure.

        Historical inspection includes deleted source rows and their recorded
        settings locations. Reading them does not restore lifecycle state.

        Returns:
            Mapping from session ID to summary.
            Empty mapping for an empty request.
            None for unavailable configuration or connection cancellation.

        Missing sessions are absent from the mapping. Other errors propagate.
        Does not publish, populate the session cache, or open scientific arrays.
        Successful resolution binds the dataset's runtime configuration.
        """
        from psycopg.rows import tuple_row

        if type(includeBehaviorLabels) is not bool:
            raise ValueError("includeBehaviorLabels must be a boolean.")

        requested = [record["sessionID"] for record in dataset.members] if sessionIDs is None else list(sessionIDs)

        for sessionID in requested:
            dataset._validateSessionID(sessionID)

        requested = sorted(set(requested))

        if not requested:
            return {}

        resolvedSettings, postgres = cls._resolveDatasetStorage(dataset, settings=settings, settingsName=settingsName)

        if postgres is None:
            return None

        if sslRootCert is None:
            sslRootCert = dataset._datasetLoadOptions.get("sslRootCert")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        settingsByRoot = {}
        identities = {}
        locationsByFile = {}

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    statement = sql.SQL("""
                        SELECT checkpoint.session_id,
                               checkpoint.status,
                               checkpoint.deleted_at,
                               checkpoint.checkpoint_name,
                               checkpoint.created_at,
                               COALESCE(checkpoint.acquired_at, root.acquired_at),
                               checkpoint.meta,
                               root.meta,
                               checkpoint.root_session_id
                        FROM {sessions} AS checkpoint
                        LEFT JOIN {sessions} AS root
                          ON root.session_id = checkpoint.root_session_id
                        WHERE checkpoint.session_id = ANY(%s::bigint[])
                        ORDER BY checkpoint.session_id
                    """).format(sessions=sessions)

                    cursor.execute(statement, (requested,))
                    rows = cursor.fetchall()
                    rootIDs = sorted({row[8] for row in rows if row[8] is not None})

                    if includeBehaviorLabels and rootIDs:
                        # Historical inspection includes deleted links/content.
                        statement = sql.SQL("""
                            SELECT link.session_id, link.name, content.file_id,
                                   content.sha256, content.size_bytes
                            FROM {} AS link
                            JOIN {} AS content ON content.file_id = link.file_id
                            WHERE link.session_id = ANY(%s::bigint[])
                              AND link.name ~ '^runs/[0-9]+/experimentSettings[.]json$'
                            ORDER BY link.session_id, link.name
                        """).format(links, files)

                        cursor.execute(statement, (rootIDs,))

                        for rootID, name, fileID, digest, size in cursor.fetchall():
                            settingsByRoot.setdefault(rootID, []).append((name, fileID))
                            identities[fileID] = {"sha256": digest, "size_bytes": size}

                    if identities:
                        statement = sql.SQL("""
                            SELECT location.file_id, backend.backend_id,
                                   backend.url_prefix, location.object_key
                            FROM {} AS location
                            JOIN {} AS backend ON backend.backend_id = location.backend_id
                            WHERE location.file_id = ANY(%s::bigint[])
                              AND location.status = 'present'
                              AND location.verified_at IS NOT NULL
                              AND backend.is_active
                            ORDER BY location.file_id, location.priority,
                                     backend.backend_id, location.object_key
                        """).format(locations, backends)

                        cursor.execute(statement, (sorted(identities),))

                        for fileID, backendID, root, key in cursor.fetchall():
                            locationsByFile.setdefault(fileID, []).append({
                                "backend_id": backendID,
                                "url_prefix": root,
                                "object_key": key,
                            })

        settingsByFile = {}

        # Default membership validation remains SQL-only. No access context
        # is needed unless verified behavioral settings are actually requested.
        if identities:
            storageOptionsByBackend = dataset._datasetLoadOptions.get("storageOptionsByBackend")

            with pglDatabaseAccess(postgres, storageOptionsByBackend) as access:
                for fileID, identity in identities.items():
                    candidates = locationsByFile.get(fileID, [])

                    if not candidates:
                        raise FileNotFoundError(f"Experiment-settings file {fileID} has no eligible registered location.")

                    settingsByFile[fileID] = cls._readPrintExperimentSettings(identity, candidates, accessContext=access)

        labelsByRoot = {}

        # ++++++++++++++++++++++++++++ addLabel +++++++++++++++++++++++++++++
        def addLabel(values, value, fieldName):
            """Collect distinct nonblank labels without changing saved text."""
            if value is None:
                return

            if not isinstance(value, str):
                raise ValueError(f"Saved {fieldName} must be a string or None.")

            if value.strip() and value not in values:
                values.append(value)

        for rootID, records in settingsByRoot.items():
            labels = {"experimentNames": [], "subjectIDs": [], "sessionNames": []}

            for name, fileID in sorted(records, key=lambda item: int(item[0].split("/")[1])):
                savedSettings = settingsByFile[fileID]
                addLabel(labels["experimentNames"], savedSettings.get("experimentName"), "experimentName")
                addLabel(labels["subjectIDs"], savedSettings.get("subjectID"), "subjectID")
                addLabel(labels["sessionNames"], savedSettings.get("sessionName"), "sessionName")

            labelsByRoot[rootID] = labels

        summaries = {}

        for sessionID, status, deletedAt, checkpointName, createdAt, acquiredAt, checkpointMeta, rootMeta, rootID in rows:
            labelMetadata = None

            for metadata in (checkpointMeta, rootMeta):
                if isinstance(metadata, dict) and "pgl_session_labels" in metadata:
                    labelMetadata = cls._validateSessionLabelMetadata(metadata["pgl_session_labels"])
                    break

            rootLabels = labelsByRoot.get(rootID, {"experimentNames": [], "subjectIDs": [], "sessionNames": []})

            if labelMetadata is not None:
                experimentNames = sorted({record["experimentName"] for record in labelMetadata["labels"] if record["experimentName"] is not None})
                subjectIDs = sorted({record["subjectID"] for record in labelMetadata["labels"] if record["subjectID"] is not None})
            else:
                experimentNames = list(rootLabels["experimentNames"])
                subjectIDs = sorted(rootLabels["subjectIDs"])

            summaries[sessionID] = {
                "sessionID": sessionID,
                "status": status,
                "deletedAt": deletedAt,
                "checkpointName": checkpointName,
                "createdAt": createdAt,
                "acquiredAt": acquiredAt,
                "experimentNames": experimentNames,
                "subjectIDs": subjectIDs,
                "sessionNames": list(rootLabels["sessionNames"]),
            }

        cls._bindDatasetStorage(dataset, resolvedSettings, postgres)
        return summaries


    # -------------------------------------------------------------------------
    # Datasets: lifecycle
    # -------------------------------------------------------------------------
    # User entry points for dataset soft deletion and restoration of source protection.

    # +++++++++++++++++++++++++++++ deleteDataset +++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def deleteDataset(cls, datasetID, *, settings=None, settingsName=None, sslRootCert=None):
        """Soft-delete one completed dataset checkpoint.

        Releases this checkpoint's session protection. Other active dataset
        versions continue protecting their own members and ancestors.

        Preserves membership, history, scientific products, and sessions.
        Does not inspect, delete, or rewrite payloads.

        Returns True for a committed change, False for an unchanged,
        unknown, or blocked operation, and None for invalid arguments,
        unavailable configuration/connection, or database failure.
        """
        return cls._changeDatasetDeletionState(datasetID, deleted=True, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert)

    # ++++++++++++++++++++++++++++ undeleteDataset ++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def undeleteDataset(cls, datasetID, *, settings=None, settingsName=None, sslRootCert=None):
        """Restore one completed dataset checkpoint.

        Every member and required ancestor must be complete and active.
        Refusal warns and leaves the dataset deleted.

        Does not restore sessions automatically or inspect payloads.

        Returns True for a committed change, False for an unchanged,
        unknown, or blocked operation, and None for invalid arguments,
        unavailable configuration/connection, or database failure.
        """
        return cls._changeDatasetDeletionState(datasetID, deleted=False, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert)


    # -------------------------------------------------------------------------
    # Database reporting
    # -------------------------------------------------------------------------
    # User-facing reports. Individual docstrings identify metadata-only reports and verified
    # settings reads.

    # +++++++++++++++++++++++++++++++++ print +++++++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @staticmethod
    def print(details=None, postgres=None, *, settings=None, settingsName=None, sessionID=None, summary=False, datasets=False, nRows=10, maxWidth=48, sslRootCert=None, includeDeleted=False, storageOptionsByBackend=None):
        """Select a session listing, dataset listing, or database report.

        Default:
            List visible sessions grouped by root experiment and subject,
            displaying actual parent-child relationships.

        datasets=True:
            List visible dataset checkpoints grouped by dataset name.
            Show checkpoint names, member counts, and scientific product names.
            Does not load datasets, sessions, or scientific payloads.

        summary=True:
            Show database table counts and session statuses.

        sessionID:
            Show one session checkpoint's metadata, subjects, and linked files.

        details:
            Show one table, an iterable of tables, or "all".

        datasets, summary, sessionID, and details are mutually exclusive.
        nRows limits table detail views only, not readable listings.

        includeDeleted includes soft-deleted checkpoints.
        storageOptionsByBackend applies to the session listing's verified
        acquisition-settings reads.

        Read-only; does not publish or modify records.
        """
        if type(summary) is not bool:
            raise ValueError("summary must be a boolean.")

        if type(datasets) is not bool:
            raise ValueError("datasets must be a boolean.")

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        if type(nRows) is not int or nRows <= 0:
            raise ValueError("nRows must be a positive integer.")

        if type(maxWidth) is not int or maxWidth < 8:
            raise ValueError("maxWidth must be an integer of at least 8.")

        if sum((details is not None, sessionID is not None, summary, datasets)) > 1:
            raise ValueError("Supply only one of details, sessionID, summary=True, or datasets=True.")

        if datasets:
            return pglDatabase.printDatasets(postgres=postgres, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeDeleted=includeDeleted)

        if sessionID is not None:
            return pglDatabase.printSession(sessionID, postgres=postgres, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeDeleted=includeDeleted)

        if details is not None or summary:
            return pglDatabase.printTables(details=details, postgres=postgres, settings=settings, settingsName=settingsName, nRows=nRows, maxWidth=maxWidth, sslRootCert=sslRootCert, includeDeleted=includeDeleted)

        return pglDatabase.printSessions(postgres=postgres, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeDeleted=includeDeleted, storageOptionsByBackend=storageOptionsByBackend)

    # +++++++++++++++++++++++++++++ printSessions +++++++++++++++++++++++++++
    @staticmethod
    def printSessions(postgres=None, *, settings=None, settingsName=None, sslRootCert=None, includeDeleted=False, storageOptionsByBackend=None):
        """Print sessions grouped by root experiment and subject.

        Called directly or by print() with its default selection.

        Parentless roots display acquisition sessionName values in brackets.
        Descendants display database checkpoint names in parentheses.
        Descriptions are omitted; actual parent-child relationships are shown.

        Reads catalog metadata in a consistent SQL snapshot. After SQL closes,
        reads verified root experiment-settings files through caller-resolved
        local/SSH access. One report-owned context reuses filesystem connections
        and closes before output is printed.

        Mirrored root labels supply grouping when present. Older behavioral
        roots fall back to archived settings. Descendants inherit root grouping.
        Acquisition settings are still read for sessionName, which is not
        included in mirrored label metadata.

        Does not reconstruct sessions, open MNE/scientific matrix payloads, or
        modify records. Returns False for unavailable configuration or
        connection cancellation; other errors propagate.
        """
        from psycopg.rows import tuple_row

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        postgres = pglDatabase._printPostgres(postgres, settings=settings, settingsName=settingsName)

        if postgres is None:
            return False

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        records = {}
        groupsByRoot = {}
        settingsByRoot = {}
        identities = {}
        locationsByFile = {}

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return False

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    cursor.execute("SELECT current_database(), transaction_timestamp()")
                    databaseName, snapshotTime = cursor.fetchone()

                    statement = sql.SQL("SELECT session_id, parent_session_id, root_session_id, origin, status, deleted_at, meta, checkpoint_name FROM {} WHERE (%s OR deleted_at IS NULL) ORDER BY session_id").format(sessionsTable)
                    cursor.execute(statement, (includeDeleted,))

                    for sessionID, parentID, rootID, origin, status, deletedAt, meta, checkpointName in cursor.fetchall():
                        records[sessionID] = {
                            "session_id": sessionID,
                            "parent_session_id": parentID,
                            "root_session_id": rootID,
                            "origin": origin,
                            "status": status,
                            "deleted_at": deletedAt,
                            "meta": meta,
                            "checkpoint_name": checkpointName,
                            "session_names": [],
                        }

                    rootIDs = {record["root_session_id"] for record in records.values()}
                    visibleRootIDs = []

                    for rootID in sorted(rootIDs):
                        record = records.get(rootID)

                        if record is None:
                            groupsByRoot[rootID] = {("Unknown", "Unknown")}
                            continue

                        visibleRootIDs.append(rootID)
                        meta = record["meta"] or {}

                        if "pgl_session_labels" in meta:
                            metadata = pglDatabase._validateSessionLabelMetadata(meta["pgl_session_labels"])
                            groupsByRoot[rootID] = {
                                (
                                    label["experimentName"] if label["experimentName"] is not None else "Unknown",
                                    label["subjectID"] if label["subjectID"] is not None else "Unknown",
                                )
                                for label in metadata["labels"]
                            }

                    # All visible behavioral roots need acquisition settings
                    # for sessionName, even when grouping labels are mirrored.
                    if visibleRootIDs:
                        statement = sql.SQL("""
                            SELECT link.session_id, link.name, content.file_id,
                                   content.sha256, content.size_bytes
                            FROM {} AS link
                            JOIN {} AS content ON content.file_id = link.file_id
                            WHERE link.session_id = ANY(%s::bigint[])
                              AND link.name ~ '^runs/[0-9]+/experimentSettings[.]json$'
                              AND (%s OR (link.deleted_at IS NULL AND content.deleted_at IS NULL))
                            ORDER BY link.session_id, link.name
                        """).format(linksTable, filesTable)
                        cursor.execute(statement, (visibleRootIDs, includeDeleted))

                        for rootID, name, fileID, digest, size in cursor.fetchall():
                            settingsByRoot.setdefault(rootID, []).append((name, fileID))
                            identities[fileID] = {"sha256": digest, "size_bytes": size}

                    if identities:
                        statement = sql.SQL("""
                            SELECT location.file_id, backend.backend_id,
                                   backend.url_prefix, location.object_key
                            FROM {} AS location
                            JOIN {} AS backend ON backend.backend_id = location.backend_id
                            WHERE location.file_id = ANY(%s::bigint[])
                              AND (%s OR location.deleted_at IS NULL)
                              AND location.status = 'present'
                              AND location.verified_at IS NOT NULL
                              AND backend.is_active
                            ORDER BY location.file_id, location.priority,
                                     backend.backend_id, location.object_key
                        """).format(locationsTable, backendsTable)
                        cursor.execute(statement, (sorted(identities), includeDeleted))

                        for fileID, backendID, root, key in cursor.fetchall():
                            locationsByFile.setdefault(fileID, []).append({"backend_id": backendID, "url_prefix": root, "object_key": key})

        # SQL is closed. No deferred reader escapes this report, so its access
        # context can be closed after collecting the verified settings.
        settingsByFile = {}

        with pglDatabaseAccess(postgres, storageOptionsByBackend) as access:
            for fileID, identity in identities.items():
                candidates = locationsByFile.get(fileID, [])

                if not candidates:
                    raise FileNotFoundError(f"Experiment-settings file {fileID} has no eligible registered location.")

                settingsByFile[fileID] = pglDatabase._readPrintExperimentSettings(identity, candidates, accessContext=access)

        for rootID in sorted(rootIDs):
            record = records.get(rootID)

            if record is None:
                continue

            sessionNames = []
            pairs = set()
            needsGrouping = rootID not in groupsByRoot
            runSettings = sorted(settingsByRoot.get(rootID, []), key=lambda item: int(item[0].split("/")[1]))

            for name, fileID in runSettings:
                savedSettings = settingsByFile[fileID]
                sessionName = savedSettings.get("sessionName")

                if sessionName is not None:
                    if not isinstance(sessionName, str):
                        raise ValueError(f"Saved sessionName must be a string or None: root {rootID}, {name}")

                    if sessionName.strip() and sessionName not in sessionNames:
                        sessionNames.append(sessionName)

                if needsGrouping:
                    experiment = pglDatabase._printGroupingValue(savedSettings.get("experimentName"), "experimentName")
                    subject = pglDatabase._printGroupingValue(savedSettings.get("subjectID"), "subjectID")
                    pairs.add((experiment, subject))

            record["session_names"] = sessionNames

            if needsGrouping:
                groupsByRoot[rootID] = pairs or {("Unknown", "Unknown")}

        grouped = {}

        for sessionID, record in records.items():
            for experiment, subject in groupsByRoot[record["root_session_id"]]:
                grouped.setdefault(experiment, {}).setdefault(subject, {})[sessionID] = record

        # +++++++++++++++++++++++++++ groupOrder ++++++++++++++++++++++++++++
        def groupOrder(value):
            """Sort labels case-insensitively, with Unknown last."""
            return (value == "Unknown", value.casefold(), value)

        output = []

        for experiment in sorted(grouped, key=groupOrder):
            output.append(pglDatabase._printValue(experiment))

            for subject in sorted(grouped[experiment], key=groupOrder):
                output.append("  " + pglDatabase._printValue(subject))
                output.extend(pglDatabase._sessionLineageLines(grouped[experiment][subject]))

            output.append("")

        pglMessages.printHeader("Sessions")
        pglMessages.message(f"Database: {databaseName}.{postgres.databaseSchema}", wrapText=False)
        pglMessages.message(f"Metadata snapshot: {snapshotTime}", wrapText=False)
        pglMessages.print("  Including deleted sessions." if includeDeleted else "  Deleted sessions excluded.")
        pglMessages.print()

        if not records:
            pglMessages.print("  (no visible sessions)")
        else:
            for line in output:
                pglMessages.print(line)

        pglMessages.printHeader()
        return True

    # +++++++++++++++++++++++++++++ printDatasets +++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    # Direct callers in this module: print.
    @staticmethod
    def printDatasets(postgres=None, *, settings=None, settingsName=None, sslRootCert=None, includeDeleted=False):
        """Print saved dataset checkpoints grouped by name.

        Shows:
            Dataset ID.
            Optional checkpoint name.
            Deleted/incomplete state markers.
            Saved member count.
            Scientific product names.

        Distinct families sharing a name are labeled by family UUID.
        No parent-child lineage is inferred between dataset checkpoints.
        Descriptions are omitted; names are not truncated.

        Counts describe recorded membership, not current source availability.
        Product names do not establish construction freshness.

        Reads database metadata in one consistent snapshot.
        Does not reconstruct datasets/sessions, open matrix payloads,
        or modify database records.

        Returns False for unavailable configuration/connection.
        Other errors propagate.
        """
        from psycopg.rows import tuple_row

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        postgres = pglDatabase._printPostgres(postgres, settings=settings, settingsName=settingsName)

        if postgres is None:
            return False

        datasetsTable = sql.Identifier(postgres.databaseSchema, "dataset")
        membersTable = sql.Identifier(postgres.databaseSchema, "dataset_members")
        productsTable = sql.Identifier(postgres.databaseSchema, "dataset_products")

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return False

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    cursor.execute("SELECT current_database(), transaction_timestamp()")
                    databaseName, snapshotTime = cursor.fetchone()

                    # Separate correlated queries avoid multiplying membership
                    # rows by product rows when calculating counts.
                    statement = sql.SQL("""
                        SELECT checkpoint.dataset_id,
                               checkpoint.family_uuid::text,
                               checkpoint.name,
                               checkpoint.checkpoint_name,
                               checkpoint.status,
                               checkpoint.deleted_at,
                               (
                                   SELECT count(*)
                                   FROM {members} AS membership
                                   WHERE membership.dataset_id = checkpoint.dataset_id
                               ) AS member_count,
                               ARRAY(
                                   SELECT product.product_name
                                   FROM {products} AS product
                                   WHERE product.dataset_id = checkpoint.dataset_id
                                   ORDER BY product.product_name
                               ) AS product_names
                        FROM {datasets} AS checkpoint
                        WHERE (%s OR checkpoint.deleted_at IS NULL)
                        ORDER BY checkpoint.dataset_id
                    """).format(datasets=datasetsTable, members=membersTable, products=productsTable)

                    cursor.execute(statement, (includeDeleted,))
                    rows = cursor.fetchall()

        grouped = {}

        for datasetID, familyUUID, name, checkpointName, status, deletedAt, memberCount, productNames in rows:
            record = {
                "datasetID": datasetID,
                "checkpointName": checkpointName,
                "status": status,
                "deletedAt": deletedAt,
                "memberCount": memberCount,
                "productNames": productNames,
            }
            grouped.setdefault(name, {}).setdefault(familyUUID, []).append(record)

        # +++++++++++++++++++++++++++++++ nameOrder +++++++++++++++++++++++++++++++
        # Local implementation within printDatasets.
        def nameOrder(value):
            return (value is None, "" if value is None else value.casefold(), "" if value is None else value)

        # +++++++++++++++++++++++++++++ checkpointLine ++++++++++++++++++++++++++++
        # Local implementation within printDatasets.
        def checkpointLine(record, prefix):
            label = f"{prefix}datasetID: {record['datasetID']}"

            if record["checkpointName"]:
                label += f" ({pglDatabase._printValue(record['checkpointName'])})"

            if record["deletedAt"] is not None:
                label += " [deleted]"

            if record["status"] != "complete":
                label += f" [{pglDatabase._printValue(record['status'])}]"

            products = ", ".join(pglDatabase._printValue(name) for name in record["productNames"]) or "(none)"
            return f"{label} | members: {record['memberCount']} | products: {products}"

        pglMessages.printHeader("Datasets")
        pglMessages.message(f"Database: {databaseName}.{postgres.databaseSchema}", wrapText=False)
        pglMessages.message(f"Metadata snapshot: {snapshotTime}", wrapText=False)
        pglMessages.print("  Including deleted datasets." if includeDeleted else "  Deleted datasets excluded.")
        pglMessages.print()

        if not rows:
            pglMessages.print("  (no visible datasets)")
        else:
            for name in sorted(grouped, key=nameOrder):
                pglMessages.print(pglDatabase._printValue(name))
                families = grouped[name]
                showFamilies = len(families) > 1

                # Order families by their first visible checkpoint ID.
                familyOrder = sorted(families, key=lambda family: min(record["datasetID"] for record in families[family]))

                for familyUUID in familyOrder:
                    if showFamilies:
                        pglMessages.print(f"  family: {familyUUID}")

                    prefix = "    " if showFamilies else "  "

                    for record in sorted(families[familyUUID], key=lambda record: record["datasetID"]):
                        pglMessages.print(checkpointLine(record, prefix))

                pglMessages.print()

        pglMessages.printHeader()
        return True

    # ++++++++++++++++++++++++++++++ printDataset +++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def printDataset(cls, dataset, *, verbose=False, settings=None, settingsName=None, sslRootCert=None):
        """Print membership labels and product references.

        Displays experiment, subject, and behavioral acquisition session names.
        Reads verified root experiment-settings files for session names.
        Does not construct member sessions or open scientific matrix arrays.

        Entry name is the dataset-local alias.
        Session name is the behavioral acquisition's sessionName.
        Checkpoint name is the saved checkpoint annotation.

        Multiple distinct labels are joined with ' + '; these are summaries,
        not an assertion that every experiment/subject/name combination exists.
        """
        from .pglData import pglDataMatrixReference

        members = dataset._validateMembers(dataset.members)
        products = cls._validateDatasetProducts(dataset)
        summaries = cls.getDatasetMemberSummaries(dataset, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeBehaviorLabels=True)

        if summaries is None:
            pglMessages.warning("Cannot print dataset members: database configuration or connection is unavailable.", level=0)
            return False

        pglMessages.printHeader(f"Dataset: {cls._printValue(dataset.name)}")
        pglMessages.print(f"  Source checkpoint: {dataset.datasetID if dataset.datasetID is not None else 'Unsaved'}")
        pglMessages.print(f"  Working members: {len(members)}")

        if dataset.checkpointName:
            pglMessages.print(f"  Checkpoint name: {cls._printValue(dataset.checkpointName)}")

        if dataset.description:
            pglMessages.print(f"  Description: {cls._printValue(dataset.description)}")

        pglMessages.print("  Acquisition dates below are not checkpoint creation dates.")
        pglMessages.print()

        rows = []

        for member in members:
            sessionID = member["sessionID"]
            summary = summaries.get(sessionID)

            if summary is None:
                experimentLabel = "Unknown"
                subjectLabel = "Unknown"
                sessionLabel = ""
                acquiredLabel = "Unknown"
                checkpointLabel = "Unavailable"
                state = "missing"
            else:
                experimentLabel = " + ".join(summary["experimentNames"]) or "Unknown"
                subjectLabel = " + ".join(summary["subjectIDs"]) or "Unknown"
                sessionLabel = " + ".join(summary["sessionNames"])
                acquiredAt = summary["acquiredAt"]
                acquiredLabel = acquiredAt.isoformat() if acquiredAt is not None else "Unknown"
                checkpointLabel = summary["checkpointName"] or ""
                state = "deleted" if summary["deletedAt"] is not None else summary["status"]

            rows.append((
                experimentLabel,
                subjectLabel,
                sessionLabel,
                member["entryName"] or "",
                acquiredLabel,
                sessionID,
                checkpointLabel,
                state,
            ))

        # Preserve the existing subject-first display order.
        # This does not change authoritative dataset membership order.
        rows.sort(key=lambda row: (row[1] == "Unknown", row[1], row[5]))
        cls._printTable(("Experiment", "Subject", "Session name", "Entry name", "Acquired", "Session ID", "Checkpoint name", "Source state"), rows, maxWidth=None)

        if products:
            pglMessages.print()
            pglMessages.printHeader("Scientific products", fillChar="-")
            productRows = []

            for name, product in sorted(products.items()):
                state = "Saved reference; unopened" if isinstance(product, pglDataMatrixReference) else "Working matrix"
                revisionID = product.matrixRevisionID or "Unpublished"
                productRows.append((name, state, revisionID))

            cls._printTable(("Product", "Attachment state", "Source revision"), productRows, maxWidth=None)
            pglMessages.print("  Attachment state does not yet report construction freshness.")

        if verbose:
            for member in members:
                if member["labels"]:
                    pglMessages.print(f"  Session {member['sessionID']} labels: {cls._printValue(member['labels'])}")

            parameterNames = sorted(str(key) for key in dataset.derivedParameters)
            pglMessages.print(f"  Derived parameter keys: {', '.join(parameterNames) or '(none)'}")
            pglMessages.print(f"  Recorded actions: {len(dataset.actionHistory)}")

        pglMessages.printHeader()
        return True

    # ++++++++++++++++++++++++++++++ printTables ++++++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    # Direct callers in this module: print.
    @staticmethod
    def printTables(details=None, postgres=None, *, settings=None, settingsName=None, nRows=10, maxWidth=48, sslRootCert=None, includeDeleted=False):
        """Print database table counts/statuses or selected table contents.

        details=None:
            Database summary.

        details="session", an iterable of names, or "all":
            First nRows rows of each requested table.

        Does not access payload storage.
        """
        from psycopg.rows import tuple_row

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        if type(nRows) is not int or nRows <= 0:
            raise ValueError("nRows must be a positive integer.")

        if type(maxWidth) is not int or maxWidth < 8:
            raise ValueError("maxWidth must be an integer of at least 8.")

        tableOrder = pglDatabase._printTableOrder()

        if details is None:
            requested = []
        elif isinstance(details, str):
            requested = list(tableOrder) if details == "all" else [details]
        else:
            try:
                requested = list(details)
            except TypeError as error:
                raise ValueError("details must be a table name or an iterable of table names.") from error

        if any(not isinstance(name, str) or name not in tableOrder for name in requested):
            raise ValueError(f"Supported detail tables: {', '.join(tableOrder)}; or 'all'.")

        requested = list(dict.fromkeys(requested))
        postgres = pglDatabase._printPostgres(postgres, settings=settings, settingsName=settingsName)

        if postgres is None:
            return False

        # ++++++++++++++++++++++++++++++ reportTable ++++++++++++++++++++++++++++++
        # Local implementation within printTables.
        def reportTable(name):
            table = sql.Identifier(postgres.databaseSchema, name)

            if not includeDeleted and name in {"session", "session_files", "files", "storage_locations"}:
                return sql.SQL("(SELECT * FROM {} WHERE deleted_at IS NULL)").format(table)

            return table

        counts = []
        checkpointCounts = []
        detailResults = []

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return False

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    cursor.execute("SELECT current_database(), current_user, transaction_timestamp()")
                    databaseName, databaseUser, snapshotTime = cursor.fetchone()

                    if requested:
                        for name in requested:
                            order = sql.SQL(", ").join(sql.Identifier(column) for column in tableOrder[name])
                            statement = sql.SQL("SELECT * FROM {} AS record ORDER BY {} LIMIT %s").format(reportTable(name), order)
                            cursor.execute(statement, (nRows + 1,))
                            headers = [column.name for column in cursor.description]
                            rows = cursor.fetchall()
                            detailResults.append((name, headers, rows[:nRows], len(rows) > nRows))
                    else:
                        for name in tableOrder:
                            cursor.execute(sql.SQL("SELECT count(*) FROM {} AS record").format(reportTable(name)))
                            counts.append((name, cursor.fetchone()[0]))

                        if includeDeleted:
                            statement = sql.SQL("SELECT origin, status, deleted_at IS NOT NULL AS deleted, count(*) FROM {} AS record GROUP BY origin, status, deleted_at IS NOT NULL ORDER BY origin, status, deleted").format(reportTable("session"))
                        else:
                            statement = sql.SQL("SELECT origin, status, count(*) FROM {} AS record GROUP BY origin, status ORDER BY origin, status").format(reportTable("session"))

                        cursor.execute(statement)
                        checkpointCounts = cursor.fetchall()

        if requested:
            for name, headers, rows, hasMore in detailResults:
                pglMessages.print()
                pglMessages.printHeader(name, fillChar="-")

                if includeDeleted:
                    pglMessages.print("  Including deleted records; deleted_at is the deletion marker where applicable.")

                pglMessages.print(f"  First {len(rows)} rows, ordered by {', '.join(tableOrder[name])}")
                pglDatabase._printTable(headers, rows, maxWidth=maxWidth)

                if hasMore:
                    pglMessages.message(f"More rows available; increase nRows above {nRows} to display more.")

        else:
            pglMessages.printHeader("PGL database snapshot")
            pglMessages.print("  Including deleted records." if includeDeleted else "  Deleted records excluded.")
            pglMessages.message(f"Database: {databaseName}.{postgres.databaseSchema}", wrapText=False)
            pglMessages.message(f"User: {databaseUser} | Transport: {postgres.connectionTransport}", wrapText=False)
            pglMessages.message(f"Snapshot transaction started: {snapshotTime}", wrapText=False)
            pglMessages.print()
            pglMessages.printHeader("Table counts", fillChar="-")
            pglDatabase._printTable(("table", "rows"), counts, maxWidth=maxWidth)
            pglMessages.print()
            pglMessages.printHeader("Checkpoint status", fillChar="-")
            checkpointHeaders = ("origin", "status", "deleted", "checkpoints") if includeDeleted else ("origin", "status", "checkpoints")
            pglDatabase._printTable(checkpointHeaders, checkpointCounts, maxWidth=maxWidth)

        pglMessages.printHeader()
        return True


    # -------------------------------------------------------------------------
    # Pipeline extraction
    # -------------------------------------------------------------------------
    # Extract recorded analysis using verified checkpoint ancestry without executing the analysis.

    # +++++++++++++++++++++++++++ getSessionPipeline ++++++++++++++++++++++++++
    # User-facing database API. See the docstring for configuration, side effects, and return
    # conventions.
    @classmethod
    def getSessionPipeline(cls, session, name=None, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None):
        """Extract recorded analysis from a session using checkpoint ancestry.

        For sessions with a checkpoint baseline, verify active ancestry and
        saved history metadata, then extract the suffix after the raw boundary.

        Without a baseline, preserve the existing history-only extraction path.
        That path may have a recorded pipeline input or no default input.

        Returns None for unavailable configuration or connection cancellation.
        Invalid ancestry and metadata errors propagate.

        Reads manifests/history only. Does not open scientific payloads,
        change session history, save a recipe, or execute actions.
        """
        from .pglPipeline import pglActionable, pglPipeline
        from .pglPostgres import pglPostgres

        if not isinstance(session, pglActionable):
            raise TypeError("Pipeline extraction requires a pglActionable object.")

        if name is not None:
            pglPipeline._validatePipelineName(name)

        boundary = cls._getCheckpointHistoryBoundary(session)

        if boundary is None:
            return pglPipeline.fromHistory(session, name=name)

        baseline = deepcopy(session._checkpointBaseline)

        if getattr(session, "sessionID", None) != baseline["sessionID"]:
            raise ValueError("Session ID differs from its checkpoint baseline. Reload the intended checkpoint before ancestry-based extraction.")

        postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)

        if postgres is None:
            return None

        if cls._checkpointDatabaseIdentity(postgres) != baseline["databaseIdentity"]:
            raise ValueError("Current PGL settings point to a different database than the session baseline.")

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        checkpoints = []
        seen = set()
        currentID = baseline["sessionID"]

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                while currentID is not None:
                    if currentID in seen:
                        raise ValueError("Checkpoint ancestry contains a cycle.")

                    if len(seen) >= 10000:
                        raise ValueError("Checkpoint ancestry exceeds the replay inspection limit.")

                    seen.add(currentID)
                    statement = sql.SQL("SELECT session_id, parent_session_id, root_session_id, origin, status, contract_version, manifest_hash, deleted_at FROM {} WHERE session_id = %s").format(sessionsTable)
                    row = connection.execute(statement, (currentID,)).fetchone()

                    if row is None:
                        raise ValueError(f"Replay ancestor {currentID} does not exist.")

                    record = dict(zip(("sessionID", "parentSessionID", "rootSessionID", "origin", "status", "contractVersion", "manifestHash", "deletedAt"), row))

                    if record["status"] != "complete" or record["deletedAt"] is not None:
                        raise ValueError(f"Replay ancestor {currentID} is incomplete or deleted.")

                    checkpoints.append(record)
                    currentID = record["parentSessionID"]

                checkpoints.reverse()

                for checkpoint in checkpoints:
                    history = cls._readReplayCheckpointHistory(connection, postgres, checkpoint, storageOptionsByBackend=storageOptionsByBackend)
                    checkpoint["historyHashes"] = [hashlib.sha256(cls.encodeRawManifest(json.loads(entry.toJSON()))).hexdigest() for entry in history.actionHistory]

                inputRecord = cls._selectAncestralReplayInput(session, checkpoints, postgres)

        # Recheck the working prefix after metadata reads. Callers must not
        # concurrently modify the session while extraction is in progress.
        cls._getCheckpointHistoryBoundary(session)

        if session._checkpointBaseline != baseline:
            raise ValueError("The working checkpoint baseline changed during pipeline extraction.")

        prefixLength = inputRecord["historyCount"]
        view = pglActionable()
        view.actionHistory = list(session.actionHistory[prefixLength:])
        view._failedActionIDs = list(session._failedActionIDs)

        # Check only the analysis history selected for extraction, not the
        # acquisition history already excluded by the raw checkpoint boundary.
        from .pglPipeline import pglActionStatus

        for entry in view.actionHistory:
            if entry.actionStatus == pglActionStatus.SUCCESS and (not entry.actionCode.get("module") or not entry.actionCode.get("qualname")):
                pglMessages.warning(f"Cannot extract pipeline: {entry.actionName!r} has old history recorded before action class identities were added. Reloading or resaving will not recover that missing information. The session and its history have not been changed.", level=2)
                return None

        # The tested execution-tree extractor rejects a boundary that splits
        # a parent/child execution tree rather than silently flattening it.
        pipeline = pglPipeline._fromHistoryRecipe(view, name=name)

        # +++++++++++++++++++++++++ restoreHistoryIndices +++++++++++++++++++++++++
        # Local implementation within getSessionPipeline.
        def restoreHistoryIndices(recipe):
            for source in recipe.stepSources.values():
                source["historyIndex"] += prefixLength

            for action in recipe.actions:
                if isinstance(action, pglPipeline):
                    restoreHistoryIndices(action)

        restoreHistoryIndices(pipeline)
        pipeline.replayInput = inputRecord

        pglMessages.message(f"Pipeline extracted from recorded ancestry: input checkpoint {inputRecord['sessionID']}; excluded {prefixLength} history entries already included in that raw checkpoint.")
        return pipeline


    # =========================================================================
    # ADVANCED API
    # =========================================================================

    # -------------------------------------------------------------------------
    # Session loading with caller-owned connections
    # -------------------------------------------------------------------------
    # Lower-level loaders used by loadSession(). Callers own database connections and transactions.

    # +++++++++++++++++++++++++++++ loadRawSession ++++++++++++++++++++++++++
    @classmethod
    def loadRawSession(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False):
        """Restore a raw behavioral checkpoint with retained caller-side access.

        Called by loadSession(), or by advanced callers owning a SQL snapshot.

        Verifies the manifest immediately. Behavioral files remain deferred
        and are verified when opened through pglCheckpointFilesystem.

        Creates one runtime access context for manifest and payload reads.
        On success, ownership transfers to the returned session's filesystem.
        On failure, the context closes before the exception propagates.

        The caller may close SQL after this method returns. Deferred file
        access uses its retained filesystem context, not the SQL connection.

        This low-level loader does not assign checkpoint annotations or bind
        the working history baseline; loadSession() performs that bookkeeping.

        Once all deferred readers are finished, the filesystem owner can call
        session.filesystem.close() to release its SSH connections.

        Does not publish or modify database records.
        """
        from .pglSession import pglSession

        access = pglDatabaseAccess(postgres, storageOptionsByBackend)

        try:
            manifest = cls.loadRawManifest(connection, postgres, sessionID, includeDeleted=includeDeleted, accessContext=access)
            fileLocations = cls.getRawFileLocations(connection, postgres, sessionID, manifest, includeDeleted=includeDeleted)
            filesystem = pglCheckpointFilesystem(manifest, fileLocations, accessContext=access, ownsAccess=True)
            runPaths = [record["path"] for record in manifest["runs"]]
            session = pglSession(filesystem=filesystem, runList=runPaths)
            return session

        except BaseException:
            access.close()
            raise

    # ++++++++++++++++++++++++++ loadRawSessionBundle +++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _saveRawMNECheckpoint.
    @classmethod
    def loadRawSessionBundle(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False, maxManifestBytes=16 * 1024 * 1024, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load an MNE-containing raw root using the caller's transaction."""
        return cls._loadSessionBundle(connection, postgres, sessionID, storageOptionsByBackend=storageOptionsByBackend, includeDeleted=includeDeleted, maxManifestBytes=maxManifestBytes, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, origin="raw")

    # +++++++++++++++++++++++++++ loadDerivedSession ++++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    @classmethod
    def loadDerivedSession(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False, maxManifestBytes=16 * 1024 * 1024, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load a completed derived checkpoint using the caller's transaction."""
        return cls._loadSessionBundle(connection, postgres, sessionID, storageOptionsByBackend=storageOptionsByBackend, includeDeleted=includeDeleted, maxManifestBytes=maxManifestBytes, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, origin="derived")

    # +++++++++++++++++++++++ reconstructDerivedSession +++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # +++++++++++++++++++++++ reconstructDerivedSession ++++++++++++++++++++++
    @classmethod
    def reconstructDerivedSession(cls, manifest, fileLocations, storageOptionsByBackend=None, *, cacheDirectory=None, directLocalPolicy="materialize", cacheContext=None, accessContext=None, ownsAccess=False):
        """Reconstruct a derived checkpoint from supplied inventory metadata.

        Advanced reconstruction entry point; does not query PostgreSQL.
        Does not restore separately registered matrix products.

        Remote database callers must supply a pglDatabaseAccess context.
        Configure backend options on that context rather than supplying a
        second nonempty storageOptionsByBackend mapping.

        ownsAccess=True transfers context ownership to the returned session's
        filesystem on success. On failure, the caller retains ownership.

        Without a context, retains local/explicit-URL compatibility behavior.
        Deferred readers require the returned filesystem to remain open.
        """
        return cls._reconstructSessionBundle(manifest, fileLocations, storageOptionsByBackend=storageOptionsByBackend, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, cacheContext=cacheContext, origin="derived", accessContext=accessContext, ownsAccess=ownsAccess)

    # -------------------------------------------------------------------------
    # Checkpoint creation, registration, and completion
    # -------------------------------------------------------------------------
    # Publication primitives used by save coordinators. They do not commit the caller's outer
    # transaction.

    # ++++++++++++++++++++++++++ createRawCheckpoint ++++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _saveRawSession, registerStagedRawCheckpoint.
    @staticmethod
    def createRawCheckpoint(connection, postgres, createdBy, description=None, ingestMethod="manual"):
        """Create an unpublished raw behavioral checkpoint.

        Uses the caller's connection and transaction; does not commit.
        File links and completion are handled separately.
        """
        if isinstance(createdBy, bool) or not isinstance(createdBy, int) or not 0 <= createdBy <= 9223372036854775807:
            raise ValueError("createdBy must be a nonnegative PostgreSQL bigint.")

        if description is not None and (not isinstance(description, str) or "\x00" in description):
            raise ValueError("description must be a string without NUL, or None.")

        if ingestMethod not in {"manual", "automated"}:
            raise ValueError("ingestMethod must be manual or automated.")

        table = sql.Identifier(postgres.databaseSchema, "session")
        statement = sql.SQL("""
            INSERT INTO {} (
                origin, kind, ingest_method, modalities, status,
                description, contract_version, created_by, started_at
            )
            VALUES (
                'raw', 'acquired', %s, ARRAY['behavior']::text[], 'in_progress',
                %s, 'pgl.raw.behavior.v1', %s, now()
            )
            RETURNING session_id
        """).format(table)

        return connection.execute(statement, (ingestMethod, description, createdBy)).fetchone()[0]

    # ++++++++++++++++++++++++ createDerivedCheckpoint ++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _saveDerivedSession.
    @staticmethod
    def createDerivedCheckpoint(connection, postgres, createdBy, parentSessionID, description=None, ingestMethod="manual", *, checkpointName=None, contractVersion="pgl.derived.behavior.v1", modalities=None):
        """Create an unpublished derived checkpoint in the caller's transaction.

        Supports legacy v1 inventories and v2 inventories containing explicit
        matrix product references.

        Database lineage rules require a completed, active parent.
        Modalities describe source content categories; matrices do not add
        a separate modality, and 'mne' does not imply EEG/MEG.

        Does not write payloads, attach products, complete the checkpoint,
        or modify the working session.
        """
        for label, value in (("createdBy", createdBy), ("parentSessionID", parentSessionID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        for label, value in (("checkpointName", checkpointName), ("description", description)):
            if value is not None and (not isinstance(value, str) or "\x00" in value):
                raise ValueError(f"{label} must be a string without NUL, or None.")

        if ingestMethod not in ("manual", "automated"):
            raise ValueError("ingestMethod must be manual or automated.")

        behaviorContracts = {"pgl.derived.behavior.v1", "pgl.derived.behavior.v2"}
        sessionContracts = {"pgl.derived.session.v1", "pgl.derived.session.v2"}

        if contractVersion not in behaviorContracts | sessionContracts:
            raise ValueError("Unsupported derived checkpoint contract.")

        hasMNE = contractVersion in sessionContracts

        if modalities is None:
            modalities = ["mne"] if hasMNE else ["behavior"]

        if not isinstance(modalities, (list, tuple)) or not modalities:
            raise ValueError("modalities must be a nonempty list or tuple.")

        if any(not isinstance(value, str) or value not in ("behavior", "mne") for value in modalities):
            raise ValueError("Supported content categories are 'behavior' and 'mne'.")

        if len(set(modalities)) != len(modalities):
            raise ValueError("Duplicate modalities are not allowed.")

        if not hasMNE and set(modalities) != {"behavior"}:
            raise ValueError("Behavioral checkpoints require only the behavior category.")

        if hasMNE and "mne" not in modalities:
            raise ValueError("MNE session checkpoints require the mne category.")

        table = sql.Identifier(postgres.databaseSchema, "session")
        statement = sql.SQL("""
            INSERT INTO {} (
                origin, kind, parent_session_id, ingest_method,
                modalities, status, checkpoint_name, description,
                contract_version, created_by, started_at
            )
            VALUES (
                'derived', 'processed', %s, %s, %s::text[], 'in_progress',
                %s, %s, %s, %s, now()
            )
            RETURNING session_id
        """).format(table)

        return connection.execute(statement, (parentSessionID, ingestMethod, list(modalities), checkpointName, description, contractVersion, createdBy)).fetchone()[0]

    # ++++++++++++++++++++++ registerStagedRawCheckpoint +++++++++++++++++++++
    @classmethod
    def registerStagedRawCheckpoint(cls, connection, postgres, createdBy, backendID, manifest, staged, storedManifest, storageOptions=None, *, sessionID=None, accessContext=None):
        """Register a staged behavioral archive without completing it.

        Called by the raw behavioral save coordinator after payload staging.

        Validates staged identities against the manifest, reads acquisition
        labels through verified checkpoint files, registers content locations,
        and attaches logical files to an unpublished checkpoint.

        Server-local roots resolve through the caller's local/SSH context.
        The catalog retains its original backend root and relative keys.

        accessContext:
            Optional borrowed context from the publication coordinator.
            Configure storage options on that context, not both arguments.
            The caller retains ownership.

        Without accessContext:
            Creates one operation-owned context using storageOptions for the
            destination backend, then closes it on success or failure.

        Uses the caller's SQL transaction; does not commit or complete the
        checkpoint. Does not copy or rewrite payloads.

        Missing subjects become SQL NULL. Missing-label consent is enforced
        by the public raw saver before staging.

        Returns:
            The allocated or supplied unpublished checkpoint ID.

        Exceptions propagate. The caller must roll back failed registration.
        Closing access resources does not undo database changes or files
        written during the earlier staging operation.
        """
        from contextlib import ExitStack

        if type(backendID) is not int or not 0 <= backendID <= 9223372036854775807:
            raise ValueError("backendID must be a nonnegative PostgreSQL bigint.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        if accessContext is not None:
            if not isinstance(accessContext, pglDatabaseAccess):
                raise TypeError("accessContext must be a pglDatabaseAccess instance.")

            if storageOptions:
                raise ValueError("Configure storage options on accessContext, not on both arguments.")

            if cls._checkpointDatabaseIdentity(accessContext._postgres) != cls._checkpointDatabaseIdentity(postgres):
                raise ValueError("The supplied payload context belongs to a different configured database.")

        cls.validateRawManifest(manifest)
        expectedByName = {record["name"]: record for record in manifest["files"]}
        stagedByName = {record["name"]: record for record in staged}

        if len(expectedByName) != len(manifest["files"]) or len(stagedByName) != len(staged):
            raise ValueError("Duplicate logical filenames are not allowed.")

        if set(expectedByName) != set(stagedByName):
            raise ValueError("Staged files do not match the manifest filenames.")

        if "manifest.json" in expectedByName:
            raise ValueError("manifest.json is reserved for the checkpoint manifest.")

        for name, expected in expectedByName.items():
            actual = stagedByName[name]

            if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
                raise ValueError(f"Staged identity differs from manifest: {name}")

        manifestBytes = cls.encodeRawManifest(manifest)

        if storedManifest["sha256"] != hashlib.sha256(manifestBytes).hexdigest() or storedManifest["size_bytes"] != len(manifestBytes):
            raise ValueError("Stored manifest identity does not match the supplied manifest.")

        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")
        statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s").format(backendsTable)
        backend = connection.execute(statement, (backendID,)).fetchone()

        if backend is None or not backend[1]:
            raise ValueError(f"Storage backend is missing or inactive: {backendID}")

        registeredRoot = backend[0]

        with ExitStack() as resources:
            if accessContext is None:
                access = resources.enter_context(pglDatabaseAccess(postgres, {backendID: storageOptions or {}}))
            else:
                access = accessContext

            fileLocations = {
                name: {
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                    "locations": [{
                        "backend_id": backendID,
                        "url_prefix": registeredRoot,
                        "object_key": record["object_key"],
                    }],
                }
                for name, record in stagedByName.items()
            }

            # The archive borrows access. Closing it must not close a context
            # supplied by the enclosing save coordinator.
            archive = pglCheckpointFilesystem(manifest, fileLocations, accessContext=access, ownsAccess=False)
            resources.callback(archive.close)

            metadata = cls._behaviorLabelMetadata(manifest, archive)
            subjectsByRun = {run["path"]: metadata["labels"][run["index"]]["subjectID"] for run in manifest["runs"]}
            subjectsByName = {}

            for name in expectedByName:
                matchingRuns = [path for path in subjectsByRun if name.startswith(path + "/")]

                if len(matchingRuns) != 1:
                    raise ValueError(f"File does not belong to exactly one run: {name}")

                subjectsByName[name] = subjectsByRun[matchingRuns[0]]

            if sessionID is None:
                checkpointID = cls.createRawCheckpoint(connection, postgres, createdBy)
            else:
                sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
                linksTable = sql.Identifier(postgres.databaseSchema, "session_files")

                statement = sql.SQL("SELECT status, origin, contract_version, created_by, deleted_at FROM {} WHERE session_id = %s FOR UPDATE").format(sessionsTable)
                checkpoint = connection.execute(statement, (sessionID,)).fetchone()

                if checkpoint != ("in_progress", "raw", "pgl.raw.behavior.v1", createdBy, None):
                    raise ValueError("Checkpoint must be an active in-progress raw behavioral checkpoint owned by the supplied creator.")

                statement = sql.SQL("SELECT 1 FROM {} WHERE session_id = %s LIMIT 1").format(linksTable)

                if connection.execute(statement, (sessionID,)).fetchone() is not None:
                    raise ValueError("Checkpoint already has file links.")

                checkpointID = sessionID

            cls._recordSessionLabelMetadata(connection, postgres, checkpointID, metadata)
            registeredPayloads = {}

            # +++++++++++++++++++++++ registerPayload +++++++++++++++++++++++
            def registerPayload(record):
                """Verify/register each distinct staged identity and location once."""
                identity = (record["sha256"], record["size_bytes"], record["object_key"])

                if identity not in registeredPayloads:
                    fileID = cls.getOrCreateFile(connection, postgres, record["sha256"], record["size_bytes"])
                    cls.verifyAndRegisterStorageLocation(connection, postgres, fileID, backendID, record["object_key"], accessContext=access)
                    registeredPayloads[identity] = fileID

                return registeredPayloads[identity]

            for name in sorted(expectedByName):
                fileID = registerPayload(stagedByName[name])
                cls.attachCheckpointFile(connection, postgres, checkpointID, name, fileID, role="acquisition", subjectID=subjectsByName[name])

            manifestFileID = registerPayload(storedManifest)
            cls.attachCheckpointFile(connection, postgres, checkpointID, "manifest.json", manifestFileID, role="manifest")

            return checkpointID

    # ++++++++++++++++++++++ registerDerivedSessionFiles ++++++++++++++++++++
    @classmethod
    def registerDerivedSessionFiles(cls, connection, postgres, sessionID, backendID, manifest, *, objectPrefix, storageOptions=None, accessContext=None):
        """Register a staged derived bundle in the caller's transaction.

        Advanced wrapper used by the derived save coordinator.

        A supplied accessContext is borrowed and forwarded unchanged.
        Otherwise the shared implementation owns a temporary context.
        Configure storage options on the context when supplying one.

        Verifies and registers files but does not copy payloads, register
        matrix-product links, complete the checkpoint, or commit.
        """
        return cls._registerSessionBundleFiles(connection, postgres, sessionID, backendID, manifest, objectPrefix=objectPrefix, storageOptions=storageOptions, origin="derived", accessContext=accessContext)
    
    # +++++++++++++++++++++++++ completeRawCheckpoint +++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _saveRawSession.
    @classmethod
    def completeRawCheckpoint(cls, connection, postgres, sessionID, manifest):
        """Validate registered identities and complete a raw checkpoint.

        Must run inside the caller's transaction.
        Checks recorded verification status; does not reread payload bytes.
        """
        cls.validateRawManifest(manifest)
        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("""
            SELECT status, origin, contract_version
            FROM {} WHERE session_id = %s FOR UPDATE
        """).format(sessionsTable)
        checkpoint = connection.execute(statement, (sessionID,)).fetchone()

        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {sessionID}")

        if checkpoint != ("in_progress", "raw", "pgl.raw.behavior.v1"):
            raise ValueError("Checkpoint must be an in-progress raw behavioral checkpoint.")

        if manifest["contract_version"] != checkpoint[2]:
            raise ValueError("Manifest contract does not match the checkpoint.")

        expected = {}
        for record in manifest["files"]:
            name = record["name"]

            if name == "manifest.json" or name in expected:
                raise ValueError(f"Reserved or duplicate manifest filename: {name}")

            expected[name] = (record["sha256"], record["size_bytes"], "acquisition")

        manifestBytes = cls.encodeRawManifest(manifest)
        manifestHash = hashlib.sha256(manifestBytes).hexdigest()
        expected["manifest.json"] = (manifestHash, len(manifestBytes), "manifest")

        statement = sql.SQL("""
            SELECT link.name, content.sha256, content.size_bytes, link.role,
                   EXISTS (
                       SELECT 1
                       FROM {} AS location
                       JOIN {} AS backend ON backend.backend_id = location.backend_id
                       WHERE location.file_id = content.file_id
                         AND location.status = 'present'
                         AND location.verified_at IS NOT NULL
                         AND backend.is_active
                   ) AS has_verified_location
            FROM {} AS link
            JOIN {} AS content ON content.file_id = link.file_id
            WHERE link.session_id = %s
        """).format(locationsTable, backendsTable, linksTable, filesTable)

        rows = connection.execute(statement, (sessionID,)).fetchall()
        actual = {name: (sha256, size, role) for name, sha256, size, role, available in rows}

        if actual != expected:
            raise ValueError("Checkpoint file links do not exactly match the manifest.")

        unavailable = [name for name, sha256, size, role, available in rows if not available]
        if unavailable:
            raise ValueError(f"Files lack verified active storage locations: {', '.join(sorted(unavailable))}")

        statement = sql.SQL("""
            UPDATE {}
            SET status = 'complete', manifest_hash = %s, completed_at = now()
            WHERE session_id = %s
        """).format(sessionsTable)
        connection.execute(statement, (manifestHash, sessionID))

        return manifestHash

    # +++++++++++++++++++++++ completeDerivedCheckpoint +++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _saveDerivedSession.
    @classmethod
    def completeDerivedCheckpoint(cls, connection, postgres, sessionID, manifest):
        """Complete a derived checkpoint using the shared validation implementation."""
        return cls._completeSessionBundleCheckpoint(connection, postgres, sessionID, manifest, origin="derived")


    # -------------------------------------------------------------------------
    # Catalog registration
    # -------------------------------------------------------------------------
    # Resolve catalog identities and attach verified locations or logical file links in the caller's
    # transaction.

    # ++++++++++++++++++++++++++++ getOrCreateUser ++++++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: saveDataset, saveRawSession, saveSession.
    @staticmethod
    def getOrCreateUser(connection, postgres, username, displayName=None):
        """Return a creator's user_id without changing an existing profile.

        Uses the caller's connection and transaction; does not commit.
        """
        if not isinstance(username, str) or not username.strip() or "\x00" in username:
            raise ValueError("username must be a nonempty string without NUL.")

        if displayName is not None and (not isinstance(displayName, str) or "\x00" in displayName):
            raise ValueError("displayName must be a string without NUL, or None.")

        table = sql.Identifier(postgres.databaseSchema, "users")
        statement = sql.SQL("INSERT INTO {} (username, display_name) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING RETURNING user_id").format(table)
        row = connection.execute(statement, (username, displayName)).fetchone()

        if row is not None:
            return row[0]

        statement = sql.SQL("SELECT user_id FROM {} WHERE username = %s").format(table)
        row = connection.execute(statement, (username,)).fetchone()

        if row is None:
            raise RuntimeError("User could not be resolved; retry the transaction.")

        return row[0]

    # +++++++++++++++++++++++++++ getOrCreateSubject ++++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _recordSessionLabelMetadata.
    @staticmethod
    def getOrCreateSubject(connection, postgres, subjectID):
        """Register or reuse a subject without changing an existing description.

        Uses the caller's connection and transaction; does not commit.
        """
        if not isinstance(subjectID, str) or not subjectID.strip() or "\x00" in subjectID:
            raise ValueError("subjectID must be a nonempty string without NUL.")

        table = sql.Identifier(postgres.databaseSchema, "subjects")
        statement = sql.SQL("INSERT INTO {} (subject_id) VALUES (%s) ON CONFLICT (subject_id) DO NOTHING RETURNING subject_id").format(table)
        row = connection.execute(statement, (subjectID,)).fetchone()

        if row is not None:
            return row[0]

        statement = sql.SQL("SELECT subject_id FROM {} WHERE subject_id = %s").format(table)
        row = connection.execute(statement, (subjectID,)).fetchone()

        if row is None:
            raise RuntimeError("Subject could not be resolved; retry the transaction.")

        return row[0]

    # +++++++++++++++++++++++ getOrCreateStorageBackend +++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _getConfiguredStorageBackend.
    @staticmethod
    def getOrCreateStorageBackend(connection, postgres, name, kind, urlPrefix):
        """Return a backend_id, rejecting conflicting existing configuration.

        Uses the caller's connection and transaction; does not commit.
        Does not probe storage or store credentials.
        """
        if not isinstance(name, str) or not name.strip() or "\x00" in name:
            raise ValueError("Backend name must be a nonempty string without NUL.")

        if kind not in {"local", "nas", "cloud"}:
            raise ValueError("Backend kind must be local, nas, or cloud.")

        if not isinstance(urlPrefix, str) or not urlPrefix.strip() or any(character in urlPrefix for character in ("\x00", "\n", "\r")):
            raise ValueError("Backend root must be a nonempty path or URL without NUL or newline characters.")

        table = sql.Identifier(postgres.databaseSchema, "storage_backends")
        statement = sql.SQL("INSERT INTO {} (name, kind, url_prefix) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING RETURNING backend_id").format(table)
        row = connection.execute(statement, (name, kind, urlPrefix)).fetchone()

        if row is not None:
            return row[0]

        statement = sql.SQL("SELECT backend_id, kind, url_prefix, is_active FROM {} WHERE name = %s").format(table)
        row = connection.execute(statement, (name,)).fetchone()

        if row is None:
            raise RuntimeError("Storage backend could not be resolved; retry the transaction.")

        backendID, existingKind, existingRoot, isActive = row

        if existingKind != kind or existingRoot != urlPrefix:
            raise ValueError(f"Storage backend {name!r} already exists with different configuration.")

        if not isActive:
            raise ValueError(f"Storage backend {name!r} is inactive.")

        return backendID

    # ++++++++++++++++++++++++++++ getOrCreateFile ++++++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _registerSessionBundleFiles, registerStagedRawCheckpoint.
    @staticmethod
    def getOrCreateFile(connection, postgres, sha256, sizeBytes):
        """Return a file_id for a content identity.

        Uses the caller's connection and transaction; does not commit.
        Storage locations are registered separately.
        """
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("SHA-256 must contain 64 lowercase hexadecimal characters.")

        if isinstance(sizeBytes, bool) or not isinstance(sizeBytes, int) or not 0 <= sizeBytes <= 9223372036854775807:
            raise ValueError("sizeBytes must be a nonnegative PostgreSQL bigint.")

        table = sql.Identifier(postgres.databaseSchema, "files")
        statement = sql.SQL("INSERT INTO {} (sha256, size_bytes, kind) VALUES (%s, %s, %s) ON CONFLICT (sha256) DO NOTHING RETURNING file_id").format(table)
        row = connection.execute(statement, (sha256, sizeBytes, "file")).fetchone()

        if row is not None:
            return row[0]

        statement = sql.SQL("SELECT file_id, size_bytes, kind FROM {} WHERE sha256 = %s").format(table)
        row = connection.execute(statement, (sha256,)).fetchone()

        if row is None:
            raise RuntimeError("File identity could not be resolved; retry the transaction.")

        fileID, existingSize, existingKind = row

        if existingSize != sizeBytes or existingKind != "file":
            raise ValueError("Existing SHA-256 has conflicting size or kind; file identity was not changed.")

        return fileID

    # ++++++++++++++++++++ verifyAndRegisterStorageLocation +++++++++++++++++
    @classmethod
    def verifyAndRegisterStorageLocation(cls, connection, postgres, fileID, backendID, objectKey, storageOptions=None, *, accessContext=None):
        """Verify stored bytes and register their backend-relative location.

        Called by raw and session-bundle registration coordinators.

        Queries the registered content identity and backend root, resolves
        caller-side filesystem access, hashes the complete stored file, then
        inserts or refreshes its verified location.

        The catalog retains:
            The original backend root.
            The supplied relative object key.
            The verified content identity.

        Caller-derived SSH URLs and credentials are never registered.

        accessContext:
            Optional borrowed publication context configured for the same
            database and backend. Its caller owns it and its storage options.
            This method does not close a supplied context.

        Without a supplied context:
            Creates a temporary context using storageOptions for backendID.
            Closes that context after verification, including on failure.

        Uses the caller's SQL transaction; does not commit.
        Does not write, copy, or delete payload files.
        A location cannot be reassigned to a different content identity.

        Returns the registered object key. Errors propagate.
        Verification establishes the bytes read now, not future immutability.
        """
        for label, value in (("fileID", fileID), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if not isinstance(objectKey, str) or objectKey.startswith("/") or "\\" in objectKey or "\x00" in objectKey or any(part in {"", ".", ".."} for part in objectKey.split("/")):
            raise ValueError("objectKey must be a valid relative POSIX path.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        if accessContext is not None:
            if not isinstance(accessContext, pglDatabaseAccess):
                raise TypeError("accessContext must be a pglDatabaseAccess instance.")

            if storageOptions:
                raise ValueError("Configure storage options on accessContext, not on both arguments.")

            # A borrowed context must describe the same configured database.
            # This is the existing runtime endpoint identity check, not a
            # globally portable database UUID.
            if cls._checkpointDatabaseIdentity(accessContext._postgres) != cls._checkpointDatabaseIdentity(postgres):
                raise ValueError("The supplied payload context belongs to a different configured database.")

        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")

        statement = sql.SQL("SELECT sha256, size_bytes FROM {} WHERE file_id = %s").format(filesTable)
        identity = connection.execute(statement, (fileID,)).fetchone()

        if identity is None:
            raise ValueError(f"Unknown file_id: {fileID}")

        statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s").format(backendsTable)
        backend = connection.execute(statement, (backendID,)).fetchone()

        if backend is None or not backend[1]:
            raise ValueError(f"Storage backend is missing or inactive: {backendID}")

        statement = sql.SQL("SELECT file_id FROM {} WHERE backend_id = %s AND object_key = %s").format(locationsTable)
        existing = connection.execute(statement, (backendID, objectKey)).fetchone()

        if existing is not None and existing[0] != fileID:
            raise ValueError("Storage path is already registered to different content.")

        ownsAccess = accessContext is None
        access = pglDatabaseAccess(postgres, {backendID: storageOptions or {}}) if ownsAccess else accessContext

        try:
            filesystem, root = access.resolve(backendID, backend[0])
            actual = pglStorage.hashFile(filesystem, posixpath.join(root, objectKey))

            if actual["sha256"] != identity[0] or actual["size_bytes"] != identity[1]:
                raise ValueError("Stored payload does not match the registered file identity.")

        finally:
            if ownsAccess:
                access.close()

        # Verification precedes catalog publication. The existing database
        # trigger coordinates deletion markers with shared content identity.
        statement = sql.SQL("""
            INSERT INTO {} AS location
                (file_id, backend_id, object_key, status, verified_at)
            VALUES (%s, %s, %s, 'present', now())
            ON CONFLICT (backend_id, object_key) DO UPDATE
            SET status = 'present', verified_at = EXCLUDED.verified_at
            WHERE location.file_id = EXCLUDED.file_id
            RETURNING object_key
        """).format(locationsTable)

        row = connection.execute(statement, (fileID, backendID, objectKey)).fetchone()

        if row is None:
            raise ValueError("Storage path was registered concurrently to different content.")

        return row[0]

    # ++++++++++++++++++++++++++ attachCheckpointFile +++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _registerSessionBundleFiles, registerStagedRawCheckpoint.
    @staticmethod
    def attachCheckpointFile(connection, postgres, sessionID, name, fileID, role=None, subjectID=None):
        """Attach a file to an in-progress checkpoint.

        Uses the caller's connection and transaction; does not commit.
        Duplicate logical names are rejected, not overwritten.
        Does not verify payload availability.
        """
        for label, value in (("sessionID", sessionID), ("fileID", fileID)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if not isinstance(name, str) or name.startswith("/") or "\\" in name or "\x00" in name or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("name must be a valid relative POSIX path.")

        for label, value in (("role", role), ("subjectID", subjectID)):
            if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
                raise ValueError(f"{label} must be a nonempty string without NUL, or None.")

        table = sql.Identifier(postgres.databaseSchema, "session_files")
        statement = sql.SQL("INSERT INTO {} (session_id, name, file_id, role, subject_id) VALUES (%s, %s, %s, %s, %s)").format(table)
        connection.execute(statement, (sessionID, name, fileID, role, subjectID))


    # -------------------------------------------------------------------------
    # Lineage and lifecycle inspection
    # -------------------------------------------------------------------------
    # Read-only previews. These queries alone do not authorize concurrent lifecycle changes.

    # +++++++++++++++++++++++++ getDownstreamSessions +++++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    # Direct callers in this module: _getSessionDeletionRecords.
    @staticmethod
    def getDownstreamSessions(connection, postgres, sessionID):
        """Return all descendants, including deleted sessions, ordered by ID.

        Follows parent_session_id recursively. The requested session itself
        is excluded, even if a cycle leads back to it.

        Returns a list of metadata dictionaries. An existing session with
        no descendants returns an empty list; an unknown session raises
        ValueError.

        Read-only: does not change rows, access payloads, or commit.
        One SQL statement provides a consistent snapshot.

        This is an informational lookup, not a concurrency-safe deletion
        decision. Deletion must repeat it under the required locks.
        """
        from psycopg.rows import dict_row

        if isinstance(sessionID, bool) or not isinstance(sessionID, int) or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")

        # Recurse over IDs only. UNION removes previously visited IDs,
        # including those reached again through a cycle.
        statement = sql.SQL("""
            WITH RECURSIVE reachable(session_id) AS (
                SELECT session_id
                FROM {sessions}
                WHERE session_id = %s

                UNION

                SELECT child.session_id
                FROM {sessions} AS child
                JOIN reachable AS parent
                  ON child.parent_session_id = parent.session_id
            )
            SELECT checkpoint.session_id,
                   checkpoint.parent_session_id,
                   checkpoint.root_session_id,
                   checkpoint.origin,
                   checkpoint.kind,
                   checkpoint.status,
                   checkpoint.description,
                   checkpoint.created_at,
                   checkpoint.deleted_at
            FROM {sessions} AS checkpoint
            JOIN reachable
              ON reachable.session_id = checkpoint.session_id
            ORDER BY checkpoint.session_id
        """).format(sessions=sessions)

        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(statement, (sessionID,))
            records = cursor.fetchall()

        if not records:
            raise ValueError(f"Unknown session: {sessionID}")

        return [record for record in records if record["session_id"] != sessionID]

    # ++++++++++++++++++++++++ getSessionDeletionImpact +++++++++++++++++++++++
    # Advanced database/checkpoint API. Observe the connection and transaction requirements in the
    # docstring.
    @staticmethod
    def getSessionDeletionImpact(connection, postgres, sessionID):
        """Preview which linked content would become eligible for soft deletion.

        Includes deleted sessions. Does not change database rows or payloads.

        Files remain shared when another nondeleted session references them,
        regardless of that session's publication status.

        This is an informational preview, not a concurrency-safe deletion
        decision. The actual delete operation must coordinate concurrent
        changes and recompute eligibility inside its transaction.
        """
        if (
            isinstance(sessionID, bool)
            or not isinstance(sessionID, int)
            or not 0 <= sessionID <= 9223372036854775807
        ):
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")

        # One statement provides a consistent snapshot for this preview.
        statement = sql.SQL("""
            SELECT checkpoint.deleted_at,
                   linked.file_id,
                   CASE WHEN linked.file_id IS NOT NULL THEN EXISTS (
                       SELECT 1
                       FROM {links} AS other_link
                       JOIN {sessions} AS other_session
                         ON other_session.session_id = other_link.session_id
                       WHERE other_link.file_id = linked.file_id
                         AND other_session.session_id <> checkpoint.session_id
                         AND other_session.deleted_at IS NULL
                   ) ELSE false END AS shared_with_active_session,
                   (
                       SELECT count(*)
                       FROM {locations} AS location
                       WHERE location.file_id = linked.file_id
                   ) AS location_count
            FROM {sessions} AS checkpoint
            LEFT JOIN (
                SELECT DISTINCT session_id, file_id
                FROM {links}
                WHERE session_id = %s
            ) AS linked
              ON linked.session_id = checkpoint.session_id
            WHERE checkpoint.session_id = %s
            ORDER BY linked.file_id
        """).format(sessions=sessions, links=links, locations=locations)

        rows = connection.execute(statement, (sessionID, sessionID)).fetchall()

        if not rows:
            raise ValueError(f"Unknown session: {sessionID}")

        exclusive = []
        shared = []
        exclusiveLocationCount = 0
        sharedLocationCount = 0

        for deletedAt, fileID, sharedWithActiveSession, locationCount in rows:
            if fileID is None:
                continue

            if sharedWithActiveSession:
                shared.append(fileID)
                sharedLocationCount += locationCount
            else:
                exclusive.append(fileID)
                exclusiveLocationCount += locationCount

        return {
            "session_id": sessionID,
            "deleted_at": rows[0][0],
            "exclusive_file_ids": exclusive,
            "shared_file_ids": shared,
            "exclusive_location_count": exclusiveLocationCount,
            "shared_location_count": sharedLocationCount,
        }


    # =========================================================================
    # IMPLEMENTATION SUPPORT
    # =========================================================================

    # -------------------------------------------------------------------------
    # Configuration, database identity, and checkpoint baselines
    # -------------------------------------------------------------------------
    # Shared database binding, destination selection, cache configuration, and saved-history
    # boundaries.

    # +++++++++++++++++++++++++ _resolveBackendAccess +++++++++++++++++++++++
    @staticmethod
    def _resolveBackendAccess(postgres, registeredRoot, *, storageOptions=None):
        """Resolve caller-side access without changing a registered root.

        Called by database payload readers and publication coordinators.
        Returns (filesystem, filesystemRoot).

        Server-local roots:
            Absolute POSIX paths and local file:// URLs identify paths on the
            installation machine, not automatically on the calling computer.

            Direct access is permitted only for a locally loaded/constructed
            configuration targeting a loopback PostgreSQL endpoint.

            SSH access requires PostgreSQL to be reached through the SSH
            endpoint's loopback interface. The configured SSH endpoint is
            therefore assumed to be the installation/storage machine.

        Explicit remote URLs:
            Use their own fsspec protocol independently of the SQL transport.
            Their existing authentication and host-verification behavior is
            governed by that backend, not by the inferred SSH adapter.

        Initial restrictions:
            Relative paths, server-side ~ expansion, filesystem chains, and
            remote authorities in file:// URLs are rejected.
            Inferred SSH access uses the caller's resolved SSH configuration.
            Nonempty storageOptions are rejected for inferred SSH access,
            rather than allowing an endpoint or security-policy override.

        Does not query SQL, write payloads, register locations, or modify the
        supplied PostgreSQL configuration.

        Resolution may open an SSH/backend connection.
        A returned inferred-SSH filesystem is caller-owned: retain it while
        lazy readers depend on it, then close it when no longer needed.

        Explicit remote URL support retains existing behavior; this function
        does not establish that every remote protocol has been tested.

        Raises for ambiguous locations or unsupported access configurations.
        Never falls back from failed SSH access to client-local access.
        """
        import ipaddress
        from urllib.parse import unquote, urlsplit

        from fsspec.core import url_to_fs
        from fsspec.implementations.local import LocalFileSystem

        if not isinstance(registeredRoot, str) or not registeredRoot.strip():
            raise ValueError("Registered storage root must be a nonempty string.")

        if any(character in registeredRoot for character in ("\x00", "\n", "\r")):
            raise ValueError("Registered storage root contains invalid control characters.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        options = dict(storageOptions or {})

        # A URI can contain :: within an IPv6 authority. Examine filesystem
        # chaining outside that authority instead of rejecting IPv6 URLs.
        explicitURL = "://" in registeredRoot
        parsed = urlsplit(registeredRoot) if explicitURL else None

        if explicitURL:
            chainOutsideAuthority = "::" in registeredRoot.split("://", 1)[0] or "::" in parsed.path

            if chainOutsideAuthority:
                raise NotImplementedError("Filesystem chains are not supported for database backend roots.")

            protocol = parsed.scheme.lower()

            if protocol not in {"file", "local"}:
                return url_to_fs(registeredRoot, **options)

            if parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
                raise ValueError("A local storage URL must contain only a local authority and an absolute path.")

            serverRoot = unquote(parsed.path)
        else:
            if "::" in registeredRoot:
                raise NotImplementedError("Filesystem chains are not supported for database backend roots.")

            serverRoot = registeredRoot

        # Decode file:// paths before checking them. Never resolve these paths
        # with Path.resolve() on a client accessing a remote installation.
        if (
            not serverRoot.startswith("/")
            or serverRoot.startswith("//")
            or "\\" in serverRoot
            or any(character in serverRoot for character in ("\x00", "\n", "\r"))
            or any(part in {".", ".."} for part in serverRoot.split("/"))
        ):
            raise ValueError("Server-local storage roots must be absolute POSIX paths without dot segments, backslashes, or control characters.")

        if postgres is None:
            raise ValueError("A database access context is required to resolve a server-local storage root.")

        postgres._validateValues()

        # ++++++++++++++++++++++++++ isLoopback +++++++++++++++++++++++++++++
        def isLoopback(host):
            """Recognize explicit loopback endpoints without DNS inference."""
            if host.lower() == "localhost":
                return True

            candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host

            try:
                return ipaddress.ip_address(candidate).is_loopback
            except ValueError:
                return False

        if postgres.connectionTransport == "ssh":
            if not isLoopback(postgres.connectionHost):
                raise ValueError("Cannot infer payload access when SSH forwards PostgreSQL to another host. Server-local payload access currently requires a loopback database endpoint on the SSH installation machine.")

            if options:
                raise ValueError("storageOptions cannot override inferred SSH access. Configure the caller's SSH identity through pglPostgres and OpenSSH configuration.")

            from .pglSSH import pglSSHFilesystem

            access = postgres.getSSHAccessConfiguration()
            filesystem = pglSSHFilesystem(access)
            return filesystem, serverRoot

        if postgres.connectionTransport == "direct":
            if not isLoopback(postgres.connectionHost):
                raise ValueError("A direct remote PostgreSQL connection does not establish access to server-local payload paths. Use SSH to the installation machine or an explicit remote storage URL.")

            if getattr(postgres, "_configurationLoadedRemotely", False):
                raise ValueError("A remotely loaded direct-loopback configuration cannot authorize access to client-local payload paths.")

            # Local direct-loopback access is an explicit deployment
            # assumption, not proof that an externally managed tunnel or
            # container shares the caller's filesystem.
            filesystem = LocalFileSystem(**options)
            return filesystem, serverRoot

        raise ValueError("Unsupported database connection transport.")

    # ++++++++++++++++++++++ _checkpointDatabaseIdentity ++++++++++++++++++++++
    # Direct callers in this module: _bindDatasetStorage, _prepareCheckpointBaseline,
    # _resolveDatasetStorage, _selectAncestralReplayInput, getDatasetBehaviorData, getDatasetData,
    # getSessionPipeline, saveDataset.
    @staticmethod
    def _checkpointDatabaseIdentity(postgres):
        """Describe the configured database endpoint without credentials.

        This runtime identity distinguishes configured destinations.
        It is not a globally stable database identity across relocation,
        aliases, restores, or alternate connection transports.
        """
        identity = {
            "transport": postgres.connectionTransport,
            "host": postgres.connectionHost,
            "port": postgres.connectionPort,
            "database": postgres.databaseName,
            "schema": postgres.databaseSchema,
        }

        if postgres.connectionTransport == "ssh":
            identity["sshHost"] = postgres.sshHost
            identity["sshPort"] = postgres.sshPort
            identity["sshUser"] = postgres.sshUser

        return identity

    # ++++++++++++++++++++++ _getConfiguredStorageBackend ++++++++++++++++++++
    @classmethod
    def _getConfiguredStorageBackend(cls, connection, postgres, storageOptions=None):
        """Resolve/register the first configured payload destination.

        Called by session and dataset save coordinators when backendID is
        omitted. Uses the caller's SQL transaction; does not commit.

        Registers the original configured root, never a caller-derived SSH
        URL. Bare absolute paths and local file URLs remain kind='local',
        regardless of whether this caller will access them through SFTP.

        Existing active registrations are reused without changing their root
        or kind. Actual destination access is resolved by the save coordinator
        before writing payloads.

        Explicit remote protocols are classified without opening a connection.
        Unknown protocols require explicit backend registration.

        storageOptions is retained for interface compatibility. This metadata
        operation does not use or persist authentication options.

        Does not write payload files or test filesystem permissions.
        Errors propagate.
        """
        from urllib.parse import unquote, urlsplit

        if not postgres.storageLocations:
            raise ValueError("No payload destination is configured in postgres.storageLocations.")

        configuredRoot = postgres.storageLocations[0]

        if not isinstance(configuredRoot, str) or not configuredRoot.strip():
            raise ValueError("The configured payload destination must be a nonempty path or URL.")

        if any(character in configuredRoot for character in ("\x00", "\n", "\r")):
            raise ValueError("The configured payload destination contains invalid control characters.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        explicitURL = "://" in configuredRoot
        parsed = urlsplit(configuredRoot) if explicitURL else None

        if explicitURL:
            if "::" in configuredRoot.split("://", 1)[0] or "::" in parsed.path:
                raise NotImplementedError("Filesystem chains are not supported for database backend roots.")

            protocol = parsed.scheme.lower()
        else:
            if "::" in configuredRoot:
                raise NotImplementedError("Filesystem chains are not supported for database backend roots.")

            protocol = "file"

        if protocol in {"file", "local"}:
            if explicitURL:
                if parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
                    raise ValueError("A local storage URL must contain only a local authority and an absolute path.")

                serverRoot = unquote(parsed.path)
            else:
                serverRoot = configuredRoot

            if (
                not serverRoot.startswith("/")
                or serverRoot.startswith("//")
                or "\\" in serverRoot
                or any(character in serverRoot for character in ("\x00", "\n", "\r"))
                or any(part in {".", ".."} for part in serverRoot.split("/"))
            ):
                raise ValueError("Server-local storage roots must be absolute POSIX paths without dot segments, backslashes, or control characters.")

            kind = "local"
        elif protocol in {"ssh", "sftp", "smb", "nfs"}:
            kind = "nas"
        elif protocol in {"s3", "gs", "gcs", "az", "abfs", "abfss", "adl"}:
            kind = "cloud"
        else:
            kind = None

        # Serialize registration for the exact configured root. Preserve its
        # spelling and identity rather than creating a personal access URL.
        lockName = f"pgl.storage_backend:{postgres.databaseSchema}:{configuredRoot}"
        postgres.acquireTransactionLock(connection, lockName, description="storage-backend registration", timeoutSeconds=30.0)

        table = sql.Identifier(postgres.databaseSchema, "storage_backends")
        statement = sql.SQL("""
            SELECT backend_id, is_active
            FROM {}
            WHERE url_prefix = %s
            ORDER BY is_active DESC, backend_id ASC
            LIMIT 1
        """).format(table)
        existing = connection.execute(statement, (configuredRoot,)).fetchone()

        if existing is not None:
            backendID, isActive = existing

            if not isActive:
                raise ValueError("The configured payload destination is registered as inactive.")

            return backendID

        if kind is None:
            raise ValueError("Cannot automatically classify the configured storage protocol. Register the backend explicitly and supply backendID.")

        rootHash = hashlib.sha256(configuredRoot.encode("utf-8")).hexdigest()
        backendName = f"configured_{rootHash}"

        return cls.getOrCreateStorageBackend(connection, postgres, backendName, kind, configuredRoot)

    # +++++++++++++++++++++++ _prepareCheckpointBaseline ++++++++++++++++++++++
    # Direct callers in this module: _bindCheckpointBaseline, saveSession.
    @classmethod
    def _prepareCheckpointBaseline(cls, session, sessionID, postgres):
        """Build runtime checkpoint metadata without modifying the session.

        Caller must establish that the supplied history corresponds to the
        checkpoint being loaded or published.

        Preparation can serialize history and fail, so saves must perform it
        before commit. Assignment to the working object happens afterward.

        Does not inspect scientific payloads or clear failure restrictions.
        """
        from .pglPipeline import pglActionable

        if not isinstance(session, pglActionable):
            raise TypeError("Checkpoint baseline tracking requires a pglActionable object.")

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        historyHashes = []

        for entry in session.actionHistory:
            document = json.loads(entry.toJSON())
            content = cls.encodeRawManifest(document)
            historyHashes.append(hashlib.sha256(content).hexdigest())

        return {
            "version": 1,
            "sessionID": sessionID,
            "databaseIdentity": cls._checkpointDatabaseIdentity(postgres),
            "historyCount": len(historyHashes),
            "historyHashes": historyHashes,
        }

    # ++++++++++++++++++++++++ _bindCheckpointBaseline ++++++++++++++++++++++++
    # Direct callers in this module: loadSession.
    @classmethod
    def _bindCheckpointBaseline(cls, session, sessionID, postgres):
        """Bind a prepared checkpoint/history boundary to a working object.

        Used after verified loading. Saving prepares its baseline separately
        before commit and installs it only after successful publication.

        Never clears failed-action safety state.
        """
        baseline = cls._prepareCheckpointBaseline(session, sessionID, postgres)
        session._checkpointBaseline = baseline

    # +++++++++++++++++++++ _getCheckpointHistoryBoundary +++++++++++++++++++++
    # Direct callers in this module: _loadPipelineInput, _selectAncestralReplayInput,
    # getSessionPipeline, saveSession.
    @classmethod
    def _getCheckpointHistoryBoundary(cls, session):
        """Return the verified working-history boundary, or None if unbound.

        Reject shortened or edited baseline history rather than guessing where
        replay should begin. Appending new history entries is permitted.

        This validates history metadata only, not scientific payload contents.
        """
        from .pglPipeline import pglActionable

        if not isinstance(session, pglActionable):
            raise TypeError("Checkpoint baseline tracking requires a pglActionable object.")

        baseline = session._checkpointBaseline

        if not baseline:
            return None

        expectedFields = {"version", "sessionID", "databaseIdentity", "historyCount", "historyHashes"}

        if set(baseline) != expectedFields or type(baseline["version"]) is not int or baseline["version"] != 1:
            raise ValueError("Invalid checkpoint baseline record.")

        count = baseline["historyCount"]
        hashes = baseline["historyHashes"]

        if type(count) is not int or count < 0 or not isinstance(hashes, list) or len(hashes) != count:
            raise ValueError("Invalid checkpoint baseline history inventory.")

        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise ValueError("Invalid checkpoint baseline history fingerprint.")

        if len(session.actionHistory) < count:
            raise ValueError("Checkpoint history was shortened after loading. Cannot safely determine the replay boundary.")

        for index, expectedHash in enumerate(hashes):
            document = json.loads(session.actionHistory[index].toJSON())
            actualHash = hashlib.sha256(cls.encodeRawManifest(document)).hexdigest()

            if actualHash != expectedHash:
                raise ValueError(f"Checkpoint history entry {index} was changed after loading. Edit an extracted pipeline instead of historical records.")

        return count

    # +++++++++++++++++++++++++ _resolveCacheDirectory ++++++++++++++++++++++++
    # Direct callers in this module: loadSession.
    @staticmethod
    def _resolveCacheDirectory(cacheDirectory=None, *, settings):
        """Resolve an explicit cache path or settings.cacheDir without creating it.

        Accepts ordinary local filesystem paths, not URLs or fsspec chains.
        An invalid explicit override raises rather than falling back silently.
        """
        import os
        from pathlib import Path

        value = settings.cacheDir if cacheDirectory is None else cacheDirectory

        try: text = os.fspath(value)
        except TypeError as error: raise ValueError("Cache directory must be a local filesystem path.") from error

        if not isinstance(text, str) or not text.strip(): raise ValueError("Cache directory must be a nonempty local filesystem path.")
        if any(character in text for character in ("\x00", "\n", "\r")): raise ValueError("Cache directory must not contain NUL or newline characters.")
        if "://" in text or "::" in text: raise ValueError("Cache directory must be an ordinary local path, not a URL or fsspec chain.")

        return str(Path(text).expanduser().resolve())

    # ++++++++++++++++++++++++++++ _mneCacheContext +++++++++++++++++++++++++
    @classmethod
    def _mneCacheContext(cls, postgres, sessionID):
        """Build a caller-local MNE cache namespace for this database endpoint.

        Called by session bundle loading before installing deferred MNE
        backings. Includes SSH endpoint information so different installations
        sharing loopback PostgreSQL settings do not collide.

        Uses the existing runtime database identity policy. Different aliases,
        transports, or SSH users may produce separate cache namespaces even
        when they reach the same database.

        Does not connect, create directories, or inspect payloads.

        Older cache directories remain untouched; this namespace version may
        materialize a new cache instead of reusing an older one.
        """
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("Invalid cache session ID.")

        document = {"cacheNamespaceVersion": 2, "databaseIdentity": cls._checkpointDatabaseIdentity(postgres)}
        digest = hashlib.sha256(pglStorage.encodeJSON(document)).hexdigest()[:16]
        label = re.sub(r"[^A-Za-z0-9_-]", "_", f"{postgres.databaseName}-{postgres.databaseSchema}")[:64]

        return {"namespace": f"{label}-{digest}", "sessionID": sessionID}

    # -------------------------------------------------------------------------
    # Session loading: bundle coordination
    # -------------------------------------------------------------------------
    # Resolve catalog inventories and products before reconstructing file-backed behavioral and MNE
    # contents.

    # +++++++++++++++++++++++++++ _loadSessionBundle ++++++++++++++++++++++++++
    # Direct callers in this module: loadDerivedSession, loadRawSessionBundle.
    # +++++++++++++++++++++++++++ _loadSessionBundle +++++++++++++++++++++++++
    @classmethod
    def _loadSessionBundle(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False, maxManifestBytes=16 * 1024 * 1024, cacheDirectory=None, directLocalPolicy="materialize", origin="derived"):
        """Load a completed bundle using caller-resolved payload access.

        Called by loadDerivedSession() and loadRawSessionBundle().
        Uses the caller's SQL transaction for catalog lookup.

        Supports raw session v1 and derived behavior/session v1/v2.
        Checks manifest integrity and encoding, exact file/product links,
        registered availability, and matrix descriptors.

        Owns one access context for manifest and checkpoint-file reads.
        Successful reconstruction transfers it to the session filesystem.
        Attached matrices each retain their independently owned access context.

        Returned objects retain no SQL connection. Behavioral/MNE fields and
        scientific matrix arrays remain deferred where supported.

        On failure, closes the checkpoint access context and every matrix
        opened by this call. Does not publish records or modify payloads.
        """
        if origin not in ("raw", "derived"):
            raise ValueError("Bundle origin must be raw or derived.")

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        if type(maxManifestBytes) is not int or maxManifestBytes <= 0:
            raise ValueError("maxManifestBytes must be a positive integer.")

        if directLocalPolicy not in ("materialize", "immutable"):
            raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("SELECT origin, status, contract_version, manifest_hash FROM {} WHERE session_id = %s AND (%s OR deleted_at IS NULL)").format(sessions)
        checkpoint = connection.execute(statement, (sessionID, includeDeleted)).fetchone()

        if checkpoint is None:
            raise ValueError(f"Unknown or deleted checkpoint: {sessionID}")

        checkpointOrigin, status, contractVersion, manifestHash = checkpoint

        if status != "complete":
            raise ValueError("Only completed checkpoints can be loaded.")

        supportedContracts = {
            "raw": ("pgl.raw.session.v1",),
            "derived": (
                "pgl.derived.behavior.v1",
                "pgl.derived.behavior.v2",
                "pgl.derived.session.v1",
                "pgl.derived.session.v2",
            ),
        }

        if checkpointOrigin != origin or contractVersion not in supportedContracts[origin]:
            raise ValueError("Unsupported checkpoint origin/contract.")

        hasMNE = contractVersion in ("pgl.raw.session.v1", "pgl.derived.session.v1", "pgl.derived.session.v2")
        payloadRole = "acquisition" if origin == "raw" else "derived"

        if hasMNE and (cacheDirectory is None or not str(cacheDirectory).strip()):
            raise ValueError("MNE restoration requires an explicit cacheDirectory.")

        statement = sql.SQL("""
            SELECT link.name, link.role, content.file_id, content.sha256,
                   content.size_bytes, link.deleted_at, content.deleted_at
            FROM {} AS link
            JOIN {} AS content ON content.file_id = link.file_id
            WHERE link.session_id = %s
            ORDER BY link.name
        """).format(links, files)

        rows = connection.execute(statement, (sessionID,)).fetchall()
        records = {}

        for name, role, fileID, digest, size, linkDeleted, contentDeleted in rows:
            if not includeDeleted and (linkDeleted is not None or contentDeleted is not None):
                raise ValueError(f"Checkpoint contains a deleted file link or content: {name}")

            records[name] = {"role": role, "file_id": fileID, "sha256": digest, "size_bytes": size}

        manifestRecord = records.get("manifest.json")

        if manifestRecord is None or manifestRecord["role"] != "manifest" or manifestRecord["sha256"] != manifestHash:
            raise ValueError("Checkpoint manifest link is missing or inconsistent.")

        manifestSize = manifestRecord["size_bytes"]

        if type(manifestSize) is not int or not 0 <= manifestSize <= maxManifestBytes:
            raise ValueError("Manifest size is invalid or exceeds the configured limit.")

        fileIDs = sorted({record["file_id"] for record in records.values()})

        statement = sql.SQL("""
            SELECT location.file_id, backend.backend_id, backend.url_prefix,
                   location.object_key
            FROM {} AS location
            JOIN {} AS backend ON backend.backend_id = location.backend_id
            WHERE location.file_id = ANY(%s::bigint[])
              AND (%s OR location.deleted_at IS NULL)
              AND location.status = 'present'
              AND location.verified_at IS NOT NULL
              AND backend.is_active
            ORDER BY location.file_id, location.priority,
                     backend.backend_id, location.object_key
        """).format(locations, backends)

        locationRows = connection.execute(statement, (fileIDs, includeDeleted)).fetchall()
        locationsByFile = {}

        for fileID, backendID, root, key in locationRows:
            if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                raise ValueError("Database contains an invalid payload object key.")

            locationsByFile.setdefault(fileID, []).append({
                "backend_id": backendID,
                "url_prefix": root,
                "object_key": key,
            })

        manifestLocations = locationsByFile.get(manifestRecord["file_id"], [])

        if not manifestLocations:
            raise FileNotFoundError("Manifest has no eligible verified storage location.")

        access = pglDatabaseAccess(postgres, storageOptionsByBackend)
        matrices = {}
        session = None

        try:
            failures = []
            manifestContent = None

            for location in manifestLocations:
                backendID = location["backend_id"]

                try:
                    filesystem, root = access.resolve(backendID, location["url_prefix"])
                    path = posixpath.join(root, location["object_key"])

                    with filesystem.open(path, "rb") as source:
                        content = source.read(manifestSize + 1)

                    if len(content) != manifestSize:
                        raise ValueError("Manifest size mismatch.")

                    if hashlib.sha256(content).hexdigest() != manifestHash:
                        raise ValueError("Manifest hash mismatch.")

                except (OSError, ValueError) as error:
                    failures.append(f"backend {backendID}: {type(error).__name__}")
                    continue

                manifestContent = content
                break

            if manifestContent is None:
                raise OSError("No manifest copy could be verified: " + "; ".join(failures))

            manifest = json.loads(manifestContent.decode("utf-8"))
            cls._validateSessionBundleManifest(manifest, origin=origin)

            if manifest["contract_version"] != contractVersion:
                raise ValueError("Manifest contract differs from the database checkpoint contract.")

            if cls.encodeRawManifest(manifest) != manifestContent:
                raise ValueError("Stored manifest is not canonically encoded.")

            expected = {
                record["name"]: (
                    record["sha256"],
                    record["size_bytes"],
                    "history" if record["name"] == "history.json" else payloadRole,
                )
                for record in manifest["files"]
            }
            expected["manifest.json"] = (manifestHash, len(manifestContent), "manifest")
            actual = {name: (record["sha256"], record["size_bytes"], record["role"]) for name, record in records.items()}

            if actual != expected:
                raise ValueError("Checkpoint file links do not exactly match the manifest identities and roles.")

            fileLocations = {}

            for record in manifest["files"]:
                name = record["name"]
                candidates = locationsByFile.get(records[name]["file_id"], [])

                if not candidates:
                    raise FileNotFoundError(f"File has no eligible verified storage location: {name}")

                fileLocations[name] = {
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                    "locations": candidates,
                }

            # Empty v1 expectations still reject unexpected product links.
            # Each returned matrix owns a separate deferred-access context.
            matrices = cls._loadSessionProducts(connection, postgres, sessionID, manifest.get("products", []), storageOptionsByBackend=storageOptionsByBackend)

            cacheContext = cls._mneCacheContext(postgres, sessionID) if hasMNE else None
            session = cls._reconstructSessionBundle(manifest, fileLocations, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, cacheContext=cacheContext, origin=origin, accessContext=access, ownsAccess=True)
            session.dataMatrices = matrices

            return session

        except BaseException:
            for matrix in matrices.values():
                try:
                    matrix.close()
                except Exception:
                    pass

            if session is not None:
                try:
                    session.filesystem.close()
                except Exception:
                    pass

            # Idempotent if the reconstructed filesystem already closed it.
            access.close()
            raise

    # +++++++++++++++++++++++ _reconstructSessionBundle +++++++++++++++++++++++
    # Direct callers in this module: _loadSessionBundle, reconstructDerivedSession.
    # +++++++++++++++++++++++ _reconstructSessionBundle ++++++++++++++++++++++
    @classmethod
    def _reconstructSessionBundle(cls, manifest, fileLocations, storageOptionsByBackend=None, *, cacheDirectory=None, directLocalPolicy="materialize", cacheContext=None, origin="derived", accessContext=None, ownsAccess=False):
        """Restore checkpoint history and deferred behavioral/MNE contents.

        Called by _loadSessionBundle() and reconstructDerivedSession().
        Matrix attachments are restored separately by the database loader.

        accessContext:
            Retained caller-side payload access for registered roots.
            Database loaders supply it explicitly for both local and SSH use.

        ownsAccess:
            Whether the returned checkpoint filesystem owns that context.
            Ownership transfers only when reconstruction succeeds.
            If reconstruction fails, the caller must close its context.

        Without a context, preserve existing local/explicit-URL reconstruction
        behavior. Remote database reconstruction must supply its access context.

        Reads history and required MNE descriptor metadata. Scientific fields
        remain deferred where supported. Does not publish data, query SQL,
        or extract behavioral matrices.

        Returns a pglSession backed by pglCheckpointFilesystem.
        That filesystem must remain open while deferred readers depend on it.
        """
        from .pglSession import pglSession, pglMNE

        if directLocalPolicy not in ("materialize", "immutable"):
            raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'.")

        if type(ownsAccess) is not bool:
            raise ValueError("ownsAccess must be a boolean.")

        if ownsAccess and accessContext is None:
            raise ValueError("ownsAccess requires an access context.")

        cls._validateSessionBundleManifest(manifest, origin=origin)
        hasMNE = "mne" in manifest

        if hasMNE and (cacheDirectory is None or not str(cacheDirectory).strip()):
            raise ValueError("MNE restoration requires an explicit cacheDirectory.")

        # Borrow during construction. Transfer ownership only when every
        # reconstruction step has succeeded.
        filesystem = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend=storageOptionsByBackend, accessContext=accessContext, ownsAccess=False)

        try:
            filesystem.mneCacheContext = deepcopy(cacheContext)
            history, metadata = cls._readSessionHistory(filesystem, len(manifest["runs"]))

            if hasMNE:
                cls._verifyMNEInventory(manifest, filesystem)

            session = pglSession(filesystem=filesystem, runList=[record["path"] for record in manifest["runs"]])
            session.uuid = history.uuid
            session.actionHistory = list(history.actionHistory)

            if not session.runs and metadata is not None:
                session.experimentName = metadata["labels"][0]["experimentName"]
                session.subjectID = metadata["labels"][0]["subjectID"]

            if hasMNE:
                session.mne = pglMNE.loadBundle(manifest["mne"]["path"], filesystem=filesystem, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

            filesystem._ownsAccess = ownsAccess
            return session

        except BaseException:
            # This filesystem still borrows the supplied context. Closing it
            # disables partial reconstruction without closing a caller-owned
            # context; the enclosing loader handles that context on failure.
            filesystem.close()
            raise


    # -------------------------------------------------------------------------
    # Session loading: legacy raw behavioral archives
    # -------------------------------------------------------------------------
    # Locate and verify raw behavioral manifests and resolve their registered acquisition files.

    # ++++++++++++++++++++++++++++ loadRawManifest +++++++++++++++++++++++++++
    @classmethod
    def loadRawManifest(cls, connection, postgres, sessionID, storageOptionsByBackend=None, maxBytes=16 * 1024 * 1024, *, includeDeleted=False, accessContext=None):
        """Read and verify a completed raw behavioral checkpoint manifest.

        Called by raw session loading and raw-checkpoint duplicate detection.

        Uses the caller's SQL connection for catalog lookup.
        Payload roots resolve through caller-side database access.

        If accessContext is supplied, its caller owns it and its backend
        options. Otherwise this method creates and closes a temporary context.

        Returns the structurally validated manifest after checking byte count,
        SHA-256, contract, and canonical encoding.

        Does not modify records, reconstruct runs, or retain a SQL connection.
        includeDeleted permits inspection without changing lifecycle state.
        """
        if type(maxBytes) is not int or maxBytes <= 0:
            raise ValueError("maxBytes must be a positive integer.")

        if accessContext is not None and not isinstance(accessContext, pglDatabaseAccess):
            raise TypeError("accessContext must be a pglDatabaseAccess instance.")

        if accessContext is not None and storageOptionsByBackend:
            raise ValueError("Configure storage options on accessContext, not on both arguments.")

        located = cls.getRawManifestLocations(connection, postgres, sessionID, includeDeleted=includeDeleted)

        if located["size_bytes"] > maxBytes:
            raise ValueError("Manifest exceeds the configured size limit.")

        ownsAccess = accessContext is None
        access = pglDatabaseAccess(postgres, storageOptionsByBackend) if ownsAccess else accessContext
        failures = []

        try:
            for location in located["locations"]:
                backendID = location["backend_id"]
                key = location["object_key"]

                if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                    raise ValueError("Database contains an invalid manifest object key.")

                try:
                    filesystem, root = access.resolve(backendID, location["url_prefix"])

                    with filesystem.open(posixpath.join(root, key), "rb") as source:
                        content = source.read(located["size_bytes"] + 1)

                    if len(content) != located["size_bytes"]:
                        raise ValueError("Manifest size mismatch.")

                    if hashlib.sha256(content).hexdigest() != located["sha256"]:
                        raise ValueError("Manifest hash mismatch.")

                except (OSError, ValueError) as error:
                    failures.append(f"backend {backendID}: {type(error).__name__}")
                    continue

                # Verified bytes identify the content. A decoding failure is
                # a malformed manifest, not a reason to try an identical copy.
                manifest = json.loads(content.decode("utf-8"))

                if not isinstance(manifest, dict):
                    raise ValueError("Manifest must contain a JSON object.")

                if manifest.get("contract_version") != "pgl.raw.behavior.v1":
                    raise ValueError("Unsupported manifest contract.")

                if cls.encodeRawManifest(manifest) != content:
                    raise ValueError("Stored manifest is not in the expected canonical encoding.")

                return cls.validateRawManifest(manifest)

            raise OSError(f"No manifest copy could be verified: {'; '.join(failures)}")

        finally:
            if ownsAccess:
                access.close()

    # ++++++++++++++++++++++++ getRawManifestLocations ++++++++++++++++++++++++
    # Direct callers in this module: getRawFileLocations, loadRawManifest.
    @staticmethod
    def getRawManifestLocations(connection, postgres, sessionID, *, includeDeleted=False):
        """
        Return the manifest identity and candidate locations for a completed checkpoint.
        Deleted sessions are hidden unless includeDeleted=True.
        The override permits inspection; it does not undelete the session.
        """
        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")
        
        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("SELECT status, origin, contract_version, manifest_hash FROM {} WHERE session_id = %s AND (%s OR deleted_at IS NULL)").format(sessionsTable)
        checkpoint = connection.execute(statement, (sessionID, includeDeleted)).fetchone()

        if checkpoint is None:
            raise ValueError(f"Unknown or deleted checkpoint: {sessionID}")
        status, origin, contractVersion, manifestHash = checkpoint

        if status != "complete":
            raise ValueError("Only completed checkpoints can be loaded.")

        if origin != "raw" or contractVersion != "pgl.raw.behavior.v1":
            raise ValueError("Unsupported checkpoint contract.")

        statement = sql.SQL("""
            SELECT content.file_id, content.sha256, content.size_bytes, link.role
            FROM {} AS link
            JOIN {} AS content ON content.file_id = link.file_id
            WHERE link.session_id = %s AND link.name = 'manifest.json'
        """).format(linksTable, filesTable)
        manifest = connection.execute(statement, (sessionID,)).fetchone()

        if manifest is None or manifest[3] != "manifest" or manifest[1] != manifestHash:
            raise ValueError("Checkpoint manifest link is missing or inconsistent.")

        fileID, sha256, sizeBytes, role = manifest

        statement = sql.SQL("""
            SELECT backend.backend_id, backend.url_prefix, location.object_key
            FROM {} AS location
            JOIN {} AS backend ON backend.backend_id = location.backend_id
            WHERE location.file_id = %s
              AND location.status = 'present'
              AND location.verified_at IS NOT NULL
              AND backend.is_active
            ORDER BY location.priority ASC, backend.backend_id ASC
        """).format(locationsTable, backendsTable)
        rows = connection.execute(statement, (fileID,)).fetchall()

        if not rows:
            raise FileNotFoundError("Manifest has no verified location on an active backend.")

        return {
            "sha256": sha256,
            "size_bytes": sizeBytes,
            "locations": [
                {"backend_id": backendID, "url_prefix": root, "object_key": key}
                for backendID, root, key in rows
            ],
        }

    # ++++++++++++++++++++++++++ getRawFileLocations ++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawSession, loadRawSession.
    @classmethod
    def getRawFileLocations(cls, connection, postgres, sessionID, manifest, *, includeDeleted=False):
        """
        Return acquisition-file identities and candidate storage locations.
        Deleted sessions are hidden unless includeDeleted=True.
        The override permits inspection; it does not undelete the session.
        """
        cls.validateRawManifest(manifest)

        manifestIdentity = cls.getRawManifestLocations(connection, postgres, sessionID, includeDeleted=includeDeleted)
        manifestBytes = cls.encodeRawManifest(manifest)

        if hashlib.sha256(manifestBytes).hexdigest() != manifestIdentity["sha256"] or len(manifestBytes) != manifestIdentity["size_bytes"]:
            raise ValueError("Supplied manifest does not match the checkpoint.")

        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("""
            SELECT link.name, content.sha256, content.size_bytes, link.role
            FROM {} AS link
            JOIN {} AS content ON content.file_id = link.file_id
            WHERE link.session_id = %s
        """).format(linksTable, filesTable)

        rows = connection.execute(statement, (sessionID,)).fetchall()
        actual = {name: (digest, size, role) for name, digest, size, role in rows}

        expected = {
            record["name"]: (record["sha256"], record["size_bytes"], "acquisition")
            for record in manifest["files"]
        }
        expected["manifest.json"] = (manifestIdentity["sha256"], manifestIdentity["size_bytes"], "manifest")

        if actual != expected:
            raise ValueError("Checkpoint file links do not exactly match the manifest.")

        result = {
            record["name"]: {
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "locations": [],
            }
            for record in manifest["files"]
        }

        statement = sql.SQL("""
            SELECT link.name, backend.backend_id, backend.url_prefix,
                   location.object_key
            FROM {} AS link
            JOIN {} AS location ON location.file_id = link.file_id
            JOIN {} AS backend ON backend.backend_id = location.backend_id
            WHERE link.session_id = %s
              AND link.role = 'acquisition'
              AND location.status = 'present'
              AND location.verified_at IS NOT NULL
              AND backend.is_active
            ORDER BY link.name, location.priority ASC, backend.backend_id ASC
        """).format(linksTable, locationsTable, backendsTable)

        for name, backendID, root, key in connection.execute(statement, (sessionID,)).fetchall():
            result[name]["locations"].append({
                "backend_id": backendID,
                "url_prefix": root,
                "object_key": key,
            })

        unavailable = [name for name, record in result.items() if not record["locations"]]

        if unavailable:
            raise FileNotFoundError(f"Files lack verified active locations: {', '.join(sorted(unavailable))}")

        return result

    # ++++++++++++++++++++++++++++ readRawSubjects ++++++++++++++++++++++++++++
    # Checkpoint implementation support. Existing external callers may use this method; see its
    # docstring and this section's purpose.
    @classmethod
    def readRawSubjects(cls, manifest, checkpointFilesystem):
        """Read each run's recorded subject ID through a verified checkpoint filesystem.

        Returns a mapping from logical run paths to subject IDs.
        Does not access session caches or write to the database.
        """
        cls.validateRawManifest(manifest)
        subjectsByRun = {}

        for record in manifest["runs"]:
            runPath = record["path"]
            settingsPath = posixpath.join(runPath, "experimentSettings.json")

            with checkpointFilesystem.open(settingsPath, "rt", encoding="utf-8") as file:
                settings = json.load(file)

            if not isinstance(settings, dict):
                raise ValueError(f"Experiment settings must contain a JSON object: {settingsPath}")

            subjectID = settings.get("subjectID")

            if not isinstance(subjectID, str) or not subjectID.strip() or "\x00" in subjectID:
                raise ValueError(f"Missing or invalid recorded subject ID: {settingsPath}")

            subjectsByRun[runPath] = subjectID

        return subjectsByRun


    # -------------------------------------------------------------------------
    # Raw session publication
    # -------------------------------------------------------------------------
    # saveRawSession() coordination, acquisition preflight, source inventory, staging, and raw
    # bundle export.

    # ++++++++++++++++++++++++++++ _saveRawSession ++++++++++++++++++++++++++
    @classmethod
    def _saveRawSession(cls, connection, postgres, session, createdBy, backendID, storageOptions=None):
        """Publish or reuse a parentless raw checkpoint.

        Called by saveRawSession() using its connection and outer transaction.

        Behavioral-only sessions:
            Inventory unchanged acquisition files.
            Reuse an identical active raw checkpoint when available.
            Otherwise stage, verify, register, and complete a new checkpoint.

        MNE-containing sessions:
            Delegate to _saveRawMNECheckpoint().

        Server-local destinations resolve through caller-side local/SSH access.
        One operation-owned context is reused for behavioral staging,
        deduplication verification, and registration.

        Registered roots and object keys remain independent of caller access.
        Existing deleted checkpoints remain deleted.

        Returns the new or reused checkpoint ID.
        Does not change the working session's identity or history.

        Uses a nested SQL transaction/savepoint. The caller must commit.
        Payload writes are not rolled back; failures may leave unregistered
        files. Destination access closes on success and failure.
        """
        for label, value in (("createdBy", createdBy), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        if session.mne is not None:
            return cls._saveRawMNECheckpoint(connection, postgres, session, createdBy, backendID, storageOptions=storageOptions)

        manifest = cls.buildRawManifest(session)
        cls.validateRawManifest(manifest)
        manifestHash = cls.hashRawManifest(manifest)

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")
        usersTable = sql.Identifier(postgres.databaseSchema, "users")

        with connection.transaction(), pglDatabaseAccess(postgres, {backendID: storageOptions or {}}) as access:
            statement = sql.SQL("SELECT 1 FROM {} WHERE user_id = %s FOR SHARE").format(usersTable)

            if connection.execute(statement, (createdBy,)).fetchone() is None:
                raise ValueError(f"Unknown creator: {createdBy}")

            statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backendsTable)
            backend = connection.execute(statement, (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError(f"Storage backend is missing or inactive: {backendID}")

            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (manifestHash,))

            statement = sql.SQL("""
                SELECT session_id, status, contract_version
                FROM {}
                WHERE manifest_hash = %s
                  AND origin = 'raw'
                  AND deleted_at IS NULL
                FOR SHARE
            """).format(sessionsTable)
            existing = connection.execute(statement, (manifestHash,)).fetchone()

            if existing is not None:
                existingID, status, contractVersion = existing

                if (status, contractVersion) != ("complete", "pgl.raw.behavior.v1"):
                    raise ValueError("Matching manifest belongs to an incomplete or unsupported checkpoint.")

                # Reuse the publication context for the archived manifest.
                # Other replica backends use their own registered roots.
                archivedManifest = cls.loadRawManifest(connection, postgres, existingID, accessContext=access)

                if archivedManifest != manifest:
                    raise ValueError("Existing checkpoint manifest differs from source manifest.")

                # Preserve existing reuse semantics: file-location checks here
                # are catalog checks, not a fresh scan of every archived file.
                cls.getRawFileLocations(connection, postgres, existingID, archivedManifest)

                pglMessages.warning(f"This raw session is already saved as active checkpoint {existingID}. No new checkpoint was created; returning the existing ID. To save another raw copy, delete checkpoint {existingID} and call saveRawSession() again. To save further work as a derived checkpoint, use saveSession().", level=2)
                return existingID

            statement = sql.SQL("""
                SELECT session_id
                FROM {}
                WHERE manifest_hash = %s
                  AND origin = 'raw'
                  AND deleted_at IS NOT NULL
                ORDER BY session_id
                FOR SHARE
            """).format(sessionsTable)
            deletedMatches = connection.execute(statement, (manifestHash,)).fetchall()

            if deletedMatches:
                deletedIDs = ", ".join(str(row[0]) for row in deletedMatches)
                pglMessages.message(f"Found matching deleted raw checkpoint(s) {deletedIDs}; leaving them unchanged and saving a new checkpoint.")

            # Keep the registered root unchanged. Resolve only the caller's
            # filesystem and physical root for this publication operation.
            filesystem, root = access.resolve(backendID, backend[0])

            sessionID = cls.createRawCheckpoint(connection, postgres, createdBy)
            staged = cls.stageRawSession(session, manifest, filesystem, root, sessionID=sessionID, connection=connection, postgres=postgres, backendID=backendID)
            storedManifest = cls.stageRawManifest(manifest, filesystem, root, sessionID=sessionID)

            cls.registerStagedRawCheckpoint(connection, postgres, createdBy, backendID, manifest, staged, storedManifest, sessionID=sessionID, accessContext=access)
            cls.completeRawCheckpoint(connection, postgres, sessionID, manifest)

            return sessionID

    # +++++++++++++++++++++++++ _saveRawMNECheckpoint +++++++++++++++++++++++
    @classmethod
    def _saveRawMNECheckpoint(cls, connection, postgres, session, createdBy, backendID, storageOptions=None):
        """Publish or reuse a parentless raw checkpoint containing MNE.

        Called by _saveRawSession() for MNE-only or mixed sessions.

        Exports current MNE contents and unchanged behavioral acquisition
        files into a verified temporary bundle. Exact encoded-manifest matches
        may reuse an active completed raw checkpoint.

        New publication:
            Resolve the registered destination through caller-side access.
            Copy and verify staged files.
            Register files using the same access context.
            Validate and complete the checkpoint.

        Matching deleted checkpoints remain deleted. New checkpoints use
        separate checkpoint-specific directories.

        Owns temporary staging and a destination access context. Source
        session readers remain caller-owned and are not closed here.

        Duplicate inspection creates a private lazy session; that session's
        matrix and checkpoint resources are closed before returning.

        Returns the new or reused checkpoint ID without changing the working
        session's identity, annotations, or history.

        Uses a nested transaction/savepoint; the caller's outer transaction
        must commit. Payload writes are not transactional and failures may
        leave unregistered files. Errors propagate.
        """
        from tempfile import TemporaryDirectory

        from fsspec.implementations.local import LocalFileSystem

        from .pglSession import pglMNE

        for label, value in (("createdBy", createdBy), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        if type(session.mne) is not pglMNE:
            raise TypeError("MNE raw checkpoints require a pglMNE instance.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        users = sql.Identifier(postgres.databaseSchema, "users")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        # +++++++++++++++++++++++++++++ isLocal +++++++++++++++++++++++++++++
        def isLocal(filesystem):
            """Determine locality from the resolved filesystem, not its root."""
            protocols = filesystem.protocol
            protocols = {protocols} if isinstance(protocols, str) else set(protocols)
            return bool(protocols.intersection({"file", "local"}))

        with connection.transaction(), pglDatabaseAccess(postgres, {backendID: storageOptions or {}}) as access:
            statement = sql.SQL("SELECT 1 FROM {} WHERE user_id = %s FOR SHARE").format(users)

            if connection.execute(statement, (createdBy,)).fetchone() is None:
                raise ValueError(f"Unknown creator: {createdBy}")

            statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backends)
            backend = connection.execute(statement, (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError("Storage backend is missing or inactive.")

            destinationFS, destinationRoot = access.resolve(backendID, backend[0])
            stagingFS = LocalFileSystem(skip_instance_cache=True)

            with TemporaryDirectory(prefix="pgl-raw-session-") as temporary:
                stagingRoot, manifest = cls.saveRawSessionFiles(session, Path(temporary) / "bundle", filesystem=stagingFS)
                manifestBytes = cls.encodeRawManifest(manifest)
                manifestHash = hashlib.sha256(manifestBytes).hexdigest()

                connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (manifestHash,))

                statement = sql.SQL("""
                    SELECT session_id, status, contract_version
                    FROM {}
                    WHERE origin = 'raw'
                      AND manifest_hash = %s
                      AND deleted_at IS NULL
                    FOR SHARE
                """).format(sessions)
                existing = connection.execute(statement, (manifestHash,)).fetchone()

                if existing is not None:
                    existingID, status, contract = existing

                    if (status, contract) != ("complete", "pgl.raw.session.v1"):
                        raise ValueError("Matching raw manifest belongs to an unsupported or incomplete checkpoint.")

                    # The loader creates its own retained contexts. This is
                    # private inspection, not a session returned to the caller.
                    inspected = None

                    try:
                        inspected = cls.loadRawSessionBundle(connection, postgres, existingID, storageOptionsByBackend={backendID: storageOptions or {}}, cacheDirectory=Path(temporary) / "unused-cache")

                    finally:
                        if inspected is not None:
                            try:
                                for matrix in inspected.dataMatrices.values():
                                    try:
                                        matrix.close()
                                    except Exception as error:
                                        pglMessages.warning(f"Could not close a duplicate-inspection matrix ({type(error).__name__}).", level=1)
                            finally:
                                inspected.filesystem.close()

                    pglMessages.warning(f"This raw session is already saved as active checkpoint {existingID}. No new checkpoint was created; returning the existing ID. To save another raw copy, delete checkpoint {existingID} and call saveRawSession() again. To save further work as a derived checkpoint, use saveSession().", level=2)
                    return existingID

                statement = sql.SQL("""
                    SELECT session_id
                    FROM {}
                    WHERE origin = 'raw'
                      AND manifest_hash = %s
                      AND deleted_at IS NOT NULL
                    ORDER BY session_id
                    FOR SHARE
                """).format(sessions)
                deletedMatches = connection.execute(statement, (manifestHash,)).fetchall()

                if deletedMatches:
                    deletedIDs = ", ".join(str(row[0]) for row in deletedMatches)
                    pglMessages.message(f"Found matching deleted raw checkpoint(s) {deletedIDs}; leaving them unchanged and saving a new checkpoint.")

                modalities = (["behavior"] if manifest["runs"] else []) + ["mne"]

                statement = sql.SQL("""
                    INSERT INTO {} (
                        origin, kind, ingest_method, modalities, status,
                        contract_version, created_by, started_at
                    )
                    VALUES (
                        'raw', 'acquired', 'manual', %s::text[], 'in_progress',
                        'pgl.raw.session.v1', %s, now()
                    )
                    RETURNING session_id
                """).format(sessions)
                sessionID = connection.execute(statement, (modalities, createdBy)).fetchone()[0]

                objectPrefix = f"session{sessionID:06d}"
                destination = posixpath.join(destinationRoot, objectPrefix)

                # Preserve existing overlap checks. Never interpret an SSH
                # destination using client-local Path.resolve().
                for run in session.runs or []:
                    if run.filesystem is None:
                        raise ValueError("Source run has no filesystem.")

                    if isLocal(destinationFS) and isLocal(run.filesystem):
                        target = Path(destination).expanduser().resolve()
                        source = Path(run.fullDataPath).expanduser().resolve()

                        if target == source or source in target.parents:
                            raise ValueError("Checkpoint destination must be outside source run directories.")

                    elif destinationFS == run.filesystem:
                        target = posixpath.normpath(destination)
                        source = posixpath.normpath(run.fullDataPath)

                        if target == source or target.startswith(source.rstrip("/") + "/"):
                            raise ValueError("Checkpoint destination must be outside source run directories.")

                if destinationFS.exists(destination):
                    raise FileExistsError(destination)

                destinationFS.makedirs(destination, exist_ok=False)

                for directory in manifest["directories"]:
                    destinationFS.makedirs(posixpath.join(destination, directory), exist_ok=True)

                records = [
                    *manifest["files"],
                    {"name": "manifest.json", "sha256": manifestHash, "size_bytes": len(manifestBytes)},
                ]

                for record in records:
                    pglStorage.copyVerifiedFile(stagingFS, posixpath.join(stagingRoot, record["name"]), destinationFS, destinationRoot, record, objectKey=posixpath.join(objectPrefix, record["name"]))

                cls._registerSessionBundleFiles(connection, postgres, sessionID, backendID, manifest, objectPrefix=objectPrefix, origin="raw", accessContext=access)
                cls._completeSessionBundleCheckpoint(connection, postgres, sessionID, manifest, origin="raw")

                return sessionID

    # +++++++++++++++++++++++++ _checkRawSessionLabels ++++++++++++++++++++++++
    # Direct callers in this module: saveRawSession.
    @classmethod
    def _checkRawSessionLabels(cls, session, *, allowMissingMetadata=False):
        """Check working-session safety and required raw acquisition labels.

        Called by the public raw saver before configuration, SQL, or export.
        allowMissingMetadata bypasses missing labels only, never failed-action
        safety or malformed/unreadable acquisition settings.
        """
        if type(allowMissingMetadata) is not bool:
            raise ValueError("allowMissingMetadata must be a boolean.")

        pglStorage.requireSafeSave(session)

        labels = session.getSessionLabels(acquisition=True)
        missing = []

        for record in labels:
            fields = [name for name in ("experimentName", "subjectID") if record[name] is None]

            if fields:
                owner = "session" if record["runIndex"] is None else f"run {record['runIndex']}"
                missing.append(f"{owner}: {', '.join(fields)}")

        if not missing:
            return True

        if session.runs:
            instructions = "Set experimentName and subjectID in the affected runs' saved acquisition experimentSettings.json. In-memory edits alone are not archived by saveRawSession()."
        else:
            instructions = "Set session.experimentName and session.subjectID before saving."

        if allowMissingMetadata:
            pglMessages.warning("Saving raw session with missing metadata because allowMissingMetadata=True: " + "; ".join(missing) + ". Unset values will display as Unknown.", level=0)
            return True

        pglMessages.warning("Raw session was not saved. Missing metadata: " + "; ".join(missing) + ". " + instructions + " To intentionally save without these labels, use saveRawSession(allowMissingMetadata=True).", level=0)
        return False

    # ++++++++++++++++++++++++++++ buildRawManifest +++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawSession, saveRawSessionFiles, stageRawSession.
    @classmethod
    def buildRawManifest(cls, session):
        """Describe saved behavioral runs without copying files or publishing a checkpoint."""
        if not session.runs:
            raise ValueError("A raw checkpoint must contain at least one run.")

        if session.mne is not None:
            raise NotImplementedError("This pilot archives behavioral run directories only; attached MNE data is not yet supported.")

        runs = []
        directories = []
        files = []

        for index, run in enumerate(session.runs):
            runName = f"runs/{index:06d}"
            inventory = cls.validateRawRun(run)

            runs.append({"index": index, "path": runName})
            directories.append(runName)
            directories.extend(posixpath.join(runName, directory) for directory in inventory["directories"])

            for relativePath in inventory["files"]:
                sourcePath = posixpath.join(run.fullDataPath, relativePath)
                identity = pglStorage.hashFile(run.filesystem, sourcePath)
                files.append({"name": posixpath.join(runName, relativePath), **identity})

        return {
            "contract_version": "pgl.raw.behavior.v1",
            "runs": runs,
            "directories": sorted(["runs", *directories]),
            "files": sorted(files, key=lambda record: record["name"]),
        }

    # +++++++++++++++++++++++++++++ validateRawRun ++++++++++++++++++++++++++++
    # Direct callers in this module: buildRawManifest.
    @classmethod
    def validateRawRun(cls, run):
        """Validate required acquisition files and return the run inventory."""
        from .pglExperiment import pglTaskBase

        inventory = cls.inventoryRawRun(run)
        files = set(inventory["files"])
        directories = set(inventory["directories"])

        requiredFiles = {"experimentSettings.json", "settings.json", "state.json", "data.json", "pgl.json"}
        missing = requiredFiles - files

        if missing:
            raise ValueError(f"Run is missing required files: {', '.join(sorted(missing))}")

        settingsPath = posixpath.join(run.fullDataPath, "experimentSettings.json")
        with run.filesystem.open(settingsPath, "rt", encoding="utf-8") as file:
            settings = json.load(file)

        if not isinstance(settings, dict):
            raise ValueError("experimentSettings.json must contain a JSON object.")

        taskNames = settings.get("tasks")
        if not isinstance(taskNames, list) or any(not isinstance(name, str) or not name.strip() for name in taskNames):
            raise ValueError("experimentSettings.json must contain a list of nonempty task names.")

        for index, name in enumerate(taskNames):
            if "/" in name or "\\" in name or "\x00" in name:
                raise ValueError(f"Task name must not contain path separators or NUL: {name!r}")

            taskDirectory = pglTaskBase.getTaskDirectoryName(taskID=index, taskName=name)
            requiredTaskFiles = {posixpath.join(taskDirectory, filename) for filename in ("settings.json", "state.json", "data.json")}
            missing = requiredTaskFiles - files

            if missing:
                raise ValueError(f"Run is missing required task files: {', '.join(sorted(missing))}")

            parametersDirectory = posixpath.join(taskDirectory, "parameters")
            if parametersDirectory not in directories:
                raise ValueError(f"Run is missing parameters directory: {parametersDirectory}")

        return inventory

    # ++++++++++++++++++++++++++++ inventoryRawRun ++++++++++++++++++++++++++++
    # Direct callers in this module: validateRawRun.
    @staticmethod
    def inventoryRawRun(run):
        """List source files and directories without loading or changing their contents.

        Paths in the returned inventory are relative to the run directory.
        This is a structural inventory, not yet a verified checkpoint.
        """
        filesystem = run.filesystem
        root = run.fullDataPath

        if filesystem is None:
            raise ValueError("Run has no resolved filesystem.")

        if not filesystem.isdir(root):
            raise NotADirectoryError(f"Run directory does not exist: {root}")

        directories = set()
        files = set()

        # ++++++++++++++++++++++++++++++ relativePath +++++++++++++++++++++++++++++
        # Local implementation within inventoryRawRun.
        def relativePath(path):
            relative = posixpath.relpath(path, root)
            parts = relative.split("/")

            if relative.startswith("/") or "\\" in relative or any(part in {"", ".", ".."} for part in parts):
                raise ValueError(f"Invalid relative path in run inventory: {relative!r}")

            return relative

        for directory, directoryNames, fileNames in filesystem.walk(root, detail=False, on_error="raise"):
            for name in directoryNames:
                directories.add(relativePath(posixpath.join(directory, name)))

            for name in fileNames:
                files.add(relativePath(posixpath.join(directory, name)))

        return {"directories": sorted(directories), "files": sorted(files)}

    # ++++++++++++++++++++++++++++ stageRawSession ++++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawSession.
    @classmethod
    def stageRawSession(cls, session, manifest, destinationFilesystem, destinationRoot, *, sessionID, connection, postgres, backendID):
        """Store each distinct payload once on the destination backend.

        Reuse registered copies only after verifying their bytes.
        Repeated content within this save reuses the same verified payload.

        Local filesystems get readable hard-link aliases where supported.
        Other filesystems retain only the canonical copy; logical filenames
        resolve through the manifest and registered content locations.

        Must run inside the caller's transaction.
        Never overwrites payloads. Failed saves may leave unregistered files.
        """
        import errno
        import os
        import stat

        if isinstance(sessionID, bool) or not isinstance(sessionID, int) or sessionID < 0:
            raise ValueError("sessionID must be a nonnegative integer.")

        cls.validateRawManifest(manifest)

        if cls.buildRawManifest(session) != manifest:
            raise ValueError("Source session no longer matches the supplied manifest.")

        # ++++++++++++++++++++++++++++++++ isLocal ++++++++++++++++++++++++++++++++
        # Local implementation within stageRawSession.
        def isLocal(filesystem):
            protocols = filesystem.protocol
            protocols = {protocols} if isinstance(protocols, str) else set(protocols)
            return bool(protocols.intersection({"file", "local"}))

        localDestination = isLocal(destinationFilesystem)
        sessionKey = f"session{sessionID:06d}"
        sessionRoot = posixpath.join(destinationRoot, sessionKey)

        # Check before creating any destination paths.
        for run in session.runs:
            if localDestination and isLocal(run.filesystem):
                destination = Path(sessionRoot).expanduser().resolve()
                source = Path(run.fullDataPath).expanduser().resolve()

                if destination == source or source in destination.parents:
                    raise ValueError("Checkpoint destination must be outside source run directories.")

            elif destinationFilesystem == run.filesystem:
                destination = posixpath.normpath(sessionRoot)
                source = posixpath.normpath(run.fullDataPath)

                if destination == source or destination.startswith(source.rstrip("/") + "/"):
                    raise ValueError("Checkpoint destination must be outside source run directories.")

        # Stable lock ordering avoids deadlocks between cooperating saves.
        # Locks remain held until the outer transaction finishes.
        for digest in sorted({record["sha256"] for record in manifest["files"]}):
            lockName = f"pgl.payload:{postgres.databaseSchema}:{backendID}:{digest}"
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lockName,))

        if destinationFilesystem.exists(sessionRoot):
            raise FileExistsError(f"Checkpoint destination already exists; it will not be overwritten: {sessionRoot}")

        destinationFilesystem.makedirs(destinationRoot, exist_ok=True)
        destinationFilesystem.makedirs(sessionRoot, exist_ok=False)

        runsByPath = {record["path"]: record for record in manifest["runs"]}

        # ++++++++++++++++++++++++++++++ physicalKey ++++++++++++++++++++++++++++++
        # Local implementation within stageRawSession.
        def physicalKey(logicalName):
            parts = logicalName.split("/", 2)
            runRecord = runsByPath["/".join(parts[:2])]
            runKey = posixpath.join(sessionKey, f"run{runRecord['index']:05d}")
            return posixpath.join(runKey, parts[2]) if len(parts) == 3 else runKey

        # Physical empty directories are a local browsing convenience.
        # The manifest preserves logical directories on every backend.
        if localDestination:
            for directory in manifest["directories"]:
                if directory != "runs":
                    destinationFilesystem.makedirs(posixpath.join(destinationRoot, physicalKey(directory)), exist_ok=True)

        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")

        # Canonical verified payload per hash for this save.
        # This cache also covers content not yet registered in SQL.
        verifiedByHash = {}
        staged = []

        unsupportedLinkErrors = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }

        # ++++++++++++++++++++++++++++ readableLocation +++++++++++++++++++++++++++
        # Local implementation within stageRawSession.
        def readableLocation(canonical, targetKey):
            if not localDestination or canonical["object_key"] == targetKey:
                return canonical

            sourcePath = posixpath.join(destinationRoot, canonical["object_key"])
            targetPath = posixpath.join(destinationRoot, targetKey)
            destinationFilesystem.makedirs(posixpath.dirname(targetPath), exist_ok=True)

            # Hard links are a filesystem metadata operation, not payload I/O.
            # Do not create an alias to a symlink or special file.
            sourceStat = os.lstat(sourcePath)

            if not stat.S_ISREG(sourceStat.st_mode):
                pglMessages.oneTimeWarning("A canonical payload is not a regular local file. Reusing its registered location without creating a hard link.", level=0)
                return canonical

            try:
                os.link(sourcePath, targetPath, follow_symlinks=False)
            except FileExistsError:
                raise
            except OSError as error:
                if error.errno not in unsupportedLinkErrors:
                    raise

                pglMessages.oneTimeWarning("Hard links are unavailable for this storage location. Reusing verified payload locations without creating duplicate copies.", level=0)
                return canonical

            targetStat = os.lstat(targetPath)

            if (
                not stat.S_ISREG(targetStat.st_mode)
                or targetStat.st_dev != sourceStat.st_dev
                or targetStat.st_ino != sourceStat.st_ino
            ):
                raise OSError(f"Hard-link identity check failed: {targetKey}")

            return {
                "object_key": targetKey,
                "sha256": canonical["sha256"],
                "size_bytes": canonical["size_bytes"],
            }

        for record in manifest["files"]:
            digest = record["sha256"]
            canonical = verifiedByHash.get(digest)

            if canonical is not None and canonical["size_bytes"] != record["size_bytes"]:
                raise ValueError("Manifest contains conflicting sizes for the same SHA-256.")

            if canonical is None:
                statement = sql.SQL("""
                    SELECT location.object_key, content.size_bytes
                    FROM {} AS content
                    JOIN {} AS location ON location.file_id = content.file_id
                    WHERE content.sha256 = %s
                      AND location.backend_id = %s
                      AND location.status = 'present'
                      AND location.verified_at IS NOT NULL
                    ORDER BY location.priority, location.object_key
                """).format(filesTable, locationsTable)
                candidates = connection.execute(statement, (digest, backendID)).fetchall()
                failures = []

                for key, registeredSize in candidates:
                    if registeredSize != record["size_bytes"]:
                        raise ValueError("Registered content size conflicts with the manifest.")

                    if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                        raise ValueError("Database contains an invalid payload object key.")

                    try:
                        actual = pglStorage.hashFile(destinationFilesystem, posixpath.join(destinationRoot, key))

                        if actual != {"sha256": digest, "size_bytes": record["size_bytes"]}:
                            raise ValueError("Registered payload integrity check failed.")

                    except (OSError, ValueError) as error:
                        failures.append(type(error).__name__)
                        continue

                    canonical = {"object_key": key, **actual}
                    break

                if candidates and canonical is None:
                    raise OSError(f"No registered copy could be verified for SHA-256 {digest}: {', '.join(failures)}")

                if canonical is None:
                    parts = record["name"].split("/", 2)
                    runRecord = runsByPath["/".join(parts[:2])]
                    run = session.runs[runRecord["index"]]
                    sourcePath = posixpath.join(run.fullDataPath, parts[2])

                    canonical = pglStorage.copyVerifiedFile(run.filesystem, sourcePath, destinationFilesystem, destinationRoot, record, objectKey=physicalKey(record["name"]))

                verifiedByHash[digest] = canonical

            stored = readableLocation(canonical, physicalKey(record["name"]))
            staged.append({"name": record["name"], **stored})

        return staged

    # ++++++++++++++++++++++++++++ stageRawManifest +++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawSession.
    @classmethod
    def stageRawManifest(cls, manifest, destinationFilesystem, destinationRoot, *, sessionID):
        """Write and verify sessionNNNNNN/manifest.json without overwriting."""
        if isinstance(sessionID, bool) or not isinstance(sessionID, int) or sessionID < 0:
            raise ValueError("sessionID must be a nonnegative integer.")

        cls.validateRawManifest(manifest)
        manifestBytes = cls.encodeRawManifest(manifest)
        expectedHash = hashlib.sha256(manifestBytes).hexdigest()
        expectedSize = len(manifestBytes)

        objectKey = f"session{sessionID:06d}/manifest.json"
        destinationPath = posixpath.join(destinationRoot, objectKey)

        with destinationFilesystem.open(destinationPath, "xb") as file:
            file.write(manifestBytes)

        stored = pglStorage.hashFile(destinationFilesystem, destinationPath)

        if stored["sha256"] != expectedHash or stored["size_bytes"] != expectedSize:
            raise ValueError(f"Stored manifest verification failed: {objectKey}")

        return {"object_key": objectKey, **stored}

    # ++++++++++++++++++++++++++ saveRawSessionFiles ++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawMNECheckpoint.
    @classmethod
    def saveRawSessionFiles(cls, session, dataPath, filesystem=None):
        """Export MNE and optional unchanged behavioral acquisition files.

        Includes resolved acquisition labels in hashed history.json.
        Does not register or publish a database checkpoint.
        Partial files may remain if export fails.
        """
        from .pglSession import pglSession, pglMNE

        if type(session.mne) is not pglMNE:
            raise TypeError("Raw session bundle export requires a pglMNE instance.")

        if dataPath is None or not str(dataPath).strip():
            raise ValueError("A destination is required.")

        if filesystem is None:
            filesystem, destination = url_to_fs(str(dataPath))
        else:
            destination = str(dataPath)

        if filesystem.exists(destination):
            raise FileExistsError(destination)

        behavior = pglSession()
        behavior.runs = list(session.runs or [])
        behavior.mne = None
        behaviorManifest = cls.buildRawManifest(behavior) if behavior.runs else None
        historyBytes = cls._sessionHistoryBytes(session, acquisition=True)

        filesystem.makedirs(destination, exist_ok=False)
        directories = set()
        files = []
        runs = []

        if behaviorManifest is not None:
            directories.update(behaviorManifest["directories"])
            runs = deepcopy(behaviorManifest["runs"])

            for directory in sorted(directories):
                filesystem.makedirs(posixpath.join(destination, directory), exist_ok=True)

            for record in behaviorManifest["files"]:
                parts = record["name"].split("/", 2)
                run = behavior.runs[int(parts[1])]
                sourcePath = posixpath.join(run.fullDataPath, parts[2])
                pglStorage.copyVerifiedFile(run.filesystem, sourcePath, filesystem, destination, record, objectKey=record["name"])
                files.append(dict(record))

            if cls.buildRawManifest(behavior) != behaviorManifest:
                raise ValueError("Behavioral acquisition changed during raw export.")

        mneRoot = posixpath.join(destination, "mne")
        session.mne.saveBundle(mneRoot, filesystem=filesystem)
        directories.add("mne")

        for current, directoryNames, fileNames in filesystem.walk(mneRoot, detail=False, on_error="raise"):
            relative = posixpath.relpath(current, destination)

            for name in directoryNames:
                directories.add(posixpath.join(relative, name))

            for name in fileNames:
                path = posixpath.join(current, name)
                files.append({"name": posixpath.join(relative, name), **pglStorage.hashFile(filesystem, path)})

        historyPath = posixpath.join(destination, "history.json")

        with filesystem.open(historyPath, "xb") as target:
            if target.write(historyBytes) != len(historyBytes):
                raise OSError("Incomplete raw session history write.")

        historyIdentity = {"sha256": hashlib.sha256(historyBytes).hexdigest(), "size_bytes": len(historyBytes)}

        if pglStorage.hashFile(filesystem, historyPath) != historyIdentity:
            raise ValueError("Raw session history verification failed.")

        files.append({"name": "history.json", **historyIdentity})
        manifest = {"contract_version": "pgl.raw.session.v1", "runs": runs, "mne": {"path": "mne"}, "directories": sorted(directories), "files": sorted(files, key=lambda record: record["name"])}
        cls.validateRawSessionManifest(manifest)

        class DestinationView:
            # ++++++++++++++++++++++++++++++++++ open +++++++++++++++++++++++++++++++++
            # Local implementation within saveRawSessionFiles.
            def open(self, path, mode="rb", **kwargs):
                return filesystem.open(posixpath.join(destination, path), mode=mode, **kwargs)

        view = DestinationView()
        cls._verifyMNEInventory(manifest, view)
        _, metadata = cls._readSessionHistory(view, len(runs))

        if runs and metadata != cls._behaviorLabelMetadata(manifest, view):
            raise ValueError("Raw history labels differ from the archived behavioral settings.")

        content = cls.encodeRawManifest(manifest)
        manifestPath = posixpath.join(destination, "manifest.json")

        with filesystem.open(manifestPath, "xb") as target:
            if target.write(content) != len(content):
                raise OSError("Incomplete raw session manifest write.")

        expected = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}

        if pglStorage.hashFile(filesystem, manifestPath) != expected:
            raise ValueError("Raw session manifest verification failed.")

        return destination, manifest


    # -------------------------------------------------------------------------
    # Derived session publication
    # -------------------------------------------------------------------------
    # saveSession() coordination and checkpoint-specific temporary bundle export.

    # ++++++++++++++++++++++++++ _saveDerivedSession ++++++++++++++++++++++++
    @classmethod
    def _saveDerivedSession(cls, connection, postgres, session, createdBy, backendID, *, parentSessionID, checkpointName=None, description=None, ingestMethod="manual", storageOptions=None):
        """Publish current contents and matrices as a derived v2 checkpoint.

        Called by saveSession(), which owns the outer SQL transaction and
        updates working-session bookkeeping only after successful commit.

        Resolves the registered destination through one operation-owned
        pglDatabaseAccess context. Server-local roots use local or SSH access
        according to the caller's configuration.

        The same destination context is used for:
            Working matrix publication.
            Checkpoint file transfers.
            Archived metadata reads.
            Stored-file verification and location registration.

        Registered roots and relative keys remain unchanged by access
        resolution. Caller-specific SSH URLs and credentials are not stored.

        Working matrices publish fresh Zarr revisions. Named revision
        references enter the hashed manifest and session_products.

        Without attached matrices, an identical active derived v2 parent may
        be reused after checking metadata, registered files, and empty products.

        Uses a nested SQL transaction/savepoint. The outer transaction must
        commit for publication to persist. Temporary staging and owned SSH
        connections are cleaned up when this method exits.

        Filesystem writes are not transactional. Failure may leave orphaned
        payloads even when SQL registration rolls back.

        Returns the new or reused checkpoint ID.

        Does not change working session identity, annotations, matrix revision
        IDs, or action history. Concurrent working-object mutation is unsupported.
        """
        from tempfile import TemporaryDirectory

        from fsspec.implementations.local import LocalFileSystem

        from .pglSession import pglMNE

        for label, value in (("createdBy", createdBy), ("backendID", backendID), ("parentSessionID", parentSessionID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        for label, value in (("checkpointName", checkpointName), ("description", description)):
            if value is not None and (not isinstance(value, str) or "\x00" in value):
                raise ValueError(f"{label} must be a string without NUL, or None.")

        if not isinstance(ingestMethod, str) or ingestMethod not in {"manual", "automated"}:
            raise ValueError("ingestMethod must be manual or automated.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        products = cls._snapshotSessionMatrices(session)
        runs = list(session.runs or [])
        hasMNE = session.mne is not None

        if not runs and not hasMNE:
            raise ValueError("A derived checkpoint must contain behavioral runs or MNE contents.")

        if hasMNE and type(session.mne) is not pglMNE:
            raise TypeError("MNE checkpoints currently support pglMNE, not subclasses.")

        # Reject unsupported matrix encodings before permanent payload writes.
        for matrix in products.values():
            pglStorage.requireSafeSave(matrix)
            pglStorage.validateDataMatrixEncoding(matrix.requireData())

        contractVersion = "pgl.derived.session.v2" if hasMNE else "pgl.derived.behavior.v2"
        modalities = (["behavior"] if runs else []) + (["mne"] if hasMNE else [])

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        users = sql.Identifier(postgres.databaseSchema, "users")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")

        # ++++++++++++++++++++++++++++ isLocal ++++++++++++++++++++++++++++++
        def isLocal(filesystem):
            """Determine actual caller-side filesystem locality."""
            protocols = filesystem.protocol
            protocols = {protocols} if isinstance(protocols, str) else set(protocols)
            return bool(protocols.intersection({"file", "local"}))

        # The access context exits before this savepoint exits. The outer
        # saveSession transaction remains responsible for final commit.
        with connection.transaction(), pglDatabaseAccess(postgres, {backendID: storageOptions or {}}) as access:
            statement = sql.SQL("SELECT 1 FROM {} WHERE user_id = %s FOR SHARE").format(users)

            if connection.execute(statement, (createdBy,)).fetchone() is None:
                raise ValueError(f"Unknown creator: {createdBy}")

            statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backends)
            backend = connection.execute(statement, (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError(f"Storage backend is missing or inactive: {backendID}")

            registeredRoot = backend[0]

            statement = sql.SQL("""
                SELECT origin, status, contract_version, manifest_hash,
                       description, ingest_method, created_by, modalities,
                       deleted_at, checkpoint_name
                FROM {}
                WHERE session_id = %s
                FOR SHARE
            """).format(sessions)

            parent = connection.execute(statement, (parentSessionID,)).fetchone()

            if parent is None or parent[1] != "complete" or parent[8] is not None:
                raise ValueError("Parent must be a completed, active checkpoint.")

            # The registered root is not edited. Only physical access is
            # selected according to this caller's database configuration.
            destinationFS, destinationRoot = access.resolve(backendID, registeredRoot)
            stagingFS = LocalFileSystem(skip_instance_cache=True)

            # Matrix publication is not transactional. Later failures may
            # leave orphaned bundles, while their catalog rows roll back.
            pglMessages.message(f"Saving session: publishing {len(products)} attached matrix product(s)...")
            productRecords = cls._publishSessionMatrices(connection, postgres, session, products, createdBy=createdBy, backendID=backendID, filesystem=destinationFS, destinationRoot=destinationRoot)

            with TemporaryDirectory(prefix="pgl-derived-session-") as temporary:
                pglMessages.message("Saving session: exporting current contents to temporary staging...")
                stagingRoot, manifest = cls.saveDerivedSessionFiles(session, dataPath=Path(temporary) / "bundle", filesystem=stagingFS, productRecords=productRecords)

                cls._requireSessionMatricesUnchanged(session, products)

                if manifest["contract_version"] != contractVersion:
                    raise ValueError("Exported manifest contract differs from the expected checkpoint contract.")

                if manifest["products"] != productRecords:
                    raise ValueError("Exported manifest products differ from the published matrix references.")

                manifestBytes = cls.encodeRawManifest(manifest)
                manifestHash = hashlib.sha256(manifestBytes).hexdigest()

                pglMessages.message("Saving session: comparing contents with the parent checkpoint...")

                sameParent = parent[0] == "derived" and parent[2] == contractVersion and parent[3] == manifestHash
                sameMetadata = parent[4] == description and parent[5] == ingestMethod and parent[6] == createdBy and sorted(parent[7]) == sorted(modalities) and parent[9] == checkpointName

                # Attached working matrices always receive new revisions.
                # Parent reuse applies only when there are no products.
                if not products and sameParent and sameMetadata:
                    cls._verifySessionProducts(connection, postgres, parentSessionID, [])

                    expected = {
                        record["name"]: (
                            record["sha256"],
                            record["size_bytes"],
                            "history" if record["name"] == "history.json" else "derived",
                        )
                        for record in manifest["files"]
                    }
                    expected["manifest.json"] = (manifestHash, len(manifestBytes), "manifest")

                    statement = sql.SQL("""
                        SELECT link.name,
                               content.sha256,
                               content.size_bytes,
                               link.role,
                               (
                                   link.deleted_at IS NULL
                                   AND content.deleted_at IS NULL
                                   AND EXISTS (
                                       SELECT 1
                                       FROM {} AS location
                                       WHERE location.file_id = content.file_id
                                         AND location.backend_id = %s
                                         AND location.deleted_at IS NULL
                                         AND location.status = 'present'
                                         AND location.verified_at IS NOT NULL
                                   )
                               ) AS available
                        FROM {} AS link
                        JOIN {} AS content ON content.file_id = link.file_id
                        WHERE link.session_id = %s
                        ORDER BY link.name
                    """).format(locations, links, files)

                    rows = connection.execute(statement, (backendID, parentSessionID)).fetchall()
                    actual = {name: (digest, size, role) for name, digest, size, role, available in rows}

                    if actual != expected:
                        raise ValueError("Parent checkpoint file links do not match its manifest identity; it cannot be reused.")

                    # Preserve existing reuse semantics: availability here is
                    # catalog metadata, not a new byte scan of the parent.
                    if all(available for name, digest, size, role, available in rows):
                        cls._requireSessionMatricesUnchanged(session, products)
                        pglMessages.warning(f"Session contents and save metadata match checkpoint {parentSessionID}. No new checkpoint was created; returning the existing ID.", level=2)
                        return parentSessionID

                    pglMessages.message("Saving session: matching parent lacks eligible files on the requested backend; creating a new checkpoint.")

                pglMessages.message("Saving session: allocating a new derived checkpoint...")
                sessionID = cls.createDerivedCheckpoint(connection, postgres, createdBy, parentSessionID, checkpointName=checkpointName, description=description, ingestMethod=ingestMethod, contractVersion=contractVersion, modalities=modalities)

                objectPrefix = f"session{sessionID:06d}"
                destination = posixpath.join(destinationRoot, objectPrefix)

                # Preserve existing source/destination overlap checks.
                # Server-local paths accessed through SSH must not be
                # interpreted with this client's Path.resolve().
                for run in runs:
                    if run.filesystem is None:
                        raise ValueError("Source run has no resolved filesystem.")

                    if isLocal(destinationFS) and isLocal(run.filesystem):
                        target = Path(destination).expanduser().resolve()
                        source = Path(run.fullDataPath).expanduser().resolve()

                        if target == source or source in target.parents:
                            raise ValueError("Derived destination must be outside source run directories.")

                    elif destinationFS == run.filesystem:
                        target = posixpath.normpath(destination)
                        source = posixpath.normpath(run.fullDataPath)

                        if target == source or target.startswith(source.rstrip("/") + "/"):
                            raise ValueError("Derived destination must be outside source run directories.")

                if destinationFS.exists(destination):
                    raise FileExistsError(f"Destination already exists: {destination}")

                destinationFS.makedirs(destination, exist_ok=False)

                for directory in manifest["directories"]:
                    destinationFS.makedirs(posixpath.join(destination, directory), exist_ok=True)

                pglMessages.message("Saving session: writing and verifying checkpoint payloads...")
                fileRecords = [
                    *manifest["files"],
                    {"name": "manifest.json", "sha256": manifestHash, "size_bytes": len(manifestBytes)},
                ]

                for record in fileRecords:
                    pglStorage.copyVerifiedFile(stagingFS, posixpath.join(stagingRoot, record["name"]), destinationFS, destinationRoot, record, objectKey=posixpath.join(objectPrefix, record["name"]))

                pglMessages.message("Saving session: registering checkpoint files and matrix links...")

                # Registration borrows this same context and therefore reuses
                # the resolved destination connection. Catalog roots remain
                # the original backend roots.
                cls.registerDerivedSessionFiles(connection, postgres, sessionID, backendID, manifest, objectPrefix=objectPrefix, accessContext=access)
                cls._registerSessionProducts(connection, postgres, sessionID, productRecords)

                cls._requireSessionMatricesUnchanged(session, products)

                for matrix in products.values():
                    pglStorage.requireSafeSave(matrix)

                pglMessages.message("Saving session: validating checkpoint completion...")
                cls.completeDerivedCheckpoint(connection, postgres, sessionID, manifest)

                return sessionID

    # ++++++++++++++++++++++++ saveDerivedSessionFiles ++++++++++++++++++++++++
    # Direct callers in this module: _saveDerivedSession.
    @classmethod
    def saveDerivedSessionFiles(cls, session, dataPath, filesystem=None, filesystemPrefix=None, *, productRecords=None):
        """Export session files and a manifest containing matrix references.

        Behavioral/MNE files and history are written into a fresh destination.
        Matrix arrays remain in separately published Zarr bundles.

        productRecords:
            None: legacy v1 export, permitted only without attached matrices.
            List: v2 export with explicit named matrix revision references,
                  including an explicitly empty collection.

        The enclosing publication transaction supplies productRecords after
        publishing matrices. This function validates record structure and
        attachment names; database correspondence is checked at completion.

        Writes files and manifest.json, but does not register database rows,
        publish matrix payloads, or modify working session attachments.
        """
        from .pglStorage import pglStorage
        from .pglSession import pglMNE

        runs = list(session.runs or [])
        mneData = session.mne
        products = cls._snapshotSessionMatrices(session)

        if productRecords is None:
            if products:
                raise ValueError("Session has attached matrices. The save coordinator must publish them and supply productRecords; refusing to omit attachments.")

            validatedProducts = None
        else:
            validatedProducts = cls._validateSessionProductRecords(productRecords)
            declaredNames = {record["name"] for record in validatedProducts}

            if declaredNames != set(products):
                raise ValueError("Manifest product names must exactly match session.dataMatrices.")

        if not runs and mneData is None:
            raise ValueError("A checkpoint must contain behavioral runs or MNE contents.")

        if mneData is not None and type(mneData) is not pglMNE:
            raise TypeError("MNE export currently supports pglMNE, not subclasses.")

        if dataPath is None or not str(dataPath).strip():
            raise ValueError("A destination path is required.")

        filesystem, destination, _ = pglStorage.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix, requireExists=False)

        if filesystem is None:
            raise ValueError("Could not resolve destination filesystem.")

        if filesystem.exists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")

        historyBytes = cls._sessionHistoryBytes(session, acquisition=False)
        filesystem.makedirs(destination, exist_ok=False)
        directories = set()
        files = []
        runRecords = []

        # +++++++++++++++++++++++++++ inventoryDirectory ++++++++++++++++++++++++++
        # Local implementation within saveDerivedSessionFiles.
        def inventoryDirectory(relativeRoot):
            fullRoot = posixpath.join(destination, relativeRoot)
            directories.add(relativeRoot)

            for currentPath, directoryNames, fileNames in filesystem.walk(fullRoot, detail=False, on_error="raise"):
                logicalPath = posixpath.relpath(currentPath, destination)

                for name in directoryNames:
                    directories.add(posixpath.join(logicalPath, name))

                for name in fileNames:
                    savedPath = posixpath.join(currentPath, name)
                    files.append({"name": posixpath.join(logicalPath, name), **pglStorage.hashFile(filesystem, savedPath)})

        if runs:
            filesystem.makedirs(posixpath.join(destination, "runs"), exist_ok=False)
            directories.add("runs")

        for index, run in enumerate(runs):
            runName = f"runs/{index:06d}"
            run.saveDerivedRun(dataPath=posixpath.join(destination, runName), filesystem=filesystem)
            runRecords.append({"index": index, "path": runName})
            inventoryDirectory(runName)

        if mneData is not None:
            mneData.saveBundle(posixpath.join(destination, "mne"), filesystem=filesystem)
            inventoryDirectory("mne")

        historyPath = posixpath.join(destination, "history.json")

        with filesystem.open(historyPath, "xb") as target:
            if target.write(historyBytes) != len(historyBytes):
                raise OSError("Incomplete session history write.")

        expectedHistory = {"sha256": hashlib.sha256(historyBytes).hexdigest(), "size_bytes": len(historyBytes)}
        storedHistory = pglStorage.hashFile(filesystem, historyPath)

        if storedHistory != expectedHistory:
            raise ValueError("Saved session history verification failed.")

        files.append({"name": "history.json", **storedHistory})

        contractVersion = "pgl.derived.session.v1" if mneData is not None else "pgl.derived.behavior.v1"

        if validatedProducts is not None:
            contractVersion = "pgl.derived.session.v2" if mneData is not None else "pgl.derived.behavior.v2"

        manifest = {
            "contract_version": contractVersion,
            "runs": runRecords,
            "directories": sorted(directories),
            "files": sorted(files, key=lambda record: record["name"]),
        }

        if mneData is not None:
            manifest["mne"] = {"path": "mne"}

        if validatedProducts is not None:
            manifest["products"] = validatedProducts

        cls.validateDerivedManifest(manifest)

        class DestinationView:
            # ++++++++++++++++++++++++++++++++++ open +++++++++++++++++++++++++++++++++
            # Local implementation within saveDerivedSessionFiles.
            def open(self, path, mode="rb", **kwargs):
                return filesystem.open(posixpath.join(destination, path), mode=mode, **kwargs)

        view = DestinationView()
        _, metadata = cls._readSessionHistory(view, len(runs))

        if runs and metadata != cls._behaviorLabelMetadata(manifest, view):
            raise ValueError("Derived history labels differ from the exported behavioral settings.")

        if mneData is not None:
            cls._verifyMNEInventory(manifest, view)

        cls._requireSessionMatricesUnchanged(session, products)
        cls.stageDerivedManifest(manifest, filesystem, destination)

        return destination, manifest

    # ++++++++++++++++++++++++++ stageDerivedManifest +++++++++++++++++++++++++
    # Direct callers in this module: saveDerivedSessionFiles.
    @classmethod
    def stageDerivedManifest(cls, manifest, destinationFilesystem, destinationRoot):
        """Write and verify a derived manifest without overwriting.

        The destination directory must already exist.
        The manifest does not include its own file identity.
        Returns the stored manifest's object key, SHA-256, and byte count.
        """
        cls.validateDerivedManifest(manifest)

        # Use the same canonical JSON encoding as raw manifests.
        manifestBytes = cls.encodeRawManifest(manifest)
        expected = {
            "sha256": hashlib.sha256(manifestBytes).hexdigest(),
            "size_bytes": len(manifestBytes),
        }

        destinationPath = posixpath.join(destinationRoot, "manifest.json")

        with destinationFilesystem.open(destinationPath, "xb") as file:
            file.write(manifestBytes)

        stored = pglStorage.hashFile(destinationFilesystem, destinationPath)

        if stored != expected:
            raise ValueError("Stored derived manifest verification failed.")

        return {"object_key": "manifest.json", **stored}


    # -------------------------------------------------------------------------
    # Shared session bundle registration and completion
    # -------------------------------------------------------------------------
    # Register exported checkpoint files and validate catalog correspondence before publication.

    # ++++++++++++++++++++++ _registerSessionBundleFiles +++++++++++++++++++++
    @classmethod
    def _registerSessionBundleFiles(cls, connection, postgres, sessionID, backendID, manifest, *, objectPrefix, storageOptions=None, origin="derived", accessContext=None):
        """Register a staged raw/derived bundle and its verified labels.

        Called by raw-MNE and derived-session publication coordinators.
        Supports derived v1/v2 inventories through manifest validation.

        Verifies the stored manifest, reads verified history/acquisition
        metadata, checks MNE inventory correspondence when applicable, and
        registers every checkpoint file through the caller's access policy.

        Matrix references are not session files. Product links are registered
        separately and checked during checkpoint completion.

        Behavioral file subjects come from archived experiment settings.
        MNE-only subjects come from hashed history metadata.
        Missing subjects are represented as SQL NULL.

        accessContext:
            Optional borrowed publication context for the same database.
            Configure storage options on that context, not both arguments.

        Without accessContext:
            Creates and closes one operation-owned context.

        Uses a nested transaction/savepoint. The caller's outer transaction
        must commit for registrations to persist.

        Does not copy payloads, complete the checkpoint, mutate working
        objects, or replace registered roots with caller-specific SSH URLs.
        Exceptions propagate; operation-owned resources close on failure.
        """
        from contextlib import ExitStack

        for label, value in (("sessionID", sessionID), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if not isinstance(objectPrefix, str) or objectPrefix.startswith("/") or "\\" in objectPrefix or "\x00" in objectPrefix or any(part in {"", ".", ".."} for part in objectPrefix.split("/")):
            raise ValueError("objectPrefix must be a valid relative POSIX path.")

        if storageOptions is not None and not isinstance(storageOptions, dict):
            raise TypeError("storageOptions must be a dictionary or None.")

        if accessContext is not None:
            if not isinstance(accessContext, pglDatabaseAccess):
                raise TypeError("accessContext must be a pglDatabaseAccess instance.")

            if storageOptions:
                raise ValueError("Configure storage options on accessContext, not on both arguments.")

            if cls._checkpointDatabaseIdentity(accessContext._postgres) != cls._checkpointDatabaseIdentity(postgres):
                raise ValueError("The supplied payload context belongs to a different configured database.")

        cls._validateSessionBundleManifest(manifest, origin=origin)

        payloadRole = "acquisition" if origin == "raw" else "derived"
        manifestBytes = cls.encodeRawManifest(manifest)
        manifestIdentity = {
            "sha256": hashlib.sha256(manifestBytes).hexdigest(),
            "size_bytes": len(manifestBytes),
        }
        hasMNE = "mne" in manifest

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        with connection.transaction():
            statement = sql.SQL("SELECT origin, status, contract_version, deleted_at FROM {} WHERE session_id = %s FOR UPDATE").format(sessionsTable)
            checkpoint = connection.execute(statement, (sessionID,)).fetchone()

            if checkpoint != (origin, "in_progress", manifest["contract_version"], None):
                raise ValueError("Checkpoint must be active, in progress, and match the manifest origin and contract.")

            statement = sql.SQL("SELECT 1 FROM {} WHERE session_id = %s LIMIT 1").format(linksTable)

            if connection.execute(statement, (sessionID,)).fetchone() is not None:
                raise ValueError("Checkpoint already has file links.")

            statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backendsTable)
            backend = connection.execute(statement, (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError("Storage backend is missing or inactive.")

            registeredRoot = backend[0]

            with ExitStack() as resources:
                if accessContext is None:
                    access = resources.enter_context(pglDatabaseAccess(postgres, {backendID: storageOptions or {}}))
                else:
                    access = accessContext

                filesystem, root = access.resolve(backendID, registeredRoot)
                manifestKey = posixpath.join(objectPrefix, "manifest.json")

                if pglStorage.hashFile(filesystem, posixpath.join(root, manifestKey)) != manifestIdentity:
                    raise ValueError("Saved manifest does not match the supplied manifest.")

                fileLocations = {
                    record["name"]: {
                        "sha256": record["sha256"],
                        "size_bytes": record["size_bytes"],
                        "locations": [{
                            "backend_id": backendID,
                            "url_prefix": registeredRoot,
                            "object_key": posixpath.join(objectPrefix, record["name"]),
                        }],
                    }
                    for record in manifest["files"]
                }

                # Retain original catalog roots in the archive inventory.
                # Physical reads use the borrowed caller-side access context.
                archive = pglCheckpointFilesystem(manifest, fileLocations, accessContext=access, ownsAccess=False)
                resources.callback(archive.close)

                if hasMNE:
                    cls._verifyMNEInventory(manifest, archive)

                _, historyMetadata = cls._readSessionHistory(archive, len(manifest["runs"]))

                if manifest["runs"]:
                    metadata = cls._behaviorLabelMetadata(manifest, archive)

                    if historyMetadata is not None and historyMetadata != metadata:
                        raise ValueError("History labels differ from archived behavioral experiment settings.")
                else:
                    metadata = historyMetadata or {
                        "version": 1,
                        "labels": [{
                            "runIndex": None,
                            "experimentName": None,
                            "subjectID": None,
                        }],
                    }

                cls._recordSessionLabelMetadata(connection, postgres, sessionID, metadata)
                subjectsByRun = {
                    run["path"]: metadata["labels"][run["index"]]["subjectID"]
                    for run in manifest["runs"]
                }
                mneSubject = metadata["labels"][0]["subjectID"] if not manifest["runs"] else None

                # +++++++++++++++++++++++ registerPayload +++++++++++++++++++
                def registerPayload(name, identity, role, subjectID=None):
                    """Verify a stored file and attach its logical checkpoint link."""
                    fileID = cls.getOrCreateFile(connection, postgres, identity["sha256"], identity["size_bytes"])
                    cls.verifyAndRegisterStorageLocation(connection, postgres, fileID, backendID, posixpath.join(objectPrefix, name), accessContext=access)
                    cls.attachCheckpointFile(connection, postgres, sessionID, name, fileID, role=role, subjectID=subjectID)

                for record in manifest["files"]:
                    name = record["name"]

                    if name == "history.json":
                        registerPayload(name, record, role="history")
                    elif hasMNE and name.startswith("mne/"):
                        registerPayload(name, record, role=payloadRole, subjectID=mneSubject)
                    else:
                        runPath = "/".join(name.split("/")[:2])
                        registerPayload(name, record, role=payloadRole, subjectID=subjectsByRun[runPath])

                registerPayload("manifest.json", manifestIdentity, role="manifest")

        return sessionID

    # ++++++++++++++++++++ _completeSessionBundleCheckpoint +++++++++++++++++++
    # Direct callers in this module: _saveRawMNECheckpoint, completeDerivedCheckpoint.
    @classmethod
    def _completeSessionBundleCheckpoint(cls, connection, postgres, sessionID, manifest, *, origin="derived"):
        """Validate registered contents and complete a raw/derived checkpoint.

        Checks:
            Checkpoint origin, contract, status, and lineage.
            Exact registered file identities and roles.
            Eligible verified file locations on active backends.
            Exact named matrix revisions and active matrix backends.

        V1 manifests imply no attached products.
        Derived v2 manifests explicitly declare products, including an empty
        collection.

        Uses recorded verification metadata, not a fresh payload-byte scan.

        Opens a transaction/savepoint. The caller's outer transaction must
        commit for publication to persist. Does not mutate working objects.
        """
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        cls._validateSessionBundleManifest(manifest, origin=origin)

        productRecords = manifest.get("products", [])
        payloadRole = "acquisition" if origin == "raw" else "derived"
        manifestBytes = cls.encodeRawManifest(manifest)
        manifestHash = hashlib.sha256(manifestBytes).hexdigest()

        expected = {
            record["name"]: (
                record["sha256"],
                record["size_bytes"],
                "history" if record["name"] == "history.json" else payloadRole,
            )
            for record in manifest["files"]
        }
        expected["manifest.json"] = (manifestHash, len(manifestBytes), "manifest")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        with connection.transaction():
            # Product-link changes also lock this owner row. Holding this
            # lock makes link verification and completion one serialized step.
            statement = sql.SQL("""
                SELECT origin, status, contract_version, deleted_at,
                       parent_session_id, root_session_id
                FROM {}
                WHERE session_id = %s
                FOR UPDATE
            """).format(sessions)

            checkpoint = connection.execute(statement, (sessionID,)).fetchone()

            if checkpoint is None or checkpoint[:4] != (origin, "in_progress", manifest["contract_version"], None):
                raise ValueError("Checkpoint must be active, in progress, and match the manifest origin and contract.")

            parentID, rootID = checkpoint[4:]

            if origin == "raw":
                if parentID is not None or rootID != sessionID:
                    raise ValueError("A raw checkpoint must be parentless and its own root.")
            else:
                statement = sql.SQL("SELECT status, deleted_at, root_session_id FROM {} WHERE session_id = %s FOR SHARE").format(sessions)
                parent = connection.execute(statement, (parentID,)).fetchone()

                if parent != ("complete", None, rootID):
                    raise ValueError("Parent must be complete, active, and have the same root.")

            statement = sql.SQL("""
                SELECT link.name,
                       content.sha256,
                       content.size_bytes,
                       link.role,
                       (
                           link.deleted_at IS NULL
                           AND content.deleted_at IS NULL
                           AND EXISTS (
                               SELECT 1
                               FROM {} AS location
                               JOIN {} AS backend
                                 ON backend.backend_id = location.backend_id
                               WHERE location.file_id = content.file_id
                                 AND location.deleted_at IS NULL
                                 AND location.status = 'present'
                                 AND location.verified_at IS NOT NULL
                                 AND backend.is_active
                           )
                       ) AS available
                FROM {} AS link
                JOIN {} AS content ON content.file_id = link.file_id
                WHERE link.session_id = %s
                ORDER BY link.name
            """).format(locations, backends, links, files)

            rows = connection.execute(statement, (sessionID,)).fetchall()
            actual = {name: (digest, size, role) for name, digest, size, role, available in rows}

            if actual != expected:
                raise ValueError("Checkpoint file links do not exactly match the manifest identities and roles.")

            unavailable = [name for name, digest, size, role, available in rows if not available]

            if unavailable:
                raise ValueError("Files lack active links, content identities, or verified active storage locations: " + ", ".join(unavailable))

            # Always verify, including an empty collection. Otherwise an old
            # manifest could conceal additional database product links.
            cls._verifySessionProducts(connection, postgres, sessionID, productRecords)

            statement = sql.SQL("""
                UPDATE {}
                SET status = 'complete',
                    manifest_hash = %s,
                    completed_at = now()
                WHERE session_id = %s
            """).format(sessions)

            connection.execute(statement, (manifestHash, sessionID))

        return manifestHash


    # -------------------------------------------------------------------------
    # Session matrix products
    # -------------------------------------------------------------------------
    # Snapshot attachments, publish revisions, register links, and restore exact manifest-declared
    # products.

    # ++++++++++++++++++++++++ _snapshotSessionMatrices +++++++++++++++++++++++
    # Direct callers in this module: _saveDerivedSession, saveDerivedSessionFiles.
    @staticmethod
    def _snapshotSessionMatrices(session):
        """Validate and snapshot the working session's matrix attachments.

        Reads attachment identities and validates array encodings without
        publishing anything. Does not extract missing products.

        Returns a new dictionary in deterministic product-name order.
        Matrix objects themselves are not copied.
        """
        from .pglData import pglDataMatrix

        products = session.dataMatrices

        if not isinstance(products, dict):
            raise TypeError("session.dataMatrices must be a dictionary.")

        for productName, matrix in products.items():
            if not isinstance(productName, str) or not productName.strip():
                raise ValueError("Session matrix names must be nonblank strings.")

            if any(character in productName for character in ("\x00", "\n", "\r")):
                raise ValueError("Session matrix names must not contain NUL or newline characters.")

            if not isinstance(matrix, pglDataMatrix):
                raise TypeError(f"Session product {productName!r} must be a pglDataMatrix.")

        return {name: products[name] for name in sorted(products)}

    # ++++++++++++++++++++ _requireSessionMatricesUnchanged +++++++++++++++++++
    # Direct callers in this module: _publishSessionMatrices, _saveDerivedSession,
    # saveDerivedSessionFiles.
    @staticmethod
    def _requireSessionMatricesUnchanged(session, products):
        """Reject changes to attachment names or object identities.

        This is not a content-freshness test and cannot detect arbitrary
        in-place edits to exposed xarray values. Concurrent mutation while
        saving is unsupported.
        """
        current = session.dataMatrices

        if not isinstance(current, dict) or set(current) != set(products):
            raise ValueError("Session matrix attachments changed while saving.")

        for productName, matrix in products.items():
            if current[productName] is not matrix:
                raise ValueError(f"Session matrix {productName!r} was replaced while saving.")

    # ++++++++++++++++++++++++ _publishSessionMatrices ++++++++++++++++++++++++
    # Direct callers in this module: _saveDerivedSession.
    @classmethod
    def _publishSessionMatrices(cls, connection, postgres, session, products, *, createdBy, backendID, filesystem, destinationRoot):
        """Publish working matrices and register revisions in caller transaction.

        products must be a snapshot from _snapshotSessionMatrices().

        Writes fresh Zarr bundles through saveDataMatrix(). Inserts catalog
        rows, but does not insert session_products links or complete a
        checkpoint. Returns deterministic records for the session manifest.

        Does not change working matrix revision IDs or session attachments.

        The caller owns the transaction. Database rollback does not remove
        payloads already written; failed saves may leave orphaned bundles.
        """
        from psycopg.types.json import Jsonb

        for label, value in (("createdBy", createdBy), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        cls._requireSessionMatricesUnchanged(session, products)

        # Validate every product before writing the first permanent bundle.
        for productName, matrix in products.items():
            pglStorage.requireSafeSave(matrix)
            pglStorage.validateDataMatrixEncoding(matrix.requireData())

        if not products:
            return []

        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")
        revisions = sql.Identifier(postgres.databaseSchema, "matrix_revisions")

        statement = sql.SQL("SELECT is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backends)
        backend = connection.execute(statement, (backendID,)).fetchone()

        if backend != (True,):
            raise ValueError("Matrix destination backend is missing or inactive.")

        # filesystem/destinationRoot must be the destination already resolved
        # from this backend by the enclosing session save implementation.
        productRoot = posixpath.join(destinationRoot, "matrices")
        records = []

        for productName, matrix in products.items():
            cls._requireSessionMatricesUnchanged(session, products)

            savedURL = pglStorage.saveDataMatrix(matrix, productRoot, filesystem=filesystem)
            savedPath = filesystem._strip_protocol(savedURL)
            descriptorPath = posixpath.join(savedPath, "matrix.json")

            with filesystem.open(descriptorPath, "rb") as source:
                descriptorBytes = source.read(16 * 1024 * 1024 + 1)

            if len(descriptorBytes) > 16 * 1024 * 1024:
                raise ValueError("Published matrix descriptor exceeds the metadata size limit.")

            descriptor = json.loads(descriptorBytes)
            revisionID = descriptor["matrixRevisionID"]

            if descriptor["contractVersion"] != "pgl.matrix.zarr.v1":
                raise ValueError("Published matrix has an unsupported persistence contract.")

            objectKey = f"matrices/matrix-{revisionID}"
            expectedPath = posixpath.join(destinationRoot, objectKey)

            if posixpath.normpath(savedPath) != posixpath.normpath(expectedPath):
                raise ValueError("Published matrix path differs from its expected backend-relative location.")

            statement = sql.SQL("""
                INSERT INTO {} (
                    matrix_revision_id, backend_id, object_key,
                    contract_version, descriptor, created_by
                )
                VALUES (%s::uuid, %s, %s, %s, %s, %s)
            """).format(revisions)

            connection.execute(statement, (revisionID, backendID, objectKey, "pgl.matrix.zarr.v1", Jsonb(descriptor), createdBy))

            records.append({
                "name": productName,
                "matrixRevisionID": revisionID,
            })

        cls._requireSessionMatricesUnchanged(session, products)

        for matrix in products.values():
            pglStorage.requireSafeSave(matrix)

        return records

    # ++++++++++++++++++++++++ _registerSessionProducts +++++++++++++++++++++++
    # Direct callers in this module: _saveDerivedSession.
    @classmethod
    def _registerSessionProducts(cls, connection, postgres, sessionID, records):
        """Attach already-registered matrix revisions to an unpublished session.

        Writes database links only. Does not publish payloads or complete the
        checkpoint. Requires an initially empty product-link collection.

        Uses a savepoint so partial link insertion rolls back on failure.
        The enclosing session-save transaction must commit for links to persist.
        """
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        records = cls._validateSessionProductRecords(records)
        sessions = sql.Identifier(postgres.databaseSchema, "session")
        products = sql.Identifier(postgres.databaseSchema, "session_products")

        with connection.transaction():
            # Serialize attachment registration against checkpoint completion.
            statement = sql.SQL("SELECT status, deleted_at FROM {} WHERE session_id = %s FOR UPDATE").format(sessions)
            owner = connection.execute(statement, (sessionID,)).fetchone()

            if owner != ("in_progress", None):
                raise ValueError("Session products require an active in_progress checkpoint.")

            statement = sql.SQL("SELECT 1 FROM {} WHERE session_id = %s LIMIT 1").format(products)

            if connection.execute(statement, (sessionID,)).fetchone() is not None:
                raise ValueError("Session products have already been registered.")

            statement = sql.SQL("""
                INSERT INTO {} (session_id, product_name, matrix_revision_id)
                VALUES (%s, %s, %s::uuid)
            """).format(products)

            for record in records:
                connection.execute(statement, (sessionID, record["name"], record["matrixRevisionID"]))

            cls._verifySessionProducts(connection, postgres, sessionID, records)

    # +++++++++++++++++++++++++ _verifySessionProducts ++++++++++++++++++++++++
    # Direct callers in this module: _completeSessionBundleCheckpoint, _loadSessionProducts,
    # _registerSessionProducts, _saveDerivedSession, loadSession.
    @classmethod
    def _verifySessionProducts(cls, connection, postgres, sessionID, records):
        """Verify exact session product links and registered matrix availability.

        Uses the caller's transaction. Checks catalog metadata and backend
        status, not payload bytes or scientific values.

        An empty expected list still checks for unexpected database links.
        This is important when loading old manifests or removing attachments.
        """
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        records = cls._validateSessionProductRecords(records)
        products = sql.Identifier(postgres.databaseSchema, "session_products")
        revisions = sql.Identifier(postgres.databaseSchema, "matrix_revisions")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("""
            SELECT product.product_name,
                   product.matrix_revision_id::text,
                   revision.contract_version,
                   backend.is_active
            FROM {} AS product
            JOIN {} AS revision
              ON revision.matrix_revision_id = product.matrix_revision_id
            JOIN {} AS backend
              ON backend.backend_id = revision.backend_id
            WHERE product.session_id = %s
        """).format(products, revisions, backends)

        rows = connection.execute(statement, (sessionID,)).fetchall()
        expected = {record["name"]: record["matrixRevisionID"] for record in records}
        actual = {name: revisionID for name, revisionID, contractVersion, isActive in rows}

        if actual != expected:
            raise ValueError("Session product links do not exactly match the manifest's named matrix revisions.")

        for name, revisionID, contractVersion, isActive in rows:
            if contractVersion != "pgl.matrix.zarr.v1":
                raise ValueError(f"Session product {name!r} has an unsupported matrix contract.")

            if not isActive:
                raise ValueError(f"Session product {name!r} has an inactive storage backend.")

    # +++++++++++++++++++++++++ _openRegisteredMatrix ++++++++++++++++++++++++
    @classmethod
    def _openRegisteredMatrix(cls, postgres, *, revisionID, backendID, objectKey, contractVersion, descriptor, registeredRoot, storageOptionsByBackend=None):
        """Open a catalog-resolved matrix and retain its payload access.

        Called by database product loaders after querying the revision and
        checking that its backend is active.

        Owns a separate access context for this matrix. Reads matrix.json,
        checks it against the catalog descriptor, then delegates standalone
        Zarr restoration to pglStorage.

        Returns:
            A pglDataMatrix whose scientific variables remain lazy.
            Its xarray close callback releases both the underlying reader
            and this matrix's owned access context.

        No SQL connection is opened or retained. The caller supplies catalog
        metadata from its own database lookup.

        Reads descriptor/Zarr metadata, not a complete scientific-value scan.
        Does not publish records, change registered paths, or mutate a dataset
        or session attachment collection.

        On failure, closes all resources created here.
        Call matrix.close() after its deferred readers are no longer needed.
        Derived xarray views must not outlive their source matrix's resources.
        """
        from uuid import UUID

        if not isinstance(revisionID, str):
            raise ValueError("Matrix revision ID must be a canonical UUID string.")

        try:
            canonicalID = str(UUID(revisionID))
        except (ValueError, AttributeError) as error:
            raise ValueError("Invalid matrix revision UUID.") from error

        if revisionID != canonicalID:
            raise ValueError("Matrix revision ID must use canonical UUID formatting.")

        if type(backendID) is not int or not 0 <= backendID <= 9223372036854775807:
            raise ValueError("backendID must be a nonnegative PostgreSQL bigint.")

        if contractVersion != "pgl.matrix.zarr.v1":
            raise ValueError("Unsupported registered matrix contract.")

        if not isinstance(descriptor, dict):
            raise ValueError("Registered matrix descriptor must be a JSON object.")

        if not isinstance(objectKey, str) or objectKey.startswith("/") or "\\" in objectKey or "\x00" in objectKey or any(part in {"", ".", ".."} for part in objectKey.split("/")):
            raise ValueError("Registered matrix has an invalid relative object key.")

        access = pglDatabaseAccess(postgres, storageOptionsByBackend)
        matrix = None

        try:
            filesystem, root = access.resolve(backendID, registeredRoot)
            bundlePath = posixpath.join(root, objectKey)
            maxBytes = 16 * 1024 * 1024

            with filesystem.open(posixpath.join(bundlePath, "matrix.json"), "rb") as source:
                descriptorBytes = source.read(maxBytes + 1)

            if len(descriptorBytes) > maxBytes:
                raise ValueError("Matrix descriptor exceeds the metadata size limit.")

            storedDescriptor = json.loads(descriptorBytes)

            if cls.encodeRawManifest(storedDescriptor) != cls.encodeRawManifest(descriptor):
                raise ValueError("Published matrix descriptor differs from its registered metadata.")

            matrix = pglStorage.loadDataMatrix(bundlePath, filesystem=filesystem)

            if matrix.matrixRevisionID != revisionID:
                raise ValueError("Loaded matrix revision differs from its registered identity.")

            data = matrix.requireData()

            # xarray exposes set_close(), but not a public callback getter.
            # Capture its existing callback before replacing it. Capturing
            # data.close instead would recurse after set_close() is called.
            if not hasattr(data, "_close"):
                raise RuntimeError("This xarray version does not expose the close callback needed to retain database payload access.")

            originalClose = data._close

            if originalClose is not None and not callable(originalClose):
                raise RuntimeError("The restored xarray close callback is invalid.")

            closed = False

            # +++++++++++++++++++++++ closePayload ++++++++++++++++++++++++++
            def closePayload():
                """Release the Zarr reader before its owned access context."""
                nonlocal closed

                if closed:
                    return

                closed = True

                try:
                    if originalClose is not None:
                        originalClose()
                finally:
                    access.close()

            data.set_close(closePayload)
            return matrix

        except BaseException:
            try:
                if matrix is not None:
                    try:
                        matrix.close()
                    except Exception:
                        # Preserve the original loading/validation error.
                        pass
            finally:
                access.close()

            raise

    # ++++++++++++++++++++++++++ _loadSessionProducts ++++++++++++++++++++++++
    @classmethod
    def _loadSessionProducts(cls, connection, postgres, sessionID, records, *, storageOptionsByBackend=None):
        """Restore exact manifest-declared session matrix attachments.

        Called by _loadSessionBundle() after manifest validation.
        Uses the caller's SQL transaction for catalog lookup and product-link
        verification.

        Each product is opened through _openRegisteredMatrix(), which resolves
        server-local roots using the caller's local/SSH access context and
        retains its own resources for deferred array reads.

        Returns:
            A dictionary mapping product names to working pglDataMatrix objects.

        Reads catalog rows, publication descriptors, and Zarr metadata.
        Does not extract behavior, compute scientific arrays, or publish data.
        Returned matrices retain payload readers, not the SQL connection.

        On failure, closes every matrix opened by this call.
        On success, the caller owns the returned matrices and must close them
        once their deferred readers are no longer needed.
        """
        records = cls._validateSessionProductRecords(records)
        cls._verifySessionProducts(connection, postgres, sessionID, records)

        if not records:
            return {}

        productsTable = sql.Identifier(postgres.databaseSchema, "session_products")
        revisionsTable = sql.Identifier(postgres.databaseSchema, "matrix_revisions")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("""
            SELECT product.product_name,
                   revision.matrix_revision_id::text,
                   revision.backend_id,
                   revision.object_key,
                   revision.contract_version,
                   revision.descriptor,
                   backend.url_prefix,
                   backend.is_active
            FROM {} AS product
            JOIN {} AS revision
              ON revision.matrix_revision_id = product.matrix_revision_id
            JOIN {} AS backend
              ON backend.backend_id = revision.backend_id
            WHERE product.session_id = %s
        """).format(productsTable, revisionsTable, backendsTable)

        rows = connection.execute(statement, (sessionID,)).fetchall()
        catalog = {row[0]: row[1:] for row in rows}
        restored = {}

        try:
            for record in records:
                productName = record["name"]
                revisionID, backendID, objectKey, contractVersion, descriptor, registeredRoot, isActive = catalog[productName]

                if revisionID != record["matrixRevisionID"]:
                    raise ValueError(f"Session product {productName!r} differs from its manifest revision.")

                if not isActive:
                    raise ValueError(f"Session product {productName!r} has an inactive backend.")

                matrix = cls._openRegisteredMatrix(postgres, revisionID=revisionID, backendID=backendID, objectKey=objectKey, contractVersion=contractVersion, descriptor=descriptor, registeredRoot=registeredRoot, storageOptionsByBackend=storageOptionsByBackend)
                restored[productName] = matrix

            return restored

        except BaseException:
            for matrix in restored.values():
                try:
                    matrix.close()
                except Exception:
                    pass

            raise

    # +++++++++++++++++++++ _validateSessionProductRecords ++++++++++++++++++++
    # Direct callers in this module: _loadSessionProducts, _registerSessionProducts,
    # _verifySessionProducts, saveDerivedSessionFiles, validateDerivedManifest.
    @staticmethod
    def _validateSessionProductRecords(records):
        """Validate named matrix references intended for a session manifest.

        Records must be sorted by case-sensitive product name.
        Revision IDs must be canonical UUID strings.

        Returns an independent list of normalized dictionaries.
        Does not read arrays, access storage, or mutate the input.
        """
        from uuid import UUID

        if not isinstance(records, list):
            raise ValueError("Session product records must be a list.")

        validated = []

        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {"name", "matrixRevisionID"}:
                raise ValueError(f"Session product {index}: invalid record fields.")

            name = record["name"]

            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Session product {index}: name must be a nonblank string.")

            if any(character in name for character in ("\x00", "\n", "\r")):
                raise ValueError(f"Session product {index}: invalid name characters.")

            revisionID = record["matrixRevisionID"]

            if not isinstance(revisionID, str):
                raise ValueError(f"Session product {name!r}: revision ID must be a string.")

            try:
                canonicalID = str(UUID(revisionID))
            except (ValueError, AttributeError) as error:
                raise ValueError(f"Session product {name!r}: invalid revision UUID.") from error

            if revisionID != canonicalID:
                raise ValueError(f"Session product {name!r}: revision UUID must use canonical formatting.")

            validated.append({"name": name, "matrixRevisionID": revisionID})

        names = [record["name"] for record in validated]

        if names != sorted(set(names)):
            raise ValueError("Session product names must be unique and sorted.")

        return validated


    # -------------------------------------------------------------------------
    # Checkpoint history and acquisition labels
    # -------------------------------------------------------------------------
    # Encode and restore checkpoint history wrappers, validate labels, and mirror verified labels
    # into SQL.

    # ++++++++++++++++++++++++++ _sessionHistoryBytes +++++++++++++++++++++++++
    # Direct callers in this module: saveDerivedSessionFiles, saveRawSessionFiles.
    @classmethod
    def _sessionHistoryBytes(cls, session, *, acquisition=False):
        """Encode history and labels after checking working-session safety.

        Runtime failure state is not serialized into checkpoint metadata.
        Labels remain part of the payload identity.
        """
        from .pglPipeline import pglActionable

        # All current derived exports and MNE raw exports pass through here.
        # Refuse unsafe working state before serializing or writing payloads.
        pglStorage.requireSafeSave(session)

        metadata = {"version": 1, "labels": session.getSessionLabels(acquisition=acquisition)}
        cls._validateSessionLabelMetadata(metadata, runCount=len(session.runs or []))

        history = pglActionable()
        history.uuid = session.uuid
        history.actionHistory = list(session.actionHistory)
        document = json.loads(history.toJSON())

        if "pgl_session_labels" in document:
            raise ValueError("History wrapper already contains the reserved session-label field.")

        document["pgl_session_labels"] = metadata
        return json.dumps(document, sort_keys=True, indent=4).encode("utf-8")

    # ++++++++++++++++++++++++++ _readSessionHistory ++++++++++++++++++++++++++
    # Direct callers in this module: _readReplayCheckpointHistory, _reconstructSessionBundle,
    # _registerSessionBundleFiles, saveDerivedSessionFiles, saveRawSessionFiles.
    @classmethod
    def _readSessionHistory(cls, filesystem, runCount):
        """Restore history and optional labels without loading behavioral fields.

        Supports older history files without session-label metadata.
        """
        from .pglPipeline import pglActionable

        with filesystem.open("history.json", "rt", encoding="utf-8") as source:
            document = json.load(source)

        if not isinstance(document, dict):
            raise ValueError("Checkpoint history must contain a JSON object.")

        hasMetadata = "pgl_session_labels" in document
        metadata = document.pop("pgl_session_labels", None)

        if hasMetadata:
            cls._validateSessionLabelMetadata(metadata, runCount=runCount)

        history = pglActionable.fromJSON(json.dumps(document), filename="history.json")

        if type(history) is not pglActionable:
            raise ValueError("Checkpoint history must contain a pglActionable wrapper.")

        if json.loads(history.toJSON()) != document:
            raise ValueError("Checkpoint history could not be reconstructed without changes.")

        return history, metadata

    # +++++++++++++++++++++++++ _behaviorLabelMetadata ++++++++++++++++++++++++
    # Direct callers in this module: _registerSessionBundleFiles, registerStagedRawCheckpoint,
    # saveDerivedSessionFiles, saveRawSessionFiles.
    @classmethod
    def _behaviorLabelMetadata(cls, manifest, filesystem):
        """Read label pairs from verified archived behavioral experiment settings."""
        from .pglSession import pglSession

        labels = []

        for runRecord in manifest["runs"]:
            path = posixpath.join(runRecord["path"], "experimentSettings.json")

            with filesystem.open(path, "rt", encoding="utf-8") as source:
                settings = json.load(source)

            if not isinstance(settings, dict):
                raise ValueError(f"Experiment settings must contain a JSON object: {path}")

            labels.append({
                "runIndex": runRecord["index"],
                "experimentName": pglSession._normalizeSessionLabel(settings.get("experimentName"), f"{path}: experimentName"),
                "subjectID": pglSession._normalizeSessionLabel(settings.get("subjectID"), f"{path}: subjectID"),
            })

        if not labels:
            raise ValueError("Behavioral label extraction requires at least one run.")

        metadata = {"version": 1, "labels": labels}
        return cls._validateSessionLabelMetadata(metadata, runCount=len(manifest["runs"]))

    # +++++++++++++++++++++ _validateSessionLabelMetadata +++++++++++++++++++++
    # Direct callers in this module: _behaviorLabelMetadata, _readSessionHistory,
    # _recordSessionLabelMetadata, _sessionHistoryBytes, getDatasetMemberSummaries, printSessions.
    @staticmethod
    def _validateSessionLabelMetadata(metadata, runCount=None):
        """Validate the versioned experiment/subject metadata record."""
        from .pglSession import pglSession

        if not isinstance(metadata, dict) or set(metadata) != {"version", "labels"}:
            raise ValueError("Invalid session-label metadata fields.")

        if type(metadata["version"]) is not int or metadata["version"] != 1:
            raise ValueError("Unsupported session-label metadata version.")

        labels = metadata["labels"]

        if not isinstance(labels, list) or not labels:
            raise ValueError("Session-label metadata must contain a nonempty labels list.")

        indices = []

        for record in labels:
            if not isinstance(record, dict) or set(record) != {"runIndex", "experimentName", "subjectID"}:
                raise ValueError("Invalid session-label record.")

            index = record["runIndex"]

            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("Invalid session-label run index.")

            indices.append(index)

            for name in ("experimentName", "subjectID"):
                normalized = pglSession._normalizeSessionLabel(record[name], name)

                if normalized != record[name]:
                    raise ValueError(f"Session-label {name} must use None for an unset value.")

        if indices != [None] and indices != list(range(len(labels))):
            raise ValueError("Session-label run indices must be consecutive, or a single None for a session without runs.")

        if runCount is not None:
            expected = list(range(runCount)) if runCount else [None]

            if indices != expected:
                raise ValueError("Session-label metadata does not match the behavioral run inventory.")

        return metadata

    # ++++++++++++++++++++++ _recordSessionLabelMetadata ++++++++++++++++++++++
    # Direct callers in this module: _registerSessionBundleFiles, registerStagedRawCheckpoint.
    @classmethod
    def _recordSessionLabelMetadata(cls, connection, postgres, sessionID, metadata):
        """Mirror verified payload labels into an active in-progress checkpoint."""
        from psycopg.types.json import Jsonb

        cls._validateSessionLabelMetadata(metadata)
        subjects = sorted({record["subjectID"] for record in metadata["labels"] if record["subjectID"] is not None})

        for subjectID in subjects:
            cls.getOrCreateSubject(connection, postgres, subjectID)

        table = sql.Identifier(postgres.databaseSchema, "session")
        statement = sql.SQL("UPDATE {} SET meta = COALESCE(meta, '{{}}'::jsonb) || %s::jsonb WHERE session_id = %s AND status = 'in_progress' AND deleted_at IS NULL RETURNING session_id").format(table)
        row = connection.execute(statement, (Jsonb({"pgl_session_labels": metadata}), sessionID)).fetchone()

        if row is None:
            raise ValueError("Session labels can only be recorded on an active in-progress checkpoint.")


    # -------------------------------------------------------------------------
    # Checkpoint manifest contracts and inventory verification
    # -------------------------------------------------------------------------
    # Shared checkpoint-format rules used during export, registration, loading, and logical
    # filesystem construction.

    # +++++++++++++++++++++++++++ encodeRawManifest +++++++++++++++++++++++++++
    # Direct callers in this module: _completeSessionBundleCheckpoint,
    # _getCheckpointHistoryBoundary, _loadSessionBundle, _loadSessionProducts,
    # _prepareCheckpointBaseline, _readReplayCheckpointHistory, _registerSessionBundleFiles,
    # _saveDerivedSession, _saveRawMNECheckpoint, completeRawCheckpoint, getDatasetData,
    # getRawFileLocations, getSessionPipeline, hashRawManifest, loadRawManifest,
    # registerStagedRawCheckpoint, saveRawSessionFiles, stageDerivedManifest, stageRawManifest.
    @classmethod
    def encodeRawManifest(cls, manifest):
        """Encode a checkpoint manifest deterministically as UTF-8 JSON.

        Shared by checkpoint hashing, publication, loading, and metadata
        comparisons. Does not validate the manifest or access storage.

        Delegates generic JSON encoding to pglStorage. The compact,
        sorted-key UTF-8 representation is unchanged from existing manifests.
        """
        return pglStorage.encodeJSON(manifest)

    # ++++++++++++++++++++++++++++ hashRawManifest ++++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawSession.
    @classmethod
    def hashRawManifest(cls, manifest):
        """Return the SHA-256 of the exact encoded manifest bytes."""
        return hashlib.sha256(cls.encodeRawManifest(manifest)).hexdigest()

    # ++++++++++++++++++++++++++ validateRawManifest ++++++++++++++++++++++++++
    # Direct callers in this module: _saveRawSession, completeRawCheckpoint, getRawFileLocations,
    # loadRawManifest, pglCheckpointFilesystem, readRawSubjects, registerStagedRawCheckpoint,
    # stageRawManifest, stageRawSession.
    @classmethod
    def validateRawManifest(cls, manifest):
        """Validate the raw behavioral manifest contract."""
        return cls._validateBehaviorManifest(manifest, "pgl.raw.behavior.v1")

    # +++++++++++++++++++++++ validateRawSessionManifest ++++++++++++++++++++++
    # Direct callers in this module: _validateSessionBundleManifest, pglCheckpointFilesystem,
    # saveRawSessionFiles.
    @classmethod
    def validateRawSessionManifest(cls, manifest):
        """Validate a raw root containing MNE and optional behavioral archives.

        Uses the same inventory layout as a derived MNE session.
        Validation does not change the manifest or its encoded identity.
        """
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a dictionary")

        if manifest.get("contract_version") != "pgl.raw.session.v1":
            raise ValueError("Unsupported raw session contract")

        validationView = dict(manifest)
        validationView["contract_version"] = "pgl.derived.session.v1"
        cls.validateDerivedSessionManifest(validationView)
        return manifest

    # ++++++++++++++++++++++++ validateDerivedManifest ++++++++++++++++++++++++
    # Direct callers in this module: _validateSessionBundleManifest, pglCheckpointFilesystem,
    # saveDerivedSessionFiles, stageDerivedManifest.
    @classmethod
    def validateDerivedManifest(cls, manifest):
        """Validate derived v1/v2 manifests without accessing payloads.

        V1 retains its original exact structure.
        V2 adds a required products list of named matrix revision references.

        Matrix references are part of the hashed manifest, but their Zarr
        files remain in separately registered matrix bundles.

        This validates structure only. Catalog correspondence and backend
        availability must be checked during completion and loading.

        Does not mutate the supplied manifest.
        """
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a dictionary.")

        contract = manifest.get("contract_version")
        baseContracts = {
            "pgl.derived.behavior.v2": "pgl.derived.behavior.v1",
            "pgl.derived.session.v2": "pgl.derived.session.v1",
        }

        if contract in baseContracts:
            if "products" not in manifest:
                raise ValueError("Derived v2 manifest requires a products list.")

            cls._validateSessionProductRecords(manifest["products"])

            # Validate the unchanged inventory using the existing v1 rules.
            # Only the temporary top-level dictionary is modified.
            inventoryManifest = dict(manifest)
            inventoryManifest.pop("products")
            inventoryManifest["contract_version"] = baseContracts[contract]
        elif contract in ("pgl.derived.behavior.v1", "pgl.derived.session.v1"):
            inventoryManifest = manifest
        else:
            raise ValueError(f"Unsupported derived manifest contract: {contract!r}")

        if inventoryManifest["contract_version"] == "pgl.derived.session.v1":
            cls.validateDerivedSessionManifest(inventoryManifest)
        else:
            cls._validateBehaviorManifest(inventoryManifest, "pgl.derived.behavior.v1")

            if "history.json" not in {record["name"] for record in inventoryManifest["files"]}:
                raise ValueError("Derived manifest is missing history.json.")

        return manifest

    # +++++++++++++++++++++ validateDerivedSessionManifest ++++++++++++++++++++
    # Direct callers in this module: validateDerivedManifest, validateRawSessionManifest.
    @staticmethod
    def validateDerivedSessionManifest(manifest):
        """Validate a derived session inventory containing an MNE bundle."""

        # +++++++++++++++++++++++++++++ requireFields +++++++++++++++++++++++++++++
        # Local implementation within validateDerivedSessionManifest.
        def requireFields(value, names, label):
            if not isinstance(value, dict) or set(value) != set(names):
                raise ValueError(f"{label} has missing or unexpected fields")

        # ++++++++++++++++++++++++++++++ validatePath +++++++++++++++++++++++++++++
        # Local implementation within validateDerivedSessionManifest.
        def validatePath(path):
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or "\\" in path
                or "\x00" in path
                or any(part in ("", ".", "..") for part in path.split("/"))
            ):
                raise ValueError(f"Invalid manifest path: {path!r}")

        requireFields(
            manifest,
            {"contract_version", "runs", "mne", "directories", "files"},
            "Manifest",
        )
        if manifest["contract_version"] != "pgl.derived.session.v1":
            raise ValueError("Unsupported derived session contract")

        for name in ("runs", "directories", "files"):
            if not isinstance(manifest[name], list):
                raise ValueError(f"Manifest {name} must be a list")

        requireFields(manifest["mne"], {"path"}, "MNE record")
        if manifest["mne"]["path"] != "mne":
            raise ValueError("MNE bundle path must be 'mne'")

        runPaths = set()
        for index, record in enumerate(manifest["runs"]):
            requireFields(record, {"index", "path"}, "Run record")
            if type(record["index"]) is not int or record["index"] != index:
                raise ValueError("Run indices must be consecutive and ordered")

            expectedPath = f"runs/{index:06d}"
            if record["path"] != expectedPath:
                raise ValueError(f"Run path must be {expectedPath!r}")
            runPaths.add(expectedPath)

        directories = manifest["directories"]
        for path in directories: validatePath(path)
        if directories != sorted(set(directories)):
            raise ValueError("Directories must be unique and sorted")

        directorySet = set(directories)
        requiredDirectories = {"mne"}
        if runPaths: requiredDirectories.update({"runs", *runPaths})
        if not requiredDirectories.issubset(directorySet):
            raise ValueError("Missing required bundle directories")

        fileNames = []
        for record in manifest["files"]:
            requireFields(record, {"name", "sha256", "size_bytes"}, "File record")
            validatePath(record["name"])

            digest = record["sha256"]
            size = record["size_bytes"]
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("File SHA256 must be 64 lowercase hexadecimal characters")
            if type(size) is not int or not 0 <= size <= 9223372036854775807:
                raise ValueError("File size must be a nonnegative PostgreSQL bigint")

            fileNames.append(record["name"])

        if fileNames != sorted(set(fileNames)):
            raise ValueError("File names must be unique and sorted")
        if directorySet.intersection(fileNames):
            raise ValueError("A path cannot be both a file and a directory")
        if not {"history.json", "mne/mne.json"}.issubset(fileNames):
            raise ValueError("Missing history.json or mne/mne.json")

        for path in directories + fileNames:
            if path in ("runs", "mne"):
                if path not in directorySet:
                    raise ValueError(f"Expected a directory: {path}")
                continue

            if path == "history.json":
                if path not in fileNames:
                    raise ValueError("history.json must be a file")
                continue

            if posixpath.dirname(path) not in directorySet:
                raise ValueError(f"Missing parent directory for {path!r}")

            if path.startswith("mne/"): continue

            owningRun = "/".join(path.split("/")[:2])
            if owningRun not in runPaths:
                raise ValueError(f"Path lies outside declared bundles: {path!r}")

        for runPath in runPaths:
            if not any(name.startswith(runPath + "/") for name in fileNames):
                raise ValueError(f"Run contains no files: {runPath}")

        return manifest

    # +++++++++++++++++++++ _validateSessionBundleManifest ++++++++++++++++++++
    # Direct callers in this module: _completeSessionBundleCheckpoint, _loadSessionBundle,
    # _readReplayCheckpointHistory, _reconstructSessionBundle, _registerSessionBundleFiles.
    @classmethod
    def _validateSessionBundleManifest(cls, manifest, *, origin):
        if origin == "raw":
            return cls.validateRawSessionManifest(manifest)
        if origin == "derived":
            return cls.validateDerivedManifest(manifest)
        raise ValueError(f"Unsupported bundle origin: {origin!r}")

    # +++++++++++++++++++++++ _validateBehaviorManifest +++++++++++++++++++++++
    # Direct callers in this module: validateDerivedManifest, validateRawManifest.
    @staticmethod
    def _validateBehaviorManifest(manifest, contractVersion):
        """Validate the v1 manifest structure without accessing payloads."""

        # +++++++++++++++++++++++++++++ requireFields +++++++++++++++++++++++++++++
        # Local implementation within _validateBehaviorManifest.
        def requireFields(value, fields, label):
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError(f"{label} has missing or unexpected fields.")

        # ++++++++++++++++++++++++++++++ validatePath +++++++++++++++++++++++++++++
        # Local implementation within _validateBehaviorManifest.
        def validatePath(path):
            if (
                not isinstance(path, str)
                or path.startswith("/")
                or "\\" in path
                or "\x00" in path
            ):
                raise ValueError(f"Invalid manifest path: {path!r}")

            if any(part in {"", ".", ".."} for part in path.split("/")):
                raise ValueError(f"Invalid manifest path: {path!r}")

        requireFields(
            manifest,
            {"contract_version", "runs", "directories", "files"},
            "Manifest",
        )

        if manifest["contract_version"] != contractVersion:
            raise ValueError("Unsupported manifest contract.")

        for field in ("runs", "directories", "files"):
            if not isinstance(manifest[field], list):
                raise ValueError(f"Manifest {field} must be a list.")

        if not manifest["runs"]:
            raise ValueError("Manifest must contain at least one run.")

        runPaths = set()

        for index, record in enumerate(manifest["runs"]):
            requireFields(record, {"index", "path"}, "Run record")

            if type(record["index"]) is not int or record["index"] != index:
                raise ValueError(
                    "Run indices must be consecutive and ordered from zero."
                )

            expectedPath = f"runs/{index:06d}"

            if record["path"] != expectedPath:
                raise ValueError(f"Run path must be {expectedPath!r}.")

            runPaths.add(expectedPath)

        directories = manifest["directories"]

        for path in directories:
            validatePath(path)

        if directories != sorted(set(directories)):
            raise ValueError("Directories must be unique and sorted.")

        directorySet = set(directories)

        if not {"runs", *runPaths}.issubset(directorySet):
            raise ValueError("Manifest is missing required run directories.")

        fileNames = []

        for record in manifest["files"]:
            requireFields(
                record,
                {"name", "sha256", "size_bytes"},
                "File record",
            )
            validatePath(record["name"])

            digest = record["sha256"]
            size = record["size_bytes"]

            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(
                    "File SHA-256 must contain 64 lowercase hexadecimal characters."
                )

            if type(size) is not int or not 0 <= size <= 9223372036854775807:
                raise ValueError(
                    "File size must be a nonnegative PostgreSQL bigint."
                )

            fileNames.append(record["name"])

        if fileNames != sorted(set(fileNames)):
            raise ValueError("File names must be unique and sorted.")

        if directorySet.intersection(fileNames):
            raise ValueError(
                "A manifest path cannot be both a file and a directory."
            )

        for path in directories + fileNames:
            if path == "runs":
                continue

            # Only derived checkpoints permit this session-level file.
            if (
                contractVersion == "pgl.derived.behavior.v1"
                and path == "history.json"
                and path in fileNames
            ):
                continue

            if posixpath.dirname(path) not in directorySet:
                raise ValueError(f"Missing parent directory for {path!r}.")

            owningRun = "/".join(path.split("/")[:2])

            if owningRun not in runPaths:
                raise ValueError(
                    f"Path lies outside the declared runs: {path!r}"
                )

        for runPath in runPaths:
            if not any(name.startswith(runPath + "/") for name in fileNames):
                raise ValueError(f"Run contains no files: {runPath}")

        return manifest

    # ++++++++++++++++++++++++++ _verifyMNEInventory ++++++++++++++++++++++++++
    # Direct callers in this module: _reconstructSessionBundle, _registerSessionBundleFiles,
    # saveDerivedSessionFiles, saveRawSessionFiles.
    @classmethod
    def _verifyMNEInventory(cls, manifest, filesystem):
        """Verify MNE descriptor references against the outer file inventory."""
        from .pglSession import pglMNEBundle

        mneRoot = manifest["mne"]["path"]
        descriptorPath = posixpath.join(mneRoot, "mne.json")
        inventory = {record["name"]: record for record in manifest["files"]}
        descriptorRecord = inventory[descriptorPath]

        maxBytes = 16 * 1024 * 1024
        if descriptorRecord["size_bytes"] > maxBytes:
            raise ValueError("MNE descriptor exceeds size limit")

        with filesystem.open(descriptorPath, "rb") as source:
            content = source.read(descriptorRecord["size_bytes"] + 1)

        if (
            len(content) != descriptorRecord["size_bytes"]
            or hashlib.sha256(content).hexdigest() != descriptorRecord["sha256"]
        ):
            raise ValueError("MNE descriptor identity mismatch")

        descriptor = json.loads(content)
        pglMNEBundle._validateDescriptor(descriptor)

        expected = {}
        for resource in descriptor["resources"].values():
            for record in resource["files"]:
                name = posixpath.join(mneRoot, resource["root"], record["name"])
                if name in expected: raise ValueError(f"Duplicate MNE payload reference: {name}")
                expected[name] = (record["sha256"], record["size_bytes"])

        actual = {
            name: (record["sha256"], record["size_bytes"])
            for name, record in inventory.items()
            if name.startswith(mneRoot + "/") and name != descriptorPath
        }

        if actual != expected:
            raise ValueError("MNE descriptor payloads do not exactly match checkpoint inventory")

        return descriptor


    # -------------------------------------------------------------------------
    # Dataset configuration, history, and publication validation
    # -------------------------------------------------------------------------
    # Resolve dataset binding and prepare lightweight metadata without embedding scientific
    # products.

    # +++++++++++++++++++++++++ _resolveDatasetStorage ++++++++++++++++++++++++
    # Direct callers in this module: getDatasetBehaviorData, getDatasetData,
    # getDatasetMemberSummaries, getDatasetSession, loadDataset, saveDataset.
    @classmethod
    def _resolveDatasetStorage(cls, dataset, *, settings=None, settingsName=None):
        """Resolve configuration and reject a different bound database.

        The binding is a runtime endpoint descriptor, not a globally portable
        database UUID. Credentials and settings are never serialized.
        """
        from .pglDataset import pglDataset
        from .pglPostgres import pglPostgres

        if not isinstance(dataset, pglDataset):
            raise TypeError("Expected a pglDataset.")

        if settings is not None and settingsName is not None:
            raise ValueError("Supply settings or settingsName, not both.")

        if settings is None and settingsName is None:
            settings = dataset._datasetSettings
            settingsName = dataset._datasetSettingsName if settings is None else None

        resolvedSettings = cls.getSettings(settings=settings, settingsName=settingsName)

        if resolvedSettings is None:
            return None, None

        postgres = pglPostgres.fromSettings(settings=resolvedSettings)

        if postgres is None:
            return None, None

        identity = cls._checkpointDatabaseIdentity(postgres)

        if dataset._datasetDatabaseIdentity and dataset._datasetDatabaseIdentity != identity:
            raise ValueError("The selected database differs from this dataset's bound database. Dataset membership cannot be resolved against another database.")

        return resolvedSettings, postgres

    # ++++++++++++++++++++++++++ _bindDatasetStorage ++++++++++++++++++++++++++
    # Direct callers in this module: getDatasetBehaviorData, getDatasetData,
    # getDatasetMemberSummaries, getDatasetSession, loadDataset.
    @classmethod
    def _bindDatasetStorage(cls, dataset, resolvedSettings, postgres):
        """Install runtime configuration after successful metadata access."""
        dataset._datasetDatabaseIdentity = deepcopy(cls._checkpointDatabaseIdentity(postgres))
        dataset._datasetSettings = resolvedSettings
        dataset._datasetSettingsName = None

    # ++++++++++++++++++++++ _datasetDatabaseIdentityJSON +++++++++++++++++++++
    # Direct callers in this module: getDatasetData, loadDataset, saveDataset.
    @staticmethod
    def _datasetDatabaseIdentityJSON(identity):
        """Canonical runtime database identity for reference comparisons."""
        return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)

    # +++++++++++++++++++++ _requireSafeDatasetCheckpoint +++++++++++++++++++++
    # Direct callers in this module: _prepareDatasetMetadata, getDatasetBehaviorData, saveDataset.
    @classmethod
    def _requireSafeDatasetCheckpoint(cls, dataset):
        """Validate failure bookkeeping, history prefix, and family identity."""
        from .pglDataset import pglDataset

        if not isinstance(dataset, pglDataset):
            raise TypeError("Expected a pglDataset.")

        pglStorage.requireSafeSave(dataset)

        baseline = dataset._datasetBaseline

        if not baseline:
            return

        expectedFields = {"version", "datasetID", "familyUUID", "databaseIdentity", "historyHashes"}

        if set(baseline) != expectedFields or type(baseline["version"]) is not int or baseline["version"] != 1:
            raise ValueError("Invalid dataset checkpoint baseline.")

        if dataset.datasetID != baseline["datasetID"]:
            raise ValueError("Dataset ID differs from its saved baseline.")

        if dataset.uuid != baseline["familyUUID"]:
            raise ValueError("Dataset family UUID differs from its saved baseline.")

        if dataset._datasetDatabaseIdentity != baseline["databaseIdentity"]:
            raise ValueError("Dataset database binding differs from its saved baseline.")

        expectedHashes = baseline["historyHashes"]

        if not isinstance(expectedHashes, list):
            raise ValueError("Invalid dataset history baseline.")

        actualHashes = cls._datasetHistoryHashes(dataset)

        if len(actualHashes) < len(expectedHashes) or actualHashes[:len(expectedHashes)] != expectedHashes:
            raise ValueError("Saved dataset history was shortened or edited. Historical entries must remain unchanged; append new analyses instead.")

    # +++++++++++++++++++++++++ _datasetHistoryHashes +++++++++++++++++++++++++
    # Direct callers in this module: _requireSafeDatasetCheckpoint, loadDataset, saveDataset.
    @staticmethod
    def _datasetHistoryHashes(dataset):
        """Fingerprint history metadata without touching scientific products."""
        return [hashlib.sha256(entry.toJSON(strict=True).encode("utf-8")).hexdigest() for entry in dataset.actionHistory]

    # ++++++++++++++++++++++++ _prepareDatasetMetadata ++++++++++++++++++++++++
    # Direct callers in this module: saveDataset.
    @classmethod
    def _prepareDatasetMetadata(cls, dataset):
        """Strictly encode lightweight metadata without embedding products.

        Membership lives in dataset_members.
        Product links live in dataset_products.
        Editable annotations live in dataset columns.
        """
        from uuid import UUID
        from .pglDataset import pglDataset
        from .pglSerialize import pglSerialize

        cls._requireSafeDatasetCheckpoint(dataset)
        dataset._validateName(dataset.name)
        dataset._validateMembers(dataset.members)
        cls._validateDatasetProducts(dataset)

        familyUUID = dataset.uuid
        _ = dataset.pglVersion
        _ = dataset.github

        try:
            parsedUUID = UUID(familyUUID)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("Dataset uuid must be a valid UUID string.") from error

        if str(parsedUUID) != familyUUID:
            raise ValueError("Dataset uuid must use canonical lowercase UUID formatting.")

        text = dataset.toJSON(strict=True)
        restored = pglSerialize.fromJSON(text, strict=True)

        if type(restored) is not pglDataset:
            raise TypeError("Dataset metadata must restore as an exact pglDataset.")

        if restored.data:
            raise ValueError("Dataset metadata must not embed scientific products.")

        if restored.toJSON(strict=True) != text:
            raise ValueError("Dataset metadata did not round-trip exactly.")

        return json.loads(text)

    # +++++++++++++++++++++++++++ _datasetAnnotation ++++++++++++++++++++++++++
    # Direct callers in this module: saveDataset.
    @staticmethod
    def _datasetAnnotation(value, fieldName):
        """Normalize blank annotations without imposing a character limit."""
        if value is None:
            return None

        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"{fieldName} must be a string without NUL, or None.")

        return value if value.strip() else None

    # ++++++++++++++++++++++++ _validateDatasetProducts +++++++++++++++++++++++
    # Direct callers in this module: _prepareDatasetMetadata, loadDataset, printDataset,
    # saveDataset.
    @staticmethod
    def _validateDatasetProducts(dataset):
        """Validate product names/types without opening references or arrays."""
        from .pglData import pglDataMatrix, pglDataMatrixReference

        products = dict(dataset.data)

        for name, product in products.items():
            if not isinstance(name, str) or not name.strip() or "\x00" in name:
                raise ValueError("Dataset product names must be nonblank strings without NUL.")

            if not isinstance(product, (pglDataMatrix, pglDataMatrixReference)):
                raise TypeError(f"Product {name!r} must be a pglDataMatrix or pglDataMatrixReference.")

        return products

    # ++++++++++++++++++++ _requireDatasetProductsUnchanged +++++++++++++++++++
    # Direct callers in this module: saveDataset.
    @classmethod
    def _requireDatasetProductsUnchanged(cls, dataset, products):
        """Check product attachment identity without comparing array contents.

        This does not detect all mutations through exposed xarray objects.
        Concurrent mutation during saving is unsupported.
        """
        if set(dataset.data) != set(products):
            raise ValueError("Dataset product names changed during saving.")

        for name, product in products.items():
            if dataset.data[name] is not product:
                raise ValueError(f"Dataset product {name!r} was replaced during saving.")


    # -------------------------------------------------------------------------
    # Session lifecycle coordination
    # -------------------------------------------------------------------------
    # Preview, confirm, lock, and apply coordinated session deletion groups or whole-group
    # restoration.

    # ++++++++++++++++++++++ _changeSessionDeletionState ++++++++++++++++++++++
    # Direct callers in this module: deleteSession, undeleteSession.
    @classmethod
    def _changeSessionDeletionState(cls, sessionID, *, deleted, deleteDownstreamSessions=False, settings=None, settingsName=None, sslRootCert=None):
        """Settings-based lifecycle wrapper with confirmation and commit handling."""
        from .pglPostgres import pglPostgres

        if isinstance(sessionID, bool) or not isinstance(sessionID, int) or not 0 <= sessionID <= 9223372036854775807:
            pglMessages.warning("sessionID must be a nonnegative PostgreSQL bigint.", level=1)
            return None

        if type(deleted) is not bool:
            pglMessages.warning("deleted must be a boolean.", level=1)
            return None

        if type(deleteDownstreamSessions) is not bool:
            pglMessages.warning("deleteDownstreamSessions must be a boolean.", level=1)
            return None

        if deleteDownstreamSessions and not deleted:
            pglMessages.warning("deleteDownstreamSessions applies only to deletion.", level=1)
            return None

        action = "delete" if deleted else "undelete"
        approvedSessionIDs = None

        try:
            try:
                postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)
            except (TypeError, ValueError) as error:
                pglMessages.warning(f"Cannot {action} session: invalid settings or configuration: {error}", level=1)
                return None

            if postgres is None:
                return None

            if deleted and deleteDownstreamSessions:
                # Finish the preview transaction AND close its connection
                # before asking the user for confirmation.
                with postgres.connect(sslRootCert=sslRootCert) as connection:
                    if connection is None:
                        return None

                    with connection.transaction():
                        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                        records = cls._getSessionDeletionRecords(connection, postgres, sessionID)
                        candidates = [record for record in records if record["deleted_at"] is None]

                        target = next(record for record in records if record["session_id"] == sessionID)

                        if target["deleted_at"] is not None and candidates:
                            raise cls._SessionLifecycleBlocked(f"Session {sessionID} is already deleted but has active descendants. No new deletion group was created; resolve the existing lifecycle inconsistency first.")

                if not candidates:
                    return False

                databaseLabel = f"{postgres.databaseName}.{postgres.databaseSchema}"
                confirmed = cls._confirmSessionDeletion(sessionID, candidates, databaseLabel=databaseLabel)

                if confirmed is not True:
                    return False

                approvedSessionIDs = [record["session_id"] for record in candidates]

            # A fresh transaction performs the authoritative checks.
            with postgres.connect(sslRootCert=sslRootCert) as connection:
                if connection is None:
                    return None

                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    changed = cls._setSessionDeleted(connection, postgres, sessionID, deleted=deleted, approvedSessionIDs=approvedSessionIDs)

        except cls._UnknownLifecycleSession as error:
            pglMessages.warning(str(error), level=0)
            return False

        except cls._SessionLifecycleBlocked as error:
            pglMessages.warning(str(error), level=2)
            return False

        except psycopg.Error as error:
            pglMessages.warning(f"Could not {action} session {sessionID}: {error}. If the connection failed during commit, the outcome may be uncertain; check the session state before retrying.", level=2)
            return None

        # Report success only after transaction and connection cleanup.
        if changed:
            if deleted:
                pglMessages.message(f"Soft deletion committed for session {sessionID} and its deletion group in {postgres.databaseName}.{postgres.databaseSchema}.")
            else:
                pglMessages.message(f"Restoration committed for session {sessionID} and its deletion group, if any, in {postgres.databaseName}.{postgres.databaseSchema}.")

        return changed

    # +++++++++++++++++++++++ _getSessionDeletionRecords ++++++++++++++++++++++
    # Direct callers in this module: _changeSessionDeletionState, _setSessionDeleted.
    @classmethod
    def _getSessionDeletionRecords(cls, connection, postgres, sessionID):
        """Return target/descendants and reject dataset-protected deletion.

        Includes deleted records in the returned inventory, but checks
        protection only for sessions that would be newly deleted.

        Used in a consistent preview snapshot and again under lifecycle
        locks before authoritative deletion. A preview alone does not
        authorize any database changes.

        Does not write, commit, reconstruct sessions, or read payloads.
        """
        from psycopg.rows import dict_row

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        statement = sql.SQL("""
            SELECT session_id, parent_session_id, root_session_id,
                   origin, kind, status, description, created_at, deleted_at
            FROM {}
            WHERE session_id = %s
        """).format(sessions)

        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(statement, (sessionID,))
            target = cursor.fetchone()

        if target is None:
            raise cls._UnknownLifecycleSession(f"Unknown session: {sessionID}")

        descendants = cls.getDownstreamSessions(connection, postgres, sessionID)
        records = sorted([target, *descendants], key=lambda record: record["session_id"])
        proposedIDs = [record["session_id"] for record in records if record["deleted_at"] is None]

        cls._requireNoDatasetSessionBlockers(connection, postgres, proposedIDs)
        return records

    # ++++++++++++++++++++++++ _confirmSessionDeletion ++++++++++++++++++++++++
    # Direct callers in this module: _changeSessionDeletionState.
    @staticmethod
    def _confirmSessionDeletion(sessionID, sessions, *, databaseLabel):
        """Print every proposed member and require an explicit 'yes'.

        No database access or payload I/O. Call without database locks.
        Cancellation or unavailable input never authorizes deletion.
        """
        records = sorted(sessions, key=lambda record: record["session_id"])

        if not records:
            return False

        # ++++++++++++++++++++++++++++++++ display ++++++++++++++++++++++++++++++++
        # Local implementation within _confirmSessionDeletion.
        def display(value):
            return "NULL" if value is None else repr(value)

        pglMessages.print()
        pglMessages.printHeader("sessions to be deleted")
        pglMessages.print(f"Database: {display(databaseLabel)}")
        pglMessages.print(f"Deletion-group top-level session: {sessionID}")
        pglMessages.print(f"Sessions to be newly soft-deleted: {len(records)}")
        pglMessages.print("Previously deleted sessions are not added to this group.")
        pglMessages.print()

        for record in records:
            label = "top-level session" if record["session_id"] == sessionID else "downstream session"
            pglMessages.print(f"  Session {record['session_id']} ({label})")
            pglMessages.print(f"    Parent: {display(record.get('parent_session_id'))}")
            pglMessages.print(f"    Origin: {display(record.get('origin'))} | Status: {display(record.get('status'))}")
            pglMessages.print(f"    Created: {display(record.get('created_at'))}")
            pglMessages.print(f"    Description: {display(record.get('description'))}")
            pglMessages.print()

        pglMessages.print("This is soft deletion only. No payload files will be removed.")
        pglMessages.print(f"Restore this entire deletion group with undeleteSession({sessionID}).")
        pglMessages.print("Individual group members cannot be restored separately.")
        pglMessages.print("Restoration requires any parent outside the group to be active.")
        pglMessages.print("All listed sessions must be included; partial selection is not supported.")

        try:
            response = input("Type yes to soft-delete ALL listed sessions; anything else cancels: ")
        except (EOFError, OSError, KeyboardInterrupt, NotImplementedError):
            pglMessages.print("\nDeletion cancelled: confirmation was interrupted or input was unavailable.")
            return False

        if response.strip().lower() != "yes":
            pglMessages.print("Deletion cancelled.")
            return False

        pglMessages.print("Confirmation received. The deletion set will be rechecked before any changes.")
        return True

    # +++++++++++++++++++++++++++ _setSessionDeleted ++++++++++++++++++++++++++
    # Direct callers in this module: _changeSessionDeletionState.
    @classmethod
    def _setSessionDeleted(cls, connection, postgres, sessionID, *, deleted, approvedSessionIDs=None):
        """Atomically delete a group or restore an existing deletion group.

        Internal lifecycle operation; no confirmation prompt or payload I/O.

        approvedSessionIDs:
            None permits deletion only when no active descendants exist.
            Otherwise, must exactly match all sessions to be newly deleted.
            The public wrapper supplies this only after user confirmation.
            Not permitted for restoration.

        Returns True for a change, False for an unchanged state.
        Unknown sessions and blocked operations raise dedicated ValueError
        subclasses; the public wrapper translates them into warnings.

        Opens a transaction/savepoint. An outer transaction must commit for
        persistence. Requires READ COMMITTED isolation.

        Conservative EXCLUSIVE locks last until the outer transaction ends.
        Ordinary SELECT queries remain allowed; competing writes are blocked.
        """
        if isinstance(sessionID, bool) or not isinstance(sessionID, int) or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        if type(deleted) is not bool:
            raise ValueError("deleted must be a boolean.")

        if approvedSessionIDs is not None:
            if not deleted:
                raise ValueError("approvedSessionIDs applies only to deletion.")

            if not isinstance(approvedSessionIDs, (list, tuple)):
                raise ValueError("approvedSessionIDs must be a list or tuple.")

            for value in approvedSessionIDs:
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9223372036854775807:
                    raise ValueError("approvedSessionIDs must contain nonnegative PostgreSQL bigints.")

            if len(set(approvedSessionIDs)) != len(approvedSessionIDs):
                raise ValueError("approvedSessionIDs must not contain duplicates.")

            approvedSessionIDs = sorted(approvedSessionIDs)

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        groups = sql.Identifier(postgres.databaseSchema, "session_deletion_groups")
        members = sql.Identifier(postgres.databaseSchema, "session_deletion_group_members")

        from psycopg.rows import tuple_row

        with connection.transaction():
            with connection.cursor(row_factory=tuple_row) as cursor:
                cursor.execute("SHOW transaction_isolation")

                if cursor.fetchone()[0] != "read committed":
                    raise ValueError("Session delete/undelete requires READ COMMITTED isolation.")

                # Parent relationships live on session, so locking session
                # also protects the authoritative descendant/parent checks.
                statement = sql.SQL("LOCK TABLE {}, {}, {}, {}, {}, {} IN EXCLUSIVE MODE").format(sessions, links, files, locations, groups, members)
                cursor.execute(statement)

                cursor.execute(sql.SQL("SELECT deleted_at FROM {} WHERE session_id = %s").format(sessions), (sessionID,))
                target = cursor.fetchone()

                if target is None:
                    raise cls._UnknownLifecycleSession(f"Unknown session: {sessionID}")

                if deleted:
                    records = cls._getSessionDeletionRecords(connection, postgres, sessionID)
                    candidates = [record for record in records if record["deleted_at"] is None]
                    affectedIDs = sorted(record["session_id"] for record in candidates)

                    # Never silently expand OR shrink the approved set.
                    if approvedSessionIDs is not None and affectedIDs != approvedSessionIDs:
                        raise cls._SessionLifecycleBlocked("The sessions to be deleted changed after confirmation. Nothing was changed; rerun deleteSession() to review the current list.")

                    if target[0] is not None:
                        if affectedIDs:
                            raise cls._SessionLifecycleBlocked(f"Session {sessionID} is already deleted but has active descendants. No new deletion group was created; resolve the existing lifecycle inconsistency first.")
                        return False

                    downstreamIDs = [value for value in affectedIDs if value != sessionID]

                    if downstreamIDs and approvedSessionIDs is None:
                        listed = ", ".join(str(value) for value in downstreamIDs)
                        raise cls._SessionLifecycleBlocked(f"Cannot delete session {sessionID}: active downstream sessions exist: {listed}. Use deleteDownstreamSessions=True to preview and explicitly confirm deletion of the complete group.")

                    # Prevent a group from spanning a previously deleted
                    # intermediate parent. Such a group could otherwise
                    # become impossible to restore under the parent rule.
                    recordsByID = {record["session_id"]: record for record in records}

                    for record in candidates:
                        if record["session_id"] == sessionID:
                            continue

                        parent = recordsByID.get(record["parent_session_id"])

                        if parent is not None and parent["deleted_at"] is not None:
                            raise cls._SessionLifecycleBlocked(f"Cannot create deletion group: active session {record['session_id']} has deleted parent {record['parent_session_id']} inside the downstream tree. Resolve this existing lifecycle inconsistency first; nothing was changed.")

                    # Active sessions must not already have outstanding
                    # group membership or lead an outstanding group.
                    statement = sql.SQL("""
                        SELECT session_id
                        FROM {}
                        WHERE session_id = ANY(%s::bigint[])
                          AND restored_at IS NULL
                        ORDER BY session_id
                    """).format(members)
                    cursor.execute(statement, (affectedIDs,))
                    conflicts = [row[0] for row in cursor.fetchall()]

                    if conflicts:
                        raise cls._SessionLifecycleBlocked(f"Active sessions already have outstanding deletion-group membership: {conflicts}. Nothing was changed; review lifecycle consistency.")

                    statement = sql.SQL("""
                        SELECT top_session_id
                        FROM {}
                        WHERE top_session_id = ANY(%s::bigint[])
                          AND restored_at IS NULL
                        ORDER BY top_session_id
                    """).format(groups)
                    cursor.execute(statement, (affectedIDs,))
                    conflicts = [row[0] for row in cursor.fetchall()]

                    if conflicts:
                        raise cls._SessionLifecycleBlocked(f"Active sessions already lead outstanding deletion groups: {conflicts}. Nothing was changed; review lifecycle consistency.")

                    statement = sql.SQL("""
                        INSERT INTO {} (top_session_id)
                        VALUES (%s)
                        RETURNING deletion_group_id, created_at
                    """).format(groups)
                    cursor.execute(statement, (sessionID,))
                    groupID, marker = cursor.fetchone()

                    statement = sql.SQL("""
                        INSERT INTO {} (deletion_group_id, session_id)
                        SELECT %s, member_id
                        FROM unnest(%s::bigint[]) AS member_ids(member_id)
                    """).format(members)
                    cursor.execute(statement, (groupID, affectedIDs))

                    cls._applySessionDeletionMarkers(connection, postgres, affectedIDs, marker)
                    return True

                # Restoration: resolve outstanding membership first.
                statement = sql.SQL("""
                    SELECT membership.deletion_group_id,
                           deletion_group.top_session_id,
                           deletion_group.restored_at
                    FROM {} AS membership
                    JOIN {} AS deletion_group
                      ON deletion_group.deletion_group_id = membership.deletion_group_id
                    WHERE membership.session_id = %s
                      AND membership.restored_at IS NULL
                """).format(members, groups)
                cursor.execute(statement, (sessionID,))
                membership = cursor.fetchone()

                statement = sql.SQL("""
                    SELECT deletion_group_id
                    FROM {}
                    WHERE top_session_id = %s
                      AND restored_at IS NULL
                """).format(groups)
                cursor.execute(statement, (sessionID,))
                ledGroup = cursor.fetchone()

                if membership is None:
                    if ledGroup is not None:
                        raise cls._SessionLifecycleBlocked(f"Session {sessionID} leads an outstanding group but lacks its membership record. Nothing was changed; review lifecycle consistency.")

                    if target[0] is None:
                        return False

                    # Legacy deletion: do not invent historical membership.
                    groupID = None
                    affectedIDs = [sessionID]

                else:
                    groupID, topID, groupRestoredAt = membership

                    if groupRestoredAt is not None:
                        raise cls._SessionLifecycleBlocked(f"Session {sessionID} has outstanding membership in an already-restored group. Nothing was changed; review lifecycle consistency.")

                    if topID != sessionID:
                        raise cls._SessionLifecycleBlocked(f"Session {sessionID} belongs to the deletion group led by session {topID}. Restore the entire group with undeleteSession({topID}); individual members cannot be restored separately.")

                    if ledGroup != (groupID,):
                        raise cls._SessionLifecycleBlocked(f"Session {sessionID} has inconsistent deletion-group ownership. Nothing was changed.")

                    statement = sql.SQL("""
                        SELECT membership.session_id,
                               membership.restored_at,
                               checkpoint.deleted_at
                        FROM {} AS membership
                        JOIN {} AS checkpoint
                          ON checkpoint.session_id = membership.session_id
                        WHERE membership.deletion_group_id = %s
                        ORDER BY membership.session_id
                    """).format(members, sessions)
                    cursor.execute(statement, (groupID,))
                    memberRows = cursor.fetchall()
                    affectedIDs = [row[0] for row in memberRows]

                    if sessionID not in affectedIDs or any(restoredAt is not None or deletedAt is None for _, restoredAt, deletedAt in memberRows):
                        raise cls._SessionLifecycleBlocked(f"Deletion group {groupID} has inconsistent member states. Nothing was changed; review lifecycle consistency.")

                # Every required parent must either be restored in this
                # operation or already be active outside the group.
                statement = sql.SQL("""
                    SELECT child.session_id, child.parent_session_id
                    FROM {sessions} AS child
                    LEFT JOIN {sessions} AS parent
                      ON parent.session_id = child.parent_session_id
                    WHERE child.session_id = ANY(%s::bigint[])
                      AND child.parent_session_id IS NOT NULL
                      AND NOT (child.parent_session_id = ANY(%s::bigint[]))
                      AND (parent.session_id IS NULL OR parent.deleted_at IS NOT NULL)
                    ORDER BY child.session_id
                """).format(sessions=sessions)
                cursor.execute(statement, (affectedIDs, affectedIDs))
                blockedParents = cursor.fetchall()

                if blockedParents:
                    details = ", ".join(f"session {childID} requires parent {parentID}" for childID, parentID in blockedParents)
                    raise cls._SessionLifecycleBlocked(f"Cannot restore session {sessionID}: {details}. Restore the required parent or its top-level deletion group first. Nothing was changed.")

                cls._applySessionDeletionMarkers(connection, postgres, affectedIDs, None)

                if groupID is not None:
                    cursor.execute("SELECT statement_timestamp()")
                    restoredAt = cursor.fetchone()[0]

                    cursor.execute(sql.SQL("UPDATE {} SET restored_at = %s WHERE deletion_group_id = %s").format(groups), (restoredAt, groupID))
                    cursor.execute(sql.SQL("UPDATE {} SET restored_at = %s WHERE deletion_group_id = %s").format(members), (restoredAt, groupID))

                return True

    # ++++++++++++++++++++++ _applySessionDeletionMarkers +++++++++++++++++++++
    # Direct callers in this module: _setSessionDeleted.
    @staticmethod
    def _applySessionDeletionMarkers(connection, postgres, sessionIDs, marker):
        """Apply coordinated markers to an already-validated session set.

        Internal write primitive. Caller must hold all lifecycle locks,
        own the transaction, and perform group/lineage validation first.

        marker is a timestamp for deletion or None for restoration.
        Does not commit, inspect payloads, or manage group records.
        """
        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")

        # Update every owner before changing links, so the existing
        # link guards see the correct session deletion marker.
        statement = sql.SQL("""
            UPDATE {}
            SET deleted_at = %s
            WHERE session_id = ANY(%s::bigint[])
              AND deleted_at IS DISTINCT FROM %s::timestamptz
        """).format(sessions)
        connection.execute(statement, (marker, sessionIDs, marker))

        statement = sql.SQL("""
            UPDATE {}
            SET deleted_at = %s
            WHERE session_id = ANY(%s::bigint[])
              AND deleted_at IS DISTINCT FROM %s::timestamptz
        """).format(links)
        connection.execute(statement, (marker, sessionIDs, marker))

        # All owners have their final markers before shared-content
        # eligibility is recomputed.
        statement = sql.SQL("""
            WITH desired AS (
                SELECT content.file_id,
                       CASE WHEN EXISTS (
                           SELECT 1
                           FROM {links} AS reference
                           JOIN {sessions} AS owner
                             ON owner.session_id = reference.session_id
                           WHERE reference.file_id = content.file_id
                             AND owner.deleted_at IS NULL
                       )
                       THEN NULL::timestamptz
                       ELSE COALESCE(content.deleted_at, %s::timestamptz)
                       END AS deleted_at
                FROM {files} AS content
                WHERE EXISTS (
                    SELECT 1
                    FROM {links} AS affected
                    WHERE affected.session_id = ANY(%s::bigint[])
                      AND affected.file_id = content.file_id
                )
            )
            UPDATE {files} AS content
            SET deleted_at = desired.deleted_at
            FROM desired
            WHERE content.file_id = desired.file_id
              AND content.deleted_at IS DISTINCT FROM desired.deleted_at
        """).format(sessions=sessions, links=links, files=files)
        connection.execute(statement, (marker, sessionIDs))

        statement = sql.SQL("""
            UPDATE {locations} AS location
            SET deleted_at = content.deleted_at
            FROM {files} AS content
            WHERE location.file_id = content.file_id
              AND EXISTS (
                  SELECT 1
                  FROM {links} AS affected
                  WHERE affected.session_id = ANY(%s::bigint[])
                    AND affected.file_id = content.file_id
              )
              AND location.deleted_at IS DISTINCT FROM content.deleted_at
        """).format(locations=locations, files=files, links=links)
        connection.execute(statement, (sessionIDs,))


    # -------------------------------------------------------------------------
    # Dataset dependency protection
    # -------------------------------------------------------------------------
    # Shared source checks used by dataset publication/restoration and session deletion.

    # +++++++++++++++++++++++ _getDatasetSessionBlockers ++++++++++++++++++++++
    # Direct callers in this module: _requireNoDatasetSessionBlockers.
    @staticmethod
    def _getDatasetSessionBlockers(connection, postgres, sessionIDs):
        """Find active datasets protecting any proposed session deletion.

        A dataset protects its exact members and every upstream ancestor.
        Historical matrix/action provenance does not create protection.

        Read-only metadata query. For an authoritative decision, the caller
        must hold the session lifecycle lock through transaction completion.
        """
        proposedIDs = list(sessionIDs)

        if any(type(value) is not int or not 0 <= value <= 9223372036854775807 for value in proposedIDs):
            raise ValueError("sessionIDs must contain nonnegative PostgreSQL bigints.")

        proposedIDs = sorted(set(proposedIDs))

        if not proposedIDs:
            return []

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        datasets = sql.Identifier(postgres.databaseSchema, "dataset")
        members = sql.Identifier(postgres.databaseSchema, "dataset_members")

        # Carry the proposed deletion ID through its descendants so each
        # result identifies both the protected ancestor and exact member.
        # UNION terminates traversal even if malformed lineage contains a cycle.
        statement = sql.SQL("""
            WITH RECURSIVE descendants(target_id, session_id) AS (
                SELECT checkpoint.session_id, checkpoint.session_id
                FROM {sessions} AS checkpoint
                WHERE checkpoint.session_id = ANY(%s::bigint[])

                UNION

                SELECT descendants.target_id, child.session_id
                FROM descendants
                JOIN {sessions} AS child
                  ON child.parent_session_id = descendants.session_id
            )
            SELECT descendants.target_id,
                   owner.dataset_id,
                   owner.name,
                   owner.checkpoint_name,
                   membership.session_id,
                   membership.entry_name
            FROM descendants
            JOIN {members} AS membership
              ON membership.session_id = descendants.session_id
            JOIN {datasets} AS owner
              ON owner.dataset_id = membership.dataset_id
            WHERE owner.status = 'complete'
              AND owner.deleted_at IS NULL
            ORDER BY owner.dataset_id,
                     membership.session_id,
                     descendants.target_id
        """).format(sessions=sessions, datasets=datasets, members=members)

        from psycopg.rows import tuple_row

        with connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(statement, (proposedIDs,))
            rows = cursor.fetchall()

        names = ("protectedSessionID", "datasetID", "datasetName", "checkpointName", "memberSessionID", "entryName")
        return [dict(zip(names, row)) for row in rows]

    # ++++++++++++++++++++ _requireNoDatasetSessionBlockers +++++++++++++++++++
    # Direct callers in this module: _getSessionDeletionRecords.
    @classmethod
    def _requireNoDatasetSessionBlockers(cls, connection, postgres, sessionIDs):
        """Refuse a complete proposed deletion set if any member is protected.

        Raises the existing lifecycle exception, which the public session
        deletion wrapper translates into pglMessages.warning.

        Does not modify database records or open scientific payloads.
        """
        blockers = cls._getDatasetSessionBlockers(connection, postgres, sessionIDs)

        if not blockers:
            return

        # One concrete blocker is sufficient to explain refusal. Report how
        # many additional protection relationships were found without printing
        # an unbounded list for large datasets.
        blocker = blockers[0]
        datasetName = cls._printValue(blocker["datasetName"])
        checkpointName = blocker["checkpointName"]
        entryName = blocker["entryName"]

        datasetLabel = f"dataset {blocker['datasetID']} ({datasetName!r})"

        if checkpointName:
            datasetLabel += f", checkpoint {cls._printValue(checkpointName)!r}"

        memberLabel = f"member session {blocker['memberSessionID']}"

        if entryName is not None:
            memberLabel += f", entryName={cls._printValue(entryName)!r}"

        extra = f" There are {len(blockers) - 1} additional blocking relationships." if len(blockers) > 1 else ""

        raise cls._SessionLifecycleBlocked(f"Cannot delete session {blocker['protectedSessionID']}: active {datasetLabel} protects {memberLabel} and its ancestors.{extra} Nothing was changed. deleteDownstreamSessions=True cannot bypass dataset protection. Removing a member in a newer dataset version does not release protection held by older active versions.")

    # +++++++++++++++++++++++ _getDatasetSourceProblems +++++++++++++++++++++++
    # Direct callers in this module: _changeDatasetDeletionState, saveDataset.
    @staticmethod
    def _getDatasetSourceProblems(connection, postgres, datasetID):
        """List members/ancestors that prevent dataset restoration.

        Returns metadata only. The caller must validate the dataset's
        existence and hold the session dependency lock when using this
        result to authorize restoration.
        """
        if type(datasetID) is not int or not 0 <= datasetID <= 9223372036854775807:
            raise ValueError("datasetID must be a nonnegative PostgreSQL bigint.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        members = sql.Identifier(postgres.databaseSchema, "dataset_members")

        statement = sql.SQL("""
            WITH RECURSIVE ancestry(member_id, source_id) AS (
                SELECT membership.session_id, membership.session_id
                FROM {members} AS membership
                WHERE membership.dataset_id = %s

                UNION

                SELECT ancestry.member_id, checkpoint.parent_session_id
                FROM ancestry
                JOIN {sessions} AS checkpoint
                  ON checkpoint.session_id = ancestry.source_id
                WHERE checkpoint.parent_session_id IS NOT NULL
            )
            SELECT ancestry.member_id,
                   membership.entry_name,
                   ancestry.source_id,
                   checkpoint.status,
                   checkpoint.deleted_at
            FROM ancestry
            JOIN {members} AS membership
              ON membership.dataset_id = %s
             AND membership.session_id = ancestry.member_id
            LEFT JOIN {sessions} AS checkpoint
              ON checkpoint.session_id = ancestry.source_id
            WHERE checkpoint.session_id IS NULL
               OR checkpoint.status <> 'complete'
               OR checkpoint.deleted_at IS NOT NULL
            ORDER BY ancestry.member_id, ancestry.source_id
        """).format(sessions=sessions, members=members)

        from psycopg.rows import tuple_row

        with connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute(statement, (datasetID, datasetID))
            rows = cursor.fetchall()

        names = ("memberSessionID", "entryName", "sourceSessionID", "status", "deletedAt")
        return [dict(zip(names, row)) for row in rows]


    # -------------------------------------------------------------------------
    # Dataset lifecycle coordination
    # -------------------------------------------------------------------------
    # Serialize dataset deletion/restoration with session lifecycle writes and revalidate required
    # sources.

    # ++++++++++++++++++++++ _changeDatasetDeletionState ++++++++++++++++++++++
    # Direct callers in this module: deleteDataset, undeleteDataset.
    @classmethod
    def _changeDatasetDeletionState(cls, datasetID, *, deleted, settings=None, settingsName=None, sslRootCert=None):
        """Coordinate a dataset lifecycle change with session lifecycle writes.

        Acquire the session dependency lock before locking the dataset row.
        This matches the protocol required for future dataset publication.

        Session deletion takes an EXCLUSIVE session-table lock, conflicting
        with this SHARE lock. Member/ancestor validation therefore remains
        valid until commit.

        Scientific payloads and membership records are never modified here.
        """
        from .pglPostgres import pglPostgres
        from psycopg.rows import tuple_row

        if type(datasetID) is not int or not 0 <= datasetID <= 9223372036854775807:
            pglMessages.warning("datasetID must be a nonnegative PostgreSQL bigint.", level=1)
            return None

        if type(deleted) is not bool:
            pglMessages.warning("deleted must be a boolean.", level=1)
            return None

        operation = "delete" if deleted else "undelete"
        outcome = None
        warning = None

        try:
            try:
                postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)
            except (TypeError, ValueError) as error:
                pglMessages.warning(f"Cannot {operation} dataset: invalid settings or configuration: {error}", level=1)
                return None

            if postgres is None:
                return None

            sessions = sql.Identifier(postgres.databaseSchema, "session")
            datasets = sql.Identifier(postgres.databaseSchema, "dataset")

            with postgres.connect(sslRootCert=sslRootCert) as connection:
                if connection is None:
                    return None

                with connection.transaction():
                    connection.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")

                    # Take this before dataset row locks and before any writes.
                    connection.execute(sql.SQL("LOCK TABLE {} IN SHARE MODE").format(sessions))

                    with connection.cursor(row_factory=tuple_row) as cursor:
                        statement = sql.SQL("SELECT status, deleted_at FROM {} WHERE dataset_id = %s FOR UPDATE").format(datasets)
                        cursor.execute(statement, (datasetID,))
                        record = cursor.fetchone()

                        if record is None:
                            outcome = "unknown"

                        elif record[0] != "complete":
                            outcome = "blocked"
                            warning = f"Cannot {operation} dataset {datasetID}: only completed dataset checkpoints support this lifecycle operation."

                        elif (record[1] is not None) == deleted:
                            outcome = "unchanged"

                        else:
                            problems = cls._getDatasetSourceProblems(connection, postgres, datasetID) if not deleted else []

                            if problems:
                                problem = problems[0]
                                sourceID = problem["sourceSessionID"]
                                memberID = problem["memberSessionID"]
                                entryName = problem["entryName"]
                                memberLabel = f"member session {memberID}"

                                if entryName is not None:
                                    memberLabel += f" (entryName={cls._printValue(entryName)!r})"

                                if problem["status"] is None:
                                    reason = "missing"
                                elif problem["deletedAt"] is not None:
                                    reason = "deleted"
                                else:
                                    reason = f"not complete (status={cls._printValue(problem['status'])!r})"

                                extra = f" There are {len(problems) - 1} additional unavailable source relationships." if len(problems) > 1 else ""
                                warning = f"Cannot undelete dataset {datasetID}: {memberLabel} requires session {sourceID}, which is {reason}.{extra} Restore the required sessions or their deletion groups first. Nothing was changed."
                                outcome = "blocked"

                            else:
                                marker = sql.SQL("statement_timestamp()") if deleted else sql.SQL("NULL")
                                statement = sql.SQL("UPDATE {} SET deleted_at = {} WHERE dataset_id = %s RETURNING dataset_id").format(datasets, marker)
                                cursor.execute(statement, (datasetID,))

                                if cursor.fetchone() != (datasetID,):
                                    raise RuntimeError("Dataset lifecycle update did not return the requested checkpoint.")

                                outcome = "changed"

        except psycopg.Error as error:
            pglMessages.warning(f"Could not {operation} dataset {datasetID}: {error}. If the connection failed during commit, check the dataset state before retrying.", level=2)
            return None

        # Report only after transaction and connection cleanup.
        if outcome == "unknown":
            pglMessages.warning(f"Unknown dataset checkpoint: {datasetID}", level=0)
            return False

        if outcome == "blocked":
            pglMessages.warning(warning, level=2)
            return False

        if outcome == "unchanged":
            return False

        if outcome != "changed":
            raise RuntimeError("Dataset lifecycle operation finished without a recognized outcome.")

        if deleted:
            pglMessages.message(f"Dataset {datasetID} soft-deleted in {postgres.databaseName}.{postgres.databaseSchema}. Its session protection is released; sessions and saved outputs are unchanged.")
        else:
            pglMessages.message(f"Dataset {datasetID} restored in {postgres.databaseName}.{postgres.databaseSchema}. Its members and ancestors are protected again.")

        return True


    # -------------------------------------------------------------------------
    # Reporting configuration and formatting
    # -------------------------------------------------------------------------
    # Shared report configuration, table formatting, and session-lineage presentation.

    # +++++++++++++++++++++++++++++ _printPostgres ++++++++++++++++++++++++++++
    # Direct callers in this module: printDatasets, printSession, printSessions, printTables.
    @staticmethod
    def _printPostgres(postgres=None, *, settings=None, settingsName=None):
        """Resolve reporting configuration without opening a connection."""
        from .pglPostgres import pglPostgres

        if postgres is not None and (settings is not None or settingsName is not None):
            raise ValueError("Supply either postgres or PGL settings, not both.")

        if postgres is None:
            postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)

        if postgres is not None and not isinstance(postgres, pglPostgres):
            raise TypeError("postgres must be a pglPostgres instance.")

        return postgres

    # ++++++++++++++++++++++++++++++ _printValue ++++++++++++++++++++++++++++++
    # Direct callers in this module: _changeDatasetDeletionState, _printTable,
    # _requireNoDatasetSessionBlockers, _sessionLineageLines, printDataset, printDatasets,
    # printSession, printSessions, setSessionDescription.
    @staticmethod
    def _printValue(value, maxWidth=None):
        """Produce single-line display text, optionally truncated."""
        text = "NULL" if value is None else str(value)
        text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")

        if maxWidth is not None and len(text) > maxWidth:
            return text[:maxWidth - 3] + "..."

        return text

    # ++++++++++++++++++++++++++++++ _printTable ++++++++++++++++++++++++++++++
    # Direct callers in this module: printDataset, printTables.
    @staticmethod
    def _printTable(headers, rows, maxWidth=48):
        """Print a compact table without accessing storage."""
        displayed = [[pglDatabase._printValue(value, maxWidth) for value in headers]]
        displayed.extend([pglDatabase._printValue(value, maxWidth) for value in row] for row in rows)
        widths = [max(len(row[index]) for row in displayed) for index in range(len(headers))]

        # ++++++++++++++++++++++++++++++++ printRow +++++++++++++++++++++++++++++++
        # Local implementation within _printTable.
        def printRow(row):
            pglMessages.print("  " + " | ".join(value.ljust(width) for value, width in zip(row, widths)))

        printRow(displayed[0])
        pglMessages.print("  " + "-+-".join("-" * width for width in widths))

        for row in displayed[1:]:
            printRow(row)

        if not rows:
            pglMessages.print("  (no rows)")

    # ++++++++++++++++++++++++++++ _printTableOrder +++++++++++++++++++++++++++
    # Direct callers in this module: printTables.
    @staticmethod
    def _printTableOrder():
        """Supported database-report tables and their stable ordering."""
        return {
            "session": ("session_id",),
            "dataset": ("dataset_id",),
            "dataset_members": ("dataset_id", "member_index"),
            "dataset_products": ("dataset_id", "product_name"),
            "matrix_revisions": ("matrix_revision_id",),
            "subjects": ("subject_id",),
            "files": ("file_id",),
            "storage_backends": ("backend_id",),
            "storage_locations": ("file_id", "backend_id", "object_key"),
            "users": ("user_id",),
            "session_files": ("session_id", "name"),
        }

    # ++++++++++++++++++++++++++ _sessionLineageLines +++++++++++++++++++++++++
    # Direct callers in this module: printSessions.
    @staticmethod
    def _sessionLineageLines(records):
        """Format visible session lineage without descriptions.

        Parentless roots use behavioral acquisition session names:
            sessionID: 92 [day1]

        Descendants use database checkpoint names:
            95 (checkpointName)

        Hidden/missing parents are not bypassed by artificial arrows.
        Iterative traversal avoids Python recursion limits.
        """
        children = {sessionID: [] for sessionID in records}

        for sessionID, record in records.items():
            parentID = record["parent_session_id"]

            if parentID in records:
                children[parentID].append(sessionID)

        for values in children.values():
            values.sort()

        roots = sorted(sessionID for sessionID, record in records.items() if record["parent_session_id"] not in records)
        visited = set()
        lines = []

        # +++++++++++++++++++++++++++++++ nodeLabel +++++++++++++++++++++++++++++++
        # Local implementation within _sessionLineageLines.
        def nodeLabel(sessionID):
            record = records[sessionID]

            if record["parent_session_id"] is None:
                label = f"sessionID: {sessionID}"
                sessionNames = record["session_names"]

                if sessionNames:
                    label += " [" + ", ".join(pglDatabase._printValue(name) for name in sessionNames) + "]"
            else:
                label = str(sessionID)
                checkpointName = record["checkpoint_name"]

                if checkpointName:
                    label += f" ({pglDatabase._printValue(checkpointName)})"

            if record["deleted_at"] is not None:
                label += " [deleted]"

            if record["status"] != "complete":
                label += f" [{pglDatabase._printValue(record['status'])}]"

            return label

        # +++++++++++++++++++++++++++++++ appendTree ++++++++++++++++++++++++++++++
        # Local implementation within _sessionLineageLines.
        def appendTree(startID, prefix):
            pending = [(startID, prefix)]

            while pending:
                currentID, currentPrefix = pending.pop()
                line = currentPrefix

                while True:
                    if currentID in visited:
                        raise ValueError("Session lineage contains a cycle or repeated node.")

                    visited.add(currentID)
                    line += nodeLabel(currentID)
                    descendants = children[currentID]

                    if not descendants:
                        lines.append(line)
                        break

                    branchPrefix = " " * (len(line) + 1) + "-> "

                    for childID in reversed(descendants[1:]):
                        pending.append((childID, branchPrefix))

                    currentID = descendants[0]
                    line += " -> "

        for sessionID in roots:
            parentID = records[sessionID]["parent_session_id"]

            if parentID is not None:
                lines.append(f"    [parent session {parentID} is not visible in this listing]")

            appendTree(sessionID, "    ")

        if len(visited) != len(records):
            raise ValueError("Session lineage contains a cycle with no visible root.")

        return lines

    # ++++++++++++++++++++++++++ _printGroupingValue ++++++++++++++++++++++++++
    # Direct callers in this module: printSessions.
    @staticmethod
    def _printGroupingValue(value, fieldName):
        """Use Unknown for unset labels, while rejecting malformed metadata."""
        if value is None:
            return "Unknown"

        if not isinstance(value, str):
            raise ValueError(f"Saved {fieldName} must be a string or None.")

        if not value.strip():
            return "Unknown"

        return value

    # ++++++++++++++++++++++ _readPrintExperimentSettings +++++++++++++++++++
    @staticmethod
    def _readPrintExperimentSettings(identity, locations, *, accessContext):
        """Read one verified experiment-settings document for a report.

        Called by printSessions() and getDatasetMemberSummaries().
        Borrows their access context so repeated file reads reuse connections.

        Tries supplied registered replicas in order. Checks bounded size and
        SHA-256 before decoding JSON. Once verified bytes are found, malformed
        JSON is a document error rather than a replica-selection failure.

        Returns a settings dictionary.
        Does not reconstruct runs, read MNE fields, query SQL, or publish data.
        Does not close the caller-owned context.
        """
        if not isinstance(accessContext, pglDatabaseAccess):
            raise TypeError("accessContext must be a pglDatabaseAccess instance.")

        maxBytes = 16 * 1024 * 1024
        expectedSize = identity["size_bytes"]
        expectedHash = identity["sha256"]

        if type(expectedSize) is not int or not 0 <= expectedSize <= maxBytes:
            raise ValueError("Experiment settings exceed the reporting size limit or have an invalid size.")

        failures = []

        for location in locations:
            backendID = location["backend_id"]
            key = location["object_key"]

            if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                raise ValueError("Database contains an invalid experiment-settings object key.")

            try:
                filesystem, root = accessContext.resolve(backendID, location["url_prefix"])

                with filesystem.open(posixpath.join(root, key), "rb") as source:
                    content = source.read(expectedSize + 1)

                if len(content) != expectedSize or hashlib.sha256(content).hexdigest() != expectedHash:
                    raise ValueError("Experiment-settings integrity mismatch.")

            except (OSError, ValueError) as error:
                failures.append(f"backend {backendID}: {type(error).__name__}")
                continue

            settings = json.loads(content.decode("utf-8"))

            if not isinstance(settings, dict):
                raise ValueError("Saved experiment settings must contain a JSON object.")

            return settings

        reason = "; ".join(failures) or "no eligible storage locations"
        raise OSError("No experiment-settings copy could be verified: " + reason)


    # -------------------------------------------------------------------------
    # Pipeline inputs, ancestry, and replay metadata
    # -------------------------------------------------------------------------
    # Resolve replay inputs and verify saved ancestry history without executing scientific actions.

    # +++++++++++++++++++++++++++ _loadPipelineInput ++++++++++++++++++++++++++
    # Checkpoint implementation support. Existing external callers may use this method; see its
    # docstring and this section's purpose.
    @classmethod
    def _loadPipelineInput(cls, sessionID, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load an active checkpoint and describe it as a pipeline input.

        Reuses the public session loader, including its validation, deferred
        loading, and connection cleanup. No SQL connection is returned.

        The database identity is the current runtime endpoint descriptor,
        not a permanent globally unique store identifier.
        """
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        session = cls.loadSession(sessionID, settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert, includeDeleted=False, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

        if session is None:
            return None, None

        boundary = cls._getCheckpointHistoryBoundary(session)
        baseline = session._checkpointBaseline

        if boundary is None or baseline["sessionID"] != sessionID or boundary != len(session.actionHistory):
            raise ValueError("Loaded pipeline input lacks a matching checkpoint history baseline.")

        inputRecord = {
            "version": 1,
            "kind": "checkpoint",
            "sessionID": sessionID,
            "databaseIdentity": deepcopy(baseline["databaseIdentity"]),
            "historyCount": boundary,
            "historyHashes": list(baseline["historyHashes"]),
        }

        return session, inputRecord

    # ++++++++++++++++++++++ _validatePipelineReplayInput +++++++++++++++++++++
    # Direct callers in this module: _selectAncestralReplayInput.
    @staticmethod
    def _validatePipelineReplayInput(record, expected=None):
        """Validate replay input metadata and optionally compare it to a plan.

        Uses metadata only; does not connect, read payloads, or mutate records.

        Endpoint identity prevents accidental use of a different configured
        database. It is not a permanent store UUID across relocation/restores.
        """
        expectedFields = {"version", "kind", "sessionID", "databaseIdentity", "historyCount", "historyHashes"}

        # ++++++++++++++++++++++++++++++++ validate +++++++++++++++++++++++++++++++
        # Local implementation within _validatePipelineReplayInput.
        def validate(value):
            if not isinstance(value, dict) or set(value) != expectedFields:
                raise ValueError("Invalid pipeline checkpoint input record.")

            if type(value["version"]) is not int or value["version"] != 1 or value["kind"] != "checkpoint":
                raise ValueError("Unsupported pipeline checkpoint input version or kind.")

            if type(value["sessionID"]) is not int or not 0 <= value["sessionID"] <= 9223372036854775807:
                raise ValueError("Invalid replay checkpoint ID.")

            identity = value["databaseIdentity"]

            if not isinstance(identity, dict):
                raise ValueError("Invalid replay database identity.")

            transport = identity.get("transport")
            identityFields = {"transport", "host", "port", "database", "schema"}

            if transport == "ssh":
                identityFields.update({"sshHost", "sshPort", "sshUser"})
            elif transport != "direct":
                raise ValueError("Unsupported replay database transport.")

            if set(identity) != identityFields:
                raise ValueError("Invalid replay database identity fields.")

            for key in ("host", "database", "schema"):
                if not isinstance(identity[key], str) or not identity[key]:
                    raise ValueError(f"Invalid replay database identity: {key}.")

            if type(identity["port"]) is not int or not 1 <= identity["port"] <= 65535:
                raise ValueError("Invalid replay database port.")

            if transport == "ssh":
                if not isinstance(identity["sshHost"], str) or not identity["sshHost"] or not isinstance(identity["sshUser"], str):
                    raise ValueError("Invalid replay SSH identity.")

                if type(identity["sshPort"]) is not int or not 1 <= identity["sshPort"] <= 65535:
                    raise ValueError("Invalid replay SSH port.")

            count = value["historyCount"]
            hashes = value["historyHashes"]

            if type(count) is not int or count < 0 or not isinstance(hashes, list) or len(hashes) != count:
                raise ValueError("Invalid replay history boundary.")

            if any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in hashes):
                raise ValueError("Invalid replay history fingerprint.")

        validate(record)

        if expected is not None:
            validate(expected)

            if record["databaseIdentity"] != expected["databaseIdentity"]:
                raise ValueError("The configured database differs from the pipeline's recorded replay database. Select the original settings, or explicitly supply sessionID to choose a different input.")

            if record["sessionID"] != expected["sessionID"]:
                raise ValueError("Loaded checkpoint differs from the pipeline's recorded replay checkpoint.")

            if record["historyCount"] != expected["historyCount"] or record["historyHashes"] != expected["historyHashes"]:
                raise ValueError("Loaded checkpoint history differs from the recorded replay input. Execution was not started.")

        return record

    # ++++++++++++++++++++++ _readReplayCheckpointHistory +++++++++++++++++++++
    # Direct callers in this module: getSessionPipeline.
    @classmethod
    def _readReplayCheckpointHistory(cls, connection, postgres, checkpoint, *, storageOptionsByBackend=None):
        """Read history metadata without constructing runs or MNE objects.

        checkpoint is a validated ancestry record from getSessionPipeline().
        Reject contracts without a recorded session-history boundary.
        """
        from io import BytesIO, TextIOWrapper

        sessionID = checkpoint["sessionID"]
        origin = checkpoint["origin"]
        contract = checkpoint["contractVersion"]

        if (origin, contract) == ("raw", "pgl.raw.behavior.v1"):
            raise ValueError(f"Raw checkpoint {sessionID} uses pgl.raw.behavior.v1 and has no session-level history boundary. Automatic ancestry replay cannot safely distinguish acquisition loading from later processing for this contract.")

        supported = {("raw", "pgl.raw.session.v1"), ("derived", "pgl.derived.session.v1"), ("derived", "pgl.derived.behavior.v1")}

        if (origin, contract) not in supported:
            raise ValueError(f"Checkpoint {sessionID} has an unsupported replay contract: {origin}, {contract}.")

        manifestBytes, manifestIdentity = cls._readReplayMetadataFile(connection, postgres, sessionID, "manifest.json", storageOptionsByBackend=storageOptionsByBackend, maxBytes=16 * 1024 * 1024)

        if manifestIdentity["sha256"] != checkpoint["manifestHash"]:
            raise ValueError(f"Checkpoint {sessionID}: manifest link differs from its checkpoint identity.")

        manifest = json.loads(manifestBytes.decode("utf-8"))
        cls._validateSessionBundleManifest(manifest, origin=origin)

        if manifest["contract_version"] != contract or cls.encodeRawManifest(manifest) != manifestBytes:
            raise ValueError(f"Checkpoint {sessionID}: manifest contract or canonical encoding differs.")

        historyRecords = [record for record in manifest["files"] if record["name"] == "history.json"]

        if len(historyRecords) != 1:
            raise ValueError(f"Checkpoint {sessionID}: manifest must contain exactly one history.json.")

        historyBytes, historyIdentity = cls._readReplayMetadataFile(connection, postgres, sessionID, "history.json", storageOptionsByBackend=storageOptionsByBackend)
        expectedHistory = historyRecords[0]

        if historyIdentity != {"sha256": expectedHistory["sha256"], "size_bytes": expectedHistory["size_bytes"]}:
            raise ValueError(f"Checkpoint {sessionID}: registered history differs from the manifest.")

        class HistoryView:
            """Expose only the already-verified history bytes to the reader."""

            # ++++++++++++++++++++++++++++++++++ open +++++++++++++++++++++++++++++++++
            # Local implementation within _readReplayCheckpointHistory.
            def open(self, path, mode="rb", **kwargs):
                if path != "history.json":
                    raise ValueError("Replay history reader requested an unexpected file.")

                if mode == "rb":
                    return BytesIO(historyBytes)

                if mode in ("r", "rt"):
                    return TextIOWrapper(BytesIO(historyBytes), encoding=kwargs.get("encoding", "utf-8"))

                raise ValueError("Replay history is read-only.")

        # Reuse the existing history/label compatibility checks.
        history, metadata = cls._readSessionHistory(HistoryView(), len(manifest["runs"]))
        return history

    # ++++++++++++++++++++++++ _readReplayMetadataFile +++++++++++++++++++++++
    @classmethod
    def _readReplayMetadataFile(cls, connection, postgres, sessionID, name, *, storageOptionsByBackend=None, maxBytes=64 * 1024 * 1024):
        """Read a verified checkpoint metadata file for replay inspection.

        Called by _readReplayCheckpointHistory().
        Only manifest.json and history.json are permitted.

        The caller owns the SQL snapshot and validates checkpoint eligibility.
        This method verifies the active file link, expected role, registered
        content identity, and bytes read from an eligible replica.

        Server-local roots resolve through the caller's local/SSH context.
        Explicit remote URLs retain their own filesystem protocol.
        Does not fall back from SSH to client-local file access.

        Returns:
            (verifiedBytes, {"sha256": digest, "size_bytes": size})

        Owns a short-lived payload access context and closes it on both success
        and failure. Returned bytes do not depend on an open filesystem.

        Does not reconstruct scientific objects, execute actions, publish
        records, or modify payloads. Does not decode the returned JSON.

        Read/integrity failures may advance to another registered replica.
        Other connection and security errors propagate.
        """
        if name not in ("manifest.json", "history.json"):
            raise ValueError("Replay inspection may read only manifest.json or history.json.")

        if type(maxBytes) is not int or maxBytes <= 0:
            raise ValueError("maxBytes must be a positive integer.")

        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("""
            SELECT content.file_id,
                   content.sha256,
                   content.size_bytes,
                   link.role
            FROM {} AS link
            JOIN {} AS content ON content.file_id = link.file_id
            WHERE link.session_id = %s
              AND link.name = %s
              AND link.deleted_at IS NULL
              AND content.deleted_at IS NULL
        """).format(links, files)

        record = connection.execute(statement, (sessionID, name)).fetchone()

        if record is None:
            raise ValueError(f"Checkpoint {sessionID} has no active registered {name}.")

        fileID, digest, size, role = record
        expectedRole = "manifest" if name == "manifest.json" else "history"

        if role != expectedRole:
            raise ValueError(f"Checkpoint {sessionID}: {name} has an unexpected role.")

        if type(size) is not int or not 0 <= size <= maxBytes:
            raise ValueError(f"Checkpoint {sessionID}: {name} exceeds the replay metadata size limit or has an invalid size.")

        statement = sql.SQL("""
            SELECT backend.backend_id,
                   backend.url_prefix,
                   location.object_key
            FROM {} AS location
            JOIN {} AS backend ON backend.backend_id = location.backend_id
            WHERE location.file_id = %s
              AND location.deleted_at IS NULL
              AND location.status = 'present'
              AND location.verified_at IS NOT NULL
              AND backend.is_active
            ORDER BY location.priority, backend.backend_id, location.object_key
        """).format(locations, backends)

        candidates = connection.execute(statement, (fileID,)).fetchall()
        failures = []

        # No deferred reader escapes this method. Close owned SSH connections
        # before returning verified in-memory bytes.
        with pglDatabaseAccess(postgres, storageOptionsByBackend) as access:
            for backendID, registeredRoot, objectKey in candidates:
                if not isinstance(objectKey, str) or objectKey.startswith("/") or "\\" in objectKey or "\x00" in objectKey or any(part in {"", ".", ".."} for part in objectKey.split("/")):
                    raise ValueError("Database contains an invalid replay metadata object key.")

                try:
                    filesystem, root = access.resolve(backendID, registeredRoot)

                    with filesystem.open(posixpath.join(root, objectKey), "rb") as source:
                        content = source.read(size + 1)

                    if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError("Metadata content identity mismatch.")

                except (OSError, ValueError) as error:
                    failures.append(f"backend {backendID}: {type(error).__name__}")
                    continue

                return content, {"sha256": digest, "size_bytes": size}

        reason = "; ".join(failures) or "no eligible storage locations"
        raise OSError(f"Cannot verify checkpoint {sessionID} {name}: {reason}")
    
    # ++++++++++++++++++++++ _selectAncestralReplayInput ++++++++++++++++++++++
    # Direct callers in this module: getSessionPipeline.
    @classmethod
    def _selectAncestralReplayInput(cls, session, checkpoints, postgres):
        """Validate root-to-current ancestry histories and choose the raw input.

        checkpoints contains records in root-to-current order, each with
        verified historyHashes.

        This checks recorded provenance consistency, not scientific equivalence
        or the absence of unrecorded in-memory mutations.
        """
        boundary = cls._getCheckpointHistoryBoundary(session)

        if boundary is None:
            raise ValueError("This working session has no verified checkpoint baseline. Reload its saved checkpoint before requesting ancestry-based extraction.")

        baseline = session._checkpointBaseline
        identity = cls._checkpointDatabaseIdentity(postgres)

        if baseline["databaseIdentity"] != identity:
            raise ValueError("Current PGL settings point to a different database than this session's checkpoint baseline.")

        if not checkpoints:
            raise ValueError("No checkpoint ancestry was supplied.")

        current = checkpoints[-1]

        if current["sessionID"] != baseline["sessionID"] or session.sessionID != baseline["sessionID"]:
            raise ValueError("Working session ID, baseline, and checkpoint ancestry do not agree.")

        if current["historyHashes"] != baseline["historyHashes"]:
            raise ValueError("Saved checkpoint history differs from the working session's baseline.")

        root = checkpoints[0]

        if root["origin"] != "raw" or root["parentSessionID"] is not None or root["rootSessionID"] != root["sessionID"]:
            raise ValueError("Replay ancestry does not begin with a parentless raw root.")

        if root["contractVersion"] != "pgl.raw.session.v1":
            raise ValueError("Automatic ancestry replay requires a raw contract with recorded session history.")

        seen = set()
        previous = None

        for checkpoint in checkpoints:
            sessionID = checkpoint["sessionID"]

            if sessionID in seen:
                raise ValueError("Checkpoint ancestry contains a cycle or repeated checkpoint.")

            seen.add(sessionID)

            if checkpoint["status"] != "complete" or checkpoint["deletedAt"] is not None:
                raise ValueError(f"Replay ancestor {sessionID} is incomplete or deleted.")

            if checkpoint["rootSessionID"] != root["sessionID"]:
                raise ValueError("Checkpoint ancestry has inconsistent root IDs.")

            if previous is not None:
                if checkpoint["origin"] != "derived" or checkpoint["parentSessionID"] != previous["sessionID"]:
                    raise ValueError("Checkpoint ancestry contains a broken parent relationship.")

                prefix = previous["historyHashes"]
                childHistory = checkpoint["historyHashes"]

                if len(childHistory) < len(prefix) or childHistory[:len(prefix)] != prefix:
                    raise ValueError(f"Checkpoint {sessionID} does not preserve its parent's recorded history prefix. Refusing to infer a replay sequence.")

            previous = checkpoint

        inputRecord = {
            "version": 1,
            "kind": "checkpoint",
            "sessionID": root["sessionID"],
            "databaseIdentity": deepcopy(identity),
            "historyCount": len(root["historyHashes"]),
            "historyHashes": list(root["historyHashes"]),
        }

        cls._validatePipelineReplayInput(inputRecord)
        return inputRecord


    # -------------------------------------------------------------------------
    # Schema initialization, migrations, and application grants
    # -------------------------------------------------------------------------
    # Local administrator-only schema setup. Existing migration names and checksums remain
    # unchanged.

    # ++++++++++++++++++++++++++ _initDatabaseSchema ++++++++++++++++++++++++++
    # Direct callers in this module: initDatabase.
    @classmethod
    def _initDatabaseSchema(cls, postgres=None, *, settings=None, settingsName=None):
        """Apply schema migrations using saved administrator credentials.

        Returns False on missing configuration, cancellation, or an
        operational connection failure. Schema/ownership errors raise.
        """
        from .pglPostgres import pglPostgres

        if postgres is not None and (settings is not None or settingsName is not None):
            raise ValueError("Supply either postgres or PGL settings, not both.")

        if postgres is None:
            postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)

            if postgres is None:
                return False

        if not isinstance(postgres, pglPostgres):
            raise TypeError("postgres must be a pglPostgres instance.")

        postgres._validateValues()
        migrations = cls._readMigrations()

        try:
            with postgres._localConnect(database=postgres.databaseName) as connection:
                if connection is None:
                    return False

                with connection.transaction():
                    connection.execute("SELECT pg_advisory_xact_lock(%s)", (cls.MIGRATION_LOCK,))
                    cls._prepareSchema(connection, postgres)
                    applied = cls._applyMigrations(connection, postgres, migrations)
                    cls._grantApplicationAccess(connection, postgres)

        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not complete schema initialization for {postgres.databaseName!r}. If the connection was lost during commit, rerun initialization to check migration state.\n{error}", level=0)
            return False

        if applied:
            pglMessages.message(f"Applied migrations: {', '.join(applied)}")
        else:
            pglMessages.message("Database schema is already current.")

        pglMessages.message(f"Scientific schema ready: {postgres.databaseName}.{postgres.databaseSchema}")
        return True

    # ++++++++++++++++++++++++++++ _readMigrations ++++++++++++++++++++++++++++
    # Direct callers in this module: _initDatabaseSchema.
    @classmethod
    def _readMigrations(cls):
        """Read and checksum the SQL migration files shipped with pglStorage."""
        directory = Path(pglInfo.getPGLDir()) / "database" / "migrations"

        if not directory.is_dir():
            raise FileNotFoundError(f"Migration directory not found: {directory}")

        migrations = []

        for path in sorted(directory.glob("*.sql")):
            match = re.fullmatch(r"(\d{4})_([a-z0-9_]+)\.sql", path.name)

            if match is None:
                raise ValueError(f"Invalid migration filename: {path.name}")

            content = path.read_bytes()

            migrations.append({
                "version": int(match.group(1)),
                "name": path.stem,
                "checksum": hashlib.sha256(content).hexdigest(),
                "sql": content.decode("utf-8"),
            })

        if not migrations:
            raise ValueError(f"No SQL migrations found in {directory}")

        versions = [migration["version"] for migration in migrations]

        if versions != list(range(1, len(migrations) + 1)):
            raise ValueError("Migration numbers must be unique and consecutive, starting at 0001.")

        return migrations

    # +++++++++++++++++++++++++++++ _prepareSchema ++++++++++++++++++++++++++++
    # Direct callers in this module: _initDatabaseSchema.
    @classmethod
    def _prepareSchema(cls, connection, postgres):
        """Ensure an administrator-owned schema and migration history table."""
        actualAdmin = connection.execute("SELECT current_user").fetchone()[0]

        if actualAdmin != postgres.adminUser:
            raise RuntimeError(f"Connected as {actualAdmin!r}, not configured administrator {postgres.adminUser!r}.")

        role = connection.execute("SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname = %s", (postgres.databaseUser,)).fetchone()

        if role is None:
            raise RuntimeError("Application role is missing. Run postgres.createDatabase() first.")

        if any(role[:5]) or not role[5]:
            raise RuntimeError("Application role has unexpected privileges or cannot log in.")

        memberships = connection.execute("SELECT 1 FROM pg_auth_members WHERE member = (SELECT oid FROM pg_roles WHERE rolname = %s) LIMIT 1", (postgres.databaseUser,)).fetchone()

        if memberships is not None:
            raise RuntimeError("Application role has role memberships requiring review.")

        schema = sql.Identifier(postgres.databaseSchema)
        admin = sql.Identifier(postgres.adminUser)
        application = sql.Identifier(postgres.databaseUser)

        schemaOwner = connection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (postgres.databaseSchema,)).fetchone()

        if schemaOwner is None:
            connection.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(schema, admin))
        elif schemaOwner[0] != postgres.adminUser:
            raise RuntimeError(f"Schema {postgres.databaseSchema!r} is owned by {schemaOwner[0]!r}, not {postgres.adminUser!r}. Ownership will not be changed automatically.")

        connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(schema))
        connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}").format(schema, application))

        # All unqualified objects in migration SQL belong to this schema.
        connection.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(schema))

        migrationTable = sql.Identifier(postgres.databaseSchema, "schema_migrations")

        connection.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                version integer PRIMARY KEY CHECK (version > 0),
                name text NOT NULL UNIQUE,
                checksum char(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{{64}}$'),
                applied_at timestamptz NOT NULL DEFAULT now(),
                applied_by text NOT NULL DEFAULT current_user
            )
        """).format(migrationTable))

    # ++++++++++++++++++++++++++++ _applyMigrations +++++++++++++++++++++++++++
    # Direct callers in this module: _initDatabaseSchema.
    @classmethod
    def _applyMigrations(cls, connection, postgres, migrations):
        """Validate history and apply pending migrations in the caller's transaction."""
        migrationTable = sql.Identifier(postgres.databaseSchema, "schema_migrations")
        history = connection.execute(sql.SQL("SELECT version, name, checksum FROM {} ORDER BY version").format(migrationTable)).fetchall()

        availableVersions = [migration["version"] for migration in migrations]
        appliedVersions = [row[0] for row in history]

        if appliedVersions != availableVersions[:len(appliedVersions)]:
            raise RuntimeError("Database migration history does not match the available migration files. Check that the correct pgl version is installed.")

        appliedByVersion = {version: (name, checksum) for version, name, checksum in history}

        # Verify all applied migrations before running any new ones.
        for migration in migrations:
            version = migration["version"]

            if version in appliedByVersion:
                expected = (migration["name"], migration["checksum"])

                if appliedByVersion[version] != expected:
                    raise RuntimeError(f"Previously applied migration has changed: {migration['name']}. Restore the original file and put changes in a new migration.")

        newlyApplied = []

        for migration in migrations:
            if migration["version"] in appliedByVersion:
                continue

            # Migration files are trusted repository code, not user input.
            connection.execute(migration["sql"], prepare=False)
            connection.execute(sql.SQL("INSERT INTO {} (version, name, checksum) VALUES (%s, %s, %s)").format(migrationTable), (migration["version"], migration["name"], migration["checksum"]))
            newlyApplied.append(migration["name"])

        return newlyApplied

    # ++++++++++++++++++++++++ _grantApplicationAccess ++++++++++++++++++++++++
    # Direct callers in this module: _initDatabaseSchema.
    # Refactor correction: explicitly grant application access to session_products.
    @classmethod
    def _grantApplicationAccess(cls, connection, postgres):
        """Grant application access after pending migrations are applied.

        Dataset products and membership are protected by publication guards.
        Matrix revision records permit registration and reads, not mutation.
        Reporting views and migration history remain read-only.
        """
        schema = sql.Identifier(postgres.databaseSchema)
        application = sql.Identifier(postgres.databaseUser)

        connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, application))

        applicationTables = [
            "users",
            "subjects",
            "storage_backends",
            "files",
            "storage_locations",
            "session",
            "session_files",
            "session_products",
            "session_deletion_groups",
            "session_deletion_group_members",
            "dataset",
            "dataset_members",
            "dataset_products",
        ]

        for name in applicationTables:
            table = sql.Identifier(postgres.databaseSchema, name)
            connection.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {} TO {}").format(table, application))

        # Published matrix records are append-only. Triggers additionally
        # reject mutation by callers with broader table permissions.
        matrixTable = sql.Identifier(postgres.databaseSchema, "matrix_revisions")
        connection.execute(sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {} FROM {}").format(matrixTable, application))
        connection.execute(sql.SQL("GRANT SELECT, INSERT ON TABLE {} TO {}").format(matrixTable, application))

        readOnlyTables = [
            "schema_migrations",
            "session_subjects",
            "raw_sessions",
            "session_subjects_including_deleted",
            "raw_sessions_including_deleted",
        ]

        for name in readOnlyTables:
            table = sql.Identifier(postgres.databaseSchema, name)
            connection.execute(sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {} FROM {}").format(table, application))
            connection.execute(sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(table, application))

        connection.execute(sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(schema, application))


    # -------------------------------------------------------------------------
    # Lifecycle exception types
    # -------------------------------------------------------------------------
    # Internal refusal categories translated into user-facing warnings by lifecycle wrappers.

    # ++++++++++++++++++++++++ _UnknownLifecycleSession +++++++++++++++++++++++
    class _UnknownLifecycleSession(ValueError):
        """Requested lifecycle session does not exist."""

    # ++++++++++++++++++++++++ _SessionLifecycleBlocked +++++++++++++++++++++++
    class _SessionLifecycleBlocked(ValueError):
        """Lifecycle operation was refused without making changes."""

# =========================================================================
# RUNTIME PAYLOAD ACCESS CONTEXT
# =========================================================================

class pglDatabaseAccess:
    """Retain caller-specific payload access independently of SQL connections.

    Created by database readers and publication coordinators.

    Owns:
        A snapshot of the caller's PostgreSQL access configuration.
        Runtime filesystem options indexed by backend ID.
        Cached filesystem/root pairs for registered backend locations.
        Inferred SSH connections created through the backend resolver.

    Does not:
        Connect to PostgreSQL.
        Change registered roots or relative object keys.
        Register database records.
        Belong in scientific serialization.

    Lifetime:
        Short-lived operations close the context after completing all I/O.
        Deferred checkpoint readers retain the context until no reader needs
        its filesystems. Closing SQL does not close this context.

        Do not close a context while lazy matrices, runs, or MNE readers still
        depend on it. Context-manager use is appropriate only when all payload
        access finishes within the with block.

    Explicit-URL filesystems may be shared by fsspec's instance cache.
    This context closes only inferred SSH connections that it owns.
    """

    # +++++++++++++++++++++++++++++ __init__ ++++++++++++++++++++++++++++++++
    def __init__(self, postgres, storageOptionsByBackend=None):
        """Snapshot access configuration without opening any connection.

        Called by pglDatabase before resolving registered payload roots.

        Backend option dictionaries are shallow-copied. Credential objects
        and nested option values must not be modified while the context is
        in use; they are not serialized or deep-copied.

        Raises for invalid configuration or malformed backend option mappings.
        """
        from threading import RLock

        from .pglPostgres import pglPostgres

        if not isinstance(postgres, pglPostgres):
            raise TypeError("postgres must be a pglPostgres instance.")

        postgres._validateValues()

        if storageOptionsByBackend is None:
            storageOptionsByBackend = {}

        if not isinstance(storageOptionsByBackend, dict):
            raise TypeError("storageOptionsByBackend must be a dictionary or None.")

        options = {}

        for backendID, backendOptions in storageOptionsByBackend.items():
            if type(backendID) is not int or not 0 <= backendID <= 9223372036854775807:
                raise ValueError("Backend option keys must be nonnegative PostgreSQL bigints.")

            if not isinstance(backendOptions, dict):
                raise TypeError("Each backend's storage options must be a dictionary.")

            options[backendID] = dict(backendOptions)

        # Copy configuration values rather than retaining the editable caller
        # object. Do not copy a live SQL connection or SSH tunnel.
        self._postgres = pglPostgres.fromJSONdict(postgres.toJSONdict())
        self._postgres._configurationLoadedRemotely = getattr(postgres, "_configurationLoadedRemotely", False)

        self._optionsByBackend = options
        self._resolved = {}
        self._ownedFilesystems = []
        self._closed = False
        self._lock = RLock()

    # +++++++++++++++++++++++++++++ resolve +++++++++++++++++++++++++++++++++
    def resolve(self, backendID, registeredRoot):
        """Return cached filesystem access for a registered backend root.

        Called by payload readers and writers before appending object keys.

        Returns:
            (filesystem, filesystemRoot)

        First access delegates to pglDatabase._resolveBackendAccess().
        Repeated access to the same backend/root reuses that filesystem.

        The lock prevents concurrent first accesses from creating duplicate
        connections. It does not serialize subsequent scientific reads or
        authorize closing the context concurrently with active I/O.

        Failed resolutions are not cached and never fall back to local access.
        """
        from urllib.parse import urlsplit

        if type(backendID) is not int or not 0 <= backendID <= 9223372036854775807:
            raise ValueError("backendID must be a nonnegative PostgreSQL bigint.")

        if not isinstance(registeredRoot, str) or not registeredRoot.strip():
            raise ValueError("registeredRoot must be a nonempty string.")

        key = (backendID, registeredRoot)

        with self._lock:
            if self._closed:
                raise ValueError("Database payload access context is closed.")

            if key in self._resolved:
                return self._resolved[key]

            options = self._optionsByBackend.get(backendID, {})
            filesystem, filesystemRoot = pglDatabase._resolveBackendAccess(self._postgres, registeredRoot, storageOptions=options)

            # The resolver creates a privately owned strict SFTP filesystem
            # only for server-local roots accessed through the SQL SSH context.
            # Explicit remote URLs continue to use fsspec's ownership model.
            explicitURL = "://" in registeredRoot
            serverLocal = not explicitURL or urlsplit(registeredRoot).scheme.lower() in {"file", "local"}
            owned = serverLocal and self._postgres.connectionTransport == "ssh"

            if owned:
                self._ownedFilesystems.append(filesystem)

            self._resolved[key] = (filesystem, filesystemRoot)
            return self._resolved[key]

    # +++++++++++++++++++++++++++++++ close +++++++++++++++++++++++++++++++++
    def close(self):
        """Release owned inferred-SSH connections without deleting anything.

        Idempotent. No SQL or catalog writes are performed.
        Explicit-URL filesystems are not closed because fsspec may share them.

        Closing invalidates dependent deferred readers. Call only after they
        have finished. Cleanup failures warn without replacing an earlier
        publication or scientific error.
        """
        with self._lock:
            if self._closed:
                return

            self._closed = True
            ownedFilesystems = self._ownedFilesystems
            self._ownedFilesystems = []
            self._resolved = {}
            self._optionsByBackend = {}

        for filesystem in reversed(ownedFilesystems):
            try:
                filesystem.close()
            except Exception as error:
                pglMessages.warning(f"Could not close a database payload SSH connection ({type(error).__name__}).", level=1)

    # +++++++++++++++++++++++++++++ __enter__ +++++++++++++++++++++++++++++++
    def __enter__(self):
        """Support bounded operations whose payload I/O finishes in the block."""
        with self._lock:
            if self._closed:
                raise ValueError("Database payload access context is closed.")

        return self

    # +++++++++++++++++++++++++++++ __exit__ ++++++++++++++++++++++++++++++++
    def __exit__(self, exceptionType, exception, traceback):
        """Release owned connections without suppressing caller errors."""
        self.close()
        return False

# =========================================================================
# READ-ONLY LOGICAL FILESYSTEM FOR DATABASE CHECKPOINTS
# =========================================================================
# Constructed by checkpoint loaders and registration validators after
# manifest validation and location resolution. It does not query SQL.
# Payload bytes are verified when opened. Matrix Zarr bundles are opened
# separately through the registered matrix-product loading path.

class pglCheckpointFilesystem(AbstractFileSystem):
    """Expose verified logical checkpoint files independently of storage paths.

    Constructed by checkpoint loaders and registration validators.

    accessContext:
        Caller-specific pglDatabaseAccess retained for deferred payload reads.
        When supplied, all physical locations resolve through this context.
        Failed remote access never falls back to client-local access.

    ownsAccess:
        True when this filesystem is responsible for closing accessContext.
        False when a publication coordinator or another owner manages it.

    Without an access context, retain the previous explicit-filesystem behavior
    for compatibility with local reconstruction callers. Bare paths then mean
    caller-local paths. Database remote loaders must always supply a context.

    Construction validates metadata but does not read payload bytes.
    Opening a file verifies its complete contents before returning a stream.
    Matrix Zarr bundles are resolved separately by matrix-product loaders.

    Keep this filesystem and its access context open while deferred session
    readers depend on them. Closing a SQL connection does not close either.
    """

    protocol = "pglcheckpoint"
    cachable = False

    # +++++++++++++++++++++++++++++ __init__ ++++++++++++++++++++++++++++++++
    def __init__(self, manifest, fileLocations, storageOptionsByBackend=None, *, accessContext=None, ownsAccess=False):
        """Validate a checkpoint inventory and retain its runtime access context.

        Returns a read-only filesystem with no scientific fields reconstructed.
        Copies inventory metadata so later caller edits do not alter identities.

        When accessContext is supplied, configure backend options on that
        context rather than supplying another options collection here.

        Ownership transfers only after successful construction. If construction
        raises, the caller remains responsible for closing its context.
        """
        super().__init__()

        if type(ownsAccess) is not bool:
            raise ValueError("ownsAccess must be a boolean.")

        if accessContext is not None and not isinstance(accessContext, pglDatabaseAccess):
            raise TypeError("accessContext must be a pglDatabaseAccess instance.")

        if ownsAccess and accessContext is None:
            raise ValueError("ownsAccess requires an access context.")

        if accessContext is not None and storageOptionsByBackend:
            raise ValueError("Configure storage options on accessContext, not on both objects.")

        if not isinstance(manifest, dict):
            raise ValueError("Manifest must contain a dictionary.")

        contractVersion = manifest.get("contract_version")

        if contractVersion == "pgl.raw.behavior.v1":
            pglDatabase.validateRawManifest(manifest)
        elif contractVersion == "pgl.raw.session.v1":
            pglDatabase.validateRawSessionManifest(manifest)
        elif contractVersion in (
            "pgl.derived.behavior.v1",
            "pgl.derived.behavior.v2",
            "pgl.derived.session.v1",
            "pgl.derived.session.v2",
        ):
            pglDatabase.validateDerivedManifest(manifest)
        else:
            raise ValueError(f"Unsupported checkpoint contract: {contractVersion!r}")

        expectedFiles = {record["name"]: record for record in manifest["files"]}

        if set(fileLocations) != set(expectedFiles):
            raise ValueError("File locations must cover exactly the manifest files.")

        self._payloads = deepcopy(fileLocations)
        self._storageOptionsByBackend = dict(storageOptionsByBackend or {})
        self._accessContext = accessContext
        self._ownsAccess = ownsAccess
        self._pglClosed = False

        for name, record in self._payloads.items():
            expected = expectedFiles[name]

            if record["sha256"] != expected["sha256"] or record["size_bytes"] != expected["size_bytes"]:
                raise ValueError(f"File identity does not match manifest: {name!r}")

            if not isinstance(record["locations"], list) or not record["locations"]:
                raise ValueError(f"File has no location candidates: {name!r}")

            for location in record["locations"]:
                key = location["object_key"]

                if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                    raise ValueError(f"Invalid payload object key: {key!r}")

                root = location["url_prefix"]

                if not isinstance(root, str) or not root.strip():
                    raise ValueError(f"Invalid storage root for {name!r}")

        self._entries = {"": {"name": "", "type": "directory", "size": 0}}

        # ++++++++++++++++++++++++++ validateName +++++++++++++++++++++++++++
        def validateName(name):
            """Validate a logical checkpoint path without accessing storage."""
            if not isinstance(name, str) or not name:
                raise ValueError("Checkpoint paths must be nonempty strings.")

            if name.startswith("/") or "\\" in name or "\x00" in name:
                raise ValueError(f"Invalid checkpoint path: {name!r}")

            if any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError(f"Invalid checkpoint path: {name!r}")

        # ++++++++++++++++++++++++++++ addEntry +++++++++++++++++++++++++++++
        def addEntry(name, kind, size):
            """Install one unique inventory entry."""
            validateName(name)

            if name in self._entries:
                raise ValueError(f"Duplicate checkpoint path: {name!r}")

            self._entries[name] = {"name": name, "type": kind, "size": size}

        for name in manifest["directories"]:
            addEntry(name, "directory", 0)

        for record in manifest["files"]:
            size = record["size_bytes"]

            if type(size) is not int or size < 0:
                raise ValueError(f"Invalid file size for {record['name']!r}")

            addEntry(record["name"], "file", size)

        for name in self._entries:
            if not name:
                continue

            parentEntry = self._entries.get(posixpath.dirname(name))

            if parentEntry is None or parentEntry["type"] != "directory":
                raise ValueError(f"Missing parent directory for checkpoint path: {name!r}")

    # +++++++++++++++++++++++++++ _logicalPath ++++++++++++++++++++++++++++++
    def _logicalPath(self, path):
        """Normalize a caller's logical lookup path, not a physical file path."""
        return self._strip_protocol(str(path)).strip("/")

    # +++++++++++++++++++++++++++++++ info ++++++++++++++++++++++++++++++++++
    def info(self, path, **kwargs):
        """Return inventory metadata without checking physical availability."""
        name = self._logicalPath(path)

        if name not in self._entries:
            raise FileNotFoundError(name)

        return dict(self._entries[name])

    # ++++++++++++++++++++++++++++++++ ls +++++++++++++++++++++++++++++++++++
    def ls(self, path, detail=True, **kwargs):
        """List logical children from the validated inventory without I/O."""
        entry = self.info(path)
        name = entry["name"]

        if entry["type"] == "file":
            entries = [entry]
        else:
            entries = [
                dict(child)
                for childName, child in self._entries.items()
                if childName and posixpath.dirname(childName) == name
            ]
            entries.sort(key=lambda child: child["name"])

        return entries if detail else [child["name"] for child in entries]

    # +++++++++++++++++++++++++ _resolveLocation ++++++++++++++++++++++++++++
    def _resolveLocation(self, location):
        """Resolve a physical replica using this reader's retained context.

        Used by verified opens and direct-local MNE eligibility checks.
        With a context, never independently reinterpret the registered root.
        """
        if self._pglClosed:
            raise ValueError("Checkpoint filesystem is closed.")

        backendID = location["backend_id"]
        registeredRoot = location["url_prefix"]

        if self._accessContext is not None:
            return self._accessContext.resolve(backendID, registeredRoot)

        # Compatibility path for existing local/explicit-URL reconstruction.
        # Remote database loaders must supply their access context.
        options = self._storageOptionsByBackend.get(backendID, {})
        return url_to_fs(registeredRoot, **options)

    # +++++++++++++++++++++++++++++++ _open +++++++++++++++++++++++++++++++++
    def _open(self, path, mode="rb", **kwargs):
        """Return fully verified bytes, trying eligible replicas in order.

        Reads each attempted replica into bounded-memory temporary storage.
        Large files spill to a caller-local temporary file.

        Returns an independent seekable binary stream. That stream no longer
        needs the SSH connection after verification has completed.

        Read and integrity failures may advance to another replica.
        Authentication/security exceptions outside those categories propagate.
        """
        if self._pglClosed:
            raise ValueError("Checkpoint filesystem is closed.")

        if mode != "rb":
            raise PermissionError("Checkpoint filesystems are read-only.")

        entry = self.info(path)

        if entry["type"] != "file":
            raise IsADirectoryError(entry["name"])

        record = self._payloads[entry["name"]]
        failures = []

        for location in record["locations"]:
            backendID = location["backend_id"]
            buffer = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")

            try:
                filesystem, root = self._resolveLocation(location)
                payloadPath = posixpath.join(root, location["object_key"])
                digest = hashlib.sha256()
                sizeBytes = 0

                with filesystem.open(payloadPath, "rb") as source:
                    while True:
                        chunk = source.read(1024 * 1024)

                        if not chunk:
                            break

                        sizeBytes += len(chunk)

                        if sizeBytes > record["size_bytes"]:
                            raise ValueError("Payload exceeds expected size.")

                        digest.update(chunk)

                        if buffer.write(chunk) != len(chunk):
                            raise OSError("Incomplete checkpoint verification-buffer write.")

                if sizeBytes != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError("Payload integrity check failed.")

                buffer.seek(0)

            except (OSError, ValueError) as error:
                buffer.close()
                failures.append(f"backend {backendID}: {type(error).__name__}")
                continue

            except BaseException:
                buffer.close()
                raise

            return buffer

        raise OSError(f"No verified copy available for {entry['name']!r}: {'; '.join(failures)}")

    # +++++++++++++++++++++++++ localBundlePaths ++++++++++++++++++++++++++++
    def localBundlePaths(self, resourceRoot, records):
        """Return a layout-compatible client-local bundle, or None.

        Called by MNE resolution only when direct-local access is requested.

        Checks inner/outer identities and resolves actual caller-side access.
        An SSH-backed server-local root is not eligible for direct-local use,
        even during a loopback test where the same path exists on the client.

        Does not hash or open scientific payloads. Resolving an access context
        can establish a filesystem connection; local candidates are stat'ed.
        The caller must verify every returned file before opening MNE.
        """
        from fsspec.implementations.local import LocalFileSystem

        if self._pglClosed:
            raise ValueError("Checkpoint filesystem is closed.")

        candidatesByName = {}

        for member in records:
            relativeName = member["name"]
            logicalName = self._logicalPath(posixpath.join(resourceRoot, relativeName))
            registered = self._payloads.get(logicalName)

            if registered is None:
                raise ValueError(f"FIF member is absent from checkpoint: {logicalName}")

            expected = {key: member[key] for key in ("sha256", "size_bytes")}
            actual = {key: registered[key] for key in ("sha256", "size_bytes")}

            if actual != expected:
                raise ValueError(f"FIF/checkpoint identity mismatch: {logicalName}")

            candidates = {}
            parts = relativeName.split("/")

            for location in registered["locations"]:
                filesystem, root = self._resolveLocation(location)

                # Determine locality from the resolved filesystem, never the
                # stored path, backend label, or SQL transport alone.
                if type(filesystem) is not LocalFileSystem:
                    continue

                source = Path(filesystem._strip_protocol(posixpath.join(root, location["object_key"]))).expanduser()

                if source.is_symlink():
                    continue

                source = source.resolve()
                bundleRoot = source

                for part in parts:
                    bundleRoot = bundleRoot.parent

                if bundleRoot.joinpath(*parts) != source:
                    continue

                candidates[bundleRoot] = source

            candidatesByName[relativeName] = candidates

        if not candidatesByName:
            return None

        commonRoots = set.intersection(*(set(candidates) for candidates in candidatesByName.values()))

        for root in sorted(commonRoots, key=str):
            paths = {name: candidates[root] for name, candidates in candidatesByName.items()}

            if all(path.is_file() for path in paths.values()):
                return paths

        return None

    # +++++++++++++++++++++++ validateBundleIdentities +++++++++++++++++++++++
    def validateBundleIdentities(self, resourceRoot, records):
        """Check MNE bundle identities against checkpoint metadata without I/O."""
        for member in records:
            logicalName = self._logicalPath(posixpath.join(resourceRoot, member["name"]))
            registered = self._payloads.get(logicalName)

            if registered is None:
                raise ValueError(f"FIF member is absent from checkpoint: {logicalName}")

            expected = {key: member[key] for key in ("sha256", "size_bytes")}
            actual = {key: registered[key] for key in ("sha256", "size_bytes")}

            if actual != expected:
                raise ValueError(f"FIF/checkpoint identity mismatch: {logicalName}")

    # +++++++++++++++++++++++++++++++ close +++++++++++++++++++++++++++++++++
    def close(self):
        """Disable future payload reads and close an owned access context.

        Call only after all deferred session readers are finished.
        Does not delete files or affect the database.

        A shared context remains the responsibility of its external owner.
        Already returned verified temporary streams remain caller-owned.
        """
        if self._pglClosed:
            return

        self._pglClosed = True

        if self._ownsAccess and self._accessContext is not None:
            self._accessContext.close()

################################################################
#   filename: pglStorage.py
#    purpose: Persistent storage for sessions and datasets
#             allowing for sharing, provenance tracking, pipelining
#         by: JLG
#       date: Oct 5, 2026
################################################################
import hashlib
import re
from pathlib import Path

import psycopg
from psycopg import sql

from .pglMessages import pglMessages
from .pglBase import pglBase
import posixpath
import json
import uuid
from fsspec.core import url_to_fs
from copy import deepcopy

class pglStorage:
    """Manage scientific storage, checkpoint metadata, and schema migrations."""

    # Stable lock key shared by all pgl schema-initialization processes.
    MIGRATION_LOCK = 734_291_806_115

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
    
    @staticmethod
    def hashFile(filesystem, path, chunkSize=1024 * 1024):
        """Compute SHA-256 and byte count without loading the whole file."""
        if isinstance(chunkSize, bool) or not isinstance(chunkSize, int) or chunkSize <= 0:
            raise ValueError("chunkSize must be a positive integer.")

        digest = hashlib.sha256()
        sizeBytes = 0

        with filesystem.open(path, "rb") as file:
            while True:
                chunk = file.read(chunkSize)
                if not chunk:
                    break

                digest.update(chunk)
                sizeBytes += len(chunk)

        return {"sha256": digest.hexdigest(), "size_bytes": sizeBytes}
    
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
                identity = cls.hashFile(run.filesystem, sourcePath)
                files.append({"name": posixpath.join(runName, relativePath), **identity})

        return {
            "contract_version": "pgl.raw.behavior.v1",
            "runs": runs,
            "directories": sorted(["runs", *directories]),
            "files": sorted(files, key=lambda record: record["name"]),
        }    
    
    @staticmethod
    def encodeRawManifest(manifest):
        """Encode a manifest deterministically as UTF-8 JSON bytes."""
        return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

    @classmethod
    def hashRawManifest(cls, manifest):
        """Return the SHA-256 of the exact encoded manifest bytes."""
        return hashlib.sha256(cls.encodeRawManifest(manifest)).hexdigest()    
    
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
    
    @classmethod
    def stageRawFile(cls, sourceFilesystem, sourcePath, destinationFilesystem, destinationRoot, expected):
        """Copy and verify a payload, returning its relative storage key.

        destinationRoot must be a resolved path for destinationFilesystem.
        A failed operation may leave an unregistered staging object.
        """
        expectedHash = expected["sha256"]
        expectedSize = expected["size_bytes"]

        if not isinstance(expectedHash, str) or re.fullmatch(r"[0-9a-f]{64}", expectedHash) is None:
            raise ValueError("Expected SHA-256 must contain 64 lowercase hexadecimal characters.")

        if isinstance(expectedSize, bool) or not isinstance(expectedSize, int) or expectedSize < 0:
            raise ValueError("Expected size must be a nonnegative integer.")

        objectKey = f"staging/{uuid.uuid4().hex}/payload"
        destinationPath = posixpath.join(destinationRoot, objectKey)
        destinationFilesystem.makedirs(posixpath.dirname(destinationPath), exist_ok=True)

        digest = hashlib.sha256()
        sizeBytes = 0

        with sourceFilesystem.open(sourcePath, "rb") as source:
            with destinationFilesystem.open(destinationPath, "wb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break

                    destination.write(chunk)
                    digest.update(chunk)
                    sizeBytes += len(chunk)

        if digest.hexdigest() != expectedHash or sizeBytes != expectedSize:
            raise ValueError(f"Source bytes do not match the manifest; staging object was not accepted: {objectKey}")

        stored = cls.hashFile(destinationFilesystem, destinationPath)

        if stored["sha256"] != expectedHash or stored["size_bytes"] != expectedSize:
            raise ValueError(f"Stored payload verification failed: {objectKey}")

        return {"object_key": objectKey, **stored}    
    
    @classmethod
    def stageRawSession(cls, session, manifest, destinationFilesystem, destinationRoot):
        """Stage and verify every file described by a raw session manifest.

        destinationRoot must be a resolved path for destinationFilesystem.
        Returns storage records; does not publish a checkpoint.
        Failed operations may leave unregistered staging objects.
        """
        currentManifest = cls.buildRawManifest(session)

        if currentManifest != manifest:
            raise ValueError("Source session no longer matches the supplied manifest.")

        sources = {}

        for runRecord in manifest["runs"]:
            run = session.runs[runRecord["index"]]
            sources[runRecord["path"]] = run

        staged = []

        for record in manifest["files"]:
            parts = record["name"].split("/", 2)
            runPath = "/".join(parts[:2])
            relativePath = parts[2]
            run = sources[runPath]
            sourcePath = posixpath.join(run.fullDataPath, relativePath)

            stored = cls.stageRawFile(run.filesystem, sourcePath, destinationFilesystem, destinationRoot, record)
            staged.append({"name": record["name"], **stored})

        return staged
    
    @classmethod
    def stageRawManifest(cls, manifest, destinationFilesystem, destinationRoot):
        """Store and verify the exact manifest bytes without publishing a checkpoint.

        destinationRoot must be a resolved path for destinationFilesystem.
        Failed operations may leave an unregistered staging object.
        """
        manifestBytes = cls.encodeRawManifest(manifest)
        expectedHash = hashlib.sha256(manifestBytes).hexdigest()
        expectedSize = len(manifestBytes)

        objectKey = f"staging/{uuid.uuid4().hex}/manifest.json"
        destinationPath = posixpath.join(destinationRoot, objectKey)
        destinationFilesystem.makedirs(posixpath.dirname(destinationPath), exist_ok=True)

        with destinationFilesystem.open(destinationPath, "wb") as file:
            file.write(manifestBytes)

        stored = cls.hashFile(destinationFilesystem, destinationPath)

        if stored["sha256"] != expectedHash or stored["size_bytes"] != expectedSize:
            raise ValueError(f"Stored manifest verification failed: {objectKey}")

        return {"object_key": objectKey, **stored}
    
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
    
    @classmethod
    def verifyAndRegisterStorageLocation(cls, connection, postgres, fileID, backendID, objectKey, storageOptions=None):
        """Verify a payload and register its location in the caller's transaction.

        Backend roots must be accessible from this client.
        Runtime storageOptions are passed to fsspec, never stored.
        Does not commit or overwrite an existing location's object key.
        """
        from fsspec.core import url_to_fs

        if not isinstance(objectKey, str) or objectKey.startswith("/") or "\\" in objectKey or "\x00" in objectKey or any(part in {"", ".", ".."} for part in objectKey.split("/")):
            raise ValueError("objectKey must be a valid relative POSIX path.")

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

        statement = sql.SQL("SELECT object_key FROM {} WHERE file_id = %s AND backend_id = %s").format(locationsTable)
        existing = connection.execute(statement, (fileID, backendID)).fetchone()

        if existing is not None and existing[0] != objectKey:
            raise ValueError("File already has a different object key on this backend.")

        filesystem, root = url_to_fs(backend[0], **(storageOptions or {}))
        actual = cls.hashFile(filesystem, posixpath.join(root, objectKey))

        if actual["sha256"] != identity[0] or actual["size_bytes"] != identity[1]:
            raise ValueError("Stored payload does not match the registered file identity.")

        statement = sql.SQL("""
            INSERT INTO {} AS location
                (file_id, backend_id, object_key, status, verified_at)
            VALUES (%s, %s, %s, 'present', now())
            ON CONFLICT (file_id, backend_id) DO UPDATE
            SET status = 'present', verified_at = EXCLUDED.verified_at
            WHERE location.object_key = EXCLUDED.object_key
            RETURNING object_key
        """).format(locationsTable)

        row = connection.execute(statement, (fileID, backendID, objectKey)).fetchone()

        if row is None:
            raise ValueError("File location changed concurrently; retry the transaction.")

        return row[0]
    
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
    
    @classmethod
    def registerStagedRawCheckpoint(cls, connection, postgres, createdBy, backendID, manifest, staged, storedManifest, storageOptions=None):
        """Register a staged behavioral archive without publishing it.

        Must be called inside the caller's transaction.
        Subject associations are read from verified archived settings.
        Unused duplicate staging objects are not deleted here.
        """
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

        if manifest["contract_version"] != "pgl.raw.behavior.v1":
            raise ValueError("Unsupported raw manifest contract.")

        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")
        statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s").format(backendsTable)
        backend = connection.execute(statement, (backendID,)).fetchone()

        if backend is None or not backend[1]:
            raise ValueError(f"Storage backend is missing or inactive: {backendID}")

        fileLocations = {
            name: {
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "locations": [{
                    "backend_id": backendID,
                    "url_prefix": backend[0],
                    "object_key": record["object_key"],
                }],
            }
            for name, record in stagedByName.items()
        }

        archiveFilesystem = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend={backendID: storageOptions or {}})
        subjectsByRun = cls.readRawSubjects(manifest, archiveFilesystem)
        runPaths = {record["path"] for record in manifest["runs"]}

        subjectsByName = {}
        for name in expectedByName:
            matchingRuns = [path for path in runPaths if name.startswith(path + "/")]
            if len(matchingRuns) != 1:
                raise ValueError(f"File does not belong to exactly one run: {name}")

            subjectsByName[name] = subjectsByRun[matchingRuns[0]]

        for subjectID in set(subjectsByRun.values()):
            cls.getOrCreateSubject(connection, postgres, subjectID)

        checkpointID = cls.createRawCheckpoint(connection, postgres, createdBy)
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")

        def registerPayload(record):
            fileID = cls.getOrCreateFile(connection, postgres, record["sha256"], record["size_bytes"])

            statement = sql.SQL("SELECT object_key FROM {} WHERE file_id = %s AND backend_id = %s").format(locationsTable)
            existing = connection.execute(statement, (fileID, backendID)).fetchone()
            objectKey = record["object_key"] if existing is None else existing[0]

            cls.verifyAndRegisterStorageLocation(connection, postgres, fileID, backendID, objectKey, storageOptions=storageOptions)
            return fileID

        for name in sorted(expectedByName):
            fileID = registerPayload(stagedByName[name])
            cls.attachCheckpointFile(connection, postgres, checkpointID, name, fileID, role="acquisition", subjectID=subjectsByName[name])

        manifestFileID = registerPayload(storedManifest)
        cls.attachCheckpointFile(connection, postgres, checkpointID, "manifest.json", manifestFileID, role="manifest")

        return checkpointID

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
    
    @classmethod
    def getRawFileLocations(cls, connection, postgres, sessionID, manifest):
        """Return acquisition-file identities and candidate storage locations."""
        cls.validateRawManifest(manifest)

        manifestIdentity = cls.getRawManifestLocations(connection, postgres, sessionID)
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

    @staticmethod
    def getRawManifestLocations(connection, postgres, sessionID):
        """Return the manifest identity and candidate locations for a completed checkpoint."""
        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("SELECT status, origin, contract_version, manifest_hash FROM {} WHERE session_id = %s").format(sessionsTable)
        checkpoint = connection.execute(statement, (sessionID,)).fetchone()

        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {sessionID}")

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
        
    @classmethod
    def loadRawManifest(cls, connection, postgres, sessionID, storageOptionsByBackend=None, maxBytes=16 * 1024 * 1024):
        """Retrieve and verify a completed checkpoint's manifest.

        Runtime storage options are keyed by backend_id and are never saved.
        This checks content identity and contract version, not the full schema.
        """
        from fsspec.core import url_to_fs

        if isinstance(maxBytes, bool) or not isinstance(maxBytes, int) or maxBytes <= 0:
            raise ValueError("maxBytes must be a positive integer.")

        located = cls.getRawManifestLocations(connection, postgres, sessionID)

        if located["size_bytes"] > maxBytes:
            raise ValueError("Manifest exceeds the configured size limit.")

        optionsByBackend = storageOptionsByBackend or {}
        failures = []

        for location in located["locations"]:
            backendID = location["backend_id"]
            key = location["object_key"]

            if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                raise ValueError("Database contains an invalid manifest object key.")

            try:
                filesystem, root = url_to_fs(location["url_prefix"], **optionsByBackend.get(backendID, {}))
                path = posixpath.join(root, key)

                # Bound the read, even if the actual file exceeds its recorded size.
                with filesystem.open(path, "rb") as file:
                    content = file.read(located["size_bytes"] + 1)

                if len(content) != located["size_bytes"]:
                    raise ValueError("Manifest size mismatch.")

                if hashlib.sha256(content).hexdigest() != located["sha256"]:
                    raise ValueError("Manifest hash mismatch.")

            except (OSError, ValueError) as error:
                failures.append(f"backend {backendID}: {type(error).__name__}")
                continue

            # Once verified bytes are found, decoding errors indicate a bad
            # manifest, not a bad replica of that same content.
            manifest = json.loads(content.decode("utf-8"))

            if not isinstance(manifest, dict):
                raise ValueError("Manifest must contain a JSON object.")

            if manifest.get("contract_version") != "pgl.raw.behavior.v1":
                raise ValueError("Unsupported manifest contract.")

            if cls.encodeRawManifest(manifest) != content:
                raise ValueError("Stored manifest is not in the expected canonical encoding.")

            return cls.validateRawManifest(manifest)

        raise OSError(f"No manifest copy could be verified: {'; '.join(failures)}")

    @staticmethod
    def validateRawManifest(manifest):
        """Validate the v1 manifest structure without accessing payloads."""
        def requireFields(value, fields, label):
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError(f"{label} has missing or unexpected fields.")

        def validatePath(path):
            if not isinstance(path, str) or path.startswith("/") or "\\" in path or "\x00" in path:
                raise ValueError(f"Invalid manifest path: {path!r}")

            if any(part in {"", ".", ".."} for part in path.split("/")):
                raise ValueError(f"Invalid manifest path: {path!r}")

        requireFields(manifest, {"contract_version", "runs", "directories", "files"}, "Manifest")

        if manifest["contract_version"] != "pgl.raw.behavior.v1":
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
                raise ValueError("Run indices must be consecutive and ordered from zero.")

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
            requireFields(record, {"name", "sha256", "size_bytes"}, "File record")
            validatePath(record["name"])

            digest = record["sha256"]
            size = record["size_bytes"]

            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("File SHA-256 must contain 64 lowercase hexadecimal characters.")

            if type(size) is not int or not 0 <= size <= 9223372036854775807:
                raise ValueError("File size must be a nonnegative PostgreSQL bigint.")

            fileNames.append(record["name"])

        if fileNames != sorted(set(fileNames)):
            raise ValueError("File names must be unique and sorted.")

        if directorySet.intersection(fileNames):
            raise ValueError("A manifest path cannot be both a file and a directory.")

        for path in directories + fileNames:
            if path == "runs":
                continue

            if posixpath.dirname(path) not in directorySet:
                raise ValueError(f"Missing parent directory for {path!r}.")

            owningRun = "/".join(path.split("/")[:2])
            if owningRun not in runPaths:
                raise ValueError(f"Path lies outside the declared runs: {path!r}")

        for runPath in runPaths:
            if not any(name.startswith(runPath + "/") for name in fileNames):
                raise ValueError(f"Run contains no files: {runPath}")

        return manifest
    
    @classmethod
    def loadRawSession(cls, connection, postgres, sessionID, storageOptionsByBackend=None):
        """Reconstruct a completed raw behavioral checkpoint as a lazy session.

        Uses the caller's SQL connection for metadata lookup only.
        Verifies the manifest immediately; acquisition payloads are verified
        when opened through the returned session's filesystem.
        """
        from .pglSession import pglSession

        manifest = cls.loadRawManifest(connection, postgres, sessionID, storageOptionsByBackend=storageOptionsByBackend)
        fileLocations = cls.getRawFileLocations(connection, postgres, sessionID, manifest)

        filesystem = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend=storageOptionsByBackend)
        runPaths = [record["path"] for record in manifest["runs"]]

        return pglSession(filesystem=filesystem, runList=runPaths)
    
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
    
    @classmethod
    def saveRawSession(cls, connection, postgres, session, createdBy, backendID, storageOptions=None):
        """Archive saved behavioral run files and return a completed checkpoint ID.

        Existing completed checkpoints with the same manifest are reused,
        without uploading another replica.

        The backend must be persistent, client-accessible, and outside the
        source run directories. Acquisition must have finished before saving.

        Opens a transaction, or a savepoint if the caller already has one.
        An outer caller transaction must commit for changes to persist.
        Payload writes are not rolled back with database changes.
        """
        from fsspec.core import url_to_fs

        manifest = cls.buildRawManifest(session)
        cls.validateRawManifest(manifest)
        manifestHash = cls.hashRawManifest(manifest)

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")
        usersTable = sql.Identifier(postgres.databaseSchema, "users")

        with connection.transaction():
            statement = sql.SQL("SELECT 1 FROM {} WHERE user_id = %s").format(usersTable)
            if connection.execute(statement, (createdBy,)).fetchone() is None:
                raise ValueError(f"Unknown creator: {createdBy}")

            statement = sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s").format(backendsTable)
            backend = connection.execute(statement, (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError(f"Storage backend is missing or inactive: {backendID}")

            # Coordinate cooperating ingestions of the same manifest.
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (manifestHash,))

            statement = sql.SQL("SELECT session_id, status, contract_version FROM {} WHERE manifest_hash = %s AND origin = 'raw'").format(sessionsTable)
            existing = connection.execute(statement, (manifestHash,)).fetchone()

            if existing is not None:
                if existing[1:] != ("complete", "pgl.raw.behavior.v1"):
                    raise ValueError("Matching manifest belongs to an incomplete or unsupported checkpoint.")

                # Verify the existing manifest and check its registered links
                # and available location metadata before reusing it.
                optionsByBackend = {backendID: storageOptions or {}}
                archivedManifest = cls.loadRawManifest(connection, postgres, existing[0], storageOptionsByBackend=optionsByBackend)

                if archivedManifest != manifest:
                    raise ValueError("Existing checkpoint manifest differs from source manifest.")

                cls.getRawFileLocations(connection, postgres, existing[0], archivedManifest)
                return existing[0]

            filesystem, root = url_to_fs(backend[0], **(storageOptions or {}))

            staged = cls.stageRawSession(session, manifest, filesystem, root)
            storedManifest = cls.stageRawManifest(manifest, filesystem, root)

            sessionID = cls.registerStagedRawCheckpoint(connection, postgres, createdBy, backendID, manifest, staged, storedManifest, storageOptions=storageOptions)
            cls.completeRawCheckpoint(connection, postgres, sessionID, manifest)

            return sessionID
    
    # ------------------------------------------------------------
    # Public schema initialization
    # ------------------------------------------------------------

    @classmethod
    def initDatabase(cls, postgres=None, *, settings=None, settingsName=None):
        """Initialize or upgrade the scientific schema.

        Uses the supplied pglPostgres object, or resolves it from pgl settings.
        Prompts privately for administrator credentials.
        Never drops tables, resets IDs, or modifies applied migration files.

        Returns True on success, or False for missing configuration or an
        operational connection failure. Schema/ownership errors raise.
        """
        from .pglPostgres import pglPostgres

        if postgres is not None and (settings is not None or settingsName is not None):
            raise ValueError("Supply either postgres or pgl settings, not both.")

        if postgres is None:
            postgres = pglPostgres.fromSettings(settings=settings, settingsName=settingsName)

            if postgres is None:
                return False

        if not isinstance(postgres, pglPostgres):
            raise TypeError("postgres must be a pglPostgres instance.")

        postgres._validateValues()
        migrations = cls._readMigrations()
        adminPassword = postgres._askPassword(f"PostgreSQL administrator password for {postgres.adminUser}: ")

        try:
            # _adminConnect verifies the actual server directory, version,
            # and networking before returning the connection.
            with postgres._adminConnect(adminPassword, postgres.databaseName) as connection:
                with connection.transaction():
                    connection.execute("SELECT pg_advisory_xact_lock(%s)", (cls.MIGRATION_LOCK,))
                    cls._prepareSchema(connection, postgres)
                    applied = cls._applyMigrations(connection, postgres, migrations)
                    cls._grantApplicationAccess(connection, postgres)

        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not complete schema initialization for {postgres.databaseName!r}. Check the server and credentials. If the connection was lost during commit, rerun initialization to check the recorded migration state.\n{error}")
            return False

        if applied:
            pglMessages.message(f"Applied migrations: {', '.join(applied)}")
        else:
            pglMessages.message("Database schema is already current.")

        pglMessages.message(f"Scientific schema ready: {postgres.databaseName}.{postgres.databaseSchema}")
        return True

    # ------------------------------------------------------------
    # Migration files
    # ------------------------------------------------------------

    @classmethod
    def _readMigrations(cls):
        """Read and checksum the SQL migration files shipped with pglStorage."""
        directory = Path(pglBase.getPGLDir()) / "database" / "migrations"

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

    # ------------------------------------------------------------
    # Schema namespace and migration history
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Apply migrations
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Application permissions
    # ------------------------------------------------------------

    @classmethod
    def _grantApplicationAccess(cls, connection, postgres):
        """Grant access to pilot tables without granting schema management."""
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
        ]

        for name in applicationTables:
            table = sql.Identifier(postgres.databaseSchema, name)
            connection.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {} TO {}").format(table, application))

        readOnlyTables = [
            "schema_migrations",
            "session_subjects",
            "raw_sessions",
        ]

        for name in readOnlyTables:
            table = sql.Identifier(postgres.databaseSchema, name)
            connection.execute(sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {} FROM {}").format(table, application))
            connection.execute(sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(table, application))

        connection.execute(sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(schema, application))
        
################################################################
# Read-only logical filesystem for archived checkpoints
################################################################

import posixpath
import hashlib
import tempfile

from fsspec import AbstractFileSystem


class pglCheckpointFilesystem(AbstractFileSystem):
    """Expose checkpoint paths independently of physical payload locations."""

    protocol = "pglcheckpoint"
    cachable = False

    def __init__(self, manifest, fileLocations, storageOptionsByBackend=None):
        super().__init__()

        from .pglStorage import pglStorage
        pglStorage.validateRawManifest(manifest)

        expectedFiles = {record["name"]: record for record in manifest["files"]}

        if set(fileLocations) != set(expectedFiles):
            raise ValueError("File locations must cover exactly the manifest files.")

        self._payloads = deepcopy(fileLocations)
        self._storageOptionsByBackend = dict(storageOptionsByBackend or {})

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

        def validateName(name):
            if not isinstance(name, str) or not name:
                raise ValueError("Checkpoint paths must be nonempty strings.")

            if name.startswith("/") or "\\" in name or "\x00" in name:
                raise ValueError(f"Invalid checkpoint path: {name!r}")

            if any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError(f"Invalid checkpoint path: {name!r}")

        def addEntry(name, kind, size):
            validateName(name)

            if name in self._entries:
                raise ValueError(f"Duplicate checkpoint path: {name!r}")

            self._entries[name] = {"name": name, "type": kind, "size": size}

        for name in manifest["directories"]:
            addEntry(name, "directory", 0)

        for record in manifest["files"]:
            size = record["size_bytes"]

            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"Invalid file size for {record['name']!r}")

            addEntry(record["name"], "file", size)

        for name in self._entries:
            if not name:
                continue

            parent = posixpath.dirname(name)
            parentEntry = self._entries.get(parent)

            if parentEntry is None or parentEntry["type"] != "directory":
                raise ValueError(f"Missing parent directory for checkpoint path: {name!r}")

    def _logicalPath(self, path):
        return self._strip_protocol(str(path)).strip("/")

    def info(self, path, **kwargs):
        name = self._logicalPath(path)

        if name not in self._entries:
            raise FileNotFoundError(name)

        return dict(self._entries[name])

    def ls(self, path, detail=True, **kwargs):
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

    def _open(self, path, mode="rb", **kwargs):
        """Return verified bytes, trying alternate locations when necessary."""
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
                options = self._storageOptionsByBackend.get(backendID, {})
                filesystem, root = url_to_fs(location["url_prefix"], **options)
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
                        buffer.write(chunk)

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
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
    def print(details=None, postgres=None, *, settings=None, settingsName=None, sessionID=None, summary=False, nRows=10, maxWidth=48, sslRootCert=None, includeDeleted=False, storageOptionsByBackend=None):
        """Select a session listing, checkpoint detail view, or database report.

        Default:
            List all visible sessions grouped by root experiment and subject,
            displaying their actual parent-child relationships.

        summary=True:
            Show the previous database summary: table counts and statuses.

        sessionID:
            Show one checkpoint's metadata, subjects, and linked files.

        details:
            Show one table, an iterable of tables, or "all".

        summary, sessionID, and details are mutually exclusive.
        nRows limits table detail views only, not the session listing.

        The grouped listing reads verified root experimentSettings.json files.
        storageOptionsByBackend supplies runtime payload credentials/options.
        No runs, tasks, or MNE payloads are reconstructed.

        Returns False for unavailable configuration/connection or an unknown
        checkpoint in the single-session view. Other errors propagate.
        """
        if type(summary) is not bool:
            raise ValueError("summary must be a boolean.")

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        if type(nRows) is not int or nRows <= 0:
            raise ValueError("nRows must be a positive integer.")

        if type(maxWidth) is not int or maxWidth < 8:
            raise ValueError("maxWidth must be an integer of at least 8.")

        if sum((details is not None, sessionID is not None, summary)) > 1:
            raise ValueError("Supply only one of details, sessionID, or summary=True.")

        if sessionID is not None:
            return pglStorage.printSession(sessionID, postgres=postgres, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeDeleted=includeDeleted)

        if details is not None or summary:
            return pglStorage.printTables(details=details, postgres=postgres, settings=settings, settingsName=settingsName, nRows=nRows, maxWidth=maxWidth, sslRootCert=sslRootCert, includeDeleted=includeDeleted)

        return pglStorage.printSessions(postgres=postgres, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeDeleted=includeDeleted, storageOptionsByBackend=storageOptionsByBackend)

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

    @staticmethod
    def _printValue(value, maxWidth=None):
        """Produce single-line display text, optionally truncated."""
        text = "NULL" if value is None else str(value)
        text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")

        if maxWidth is not None and len(text) > maxWidth:
            return text[:maxWidth - 3] + "..."

        return text

    @staticmethod
    def _printTable(headers, rows, maxWidth=48):
        """Print a compact table without accessing storage."""
        displayed = [[pglStorage._printValue(value, maxWidth) for value in headers]]
        displayed.extend([pglStorage._printValue(value, maxWidth) for value in row] for row in rows)
        widths = [max(len(row[index]) for row in displayed) for index in range(len(headers))]

        def printRow(row):
            pglMessages.print("  " + " | ".join(value.ljust(width) for value, width in zip(row, widths)))

        printRow(displayed[0])
        pglMessages.print("  " + "-+-".join("-" * width for width in widths))

        for row in displayed[1:]:
            printRow(row)

        if not rows:
            pglMessages.print("  (no rows)")

    @staticmethod
    def _printTableOrder():
        """Supported database-report tables and their stable ordering."""
        return {
            "session": ("session_id",),
            "subjects": ("subject_id",),
            "files": ("file_id",),
            "storage_backends": ("backend_id",),
            "storage_locations": ("file_id", "backend_id", "object_key"),
            "users": ("user_id",),
            "session_files": ("session_id", "name"),
        }
        
    @staticmethod
    def _readPrintExperimentSettings(identity, locations, storageOptionsByBackend=None):
        """Read one verified behavioral experiment-settings JSON document.

        Reads only the settings document, not a run or MNE object.
        Tries eligible registered replicas in their supplied order.
        """
        maxBytes = 16 * 1024 * 1024
        expectedSize = identity["size_bytes"]
        expectedHash = identity["sha256"]

        if type(expectedSize) is not int or not 0 <= expectedSize <= maxBytes:
            raise ValueError("Experiment settings exceed the reporting size limit or have an invalid size.")

        optionsByBackend = storageOptionsByBackend or {}
        failures = []

        for location in locations:
            backendID = location["backend_id"]
            key = location["object_key"]

            if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                raise ValueError("Database contains an invalid experiment-settings object key.")

            try:
                filesystem, root = url_to_fs(location["url_prefix"], **optionsByBackend.get(backendID, {}))

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

        raise OSError("No experiment-settings copy could be verified: " + "; ".join(failures))

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

    @staticmethod
    def _sessionLineageLines(records):
        """Format a visible checkpoint forest with descriptions.

        Hidden/missing parents are not bypassed by artificial arrows.
        Descriptions are rendered on one line without truncation.
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

        def nodeLabel(sessionID):
            record = records[sessionID]
            label = str(sessionID)

            if record["deleted_at"] is not None:
                label += " [deleted]"

            if record["status"] != "complete":
                label += f" [{pglStorage._printValue(record['status'])}]"

            description = record.get("description")

            if description:
                label += " (" + pglStorage._printValue(description) + ")"

            return label

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

            appendTree(sessionID, "    sessionID: ")

        if len(visited) != len(records):
            raise ValueError("Session lineage contains a cycle with no visible root.")

        return lines
    
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
    
    @staticmethod
    def printSessions(postgres=None, *, settings=None, settingsName=None, sslRootCert=None, includeDeleted=False, storageOptionsByBackend=None):
        """Print sessions grouped by root experiment and subject.

        Shows checkpoint descriptions alongside IDs and lineage.

        Uses recorded root-session labels when available.
        Older behavioral roots fall back to verified experimentSettings.json.
        Descendants inherit their root's grouping.

        Unset labels display as Unknown.
        Does not reconstruct runs, tasks, or MNE payloads.
        """
        from psycopg.rows import tuple_row

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        postgres = pglStorage._printPostgres(postgres, settings=settings, settingsName=settingsName)

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

                    statement = sql.SQL("SELECT session_id, parent_session_id, root_session_id, origin, status, deleted_at, meta, description FROM {} WHERE (%s OR deleted_at IS NULL) ORDER BY session_id").format(sessionsTable)
                    cursor.execute(statement, (includeDeleted,))

                    for sessionID, parentID, rootID, origin, status, deletedAt, meta, description in cursor.fetchall():
                        records[sessionID] = {"session_id": sessionID, "parent_session_id": parentID, "root_session_id": rootID, "origin": origin, "status": status, "deleted_at": deletedAt, "meta": meta, "description": description}

                    rootIDs = {record["root_session_id"] for record in records.values()}
                    fallbackRoots = []

                    for rootID in sorted(rootIDs):
                        record = records.get(rootID)

                        if record is None:
                            groupsByRoot[rootID] = {("Unknown", "Unknown")}
                            continue

                        meta = record["meta"] or {}

                        if "pgl_session_labels" in meta:
                            metadata = pglStorage._validateSessionLabelMetadata(meta["pgl_session_labels"])
                            groupsByRoot[rootID] = {(label["experimentName"] if label["experimentName"] is not None else "Unknown", label["subjectID"] if label["subjectID"] is not None else "Unknown") for label in metadata["labels"]}
                        else:
                            fallbackRoots.append(rootID)

                    if fallbackRoots:
                        statement = sql.SQL("SELECT link.session_id, link.name, content.file_id, content.sha256, content.size_bytes FROM {} AS link JOIN {} AS content ON content.file_id = link.file_id WHERE link.session_id = ANY(%s::bigint[]) AND link.name ~ '^runs/[0-9]+/experimentSettings[.]json$' AND (%s OR (link.deleted_at IS NULL AND content.deleted_at IS NULL)) ORDER BY link.session_id, link.name").format(linksTable, filesTable)
                        cursor.execute(statement, (fallbackRoots, includeDeleted))

                        for rootID, name, fileID, digest, size in cursor.fetchall():
                            settingsByRoot.setdefault(rootID, []).append((name, fileID))
                            identities[fileID] = {"sha256": digest, "size_bytes": size}

                    if identities:
                        statement = sql.SQL("SELECT location.file_id, backend.backend_id, backend.url_prefix, location.object_key FROM {} AS location JOIN {} AS backend ON backend.backend_id = location.backend_id WHERE location.file_id = ANY(%s::bigint[]) AND (%s OR location.deleted_at IS NULL) AND location.status = 'present' AND location.verified_at IS NOT NULL AND backend.is_active ORDER BY location.file_id, location.priority, backend.backend_id, location.object_key").format(locationsTable, backendsTable)
                        cursor.execute(statement, (sorted(identities), includeDeleted))

                        for fileID, backendID, root, key in cursor.fetchall():
                            locationsByFile.setdefault(fileID, []).append({"backend_id": backendID, "url_prefix": root, "object_key": key})

        pairByFile = {}

        for fileID, identity in identities.items():
            locations = locationsByFile.get(fileID, [])

            if not locations:
                raise FileNotFoundError(f"Experiment-settings file {fileID} has no eligible registered location.")

            savedSettings = pglStorage._readPrintExperimentSettings(identity, locations, storageOptionsByBackend=storageOptionsByBackend)
            experiment = pglStorage._printGroupingValue(savedSettings.get("experimentName"), "experimentName")
            subject = pglStorage._printGroupingValue(savedSettings.get("subjectID"), "subjectID")
            pairByFile[fileID] = (experiment, subject)

        for rootID in {record["root_session_id"] for record in records.values()}:
            if rootID not in groupsByRoot:
                pairs = {pairByFile[fileID] for name, fileID in settingsByRoot.get(rootID, [])}
                groupsByRoot[rootID] = pairs or {("Unknown", "Unknown")}

        grouped = {}

        for sessionID, record in records.items():
            for experiment, subject in groupsByRoot[record["root_session_id"]]:
                grouped.setdefault(experiment, {}).setdefault(subject, {})[sessionID] = record

        def groupOrder(value):
            return (value == "Unknown", value.casefold(), value)

        output = []

        for experiment in sorted(grouped, key=groupOrder):
            output.append(pglStorage._printValue(experiment))

            for subject in sorted(grouped[experiment], key=groupOrder):
                output.append("  " + pglStorage._printValue(subject))
                output.extend(pglStorage._sessionLineageLines(grouped[experiment][subject]))

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
    
    @staticmethod
    def printSession(sessionID, postgres=None, *, settings=None, settingsName=None, sslRootCert=None, includeDeleted=False):
        """Print one checkpoint's metadata, subjects, and linked files.

        Shows one preferred registered location per logical file.
        Other visible locations are counted.

        Does not read or verify payload bytes.
        Full file paths are printed without truncation.
        """
        from psycopg.rows import tuple_row

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        postgres = pglStorage._printPostgres(postgres, settings=settings, settingsName=settingsName)

        if postgres is None:
            return False

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        usersTable = sql.Identifier(postgres.databaseSchema, "users")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        filesTable = sql.Identifier(postgres.databaseSchema, "files")
        locationsTable = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        sessionRecord = None
        sessionFiles = {}
        locationsByFile = {}

        with postgres.connect(sslRootCert=sslRootCert) as connection:
            if connection is None:
                return False

            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

                with connection.cursor(row_factory=tuple_row) as cursor:
                    cursor.execute("SELECT current_database(), transaction_timestamp()")
                    databaseName, snapshotTime = cursor.fetchone()

                    statement = sql.SQL("SELECT checkpoint.*, creator.username AS creator_username, creator.display_name AS creator_display_name FROM {} AS checkpoint LEFT JOIN {} AS creator ON creator.user_id = checkpoint.created_by WHERE checkpoint.session_id = %s AND (%s OR checkpoint.deleted_at IS NULL)").format(sessionsTable, usersTable)
                    cursor.execute(statement, (sessionID, includeDeleted))
                    row = cursor.fetchone()

                    if row is not None:
                        sessionRecord = dict(zip((column.name for column in cursor.description), row))

                        statement = sql.SQL("SELECT link.name, link.role, link.subject_id, content.file_id, content.sha256, content.size_bytes, link.deleted_at, content.deleted_at FROM {} AS link JOIN {} AS content ON content.file_id = link.file_id WHERE link.session_id = %s AND (%s OR (link.deleted_at IS NULL AND content.deleted_at IS NULL)) ORDER BY link.name").format(linksTable, filesTable)
                        cursor.execute(statement, (sessionID, includeDeleted))

                        for name, role, subjectID, fileID, digest, size, linkDeleted, contentDeleted in cursor.fetchall():
                            sessionFiles[name] = {"role": role, "subject_id": subjectID, "file_id": fileID, "sha256": digest, "size_bytes": size, "link_deleted_at": linkDeleted, "content_deleted_at": contentDeleted}

                        fileIDs = sorted({record["file_id"] for record in sessionFiles.values()})

                        if fileIDs:
                            statement = sql.SQL("SELECT location.file_id, location.object_key, backend.backend_id, backend.name, backend.url_prefix, backend.is_active, location.status, location.priority, location.verified_at, location.deleted_at FROM {} AS location JOIN {} AS backend ON backend.backend_id = location.backend_id WHERE location.file_id = ANY(%s::bigint[]) AND (%s OR location.deleted_at IS NULL) ORDER BY location.file_id, location.priority, backend.backend_id, location.object_key").format(locationsTable, backendsTable)
                            cursor.execute(statement, (fileIDs, includeDeleted))

                            for fileID, key, backendID, backendName, root, active, status, priority, verifiedAt, deletedAt in cursor.fetchall():
                                locationsByFile.setdefault(fileID, []).append({"object_key": key, "backend_id": backendID, "backend_name": backendName, "root": root, "active": active, "status": status, "priority": priority, "verified_at": verifiedAt, "deleted_at": deletedAt})

        if sessionRecord is None:
            pglMessages.warning(f"No visible session {sessionID} found in {databaseName}.{postgres.databaseSchema}; it may be absent or excluded by the deletion filter.", level=0)
            return False

        def expectedKey(name):
            if sessionRecord["contract_version"] == "pgl.raw.behavior.v1":
                parts = name.split("/", 2)

                if len(parts) == 3 and parts[0] == "runs" and parts[1].isascii() and parts[1].isdigit():
                    return f"session{sessionID:06d}/run{int(parts[1]):05d}/{parts[2]}"

            return f"session{sessionID:06d}/{name}"

        def eligible(location):
            return location["active"] and location["status"] == "present" and location["verified_at"] is not None and (includeDeleted or location["deleted_at"] is None)

        deletionLabel = " [DELETED]" if sessionRecord["deleted_at"] is not None else ""
        pglMessages.printHeader(f"Session {sessionID}{deletionLabel}")
        pglMessages.message(f"Database: {databaseName}.{postgres.databaseSchema}", wrapText=False)
        pglMessages.message(f"Snapshot transaction started: {snapshotTime}", wrapText=False)
        pglMessages.print()
        pglMessages.printHeader("Session metadata", fillChar="-")
        labelWidth = max(len(name) for name in sessionRecord)

        for name, value in sessionRecord.items():
            pglMessages.print(f"  {name:<{labelWidth}} : {pglStorage._printValue(value)}")

        subjects = sorted({record["subject_id"] for record in sessionFiles.values() if record["subject_id"] is not None})
        pglMessages.print()
        pglMessages.printHeader("Subjects", fillChar="-")
        pglMessages.print("  " + (", ".join(pglStorage._printValue(subject) for subject in subjects) if subjects else "(none linked)"))

        pglMessages.print()
        pglMessages.printHeader(f"Files and locations: {len(sessionFiles)} linked files", fillChar="-")
        pglMessages.print("  One recorded content location per file; shared files may resolve into another run or session.")
        pglMessages.print("  Status is recorded metadata; payloads have not been checked.")
        pglMessages.print("  Registered aliases/copies are paths, not necessarily separate physical byte copies.")

        if includeDeleted:
            pglMessages.print("  Including deleted records. Read eligibility below assumes this inspection override.")

        if not sessionFiles:
            pglMessages.print("  (no linked files)")

        for name, record in sessionFiles.items():
            preferredKey = expectedKey(name)
            candidates = locationsByFile.get(record["file_id"], [])
            ordered = sorted(candidates, key=lambda location: (not eligible(location), location["object_key"] != preferredKey, location["priority"], location["backend_id"], location["object_key"]))

            pglMessages.print()
            pglMessages.print(f"  {pglStorage._printValue(name)}")
            pglMessages.print(f"    file_id: {record['file_id']} | bytes: {record['size_bytes']} | role: {pglStorage._printValue(record['role'])}")

            if includeDeleted:
                linkState = "DELETED" if record["link_deleted_at"] is not None else "active"
                contentState = "DELETED" if record["content_deleted_at"] is not None else "active"
                pglMessages.print(f"    File link: {linkState} | shared content identity: {contentState}")

            if not ordered:
                pglMessages.print("    No visible registered storage location.")
                continue

            location = ordered[0]
            separator = "" if location["root"].endswith("/") else "/"
            fullLocation = location["root"] + separator + location["object_key"]

            pglMessages.print(f"    Location: {pglStorage._printValue(fullLocation)}")
            pglMessages.print(f"    Backend: {location['backend_id']} | status: {pglStorage._printValue(location['status'])} | active: {location['active']} | eligible for reads: {eligible(location)}")
            pglMessages.print(f"    Last verified: {pglStorage._printValue(location['verified_at'])}")

            if includeDeleted:
                locationState = "DELETED" if location["deleted_at"] is not None else "active"
                pglMessages.print(f"    Location deletion state: {locationState}")

            if len(ordered) > 1:
                pglMessages.print(f"    Other visible registered aliases/copies: {len(ordered) - 1}")

        pglMessages.printHeader()
        return True
    
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

        tableOrder = pglStorage._printTableOrder()

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
        postgres = pglStorage._printPostgres(postgres, settings=settings, settingsName=settingsName)

        if postgres is None:
            return False

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
                pglStorage._printTable(headers, rows, maxWidth=maxWidth)

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
            pglStorage._printTable(("table", "rows"), counts, maxWidth=maxWidth)
            pglMessages.print()
            pglMessages.printHeader("Checkpoint status", fillChar="-")
            checkpointHeaders = ("origin", "status", "deleted", "checkpoints") if includeDeleted else ("origin", "status", "checkpoints")
            pglStorage._printTable(checkpointHeaders, checkpointCounts, maxWidth=maxWidth)

        pglMessages.printHeader()
        return True
    
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

    @classmethod
    def _checkRawSessionLabels(cls, session, *, allowMissingMetadata=False):
        """Check working-session safety and required raw acquisition labels.

        Called by the public raw saver before configuration, SQL, or export.
        allowMissingMetadata bypasses missing labels only, never failed-action
        safety or malformed/unreadable acquisition settings.
        """
        if type(allowMissingMetadata) is not bool:
            raise ValueError("allowMissingMetadata must be a boolean.")

        cls._requireSafeCheckpoint(session)

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
    
    @classmethod
    def _bindCheckpointBaseline(cls, session, sessionID, postgres):
        """Bind a prepared checkpoint/history boundary to a working object.

        Used after verified loading. Saving prepares its baseline separately
        before commit and installs it only after successful publication.

        Never clears failed-action safety state.
        """
        baseline = cls._prepareCheckpointBaseline(session, sessionID, postgres)
        session._checkpointBaseline = baseline
        
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

    @classmethod
    def _sessionHistoryBytes(cls, session, *, acquisition=False):
        """Encode history and labels after checking working-session safety.

        Runtime failure state is not serialized into checkpoint metadata.
        Labels remain part of the payload identity.
        """
        from .pglPipeline import pglActionable

        # All current derived exports and MNE raw exports pass through here.
        # Refuse unsafe working state before serializing or writing payloads.
        cls._requireSafeCheckpoint(session)

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
    def stageRawFile(cls, sourceFilesystem, sourcePath, destinationFilesystem, destinationRoot, expected, *, objectKey):
        """Copy and verify a payload at an explicit relative storage key.

        Never intentionally overwrites an existing file.
        Failure may leave an unregistered partial payload.
        """
        expectedHash = expected["sha256"]
        expectedSize = expected["size_bytes"]

        if not isinstance(expectedHash, str) or re.fullmatch(r"[0-9a-f]{64}", expectedHash) is None:
            raise ValueError("Expected SHA-256 must contain 64 lowercase hexadecimal characters.")

        if isinstance(expectedSize, bool) or not isinstance(expectedSize, int) or expectedSize < 0:
            raise ValueError("Expected size must be a nonnegative integer.")

        if not isinstance(objectKey, str) or objectKey.startswith("/") or "\\" in objectKey or "\x00" in objectKey or any(part in {"", ".", ".."} for part in objectKey.split("/")):
            raise ValueError("objectKey must be a valid relative POSIX path.")

        destinationPath = posixpath.join(destinationRoot, objectKey)
        destinationFilesystem.makedirs(posixpath.dirname(destinationPath), exist_ok=True)

        digest = hashlib.sha256()
        sizeBytes = 0

        with sourceFilesystem.open(sourcePath, "rb") as source:
            with destinationFilesystem.open(destinationPath, "xb") as destination:
                while True:
                    chunk = source.read(1024 * 1024)

                    if not chunk:
                        break

                    sizeBytes += len(chunk)

                    if sizeBytes > expectedSize:
                        raise ValueError(f"Source payload exceeds its manifest size: {objectKey}")

                    destination.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest() != expectedHash or sizeBytes != expectedSize:
            raise ValueError(f"Source bytes do not match the manifest: {objectKey}")

        stored = cls.hashFile(destinationFilesystem, destinationPath)

        if stored["sha256"] != expectedHash or stored["size_bytes"] != expectedSize:
            raise ValueError(f"Stored payload verification failed: {objectKey}")

        return {"object_key": objectKey, **stored}

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
                        actual = cls.hashFile(destinationFilesystem, posixpath.join(destinationRoot, key))

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

                    canonical = cls.stageRawFile(run.filesystem, sourcePath, destinationFilesystem, destinationRoot, record, objectKey=physicalKey(record["name"]))

                verifiedByHash[digest] = canonical

            stored = readableLocation(canonical, physicalKey(record["name"]))
            staged.append({"name": record["name"], **stored})

        return staged

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
        """Verify and register an exact payload location.

        Allows multiple copies of identical content on the same backend.
        A physical path cannot be reassigned to different content.

        Uses the caller's transaction; does not commit.
        Runtime storage options are never stored.
        """
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

        statement = sql.SQL("SELECT file_id FROM {} WHERE backend_id = %s AND object_key = %s").format(locationsTable)
        existing = connection.execute(statement, (backendID, objectKey)).fetchone()

        if existing is not None and existing[0] != fileID:
            raise ValueError("Storage path is already registered to different content.")

        filesystem, root = url_to_fs(backend[0], **(storageOptions or {}))
        actual = cls.hashFile(filesystem, posixpath.join(root, objectKey))

        if actual["sha256"] != identity[0] or actual["size_bytes"] != identity[1]:
            raise ValueError("Stored payload does not match the registered file identity.")

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
    def registerStagedRawCheckpoint(cls, connection, postgres, createdBy, backendID, manifest, staged, storedManifest, storageOptions=None, *, sessionID=None):
        """Register a staged behavioral archive without publishing it.

        Reads labels from verified archived experiment settings.
        Missing subjects are SQL NULL, not synthetic subject records.
        The public raw saver controls missing-metadata consent.

        Must run inside the caller's transaction.
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

        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")
        backend = connection.execute(sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s").format(backendsTable), (backendID,)).fetchone()

        if backend is None or not backend[1]:
            raise ValueError(f"Storage backend is missing or inactive: {backendID}")

        fileLocations = {name: {"sha256": record["sha256"], "size_bytes": record["size_bytes"], "locations": [{"backend_id": backendID, "url_prefix": backend[0], "object_key": record["object_key"]}]} for name, record in stagedByName.items()}
        archive = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend={backendID: storageOptions or {}})
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
            checkpoint = connection.execute(sql.SQL("SELECT status, origin, contract_version, created_by, deleted_at FROM {} WHERE session_id = %s FOR UPDATE").format(sessionsTable), (sessionID,)).fetchone()

            if checkpoint != ("in_progress", "raw", "pgl.raw.behavior.v1", createdBy, None):
                raise ValueError("Checkpoint must be an active in-progress raw behavioral checkpoint owned by the supplied creator.")

            if connection.execute(sql.SQL("SELECT 1 FROM {} WHERE session_id = %s LIMIT 1").format(linksTable), (sessionID,)).fetchone() is not None:
                raise ValueError("Checkpoint already has file links.")

            checkpointID = sessionID

        cls._recordSessionLabelMetadata(connection, postgres, checkpointID, metadata)
        registeredPayloads = {}

        def registerPayload(record):
            identity = (record["sha256"], record["size_bytes"], record["object_key"])

            if identity not in registeredPayloads:
                fileID = cls.getOrCreateFile(connection, postgres, record["sha256"], record["size_bytes"])
                cls.verifyAndRegisterStorageLocation(connection, postgres, fileID, backendID, record["object_key"], storageOptions=storageOptions)
                registeredPayloads[identity] = fileID

            return registeredPayloads[identity]

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
        
    @classmethod
    def loadRawManifest(cls, connection, postgres, sessionID, storageOptionsByBackend=None, maxBytes=16 * 1024 * 1024, *, includeDeleted=False):
        """Retrieve and verify a completed checkpoint's manifest.

        Runtime storage options are keyed by backend_id and are never saved.
        This checks content identity and contract version, not the full schema.
        
        Deleted sessions are hidden unless includeDeleted=True.
        The override permits inspection; it does not undelete the session.
        """
        from fsspec.core import url_to_fs

        if isinstance(maxBytes, bool) or not isinstance(maxBytes, int) or maxBytes <= 0:
            raise ValueError("maxBytes must be a positive integer.")

        located = cls.getRawManifestLocations(connection, postgres, sessionID, includeDeleted=includeDeleted)

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

    @classmethod
    def validateRawManifest(cls, manifest):
        """Validate the raw behavioral manifest contract."""
        return cls._validateBehaviorManifest(manifest, "pgl.raw.behavior.v1")

    @classmethod
    def validateDerivedManifest(cls, manifest):
        """Validate supported derived contracts without accessing payloads."""
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a dictionary")

        contract = manifest.get("contract_version")

        if contract == "pgl.derived.session.v1":
            return cls.validateDerivedSessionManifest(manifest)

        cls._validateBehaviorManifest(manifest, "pgl.derived.behavior.v1")

        if "history.json" not in {record["name"] for record in manifest["files"]}:
            raise ValueError("Derived manifest is missing history.json")

        return manifest
    
    @staticmethod
    def validateDerivedSessionManifest(manifest):
        """Validate a derived session inventory containing an MNE bundle."""

        def requireFields(value, names, label):
            if not isinstance(value, dict) or set(value) != set(names):
                raise ValueError(f"{label} has missing or unexpected fields")

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
    
    @staticmethod
    def _validateBehaviorManifest(manifest, contractVersion):
        """Validate the v1 manifest structure without accessing payloads."""

        def requireFields(value, fields, label):
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError(f"{label} has missing or unexpected fields.")

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
    
    @staticmethod
    def _mneCacheContext(postgres, sessionID):
        """Build a cache namespace from configured database identity."""
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807: raise ValueError("Invalid cache session ID.")

        identity = {"host": postgres.postgresHost, "port": postgres.postgresPort, "database": postgres.databaseName, "schema": postgres.databaseSchema}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        label = re.sub(r"[^A-Za-z0-9_-]", "_", f"{postgres.databaseName}-{postgres.databaseSchema}")[:64]
        return {"namespace": f"{label}-{digest}", "sessionID": sessionID}    

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

    @classmethod
    def loadSession(cls, sessionID, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, includeDeleted=False, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load a completed raw or derived checkpoint.

        Restores checkpoint ID and description, then binds a runtime history
        baseline for subsequent pipeline extraction and save bookkeeping.

        Supports behavioral-only, MNE-only, and mixed checkpoints.
        MNE cache precedence is explicit cacheDirectory, then settings.cacheDir.

        Unknown/hidden sessions warn at level 0 and return None.
        Invalid public arguments/configuration warn at level 1 and return None.
        Database failures warn at level 2 and return None.
        Missing configuration or connection cancellation returns None.

        Manifest, payload, baseline, and contract-validation errors propagate.
        No SQL connection is retained by the returned session.

        directLocalPolicy='immutable' requires source files and resolved paths
        to remain unchanged and available while readers use them.
        includeDeleted permits inspection without undeleting anything.
        """
        from .pglPostgres import pglPostgres
        from .pglSettings import pglSettingsManager

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

        try:
            try:
                resolvedSettings = pglSettingsManager.getSettings(settings=settings, settingsName=settingsName)

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
                    checkpoint = connection.execute(sql.SQL("SELECT origin, contract_version, description FROM {} WHERE session_id = %s AND (%s OR deleted_at IS NULL)").format(table), (sessionID, includeDeleted)).fetchone()

                    if checkpoint is not None:
                        origin, contractVersion, description = checkpoint
                        loadOptions = {"storageOptionsByBackend": storageOptionsByBackend, "includeDeleted": includeDeleted}

                        if (origin, contractVersion) == ("raw", "pgl.raw.behavior.v1"):
                            loader = cls.loadRawSession

                        elif (origin, contractVersion) == ("raw", "pgl.raw.session.v1"):
                            loader = cls.loadRawSessionBundle
                            resolvedCache = cls._resolveCacheDirectory(cacheDirectory, settings=resolvedSettings)
                            loadOptions.update(cacheDirectory=resolvedCache, directLocalPolicy=directLocalPolicy)

                        elif origin == "derived" and contractVersion in ("pgl.derived.behavior.v1", "pgl.derived.session.v1"):
                            loader = cls.loadDerivedSession
                            resolvedCache = cls._resolveCacheDirectory(cacheDirectory, settings=resolvedSettings) if contractVersion == "pgl.derived.session.v1" else cacheDirectory
                            loadOptions.update(cacheDirectory=resolvedCache, directLocalPolicy=directLocalPolicy)

                        else:
                            raise ValueError(f"Unsupported checkpoint origin/contract: {origin!r}, {contractVersion!r}")

                        session = loader(connection, postgres, sessionID, **loadOptions)

        except psycopg.Error as error:
            pglMessages.warning(f"Could not load session {sessionID}: {error}", level=2)
            return None

        if checkpoint is None:
            reason = "Unknown checkpoint" if includeDeleted else "Unknown or deleted checkpoint"
            pglMessages.warning(f"{reason}: {sessionID}", level=0)
            return None

        # The loader has established the checkpoint's contents and history.
        # Runtime metadata is attached without opening scientific payloads.
        session.sessionID = sessionID
        session.description = checkpoint[2]
        cls._bindCheckpointBaseline(session, sessionID, postgres)

        pglMessages.message(f"Loaded {checkpoint[0]} checkpoint {sessionID}; backed fields will load on demand.")
        return session
    
    @staticmethod
    def _requireSafeCheckpoint(session):
        """Reject working objects affected by failed action attempts.

        This guard checks runtime failure bookkeeping only.
        It does not perform scientific revalidation or read payloads.
        A clean guard result is not proof of complete scientific validity.
        """
        from .pglPipeline import pglActionable

        if not isinstance(session, pglActionable):
            raise TypeError("Checkpoint saving requires a pglActionable object.")

        failedIDs = tuple(session._failedActionIDs)

        if not failedIDs:
            return

        failedIDSet = set(failedIDs)
        names = list(dict.fromkeys(entry.actionName for entry in session.actionHistory if entry.actionID in failedIDSet))
        details = ", ".join(names) if names else ", ".join(failedIDs)

        raise ValueError(f"Cannot save checkpoint: this working session may contain partial changes from failed actions: {details}. Reload a saved checkpoint and rerun the corrected analysis. A later successful action does not clear this restriction.")
    
    @classmethod
    def _saveDerivedSession(cls, connection, postgres, session, createdBy, backendID, *, parentSessionID, description=None, ingestMethod="manual", storageOptions=None):
        """Save current contents as a derived checkpoint, or reuse its parent.

        Reuses only an active completed derived parent with identical encoded
        manifest, description, ingest method, creator, and modalities, whose
        linked files have eligible locations on the requested backend.

        A raw parent always produces a new derived checkpoint.

        Exports a verified temporary snapshot before deciding whether a new
        checkpoint is needed. This avoids duplicate permanent writes, but
        does not avoid serialization and hashing.

        Uses the caller's transaction with a nested transaction/savepoint.
        Temporary staging is removed. Failed destination writes may leave
        unregistered files.

        Returns the new or reused checkpoint ID. Does not update the
        in-memory session ID or clear its action history.
        """
        from tempfile import TemporaryDirectory
        from fsspec.implementations.local import LocalFileSystem
        from .pglSession import pglMNE

        for label, value in (("createdBy", createdBy), ("backendID", backendID), ("parentSessionID", parentSessionID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if description is not None and (not isinstance(description, str) or "\x00" in description):
            raise ValueError("description must be a string without NUL, or None.")

        if not isinstance(ingestMethod, str) or ingestMethod not in {"manual", "automated"}:
            raise ValueError("ingestMethod must be manual or automated.")

        hasMNE = session.mne is not None

        if not session.runs and not hasMNE:
            raise ValueError("A derived checkpoint must contain behavioral runs or MNE contents.")

        if hasMNE and type(session.mne) is not pglMNE:
            raise TypeError("MNE checkpoints currently support pglMNE, not subclasses.")

        contractVersion = "pgl.derived.session.v1" if hasMNE else "pgl.derived.behavior.v1"
        modalities = (["behavior"] if session.runs else []) + (["mne"] if hasMNE else [])

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        users = sql.Identifier(postgres.databaseSchema, "users")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")

        def isLocal(filesystem):
            protocols = filesystem.protocol
            protocols = {protocols} if isinstance(protocols, str) else set(protocols)
            return bool(protocols.intersection({"file", "local"}))

        with connection.transaction():
            creator = connection.execute(sql.SQL("SELECT 1 FROM {} WHERE user_id = %s FOR SHARE").format(users), (createdBy,)).fetchone()

            if creator is None:
                raise ValueError(f"Unknown creator: {createdBy}")

            backend = connection.execute(sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backends), (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError(f"Storage backend is missing or inactive: {backendID}")

            statement = sql.SQL("SELECT origin, status, contract_version, manifest_hash, description, ingest_method, created_by, modalities, deleted_at FROM {} WHERE session_id = %s FOR SHARE").format(sessions)
            parent = connection.execute(statement, (parentSessionID,)).fetchone()

            if parent is None or parent[1] != "complete" or parent[8] is not None:
                raise ValueError("Parent must be a completed, active checkpoint.")

            destinationFS, destinationRoot = url_to_fs(backend[0], **(storageOptions or {}))
            stagingFS = LocalFileSystem(skip_instance_cache=True)

            with TemporaryDirectory(prefix="pgl-derived-session-") as temporary:
                pglMessages.message("Saving session: exporting current contents to temporary staging...")
                stagingRoot, manifest = cls.saveDerivedSessionFiles(session, dataPath=Path(temporary) / "bundle", filesystem=stagingFS)

                if manifest["contract_version"] != contractVersion:
                    raise ValueError("Exported manifest contract differs from the expected checkpoint contract.")

                manifestBytes = cls.encodeRawManifest(manifest)
                manifestHash = hashlib.sha256(manifestBytes).hexdigest()

                pglMessages.message("Saving session: comparing contents with the parent checkpoint...")
                sameParent = parent[0] == "derived" and parent[2] == contractVersion and parent[3] == manifestHash
                sameMetadata = parent[4] == description and parent[5] == ingestMethod and parent[6] == createdBy and sorted(parent[7]) == sorted(modalities)

                if sameParent and sameMetadata:
                    expected = {record["name"]: (record["sha256"], record["size_bytes"], "history" if record["name"] == "history.json" else "derived") for record in manifest["files"]}
                    expected["manifest.json"] = (manifestHash, len(manifestBytes), "manifest")

                    statement = sql.SQL("SELECT link.name, content.sha256, content.size_bytes, link.role, (link.deleted_at IS NULL AND content.deleted_at IS NULL AND EXISTS (SELECT 1 FROM {} AS location WHERE location.file_id = content.file_id AND location.backend_id = %s AND location.deleted_at IS NULL AND location.status = 'present' AND location.verified_at IS NOT NULL)) AS available FROM {} AS link JOIN {} AS content ON content.file_id = link.file_id WHERE link.session_id = %s ORDER BY link.name").format(locations, links, files)
                    rows = connection.execute(statement, (backendID, parentSessionID)).fetchall()
                    actual = {name: (digest, size, role) for name, digest, size, role, available in rows}

                    if actual != expected:
                        raise ValueError("Parent checkpoint file links do not match its manifest identity; it cannot be reused.")

                    if all(available for name, digest, size, role, available in rows):
                        pglMessages.warning(f"Session contents and save metadata match checkpoint {parentSessionID}. No new checkpoint was created; returning the existing ID. Make changes before calling saveSession() again.", level=2)
                        return parentSessionID

                    pglMessages.message("Saving session: matching parent lacks eligible files on the requested backend; creating a new checkpoint.")

                pglMessages.message("Saving session: allocating a new derived checkpoint...")
                sessionID = cls.createDerivedCheckpoint(connection, postgres, createdBy, parentSessionID, description=description, ingestMethod=ingestMethod, contractVersion=contractVersion, modalities=modalities)
                objectPrefix = f"session{sessionID:06d}"
                destination = posixpath.join(destinationRoot, objectPrefix)

                for run in session.runs:
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
                records = [*manifest["files"], {"name": "manifest.json", "sha256": manifestHash, "size_bytes": len(manifestBytes)}]

                for record in records:
                    cls.stageRawFile(stagingFS, posixpath.join(stagingRoot, record["name"]), destinationFS, destinationRoot, record, objectKey=posixpath.join(objectPrefix, record["name"]))

                pglMessages.message("Saving session: registering checkpoint files...")
                cls.registerDerivedSessionFiles(connection, postgres, sessionID, backendID, manifest, objectPrefix=objectPrefix, storageOptions=storageOptions)

                pglMessages.message("Saving session: validating checkpoint completion...")
                cls.completeDerivedCheckpoint(connection, postgres, sessionID, manifest)

                return sessionID

    @classmethod
    def saveSession(cls, session, *, parentSessionID=None, settings=None, settingsName=None, backendID=None, description=None, ingestMethod="manual", storageOptions=None, sslRootCert=None):
        """Save current contents and history as a derived checkpoint.

        Defaults parentSessionID to session.sessionID.
        description=None carries forward session.description.
        An explicit blank description clears it.

        Rejects failed working state and changes to an established baseline's
        historical prefix before opening a database connection.

        Returns the new or reused checkpoint ID, or None for missing-parent
        rejection, unavailable configuration, or connection cancellation.

        Prepares new baseline metadata before commit. Updates sessionID,
        description, and baseline only after commit and connection cleanup.

        Database changes are transactional; payload writes are not.
        Failed saves may leave unregistered files. Other errors propagate.
        Accumulated action history is retained.

        The working object must not be modified concurrently while saving.
        """
        from time import perf_counter
        from .pglPostgres import pglPostgres
        from .pglSettings import pglSettingsManager

        # These checks inspect provenance bookkeeping, not scientific payloads.
        # A later successful action must not conceal an earlier failed mutation.
        cls._requireSafeCheckpoint(session)
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

        resolvedDescription = session.description if description is None else description

        if resolvedDescription is not None:
            if not isinstance(resolvedDescription, str) or "\x00" in resolvedDescription:
                raise ValueError("description must be a string without NUL, or None.")

            if not resolvedDescription.strip():
                resolvedDescription = None

        if not isinstance(ingestMethod, str) or ingestMethod not in {"manual", "automated"}:
            raise ValueError("ingestMethod must be manual or automated.")

        resolvedSettings = pglSettingsManager.getSettings(settings=settings, settingsName=settingsName)

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
                sessionID = cls._saveDerivedSession(connection, postgres, session, createdBy, backendID, parentSessionID=resolvedParentID, description=resolvedDescription, ingestMethod=ingestMethod, storageOptions=storageOptions)

                # Recheck the historical prefix and prepare all new fingerprints
                # while an error can still roll back database publication.
                cls._requireSafeCheckpoint(session)
                cls._getCheckpointHistoryBoundary(session)
                pendingBaseline = cls._prepareCheckpointBaseline(session, sessionID, postgres)

                pglMessages.message(f"Saving session: checkpoint ready ({perf_counter() - contentsStarted:.1f} s); committing transaction...")

        # Do not serialize history or resolve storage here. Publication and
        # connection cleanup have succeeded; install the prepared metadata.
        session.sessionID = sessionID
        session.description = resolvedDescription
        session._checkpointBaseline = pendingBaseline

        pglMessages.message(f"Checkpoint {sessionID} is complete. Database configuration: {databasePath} (database={postgres.databaseName}, schema={postgres.databaseSchema}; {perf_counter() - saveStarted:.1f} s total).", wrapText=False)
        return sessionID
    
    @classmethod
    def loadRawSession(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False):
        """Reconstruct a completed raw behavioral checkpoint as a lazy session.

        Uses the caller's SQL connection for metadata lookup only.
        Verifies the manifest immediately; acquisition payloads are verified
        when opened through the returned session's filesystem.
        Deleted sessions are hidden unless includeDeleted=True.
        The override permits inspection; it does not undelete the session.
        """
        from .pglSession import pglSession

        manifest = cls.loadRawManifest(connection, postgres, sessionID, storageOptionsByBackend=storageOptionsByBackend, includeDeleted=includeDeleted)
        fileLocations = cls.getRawFileLocations(connection, postgres, sessionID, manifest, includeDeleted=includeDeleted)

        filesystem = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend=storageOptionsByBackend)
        runPaths = [record["path"] for record in manifest["runs"]]

        return pglSession(filesystem=filesystem, runList=runPaths)
    
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
                cls.stageRawFile(run.filesystem, sourcePath, filesystem, destination, record, objectKey=record["name"])
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
                files.append({"name": posixpath.join(relative, name), **cls.hashFile(filesystem, path)})

        historyPath = posixpath.join(destination, "history.json")

        with filesystem.open(historyPath, "xb") as target:
            if target.write(historyBytes) != len(historyBytes):
                raise OSError("Incomplete raw session history write.")

        historyIdentity = {"sha256": hashlib.sha256(historyBytes).hexdigest(), "size_bytes": len(historyBytes)}

        if cls.hashFile(filesystem, historyPath) != historyIdentity:
            raise ValueError("Raw session history verification failed.")

        files.append({"name": "history.json", **historyIdentity})
        manifest = {"contract_version": "pgl.raw.session.v1", "runs": runs, "mne": {"path": "mne"}, "directories": sorted(directories), "files": sorted(files, key=lambda record: record["name"])}
        cls.validateRawSessionManifest(manifest)

        class DestinationView:
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

        if cls.hashFile(filesystem, manifestPath) != expected:
            raise ValueError("Raw session manifest verification failed.")

        return destination, manifest
    
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
    
    @staticmethod
    def _validatePipelineReplayInput(record, expected=None):
        """Validate replay input metadata and optionally compare it to a plan.

        Uses metadata only; does not connect, read payloads, or mutate records.

        Endpoint identity prevents accidental use of a different configured
        database. It is not a permanent store UUID across relocation/restores.
        """
        expectedFields = {"version", "kind", "sessionID", "databaseIdentity", "historyCount", "historyHashes"}

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

        stored = cls.hashFile(destinationFilesystem, destinationPath)

        if stored != expected:
            raise ValueError("Stored derived manifest verification failed.")

        return {"object_key": "manifest.json", **stored}

    @staticmethod
    def createDerivedCheckpoint(connection, postgres, createdBy, parentSessionID, description=None, ingestMethod="manual", *, contractVersion="pgl.derived.behavior.v1", modalities=None):
        """Create an unpublished derived checkpoint in the caller's transaction.

        Database lineage rules require a completed, active parent.
        modalities describes content categories; 'mne' does not imply EEG/MEG.
        """
        for label, value in (("createdBy", createdBy), ("parentSessionID", parentSessionID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807: raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if description is not None and (not isinstance(description, str) or "\x00" in description): raise ValueError("description must be a string without NUL, or None.")
        if ingestMethod not in ("manual", "automated"): raise ValueError("ingestMethod must be manual or automated.")
        if contractVersion not in ("pgl.derived.behavior.v1", "pgl.derived.session.v1"): raise ValueError("Unsupported derived checkpoint contract.")

        if modalities is None: modalities = ["behavior"] if contractVersion == "pgl.derived.behavior.v1" else ["mne"]
        if not isinstance(modalities, (list, tuple)) or not modalities: raise ValueError("modalities must be a nonempty list or tuple.")
        if any(not isinstance(value, str) or value not in ("behavior", "mne") for value in modalities): raise ValueError("Supported content categories are 'behavior' and 'mne'.")
        if len(set(modalities)) != len(modalities): raise ValueError("Duplicate modalities are not allowed.")
        if contractVersion == "pgl.derived.behavior.v1" and set(modalities) != {"behavior"}: raise ValueError("Behavioral checkpoints require only the behavior category.")
        if contractVersion == "pgl.derived.session.v1" and "mne" not in modalities: raise ValueError("MNE session checkpoints require the mne category.")

        table = sql.Identifier(postgres.databaseSchema, "session")
        statement = sql.SQL("""
            INSERT INTO {} (
                origin, kind, parent_session_id, ingest_method,
                modalities, status, description, contract_version,
                created_by, started_at
            )
            VALUES ('derived', 'processed', %s, %s, %s::text[], 'in_progress', %s, %s, %s, now())
            RETURNING session_id
        """).format(table)

        return connection.execute(statement, (parentSessionID, ingestMethod, list(modalities), description, contractVersion, createdBy)).fetchone()[0]

    @classmethod
    def saveDerivedSessionFiles(cls, session, dataPath, filesystem=None, filesystemPrefix=None):
        """Export current behavioral/MNE contents, history, labels, and manifest.

        Labels are included in hashed history.json.
        Requires a fresh destination and writes manifest.json last.
        Does not register or complete a database checkpoint.
        """
        from .pglBase import pglBase
        from .pglSession import pglMNE

        runs = list(session.runs or [])
        mneData = session.mne

        if not runs and mneData is None:
            raise ValueError("A checkpoint must contain behavioral runs or MNE contents")

        if mneData is not None and type(mneData) is not pglMNE:
            raise TypeError("MNE export currently supports pglMNE, not subclasses")

        if dataPath is None or not str(dataPath).strip():
            raise ValueError("A destination path is required")

        filesystem, destination, _ = pglBase.validateFilesystem(filesystem=filesystem, dataPath=dataPath, filesystemPrefix=filesystemPrefix, requireExists=False)

        if filesystem is None:
            raise ValueError("Could not resolve destination filesystem")

        if filesystem.exists(destination):
            raise FileExistsError(f"Destination already exists: {destination}")

        historyBytes = cls._sessionHistoryBytes(session, acquisition=False)
        filesystem.makedirs(destination, exist_ok=False)
        directories = set()
        files = []
        runRecords = []

        def inventoryDirectory(relativeRoot):
            fullRoot = posixpath.join(destination, relativeRoot)
            directories.add(relativeRoot)

            for currentPath, directoryNames, fileNames in filesystem.walk(fullRoot, detail=False, on_error="raise"):
                logicalPath = posixpath.relpath(currentPath, destination)

                for name in directoryNames:
                    directories.add(posixpath.join(logicalPath, name))

                for name in fileNames:
                    savedPath = posixpath.join(currentPath, name)
                    files.append({"name": posixpath.join(logicalPath, name), **cls.hashFile(filesystem, savedPath)})

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
                raise OSError("Incomplete session history write")

        expectedHistory = {"sha256": hashlib.sha256(historyBytes).hexdigest(), "size_bytes": len(historyBytes)}
        storedHistory = cls.hashFile(filesystem, historyPath)

        if storedHistory != expectedHistory:
            raise ValueError("Saved session history verification failed")

        files.append({"name": "history.json", **storedHistory})
        manifest = {"contract_version": "pgl.derived.session.v1" if mneData is not None else "pgl.derived.behavior.v1", "runs": runRecords, "directories": sorted(directories), "files": sorted(files, key=lambda record: record["name"])}

        if mneData is not None:
            manifest["mne"] = {"path": "mne"}

        cls.validateDerivedManifest(manifest)

        class DestinationView:
            def open(self, path, mode="rb", **kwargs):
                return filesystem.open(posixpath.join(destination, path), mode=mode, **kwargs)

        view = DestinationView()
        _, metadata = cls._readSessionHistory(view, len(runs))

        if runs and metadata != cls._behaviorLabelMetadata(manifest, view):
            raise ValueError("Derived history labels differ from the exported behavioral settings.")

        if mneData is not None:
            cls._verifyMNEInventory(manifest, view)

        cls.stageDerivedManifest(manifest, filesystem, destination)
        return destination, manifest
    
    @classmethod
    def _saveRawMNECheckpoint(cls, connection, postgres, session, createdBy, backendID, storageOptions=None):
        """Save/register/complete an MNE-containing parentless raw root.

        Uses the caller's transaction with a nested transaction/savepoint.
        Exports a verified temporary snapshot before allocating destination
        payload paths.

        Exact encoded-manifest matches reuse completed active roots with a
        warning. This is byte-identity deduplication, not semantic MNE
        equivalence.

        Matching deleted checkpoints remain unchanged; a new checkpoint
        is created with a separate checkpoint-specific destination.

        Temporary staging is removed. Failed destination writes may leave
        unregistered files.
        """
        from tempfile import TemporaryDirectory
        from fsspec.implementations.local import LocalFileSystem
        from .pglSession import pglMNE

        for label, value in (("createdBy", createdBy), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if type(session.mne) is not pglMNE:
            raise TypeError("MNE raw checkpoints require a pglMNE instance.")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        users = sql.Identifier(postgres.databaseSchema, "users")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        with connection.transaction():
            creator = connection.execute(sql.SQL("SELECT 1 FROM {} WHERE user_id = %s FOR SHARE").format(users), (createdBy,)).fetchone()

            if creator is None:
                raise ValueError(f"Unknown creator: {createdBy}")

            backend = connection.execute(sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backends), (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError("Storage backend is missing or inactive.")

            destinationFS, destinationRoot = url_to_fs(backend[0], **(storageOptions or {}))
            stagingFS = LocalFileSystem(skip_instance_cache=True)

            with TemporaryDirectory(prefix="pgl-raw-session-") as temporary:
                stagingRoot, manifest = cls.saveRawSessionFiles(session, Path(temporary) / "bundle", filesystem=stagingFS)
                manifestBytes = cls.encodeRawManifest(manifest)
                manifestHash = hashlib.sha256(manifestBytes).hexdigest()

                # Serialize saves of identical raw manifests.
                connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (manifestHash,))
                existing = connection.execute(sql.SQL("SELECT session_id, status, contract_version FROM {} WHERE origin = 'raw' AND manifest_hash = %s AND deleted_at IS NULL FOR SHARE").format(sessions), (manifestHash,)).fetchone()

                if existing is not None:
                    existingID, status, contract = existing

                    if (status, contract) != ("complete", "pgl.raw.session.v1"):
                        raise ValueError("Matching raw manifest belongs to an unsupported or incomplete checkpoint.")

                    # Validate registered inventory and metadata without resolving MNE fields.
                    cls.loadRawSessionBundle(connection, postgres, existingID, storageOptionsByBackend={backendID: storageOptions or {}}, cacheDirectory=Path(temporary) / "unused-cache")
                    pglMessages.warning(f"This raw session is already saved as active checkpoint {existingID}. No new checkpoint was created; returning the existing ID. To save another raw copy, delete checkpoint {existingID} and call saveRawSession() again. To save further work as a derived checkpoint, use saveSession().", level=2)
                    return existingID

                deletedMatches = connection.execute(sql.SQL("SELECT session_id FROM {} WHERE origin = 'raw' AND manifest_hash = %s AND deleted_at IS NOT NULL ORDER BY session_id FOR SHARE").format(sessions), (manifestHash,)).fetchall()

                if deletedMatches:
                    deletedIDs = ", ".join(str(row[0]) for row in deletedMatches)
                    pglMessages.message(f"Found matching deleted raw checkpoint(s) {deletedIDs}; leaving them unchanged and saving a new checkpoint.")

                modalities = (["behavior"] if manifest["runs"] else []) + ["mne"]
                statement = sql.SQL("INSERT INTO {} (origin, kind, ingest_method, modalities, status, contract_version, created_by, started_at) VALUES ('raw', 'acquired', 'manual', %s::text[], 'in_progress', 'pgl.raw.session.v1', %s, now()) RETURNING session_id").format(sessions)
                sessionID = connection.execute(statement, (modalities, createdBy)).fetchone()[0]
                objectPrefix = f"session{sessionID:06d}"
                destination = posixpath.join(destinationRoot, objectPrefix)

                def isLocal(filesystem):
                    protocols = filesystem.protocol
                    protocols = {protocols} if isinstance(protocols, str) else set(protocols)
                    return bool(protocols.intersection({"file", "local"}))

                for run in session.runs:
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

                records = [*manifest["files"], {"name": "manifest.json", "sha256": manifestHash, "size_bytes": len(manifestBytes)}]

                for record in records:
                    cls.stageRawFile(stagingFS, posixpath.join(stagingRoot, record["name"]), destinationFS, destinationRoot, record, objectKey=posixpath.join(objectPrefix, record["name"]))

                cls._registerSessionBundleFiles(connection, postgres, sessionID, backendID, manifest, objectPrefix=objectPrefix, storageOptions=storageOptions, origin="raw")
                cls._completeSessionBundleCheckpoint(connection, postgres, sessionID, manifest, origin="raw")
                return sessionID
    
    @classmethod
    def _readReplayMetadataFile(cls, connection, postgres, sessionID, name, *, storageOptionsByBackend=None, maxBytes=64 * 1024 * 1024):
        """Read a verified registered metadata file for replay inspection.

        Caller owns the read-only database snapshot and validates the session.
        Only manifest.json and history.json are permitted.

        Returns the bytes and registered content identity.
        """
        if name not in ("manifest.json", "history.json"):
            raise ValueError("Replay inspection may read only manifest.json or history.json.")

        if type(maxBytes) is not int or maxBytes <= 0:
            raise ValueError("maxBytes must be a positive integer.")

        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        statement = sql.SQL("SELECT content.file_id, content.sha256, content.size_bytes, link.role FROM {} AS link JOIN {} AS content ON content.file_id = link.file_id WHERE link.session_id = %s AND link.name = %s AND link.deleted_at IS NULL AND content.deleted_at IS NULL").format(links, files)
        record = connection.execute(statement, (sessionID, name)).fetchone()

        if record is None:
            raise ValueError(f"Checkpoint {sessionID} has no active registered {name}.")

        fileID, digest, size, role = record
        expectedRole = "manifest" if name == "manifest.json" else "history"

        if role != expectedRole:
            raise ValueError(f"Checkpoint {sessionID}: {name} has an unexpected role.")

        if type(size) is not int or not 0 <= size <= maxBytes:
            raise ValueError(f"Checkpoint {sessionID}: {name} exceeds the replay metadata size limit.")

        statement = sql.SQL("SELECT backend.backend_id, backend.url_prefix, location.object_key FROM {} AS location JOIN {} AS backend ON backend.backend_id = location.backend_id WHERE location.file_id = %s AND location.deleted_at IS NULL AND location.status = 'present' AND location.verified_at IS NOT NULL AND backend.is_active ORDER BY location.priority, backend.backend_id, location.object_key").format(locations, backends)
        candidates = connection.execute(statement, (fileID,)).fetchall()
        failures = []
        optionsByBackend = storageOptionsByBackend or {}

        for backendID, root, key in candidates:
            if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                raise ValueError("Database contains an invalid replay metadata object key.")

            try:
                filesystem, prefix = url_to_fs(root, **optionsByBackend.get(backendID, {}))

                with filesystem.open(posixpath.join(prefix, key), "rb") as source:
                    content = source.read(size + 1)

                if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("Metadata content identity mismatch.")

            except (OSError, ValueError) as error:
                failures.append(f"backend {backendID}: {type(error).__name__}")
                continue

            return content, {"sha256": digest, "size_bytes": size}

        raise OSError(f"Cannot verify checkpoint {sessionID} {name}: " + ("; ".join(failures) or "no eligible storage locations"))
    
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
    
    @staticmethod
    def exportFigRows(filename=None, *, filesystem=None, storageOptions=None, dpi=100, overwrite=True, maxPixels=100_000_000):
        """Create a collector for exporting figure rows as one PNG.

        None creates a no-op collector. No browser or inline display is opened.
        Use add(fig), finish(), and close(), or a with block for cleanup.
        """
        return pglFigureRowExport(filename, filesystem=filesystem, storageOptions=storageOptions, dpi=dpi, overwrite=overwrite, maxPixels=maxPixels)
    
    @staticmethod
    def exportFig(fig, filename=None, *, filesystem=None, storageOptions=None, dpi=100, overwrite=True):
        """Export a Matplotlib figure through fsspec or open it in a browser.

        Without a filename, render a temporary PNG and request a browser tab.
        No inline Jupyter display is produced. Temporary browser files remain
        available after this method returns.

        Supported formats: PNG, SVG, PDF, JPEG, and WebP.
        Missing or unsupported filename extensions become .png.

        Plain local paths expand ~. Remote paths use backend semantics.
        Existing output is protected unless overwrite=True.

        Render before opening the output file. Writes are not transactional;
        a failed write may leave partial output.

        Returns a resolved filesystem path or temporary browser-image path.
        Failures warn and return None. Browser-launch failure warns but returns
        the successfully rendered temporary path.

        Does not close or clear the supplied figure.
        """
        import os
        import tempfile
        import webbrowser
        from io import BytesIO
        from pathlib import Path
        from matplotlib.figure import Figure
        from fsspec import AbstractFileSystem
        from fsspec.core import url_to_fs

        try:
            if not isinstance(fig, Figure):
                raise TypeError("fig must be a Matplotlib Figure.")

            if type(dpi) is not int or dpi <= 0:
                raise ValueError("dpi must be a positive integer.")

            if type(overwrite) is not bool:
                raise ValueError("overwrite must be a boolean.")

            if filesystem is not None and storageOptions:
                raise ValueError("Supply filesystem or storageOptions, not both.")

            if filesystem is not None and not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec AbstractFileSystem.")

            displayOnly = filename is None
            outputFormat = "png"
            destination = None
            destinationFS = None

            if displayOnly:
                if filesystem is not None or storageOptions:
                    raise ValueError("Filesystem options require an export filename.")

            else:
                location = os.fspath(filename)

                if not isinstance(location, str) or not location.strip():
                    raise ValueError("filename must be a nonempty path or URL.")

                if any(character in location for character in ("\x00", "\n", "\r")):
                    raise ValueError("filename must not contain NUL or newline characters.")

                if location.endswith(("/", "\\")):
                    raise ValueError("Supply a figure filename, not only a directory.")

                if filesystem is None:
                    # Expand local home paths only. Remote ~ belongs to the
                    # remote filesystem and must not use this client's home.
                    if "://" not in location and "::" not in location:
                        location = str(Path(location).expanduser())

                    destinationFS, destination = url_to_fs(location, **(storageOptions or {}))
                else:
                    destinationFS = filesystem
                    destination = destinationFS._strip_protocol(location)

                protocols = destinationFS.protocol
                protocols = {protocols} if isinstance(protocols, str) else set(protocols)

                if protocols.intersection({"file", "local"}):
                    destination = str(Path(destination).expanduser().resolve())

                if not destination or destination.endswith("/"):
                    raise ValueError("Supply a figure filename.")

                if destinationFS.isdir(destination):
                    raise IsADirectoryError("The requested figure destination is a directory.")

                stem, extension = posixpath.splitext(destination)
                extension = extension.lower()
                formats = {".png": "png", ".svg": "svg", ".pdf": "pdf", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}

                if extension not in formats:
                    destination = stem + ".png"

                    if extension:
                        pglMessages.warning(f"Unsupported figure extension {extension!r}; exporting as PNG instead.", level=0)
                else:
                    destination = stem + extension
                    outputFormat = formats[extension]

                # Check the normalized output path, not just the requested path.
                if destinationFS.exists(destination):
                    if destinationFS.isdir(destination):
                        raise IsADirectoryError("The normalized figure destination is a directory.")

                    if not overwrite:
                        raise FileExistsError("The figure already exists; use overwrite=True to replace it.")

            # Complete rendering before creating or truncating the destination.
            with BytesIO() as buffer:
                fig.savefig(buffer, format=outputFormat, dpi=dpi, facecolor="white", transparent=False)
                content = buffer.getvalue()

            if displayOnly:
                # Browser-only display in notebooks and ordinary Python alike.
                # Do not delete immediately: browsers may read asynchronously.
                with tempfile.NamedTemporaryFile(prefix="pgl-figure-", suffix=".png", delete=False) as temporary:
                    written = temporary.write(content)

                    if written != len(content):
                        raise OSError("Incomplete temporary figure write.")

                    displayPath = Path(temporary.name).resolve()

                try:
                    opened = webbrowser.open(displayPath.as_uri(), new=2)
                except Exception:
                    opened = False

                if opened:
                    pglMessages.message(f"Requested browser display: {displayPath}")
                else:
                    pglMessages.warning(f"Could not open a browser. The figure is available at {displayPath}.", level=1)

                return str(displayPath)

            parent = posixpath.dirname(destination)

            if parent:
                destinationFS.makedirs(parent, exist_ok=True)

            # Use exclusive creation where the backend supports it. Do not
            # silently fall back to overwriting an existing destination.
            mode = "wb" if overwrite else "xb"

            with destinationFS.open(destination, mode) as target:
                written = target.write(content)

                if written != len(content):
                    raise OSError("Incomplete figure write.")

            pglMessages.message(f"Figure exported as {outputFormat.upper()}: {posixpath.basename(destination)}")
            return destination

        except FileExistsError:
            pglMessages.warning("Figure export skipped: the output file already exists. Use overwrite=True to replace it.", level=1)
            return None

        except IsADirectoryError:
            pglMessages.warning("Figure export skipped: the destination is a directory. Supply a filename such as evoked.png.", level=1)
            return None

        except Exception as error:
            # Do not print backend exception text or full URLs: they may expose
            # credentials. KeyboardInterrupt and SystemExit still propagate.
            pglMessages.warning(f"Could not export/display figure ({type(error).__name__}). Check the figure, filename, format support, storage options, and permissions. Processing will continue.", level=1)
            return None
    
    @classmethod
    def saveRawSession(cls, session, *, settings=None, settingsName=None, backendID=None, storageOptions=None, sslRootCert=None, allowMissingMetadata=False):
        """Save a parentless acquisition/root checkpoint.

        Behavioral labels/files come from saved acquisition sources.
        MNE-only labels come from session.experimentName and session.subjectID.

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
        from .pglSettings import pglSettingsManager

        if not cls._checkRawSessionLabels(session, allowMissingMetadata=allowMissingMetadata):
            return None

        resolvedSettings = pglSettingsManager.getSettings(settings=settings, settingsName=settingsName)

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

        if backendID is None and not postgres.storageLocations:
            pglMessages.warning("Cannot save raw session: no storageLocations are configured in the PostgreSQL configuration.", level=0)
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

    @classmethod
    def _saveRawSession(cls, connection, postgres, session, createdBy, backendID, storageOptions=None):
        """Save a parentless raw checkpoint using the caller's connection.

        Behavioral-only sessions use the existing acquisition archive path.
        MNE-only and mixed sessions use the raw session bundle exporter.

        Behavioral acquisition files are archived unchanged.
        MNE objects are serialized from their current contents.

        Opens a transaction/savepoint. An outer transaction must commit for
        persistence. Payload writes are not rolled back.

        Exact raw manifest matches reuse completed active checkpoints with
        a warning and return the existing checkpoint ID.

        Matching deleted checkpoints remain unchanged; a new checkpoint
        is created.
        """
        if session.mne is not None:
            return cls._saveRawMNECheckpoint(connection, postgres, session, createdBy, backendID, storageOptions=storageOptions)

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

            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (manifestHash,))
            statement = sql.SQL("SELECT session_id, status, contract_version FROM {} WHERE manifest_hash = %s AND origin = 'raw' AND deleted_at IS NULL FOR SHARE").format(sessionsTable)
            existing = connection.execute(statement, (manifestHash,)).fetchone()

            if existing is not None:
                existingID, status, contractVersion = existing

                if (status, contractVersion) != ("complete", "pgl.raw.behavior.v1"):
                    raise ValueError("Matching manifest belongs to an incomplete or unsupported checkpoint.")

                optionsByBackend = {backendID: storageOptions or {}}
                archivedManifest = cls.loadRawManifest(connection, postgres, existingID, storageOptionsByBackend=optionsByBackend)

                if archivedManifest != manifest:
                    raise ValueError("Existing checkpoint manifest differs from source manifest.")

                cls.getRawFileLocations(connection, postgres, existingID, archivedManifest)
                pglMessages.warning(f"This raw session is already saved as active checkpoint {existingID}. No new checkpoint was created; returning the existing ID. To save another raw copy, delete checkpoint {existingID} and call saveRawSession() again. To save further work as a derived checkpoint, use saveSession().", level=2)
                return existingID

            statement = sql.SQL("SELECT session_id FROM {} WHERE manifest_hash = %s AND origin = 'raw' AND deleted_at IS NOT NULL ORDER BY session_id FOR SHARE").format(sessionsTable)
            deletedMatches = connection.execute(statement, (manifestHash,)).fetchall()

            if deletedMatches:
                deletedIDs = ", ".join(str(row[0]) for row in deletedMatches)
                pglMessages.message(f"Found matching deleted raw checkpoint(s) {deletedIDs}; leaving them unchanged and saving a new checkpoint.")

            filesystem, root = url_to_fs(backend[0], **(storageOptions or {}))
            sessionID = cls.createRawCheckpoint(connection, postgres, createdBy)
            staged = cls.stageRawSession(session, manifest, filesystem, root, sessionID=sessionID, connection=connection, postgres=postgres, backendID=backendID)
            storedManifest = cls.stageRawManifest(manifest, filesystem, root, sessionID=sessionID)
            cls.registerStagedRawCheckpoint(connection, postgres, createdBy, backendID, manifest, staged, storedManifest, storageOptions=storageOptions, sessionID=sessionID)
            cls.completeRawCheckpoint(connection, postgres, sessionID, manifest)
            return sessionID
    
    @classmethod
    def _getConfiguredStorageBackend(cls, connection, postgres, storageOptions=None):
        """Resolve/register the first configured payload destination.

        Uses the caller's transaction; does not commit or write payloads.
        Runtime storage options are never stored in the database.
        """
        if not postgres.storageLocations:
            raise ValueError("No payload destination is configured in postgres.storageLocations.")

        configuredRoot = postgres.storageLocations[0]

        if not isinstance(configuredRoot, str) or not configuredRoot.strip() or any(character in configuredRoot for character in ("\x00", "\n", "\r")):
            raise ValueError("The configured payload destination must be a nonempty path or URL without NUL or newline characters.")

        # Coordinate registration for this configured destination.
        lockName = f"pgl.storage_backend:{postgres.databaseSchema}:{configuredRoot}"
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lockName,))

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

        filesystem, _ = url_to_fs(configuredRoot, **(storageOptions or {}))
        protocols = filesystem.protocol
        protocols = {protocols} if isinstance(protocols, str) else set(protocols)

        if protocols.intersection({"file", "local"}):
            kind = "local"
        elif protocols.intersection({"ssh", "sftp", "smb", "nfs"}):
            kind = "nas"
        elif protocols.intersection({"s3", "gs", "gcs", "az", "abfs", "abfss", "adl"}):
            kind = "cloud"
        else:
            raise ValueError("Cannot automatically classify the configured storage protocol. Register the backend explicitly and supply backendID.")

        rootHash = hashlib.sha256(configuredRoot.encode("utf-8")).hexdigest()
        backendName = f"configured_{rootHash}"

        return cls.getOrCreateStorageBackend(connection, postgres, backendName, kind, configuredRoot)
    
    class _UnknownLifecycleSession(ValueError):
        """Requested lifecycle session does not exist."""

    class _SessionLifecycleBlocked(ValueError):
        """Lifecycle operation was refused without making changes."""

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

    @classmethod
    def _getSessionDeletionRecords(cls, connection, postgres, sessionID):
        """Return target and all descendants, including deleted records.

        Uses two queries. Call within a consistent preview snapshot or
        after acquiring lifecycle locks. Does not write or commit.
        """
        from psycopg.rows import dict_row

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
        return sorted([target, *descendants], key=lambda record: record["session_id"])

    @staticmethod
    def _confirmSessionDeletion(sessionID, sessions, *, databaseLabel):
        """Print every proposed member and require an explicit 'yes'.

        No database access or payload I/O. Call without database locks.
        Cancellation or unavailable input never authorizes deletion.
        """
        records = sorted(sessions, key=lambda record: record["session_id"])

        if not records:
            return False

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

    @classmethod
    def _registerSessionBundleFiles(cls, connection, postgres, sessionID, backendID, manifest, *, objectPrefix, storageOptions=None, origin="derived"):
        """Register a raw/derived bundle and mirror its verified session labels.

        Behavioral file subjects come from archived experiment settings.
        MNE-only file subjects come from hashed history metadata.
        Missing subjects are represented as SQL NULL.

        Uses the caller's transaction; does not publish the checkpoint.
        """
        for label, value in (("sessionID", sessionID), ("backendID", backendID)):
            if type(value) is not int or not 0 <= value <= 9223372036854775807:
                raise ValueError(f"{label} must be a nonnegative PostgreSQL bigint.")

        if not isinstance(objectPrefix, str) or objectPrefix.startswith("/") or "\\" in objectPrefix or "\x00" in objectPrefix or any(part in {"", ".", ".."} for part in objectPrefix.split("/")):
            raise ValueError("objectPrefix must be a valid relative POSIX path.")

        cls._validateSessionBundleManifest(manifest, origin=origin)
        payloadRole = "acquisition" if origin == "raw" else "derived"
        manifestBytes = cls.encodeRawManifest(manifest)
        manifestIdentity = {"sha256": hashlib.sha256(manifestBytes).hexdigest(), "size_bytes": len(manifestBytes)}
        hasMNE = manifest["contract_version"] in ("pgl.raw.session.v1", "pgl.derived.session.v1")

        sessionsTable = sql.Identifier(postgres.databaseSchema, "session")
        linksTable = sql.Identifier(postgres.databaseSchema, "session_files")
        backendsTable = sql.Identifier(postgres.databaseSchema, "storage_backends")

        with connection.transaction():
            checkpoint = connection.execute(sql.SQL("SELECT origin, status, contract_version, deleted_at FROM {} WHERE session_id = %s FOR UPDATE").format(sessionsTable), (sessionID,)).fetchone()

            if checkpoint != (origin, "in_progress", manifest["contract_version"], None):
                raise ValueError("Checkpoint must be active, in progress, and match the manifest origin and contract.")

            if connection.execute(sql.SQL("SELECT 1 FROM {} WHERE session_id = %s LIMIT 1").format(linksTable), (sessionID,)).fetchone() is not None:
                raise ValueError("Checkpoint already has file links.")

            backend = connection.execute(sql.SQL("SELECT url_prefix, is_active FROM {} WHERE backend_id = %s FOR SHARE").format(backendsTable), (backendID,)).fetchone()

            if backend is None or not backend[1]:
                raise ValueError("Storage backend is missing or inactive.")

            filesystem, root = url_to_fs(backend[0], **(storageOptions or {}))
            manifestKey = posixpath.join(objectPrefix, "manifest.json")

            if cls.hashFile(filesystem, posixpath.join(root, manifestKey)) != manifestIdentity:
                raise ValueError("Saved manifest does not match the supplied manifest.")

            fileLocations = {record["name"]: {"sha256": record["sha256"], "size_bytes": record["size_bytes"], "locations": [{"backend_id": backendID, "url_prefix": backend[0], "object_key": posixpath.join(objectPrefix, record["name"])}]} for record in manifest["files"]}
            archive = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend={backendID: storageOptions or {}})

            if hasMNE:
                cls._verifyMNEInventory(manifest, archive)

            _, historyMetadata = cls._readSessionHistory(archive, len(manifest["runs"]))

            if manifest["runs"]:
                metadata = cls._behaviorLabelMetadata(manifest, archive)

                if historyMetadata is not None and historyMetadata != metadata:
                    raise ValueError("History labels differ from archived behavioral experiment settings.")
            else:
                metadata = historyMetadata or {"version": 1, "labels": [{"runIndex": None, "experimentName": None, "subjectID": None}]}

            cls._recordSessionLabelMetadata(connection, postgres, sessionID, metadata)
            subjectsByRun = {run["path"]: metadata["labels"][run["index"]]["subjectID"] for run in manifest["runs"]}
            mneSubject = metadata["labels"][0]["subjectID"] if not manifest["runs"] else None

            def registerPayload(name, identity, role, subjectID=None):
                fileID = cls.getOrCreateFile(connection, postgres, identity["sha256"], identity["size_bytes"])
                cls.verifyAndRegisterStorageLocation(connection, postgres, fileID, backendID, posixpath.join(objectPrefix, name), storageOptions=storageOptions)
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

    @classmethod
    def _completeSessionBundleCheckpoint(cls, connection, postgres, sessionID, manifest, *, origin="derived"):
        """Validate registered metadata and complete a raw or derived checkpoint.

        Raw checkpoints must be parentless and their own root.
        Derived checkpoints require a completed active parent with the same root.

        Checks exact registered file identities/roles and eligible storage
        locations. Uses recorded verification metadata rather than rereading
        payload bytes.

        Opens a transaction/savepoint. An outer transaction must commit for
        publication to persist.
        """
        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        cls._validateSessionBundleManifest(manifest, origin=origin)
        payloadRole = "acquisition" if origin == "raw" else "derived"
        manifestBytes = cls.encodeRawManifest(manifest)
        manifestHash = hashlib.sha256(manifestBytes).hexdigest()

        expected = {record["name"]: (record["sha256"], record["size_bytes"], "history" if record["name"] == "history.json" else payloadRole) for record in manifest["files"]}
        expected["manifest.json"] = (manifestHash, len(manifestBytes), "manifest")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        with connection.transaction():
            statement = sql.SQL("SELECT origin, status, contract_version, deleted_at, parent_session_id, root_session_id FROM {} WHERE session_id = %s FOR UPDATE").format(sessions)
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

            statement = sql.SQL("SELECT link.name, content.sha256, content.size_bytes, link.role, (link.deleted_at IS NULL AND content.deleted_at IS NULL AND EXISTS (SELECT 1 FROM {} AS location JOIN {} AS backend ON backend.backend_id = location.backend_id WHERE location.file_id = content.file_id AND location.deleted_at IS NULL AND location.status = 'present' AND location.verified_at IS NOT NULL AND backend.is_active)) AS available FROM {} AS link JOIN {} AS content ON content.file_id = link.file_id WHERE link.session_id = %s ORDER BY link.name").format(locations, backends, links, files)
            rows = connection.execute(statement, (sessionID,)).fetchall()
            actual = {name: (digest, size, role) for name, digest, size, role, available in rows}

            if actual != expected:
                raise ValueError("Checkpoint file links do not exactly match the manifest identities and roles.")

            unavailable = [name for name, digest, size, role, available in rows if not available]

            if unavailable:
                raise ValueError("Files lack active links, content identities, or verified active storage locations: " + ", ".join(unavailable))

            statement = sql.SQL("UPDATE {} SET status = 'complete', manifest_hash = %s, completed_at = now() WHERE session_id = %s").format(sessions)
            connection.execute(statement, (manifestHash, sessionID))

        return manifestHash

    @classmethod
    def _reconstructSessionBundle(cls, manifest, fileLocations, storageOptionsByBackend=None, *, cacheDirectory=None, directLocalPolicy="materialize", cacheContext=None, origin="derived"):
        """Restore verified history, labels, and deferred behavioral/MNE fields.

        Older histories without label metadata remain supported.
        Behavioral labels derive from run experiment settings on access.
        MNE-only labels restore from the hashed history payload.
        """
        from .pglSession import pglSession, pglMNE

        if directLocalPolicy not in ("materialize", "immutable"):
            raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'")

        cls._validateSessionBundleManifest(manifest, origin=origin)
        hasMNE = manifest["contract_version"] in ("pgl.raw.session.v1", "pgl.derived.session.v1")

        if hasMNE and (cacheDirectory is None or not str(cacheDirectory).strip()):
            raise ValueError("MNE restoration requires an explicit cacheDirectory")

        filesystem = pglCheckpointFilesystem(manifest, fileLocations, storageOptionsByBackend=storageOptionsByBackend)
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

        return session

    @classmethod
    def _loadSessionBundle(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False, maxManifestBytes=16 * 1024 * 1024, cacheDirectory=None, directLocalPolicy="materialize", origin="derived"):
        """Load a completed raw or derived session bundle.

        Supports raw pgl.raw.session.v1 and both derived contracts.
        Uses the caller's connection for metadata lookup.
        REPEATABLE READ provides a consistent metadata snapshot.

        Verifies manifest, history, and MNE descriptor immediately.
        Behavioral and analysis payloads remain deferred.
        No SQL connection is retained by the restored session.
        """
        if origin not in ("raw", "derived"):
            raise ValueError("Bundle origin must be raw or derived")

        if type(sessionID) is not int or not 0 <= sessionID <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        if type(includeDeleted) is not bool:
            raise ValueError("includeDeleted must be a boolean.")

        if type(maxManifestBytes) is not int or maxManifestBytes <= 0:
            raise ValueError("maxManifestBytes must be a positive integer.")

        if directLocalPolicy not in ("materialize", "immutable"):
            raise ValueError("directLocalPolicy must be 'materialize' or 'immutable'")

        sessions = sql.Identifier(postgres.databaseSchema, "session")
        links = sql.Identifier(postgres.databaseSchema, "session_files")
        files = sql.Identifier(postgres.databaseSchema, "files")
        locations = sql.Identifier(postgres.databaseSchema, "storage_locations")
        backends = sql.Identifier(postgres.databaseSchema, "storage_backends")

        checkpoint = connection.execute(sql.SQL("SELECT origin, status, contract_version, manifest_hash FROM {} WHERE session_id = %s AND (%s OR deleted_at IS NULL)").format(sessions), (sessionID, includeDeleted)).fetchone()

        if checkpoint is None:
            raise ValueError(f"Unknown or deleted checkpoint: {sessionID}")

        checkpointOrigin, status, contractVersion, manifestHash = checkpoint

        if status != "complete":
            raise ValueError("Only completed checkpoints can be loaded.")

        supportedContracts = {"raw": ("pgl.raw.session.v1",), "derived": ("pgl.derived.behavior.v1", "pgl.derived.session.v1")}

        if checkpointOrigin != origin or contractVersion not in supportedContracts[origin]:
            raise ValueError("Unsupported checkpoint origin/contract.")

        hasMNE = contractVersion in ("pgl.raw.session.v1", "pgl.derived.session.v1")
        payloadRole = "acquisition" if origin == "raw" else "derived"

        if hasMNE and (cacheDirectory is None or not str(cacheDirectory).strip()):
            raise ValueError("MNE restoration requires an explicit cacheDirectory.")

        statement = sql.SQL("SELECT link.name, link.role, content.file_id, content.sha256, content.size_bytes, link.deleted_at, content.deleted_at FROM {} AS link JOIN {} AS content ON content.file_id = link.file_id WHERE link.session_id = %s ORDER BY link.name").format(links, files)
        rows = connection.execute(statement, (sessionID,)).fetchall()
        records = {}

        for name, role, fileID, digest, size, linkDeleted, contentDeleted in rows:
            if not includeDeleted and (linkDeleted is not None or contentDeleted is not None):
                raise ValueError(f"Checkpoint contains a deleted file link or content: {name}")

            records[name] = {"role": role, "file_id": fileID, "sha256": digest, "size_bytes": size}

        manifestRecord = records.get("manifest.json")

        if manifestRecord is None or manifestRecord["role"] != "manifest" or manifestRecord["sha256"] != manifestHash:
            raise ValueError("Checkpoint manifest link is missing or inconsistent.")

        if manifestRecord["size_bytes"] > maxManifestBytes:
            raise ValueError("Manifest exceeds the configured size limit.")

        fileIDs = sorted({record["file_id"] for record in records.values()})
        statement = sql.SQL("SELECT location.file_id, backend.backend_id, backend.url_prefix, location.object_key FROM {} AS location JOIN {} AS backend ON backend.backend_id = location.backend_id WHERE location.file_id = ANY(%s::bigint[]) AND (%s OR location.deleted_at IS NULL) AND location.status = 'present' AND location.verified_at IS NOT NULL AND backend.is_active ORDER BY location.file_id, location.priority, backend.backend_id, location.object_key").format(locations, backends)
        locationRows = connection.execute(statement, (fileIDs, includeDeleted)).fetchall()
        locationsByFile = {}

        for fileID, backendID, root, key in locationRows:
            if not isinstance(key, str) or key.startswith("/") or "\\" in key or "\x00" in key or any(part in {"", ".", ".."} for part in key.split("/")):
                raise ValueError("Database contains an invalid payload object key.")

            locationsByFile.setdefault(fileID, []).append({"backend_id": backendID, "url_prefix": root, "object_key": key})

        manifestLocations = locationsByFile.get(manifestRecord["file_id"], [])

        if not manifestLocations:
            raise FileNotFoundError("Manifest has no eligible verified storage location.")

        optionsByBackend = storageOptionsByBackend or {}
        failures = []
        manifestContent = None

        for location in manifestLocations:
            backendID = location["backend_id"]

            try:
                filesystem, root = url_to_fs(location["url_prefix"], **optionsByBackend.get(backendID, {}))
                path = posixpath.join(root, location["object_key"])

                with filesystem.open(path, "rb") as source:
                    content = source.read(manifestRecord["size_bytes"] + 1)

                if len(content) != manifestRecord["size_bytes"]:
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

        expected = {record["name"]: (record["sha256"], record["size_bytes"], "history" if record["name"] == "history.json" else payloadRole) for record in manifest["files"]}
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

            fileLocations[name] = {"sha256": record["sha256"], "size_bytes": record["size_bytes"], "locations": candidates}

        cacheContext = cls._mneCacheContext(postgres, sessionID) if hasMNE else None
        return cls._reconstructSessionBundle(manifest, fileLocations, storageOptionsByBackend=optionsByBackend, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, cacheContext=cacheContext, origin=origin)
    
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

    @classmethod
    def _validateSessionBundleManifest(cls, manifest, *, origin):
        if origin == "raw":
            return cls.validateRawSessionManifest(manifest)
        if origin == "derived":
            return cls.validateDerivedManifest(manifest)
        raise ValueError(f"Unsupported bundle origin: {origin!r}")

    # Compatibility wrappers: existing callers retain their interfaces.

    @classmethod
    def registerDerivedSessionFiles(cls, connection, postgres, sessionID, backendID, manifest, *, objectPrefix, storageOptions=None):
        """Register a derived bundle using the shared registration implementation."""
        return cls._registerSessionBundleFiles(connection, postgres, sessionID, backendID, manifest, objectPrefix=objectPrefix, storageOptions=storageOptions, origin="derived")

    @classmethod
    def completeDerivedCheckpoint(cls, connection, postgres, sessionID, manifest):
        """Complete a derived checkpoint using the shared validation implementation."""
        return cls._completeSessionBundleCheckpoint(connection, postgres, sessionID, manifest, origin="derived")

    @classmethod
    def reconstructDerivedSession(cls, manifest, fileLocations, storageOptionsByBackend=None, *, cacheDirectory=None, directLocalPolicy="materialize", cacheContext=None):
        """Restore a derived bundle with deferred behavioral/MNE fields."""
        return cls._reconstructSessionBundle(manifest, fileLocations, storageOptionsByBackend=storageOptionsByBackend, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, cacheContext=cacheContext, origin="derived")

    @classmethod
    def loadDerivedSession(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False, maxManifestBytes=16 * 1024 * 1024, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load a completed derived checkpoint using the caller's transaction."""
        return cls._loadSessionBundle(connection, postgres, sessionID, storageOptionsByBackend=storageOptionsByBackend, includeDeleted=includeDeleted, maxManifestBytes=maxManifestBytes, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, origin="derived")

    @classmethod
    def loadRawSessionBundle(cls, connection, postgres, sessionID, storageOptionsByBackend=None, *, includeDeleted=False, maxManifestBytes=16 * 1024 * 1024, cacheDirectory=None, directLocalPolicy="materialize"):
        """Load an MNE-containing raw root using the caller's transaction."""
        return cls._loadSessionBundle(connection, postgres, sessionID, storageOptionsByBackend=storageOptionsByBackend, includeDeleted=includeDeleted, maxManifestBytes=maxManifestBytes, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy, origin="raw")

    # ------------------------------------------------------------
    # Public schema initialization
    # ------------------------------------------------------------
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
        from .pglSettings import pglSettingsManager

        if postgres is not None and (settings is not None or settingsName is not None):
            raise ValueError("Supply either postgres or PGL settings, not both.")

        if postgres is None:
            resolvedSettings = pglSettingsManager.getSettings(settings=settings, settingsName=settingsName)

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
        """Grant application table access and read-only reporting-view access."""
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
            "session_deletion_groups",
            "session_deletion_group_members",
        ]

        for name in applicationTables:
            table = sql.Identifier(postgres.databaseSchema, name)
            connection.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {} TO {}").format(table, application))

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

class pglFigureRowExport:
    """Collect rendered figure rows and export one vertically stacked PNG.

    Row figures are rendered immediately to temporary files; figure handles
    are not retained. The completed image is written through fsspec.

    Export errors warn and disable this collector without failing analysis.
    Only finish() publishes output. close() discards unfinished output.

    maxPixels limits the final image area, not total process memory.
    """

    def __init__(self, filename, *, filesystem=None, storageOptions=None, dpi=200, overwrite=False, maxPixels=40_000_000):
        """Prepare an export destination without creating its output file."""
        import os
        from tempfile import TemporaryDirectory
        from fsspec import AbstractFileSystem

        self._temporary = None
        self._rows = []
        self._width = 0
        self._height = 0
        self._filesystem = None
        self._destination = None
        self._result = None
        self._closed = False
        self._finished = False
        self._disabled = filename is None
        self._dpi = dpi
        self._overwrite = overwrite
        self._maxPixels = maxPixels
        self._exportRequested = filename is not None

        # None means no optional export, not browser display.
        if self._disabled:
            return

        try:
            if type(dpi) is not int or dpi <= 0:
                raise ValueError("dpi must be a positive integer.")

            if type(overwrite) is not bool:
                raise ValueError("overwrite must be a boolean.")

            if type(maxPixels) is not int or maxPixels <= 0:
                raise ValueError("maxPixels must be a positive integer.")

            if filesystem is not None and storageOptions:
                raise ValueError("Supply filesystem or storageOptions, not both.")

            if filesystem is not None and not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec AbstractFileSystem.")

            location = os.fspath(filename)

            if not isinstance(location, str) or not location.strip():
                raise ValueError("filename must be a nonempty path or URL.")

            if any(character in location for character in ("\x00", "\n", "\r")):
                raise ValueError("filename must not contain NUL or newline characters.")

            if location.endswith(("/", "\\")):
                raise ValueError("Supply a filename, not only a directory.")

            if filesystem is None:
                # Expand client-local ~ only. Remote paths use their backend's
                # conventions rather than the client's home directory.
                if "://" not in location and "::" not in location:
                    location = str(Path(location).expanduser())

                filesystem, destination = url_to_fs(location, **(storageOptions or {}))
            else:
                destination = filesystem._strip_protocol(location)

            protocols = filesystem.protocol
            protocols = {protocols} if isinstance(protocols, str) else set(protocols)

            if protocols.intersection({"file", "local"}):
                destination = str(Path(destination).expanduser().resolve())

            if not destination or destination.endswith("/") or filesystem.isdir(destination):
                raise ValueError("Supply a figure filename rather than a directory.")

            stem, extension = posixpath.splitext(destination)
            destination = stem + ".png"

            if extension and extension.lower() != ".png":
                pglMessages.warning("Combined figure export uses PNG; the filename extension was changed to .png.", level=0)

            if filesystem.exists(destination):
                if filesystem.isdir(destination):
                    raise IsADirectoryError("The normalized destination is a directory.")

                if not overwrite:
                    raise FileExistsError("The output already exists.")

            self._filesystem = filesystem
            self._destination = destination
            self._temporary = TemporaryDirectory(prefix="pgl-figure-rows-")

        except Exception as error:
            self._fail("prepare figure export", error)

    def _fail(self, operation, error):
        """Disable optional export and release temporary files."""
        self._disabled = True

        if isinstance(error, FileExistsError):
            pglMessages.warning("Combined figure export skipped: the output already exists. Use overwrite=True to replace it.", level=2)
        elif isinstance(error, MemoryError):
            pglMessages.warning("Combined figure export stopped: insufficient memory. Reduce dpi or export fewer rows. Analysis and notebook display will continue.", level=2)
        else:
            # Backend exception messages can contain URLs or credentials.
            # Report the failure category without exposing those details.
            pglMessages.warning(f"Could not {operation} ({type(error).__name__}); combined export was disabled. Analysis and notebook display will continue.", level=2)

        self.close()

    def add(self, fig):
        """Render one completed row without retaining its Matplotlib figure.

        Returns True when collected, otherwise False.
        Does not display, clear, or close the supplied figure.
        """
        if self._disabled or self._closed or self._finished:
            # A None filename stays silent. A failed export warns for each
            # subsequent row so the failure remains visible during plotting.
            if self._exportRequested and not self._finished:
                pglMessages.warning("This figure row was NOT collected: combined PNG export was cancelled earlier. Notebook display and report generation are continuing, but no combined PNG will be saved.", level=1)

            return False
        try:
            import math
            from matplotlib.figure import Figure
            from PIL import Image

            if not isinstance(fig, Figure):
                raise TypeError("fig must be a Matplotlib Figure.")

            # Check approximate dimensions before allocating the render buffer.
            widthInches, heightInches = fig.get_size_inches()
            estimatedWidth = math.ceil(float(widthInches) * self._dpi)
            estimatedHeight = math.ceil(float(heightInches) * self._dpi)

            if estimatedWidth <= 0 or estimatedHeight <= 0:
                raise ValueError("Figure dimensions must be positive.")

            estimatedPixels = max(self._width, estimatedWidth) * (self._height + estimatedHeight)

            if estimatedPixels > self._maxPixels:
                pglMessages.warning(f"Combined figure export would exceed maxPixels={self._maxPixels:,}. Reduce dpi, reduce the number of rows, or explicitly increase maxPixels.", level=1)
                self._disabled = True
                self.close()
                return False

            rowPath = Path(self._temporary.name) / f"row{len(self._rows):06d}.png"

            # No tight bounding box: preserve each row's designed canvas size.
            # Explicitly override a possible savefig.bbox='tight' global setting.
            import matplotlib

            with matplotlib.rc_context({"savefig.bbox": None}):
                fig.savefig(rowPath, format="png", dpi=self._dpi, facecolor="white", transparent=False, bbox_inches=None)

            with Image.open(rowPath) as image:
                width, height = image.size

            combinedWidth = max(self._width, width)
            combinedHeight = self._height + height

            if combinedWidth * combinedHeight > self._maxPixels:
                pglMessages.warning(f"Rendered rows exceed maxPixels={self._maxPixels:,}; combined figure export was cancelled.", level=1)
                self._disabled = True
                self.close()
                return False

            self._rows.append((rowPath, width, height))
            self._width = combinedWidth
            self._height = combinedHeight
            return True

        except Exception as error:
            self._fail("render figure row", error)
            return False

    def finish(self):
        """Stack collected rows vertically and save one PNG.

        Narrower rows are centered on a white background without resizing.
        Returns the resolved destination on success, otherwise None.

        Repeated finish() after success returns the same result.
        A failed remote write may leave a partial output file.
        """
        if self._finished:
            return self._result

        if self._disabled or self._closed:
            return None

        try:
            from PIL import Image

            if not self._rows:
                pglMessages.warning("No figure rows were collected; no PNG was exported.", level=1)
                self.close()
                return None

            if self._width * self._height > self._maxPixels:
                raise ValueError("Combined image exceeds maxPixels.")

            # Recheck before assembling or writing. Exclusive creation below
            # provides another safeguard where supported by the backend.
            if self._filesystem.exists(self._destination):
                if self._filesystem.isdir(self._destination):
                    raise IsADirectoryError("Figure destination is a directory.")

                if not self._overwrite:
                    raise FileExistsError("Figure destination already exists.")

            pglMessages.message(f"Assembling {len(self._rows)} figure rows into one PNG ({self._width} × {self._height} pixels)...")
            combinedPath = Path(self._temporary.name) / "combined.png"
            canvas = Image.new("RGB", (self._width, self._height), color="white")

            try:
                top = 0

                for rowPath, width, height in self._rows:
                    with Image.open(rowPath) as source:
                        rowImage = source.convert("RGBA")

                        try:
                            left = (self._width - width) // 2
                            canvas.paste(rowImage, (left, top), rowImage)
                        finally:
                            rowImage.close()

                    top += height

                # Write the complete local PNG before opening remote output.
                canvas.save(combinedPath, format="PNG", dpi=(self._dpi, self._dpi))

            finally:
                canvas.close()

            parent = posixpath.dirname(self._destination)

            if parent:
                self._filesystem.makedirs(parent, exist_ok=True)

            mode = "wb" if self._overwrite else "xb"

            with combinedPath.open("rb") as source:
                with self._filesystem.open(self._destination, mode) as target:
                    while True:
                        chunk = source.read(1024 * 1024)

                        if not chunk:
                            break

                        if target.write(chunk) != len(chunk):
                            raise OSError("Incomplete combined figure write.")

            self._result = self._destination
            self._finished = True
            pglMessages.message(f"Combined figure exported: {posixpath.basename(self._destination)}")
            return self._result

        except Exception as error:
            self._fail("save combined figure", error)
            return None

        finally:
            self.close()

    def close(self):
        """Remove temporary files without publishing unfinished output.

        Safe to call repeatedly, including after finish() or an export error.
        """
        self._closed = True
        temporary = self._temporary
        self._temporary = None
        self._rows = []

        if temporary is not None:
            try:
                temporary.cleanup()
            except Exception as error:
                pglMessages.warning(f"Could not remove temporary figure-export files ({type(error).__name__}).", level=1)

    def __enter__(self):
        """Allow deterministic cleanup with a with block."""
        return self

    def __exit__(self, exceptionType, exception, traceback):
        """Clean up without suppressing scientific errors or auto-publishing."""
        self.close()
        return False

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

        if not isinstance(manifest, dict):
            raise ValueError("Manifest must contain a dictionary.")

        contractVersion = manifest.get("contract_version")

        if contractVersion == "pgl.raw.behavior.v1":
            pglStorage.validateRawManifest(manifest)
        elif contractVersion == "pgl.raw.session.v1":
            pglStorage.validateRawSessionManifest(manifest)
        elif contractVersion in ("pgl.derived.behavior.v1", "pgl.derived.session.v1"):
            pglStorage.validateDerivedManifest(manifest)
        else:
            raise ValueError(f"Unsupported checkpoint contract: {contractVersion!r}")

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
    
    def localBundlePaths(self, resourceRoot, records):
        """Return a layout-compatible local bundle, or None.

        Called only when resolving a field, never during reconstruction.

        Checks bundle identities against checkpoint identities, then selects
        registered local locations sharing the required relative layout.

        Does not open, hash, spool, download, or copy payloads.
        The caller must verify every returned file before opening MNE.
        """
        from pathlib import Path
        from fsspec.implementations.local import LocalFileSystem

        candidatesByName = {}

        for member in records:
            relativeName = member["name"]
            logicalName = self._logicalPath(
                posixpath.join(resourceRoot, relativeName)
            )

            registered = self._payloads.get(logicalName)
            if registered is None:
                raise ValueError(
                    f"FIF member is absent from checkpoint: {logicalName}"
                )

            expected = {
                key: member[key]
                for key in ("sha256", "size_bytes")
            }
            checkpointIdentity = {
                key: registered[key]
                for key in ("sha256", "size_bytes")
            }
            if checkpointIdentity != expected:
                raise ValueError(
                    f"FIF/checkpoint identity mismatch: {logicalName}"
                )

            # Map possible physical bundle roots to member paths.
            candidates = {}
            parts = relativeName.split("/")

            for location in registered["locations"]:
                options = self._storageOptionsByBackend.get(
                    location["backend_id"], {}
                )
                filesystem, root = url_to_fs(
                    location["url_prefix"], **options
                )

                # Do not infer locality from the checkpoint's protocol,
                # backend label, or a wrapper's apparent filename.
                if type(filesystem) is not LocalFileSystem:
                    continue

                source = Path(
                    filesystem._strip_protocol(
                        posixpath.join(root, location["object_key"])
                    )
                ).expanduser()

                # Leaf symlinks are conservatively ineligible.
                if source.is_symlink():
                    continue

                source = source.resolve()
                bundleRoot = source
                for _ in parts:
                    bundleRoot = bundleRoot.parent

                # Content-addressed names generally fail this comparison.
                if bundleRoot.joinpath(*parts) != source:
                    continue

                candidates[bundleRoot] = source

            candidatesByName[relativeName] = candidates

        if not candidatesByName:
            return None

        commonRoots = set.intersection(
            *(set(candidates) for candidates in candidatesByName.values())
        )

        # A bundle must be selected together, not one preferred replica
        # independently for each split member.
        for root in sorted(commonRoots, key=str):
            paths = {
                name: candidates[root]
                for name, candidates in candidatesByName.items()
            }
            if all(path.is_file() for path in paths.values()):
                return paths

        return None
    
    def validateBundleIdentities(self, resourceRoot, records):
        """Check inner bundle identities against checkpoint metadata without I/O."""
        for member in records:
            logicalName = self._logicalPath(posixpath.join(resourceRoot, member["name"]))
            registered = self._payloads.get(logicalName)
            if registered is None: raise ValueError(f"FIF member is absent from checkpoint: {logicalName}")

            expected = {key: member[key] for key in ("sha256", "size_bytes")}
            actual = {key: registered[key] for key in ("sha256", "size_bytes")}
            if actual != expected: raise ValueError(f"FIF/checkpoint identity mismatch: {logicalName}")
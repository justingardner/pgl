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

class pglStorage:
    """Manage scientific storage, checkpoint metadata, and schema migrations."""

    # Stable lock key shared by all pgl schema-initialization processes.
    MIGRATION_LOCK = 734_291_806_115

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
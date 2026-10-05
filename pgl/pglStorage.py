################################################################
#   filename: pglStorage.py
#    purpose: Persistent storage for sessions and datasets
#             allowing for sharing, provenance tracking, pipelining
#         by: JLG
#       date: Oct 5, 2026
################################################################

from hashlib import sha256
from pathlib import Path
import re
import getpass
import os
import platform
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import shlex
import socket
import uuid

import psycopg
from psycopg import sql
from .pglMessages import pglMessages


class pglStorage:
    """Postgres-backed store. Initial implementation: schema management."""

    # Stable, application-specific advisory-lock key.
    _migrationLock = 734_291_806_115

    def __init__(self, connection, schema="pgl"):
        self.connection = connection
        self.schema = schema

    @classmethod
    def connect(cls, conninfo="", schema="pgl", **kwargs):
        """Connect and select the application's database schema."""
        if "autocommit" in kwargs:
            raise TypeError("pglStorage manages transactions; do not pass autocommit")

        connection = psycopg.connect(conninfo, autocommit=True, **kwargs)

        try:
            connection.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(schema)))
        except Exception:
            connection.close()
            raise

        return cls(connection, schema=schema)
    
    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def initDatabase(cls, *, dbname, adminUser, adminPassword, applicationRole, host="127.0.0.1", port=5432, schema="pgl", connect_timeout=10, migrationsDirectory=None):
        """Initialize the schema and report newly applied migrations."""
        with cls.connect(dbname=dbname, user=adminUser, password=adminPassword, host=host, port=port, schema=schema, connect_timeout=connect_timeout) as admin:
            applied = admin._initDatabase(applicationRole=applicationRole, migrationsDirectory=migrationsDirectory)

        print("Applied migrations:", ", ".join(applied) if applied else "None; schema is already current")
        
    def _initDatabase(self, applicationRole, migrationsDirectory=None):
        """Create the schema, apply migrations, and grant application access.

        Call using the administrator connection, not the application connection.
        Safe to rerun. Does not create roles or change passwords.
        """
        with self.connection.transaction():
            self.connection.execute("SELECT pg_advisory_xact_lock(%s)", (self._migrationLock,))

            adminRole = self.connection.execute("SELECT current_user").fetchone()[0]

            if applicationRole == adminRole:
                raise ValueError("Initialize using a separate administrator role")

            roleExists = self.connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (applicationRole,)).fetchone()

            if roleExists is None:
                raise ValueError(f"Application role does not exist: {applicationRole}")

            schemaOwner = self.connection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (self.schema,)).fetchone()

            if schemaOwner is None:
                self.connection.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(sql.Identifier(self.schema), sql.Identifier(adminRole)))
            elif schemaOwner[0] != adminRole:
                raise RuntimeError(f"Schema {self.schema!r} is owned by {schemaOwner[0]!r}, not {adminRole!r}; refusing to change ownership automatically")

            self.connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(self.schema)))
            self.connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}").format(sql.Identifier(self.schema), sql.Identifier(applicationRole)))

            applied = self.migrate(migrationsDirectory=migrationsDirectory)
            self._grantApplicationAccess(applicationRole)

        return applied


    def _grantApplicationAccess(self, applicationRole):
        """Grant pilot application privileges without migration-write access."""
        schema = sql.Identifier(self.schema)
        role = sql.Identifier(applicationRole)

        self.connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, role))

        applicationTables = ["users", "subjects", "storage_backends", "files", "storage_locations", "session", "session_files"]

        for table in applicationTables:
            self.connection.execute(sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {}.{} TO {}").format(schema, sql.Identifier(table), role))

        readOnlyTables = ["schema_migrations", "session_subjects", "raw_sessions"]

        for table in readOnlyTables:
            self.connection.execute(sql.SQL("REVOKE ALL PRIVILEGES ON TABLE {}.{} FROM {}").format(schema, sql.Identifier(table), role))
            self.connection.execute(sql.SQL("GRANT SELECT ON TABLE {}.{} TO {}").format(schema, sql.Identifier(table), role))

        self.connection.execute(sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(schema, role))
    def migrate(self, migrationsDirectory=None):
        """Apply numbered SQL migrations atomically, checking old checksums."""

        if migrationsDirectory is None:
            migrationsDirectory = Path(__file__).resolve().parent / "migrations"

        directory = Path(migrationsDirectory).expanduser().resolve()

        if not directory.is_dir():
            raise FileNotFoundError(f"Migration directory not found: {directory}")

        migrations = []

        for path in sorted(directory.glob("*.sql")):
            match = re.fullmatch(r"(\d{4})_([a-z0-9_]+)\.sql", path.name)

            if match is None:
                raise ValueError(f"Invalid migration filename: {path.name}")

            raw = path.read_bytes()
            migrations.append({
                "version": int(match.group(1)),
                "name": path.stem,
                "checksum": sha256(raw).hexdigest(),
                "sql": raw.decode("utf-8"),
            })

        if not migrations:
            raise ValueError(f"No SQL migrations found in {directory}")

        versions = [migration["version"] for migration in migrations]

        if versions != list(range(1, len(migrations) + 1)):
            raise ValueError("Migration numbers must be unique and consecutive, starting at 0001")

        newlyApplied = []

        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                # Protect initialization and migration from concurrent runners.
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (self._migrationLock,))
                cursor.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(sql.Identifier(self.schema)))

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS public.schema_migrations (
                        version integer PRIMARY KEY CHECK (version > 0),
                        name text NOT NULL UNIQUE,
                        checksum char(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
                        applied_at timestamptz NOT NULL DEFAULT now(),
                        applied_by text NOT NULL DEFAULT current_user
                    )
                """)

                cursor.execute("SELECT version, name, checksum FROM public.schema_migrations ORDER BY version")
                appliedRows = cursor.fetchall()
                applied = {version: (name, checksum) for version, name, checksum in appliedRows}

                if [row[0] for row in appliedRows] != versions[:len(appliedRows)]:
                    raise RuntimeError("Database migration history is not a prefix of the available migrations")

                for migration in migrations:
                    version = migration["version"]

                    if version in applied:
                        expected = (migration["name"], migration["checksum"])

                        if applied[version] != expected:
                            raise RuntimeError(f"Applied migration was changed: {migration['name']}; restore it and add a new migration")

                        continue

                    # SQL files are trusted repository code, not user input.
                    cursor.execute(migration["sql"], prepare=False)
                    cursor.execute("INSERT INTO public.schema_migrations (version, name, checksum) VALUES (%s, %s, %s)", (version, migration["name"], migration["checksum"]))
                    newlyApplied.append(migration["name"])

        return newlyApplied
    
    @staticmethod
    def validateConfiguration(*, postgresVersion="17", postgresPort=5432, databaseName="pgl", databaseSchema="pgl", databaseUser="pgl_app", adminUser="pgl_admin", installDirectory="~/.local/share/pgl/postgres", installPostgresIfMissing=True, resetApplicationPassword=False):
        """Validate local macOS setup options and return resolved configuration.

        Does not install software, create directories, start Postgres,
        connect to a database, or request passwords.
        """
        POSTGRES_HOST = "127.0.0.1"
        STARTUP_TIMEOUT_SECONDS = 60
        CONNECT_TIMEOUT_SECONDS = 10

        if platform.system() != "Darwin":
            raise RuntimeError("Local Postgres installation currently supports macOS only.")

        if os.geteuid() == 0:
            raise RuntimeError("Run setup as your normal macOS user, not as root.")

        if isinstance(postgresPort, bool) or not isinstance(postgresPort, int) or not 1 <= postgresPort <= 65535:
            raise ValueError("postgresPort must be an integer between 1 and 65535.")

        if not isinstance(postgresVersion, str) or not re.fullmatch(r"[0-9]+", postgresVersion):
            raise ValueError("postgresVersion must be a major version string such as '17'.")

        names = {
            "databaseName": databaseName,
            "databaseSchema": databaseSchema,
            "databaseUser": databaseUser,
            "adminUser": adminUser,
        }

        for label, name in names.items():
            if not isinstance(name, str) or not name.strip() or "\x00" in name or len(name.encode("utf-8")) > 63:
                raise ValueError(f"{label} must contain 1–63 UTF-8 bytes, must not be blank, and must not contain NUL.")

        if databaseUser == adminUser:
            raise ValueError("Use different administrator and application accounts.")

        if databaseUser.startswith("pg_") or adminUser.startswith("pg_"):
            raise ValueError("Postgres reserves role names beginning with 'pg_'.")

        if databaseSchema.startswith("pg_") or databaseSchema in {"public", "information_schema"}:
            raise ValueError("Choose a dedicated application schema.")

        if databaseName in {"postgres", "template0", "template1"}:
            raise ValueError("Choose a dedicated application database.")

        if not isinstance(installPostgresIfMissing, bool):
            raise ValueError("installPostgresIfMissing must be True or False.")

        if not isinstance(resetApplicationPassword, bool):
            raise ValueError("resetApplicationPassword must be True or False.")

        if not isinstance(installDirectory, (str, Path)) or not str(installDirectory).strip():
            raise ValueError("installDirectory must be a nonempty path.")

        rootDirectory = Path(installDirectory).expanduser().resolve()

        if rootDirectory.exists() and not rootDirectory.is_dir():
            raise ValueError(f"Installation path is not a directory: {rootDirectory}")

        if rootDirectory in {Path("/"), Path.home().resolve()}:
            raise ValueError("Choose a dedicated installation subdirectory.")

        brewCandidates = [shutil.which("brew"), "/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
        brew = next((Path(path) for path in brewCandidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)

        if brew is None:
            raise RuntimeError("Homebrew was not found. Install Homebrew before running local Postgres setup.")

        configuration = SimpleNamespace(
            postgresVersion=postgresVersion,
            postgresHost=POSTGRES_HOST,
            postgresPort=postgresPort,
            databaseName=databaseName,
            databaseSchema=databaseSchema,
            databaseUser=databaseUser,
            adminUser=adminUser,
            installDirectory=rootDirectory,
            dataDirectory=rootDirectory / "data",
            logFile=rootDirectory / "postgres.log",
            installPostgresIfMissing=installPostgresIfMissing,
            resetApplicationPassword=resetApplicationPassword,
            startupTimeoutSeconds=STARTUP_TIMEOUT_SECONDS,
            connectTimeoutSeconds=CONNECT_TIMEOUT_SECONDS,
            brew=brew,
            formula=f"postgresql@{postgresVersion}",
        )
        print("Postgres configuration")
        print("=" * 72)

        for name, value in vars(configuration).items():
            print(f"{name:<28} {value}")

        return configuration

    @staticmethod
    def _askPassword(prompt, *, confirm=False):
        """Always request a password privately; never read it from the environment."""
        pglMessages.message("Enter password", emphasize=True)
        password = getpass.getpass(prompt)

        if not password or any(character in password for character in ("\n", "\r", "\x00")):
            raise ValueError("Passwords must be nonempty and contain no newline or NUL characters.")

        if confirm and password != getpass.getpass("Confirm password: "):
            raise ValueError("Passwords do not match.")

        return password
    
    @staticmethod
    def _runCommand(arguments, configuration, *, capture=False, check=True):
        """Run an external command without a shell."""
        commandEnvironment = os.environ.copy()
        commandEnvironment["PATH"] = f"{configuration.brew.parent}{os.pathsep}{commandEnvironment.get('PATH', '')}"

        return subprocess.run([str(argument) for argument in arguments], check=check, text=True, stdout=subprocess.PIPE if capture else None, env=commandEnvironment)    
    
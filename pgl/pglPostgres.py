################################################################
#   filename: pglPostgres.py
#    purpose: Local PostgreSQL configuration and server management
#         by: JLG
#       date: Oct 5, 2026
################################################################

import getpass
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from .pglMessages import pglMessages

import psycopg
from psycopg import sql
from traitlets import Bool, Int, Unicode

from .pglSettings import pglTraitSettings


class pglPostgres(pglTraitSettings):
    """Saved PostgreSQL configuration and explicit server-management methods.

    Loading configuration does not start Postgres or connect to a database.
    Passwords, connections, executable paths, and live status are not saved.

    Server setup currently supports local macOS/Homebrew installations.
    """

    postgresVersion = Unicode("17", postgresConfiguration=True, help="PostgreSQL major version to install and use.")
    postgresHost = Unicode("127.0.0.1", postgresConfiguration=True, help="Server address; local Postgres management requires 127.0.0.1.")
    postgresPort = Int(5432, min=1, max=65535, postgresConfiguration=True, help="TCP port on which the Postgres server listens.")

    databaseName = Unicode("pgl", postgresConfiguration=True, help="Database containing the session data store.")
    databaseSchema = Unicode("pgl", postgresConfiguration=True, help="Schema containing the session data store tables.")
    databaseUser = Unicode("pgl_app", postgresConfiguration=True, help="Application role used for ordinary database reads and writes.")
    adminUser = Unicode("pgl_admin", postgresConfiguration=True, help="Administrator role used to initialize and manage the database.")

    installDirectory = Unicode("~/.local/share/pgl/postgres", postgresConfiguration=True, help="Root directory for the saved configuration, server log, and Postgres data directory.")
    installPostgresIfMissing = Bool(True, postgresConfiguration=True, help="Allow installation through Homebrew if the configured PostgreSQL version is missing.")

    # Constants: not serialized.
    CONFIGURATION_VERSION = 1
    CONFIGURATION_FILENAME = "configuration.json"
    STARTUP_TIMEOUT_SECONDS = 60
    CONNECT_TIMEOUT_SECONDS = 10

    # ------------------------------------------------------------
    # Derived paths and values
    # ------------------------------------------------------------

    @property
    def rootDirectory(self):
        return Path(self.installDirectory).expanduser().resolve()

    @property
    def dataDirectory(self):
        return self.rootDirectory / "data"

    @property
    def logFile(self):
        return self.rootDirectory / "postgres.log"

    @property
    def configurationFile(self):
        return self.rootDirectory / self.CONFIGURATION_FILENAME

    @property
    def formula(self):
        return f"postgresql@{self.postgresVersion}"

    @property
    def startupTimeoutSeconds(self):
        return self.STARTUP_TIMEOUT_SECONDS

    @property
    def connectTimeoutSeconds(self):
        return self.CONNECT_TIMEOUT_SECONDS

    @property
    def brew(self):
        return self._findBrew()

    @property
    def binaryDirectory(self):
        result = self._runCommand([self.brew, "--prefix"], capture=True)
        return Path(result.stdout.strip()) / "opt" / self.formula / "bin"

    @property
    def initdb(self):
        return self.binaryDirectory / "initdb"

    @property
    def pgCtl(self):
        return self.binaryDirectory / "pg_ctl"

    # ------------------------------------------------------------
    # Configuration serialization
    # ------------------------------------------------------------

    @classmethod
    def _configurationNames(cls):
        return sorted(name for name, trait in cls.class_traits().items() if trait.metadata.get("postgresConfiguration", False))

    def _configurationValues(self):
        return {name: getattr(self, name) for name in self._configurationNames()}

    def toJSONdict(self, type="all"):
        """Serialize only explicitly designated configuration fields."""
        return {"configurationVersion": self.CONFIGURATION_VERSION, **self._configurationValues()}

    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore configuration without starting or connecting to Postgres."""
        values = dict(data)
        version = values.pop("configurationVersion", None)

        if version != cls.CONFIGURATION_VERSION:
            raise ValueError(f"Unsupported Postgres configuration version: {version!r}")

        expected = set(cls._configurationNames())
        supplied = set(values)

        if supplied != expected:
            raise ValueError(f"Invalid configuration fields; missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}")

        obj = cls(**values)
        obj._validateValues()
        return obj

    def save(self, filename=None, *, overwrite=False):
        """Atomically save configuration with owner-only permissions.

        Identical existing configuration is left unchanged.
        Changed configuration requires overwrite=True.
        No passwords or runtime state are saved.
        """
        self._validateValues()
        target = self.configurationFile if filename is None else Path(filename).expanduser().resolve()

        if target == self.dataDirectory or self.dataDirectory in target.parents:
            raise ValueError("Do not store configuration inside Postgres's internal data directory.")

        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if target.exists() and not overwrite:
            existing = self.load(target)

            if existing.toJSONdict() != self.toJSONdict():
                raise RuntimeError(f"Different configuration already exists at {target}; use overwrite=True only for an intentional change.")

            pglMessages.message(f"Configuration already saved: {target}")
            return

        encoded = self.toJSON()
        temporaryPath = None

        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, prefix=".pgl-postgres-", delete=False) as temporary:
                temporaryPath = Path(temporary.name)
                os.chmod(temporaryPath, 0o600)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())

            if overwrite:
                os.replace(temporaryPath, target)
            else:
                # Publish without overwriting a concurrently created file.
                os.link(temporaryPath, target)

        finally:
            if temporaryPath is not None:
                temporaryPath.unlink(missing_ok=True)

        pglMessages.message(f"Configuration saved: {target}")

    @classmethod
    def load(cls, filename=None):
        """Load configuration only; never start or connect to Postgres."""
        if filename is None:
            defaultDirectory = cls.class_traits()["installDirectory"].default_value
            target = Path(defaultDirectory) / cls.CONFIGURATION_FILENAME
        else:
            target = Path(filename)

        target = target.expanduser().resolve()
        data = json.loads(target.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError("Postgres configuration must be a JSON object.")

        if data.pop("__class__", None) != cls.__name__:
            raise ValueError(f"Not a {cls.__name__} configuration: {target}")

        return cls.fromJSONdict(data, filename=str(target))

    @classmethod
    def fromSettings(cls, settings=None, settingsName=None):
        """Load configured Postgres settings, or warn and return None if not configured."""
        from .pglSettings import pglSettingsManager

        resolvedSettings = pglSettingsManager.getSettings(settings=settings, settingsName=settingsName)

        if resolvedSettings is None:
            pglMessages.warning("Could not resolve pgl settings.")
            return None

        if not resolvedSettings.databasePath.strip():
            pglMessages.warning("No databasePath is configured in pgl settings.")
            return None

        directory = Path(resolvedSettings.databasePath).expanduser().resolve()
        filename = directory / cls.CONFIGURATION_FILENAME

        if not filename.is_file():
            pglMessages.warning(f"Postgres has not been configured at {directory}. Create a pglPostgres instance with installDirectory pointing there, then call install().")
            return None

        return cls.load(filename)

    # ------------------------------------------------------------
    # Configuration and environment validation
    # ------------------------------------------------------------

    def _validateValues(self):
        """Validate settings without requiring Homebrew or a running server."""
        if self.postgresHost != "127.0.0.1":
            raise ValueError("Local Postgres management only supports 127.0.0.1.")

        if not re.fullmatch(r"[0-9]+", self.postgresVersion):
            raise ValueError("postgresVersion must be a major version string such as '17'.")

        if int(self.postgresVersion) < 10:
            raise ValueError("This installer supports PostgreSQL major versions 10 and later.")

        for name in ("databaseName", "databaseSchema", "databaseUser", "adminUser"):
            value = getattr(self, name)

            if not value.strip() or "\x00" in value or len(value.encode("utf-8")) > 63:
                raise ValueError(f"{name} must contain 1–63 UTF-8 bytes, must not be blank, and must not contain NUL.")

        if self.databaseUser == self.adminUser:
            raise ValueError("Use separate administrator and application accounts.")

        if self.databaseUser.startswith("pg_") or self.adminUser.startswith("pg_"):
            raise ValueError("Postgres reserves role names beginning with 'pg_'.")

        if self.databaseSchema.startswith("pg_") or self.databaseSchema in {"public", "information_schema"}:
            raise ValueError("Choose a dedicated application schema.")

        if self.databaseName in {"postgres", "template0", "template1"}:
            raise ValueError("Choose a dedicated application database.")

        if not self.installDirectory.strip() or "\x00" in self.installDirectory:
            raise ValueError("installDirectory must not be blank or contain NUL.")

        if self.rootDirectory in {Path("/"), Path.home().resolve()}:
            raise ValueError("Choose a dedicated installation subdirectory.")

        if self.rootDirectory.exists() and not self.rootDirectory.is_dir():
            raise ValueError(f"Installation path is not a directory: {self.rootDirectory}")

    @staticmethod
    def _findBrew():
        if platform.system() != "Darwin":
            raise RuntimeError("Local Postgres management currently supports macOS only.")

        if os.geteuid() == 0:
            raise RuntimeError("Run setup as your normal macOS user, not as root.")

        candidates = [shutil.which("brew"), "/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
        brew = next((Path(path) for path in candidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)

        if brew is None:
            raise RuntimeError("Homebrew was not found. Install it before local Postgres setup.")

        return brew

    def validateConfiguration(self):
        """Validate local setup requirements and print resolved settings."""
        self._validateValues()
        brew = self._findBrew()

        pglMessages.printHeader("Postgres configuration")

        values = self._configurationValues()
        values.update({
            "configurationVersion": self.CONFIGURATION_VERSION,
            "rootDirectory": self.rootDirectory,
            "dataDirectory": self.dataDirectory,
            "logFile": self.logFile,
            "configurationFile": self.configurationFile,
            "formula": self.formula,
            "brew": brew,
            "startupTimeoutSeconds": self.startupTimeoutSeconds,
            "connectTimeoutSeconds": self.connectTimeoutSeconds,
        })

        for name, value in values.items():
            pglMessages.print(f"{name:<28} {value}")

    # ------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------

    def _runCommand(self, arguments, *, capture=False, check=True):
        """Run an external command without invoking a shell."""
        environment = os.environ.copy()
        environment["PATH"] = f"{self._findBrew().parent}{os.pathsep}{environment.get('PATH', '')}"

        return subprocess.run([str(argument) for argument in arguments], check=check, text=True, stdout=subprocess.PIPE if capture else None, env=environment)

    @staticmethod
    def _askPassword(prompt, *, confirm=False):
        """Always prompt privately; never retrieve passwords from environment variables."""
        pglMessages.message("Enter Password", emphasize=True)
        password = getpass.getpass(prompt)

        if not password or any(character in password for character in ("\n", "\r", "\x00")):
            raise ValueError("Passwords must be nonempty and contain no newline or NUL characters.")

        if confirm and password != getpass.getpass("Confirm password: "):
            raise ValueError("Passwords do not match.")

        return password

    def _requireExecutable(self, name):
        executable = self.binaryDirectory / name

        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Required executable is missing or not executable: {executable}. Run install() first.")

        return executable

    def _checkDataDirectory(self):
        """Check that an initialized data directory uses the expected version."""
        versionFile = self.dataDirectory / "PG_VERSION"

        if not versionFile.is_file():
            raise RuntimeError("Postgres data directory is not initialized. Run init() first.")

        actualVersion = versionFile.read_text(encoding="utf-8").strip()

        if actualVersion != self.postgresVersion:
            raise RuntimeError(f"Existing data directory uses PostgreSQL {actualVersion}, not {self.postgresVersion}. An explicit upgrade is required.")

    def _verifyServerConnection(self, connection):
        """Verify server identity before performing administrative operations."""
        actualDirectory = Path(connection.execute("SHOW data_directory").fetchone()[0]).resolve()
        listenAddresses = connection.execute("SHOW listen_addresses").fetchone()[0]
        socketDirectories = connection.execute("SHOW unix_socket_directories").fetchone()[0]
        serverVersion = connection.execute("SHOW server_version").fetchone()[0]
        versionNumber = int(connection.execute("SHOW server_version_num").fetchone()[0])
        actualPort = int(connection.execute("SHOW port").fetchone()[0])

        if actualDirectory != self.dataDirectory.resolve():
            raise RuntimeError(f"Connected to an unexpected data directory: {actualDirectory}")

        if listenAddresses != self.postgresHost or socketDirectories != "" or actualPort != self.postgresPort:
            raise RuntimeError("Server networking does not match the configured local-only endpoint.")

        if versionNumber // 10000 != int(self.postgresVersion):
            raise RuntimeError(f"Expected PostgreSQL {self.postgresVersion}, found {serverVersion}.")

        return serverVersion, actualDirectory

    def _adminConnect(self, password, database="postgres"):
        """Open and verify an administrator connection; caller must close it."""
        self._validateValues()
        connection = psycopg.connect(host=self.postgresHost, port=self.postgresPort, dbname=database, user=self.adminUser, password=password, connect_timeout=self.connectTimeoutSeconds, autocommit=True)

        try:
            self._verifyServerConnection(connection)
        except Exception:
            connection.close()
            raise

        return connection

    # ------------------------------------------------------------
    # Install PostgreSQL software
    # ------------------------------------------------------------

    def install(self):
        """Locate or install PostgreSQL and verify its executables.

        Does not initialize the pgl-managed data directory or start its server.
        Homebrew may create a separate default data directory during installation.
        """
        self._validateValues()
        binaryDirectory = self.binaryDirectory

        if not (binaryDirectory / "postgres").is_file():
            if not self.installPostgresIfMissing:
                raise RuntimeError(f"{self.formula} is missing and installation is disabled.")

            self._runCommand([self.brew, "install", self.formula])

        for name in ("postgres", "initdb", "pg_ctl"):
            executable = binaryDirectory / name

            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise RuntimeError(f"Required executable is missing or not executable: {executable}")

        result = self._runCommand([binaryDirectory / "postgres", "--version"], capture=True)
        version = result.stdout.strip()

        if not re.search(rf"\b{re.escape(self.postgresVersion)}(?:\.|\s|$)", version):
            raise RuntimeError(f"Expected PostgreSQL {self.postgresVersion}, found: {version}")

        self.save()
        pglMessages.message(f"PostgreSQL available: {version}")
        pglMessages.message(f"Executable directory: {binaryDirectory}")
    # ------------------------------------------------------------
    # Initialize PostgreSQL data directory
    # ------------------------------------------------------------

    def init(self):
        """Initialize the managed data directory or reuse a compatible one.

        Does not start the server, create application tables, or reset passwords.
        Never overwrites a nonempty, unrecognized data directory.
        """
        self._validateValues()
        initdb = self._requireExecutable("initdb")
        dataDirectory = self.dataDirectory
        versionFile = dataDirectory / "PG_VERSION"

        self.rootDirectory.mkdir(parents=True, exist_ok=True, mode=0o700)

        if dataDirectory.exists() and not dataDirectory.is_dir():
            raise RuntimeError(f"Data path is not a directory: {dataDirectory}")

        if versionFile.exists():
            self._checkDataDirectory()
            pglMessages.message(f"Using existing PostgreSQL data directory: {dataDirectory}")
            return

        if dataDirectory.exists() and any(dataDirectory.iterdir()):
            raise RuntimeError(f"Refusing to initialize a nonempty directory: {dataDirectory}")

        adminPassword = self._askPassword(f"New PostgreSQL administrator password for {self.adminUser}: ", confirm=True)

        # The temporary password file is private and removed on exit.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as passwordFile:
            passwordFile.write(adminPassword + "\n")
            passwordFile.flush()
            self._runCommand([initdb, "-D", dataDirectory, "-U", self.adminUser, "--encoding=UTF8", "--auth=scram-sha-256", f"--pwfile={passwordFile.name}"])

        pglMessages.message(f"Initialized PostgreSQL data directory: {dataDirectory}")

    # ------------------------------------------------------------
    # Start PostgreSQL
    # ------------------------------------------------------------

    def start(self):
        """Start the managed server if it is not already running.

        Does not restart or reconfigure an already-running instance.
        Use verify() to check its actual endpoint and configuration.
        """
        self._validateValues()
        pgCtl = self._requireExecutable("pg_ctl")
        self._checkDataDirectory()

        status = self._runCommand([pgCtl, "-D", self.dataDirectory, "status"], capture=True, check=False)

        if status.returncode == 0:
            pglMessages.message("This Postgres instance is already running.")

        elif status.returncode == 3:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind((self.postgresHost, self.postgresPort))
                except OSError as error:
                    raise RuntimeError(f"Port {self.postgresPort} is unavailable. Choose another postgresPort.") from error

            serverOptions = shlex.join(["-h", self.postgresHost, "-p", str(self.postgresPort), "-k", ""])

            try:
                self._runCommand([pgCtl, "-D", self.dataDirectory, "-l", self.logFile, "-o", serverOptions, "-t", self.startupTimeoutSeconds, "-w", "start"])
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Postgres startup failed or timed out. Check server status and log: {self.logFile}") from error

            pglMessages.message(f"Postgres started at {self.postgresHost}:{self.postgresPort}.")

        else:
            raise RuntimeError(f"Could not inspect Postgres; pg_ctl exited with status {status.returncode}.")

        pglMessages.message(f"Server log: {self.logFile}")

    # ------------------------------------------------------------
    # Verify PostgreSQL server
    # ------------------------------------------------------------

    def verify(self):
        """Verify the running server's identity, version, and networking."""
        self._validateValues()
        adminPassword = self._askPassword(f"PostgreSQL administrator password for {self.adminUser}: ")

        try:
            with self._adminConnect(adminPassword) as connection:
                serverVersion = connection.execute("SHOW server_version").fetchone()[0]
        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not verify Postgres at {self.postgresHost}:{self.postgresPort}. Check that it is running and the credentials are correct.\n{error}")
            return False

        pglMessages.message(f"Verified PostgreSQL {serverVersion}")
        pglMessages.message(f"Verified data directory: {self.dataDirectory}")
        pglMessages.message(f"Verified endpoint: {self.postgresHost}:{self.postgresPort}")
        pglMessages.message("Verified local-only networking.")

    # ------------------------------------------------------------
    # Create application role, database, and schema
    # ------------------------------------------------------------
    def createDatabase(self, *, resetApplicationPassword=False):
        """Create the database; warn and return False on connection failures."""
        try:
            self._createDatabase(resetApplicationPassword=resetApplicationPassword)
        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not complete database setup at {self.postgresHost}:{self.postgresPort}. Check that Postgres is running and the credentials are correct. Earlier setup steps may have completed; rerun after resolving the issue.\n{error}")
            return False

        return True
    
    def _createDatabase(self, *, resetApplicationPassword=False):
        """Create the application role, database, and administrator-owned schema.

        Existing application passwords are preserved unless explicitly reset.
        Does not create scientific tables; those belong to pglStorage.

        Run sequentially, not concurrently. CREATE DATABASE cannot run inside
        a transaction, so a failure can leave earlier setup steps completed.
        """
        self._validateValues()

        if not isinstance(resetApplicationPassword, bool):
            raise ValueError("resetApplicationPassword must be True or False.")

        adminPassword = self._askPassword(f"PostgreSQL administrator password for {self.adminUser}: ")

        with self._adminConnect(adminPassword) as connection:
            existingRole = connection.execute("SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname = %s", (self.databaseUser,)).fetchone()

            if existingRole is not None:
                if any(existingRole[:5]) or not existingRole[5]:
                    raise RuntimeError("Existing application role has unexpected privileges or cannot log in.")

                memberships = connection.execute("SELECT parent.rolname FROM pg_auth_members membership JOIN pg_roles member ON member.oid = membership.member JOIN pg_roles parent ON parent.oid = membership.roleid WHERE member.rolname = %s", (self.databaseUser,)).fetchall()

                if memberships:
                    raise RuntimeError(f"Existing application role has role memberships requiring review: {memberships}")

            databaseOwner = connection.execute("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = %s", (self.databaseName,)).fetchone()

            if databaseOwner is not None and databaseOwner[0] != self.adminUser:
                raise RuntimeError(f"Database is owned by {databaseOwner[0]!r}, not {self.adminUser!r}.")

            # Check an existing schema before making changes.
            if databaseOwner is not None:
                with self._adminConnect(adminPassword, self.databaseName) as databaseConnection:
                    schemaOwner = databaseConnection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

                    if schemaOwner is not None and schemaOwner[0] != self.adminUser:
                        raise RuntimeError(f"Schema is owned by {schemaOwner[0]!r}, not {self.adminUser!r}.")

            settingPassword = existingRole is None or resetApplicationPassword
            prompt = "Choose the application password: " if settingPassword else "Existing application password: "
            applicationPassword = self._askPassword(prompt, confirm=settingPassword)

            if existingRole is None:
                statement = sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}").format(sql.Identifier(self.databaseUser), sql.Literal(applicationPassword))
                connection.execute(statement)

            elif resetApplicationPassword:
                statement = sql.SQL("ALTER ROLE {} PASSWORD {}").format(sql.Identifier(self.databaseUser), sql.Literal(applicationPassword))
                connection.execute(statement)

            if databaseOwner is None:
                connection.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(self.databaseName), sql.Identifier(self.adminUser)))

            connection.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(self.databaseName), sql.Identifier(self.databaseUser)))

        with self._adminConnect(adminPassword, self.databaseName) as connection:
            with connection.transaction():
                schemaOwner = connection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

                if schemaOwner is None:
                    connection.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(sql.Identifier(self.databaseSchema), sql.Identifier(self.adminUser)))
                elif schemaOwner[0] != self.adminUser:
                    raise RuntimeError(f"Schema is owned by {schemaOwner[0]!r}, not {self.adminUser!r}.")

                connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(self.databaseSchema)))
                connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}").format(sql.Identifier(self.databaseSchema), sql.Identifier(self.databaseUser)))
                connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(self.databaseSchema), sql.Identifier(self.databaseUser)))

                connection.execute(sql.SQL("ALTER ROLE {} IN DATABASE {} SET search_path = {}, pg_catalog").format(sql.Identifier(self.databaseUser), sql.Identifier(self.databaseName), sql.Identifier(self.databaseSchema)))

        # Verify the supplied application credentials and default schema.
        with psycopg.connect(host=self.postgresHost, port=self.postgresPort, dbname=self.databaseName, user=self.databaseUser, password=applicationPassword, connect_timeout=self.connectTimeoutSeconds, autocommit=True) as connection:
            self._verifyApplicationConnection(connection)

        pglMessages.message(f"Application account ready: {self.databaseUser}")
        pglMessages.message(f"Database ready: {self.databaseName}")
        pglMessages.message(f"Schema ready: {self.databaseSchema} (owner: {self.adminUser})")
        pglMessages.message("Application login verified.")

    # ------------------------------------------------------------
    # Verify application connection
    # ------------------------------------------------------------

    def _verifyApplicationConnection(self, connection):
        """Check application identity and execute a simple read-only query."""
        identity = connection.execute("SELECT current_database(), current_user, current_schema()").fetchone()
        expected = (self.databaseName, self.databaseUser, self.databaseSchema)

        if identity != expected:
            raise RuntimeError(f"Unexpected connection identity: {identity}; expected {expected}")

        result = connection.execute("SELECT 42").fetchone()

        if result != (42,):
            raise RuntimeError("Database query test failed.")

        return identity

    def verifyDatabase(self):
        """Verify application access; warn and return False if connection fails."""
        self._validateValues()
        applicationPassword = self._askPassword(f"PostgreSQL application password for {self.databaseUser}: ")

        try:
            with psycopg.connect(host=self.postgresHost, port=self.postgresPort, dbname=self.databaseName, user=self.databaseUser, password=applicationPassword, connect_timeout=self.connectTimeoutSeconds) as connection:
                try:
                    identity = self._verifyApplicationConnection(connection)
                finally:
                    connection.rollback()
        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not verify database {self.databaseName!r} at {self.postgresHost}:{self.postgresPort}. Check that Postgres is running and the credentials are correct.\n{error}")
            return False

        pglMessages.message(f"Database: {identity[0]}")
        pglMessages.message(f"User: {identity[1]}")
        pglMessages.message(f"Schema: {identity[2]}")
        pglMessages.message("Application connection, schema selection, and query test passed.")
        pglMessages.message("Table read/write testing is deferred until schema migrations have run.")
        return True        
    
    def stop(self):
        """Stop the managed server cleanly without deleting any data.

        Disconnects clients and rolls back active transactions before shutdown.
        Safe to call when the server is already stopped.
        """
        self._validateValues()
        pgCtl = self._requireExecutable("pg_ctl")
        self._checkDataDirectory()

        status = self._runCommand([pgCtl, "-D", self.dataDirectory, "status"], capture=True, check=False)

        if status.returncode == 3:
            pglMessages.message("This Postgres instance is already stopped.")
            return

        if status.returncode != 0:
            raise RuntimeError(f"Could not inspect Postgres; pg_ctl exited with status {status.returncode}.")

        try:
            self._runCommand([pgCtl, "-D", self.dataDirectory, "stop", "-m", "fast", "-t", self.startupTimeoutSeconds, "-w"])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Postgres shutdown failed or timed out. Check server status and log: {self.logFile}") from error

        pglMessages.message("Postgres stopped.")
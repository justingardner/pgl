################################################################
#   filename: pglPostgres.py
#    purpose: PostgreSQL configuration, connections, and local
#             server management
#         by: JLG
#       date: Oct 5, 2026
################################################################

import getpass
import json
import os
import platform
import posixpath
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
from psycopg import sql
from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from traitlets import Bool, Int, Unicode, List

from .pglMessages import pglMessages
from .pglSettings import pglTraitSettings


################################################################
# PostgreSQL connection transport
################################################################

class pglPostgresTransport:
    """Open direct or SSH-tunneled PostgreSQL connections.

    Connections and tunnels are runtime resources, never serialized.

    SSH uses the system OpenSSH executable, existing SSH keys/agent,
    SSH configuration, and known_hosts. Unknown host keys are not
    accepted automatically.
    """

    @classmethod
    @contextmanager
    def connect(cls, postgres, *, user, password, database=None, sslRootCert=None):
        """Open a connection and close its resources when the context exits."""
        database = postgres.databaseName if database is None else database

        options = {
            "host": postgres.connectionHost,
            "port": postgres.connectionPort,
            "dbname": database,
            "user": user,
            "password": password,
            "connect_timeout": postgres.connectTimeoutSeconds,
            "sslmode": postgres.databaseSSLMode,
            "autocommit": True,
        }

        if sslRootCert:
            options["sslrootcert"] = str(Path(sslRootCert).expanduser().resolve())

        if postgres.connectionTransport == "direct":
            with cls._direct(options) as connection:
                yield connection

        elif postgres.connectionTransport == "ssh":
            with cls._ssh(postgres, options) as connection:
                yield connection

        else:
            raise ValueError(f"Unsupported connection transport: {postgres.connectionTransport}")

    @staticmethod
    @contextmanager
    def _direct(options):
        connection = psycopg.connect(**options)

        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _sshDiagnostic(file):
        file.flush()
        file.seek(0)
        return file.read().decode("utf-8", errors="replace").strip()

    @classmethod
    @contextmanager
    def _ssh(cls, postgres, options):
        """Open an SSH tunnel whose lifetime matches the database connection."""
        ssh = shutil.which("ssh")

        if ssh is None:
            raise RuntimeError("OpenSSH was not found.")

        # Reserve a candidate port. OpenSSH takes ownership after release.
        # ExitOnForwardFailure makes any binding race fail explicitly.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            localPort = reservation.getsockname()[1]

        host = postgres.connectionHost
        forwardHost = f"[{host}]" if ":" in host and not host.startswith("[") else host
        forwarding = f"127.0.0.1:{localPort}:{forwardHost}:{postgres.connectionPort}"

        # A short private directory avoids Unix control-socket path limits.
        temporaryRoot = "/tmp" if os.name == "posix" and Path("/tmp").is_dir() else None

        with tempfile.TemporaryDirectory(prefix="pgl-ssh-", dir=temporaryRoot) as directory:
            controlPath = str(Path(directory) / "control")

            arguments = [
                ssh,
                "-N",
                "-T",
                "-M",
                "-S", controlPath,
                "-p", str(postgres.sshPort),
                "-L", forwarding,
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ControlPersist=no",
                "-o", "ForkAfterAuthentication=no",
                "-o", f"ConnectTimeout={postgres.connectTimeoutSeconds}",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3",
            ]

            if postgres.sshUser:
                arguments.extend(["-l", postgres.sshUser])

            arguments.append(postgres.sshHost)

            with tempfile.TemporaryFile(mode="w+b") as diagnostic:
                process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=diagnostic)

                try:
                    deadline = time.monotonic() + postgres.connectTimeoutSeconds

                    while True:
                        if process.poll() is not None:
                            detail = cls._sshDiagnostic(diagnostic)
                            raise RuntimeError(f"SSH tunnel failed: {detail or 'SSH exited before the tunnel was ready.'}")

                        if Path(controlPath).exists():
                            try:
                                check = subprocess.run([ssh, "-S", controlPath, "-O", "check", postgres.sshHost], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1, check=False)
                            except subprocess.TimeoutExpired:
                                check = None

                            if check is not None and check.returncode == 0:
                                break

                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out waiting for the SSH tunnel.")

                        time.sleep(0.1)

                    if process.poll() is not None:
                        detail = cls._sshDiagnostic(diagnostic)
                        raise RuntimeError(f"SSH exited while establishing the tunnel: {detail}")

                    tunnelOptions = dict(options)

                    # Preserve the logical database hostname for TLS hostname
                    # verification, while connecting to the local tunnel.
                    tunnelOptions["hostaddr"] = "127.0.0.1"
                    tunnelOptions["port"] = localPort

                    connection = psycopg.connect(**tunnelOptions)

                    try:
                        yield connection
                    finally:
                        connection.close()

                finally:
                    if process.poll() is None:
                        process.terminate()

                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()


################################################################
# PostgreSQL configuration and management
################################################################

class pglPostgres(pglTraitSettings):
    """Saved configuration, client connections, and local server management.

    Configuration can be discovered through fsspec.
    SQL connections use psycopg over direct TCP/TLS or an SSH tunnel.
    Scientific payload access belongs to pglStorage.

    Loading does not install software, start a server, or connect.
    Passwords, connections, filesystem objects, and tunnels are not saved.

    Local server management currently supports macOS/Homebrew only.
    """

    # Local server administration.
    postgresVersion = Unicode("17", postgresConfiguration=True, help="PostgreSQL major version used by the locally managed server.")
    postgresHost = Unicode("127.0.0.1", postgresConfiguration=True, help="Address for local server management; must be 127.0.0.1.")
    postgresPort = Int(5432, min=1, max=65535, postgresConfiguration=True, help="Port used by the locally managed PostgreSQL server.")

    databaseName = Unicode("pgl", postgresConfiguration=True, help="Database containing the session data store.")
    databaseSchema = Unicode("pgl", postgresConfiguration=True, help="Schema containing the session data store tables.")
    databaseUser = Unicode("pgl_app", postgresConfiguration=True, help="Default application role used for ordinary reads and writes.")
    adminUser = Unicode("pgl_admin", postgresConfiguration=True, help="Administrator role used for local setup and schema migrations.")

    installDirectory = Unicode("~/data/pgl", postgresConfiguration=True, help="Local server directory containing configuration, log, and database files. Supports ~.")
    storageLocations = List(Unicode(), default_value=[], postgresConfiguration=True, help="Ordered payload storage paths or URLs. Plain paths refer to the installation machine. Setup defaults to installDirectory/storage.")
    installPostgresIfMissing = Bool(True, postgresConfiguration=True, help="Allow Homebrew installation when the configured PostgreSQL version is missing.")

    # Client connection transport.
    connectionTransport = Unicode("direct", postgresConfiguration=True, help="Client database transport: direct or ssh.")
    databaseHost = Unicode("", postgresConfiguration=True, help="Database hostname reached directly or from the SSH server. Empty uses postgresHost.")
    databasePort = Int(default_value=None, allow_none=True, min=1, max=65535, postgresConfiguration=True, help="Database port reached directly or from the SSH server. None uses postgresPort.")

    sshHost = Unicode("", postgresConfiguration=True, help="SSH server hostname, IP address, or alias from the client's SSH configuration.")
    sshPort = Int(22, min=1, max=65535, postgresConfiguration=True, help="SSH server port.")
    sshUser = Unicode("", postgresConfiguration=True, help="SSH username. Empty uses the client's SSH configuration or current username.")

    databaseSSLMode = Unicode("prefer", postgresConfiguration=True, help="PostgreSQL TLS mode: disable, allow, prefer, require, verify-ca, or verify-full.")

    CONFIGURATION_VERSION = 3
    CONFIGURATION_FILENAME = "configuration.json"
    STARTUP_TIMEOUT_SECONDS = 60
    CONNECT_TIMEOUT_SECONDS = 10

    # ------------------------------------------------------------
    # Derived values: not serialized separately
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
    def connectionHost(self):
        return self.databaseHost or self.postgresHost

    @property
    def connectionPort(self):
        return self.postgresPort if self.databasePort is None else self.databasePort

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
        """Only explicitly marked configuration traits are serialized."""
        return {"configurationVersion": self.CONFIGURATION_VERSION, **self._configurationValues()}

    @classmethod
    def fromJSONdict(cls, data, type="all", filename=None):
        """Restore settings and upgrade older configurations in memory."""
        values = dict(data)
        version = values.pop("configurationVersion", None)

        transportFields = {
            "connectionTransport",
            "databaseHost",
            "databasePort",
            "sshHost",
            "sshPort",
            "sshUser",
            "databaseSSLMode",
        }

        expected = set(cls._configurationNames())

        if version == 1:
            legacyFields = expected - transportFields - {"storageLocations"}

            if set(values) != legacyFields:
                raise ValueError("Invalid version-1 Postgres configuration fields.")

            values.update({
                "connectionTransport": "direct",
                "databaseHost": "",
                "databasePort": None,
                "sshHost": "",
                "sshPort": 22,
                "sshUser": "",
                "databaseSSLMode": "prefer",
            })

            version = 2

        if version == 2:
            if set(values) != expected - {"storageLocations"}:
                raise ValueError("Invalid version-2 Postgres configuration fields.")

            # Do not expand ~ or resolve paths here: the configuration may
            # describe a remote installation.
            values["storageLocations"] = [posixpath.join(values["installDirectory"], "storage")]
            version = 3

        if version != cls.CONFIGURATION_VERSION:
            raise ValueError(f"Unsupported Postgres configuration version: {version!r}")

        supplied = set(values)

        if supplied != expected:
            raise ValueError(f"Invalid configuration fields; missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}")

        obj = cls(**values)
        obj._validateValues()
        return obj

    @staticmethod
    def _resolveConfigurationFile(filename, filesystem=None, storageOptions=None):
        """Resolve a configuration file using fsspec's native path handling."""
        location = str(filename)

        if filesystem is not None and storageOptions:
            raise ValueError("Supply either filesystem or storageOptions, not both.")

        if filesystem is None:
            if "://" not in location:
                location = str(Path(location).expanduser().resolve())

            filesystem, path = url_to_fs(location, **(storageOptions or {}))

        else:
            if not isinstance(filesystem, AbstractFileSystem):
                raise TypeError("filesystem must be an fsspec AbstractFileSystem.")

            path = filesystem._strip_protocol(location)

        protocols = filesystem.protocol
        protocols = (protocols,) if isinstance(protocols, str) else protocols
        isLocal = any(protocol in {"file", "local"} for protocol in protocols)

        if isLocal:
            path = str(Path(path).expanduser().resolve())

        return filesystem, path, isLocal

    @classmethod
    def load(cls, filename=None, *, filesystem=None, storageOptions=None):
        """Load configuration through fsspec and infer a missing SSH hostname."""
        if filename is None:
            directory = cls.class_traits()["installDirectory"].default_value
            filename = posixpath.join(directory, cls.CONFIGURATION_FILENAME)

        location = str(filename)
        filesystem, path, isLocal = cls._resolveConfigurationFile(location, filesystem=filesystem, storageOptions=storageOptions)

        with filesystem.open(path, "rt", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Postgres configuration must be a JSON object.")

        if data.pop("__class__", None) != cls.__name__:
            raise ValueError("The file is not a pglPostgres configuration.")

        if data.get("connectionTransport") == "ssh" and not data.get("sshHost"):
            url = urlsplit(location)

            if url.scheme in {"ssh", "sftp"} and url.hostname:
                data["sshHost"] = url.hostname
            else:
                protocols = filesystem.protocol
                protocols = (protocols,) if isinstance(protocols, str) else protocols

                if any(protocol in {"ssh", "sftp"} for protocol in protocols):
                    data["sshHost"] = getattr(filesystem, "host", "") or ""

        obj = cls.fromJSONdict(data)
        obj._configurationLoadedRemotely = not isLocal
        return obj

    @classmethod
    def fromSettings(cls, settings=None, settingsName=None, *, filesystem=None, storageOptions=None):
        """Discover configuration from databasePath in pgl settings.

        Missing configuration produces a warning and returns None.
        Invalid configuration and authentication failures remain errors.
        """
        from .pglSettings import pglSettingsManager

        resolvedSettings = pglSettingsManager.getSettings(settings=settings, settingsName=settingsName)

        if resolvedSettings is None:
            pglMessages.warning("Could not resolve pgl settings.")
            return None

        directory = resolvedSettings.databasePath.strip()

        if not directory:
            pglMessages.warning("No databasePath is configured in pgl settings.")
            return None

        filename = posixpath.join(directory.rstrip("/") + "/", cls.CONFIGURATION_FILENAME)

        try:
            return cls.load(filename, filesystem=filesystem, storageOptions=storageOptions)
        except FileNotFoundError:
            pglMessages.warning("No PostgreSQL configuration was found in the configured databasePath.")
            return None

    def save(self, filename=None, *, overwrite=False):
        """Atomically save local configuration with owner-only permissions.

        Remote configuration discovery is supported, but remote publication
        is not implemented here. Publish a password-free configuration using
        the remote backend's appropriate deployment procedure.

        Identical existing settings are left unchanged.
        Changed settings require overwrite=True.
        """
        self._requireLocalConfiguration()
        self.installDirectory = str(self.rootDirectory)
        self._setDefaultStorageLocations()
        self._validateValues()

        filename = self.configurationFile if filename is None else filename
        _, path, isLocal = self._resolveConfigurationFile(filename)

        if not isLocal:
            raise NotImplementedError("Remote configuration publication is not implemented. Save locally and publish using the target backend's deployment procedure.")

        target = Path(path)
        dataDirectory = self.dataDirectory.resolve()

        if target == dataDirectory or dataDirectory in target.parents:
            raise ValueError("Do not store configuration inside Postgres's internal data directory.")

        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if target.exists() and not overwrite:
            existing = self.load(target)

            if existing.toJSONdict() != self.toJSONdict():
                raise RuntimeError(f"Different configuration already exists at {target}; use overwrite=True only for an intentional change.")

            pglMessages.message(f"Configuration already saved: {target}")
            return

        temporaryPath = None

        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, prefix=".pgl-postgres-", delete=False) as temporary:
                temporaryPath = Path(temporary.name)
                os.chmod(temporaryPath, 0o600)
                temporary.write(self.toJSON())
                temporary.flush()
                os.fsync(temporary.fileno())

            if overwrite:
                os.replace(temporaryPath, target)
            else:
                os.link(temporaryPath, target)

        finally:
            if temporaryPath is not None:
                temporaryPath.unlink(missing_ok=True)

        pglMessages.message(f"Configuration saved: {target}")
        
    def _setDefaultStorageLocations(self):
        """Set the initial payload location without replacing configured locations."""
        self._requireLocalConfiguration()

        if not self.storageLocations:
            self.storageLocations = [str(self.rootDirectory / "storage")]        

    # ------------------------------------------------------------
    # Validation and display
    # ------------------------------------------------------------

    def _validateValues(self):
        """Validate saved values without probing the server or local hardware."""
        if self.postgresHost != "127.0.0.1":
            raise ValueError("postgresHost is for local management and must be 127.0.0.1. Use databaseHost for a remote endpoint.")

        if not re.fullmatch(r"[0-9]+", self.postgresVersion) or int(self.postgresVersion) < 10:
            raise ValueError("postgresVersion must be a PostgreSQL major version string of 10 or later.")

        for name in ("databaseName", "databaseSchema", "databaseUser", "adminUser"):
            value = getattr(self, name)

            if not value.strip() or "\x00" in value or len(value.encode("utf-8")) > 63:
                raise ValueError(f"{name} must contain 1–63 UTF-8 bytes, must not be blank, and must not contain NUL.")

        if self.databaseUser == self.adminUser:
            raise ValueError("Use separate administrator and application accounts.")

        if self.databaseUser.startswith("pg_") or self.adminUser.startswith("pg_"):
            raise ValueError("Postgres reserves role names beginning with pg_.")

        if self.databaseSchema.startswith("pg_") or self.databaseSchema in {"public", "information_schema"}:
            raise ValueError("Choose a dedicated application schema.")

        if self.databaseName in {"postgres", "template0", "template1"}:
            raise ValueError("Choose a dedicated application database.")

        if not self.installDirectory.strip() or "\x00" in self.installDirectory:
            raise ValueError("installDirectory must not be blank or contain NUL.")

        if "://" in self.installDirectory:
            raise ValueError("installDirectory is a server-local path, not an fsspec URL. Use databasePath in pgl settings for configuration discovery.")

        if self.connectionTransport not in {"direct", "ssh"}:
            raise ValueError("connectionTransport must be direct or ssh.")

        if not self.connectionHost.strip() or any(character.isspace() for character in self.connectionHost) or "\x00" in self.connectionHost or "," in self.connectionHost or "/" in self.connectionHost:
            raise ValueError("databaseHost must be a single hostname or IP address.")

        if self.sshHost and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:%-]*", self.sshHost):
            raise ValueError("sshHost must be a hostname, IP address, or SSH alias—not a URL.")
        
        if self.sshUser and (self.sshUser.startswith("-") or any(character.isspace() for character in self.sshUser) or "\x00" in self.sshUser):
            raise ValueError("Invalid SSH username.")

        if self.databaseSSLMode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            raise ValueError("Invalid databaseSSLMode.")
        
        for location in self.storageLocations:
            if not location.strip() or any(character in location for character in ("\x00", "\n", "\r")):
                raise ValueError("Storage locations must be nonempty paths or URLs without NUL or newline characters.")

        if len(self.storageLocations) != len(set(self.storageLocations)):
            raise ValueError("Storage locations must not contain duplicate entries.")

    def _requireLocalConfiguration(self):
        if getattr(self, "_configurationLoadedRemotely", False):
            raise RuntimeError("This configuration was loaded remotely. Local server administration and local configuration saving are disabled.")

    def _validateLocalEnvironment(self):
        self._requireLocalConfiguration()
        self._validateValues()

        if self.rootDirectory in {Path("/"), Path.home().resolve()}:
            raise ValueError("Choose a dedicated installation subdirectory.")

        if self.rootDirectory.exists() and not self.rootDirectory.is_dir():
            raise ValueError(f"Installation path is not a directory: {self.rootDirectory}")

        return self._findBrew()

    def showConfiguration(self):
        """Display configuration without requiring Homebrew or a connection."""
        pglMessages.message("Postgres configuration")

        for name, value in self._configurationValues().items():
            pglMessages.message(f"{name:<28} {value}")

        pglMessages.message(f"{'connectionHost':<28} {self.connectionHost}")
        pglMessages.message(f"{'connectionPort':<28} {self.connectionPort}")

        source = "remote" if getattr(self, "_configurationLoadedRemotely", False) else "local or newly constructed"
        pglMessages.message(f"{'configurationSource':<28} {source}")

    def validateConfiguration(self):
        """Validate and display settings without requiring a running server."""
        self._validateValues()
        self.showConfiguration()

        if getattr(self, "_configurationLoadedRemotely", False):
            pglMessages.message("Remote client configuration; local server checks skipped.")
            return

        brew = self._validateLocalEnvironment()

        values = {
            "configurationVersion": self.CONFIGURATION_VERSION,
            "rootDirectory": self.rootDirectory,
            "dataDirectory": self.dataDirectory,
            "logFile": self.logFile,
            "configurationFile": self.configurationFile,
            "brew": brew,
            "formula": self.formula,
            "startupTimeoutSeconds": self.startupTimeoutSeconds,
            "connectTimeoutSeconds": self.connectTimeoutSeconds,
        }

        for name, value in values.items():
            pglMessages.message(f"{name:<28} {value}")

    # ------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------

    @staticmethod
    def _findBrew():
        if platform.system() != "Darwin":
            raise RuntimeError("Local Postgres management currently supports macOS only.")

        if os.geteuid() == 0:
            raise RuntimeError("Run local setup as your normal macOS user, not as root.")

        candidates = [shutil.which("brew"), "/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
        brew = next((Path(path) for path in candidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)

        if brew is None:
            raise RuntimeError("Homebrew was not found.")

        return brew

    def _runCommand(self, arguments, *, capture=False, check=True):
        self._requireLocalConfiguration()
        environment = os.environ.copy()
        environment["PATH"] = f"{self._findBrew().parent}{os.pathsep}{environment.get('PATH', '')}"

        return subprocess.run([str(argument) for argument in arguments], check=check, text=True, stdout=subprocess.PIPE if capture else None, env=environment)

    @staticmethod
    def _askPassword(prompt, *, confirm=False):
        """Always prompt; do not retrieve passwords from environment variables."""
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
        versionFile = self.dataDirectory / "PG_VERSION"

        if not versionFile.is_file():
            raise RuntimeError("Postgres data directory is not initialized. Run init() first.")

        version = versionFile.read_text(encoding="utf-8").strip()

        if version != self.postgresVersion:
            raise RuntimeError(f"Existing data directory uses PostgreSQL {version}, not {self.postgresVersion}. An explicit upgrade is required.")

    # ------------------------------------------------------------
    # Application connections: local or remote
    # ------------------------------------------------------------

    @contextmanager
    def connect(self, *, user=None, sslRootCert=None):
        """Open a client connection and close it, and its tunnel, on exit.

        Prompts for the database password.
        Uses autocommit; wrap related writes in connection.transaction().
        Never starts or administers a server.
        """
        self._validateValues()
        if self.connectionTransport == "ssh" and not self.sshHost:
            raise RuntimeError("No SSH hostname is available. Load configuration from an ssh:// URL or specify sshHost.")
        databaseUser = self.databaseUser if user is None else user

        if not isinstance(databaseUser, str) or not databaseUser.strip() or "\x00" in databaseUser:
            raise ValueError("Database username must be a nonempty string without NUL.")

        if getattr(self, "_configurationLoadedRemotely", False):
            if self.connectionTransport == "direct" and self.connectionHost.lower() in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError("Remote configuration points directly to loopback. Configure an SSH transport or a directly reachable database endpoint.")

        password = self._askPassword(f"PostgreSQL password for {databaseUser}: ")

        with pglPostgresTransport.connect(self, user=databaseUser, password=password, sslRootCert=sslRootCert) as connection:
            identity = connection.execute("SELECT current_database(), current_user").fetchone()

            if identity != (self.databaseName, databaseUser):
                raise RuntimeError(f"Unexpected connection identity: {identity}")

            schemaExists = connection.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

            if schemaExists is None:
                raise RuntimeError(f"Database schema does not exist: {self.databaseSchema}")

            connection.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(self.databaseSchema)))

            if connection.execute("SELECT current_schema()").fetchone()[0] != self.databaseSchema:
                raise RuntimeError(f"Cannot access configured schema: {self.databaseSchema}")

            yield connection

    def verifyDatabase(self, *, user=None, sslRootCert=None):
        """Verify client access; warn and return False on connection failures."""
        try:
            with self.connect(user=user, sslRootCert=sslRootCert) as connection:
                identity = connection.execute("SELECT current_database(), current_user, current_schema()").fetchone()

                if connection.execute("SELECT 42").fetchone() != (42,):
                    raise RuntimeError("Database query test failed.")

        except (psycopg.OperationalError, OSError, RuntimeError) as error:
            pglMessages.warning(f"Could not verify database access: {error}")
            return False

        pglMessages.message(f"Database: {identity[0]}")
        pglMessages.message(f"User: {identity[1]}")
        pglMessages.message(f"Schema: {identity[2]}")
        pglMessages.message(f"Transport: {self.connectionTransport}")
        pglMessages.message("Application connection and query test passed.")
        return True

    # ------------------------------------------------------------
    # Local administrator connections
    # ------------------------------------------------------------

    def _verifyServerConnection(self, connection):
        """Verify a locally managed server before administrative operations."""
        actualDirectory = Path(connection.execute("SHOW data_directory").fetchone()[0]).resolve()
        listenAddresses = connection.execute("SHOW listen_addresses").fetchone()[0]
        socketDirectories = connection.execute("SHOW unix_socket_directories").fetchone()[0]
        serverVersion = connection.execute("SHOW server_version").fetchone()[0]
        versionNumber = int(connection.execute("SHOW server_version_num").fetchone()[0])
        actualPort = int(connection.execute("SHOW port").fetchone()[0])

        if actualDirectory != self.dataDirectory.resolve():
            raise RuntimeError(f"Connected to an unexpected data directory: {actualDirectory}")

        if listenAddresses != self.postgresHost or socketDirectories != "" or actualPort != self.postgresPort:
            raise RuntimeError("Server networking does not match the local management configuration.")

        if versionNumber // 10000 != int(self.postgresVersion):
            raise RuntimeError(f"Expected PostgreSQL {self.postgresVersion}, found {serverVersion}.")

        return serverVersion, actualDirectory

    def _adminConnect(self, password, database="postgres"):
        """Open a verified local administrator connection; caller closes it.

        Retained for compatibility with pglStorage's local migration runner.
        Does not use the client SSH/direct transport settings.
        """
        self._requireLocalConfiguration()
        self._validateValues()

        connection = psycopg.connect(host=self.postgresHost, port=self.postgresPort, dbname=database, user=self.adminUser, password=password, connect_timeout=self.connectTimeoutSeconds, sslmode="prefer", autocommit=True)

        try:
            self._verifyServerConnection(connection)
        except Exception:
            connection.close()
            raise

        return connection

    # ------------------------------------------------------------
    # Local software installation
    # ------------------------------------------------------------

    def install(self):
        """Locate/install PostgreSQL, verify executables, and save configuration."""
        self._validateLocalEnvironment()
        self.installDirectory = str(self.rootDirectory)
        self._setDefaultStorageLocations()
        binaryDirectory = self.binaryDirectory

        # Catch conflicting saved settings before invoking Homebrew.
        if self.configurationFile.exists():
            existing = self.load(self.configurationFile)

            if existing.toJSONdict() != self.toJSONdict():
                raise RuntimeError("A different configuration is already saved. Review it and use save(overwrite=True) explicitly before installing.")

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
    # Local data-directory initialization
    # ------------------------------------------------------------

    def init(self):
        """Initialize the local data directory without overwriting existing data."""
        self._validateLocalEnvironment()
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

        password = self._askPassword(f"New PostgreSQL administrator password for {self.adminUser}: ", confirm=True)

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as passwordFile:
            passwordFile.write(password + "\n")
            passwordFile.flush()
            self._runCommand([initdb, "-D", dataDirectory, "-U", self.adminUser, "--encoding=UTF8", "--auth=scram-sha-256", f"--pwfile={passwordFile.name}"])

        pglMessages.message(f"Initialized PostgreSQL data directory: {dataDirectory}")

    # ------------------------------------------------------------
    # Local server lifecycle
    # ------------------------------------------------------------

    def start(self):
        """Start the managed server, leaving an already-running server unchanged."""
        self._validateLocalEnvironment()
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
                    raise RuntimeError(f"Port {self.postgresPort} is unavailable.") from error

            options = shlex.join(["-h", self.postgresHost, "-p", str(self.postgresPort), "-k", ""])

            try:
                self._runCommand([pgCtl, "-D", self.dataDirectory, "-l", self.logFile, "-o", options, "-t", self.startupTimeoutSeconds, "-w", "start"])
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Postgres startup failed or timed out. Check {self.logFile}") from error

            pglMessages.message(f"Postgres started at {self.postgresHost}:{self.postgresPort}.")

        else:
            raise RuntimeError(f"Could not inspect Postgres; pg_ctl exited with status {status.returncode}.")

        pglMessages.message(f"Server log: {self.logFile}")

    def stop(self):
        """Stop the local server cleanly without deleting data.

        Disconnects clients and rolls back active transactions.
        """
        self._validateLocalEnvironment()
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
            raise RuntimeError(f"Postgres shutdown failed or timed out. Check {self.logFile}") from error

        pglMessages.message("Postgres stopped.")

    def verify(self):
        """Verify the locally managed server; warn on connection failure."""
        self._requireLocalConfiguration()
        self._validateValues()
        password = self._askPassword(f"PostgreSQL administrator password for {self.adminUser}: ")

        try:
            with self._adminConnect(password) as connection:
                version = connection.execute("SHOW server_version").fetchone()[0]
        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not verify local Postgres. Check that it is running and the credentials are correct.\n{error}")
            return False

        pglMessages.message(f"Verified PostgreSQL {version}")
        pglMessages.message(f"Verified data directory: {self.dataDirectory}")
        pglMessages.message(f"Verified endpoint: {self.postgresHost}:{self.postgresPort}")
        pglMessages.message("Verified local-only networking.")
        return True

    # ------------------------------------------------------------
    # Local role/database/schema creation
    # ------------------------------------------------------------

    def createDatabase(self, *, resetApplicationPassword=False):
        """Create the local application database; warn on connection failure."""
        try:
            self._createDatabase(resetApplicationPassword=resetApplicationPassword)
        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not complete database setup. Check the server and credentials. Earlier setup steps may have completed; rerun after resolving the issue.\n{error}")
            return False

        return True

    def _createDatabase(self, *, resetApplicationPassword=False):
        """Create the role, database, and administrator-owned schema.

        Does not create scientific tables.
        CREATE DATABASE cannot be part of a transaction; run sequentially.
        """
        self._requireLocalConfiguration()
        self._validateValues()

        if not isinstance(resetApplicationPassword, bool):
            raise ValueError("resetApplicationPassword must be True or False.")

        adminPassword = self._askPassword(f"PostgreSQL administrator password for {self.adminUser}: ")

        with self._adminConnect(adminPassword) as connection:
            role = connection.execute("SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls, rolcanlogin FROM pg_roles WHERE rolname = %s", (self.databaseUser,)).fetchone()

            if role is not None:
                if any(role[:5]) or not role[5]:
                    raise RuntimeError("Application role has unexpected privileges or cannot log in.")

                memberships = connection.execute("SELECT parent.rolname FROM pg_auth_members membership JOIN pg_roles member ON member.oid = membership.member JOIN pg_roles parent ON parent.oid = membership.roleid WHERE member.rolname = %s", (self.databaseUser,)).fetchall()

                if memberships:
                    raise RuntimeError(f"Application role has memberships requiring review: {memberships}")

            owner = connection.execute("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname = %s", (self.databaseName,)).fetchone()

            if owner is not None and owner[0] != self.adminUser:
                raise RuntimeError(f"Database is owned by {owner[0]!r}, not {self.adminUser!r}.")

            if owner is not None:
                with self._adminConnect(adminPassword, self.databaseName) as databaseConnection:
                    schemaOwner = databaseConnection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

                    if schemaOwner is not None and schemaOwner[0] != self.adminUser:
                        raise RuntimeError(f"Schema is owned by {schemaOwner[0]!r}, not {self.adminUser!r}.")

            settingPassword = role is None or resetApplicationPassword
            prompt = "Choose the application password: " if settingPassword else "Existing application password: "
            applicationPassword = self._askPassword(prompt, confirm=settingPassword)

            if role is None:
                statement = sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}").format(sql.Identifier(self.databaseUser), sql.Literal(applicationPassword))
                connection.execute(statement)

            elif resetApplicationPassword:
                statement = sql.SQL("ALTER ROLE {} PASSWORD {}").format(sql.Identifier(self.databaseUser), sql.Literal(applicationPassword))
                connection.execute(statement)

            if owner is None:
                connection.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(self.databaseName), sql.Identifier(self.adminUser)))

            connection.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(self.databaseName), sql.Identifier(self.databaseUser)))

        with self._adminConnect(adminPassword, self.databaseName) as connection:
            with connection.transaction():
                owner = connection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

                if owner is None:
                    connection.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(sql.Identifier(self.databaseSchema), sql.Identifier(self.adminUser)))
                elif owner[0] != self.adminUser:
                    raise RuntimeError(f"Schema is owned by {owner[0]!r}, not {self.adminUser!r}.")

                connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(self.databaseSchema)))
                connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}").format(sql.Identifier(self.databaseSchema), sql.Identifier(self.databaseUser)))
                connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(self.databaseSchema), sql.Identifier(self.databaseUser)))
                connection.execute(sql.SQL("ALTER ROLE {} IN DATABASE {} SET search_path = {}, pg_catalog").format(sql.Identifier(self.databaseUser), sql.Identifier(self.databaseName), sql.Identifier(self.databaseSchema)))

        # This is local setup verification, independent of client transport.
        with psycopg.connect(host=self.postgresHost, port=self.postgresPort, dbname=self.databaseName, user=self.databaseUser, password=applicationPassword, connect_timeout=self.connectTimeoutSeconds, sslmode="prefer", autocommit=True) as connection:
            identity = connection.execute("SELECT current_database(), current_user, current_schema()").fetchone()

            if identity != (self.databaseName, self.databaseUser, self.databaseSchema):
                raise RuntimeError(f"Unexpected application connection identity: {identity}")

            if connection.execute("SELECT 42").fetchone() != (42,):
                raise RuntimeError("Application query test failed.")

        pglMessages.message(f"Application account ready: {self.databaseUser}")
        pglMessages.message(f"Database ready: {self.databaseName}")
        pglMessages.message(f"Schema ready: {self.databaseSchema} (owner: {self.adminUser})")
        pglMessages.message("Application login verified.")
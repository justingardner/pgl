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
            "application_name": f"pgl:{getpass.getuser()}@{socket.gethostname()}:pid={os.getpid()}"[:63],
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

    @property
    def connectTimeoutSeconds(self):
        return getattr(self, "_connectTimeoutSeconds", self.CONNECT_TIMEOUT_SECONDS)

    @property
    def lockTimeoutSeconds(self):
        return getattr(self, "_lockTimeoutSeconds", 30.0)

    @property
    def ioTimeoutSeconds(self):
        return getattr(self, "_ioTimeoutSeconds", 60)

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
    def _readConfigurationDocument(cls, location, *, filesystem=None, storageOptions=None):
        """Read configuration with explicit filesystem ownership.

        Internally created SFTP filesystems:
            Bypass the instance cache.
            Close their SFTP/SSH resources after reading.
            Retry once if the connection drops during the read.

        Caller-supplied filesystems:
            Are not replaced, closed, or retried.

        Returns filesystem metadata plus the parsed document. An internally
        created SFTP filesystem is already closed when this method returns;
        its configuration attributes remain available for transport inference.
        """
        import errno

        def connectionDropped(error):
            """Recognize transport failures, not authentication or file errors."""
            from paramiko import SSHException
            from paramiko.ssh_exception import AuthenticationException, BadHostKeyException

            if isinstance(error, (AuthenticationException, BadHostKeyException)):
                return False

            if isinstance(error, (EOFError, ConnectionError, TimeoutError)):
                return True

            if isinstance(error, OSError):
                return error.errno in {
                    errno.ECONNRESET,
                    errno.ECONNABORTED,
                    errno.EPIPE,
                    errno.ENOTCONN,
                    errno.ETIMEDOUT,
                }

            if isinstance(error, SSHException):
                message = str(error).lower()
                return (
                    message.startswith("server connection dropped")
                    or message in {"socket is closed", "socket is closed.", "no existing session"}
                )

            return False

        def closeOwnedSFTP(ownedFilesystem):
            """Close the resources used by fsspec's SFTP implementation."""
            for name in ("ftp", "client"):
                resource = getattr(ownedFilesystem, name, None)

                if resource is not None:
                    try:
                        resource.close()
                    except Exception as error:
                        # Do not replace a read error with a cleanup error.
                        pglMessages.warning(
                            f"Could not completely close configuration SFTP "
                            f"resource {name!r} ({type(error).__name__}).",
                            level=0,
                        )

        # Preserve caller ownership and the existing argument validation.
        if filesystem is not None:
            resolvedFilesystem, path, isLocal = cls._resolveConfigurationFile(
                location,
                filesystem=filesystem,
                storageOptions=storageOptions,
            )

            with resolvedFilesystem.open(path, "rt", encoding="utf-8") as file:
                data = json.load(file)

            return resolvedFilesystem, path, isLocal, data

        # Never reuse a cached connection for this short configuration read.
        options = dict(storageOptions or {})
        options["skip_instance_cache"] = True

        for attempt in range(2):
            ownedFilesystem = None
            ownedSFTP = False

            try:
                ownedFilesystem, path, isLocal = cls._resolveConfigurationFile(
                    location,
                    storageOptions=options,
                )

                protocols = ownedFilesystem.protocol
                protocols = (protocols,) if isinstance(protocols, str) else protocols
                isSFTP = any(protocol in {"ssh", "sftp"} for protocol in protocols)

                if isSFTP:
                    from fsspec.implementations.sftp import SFTPFileSystem

                    # Only assume resource ownership details for the known
                    # fsspec implementation and its subclasses.
                    ownedSFTP = isinstance(ownedFilesystem, SFTPFileSystem)

                with ownedFilesystem.open(path, "rt", encoding="utf-8") as file:
                    data = json.load(file)

                return ownedFilesystem, path, isLocal, data

            except Exception as error:
                if not ownedSFTP or not connectionDropped(error):
                    raise

                if attempt == 1:
                    raise RuntimeError(
                        "Could not read PostgreSQL configuration: the SFTP "
                        "connection failed again after reconnecting. "
                        "Check SSH access and retry."
                    ) from error

                pglMessages.warning(
                    "Configuration SFTP connection was lost; reconnecting once...",
                    level=0,
                )

            finally:
                if ownedSFTP:
                    closeOwnedSFTP(ownedFilesystem)

    @classmethod
    def load(cls, filename=None, *, filesystem=None, storageOptions=None):
        """Load configuration and resolve caller-side SSH transport.

        A direct-loopback configuration discovered through SSH/SFTP describes
        a database on the discovery machine. Adapt the returned object to use
        an SSH tunnel to that machine without rewriting the saved file.

        Explicit SSH endpoints and direct non-loopback endpoints are preserved.
        Other remote protocols do not authorize automatic SSH inference.

        Does not open a SQL connection or administer the server.
        """
        import ipaddress
        from urllib.parse import unquote

        if filename is None:
            directory = cls.class_traits()["installDirectory"].default_value
            filename = posixpath.join(directory, cls.CONFIGURATION_FILENAME)

        location = str(filename)
        filesystem, path, isLocal, data = cls._readConfigurationDocument(location, filesystem=filesystem, storageOptions=storageOptions)
        
        if not isinstance(data, dict):
            raise ValueError("Postgres configuration must be a JSON object.")

        if data.pop("__class__", None) != cls.__name__:
            raise ValueError("The file is not a pglPostgres configuration.")

        # Upgrade and validate the saved document before adapting access.
        obj = cls.fromJSONdict(data)
        obj._configurationLoadedRemotely = not isLocal

        protocols = filesystem.protocol
        protocols = (protocols,) if isinstance(protocols, str) else protocols
        discoveredThroughSSH = not isLocal and any(protocol in {"ssh", "sftp"} for protocol in protocols)

        host = obj.connectionHost
        candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host

        try:
            loopback = ipaddress.ip_address(candidate).is_loopback
        except ValueError:
            loopback = candidate.lower() == "localhost"

        inferTransport = discoveredThroughSSH and obj.connectionTransport == "direct" and loopback
        inferEndpoint = discoveredThroughSSH and obj.connectionTransport == "ssh" and not obj.sshHost

        if inferTransport or inferEndpoint:
            url = urlsplit(location)
            sourceOptions = dict(getattr(filesystem, "storage_options", {}) or {})
            sourceOptions.update(storageOptions or {})

            if url.scheme in {"ssh", "sftp"} and url.hostname:
                # Preserve the caller's alias instead of replacing it with
                # a resolved hostname; OpenSSH applies the caller's policy.
                sshHost = url.hostname
                sshUser = unquote(url.username) if url.username is not None else sourceOptions.get("username", "")
                sshPort = url.port if url.port is not None else sourceOptions.get("port", 22)
            else:
                # Support an explicitly supplied SSH/SFTP filesystem with
                # a protocol-free configuration path.
                sshHost = getattr(filesystem, "host", "") or sourceOptions.get("host", "")
                sshUser = sourceOptions.get("username", "")
                sshPort = sourceOptions.get("port", getattr(filesystem, "port", 22))

            if not sshHost:
                raise ValueError("Cannot infer the SSH configuration source hostname.")

            if inferTransport:
                # The discovery endpoint, not unused server-side SSH fields,
                # determines access to this installation.
                obj.connectionTransport = "ssh"
                obj.sshHost = sshHost
                obj.sshUser = sshUser or ""
                obj.sshPort = sshPort
            else:
                # Preserve explicitly configured user/port values when
                # completing an existing SSH configuration.
                obj.sshHost = sshHost

                if not obj.sshUser:
                    obj.sshUser = sshUser or ""

                if obj.sshPort == 22:
                    obj.sshPort = sshPort

            obj._validateValues()

        return obj

    @classmethod
    def fromSettings(cls, settings=None, settingsName=None, *, filesystem=None, storageOptions=None):
        """Discover configuration from databasePath in database settings.

        Missing configuration produces a warning and returns None.
        Invalid configuration and authentication failures remain errors.
        """
        from .pglDatabase import pglDatabase

        resolvedSettings = pglDatabase.getSettings(settings=settings, settingsName=settingsName)
        
        if resolvedSettings is None:
            pglMessages.warning("Could not resolve database settings.")
            return None

        directory = resolvedSettings.databasePath.strip()

        if not directory:
            pglMessages.warning("No databasePath is configured in database settings.")
            return None

        filename = posixpath.join(directory.rstrip("/") + "/", cls.CONFIGURATION_FILENAME)

        try:
            return cls.load(filename, filesystem=filesystem, storageOptions=storageOptions)
        except FileNotFoundError:
            pglMessages.warning("No PostgreSQL configuration was found in the configured databasePath.")
            return None

    # ++++++++++++++++++++++++ saveClientConfiguration ++++++++++++++++++++++
    def saveClientConfiguration(self, filename, *, overwrite=False):
        """Write an explicit caller-local PostgreSQL client configuration.

        Used when configuring analysis clients for an existing installation.
        Unlike save(), this method performs no installation-path normalization,
        default storage-location assignment, or local-server administration.

        Preserves:
            installDirectory as the installation machine's path.
            storageLocations as registered/configured server roots or URLs.
            Caller-specific direct/SSH connection settings.

        Writes only the existing password-free PostgreSQL configuration
        fields. Does not include agent contents, passwords, or runtime handles.

        filename must be an ordinary local path, preferably a dedicated file
        under ~/.pgl/database-clients/<profile>/configuration.json.

        Existing identical configuration is left unchanged.
        A different existing file requires overwrite=True.
        Publication uses a same-directory temporary file and owner-only mode.

        Returns:
            Absolute path of the saved configuration.

        Does not modify this object or connect to PostgreSQL/SFTP.
        Errors propagate. Parent directories may remain after a failed save.

        Usage:
            postgres.saveClientConfiguration("~/.pgl/database-clients/desktop/configuration.json")
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        if type(overwrite) is not bool:
            raise ValueError("overwrite must be a boolean.")

        try:
            location = os.fspath(filename)
        except TypeError as error:
            raise ValueError("filename must be an ordinary local path.") from error

        if not isinstance(location, str) or not location.strip():
            raise ValueError("filename must be a nonempty local path.")

        if any(character in location for character in ("\x00", "\n", "\r")):
            raise ValueError("filename must not contain NUL or newline characters.")

        if "://" in location or "::" in location:
            raise ValueError("Client configuration publication requires an ordinary local path, not a URL.")

        if location.endswith(("/", "\\")):
            raise ValueError("Supply a configuration filename, not only a directory.")

        self._validateValues()

        if self.connectionTransport == "ssh" and not self.sshHost:
            raise ValueError("A locally saved SSH client configuration requires an explicit sshHost.")

        # Construct the exact configuration wrapper expected by load().
        # Do not serialize arbitrary runtime attributes.
        document = {"__class__": type(self).__name__, **self.toJSONdict()}
        content = json.dumps(document, sort_keys=True, indent=4, ensure_ascii=False, allow_nan=False).encode("utf-8")

        target = Path(location).expanduser().absolute()

        if target.is_symlink():
            raise ValueError("Client configuration destination must not be a symlink.")

        if target.exists():
            if not target.is_file():
                raise IsADirectoryError("Client configuration destination is not a regular file.")

            with target.open("rt", encoding="utf-8") as source:
                existing = json.load(source)

            if existing == document:
                pglMessages.message(f"Client configuration is already current: {target}")
                return str(target)

            if not overwrite:
                raise FileExistsError(f"Different client configuration exists at {target}; use overwrite=True only for an intentional replacement.")

        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporaryPath = None

        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".pgl-client-", delete=False) as temporary:
                temporaryPath = Path(temporary.name)
                os.chmod(temporaryPath, 0o600)

                if temporary.write(content) != len(content):
                    raise OSError("Incomplete client configuration write.")

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

        pglMessages.message(f"Client configuration saved: {target}")
        return str(target)

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
            raise ValueError("installDirectory is a server-local path, not an fsspec URL. Use databasePath in databasepg settings for configuration discovery.")

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

    @contextmanager
    def _localConnect(self, *, user=None, database=None, password=None):
        """Open a verified local connection, or yield None on cancellation.

        Used by setup and migrations, independently of client transport.
        An explicitly supplied password is attempted first; otherwise the
        OS credential store is consulted before prompting.

        Newly entered credentials are saved only after validation succeeds.
        Caller operations are outside the authentication retry loop.
        """
        from contextlib import ExitStack

        self._requireLocalConfiguration()
        self._validateValues()

        databaseUser = self.adminUser if user is None else user
        database = self.databaseName if database is None else database
        self._passwordStoreIdentity(user=databaseUser, local=True)

        fromSavedPassword = False

        if password is None:
            password = self._readSavedPassword(databaseUser, local=True)
            fromSavedPassword = password is not None

        with ExitStack() as resources:
            try:
                while True:
                    if password is None:
                        password = self._askPassword(f"PostgreSQL password for {databaseUser}: ")
                        fromSavedPassword = False

                    if password is None:
                        yield None
                        return

                    try:
                        if databaseUser == self.adminUser:
                            connection = self._adminConnect(password, database)
                        else:
                            connection = psycopg.connect(host=self.postgresHost, port=self.postgresPort, dbname=database, user=databaseUser, password=password, connect_timeout=self.connectTimeoutSeconds, sslmode="prefer", autocommit=True)

                    except psycopg.OperationalError as error:
                        passwordRejected = error.sqlstate == "28P01" or "password authentication failed" in str(error).lower()

                        if not passwordRejected:
                            raise

                        if fromSavedPassword:
                            self.clearSavedPassword(user=databaseUser, local=True)

                        pglMessages.warning(f"PostgreSQL rejected the password for {databaseUser!r}. Try again, or press Enter without typing to cancel.", level=0)
                        password = None
                        fromSavedPassword = False
                        continue

                    resources.callback(connection.close)
                    break

                identity = connection.execute("SELECT current_database(), current_user").fetchone()

                if identity != (database, databaseUser):
                    raise RuntimeError(f"Unexpected connection identity: {identity}")

                # Administrator server identity was checked by _adminConnect.
                # Application setup validation also checks schema access.
                if databaseUser != self.adminUser:
                    connection.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(self.databaseSchema)))

                    if connection.execute("SELECT current_schema()").fetchone()[0] != self.databaseSchema:
                        raise RuntimeError(f"Cannot access configured schema: {self.databaseSchema}")

                    if connection.execute("SELECT 42").fetchone() != (42,):
                        raise RuntimeError("Application query test failed.")

                if not fromSavedPassword:
                    self._saveVerifiedPassword(databaseUser, password, local=True)

            finally:
                password = None

            yield connection

    @staticmethod
    def _askPassword(prompt, *, confirm=False):
        """Prompt privately; return None on empty input, Ctrl-C, or EOF."""
        pglMessages.message("Enter password; press Enter without typing to cancel.", emphasize=True)

        try:
            while True:
                password = getpass.getpass(prompt)

                if not password:
                    pglMessages.warning("Password entry cancelled; operation aborted.", level=0)
                    return None

                if any(character in password for character in ("\n", "\r", "\x00")):
                    pglMessages.warning("Passwords must contain no newline or NUL characters. Please try again.", level=0)
                    continue

                if confirm:
                    confirmation = getpass.getpass("Confirm password (Enter to cancel): ")

                    if not confirmation:
                        pglMessages.warning("Password entry cancelled; operation aborted.", level=0)
                        return None

                    if password != confirmation:
                        pglMessages.warning("Passwords do not match. Please try again.", level=0)
                        continue

                return password

        except (KeyboardInterrupt, EOFError):
            pglMessages.warning("Password entry cancelled; operation aborted.", level=0)
            return None

    @staticmethod
    def _passwordStore():
        """Return a supported OS credential store, or None.

        Explicitly reject plaintext, encrypted-file, and unknown backends.
        Password-store unavailability does not prevent a prompted login.
        """
        try:
            import keyring

            backend = keyring.get_keyring()
            backendType = (type(backend).__module__, type(backend).__name__)

            supported = {
                ("keyring.backends.macOS", "Keyring"),
                ("keyring.backends.Windows", "WinVaultKeyring"),
                ("keyring.backends.SecretService", "Keyring"),
            }

            if backendType not in supported:
                raise RuntimeError(f"Unsupported credential backend: {backendType[0]}.{backendType[1]}")

            return backend

        except Exception as error:
            pglMessages.oneTimeWarning(f"Secure password storage is unavailable ({type(error).__name__}). Install keyring and configure macOS Keychain, Windows Credential Manager, or Linux Secret Service. Passwords will not be saved.", level=0)
            return None

    def _passwordStoreIdentity(self, user=None, *, local=False):
        """Identify a credential without storing it in PGL configuration.

        local=True uses the local administration endpoint rather than
        the configured client transport. Both use the same identity when
        the client configuration describes that same local endpoint.
        """
        import hashlib

        databaseUser = self.databaseUser if user is None else user

        if not isinstance(databaseUser, str) or not databaseUser.strip() or "\x00" in databaseUser:
            raise ValueError("Database username must be a nonempty string without NUL.")

        endpoint = {
            "transport": "direct" if local else self.connectionTransport,
            "host": self.postgresHost if local else self.connectionHost,
            "port": self.postgresPort if local else self.connectionPort,
            "database": self.databaseName,
            "ssl_mode": "prefer" if local else self.databaseSSLMode,
        }

        if not local and self.connectionTransport == "ssh":
            endpoint.update({
                "ssh_host": self.sshHost,
                "ssh_port": self.sshPort,
                "ssh_user": self.sshUser or getpass.getuser(),
            })

        encoded = json.dumps(endpoint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        service = "pgl.postgres.v1." + hashlib.sha256(encoded).hexdigest()

        return service, databaseUser

    def _readSavedPassword(self, user, *, local=False):
        """Read a saved credential, or return None without blocking a prompt."""
        service, databaseUser = self._passwordStoreIdentity(user=user, local=local)
        store = self._passwordStore()

        if store is None:
            return None

        try:
            return store.get_password(service, databaseUser) or None
        except Exception as error:
            pglMessages.warning(f"Could not read the saved password for {databaseUser!r} ({type(error).__name__}). Please enter it instead.", level=0)
            return None

    def _saveVerifiedPassword(self, user, password, *, local=False):
        """Save only a credential the caller has already verified or installed."""
        if not isinstance(password, str) or not password:
            raise ValueError("Cannot save an empty password.")

        service, databaseUser = self._passwordStoreIdentity(user=user, local=local)
        store = self._passwordStore()

        if store is None:
            return False

        try:
            store.set_password(service, databaseUser, password)
        except Exception as error:
            pglMessages.warning(f"Operation succeeded, but the password for {databaseUser!r} could not be saved securely ({type(error).__name__}).", level=0)
            return False

        pglMessages.message(f"Password saved in the OS credential store for {databaseUser!r}.")
        return True

    def clearSavedPassword(self, *, user=None, local=False):
        """Remove a saved credential without changing the PostgreSQL password.

        Defaults to the application user's client credential.
        Use user=self.adminUser, local=True for local administration.
        Returns True if removed or already absent, otherwise False.
        """
        self._validateValues()
        service, databaseUser = self._passwordStoreIdentity(user=user, local=local)
        store = self._passwordStore()

        if store is None:
            return False

        try:
            if store.get_password(service, databaseUser) is None:
                pglMessages.message(f"No saved PostgreSQL password for {databaseUser!r}.")
                return True

            store.delete_password(service, databaseUser)

        except Exception as error:
            pglMessages.warning(f"Could not clear the saved PostgreSQL password ({type(error).__name__}).", level=0)
            return False

        pglMessages.message(f"Cleared the saved PostgreSQL password for {databaseUser!r}.")
        return True

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
    # ++++++++++++++++++++++++ getSSHAccessConfiguration +++++++++++++++++++++
    def getSSHAccessConfiguration(self):
        """Resolve caller-side SSH settings for database payload access.

        Called by pglDatabase when a server-local storage root must be
        accessed through the same installation machine as the SQL tunnel.

        Uses OpenSSH's configuration evaluation rather than independently
        interpreting aliases with Paramiko. The explicit SSH port and optional
        username match those supplied by pglPostgresTransport._ssh().

        Returns:
            A new runtime dictionary containing the resolved endpoint,
            identity-file paths, known-hosts-file paths, and relevant
            authentication/security settings.

        Does not:
            Open an SSH or PostgreSQL connection.
            Read private-key contents.
            Save configuration or credentials.
            Verify that authentication or SFTP access will succeed.

        OpenSSH may read its configuration files and evaluate trusted local
        configuration directives such as Match exec.

        First implementation restrictions:
            No ProxyJump or ProxyCommand.
            No HostKeyAlias.
            No custom IdentityAgent socket.
            No explicitly configured SSH certificate files.
            Standard user/global known_hosts files are supported.

        These restrictions prevent the payload client from silently ignoring
        routing or identity settings used by the SQL tunnel.

        Host-key verification must still be enforced by the filesystem
        connection implementation. Resolving this dictionary alone does
        not authenticate the server.

        Raises:
            ValueError for invalid configuration.
            NotImplementedError for unsupported SSH arrangements.
            RuntimeError for unavailable OpenSSH or failed resolution.
        """
        import os
        import shlex
        import shutil
        import subprocess
        from pathlib import Path

        self._validateValues()

        if self.connectionTransport != "ssh":
            raise ValueError("SSH access configuration requires connectionTransport='ssh'.")

        if not self.sshHost:
            raise ValueError("SSH access configuration requires sshHost.")

        executable = shutil.which("ssh")

        if executable is None:
            raise RuntimeError("OpenSSH was not found.")

        # Match the SQL tunnel's explicit endpoint selection. In particular,
        # sshPort overrides a Port directive in the user's SSH configuration.
        arguments = [
            executable,
            "-G",
            "-T",
            "-p", str(self.sshPort),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"ConnectTimeout={self.connectTimeoutSeconds}",
        ]

        if self.sshUser:
            arguments.extend(["-l", self.sshUser])

        arguments.append(self.sshHost)

        try:
            result = subprocess.run(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=self.connectTimeoutSeconds, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Timed out resolving OpenSSH access configuration.") from error

        if result.returncode != 0:
            # Configuration diagnostics can expose local command arguments
            # or other private configuration details.
            raise RuntimeError(f"OpenSSH configuration resolution failed with exit status {result.returncode}.")

        values = {}

        for line in result.stdout.splitlines():
            parts = line.split(None, 1)

            if len(parts) != 2:
                continue

            name, value = parts
            values.setdefault(name.lower(), []).append(value.strip())

        # ++++++++++++++++++++++++++++ single +++++++++++++++++++++++++++++++
        def single(name, default=None):
            """Read one resolved setting, rejecting conflicting duplicates."""
            entries = values.get(name, [])

            if not entries:
                return default

            if len(set(entries)) != 1:
                raise ValueError(f"OpenSSH returned conflicting values for {name}.")

            return entries[0]

        # ++++++++++++++++++++++++++++ enabled ++++++++++++++++++++++++++++++
        def enabled(name, default):
            """Decode OpenSSH's Boolean output without permissive coercion."""
            value = single(name, default).lower()

            if value in {"yes", "true"}:
                return True

            if value in {"no", "false"}:
                return False

            raise ValueError(f"Unsupported resolved SSH Boolean setting: {name}.")

        # ++++++++++++++++++++++++++ configured +++++++++++++++++++++++++++++
        def configured(name):
            """Identify an explicitly effective nonempty setting."""
            value = single(name)
            return value is not None and value.lower() not in {"", "none"}

        host = single("hostname")
        username = single("user")
        portText = single("port")

        if not host or any(character.isspace() for character in host) or "\x00" in host:
            raise ValueError("OpenSSH did not resolve a valid SSH hostname.")

        if not username or "\x00" in username or any(character.isspace() for character in username):
            raise ValueError("OpenSSH did not resolve a valid SSH username.")

        try:
            port = int(portText)
        except (TypeError, ValueError) as error:
            raise ValueError("OpenSSH did not resolve a valid SSH port.") from error

        if not 1 <= port <= 65535:
            raise ValueError("Resolved SSH port is outside 1–65535.")

        unsupported = []

        for name in ("proxyjump", "proxycommand", "hostkeyalias", "knownhostscommand"):
            if configured(name):
                unsupported.append(name)

        certificateFiles = [value for value in values.get("certificatefile", []) if value.lower() != "none"]

        if certificateFiles:
            unsupported.append("certificatefile")

        identityAgent = single("identityagent")

        if identityAgent is not None and identityAgent.lower() not in {"none", "ssh_auth_sock", "$ssh_auth_sock"}:
            unsupported.append("identityagent")

        if unsupported:
            raise NotImplementedError("Automatic database payload access does not yet support these SSH settings: " + ", ".join(sorted(unsupported)) + ". SQL tunneling and payload access must not silently use different connection policies.")

        # +++++++++++++++++++++++++++ localPath +++++++++++++++++++++++++++++
        def localPath(value, settingName):
            """Expand an already-resolved local SSH configuration path.

            Unsupported token expressions are rejected rather than guessed.
            Paths identify caller-local files, not server storage roots.
            """
            if not value or any(character in value for character in ("\x00", "\n", "\r")):
                raise ValueError(f"Invalid path in SSH setting {settingName}.")

            if "%" in value or "${" in value or value.startswith("$"):
                raise NotImplementedError(f"Unresolved path tokens in SSH setting {settingName} are not supported.")

            return str(Path(value).expanduser().resolve())

        identityFiles = []

        for value in values.get("identityfile", []):
            if value.lower() == "none":
                continue

            # IdentityFile output is one path per line. Do not split it:
            # a legitimate local filename can contain spaces.
            path = localPath(value, "identityfile")

            if path not in identityFiles:
                identityFiles.append(path)

        knownHostsFiles = []

        for settingName in ("userknownhostsfile", "globalknownhostsfile"):
            for value in values.get(settingName, []):
                # These settings can contain multiple quoted paths.
                for item in shlex.split(value):
                    if item.lower() == "none":
                        continue

                    path = localPath(item, settingName)

                    if path not in knownHostsFiles:
                        knownHostsFiles.append(path)

        if not knownHostsFiles:
            raise ValueError("No known_hosts files are configured for strict SSH payload verification.")

        # Do not copy agent credentials. Only report whether the caller's
        # current process has an agent available and the SSH policy permits it.
        agentPermitted = identityAgent is None or identityAgent.lower() != "none"
        agentAvailable = agentPermitted and bool(os.environ.get("SSH_AUTH_SOCK"))

        return {
            "version": 1,
            "host": host,
            "port": port,
            "username": username,
            "connectTimeoutSeconds": self.connectTimeoutSeconds,
            "identityFiles": identityFiles,
            "knownHostsFiles": knownHostsFiles,
            "identitiesOnly": enabled("identitiesonly", "no"),
            "publicKeyAuthentication": enabled("pubkeyauthentication", "yes"),
            "agentAvailable": agentAvailable,
            "strictHostKeyChecking": True,
        }

    @contextmanager
    def connect(self, *, user=None, sslRootCert=None, rememberPassword=True, useSavedPassword=True, password=None):
        """Open a client connection, or yield None on password-entry cancellation.

        Without an explicit password:
            Uses the existing saved-password and interactive-prompt behavior.
            Rejected saved passwords are removed.

        With an explicit password:
            Attempts that password once.
            Never reads or removes an existing saved password.
            Never prompts for a PostgreSQL password or retries authentication.
            Authentication failures propagate.

        rememberPassword controls saving after successful login and
        database/schema validation. Set it False when the caller will
        explicitly store the credential after connection cleanup.

        Connections and SSH tunnels close on exit.
        SSH authentication retains its existing transport behavior.
        """
        from contextlib import ExitStack

        self._validateValues()

        if not isinstance(rememberPassword, bool) or not isinstance(useSavedPassword, bool):
            raise ValueError("rememberPassword and useSavedPassword must be True or False.")

        explicitPassword = password is not None

        if explicitPassword:
            if not isinstance(password, str) or not password:
                raise ValueError("An explicit password must be a nonempty string.")

            if any(character in password for character in ("\n", "\r", "\x00")):
                raise ValueError("Passwords must contain no newline or NUL characters.")

        if self.connectionTransport == "ssh" and not self.sshHost:
            raise RuntimeError("No SSH hostname is available. Load configuration from an ssh:// URL or specify sshHost.")

        databaseUser = self.databaseUser if user is None else user
        service, databaseUser = self._passwordStoreIdentity(user=databaseUser)

        if getattr(self, "_configurationLoadedRemotely", False):
            if self.connectionTransport == "direct" and self.connectionHost.lower() in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError("Remote configuration points directly to loopback. Configure an SSH transport or a directly reachable database endpoint.")

        store = self._passwordStore() if rememberPassword or (useSavedPassword and not explicitPassword) else None
        fromSavedPassword = False

        if not explicitPassword and store is not None and useSavedPassword:
            try:
                password = store.get_password(service, databaseUser)
                fromSavedPassword = bool(password)
            except Exception as error:
                pglMessages.warning(f"Could not read the saved PostgreSQL password ({type(error).__name__}). Please enter it instead.", level=0)
                password = None

        with ExitStack() as resources:
            try:
                while True:
                    if not password:
                        password = self._askPassword(f"PostgreSQL password for {databaseUser}: ")
                        fromSavedPassword = False

                    if password is None:
                        yield None
                        return

                    try:
                        connection = resources.enter_context(
                            pglPostgresTransport.connect(
                                self,
                                user=databaseUser,
                                password=password,
                                sslRootCert=sslRootCert,
                            )
                        )
                    except psycopg.OperationalError as error:
                        # Explicit credentials get exactly one attempt.
                        # Do not delete an existing credential or prompt.
                        if explicitPassword:
                            raise

                        passwordRejected = error.sqlstate == "28P01" or "password authentication failed" in str(error).lower()

                        if not passwordRejected:
                            raise

                        if fromSavedPassword:
                            try:
                                store.delete_password(service, databaseUser)
                            except Exception as deleteError:
                                pglMessages.warning(f"Could not remove the rejected saved password ({type(deleteError).__name__}).", level=0)

                            pglMessages.warning(f"PostgreSQL rejected the saved password for {databaseUser!r}. Enter a replacement, or press Enter without typing to cancel.", level=0)
                        else:
                            pglMessages.warning(f"PostgreSQL rejected the password for {databaseUser!r}. Try again, or press Enter without typing to cancel.", level=0)

                        password = None
                        fromSavedPassword = False
                        continue

                    break

                identity = connection.execute("SELECT current_database(), current_user").fetchone()

                if identity != (self.databaseName, databaseUser):
                    raise RuntimeError(f"Unexpected connection identity: {identity}")

                schemaExists = connection.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

                if schemaExists is None:
                    raise RuntimeError(f"Database schema does not exist: {self.databaseSchema}")

                connection.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(self.databaseSchema)))

                if connection.execute("SELECT current_schema()").fetchone()[0] != self.databaseSchema:
                    raise RuntimeError(f"Cannot access configured schema: {self.databaseSchema}")

                if rememberPassword and store is not None and not fromSavedPassword:
                    try:
                        store.set_password(service, databaseUser, password)
                    except Exception as error:
                        pglMessages.warning(f"Connected successfully, but could not save the password securely ({type(error).__name__}).", level=0)
                    else:
                        pglMessages.message(f"PostgreSQL password saved in the OS credential store for {databaseUser!r}.")

            finally:
                # Drop this local reference; Python does not guarantee
                # secure memory erasure.
                password = None

            # Caller errors remain outside the authentication retry loop.
            yield connection

    def verifyDatabase(self, *, user=None, sslRootCert=None):
        """Verify client access; return False on cancellation or connection failure."""
        try:
            with self.connect(user=user, sslRootCert=sslRootCert) as connection:
                if connection is None:
                    return False

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
    # Inspect database activity
    # ------------------------------------------------------------

    def inspectDatabaseActivity(self, *, user=None, sslRootCert=None):
        """Return other sessions in this database for lock diagnostics.

        Uses the normal direct/SSH client connection.
        Returns None on password-entry cancellation.

        Query visibility depends on the connected role's permissions.
        Does not cancel queries, terminate sessions, or change data.
        """
        from psycopg.rows import dict_row

        statement = """
            SELECT
                a.pid,
                a.backend_start,
                a.usename,
                a.application_name,
                a.client_addr,
                a.state,
                a.xact_start,
                clock_timestamp() - a.xact_start AS transaction_age,
                a.wait_event_type,
                a.wait_event,
                pg_blocking_pids(a.pid) AS blocking_pids,
                EXISTS (
                    SELECT 1
                    FROM pg_locks AS l
                    WHERE l.pid = a.pid
                      AND l.locktype = 'advisory'
                      AND l.granted
                ) AS holds_advisory_lock,
                a.query AS last_query
            FROM pg_stat_activity AS a
            WHERE a.datname = current_database()
              AND a.pid <> pg_backend_pid()
            ORDER BY a.xact_start NULLS LAST, a.pid
        """

        with self.connect(user=user, sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(statement)
                return cursor.fetchall()
    
    
    # ------------------------------------------------------------
    # acquire transaction lock
    # ------------------------------------------------------------ 
    @staticmethod
    def acquireTransactionLock(connection, lockName, *, description="database operation", timeoutSeconds=30.0, pollSeconds=0.25):
        """Acquire a transaction advisory lock with bounded contention waiting.

        Requires an active explicit transaction.
        The acquired lock remains held until that transaction ends.

        On initial contention:
            Report the operation and timeout.
            Inspect the exact advisory lock and display its current holders.
            Remember displayed connection identities for PID-only recovery.

        Lock diagnostics use a savepoint so a diagnostic SQL failure does not
        invalidate the enclosing save transaction.

        Raises TimeoutError on contention deadline. The caller must let that
        exception unwind its transaction before handling it.

        The deadline bounds contention retries, not network stalls inside an
        individual SQL request. No connection is automatically terminated.
        """
        import math
        import time

        from psycopg.pq import TransactionStatus
        from psycopg.rows import dict_row

        from .pglDatabase import pglDatabase

        if not isinstance(lockName, str) or not lockName or "\x00" in lockName:
            raise ValueError("lockName must be a nonempty string without NUL.")

        for name, value in (("timeoutSeconds", timeoutSeconds), ("pollSeconds", pollSeconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")

        if connection.info.transaction_status != TransactionStatus.INTRANS:
            raise RuntimeError("acquireTransactionLock() requires an active explicit transaction.")

        def reportLockOwners():
            """Inspect holders of this exact bigint advisory-lock key."""
            statement = """
                WITH requested AS (
                    SELECT hashtextextended(%s, 0) AS lock_key
                )
                SELECT
                    a.pid,
                    a.backend_start AS "backendStart",
                    a.usename AS "databaseUser",
                    a.application_name AS "clientIdentity",
                    a.client_addr AS "clientAddress",
                    a.state,
                    clock_timestamp() - a.backend_start AS "connectionAge",
                    clock_timestamp() - a.xact_start AS "transactionAge",
                    clock_timestamp() - a.state_change AS "stateAge",
                    CASE
                        WHEN a.state IN (
                            'idle',
                            'idle in transaction',
                            'idle in transaction (aborted)'
                        )
                        THEN clock_timestamp() - a.state_change
                        ELSE NULL
                    END AS "idleTime",
                    a.wait_event_type AS "waitType",
                    a.wait_event AS "waitEvent",
                    pg_blocking_pids(a.pid) AS "blockingPIDs",
                    (
                        SELECT count(*)
                        FROM pg_locks AS held
                        WHERE held.pid = a.pid
                          AND held.locktype = 'advisory'
                          AND held.granted
                    ) AS "heldAdvisoryLocks",
                    left(a.query, 1000) AS "lastQuery"
                FROM requested AS requested_lock
                JOIN pg_locks AS lock
                  ON lock.locktype = 'advisory'
                 AND lock.objsubid = 1
                 AND lock.classid::bigint =
                     ((requested_lock.lock_key >> 32) & 4294967295::bigint)
                 AND lock.objid::bigint =
                     (requested_lock.lock_key & 4294967295::bigint)
                 AND lock.database = (
                     SELECT oid
                     FROM pg_database
                     WHERE datname = current_database()
                 )
                JOIN pg_stat_activity AS a ON a.pid = lock.pid
                WHERE lock.granted
                  AND a.pid <> pg_backend_pid()
                ORDER BY a.pid
            """

            try:
                # This creates a savepoint within the caller's transaction.
                with connection.transaction():
                    with connection.cursor(row_factory=dict_row) as cursor:
                        cursor.execute(statement, (lockName,))
                        records = cursor.fetchall()

            except Exception as error:
                pglMessages.warning(f"Could not inspect the lock owner ({type(error).__name__}). The bounded lock wait will continue.", level=0)
                return

            if not records:
                pglMessages.message("Lock owner changed or is unavailable to inspection; retrying acquisition.")
                return

            for record in records:
                pglMessages.message(f"Lock holder: {pglDatabase._formatPostgresLock(record)}", wrapText=False)

        started = time.monotonic()
        deadline = started + timeoutSeconds
        reportedWait = False

        while True:
            if reportedWait and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeoutSeconds:g} s waiting for {description}. "
                    "Another database session held a conflicting lock. "
                    "Run pglDatabase.inspectPostgresLocks() to inspect current owners. "
                    "An idle database connection may still be uploading or verifying files."
                )

            acquired = connection.execute("SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))", (lockName,)).fetchone()[0]

            if acquired:
                if reportedWait:
                    pglMessages.message(f"Lock acquired for {description} after {time.monotonic() - started:.1f} s.")

                return

            if not reportedWait:
                pglMessages.message(f"Waiting for {description}: another database session holds a conflicting lock (timeout: {timeoutSeconds:g} s)...")
                reportedWait = True
                reportLockOwners()

            remaining = deadline - time.monotonic()

            if remaining > 0:
                time.sleep(min(pollSeconds, remaining))
                
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

        Retained for compatibility with pglDatabase's local migration runner.
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
        """Initialize local server data without overwriting existing data.

        Reuses a saved administrator password, or asks for a new one.
        Saves a newly entered password only after initdb succeeds.
        Returns False on cancellation, otherwise True.
        """
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
            return True

        if dataDirectory.exists() and any(dataDirectory.iterdir()):
            raise RuntimeError(f"Refusing to initialize a nonempty directory: {dataDirectory}")

        password = self._readSavedPassword(self.adminUser, local=True)
        fromSavedPassword = password is not None

        if password is None:
            password = self._askPassword(f"Choose PostgreSQL administrator password for {self.adminUser}: ", confirm=True)

        if password is None:
            return False

        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as passwordFile:
                passwordFile.write(password + "\n")
                passwordFile.flush()
                self._runCommand([initdb, "-D", dataDirectory, "-U", self.adminUser, "--encoding=UTF8", "--auth=scram-sha-256", f"--pwfile={passwordFile.name}"])

            if not fromSavedPassword:
                self._saveVerifiedPassword(self.adminUser, password, local=True)

        finally:
            password = None

        pglMessages.message(f"Initialized PostgreSQL data directory: {dataDirectory}")
        return True

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


    # ------------------------------------------------------------
    # Local role/database/schema creation
    # ------------------------------------------------------------

    def createDatabase(self, *, resetApplicationPassword=False):
        """Create the role/database; return False on cancellation or connection failure."""
        try:
            return self._createDatabase(resetApplicationPassword=resetApplicationPassword)
        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not complete database setup. Check the server and credentials. Earlier setup steps may have completed; rerun after resolving the issue.\n{error}")
            return False
        
    def _createDatabase(self, *, resetApplicationPassword=False):
        """Create local role/database/schema and verify application access.

        Reuses saved credentials. Explicit password resets always prompt.
        New application credentials are saved only after successful login.
        CREATE DATABASE cannot be transactional; partial setup may persist.
        """
        self._requireLocalConfiguration()
        self._validateValues()

        if not isinstance(resetApplicationPassword, bool):
            raise ValueError("resetApplicationPassword must be True or False.")

        applicationPassword = None

        try:
            with self._localConnect(database="postgres") as connection:
                if connection is None:
                    return False

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

                # Check existing ownership before making changes.
                if owner is not None:
                    with self._localConnect(database=self.databaseName) as databaseConnection:
                        if databaseConnection is None:
                            return False

                        schemaOwner = databaseConnection.execute("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = %s", (self.databaseSchema,)).fetchone()

                        if schemaOwner is not None and schemaOwner[0] != self.adminUser:
                            raise RuntimeError(f"Schema is owned by {schemaOwner[0]!r}, not {self.adminUser!r}.")

                settingPassword = role is None or resetApplicationPassword

                if settingPassword:
                    # Rebuilding a missing role may reuse its saved password.
                    # An explicit reset instead requires a new choice.
                    if not resetApplicationPassword:
                        applicationPassword = self._readSavedPassword(self.databaseUser, local=True)

                    if applicationPassword is None:
                        applicationPassword = self._askPassword(f"Choose PostgreSQL application password for {self.databaseUser}: ", confirm=True)

                    if applicationPassword is None:
                        return False

                if role is None:
                    statement = sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {}").format(sql.Identifier(self.databaseUser), sql.Literal(applicationPassword))
                    connection.execute(statement)

                elif resetApplicationPassword:
                    statement = sql.SQL("ALTER ROLE {} PASSWORD {}").format(sql.Identifier(self.databaseUser), sql.Literal(applicationPassword))
                    connection.execute(statement)

                if owner is None:
                    connection.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(self.databaseName), sql.Identifier(self.adminUser)))

                connection.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(self.databaseName), sql.Identifier(self.databaseUser)))

            with self._localConnect(database=self.databaseName) as connection:
                if connection is None:
                    return False

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

            # Verifies and saves a new password, or retrieves/retries an
            # existing one. No setup SQL is rerun on authentication failure.
            with self._localConnect(user=self.databaseUser, database=self.databaseName, password=applicationPassword) as connection:
                if connection is None:
                    return False

        finally:
            applicationPassword = None

        pglMessages.message(f"Application account ready: {self.databaseUser}")
        pglMessages.message(f"Database ready: {self.databaseName}")
        pglMessages.message(f"Schema ready: {self.databaseSchema} (owner: {self.adminUser})")
        pglMessages.message("Application login verified.")
        return True

    def verify(self):
        """Verify the local server using saved credentials when available."""
        try:
            with self._localConnect(database="postgres") as connection:
                if connection is None:
                    return False

                version = connection.execute("SHOW server_version").fetchone()[0]

        except psycopg.OperationalError as error:
            pglMessages.warning(f"Could not verify local PostgreSQL: {error}", level=0)
            return False

        pglMessages.message(f"Verified PostgreSQL {version}")
        pglMessages.message(f"Verified data directory: {self.dataDirectory}")
        pglMessages.message(f"Verified endpoint: {self.postgresHost}:{self.postgresPort}")
        return True
    
    def inspectLocks(self, *, user=None, sslRootCert=None):
        """Return advisory-lock holders and blocked sessions in this database.

        Uses normal direct/SSH client access.
        Returns None on connection cancellation.

        A transaction's age or idle state does not establish abandonment.
        Query text and other-role visibility depend on PostgreSQL permissions.

        Does not cancel queries, terminate connections, or change data.
        """
        from psycopg.rows import dict_row

        statement = """
            SELECT
                a.pid,
                a.backend_start AS "backendStart",
                a.usename AS "databaseUser",
                a.application_name AS "clientIdentity",
                a.client_addr AS "clientAddress",
                a.state,
                clock_timestamp() - a.backend_start AS "connectionAge",
                clock_timestamp() - a.xact_start AS "transactionAge",
                clock_timestamp() - a.state_change AS "stateAge",
                CASE
                    WHEN a.state IN ('idle', 'idle in transaction', 'idle in transaction (aborted)')
                    THEN clock_timestamp() - a.state_change
                    ELSE NULL
                END AS "idleTime",
                a.wait_event_type AS "waitType",
                a.wait_event AS "waitEvent",
                pg_blocking_pids(a.pid) AS "blockingPIDs",
                (
                    SELECT count(*)
                    FROM pg_locks AS l
                    WHERE l.pid = a.pid
                      AND l.locktype = 'advisory'
                      AND l.granted
                ) AS "heldAdvisoryLocks",
                left(a.query, 1000) AS "lastQuery"
            FROM pg_stat_activity AS a
            WHERE a.datname = current_database()
              AND a.pid <> pg_backend_pid()
              AND (
                  EXISTS (
                      SELECT 1
                      FROM pg_locks AS l
                      WHERE l.pid = a.pid
                        AND l.locktype = 'advisory'
                  )
                  OR cardinality(pg_blocking_pids(a.pid)) > 0
              )
            ORDER BY a.xact_start NULLS LAST, a.pid
        """

        with self.connect(user=user, sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(statement)
                return cursor.fetchall()
            
    def terminateLockOwner(self, pid, backendStart, *, sslRootCert=None):
        """Administratively terminate a confirmed abandoned advisory-lock owner.

        Uses the configured administrator role over normal client transport.
        Requires PID plus backendStart from inspectLocks() and typed confirmation.

        Refuses an owner with currently blocked followers. Resolve abandoned
        waiting saves first; otherwise releasing the owner could start them.

        Terminates the whole connection, not just one lock. Its uncommitted
        transaction rolls back. Payload files are not removed.

        Returns True for confirmed termination, False for refusal/cancellation,
        or None for administrator connection cancellation.

        PostgreSQL permissions remain authoritative. This method's admin-only
        interface does not revoke PostgreSQL's ability for a role to signal
        its own other sessions through direct SQL.
        """
        from datetime import datetime
        from pprint import pformat

        from psycopg.rows import dict_row

        if type(pid) is not int or pid <= 0:
            raise ValueError("pid must be a positive integer.")

        if not isinstance(backendStart, datetime) or backendStart.tzinfo is None:
            raise ValueError("backendStart must be the timezone-aware value returned by inspectLocks().")

        statement = """
            SELECT
                a.pid,
                a.backend_start AS "backendStart",
                a.usename AS "databaseUser",
                a.application_name AS "clientIdentity",
                a.state,
                clock_timestamp() - a.xact_start AS "transactionAge",
                clock_timestamp() - a.state_change AS "stateAge",
                left(a.query, 1000) AS "lastQuery",
                ARRAY(
                    SELECT follower.pid
                    FROM pg_stat_activity AS follower
                    WHERE a.pid = ANY(pg_blocking_pids(follower.pid))
                    ORDER BY follower.pid
                ) AS "waitingPIDs"
            FROM pg_stat_activity AS a
            WHERE a.pid = %s
              AND a.backend_start = %s
              AND a.datname = current_database()
              AND a.pid <> pg_backend_pid()
              AND EXISTS (
                  SELECT 1
                  FROM pg_locks AS l
                  WHERE l.pid = a.pid
                    AND l.locktype = 'advisory'
                    AND l.granted
              )
        """

        with self.connect(user=self.adminUser, sslRootCert=sslRootCert) as connection:
            if connection is None:
                return None

            if connection.execute("SELECT current_user").fetchone() != (self.adminUser,):
                raise PermissionError("Lock-owner termination requires the configured administrator account.")

            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(statement, (pid, backendStart))
                owner = cursor.fetchone()

            if owner is None:
                pglMessages.warning("That exact connection no longer holds an advisory lock in this database. Nothing was terminated.", level=0)
                return False

            pglMessages.message(pformat(owner, sort_dicts=False), wrapText=False)

            if owner["waitingPIDs"]:
                pglMessages.warning(f"Refusing to terminate this owner while sessions {owner['waitingPIDs']} are queued behind it. Cancel confirmed abandoned waiting saves first, then inspect again.", level=2)
                return False

            pglMessages.warning("Only continue if this connection belongs to an abandoned operation. A database-idle connection may still be uploading files. Termination rolls back uncommitted SQL but does not remove uploaded files.", level=2)
            expected = f"terminate {pid}"

            try:
                response = input(f"Type '{expected}' to terminate this connection; anything else cancels: ")
            except (KeyboardInterrupt, EOFError, OSError):
                pglMessages.message("Termination cancelled.")
                return False

            if response.strip() != expected:
                pglMessages.message("Termination cancelled.")
                return False

            # Fresh autocommit query after confirmation: no diagnostic snapshot
            # or transaction is kept open while waiting for user input.
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(statement, (pid, backendStart))
                current = cursor.fetchone()

            if current is None:
                pglMessages.warning("The connection is gone or no longer holds an advisory lock. Nothing was terminated.", level=0)
                return False

            if current["waitingPIDs"]:
                pglMessages.warning(f"New waiting sessions appeared: {current['waitingPIDs']}. Nothing was terminated; inspect again.", level=2)
                return False

            result = connection.execute(
                """
                SELECT pg_terminate_backend(a.pid, 5000)
                FROM pg_stat_activity AS a
                WHERE a.pid = %s
                  AND a.backend_start = %s
                  AND a.datname = current_database()
                  AND a.pid <> pg_backend_pid()
                  AND EXISTS (
                      SELECT 1
                      FROM pg_locks AS l
                      WHERE l.pid = a.pid
                        AND l.locktype = 'advisory'
                        AND l.granted
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM pg_stat_activity AS follower
                      WHERE a.pid = ANY(pg_blocking_pids(follower.pid))
                  )
                """,
                (pid, backendStart),
            ).fetchone()

        if result is None:
            pglMessages.warning("The termination conditions changed. Nothing was terminated; inspect again.", level=0)
            return False

        if not result[0]:
            pglMessages.warning("Termination was not confirmed. Inspect PostgreSQL activity before retrying.", level=2)
            return False

        pglMessages.message(f"Connection {pid} terminated. Its transaction locks are released; uploaded payload files were not removed.")
        return True
    
    @staticmethod
    def validateTimeouts(*, lockTimeoutSeconds, ioTimeoutSeconds, connectTimeoutSeconds):
        """Validate runtime timeout values without opening connections."""
        import math

        if isinstance(lockTimeoutSeconds, bool) or not isinstance(lockTimeoutSeconds, (int, float)) or not math.isfinite(lockTimeoutSeconds) or lockTimeoutSeconds <= 0:
            raise ValueError("lockTimeoutSeconds must be a finite positive number.")

        if type(ioTimeoutSeconds) is not int or ioTimeoutSeconds <= 0:
            raise ValueError("ioTimeoutSeconds must be a positive integer.")

        if type(connectTimeoutSeconds) is not int or connectTimeoutSeconds < 2:
            raise ValueError("connectTimeoutSeconds must be an integer of at least 2 seconds.")
        
    def withTimeouts(self, *, lockTimeoutSeconds=None, ioTimeoutSeconds=None, connectTimeoutSeconds=None):
        """Return an independent configuration copy with runtime timeouts.

        None preserves this object's effective timeout.
        Copies configuration and remote-discovery status, not connections.

        Overrides are ordinary private attributes, not serialized traits.
        The original object and its saved configuration remain unchanged.
        """
        lockTimeoutSeconds = self.lockTimeoutSeconds if lockTimeoutSeconds is None else lockTimeoutSeconds
        ioTimeoutSeconds = self.ioTimeoutSeconds if ioTimeoutSeconds is None else ioTimeoutSeconds
        connectTimeoutSeconds = self.connectTimeoutSeconds if connectTimeoutSeconds is None else connectTimeoutSeconds

        self.validateTimeouts(lockTimeoutSeconds=lockTimeoutSeconds, ioTimeoutSeconds=ioTimeoutSeconds, connectTimeoutSeconds=connectTimeoutSeconds)

        copied = type(self).fromJSONdict(self.toJSONdict())
        copied._configurationLoadedRemotely = getattr(self, "_configurationLoadedRemotely", False)
        copied._lockTimeoutSeconds = float(lockTimeoutSeconds)
        copied._ioTimeoutSeconds = ioTimeoutSeconds
        copied._connectTimeoutSeconds = connectTimeoutSeconds
        return copied
    
    def acquireTableShareLock(self, connection, tableName, *, description="dataset dependency lock", progress=None, pollSeconds=0.25):
        """Acquire a SHARE table lock with bounded contention waiting.

        Uses this configuration's schema and lockTimeoutSeconds.
        Requires an active explicit transaction.

        Each NOWAIT attempt uses a savepoint. Successful acquisition remains
        held until the outer transaction ends.

        Initial contention reports conflicting granted relation-lock holders.
        Diagnostics are snapshots, not proof of abandonment, and do not cache
        targets for termination.

        Does not change session-wide SQL timeout settings.
        The deadline bounds retries, not network stalls within SQL requests.
        """
        import math
        import time

        from psycopg.pq import TransactionStatus
        from psycopg.rows import dict_row

        from .pglDatabase import pglDatabase

        if not isinstance(tableName, str) or not tableName or "\x00" in tableName:
            raise ValueError("tableName must be a nonempty string without NUL.")

        if isinstance(pollSeconds, bool) or not isinstance(pollSeconds, (int, float)) or not math.isfinite(pollSeconds) or pollSeconds <= 0:
            raise ValueError("pollSeconds must be a finite positive number.")

        if connection.info.transaction_status != TransactionStatus.INTRANS:
            raise RuntimeError("acquireTableShareLock() requires an active explicit transaction.")

        timeoutSeconds = self.lockTimeoutSeconds
        table = sql.Identifier(self.databaseSchema, tableName)
        statement = sql.SQL("LOCK TABLE {} IN SHARE MODE NOWAIT").format(table)
        started = time.monotonic()
        deadline = started + timeoutSeconds
        reportedWait = False

        def reportOwners():
            """Describe granted relation locks conflicting with SHARE."""
            diagnostic = """
                SELECT DISTINCT
                    a.pid,
                    a.backend_start AS "backendStart",
                    a.usename AS "databaseUser",
                    a.application_name AS "clientIdentity",
                    a.state,
                    clock_timestamp() - a.backend_start AS "connectionAge",
                    clock_timestamp() - a.xact_start AS "transactionAge",
                    clock_timestamp() - a.state_change AS "stateAge",
                    a.wait_event_type AS "waitType",
                    a.wait_event AS "waitEvent",
                    pg_blocking_pids(a.pid) AS "blockingPIDs",
                    (
                        SELECT count(*)
                        FROM pg_locks AS held
                        WHERE held.pid = a.pid
                          AND held.locktype = 'advisory'
                          AND held.granted
                    ) AS "heldAdvisoryLocks",
                    l.mode AS "tableLockMode"
                FROM pg_locks AS l
                JOIN pg_stat_activity AS a ON a.pid = l.pid
                JOIN pg_class AS c ON c.oid = l.relation
                JOIN pg_namespace AS n ON n.oid = c.relnamespace
                WHERE l.locktype = 'relation'
                  AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())
                  AND n.nspname = %s
                  AND c.relname = %s
                  AND l.granted
                  AND l.mode IN (
                      'RowExclusiveLock',
                      'ShareUpdateExclusiveLock',
                      'ShareRowExclusiveLock',
                      'ExclusiveLock',
                      'AccessExclusiveLock'
                  )
                  AND a.pid <> pg_backend_pid()
                ORDER BY a.pid
            """

            try:
                with connection.transaction():
                    with connection.cursor(row_factory=dict_row) as cursor:
                        cursor.execute(diagnostic, (self.databaseSchema, tableName))
                        records = cursor.fetchall()
            except Exception as error:
                pglMessages.warning(f"Could not inspect dependency-lock holders ({type(error).__name__}).", level=0)
                return

            if not records:
                pglMessages.message("No conflicting granted holder is currently visible; the lock may have changed or be blocked behind a queued request.")
                return

            for record in records:
                pglMessages.message(f"Table-lock holder: {pglDatabase._formatPostgresLock(record)} | table lock: {record['tableLockMode']}", wrapText=False)

        while True:
            if reportedWait and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out after {timeoutSeconds:g} s waiting for {description}.")

            try:
                with connection.transaction():
                    connection.execute(statement)
            except psycopg.errors.LockNotAvailable:
                if not reportedWait:
                    reportedWait = True

                    if progress is not None:
                        progress(None, f"Waiting for {description} (timeout {timeoutSeconds:g} s)")

                    pglMessages.message(f"Waiting for {description} (timeout: {timeoutSeconds:g} s)...")
                    reportOwners()

                remaining = deadline - time.monotonic()

                if remaining > 0:
                    time.sleep(min(pollSeconds, remaining))
            else:
                if reportedWait:
                    pglMessages.message(f"Acquired {description} after {time.monotonic() - started:.1f} s.")

                return
################################################################
#   filename: pglSSHFilesystem.py
#    purpose: Host-key-verified SFTP filesystem for database
#             payload access using caller-resolved SSH settings
#       date: 2024-06-05
#         by: JLG
################################################################

import io
from pathlib import Path
from threading import RLock
from uuid import uuid4

import paramiko
from fsspec.implementations.sftp import SFTPFileSystem
from fsspec.mapping import FSMap


# =========================================================================
# BINARY FILE ADAPTER
# =========================================================================

class _pglSFTPFile(io.BufferedIOBase):
    """Normalize Paramiko file behavior and serialize channel operations.

    Created by _pglSFTPClient.open().

    All file operations share the filesystem's reentrant lock. This prevents
    separate threads from concurrently consuming responses on one SFTP client.

    Successful Paramiko writes returning None are normalized to byte counts.
    Stored-byte verification remains the publication caller's responsibility.

    Closing this object closes its file handle, not the SSH connection.
    """

    # +++++++++++++++++++++++++++++ __init__ ++++++++++++++++++++++++++++++++
    def __init__(self, handle, mode, lock):
        """Wrap an open binary handle without reading or writing."""
        super().__init__()
        self._handle = handle
        self._mode = mode
        self._ioLock = lock

    # +++++++++++++++++++++++++++++ readable ++++++++++++++++++++++++++++++++
    def readable(self):
        """Report whether the requested mode permits reads."""
        return "r" in self._mode or "+" in self._mode

    # +++++++++++++++++++++++++++++ writable ++++++++++++++++++++++++++++++++
    def writable(self):
        """Report whether the requested mode permits writes."""
        return any(character in self._mode for character in ("w", "a", "x", "+"))

    # +++++++++++++++++++++++++++++ seekable ++++++++++++++++++++++++++++++++
    def seekable(self):
        """SFTP files support explicit positioning."""
        return True

    # +++++++++++++++++++++++++++++++ read ++++++++++++++++++++++++++++++++++
    def read(self, size=-1):
        """Read bytes while holding the shared SFTP lock."""
        with self._ioLock:
            self._checkClosed()

            if not self.readable():
                raise io.UnsupportedOperation("File is not readable.")

            return self._handle.read(size)

    # ++++++++++++++++++++++++++++++ read1 ++++++++++++++++++++++++++++++++++
    def read1(self, size=-1):
        """Provide the binary interface used by text wrappers."""
        return self.read(size)

    # +++++++++++++++++++++++++++++ readinto ++++++++++++++++++++++++++++++++
    def readinto(self, buffer):
        """Fill a writable buffer and return the number of bytes read."""
        view = memoryview(buffer).cast("B")
        content = self.read(len(view))
        view[:len(content)] = content
        return len(content)

    # +++++++++++++++++++++++++++++++ write +++++++++++++++++++++++++++++++++
    def write(self, content):
        """Write bytes under the shared lock and return an ordinary byte count."""
        content = bytes(memoryview(content))

        with self._ioLock:
            self._checkClosed()

            if not self.writable():
                raise io.UnsupportedOperation("File is not writable.")

            written = self._handle.write(content)

            if written is None:
                return len(content)

            if type(written) is not int or not 0 <= written <= len(content):
                raise OSError("SFTP returned an invalid write count.")

            return written

    # +++++++++++++++++++++++++++++++ seek ++++++++++++++++++++++++++++++++++
    def seek(self, offset, whence=io.SEEK_SET):
        """Position the remote file and return its resulting offset."""
        with self._ioLock:
            self._checkClosed()
            self._handle.seek(offset, whence)
            return self._handle.tell()

    # +++++++++++++++++++++++++++++++ tell ++++++++++++++++++++++++++++++++++
    def tell(self):
        """Return the current remote-file offset."""
        with self._ioLock:
            self._checkClosed()
            return self._handle.tell()

    # +++++++++++++++++++++++++++++++ flush +++++++++++++++++++++++++++++++++
    def flush(self):
        """Flush the remote handle under the shared lock."""
        with self._ioLock:
            self._checkClosed()
            self._handle.flush()

    # +++++++++++++++++++++++++++++++ close +++++++++++++++++++++++++++++++++
    def close(self):
        """Close deterministically, including when flushing fails.

        The lock is reentrant because IOBase.close() calls flush().
        """
        with self._ioLock:
            if self.closed:
                return

            try:
                super().close()
            finally:
                self._handle.close()


# =========================================================================
# SERIALIZED SFTP CLIENT FACADE
# =========================================================================

class _pglSFTPClient:
    """Serialize the SFTP operations used by inherited fsspec methods.

    Installed as pglSSHFilesystem.ftp.

    Wrapping only filesystem._open() would leave stat(), directory listing,
    mkdir(), remove(), rename(), and other inherited operations unprotected.

    Complete synchronous client calls run under one shared lock.
    Returned file handles use the same lock for subsequent operations.

    Raw channel access and lazy directory iterators are handled explicitly.
    This is an internal facade, not a general replacement for Paramiko's API.
    """

    # +++++++++++++++++++++++++++++ __init__ ++++++++++++++++++++++++++++++++
    def __init__(self, client, lock):
        """Retain the connected SFTP client and its filesystem-owned lock."""
        self._client = client
        self._ioLock = lock

    # +++++++++++++++++++++++++++++++ open ++++++++++++++++++++++++++++++++++
    def open(self, filename, mode="rb", bufsize=-1):
        """Open a file with exclusive-mode and write-count compatibility.

        Disable file buffering and do not enable Paramiko write pipelining.
        This keeps each operation synchronous while it holds the shared lock.
        """
        remoteMode = mode.replace("x", "wx") if "x" in mode else mode

        with self._ioLock:
            handle = self._client.open(filename, remoteMode, bufsize=0)
            return _pglSFTPFile(handle, mode, self._ioLock)

    # +++++++++++++++++++++++++++++++ file ++++++++++++++++++++++++++++++++++
    def file(self, filename, mode="rb", bufsize=-1):
        """Provide Paramiko's file() alias without exposing an unwrapped handle."""
        return self.open(filename, mode=mode, bufsize=bufsize)

    # +++++++++++++++++++++++++++ listdir_iter ++++++++++++++++++++++++++++++
    def listdir_iter(self, path=".", read_aheads=50):
        """Return a local iterator after synchronously collecting the listing.

        Do not return a remote generator that could perform I/O outside the
        lock. This uses memory proportional to one directory's entries.
        """
        with self._ioLock:
            entries = self._client.listdir_attr(path)

        return iter(entries)

    # +++++++++++++++++++++++++++ __getattr__ +++++++++++++++++++++++++++++++
    def __getattr__(self, name):
        """Delegate synchronous client methods under the shared lock.

        Transfer helpers disable prefetch because its background requests
        would otherwise escape the serialized operation.

        Private state and direct channel access are intentionally unavailable
        through this facade.
        """
        if name.startswith("_") or name == "get_channel":
            raise AttributeError(name)

        target = getattr(self._client, name)

        if not callable(target):
            raise AttributeError(f"Raw SFTP client attribute is not exposed: {name}")

        # ++++++++++++++++++++++++++++ invoke ++++++++++++++++++++++++++++++
        def invoke(*args, **kwargs):
            """Run one synchronous SFTP client operation."""
            if name in {"get", "getfo"}:
                # Both Paramiko signatures place prefetch after callback:
                # get(remote, local, callback=None, prefetch=True, ...)
                arguments = list(args)
                kwargs = dict(kwargs)

                if len(arguments) >= 4:
                    arguments[3] = False
                    kwargs.pop("prefetch", None)
                else:
                    kwargs["prefetch"] = False

                args = tuple(arguments)

            with self._ioLock:
                return target(*args, **kwargs)

        return invoke


# =========================================================================
# DASK-TOKENIZABLE ZARR MAPPING
# =========================================================================

class _pglSSHMap(FSMap):
    """Give an SSH-backed mapping a connection-free Dask identity.

    Prevents task-name generation from falling back to pickle reconstruction
    of the embedded filesystem, which would open another SSH connection.

    Tokens describe runtime readers, not immutable scientific contents.
    Process/distributed transfer of live readers is not supported here.
    """

    # +++++++++++++++++++++++++ __dask_tokenize__ +++++++++++++++++++++++++++
    def __dask_tokenize__(self):
        """Describe this mapping without pickling it or accessing storage."""
        return (
            "pglSSHMap",
            self.fs.__dask_tokenize__(),
            self.root,
            self.check,
            self.create,
            self.missing_exceptions,
        )


# =========================================================================
# VERIFIED SSH FILESYSTEM
# =========================================================================

class pglSSHFilesystem(SFTPFileSystem):
    """Provide verified SFTP access with serialized channel operations.

    Constructed by pglDatabase's caller-side backend resolver.

    Security:
        Load configured known-hosts files.
        Reject unknown or changed host keys.
        Do not automatically add keys or prompt for authentication.
        This is not a complete implementation of every OpenSSH policy.

    Authentication:
        Use resolved identity files and the caller's available SSH agent.
        Encrypted keys should be unlocked in the agent before use.

    Concurrency:
        Supported fsspec operations and returned file handles share one lock.
        Threaded Dask execution is permitted, but network I/O on this
        filesystem is serialized.
        Separate filesystem instances have independent locks/connections.
        Multiprocessing/distributed transfer of live readers is not supported.

    Timeouts:
        ioTimeoutSeconds bounds blocking channel reads/writes.
        It does not bound a complete transfer, computation, or lock wait.
        Timeout failures propagate; no transparent reconnect is attempted.

    Lifetime:
        Keep the filesystem open while deferred readers use it.
        close() invalidates those readers.
        Never close it concurrently with active scientific I/O.
    """

    cachable = False

    # +++++++++++++++++++++++++++++ __init__ ++++++++++++++++++++++++++++++++
    def __init__(self, accessConfiguration, *, ioTimeoutSeconds=60):
        """Validate runtime configuration and open a verified SFTP connection.

        accessConfiguration comes from
        pglPostgres.getSSHAccessConfiguration().

        Opens SSH/SFTP, not PostgreSQL. Does not modify registered roots,
        scientific metadata, or authentication files.
        """
        expectedFields = {
            "version",
            "host",
            "port",
            "username",
            "connectTimeoutSeconds",
            "identityFiles",
            "knownHostsFiles",
            "identitiesOnly",
            "publicKeyAuthentication",
            "agentAvailable",
            "strictHostKeyChecking",
        }

        if not isinstance(accessConfiguration, dict) or set(accessConfiguration) != expectedFields:
            raise ValueError("Invalid SSH access configuration fields.")

        if type(ioTimeoutSeconds) is not int or ioTimeoutSeconds <= 0:
            raise ValueError("ioTimeoutSeconds must be a positive integer.")

        access = dict(accessConfiguration)

        if type(access["version"]) is not int or access["version"] != 1:
            raise ValueError("Unsupported SSH access configuration version.")

        for name in ("host", "username"):
            value = access[name]

            if not isinstance(value, str) or not value or any(character.isspace() for character in value) or "\x00" in value:
                raise ValueError(f"Invalid SSH access setting: {name}.")

        if type(access["port"]) is not int or not 1 <= access["port"] <= 65535:
            raise ValueError("Invalid SSH port.")

        if type(access["connectTimeoutSeconds"]) is not int or access["connectTimeoutSeconds"] <= 0:
            raise ValueError("SSH timeout must be a positive integer.")

        for name in ("identitiesOnly", "publicKeyAuthentication", "agentAvailable", "strictHostKeyChecking"):
            if type(access[name]) is not bool:
                raise ValueError(f"SSH setting {name} must be a boolean.")

        if not access["strictHostKeyChecking"]:
            raise ValueError("SSH payload access requires strict host-key checking.")

        if not access["publicKeyAuthentication"]:
            raise NotImplementedError("SSH payload access currently requires public-key authentication.")

        if access["identitiesOnly"]:
            raise NotImplementedError("IdentitiesOnly=True requires explicit agent-key filtering and is not supported by this adapter.")

        for name in ("identityFiles", "knownHostsFiles"):
            paths = access[name]

            if not isinstance(paths, list) or any(not isinstance(path, str) or not path or "\x00" in path for path in paths):
                raise ValueError(f"Invalid SSH path collection: {name}.")

            access[name] = list(paths)

        self._accessConfiguration = access
        self._ioTimeoutSeconds = ioTimeoutSeconds
        self._ioLock = RLock()
        self._pglDaskToken = uuid4().hex
        self._pglClosed = False
        self.client = None
        self.ftp = None

        # The superclass calls our _connect(), not its AutoAddPolicy path.
        super().__init__(access["host"])

    # +++++++++++++++++++++++++ __dask_tokenize__ +++++++++++++++++++++++++++
    def __dask_tokenize__(self):
        """Return a runtime token without serializing credentials or reconnecting."""
        return ("pglSSHFilesystem", self._pglDaskToken)

    # ++++++++++++++++++++++++++++ get_mapper +++++++++++++++++++++++++++++++
    def get_mapper(self, root="", check=False, create=False, missing_exceptions=None):
        """Return a Zarr mapping that borrows this filesystem's lifetime.

        Task tokenization performs no I/O.
        Explicit check/create requests retain ordinary FSMap behavior.
        """
        if self._pglClosed:
            raise ValueError("SSH filesystem is closed.")

        return _pglSSHMap(root, self, check=check, create=create, missing_exceptions=missing_exceptions)

    # +++++++++++++++++++++++++++++ _connect ++++++++++++++++++++++++++++++++
    def _connect(self):
        """Verify the host, authenticate, and install serialized SFTP access.

        Called by construction. Failures close the newly created SSH client.
        Known-host marker directives and certificate identities remain
        unsupported rather than being silently ignored.
        """
        access = self._accessConfiguration
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

        try:
            loadedFiles = 0

            for filename in access["knownHostsFiles"]:
                path = Path(filename)

                if not path.exists():
                    continue

                if not path.is_file():
                    raise ValueError("A configured known-hosts path is not a regular file.")

                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()

                    if stripped and not stripped.startswith("#") and stripped.startswith("@"):
                        raise NotImplementedError("Known-hosts marker directives require additional support; this adapter refuses to ignore them.")

                client.load_host_keys(str(path))
                loadedFiles += 1

            if not loadedFiles:
                raise ValueError("No configured known-hosts file could be loaded.")

            keyFiles = []

            for filename in access["identityFiles"]:
                path = Path(filename)

                if not path.is_file():
                    continue

                if path.name.endswith("-cert.pub") or Path(str(path) + "-cert.pub").exists():
                    raise NotImplementedError("SSH certificate identities are not supported by this adapter.")

                keyFiles.append(str(path))

            timeout = access["connectTimeoutSeconds"]
            client.connect(hostname=access["host"], port=access["port"], username=access["username"], key_filename=keyFiles or None, allow_agent=access["agentAvailable"], look_for_keys=False, timeout=timeout, banner_timeout=timeout, auth_timeout=timeout, channel_timeout=timeout)

            transport = client.get_transport()

            if transport is None or not transport.is_active():
                raise OSError("SSH transport is not active after authentication.")

            transport.set_keepalive(15)

            # Set the channel timeout before SFTP's initial protocol exchange,
            # not just after open_sftp() has completed.
            channel = transport.open_session(timeout=timeout)
            channel.settimeout(self._ioTimeoutSeconds)
            channel.invoke_subsystem("sftp")
            sftp = paramiko.SFTPClient(channel)

            self.client = client
            self.ftp = _pglSFTPClient(sftp, self._ioLock)

        except BaseException:
            client.close()
            raise

    # +++++++++++++++++++++++++++++++ _open +++++++++++++++++++++++++++++++++
    def _open(self, path, mode="rb", block_size=None, **kwargs):
        """Open a binary file protected by the shared SFTP operation lock.

        Used by fsspec, including its text wrappers.
        Filesystem transactions are unsupported; database coordinators manage
        their own staging and publication.

        Exclusive x modes retain exclusive-create protection.
        No fallback to overwrite mode is attempted.
        """
        with self._ioLock:
            if self._pglClosed:
                raise ValueError("SSH filesystem is closed.")

            if kwargs.get("autocommit", True) is False:
                raise NotImplementedError("Filesystem transactions are not supported by this SSH adapter.")

            supportedModes = {"rb", "wb", "ab", "xb", "rb+", "wb+", "ab+", "xb+", "r+b", "w+b", "a+b", "x+b"}

            if mode not in supportedModes:
                raise ValueError(f"Unsupported binary SFTP file mode: {mode!r}")

            path = self._strip_protocol(path)

            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError("SSH payload paths must be absolute server paths.")

            return self.ftp.open(path, mode, bufsize=0)

    # +++++++++++++++++++++++++++++++ close +++++++++++++++++++++++++++++++++
    def close(self):
        """Close SFTP and SSH after all dependent readers have finished.

        Idempotent. Does not delete files or modify catalog records.
        Acquires the operation lock; it is not a cancellation mechanism for
        active transfers or an alternative to joining worker tasks.
        """
        with self._ioLock:
            if self._pglClosed:
                return

            self._pglClosed = True

            try:
                if self.ftp is not None:
                    self.ftp.close()
            finally:
                if self.client is not None:
                    self.client.close()
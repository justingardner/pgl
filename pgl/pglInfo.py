
################################################################
#   filename: pglInfo.py
#    purpose: Provides package version and repository provenance utilities for pgl
#         by: JLG
#       date: Oct 21, 2026
################################################################
            
import functools
import inspect
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import numpy as np


class pglInfo:
    """Package metadata without graphics or device dependencies."""

    @classmethod
    def getPGLDir(cls):
        pglDir = inspect.getfile(cls)
        pglDir = os.path.dirname(os.path.dirname(pglDir))
        return(Path(pglDir))

    @classmethod
    @functools.lru_cache(maxsize=1)
    def getRepoRevision(cls):
        '''
        Get the github revision        
        '''
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cls.getPGLDir())
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "Unknown"

    @classmethod
    def getRepoDirtyFiles(cls):
        '''
        Get the list of files with uncommitted changes
        '''
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True, cwd=cls.getPGLDir()
            )
            # each line looks like " M pgl/pglDraw.py" or "?? newfile.py"
            return [line[3:] for line in result.stdout.splitlines()]
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return []

    @classmethod
    @functools.lru_cache(maxsize=1)
    def getRepoBranch(cls):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True, cwd=cls.getPGLDir()
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    @classmethod
    def version(cls):
        try:
            v = _pkg_version("pgl")
        except PackageNotFoundError:
            v = "unknown"
        return v

    @classmethod
    def versionDict(cls):
        return {
            "version": cls.version(),
            "github": cls.getRepoRevision(),
            "dirtyFiles": cls.getRepoDirtyFiles(),
            "branch": cls.getRepoBranch(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__
        }

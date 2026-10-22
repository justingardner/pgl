#!/usr/bin/env python3
"""
PGL developer tasks without Make or shell commands.

Usage (from the repository root):
    python dev.py build   # Build native extensions in place (macOS only).
    python dev.py force   # Force recompilation of native extensions (macOS only).
    python dev.py clean   # Remove native build artifacts.
    python dev.py yml     # Regenerate all curated Conda environment YAML files.

Use the Python interpreter from your PGL development environment.
If your Python command is named python3, substitute python3 above.

Environment activation is optional. For example:
    conda run --no-capture-output -n pgl_macos python dev.py build

These tasks do not create environments or install PGL.
For installation, use:
    python install.py analysis
    python install.py mne
    python install.py native   # macOS only.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

# Only remove artifacts belonging to these native extensions.
NATIVE_MODULES = (
    "_resolution",
    "_pglGammaTable",
    "_pglTimestamp",
    "_pglEventListener",
)


def run(command, env=None):
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def build(force=False):
    if sys.platform != "darwin":
        raise RuntimeError(
            "PGL native extension builds require macOS. "
            "Analysis installations do not need this task."
        )

    env = os.environ.copy()
    env["PGL_BUILD_NATIVE"] = "1"

    command = [
        sys.executable,
        str(ROOT / "setup.py"),
        "build_ext",
        "--inplace",
    ]
    if force:
        command.append("--force")

    run(command, env=env)


def clean():
    build_dir = ROOT / "build"

    if build_dir.is_symlink():
        build_dir.unlink()
        print("Removed:", build_dir)
    elif build_dir.exists():
        shutil.rmtree(build_dir)
        print("Removed:", build_dir)

    package_dir = ROOT / "pgl"

    for module in NATIVE_MODULES:
        for extension in (".so", ".pyd"):
            for artifact in package_dir.glob(module + "*" + extension):
                if artifact.is_file() or artifact.is_symlink():
                    artifact.unlink()
                    print("Removed:", artifact)

    print("Native build artifacts cleaned.")


def make_environments():
    run([sys.executable, str(ROOT / "scripts" / "makeEnvironments.py")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=("build", "force", "clean", "yml"),
    )
    args = parser.parse_args()

    if args.task == "build":
        build()
    elif args.task == "force":
        build(force=True)
    elif args.task == "clean":
        clean()
    else:
        make_environments()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print("\nTask failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        print(
            "\nTask failed with exit code {}. See the output above.".format(
                error.returncode
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTask cancelled.", file=sys.stderr)
        sys.exit(130)
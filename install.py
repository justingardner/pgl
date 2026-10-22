#!/usr/bin/env python3
"""Create or update a Conda environment and install PGL."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

PROFILES = {
    "analysis": ("pgl.yml", "pgl", False),
    "mne": ("pgl_mne.yml", "pgl_mne", False),
    "native": ("pgl_macos.yml", "pgl_macos", True),
}


def find_conda():
    # CONDA_EXE can locate Conda even when its executable is not on PATH.
    candidates = [os.environ.get("CONDA_EXE"), "conda"]

    for candidate in candidates:
        if candidate:
            executable = shutil.which(candidate)
            if executable:
                return executable

    raise RuntimeError(
        "Conda was not found. Install Miniforge or another Conda distribution "
        "and run this script from a terminal where Conda is available."
    )


def check_native_requirements():
    if sys.platform != "darwin":
        raise RuntimeError(
            "The native profile requires macOS. "
            "Use the analysis or mne profile on Windows and Linux."
        )

    xcrun = shutil.which("xcrun")
    if xcrun:
        result = subprocess.run(
            [xcrun, "--find", "clang"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return

    raise RuntimeError(
        "Apple's command-line compiler tools were not found. "
        "Run 'xcode-select --install', complete the installation, "
        "then run this script again."
    )


def run(command, env=None):
    # Display only; subprocess receives an argument list, not a shell command.
    print("\nRunning:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        choices=sorted(PROFILES),
        help="analysis, analysis with MNE, or native macOS acquisition",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update an existing environment instead of creating a new one.",
    )
    args = parser.parse_args()

    yaml_name, environment_name, build_native = PROFILES[args.profile]
    yaml_path = ROOT / yaml_name

    if not yaml_path.is_file():
        raise RuntimeError("Environment file not found: {}".format(yaml_path))

    conda = find_conda()

    if build_native:
        check_native_requirements()

    run([
        conda,
        "env",
        "update" if args.update else "create",
        "--name",
        environment_name,
        "--file",
        str(yaml_path),
    ])

    # Apply the build option only to the installation subprocess.
    # This overrides any inherited value without changing the user's shell.
    build_env = os.environ.copy()
    build_env["PGL_BUILD_NATIVE"] = "1" if build_native else "0"

    run([
        conda,
        "run",
        "--no-capture-output",
        "--name",
        environment_name,
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(ROOT),
    ], env=build_env)

    print("\nPGL installed in Conda environment: {}".format(environment_name))
    print(
        "Run Python with: conda run --no-capture-output -n {} python".format(
            environment_name
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print("\nInstallation failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        print(
            "\nInstallation failed with exit code {}. "
            "See the command output above.".format(error.returncode),
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInstallation cancelled.", file=sys.stderr)
        sys.exit(130)
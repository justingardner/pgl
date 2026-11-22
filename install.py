#!/usr/bin/env python3
"""
Install PGL (requires Conda / Miniforge).

Run from the PGL repository folder:

    python install.py all        # Install all environments; skip native on non-Macs.
    python install.py analysis   # General analysis: pgl
    python install.py mne        # Analysis with MNE: pgl_mne
    python install.py native     # macOS graphics/acquisition: pgl_macos

Already installed? Add --update:
    python install.py all --update

On macOS, native builds require Apple's compiler tools:
    xcode-select --install

After installation, activate the environment you want:
    conda activate pgl           # Or pgl_mne / pgl_macos

No activation is needed before installation. Use python3 if needed.
PGL is installed in editable mode; keep this repository folder in place.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

# (yaml, env name, native: "required" | "optional" | "never")
PROFILES = {
    "analysis": ("pgl.yml", "pgl", "never"),
    "mne": ("pgl_mne.yml", "pgl_mne", "never"),
    "mri": ("pgl_mri.yml", "pgl_mri", "optional"),
    "native": ("pgl_macos.yml", "pgl_macos", "required"),
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "profile",
        choices=("all",) + tuple(PROFILES),
        help="Install all environments or choose one profile.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing environments instead of creating new ones.",
    )
    args = parser.parse_args()

    conda = find_conda()
    profiles = list(PROFILES) if args.profile == "all" else [args.profile]
    selectedProfiles = []

    for profile in profiles:
        yamlName, environmentName, buildNative = PROFILES[profile]

        # Skip native profiles only when installing all on Windows/Linux.
        # An explicit native request still reports the platform requirement.
        if args.profile == "all" and buildNative and sys.platform != "darwin":
            print(f"Skipping {environmentName}: requires macOS.", flush=True)
            continue

        yamlPath = ROOT / yamlName
        if not yamlPath.is_file():
            raise RuntimeError(f"Environment file not found: {yamlPath}")

        selectedProfiles.append((yamlPath, environmentName, buildNative))

    # Check prerequisites before creating or updating any environments.
    if any(buildNative for _, _, buildNative in selectedProfiles):
        check_native_requirements()

    for yamlPath, environmentName, buildNative in selectedProfiles:
        print(f"\nInstalling PGL in: {environmentName}", flush=True)

        run([
            conda,
            "env",
            "update" if args.update else "create",
            "--name",
            environmentName,
            "--file",
            str(yamlPath),
        ])

        # Apply the build option only to the installation subprocess.
        # This overrides any inherited value without changing the user's shell.
        buildEnv = os.environ.copy()
        buildEnv["PGL_BUILD_NATIVE"] = "1" if buildNative else "0"

        run([
            conda,
            "run",
            "--no-capture-output",
            "--name",
            environmentName,
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(ROOT),
        ], env=buildEnv)

    print("\nInstallation complete. Activate an environment with:")
    for _, environmentName, _ in selectedProfiles:
        print(f"    conda activate {environmentName}")

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
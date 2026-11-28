#!/usr/bin/env python3
"""
Install PGL (requires Conda / Miniforge).

Run from the PGL repository folder:

    python install.py all        # pgl, pgl_mne, and pgl_mri
    python install.py analysis   # pgl
    python install.py mne        # pgl_mne
    python install.py mri        # pgl_mri

Already installed? Add --update:
    python install.py all --update

On macOS, all profiles include graphics and acquisition support.
Native builds require Apple's compiler tools:
    xcode-select --install

After installation, activate the environment you want:
    conda activate pgl           # Or pgl_mne / pgl_mri

No activation is needed before installation. Use python3 if needed.
PGL is installed in editable mode; keep this repository folder in place.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts.makeEnvironments import renderEnvironment


ROOT = Path(__file__).resolve().parent

PROFILES = {
    "analysis": "pgl",
    "mne": "pgl_mne",
    "mri": "pgl_mri",
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
        choices=("all", *PROFILES, "native"),
        help="Install all environments or choose one profile. On macOS, native is an alias for analysis.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing environments instead of creating new ones.",
    )
    args = parser.parse_args()

    buildNative = sys.platform == "darwin"
    profile = args.profile
    if profile == "native":
        if not buildNative:
            raise RuntimeError("The native alias requires macOS. Use analysis, mne, or mri on Windows and Linux.")
        profile = "analysis"
    environmentNames = list(PROFILES.values()) if profile == "all" else [PROFILES[profile]]

    # Check prerequisites before creating or updating any environments.
    conda = find_conda()
    if buildNative: check_native_requirements()

    # Override any inherited setting in the editable-install subprocess.
    buildEnv = os.environ.copy()
    buildEnv["PGL_BUILD_NATIVE"] = "1" if buildNative else "0"

    with TemporaryDirectory(prefix="pgl-environments-") as temporaryDir:
        for environmentName in environmentNames:
            yamlPath = Path(temporaryDir) / f"{environmentName}.yml"
            yamlPath.write_text(renderEnvironment(environmentName, macOS=buildNative), encoding="utf-8")
            print(f"\nInstalling PGL in {environmentName}", flush=True)

            run([
                conda,
                "env",
                "update" if args.update else "create",
                "--name",
                environmentName,
                "--file",
                str(yamlPath),
            ])
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
    for environmentName in environmentNames:
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

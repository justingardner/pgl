#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import yaml


def CondaPackageName(spec):
    """Convert numpy=2.5.1=... or numpy>=2 into numpy."""
    return re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]


def PipPackageName(spec):
    """Convert package==1.2 or package>=1.2 into package."""
    return re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--python", default="python=3.12")
parser.add_argument("--addConda", action="append", default=[])

args = parser.parse_args()

environment = yaml.safe_load(Path(args.input).read_text())

cleanDependencies = []
seenCondaPackages = set()

for dependency in environment.get("dependencies", []):
    if isinstance(dependency, str):
        packageName = CondaPackageName(dependency)

        # Replace any exported Python pin/build with your desired flexible spec.
        if packageName == "python":
            dependency = args.python
            packageName = "python"
        else:
            dependency = packageName

        if packageName not in seenCondaPackages:
            cleanDependencies.append(dependency)
            seenCondaPackages.add(packageName)

    elif isinstance(dependency, dict) and "pip" in dependency:
        cleanPipPackages = []
        seenPipPackages = set()

        for package in dependency["pip"]:
            # Do not export local editable installs such as:
            # -e file:///Users/justin/proj/pgl
            if package.startswith("-e ") or "file://" in package:
                continue

            packageName = PipPackageName(package)

            if packageName and packageName not in seenPipPackages:
                cleanPipPackages.append(packageName)
                seenPipPackages.add(packageName)

        if cleanPipPackages:
            cleanDependencies.append({"pip": cleanPipPackages})

# Make sure Python exists even if the export did not include it.
if "python" not in seenCondaPackages:
    cleanDependencies.insert(0, args.python)
    seenCondaPackages.add("python")

# Add requested top-level Conda packages, e.g. mne.
for package in args.addConda:
    packageName = CondaPackageName(package)

    if packageName not in seenCondaPackages:
        cleanDependencies.append(package)
        seenCondaPackages.add(packageName)

environment["name"] = args.name
environment["channels"] = ["conda-forge"]
environment["dependencies"] = cleanDependencies
environment.pop("prefix", None)

Path(args.output).write_text(
    yaml.safe_dump(environment, sort_keys=False, default_flow_style=False)
)

print(f"Wrote {args.output}")
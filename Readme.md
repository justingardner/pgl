# PGL

## Setup

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution before proceeding.

### 1. Clone the library

```bash
git clone https://github.com/justingardner/pgl.git pgl
cd pgl
```

Run the remaining installation commands from this directory.

### 2. Choose an environment

| Environment file | Purpose |
|---|---|
| `pgl.yml` | Analysis |
| `pgl_mne.yml` | Analysis with MNE |
| `pgl_macos.yml` | macOS graphics and experiment acquisition |

The YAML files install dependencies. PGL itself is installed separately in the next step.

#### Option A: Analysis

```bash
conda env create -f pgl.yml
conda activate pgl
env PGL_BUILD_NATIVE=0 python -m pip install --no-deps --no-build-isolation -e .
```

#### Option B: Analysis with MNE

```bash
conda env create -f pgl_mne.yml
conda activate pgl_mne
env PGL_BUILD_NATIVE=0 python -m pip install --no-deps --no-build-isolation -e .
```

These commands use macOS/Linux shell syntax. In Windows PowerShell, set the build option before running pip:

```powershell
$env:PGL_BUILD_NATIVE = "0"
python -m pip install --no-deps --no-build-isolation -e .
Remove-Item Env:PGL_BUILD_NATIVE
```

#### Option C: macOS graphics and experiment acquisition

Install Apple's Command Line Tools if they are not already installed:

```bash
xcode-select --install
```

Then create the environment and install PGL with native extensions:

```bash
conda env create -f pgl_macos.yml
conda activate pgl_macos
env PGL_BUILD_NATIVE=1 python -m pip install --no-deps --no-build-isolation -e .
```

Graphics and native acquisition extensions require macOS. Device-specific integrations, including Eyelink, VPixx, and LabJack, may require separately installed vendor SDKs and drivers.

For Eyelink, use SR Research's Python SDK; do not assume that a PyPI package named `pylink` is the correct SDK.

### 3. Allow Accessibility access for keyboard and mouse events

This step is only needed for macOS keyboard/mouse acquisition.

Open **System Settings → Privacy & Security → Accessibility** and enable access for the application running PGL, such as Terminal or Visual Studio Code. Quit and reopen the application after changing permissions if necessary.

If events are not received from VS Code, try launching it from an authorized Terminal:

```bash
/Applications/Visual\ Studio\ Code.app/Contents/MacOS/Code
```

Depending on the macOS version and how Python is launched, VS Code or the actual host application may also need its own permission. Launching from Terminal does not guarantee permission inheritance.


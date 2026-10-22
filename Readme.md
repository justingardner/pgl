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

Analysis environments are intended to be portable. The current setup has been tested on macOS; Linux and Windows validation is pending.

**Current limitation:** on macOS, `pglTimestamp.getSecs()` still requires the native timestamp extension. Analysis imports do not require it, but workflows that call this method do.

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

### 3. Verify installation

For analysis:

```bash
python -c "from pgl import pglStorage, pglDatabase, pglSession, pglDataset, pglPostgres; print('PGL analysis imports succeeded.')"
```

For macOS graphics:

```bash
python -c "from pgl import pgl; print('PGL graphics class imported successfully.')"
```

The graphics check imports the controller class without creating a controller or explicitly opening a display.

### 4. Allow Accessibility access for keyboard and mouse events

This step is only needed for macOS keyboard/mouse acquisition.

Open **System Settings → Privacy & Security → Accessibility** and enable access for the application running PGL, such as Terminal or Visual Studio Code. Quit and reopen the application after changing permissions if necessary.

If events are not received from VS Code, try launching it from an authorized Terminal:

```bash
/Applications/Visual\ Studio\ Code.app/Contents/MacOS/Code
```

Depending on the macOS version and how Python is launched, VS Code or the actual host application may also need its own permission. Launching from Terminal does not guarantee permission inheritance.

## Using an existing environment

Activate the environment before starting Python or a notebook:

```bash
conda activate pgl
```

Use `pgl_mne` or `pgl_macos` instead when appropriate. In notebooks, select the kernel belonging to that environment.

If the environment name already exists, either reuse it or create a new one with an explicit name:

```bash
conda env create -n pgl_analysis_test -f pgl.yml
conda activate pgl_analysis_test
env PGL_BUILD_NATIVE=0 python -m pip install --no-deps --no-build-isolation -e .
```

## Development

PGL is installed in editable mode (`-e .`), so Python source changes do not require reinstalling. Restart Python or the notebook kernel to reliably load changed code.

Run Makefile commands from the repository root:

| Command | Purpose |
|---|---|
| `make` | Build macOS native extensions in place |
| `make force` | Force recompilation of native extensions |
| `make yml` | Regenerate the environment YAML files |
| `make clean` | Remove build outputs and native extension binaries |

To change environment dependencies, edit `scripts/makeEnvironments.py`, then run:

```bash
make yml
```

Alternatively, without Make:

```bash
python scripts/makeEnvironments.py
```

Regenerating YAML files does not change installed environments. To update one:

```bash
conda env update -n pgl -f pgl.yml
```

Use the matching environment name and YAML file for MNE or macOS graphics.

**Note:** editable installations share the repository checkout. Running `make clean` removes its native extension binaries for every environment using that checkout. Rebuild them from your graphics environment when needed.
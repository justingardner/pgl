## Setup

Install [Miniforge](https://github.com/conda-forge/miniforge) or another
Conda distribution.

Run the commands below in a terminal where Python and Conda are available.
On Windows, the Miniforge Prompt is a convenient option. If your system
uses `python3` rather than `python`, substitute that command.

### 1. Clone the library

```text
git clone https://github.com/justingardner/pgl.git pgl
cd pgl
```

### 2. Install

Choose one profile:

| Profile | Purpose | Supported platforms |
|---|---|---|
| `analysis` | Analysis | Windows, Linux, macOS |
| `mne` | Analysis with MNE | Windows, Linux, macOS |
| `native` | Graphics and experiment acquisition | macOS only |

#### Analysis

```text
python install.py analysis
```

#### Analysis with MNE

```text
python install.py mne
```

#### macOS graphics and experiment acquisition

Install Apple's Command Line Tools if they are not already installed:

```text
xcode-select --install
```

After the tools finish installing:

```text
python install.py native
```

The installer creates the selected Conda environment and installs PGL
in editable mode. No environment activation or shell-specific
environment-variable commands are needed.

To update an existing installation, add `--update`:

```text
python install.py analysis --update
```

Use `mne` or `native` instead of `analysis` for those profiles.

### 3. Use the environment

Select the environment in your IDE, or run a Python script through Conda:

```text
conda run --no-capture-output -n pgl python your_script.py
```

For the other profiles, replace `pgl` with `pgl_mne` or `pgl_macos`.

### Native acquisition requirements

Graphics and native acquisition extensions require macOS. Device-specific
integrations, including Eyelink, VPixx, and LabJack, may require separately
installed vendor SDKs and drivers.

For Eyelink, use SR Research's Python SDK; do not assume that a PyPI
package named `pylink` is the correct SDK.

### Accessibility access for keyboard and mouse events

This step is only needed for macOS keyboard/mouse acquisition.

Open **System Settings → Privacy & Security → Accessibility** and enable
access for the application running PGL, such as Terminal or Visual Studio
Code. Quit and reopen the application after changing permissions if needed.

Depending on the macOS version and how Python is launched, the actual
host application may need its own permission. Launching VS Code from an
authorized Terminal does not guarantee permission inheritance.

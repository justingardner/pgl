# PGL

## Setup

```bash
git clone https://github.com/justingardner/pgl.git pgl
cd pgl
pip install -e .
```

### Install Conda Environment
```bash
# adding this solver can speed up environment creation, but is not necessary on all systems
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# must run this to create environment
conda env create -f pgl.yml
```
### Keyboard and mouse events

To get keyboard/mouse events you need to go to System Settings (in Apple menu at top left), choose Privacy & Security then Accessibility and make sure that Terminal.app is turned on.

If you are running VS Code, make sure to **run it from Terminal** instead of by double-clicking on the icon (so that Apple gives it the Accessibility permissions):
 ```bash
 /Applications/Visual\ Studio\ Code.app/Contents/MacOS/Code
```

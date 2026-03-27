# PGL

## Setup

### Install Conda Environment

```bash
conda env create -f pgl.yml

### Keyboard and mouse events

To get keyboard/mouse events you need to go to System Settings (in Apple menu at top left), choose Privacy & Security then Accessibility and make sure that Terminal.app is turned on.

Then when you run VS Code, make sure to run it from Terminal and run Electron (this should start VS Code as normal):
 /Applications/Visual\ Studio\ Code.app/Contents/MacOS/Electron
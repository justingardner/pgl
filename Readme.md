## Setup

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution, then run:

```sh
git clone https://github.com/justingardner/pgl.git
cd pgl
python install.py all
```

Select `pgl`, `pgl_mri`, or `pgl_mne` in your IDE, or run `conda activate pgl`.

Use `analysis`, `mri`, or `mne` in place of `all` to install just one environment. Add `--update` to update an existing setup. On macOS, the installer adds graphics support and tells you how to install Apple’s compiler tools if they’re missing.

PYTHON ?= python

.DEFAULT_GOAL := build

.PHONY: build force clean yml pgl_yml pgl_mne_yml pgl_macos_yml

# Developer-only native extension builds.
# Requires the conditional setup.py discussed earlier.
# Explicitly requesting native builds should fail on unsupported platforms.
build:
	env PGL_BUILD_NATIVE=1 $(PYTHON) setup.py build_ext --inplace

force:
	env PGL_BUILD_NATIVE=1 $(PYTHON) setup.py build_ext --inplace --force

clean:
	rm -rf build
	rm -f pgl/_resolution*.so
	rm -f pgl/_pglGammaTable*.so
	rm -f pgl/_pglTimestamp*.so
	rm -f pgl/_pglEventListener*.so

# Generate curated specifications, not snapshots of installed environments.
yml:
	$(PYTHON) scripts/makeEnvironments.py

pgl_yml pgl_mne_yml pgl_macos_yml: yml
# Makefile
build: pgl/_resolution.m pgl/_pglGammaTable.m pgl/_pglTimestamp.c pgl/_pglEventListener.cpp
	python setup.py build_ext --inplace

force:
	python setup.py build_ext --inplace

clean:
	rm -rf build *.so *.egg-info __pycache__

###############################################
# update of yml file for dependencies
###############################################
.PHONY: refreshPglPinned pgl_yml pgl_mne_yml

# Create the pinned snapshot only if it does not already exist.
pglPinned.yml:
	conda env export -n pgl --from-history > $@

# Explicitly refresh the snapshot from the current pgl environment.
# Only replace it when the exported contents actually changed.
refreshPglPinned:
	conda env export -n pgl --from-history > pglPinned.yml.tmp
	cmp -s pglPinned.yml.tmp pglPinned.yml || mv pglPinned.yml.tmp pglPinned.yml
	rm -f pglPinned.yml.tmp

# Create/update the flexible PGL environment description.
pgl.yml: pglPinned.yml scripts/makeFlexibleEnvironment.py
	python scripts/makeFlexibleEnvironment.py \
		--input pglPinned.yml \
		--output pgl.yml \
		--name pgl \
		--python "python=3.12"

# Create/update the flexible PGL + MNE environment description.
pgl_mne.yml: pglPinned.yml scripts/makeFlexibleEnvironment.py
	python scripts/makeFlexibleEnvironment.py \
		--input pglPinned.yml \
		--output pgl_mne.yml \
		--name pgl_mne \
		--python "python=3.12" \
		--addConda mne

pgl_yml: pgl.yml

pgl_mne_yml: pgl_mne.yml
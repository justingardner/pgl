# Makefile
build: pgl/_resolution.m pgl/_pglGammaTable.m pgl/_pglTimestamp.c pgl/_pglEventListener.cpp
	python setup.py build_ext --inplace

force:
	python setup.py build_ext --inplace

clean:
	rm -rf build *.so *.egg-info __pycache__
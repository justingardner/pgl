# Makefile
build: pgl/_resolution.m
	python setup.py build_ext --inplace

clean:
	rm -rf build *.so *.egg-info __pycache__
import os
import sys

from setuptools import Extension, find_packages, setup


native_setting = os.environ.get("PGL_BUILD_NATIVE", "auto").strip().lower()

if native_setting not in {"auto", "0", "1"}:
    raise RuntimeError("PGL_BUILD_NATIVE must be 'auto', '0', or '1'.")

build_native = (
    sys.platform == "darwin"
    if native_setting == "auto"
    else native_setting == "1"
)

if build_native and sys.platform != "darwin":
    raise RuntimeError(
        "PGL native extensions require macOS. "
        "Use PGL_BUILD_NATIVE=0 for an analysis-only installation."
    )

extensions = []

if build_native:
    import numpy

    extensions = [
        Extension(
            "pgl._resolution",
            sources=["pgl/_resolution.m"],
            extra_compile_args=["-ObjC"],
            extra_link_args=[
                "-framework", "CoreGraphics",
                "-framework", "Cocoa",
                "-framework", "CoreFoundation",
            ],
        ),
        Extension(
            "pgl._pglGammaTable",
            sources=["pgl/_pglGammaTable.m"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-ObjC"],
            extra_link_args=[
                "-framework", "CoreGraphics",
                "-framework", "Cocoa",
                "-framework", "CoreFoundation",
            ],
        ),
        Extension(
            "pgl._pglTimestamp",
            sources=["pgl/_pglTimestamp.c"],
        ),
        Extension(
            "pgl._pglEventListener",
            sources=["pgl/_pglEventListener.cpp"],
            extra_link_args=[
                "-framework", "ApplicationServices",
                "-framework", "Carbon",
            ],
        ),
    ]

setup(
    packages=find_packages(),
    ext_modules=extensions,
)
import numpy
from setuptools import setup, find_packages, Extension

displayInfoExtension = Extension(
    'pgl._resolution', 
    sources=['pgl/_resolution.m'],
    extra_compile_args=['-ObjC'],
    extra_link_args=[
        '-framework', 'CoreGraphics',
        '-framework', 'Cocoa',
        '-framework', 'CoreFoundation'
    ]
)

gammaTableExtension = Extension(
    'pgl._pglGammaTable',
    sources=['pgl/_pglGammaTable.m'],
    include_dirs=[numpy.get_include()], 
    extra_compile_args=['-ObjC'],
    extra_link_args=[
        '-framework', 'CoreGraphics',
        '-framework', 'Cocoa',
        '-framework', 'CoreFoundation'
    ]
)

timestampExtension = Extension(
    'pgl._pglTimestamp',          # module name
    sources=['pgl/_pglTimestamp.m'],  
    extra_compile_args=[],    # no ObjC needed
    extra_link_args=[]        # no extra frameworks needed
)

setup(
    name='pgl',  
    version='0.1.0',
    packages=find_packages(), 
    description='PGL Psychophysics and experiment library',
    author='Justin Gardner',
    author_email='justin@justingardner.net',
    license='MIT',
    python_requires='>=3.9',
    ext_modules=[displayInfoExtension,gammaTableExtension,timestampExtension]
)
from setuptools import setup, find_packages, Extension

displayInfoExtension = Extension(
    'pgl._displayInfo', 
    sources=['pgl/_displayInfo.m'],
    extra_compile_args=['-ObjC'],
    extra_link_args=[
        '-framework', 'CoreGraphics',
        '-framework', 'Cocoa',
        '-framework', 'CoreFoundation'
    ]
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
    ext_modules=[displayInfoExtension]
)
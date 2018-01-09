from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext = Extension('clblaswrap', ['clblaswrap.pyx'],
                include_dirs=[],
                libraries=['OpenCL', 'clBLAS'],
                library_dirs=[])

setup(ext_modules=cythonize(ext))

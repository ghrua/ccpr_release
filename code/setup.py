from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import sys
import pybind11

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        extra_compile_args = []
        if ct == "unix":
            extra_compile_args.append("-std=c++11")
        for ext in self.extensions:
            ext.extra_compile_args = extra_compile_args
        build_ext.build_extensions(self)

cpp_module = Extension(
    "efficient_structure",
    sources=["efficient_structure.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++11"],
)

cython_module = Extension(
    "data_utils_fast",
    sources=["data_utils_fast.pyx"],
    include_dirs=[np.get_include()],
    language="c++",
    extra_compile_args=["-std=c++11"],
)


setup(
    name="efficient_structure",
    version="0.1",
    ext_modules=[cpp_module, cython_module],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False
)
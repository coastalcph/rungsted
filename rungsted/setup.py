from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
from distutils.command.build_clib import build_clib


# TODO Change the setup to be more like the one described here, which handles dependencies between modules better
# https://github.com/cython/cython/wiki/enhancements-distutils_preprocessing


extra_compile_args=['-Wno-deprecated', '-Wno-unused-function', '-Wno-#warnings', '-Wno-deprecated-writable-strings']

setup(
  name='Structured perceptron',
  cmdclass={'build_ext': build_ext, 'build_clib': build_clib},
  ext_modules=[
      Extension('struct_perceptron', sources=['struct_perceptron.pyx'],
                extra_compile_args=extra_compile_args, language='c++'),
      Extension('input', sources=['input.pyx'],
                extra_compile_args=extra_compile_args, language='c++'),
      Extension('decoding', sources=['decoding.pyx'],
                extra_compile_args=extra_compile_args, language='c++'),
      Extension('feat_map', sources=['feat_map.pyx',
                                    'MurmurHash3.cpp'],
                extra_compile_args=extra_compile_args, language='c++')
  ],
  include_dirs = [np.get_include()]
)

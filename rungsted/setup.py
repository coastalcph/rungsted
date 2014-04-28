from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
from distutils.command.build_clib import build_clib



# ext_modules = [Extension('murmurhash', sources=['rungsted/MurmurHash3.cpp',
#                                                 'rungsted/MurmurHash3.h'], language='c++')]
# ext_modules += cythonize(["rungsted/*.pyx"],
#                          language='c++',
#                          define_macros=[("NPY_NO_DEPRECATED_API", None)],
#                          )



# define_macros=[("NPY_NO_DEPRECATED_API", None)]

# setup(
#     cmdclass={'build_ext': build_ext},
#     ext_modules=[Extension("cython_test", sources=["cython_test.pyx", "c_test.cc"])]
# )

# TODO Change the setup to be more like the one described here, which handles dependencies between modules better
# https://github.com/cython/cython/wiki/enhancements-distutils_preprocessing


extra_compile_args=['-Wno-deprecated', '-Wno-unused-function', '-Wno-#warnings', '-Wno-deprecated-writable-strings']


setup(
  name='Structured perceptron',
  cmdclass={'build_ext': build_ext, 'build_clib': build_clib},
  # libraries=[('MurmurHash3', {'sources': ['rungsted/MurmurHash3.cpp'],
  #                             'language': 'c++',
  #                             })],

  # ext_modules=cythonize("rungsted/*.pyx", sources=['rungsted/MurmurHash3.cpp'])
  ext_modules=[
      Extension('rungsted.struct_perceptron', sources=['rungsted/struct_perceptron.pyx'],
                extra_compile_args=extra_compile_args, language='c++',),
      Extension('rungsted.input', sources=['rungsted/input.pyx',
                                  'rungsted/MurmurHash3.cpp'],
                extra_compile_args=extra_compile_args, language='c++'),
      Extension('rungsted.hashing', sources=['rungsted/hashing.pyx',
                                    'rungsted/MurmurHash3.cpp'],
                extra_compile_args=extra_compile_args, language='c++')
  ],
  include_dirs = [np.get_include()]
)

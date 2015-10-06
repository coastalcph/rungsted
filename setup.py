import os
from os.path import splitext
import subprocess
import platform
from setuptools import setup, Extension
import numpy as np
import sys


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


mac_version_str, _, _ = platform.mac_ver()

extra_compile_args = ['-Wno-deprecated', '-Wno-unused-function', '-Wno-#warnings', '-Wno-deprecated-writable-strings',
                      '-std=c++11']

if mac_version_str:
    extra_compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.8'])

cython_modules = [['rungsted/feat_map.pyx', 'rungsted/MurmurHash3.cpp'],
                  ['rungsted/weights.pyx'],
                  ['rungsted/struct_perceptron.pyx'],
                  ['rungsted/input.pyx'],
                  ['rungsted/decoding.pyx'],
                  ['rungsted/corruption.pyx'],
                  ]

pwd = os.path.dirname(__file__)

includes = [os.path.join(pwd, 'rungsted'),
            os.path.join(pwd, 'src')]

ext_modules = []

recreate_cython = os.environ.get('CYTHON') == '1'

for cython_files in cython_modules:
    source_files = []
    lead_path, ext = splitext(cython_files[0])
    module_name = lead_path.replace("/", ".")

    for fname in cython_files:
        if fname.endswith("pyx"):
            cython_fname = fname
            cpp_fname = fname.replace(".pyx", ".cpp")

            # Execute Cython to generate c++ files if necessary
            cpp_file_exists = os.path.exists(cpp_fname)
            cython_file_exists = os.path.exists(cython_fname)

            cpp_older_than_cython = cpp_file_exists and cython_file_exists \
                                    and os.path.getmtime(cython_fname) > os.path.getmtime(cpp_fname)

            if recreate_cython and (not cpp_file_exists or cpp_older_than_cython):
                if cpp_file_exists:
                    os.remove(cpp_fname)

                subprocess.check_call("cython -3 --cplus {} --output-file {} -v".format(cython_fname, cpp_fname), shell=True)

            source_files.append(cpp_fname)
        else:
            source_files.append(fname)

    ext_modules.append(Extension(module_name, sources=source_files, include_dirs=includes,
                       extra_compile_args=extra_compile_args, language='c++'))

setup(
    name="rungsted",
    version="1.2.4",
    author="Anders Johannsen",
    author_email="ajohannsen@hum.ku.dk",
    description=("Rungsted. An efficient HMM-based structured prediction model for sequential labeling tasks, with extras. "),
    keywords="hmm perceptron structured_model",
    packages=['rungsted'],
    entry_points={
        'console_scripts': [
            'rungsted = rungsted.labeler:main',
        ]},
    long_description=read('README.md'),
    url="https://github.com/coastalcph/rungsted",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.4",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=['pandas>=0.16'],

    ext_modules=ext_modules,
    include_dirs=[np.get_include(), 'src', '.', 'rungsted']
)

import os
from os.path import splitext
import subprocess
from setuptools import setup, Extension
import numpy as np

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


extra_compile_args = ['-Wno-deprecated', '-Wno-unused-function', '-Wno-#warnings', '-Wno-deprecated-writable-strings',
                      '-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.8']

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

for cython_files in cython_modules:
    source_files = []
    module_name = None
    for fname in cython_files:
        if fname.endswith("pyx"):
            cython_fname = fname
            fname = fname.replace(".pyx", ".cpp")
            fname_exists = os.path.exists(fname)

            # Force re-generate
            # if fname_exists:
            #     os.remove(fname)
            #     fname_exists = False

            if not fname_exists or (fname_exists and os.path.getmtime(cython_fname) > os.path.getmtime(fname)):
                if fname_exists:
                    os.remove(fname)
                print "calling cython"
                #subprocess.check_call(["cython", "--cplus", cython_fname,
                #                       "--output-file", fname, '-v'], shell=True)

                subprocess.check_call("cython --cplus {} --output-file {} -v".format(cython_fname, fname), shell=True)


            if not module_name:
                lead_path, ext = splitext(cython_fname)
                #module_name = lead_path.split("/")[1]
                module_name = lead_path.replace("/", ".")


        source_files.append(fname)

    ext_modules.append(Extension(module_name, sources=source_files, include_dirs=includes,
                       extra_compile_args=extra_compile_args, language='c++'))

setup(
    name="rungsted",
    version="0.1",
    author="Anders Johannsen",
    author_email="ajohannsen@hum.ku.dk",
    description=("Rungsted. A structured perceptron sequential tagger"),
    keywords="hmm perceptron structured_model",
    packages=['rungsted'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    ext_modules=ext_modules,
    include_dirs = [np.get_include(), 'src', '.', 'rungsted']
)



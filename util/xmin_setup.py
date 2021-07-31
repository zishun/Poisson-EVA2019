from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import platform

ext_modules = [
        Extension("xmin",
                sources=["xmin.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']
                )
    ]

if platform.system() == 'Windows':
    ext_modules = [
        Extension("xmin",
                sources=["xmin.pyx"],
                extra_compile_args=['/openmp'],
                compiler_directives={'language_level' : "3"}
                )
    ]


setup(name="xmin",
      ext_modules=cythonize(ext_modules,compiler_directives={'language_level' : "3"}))

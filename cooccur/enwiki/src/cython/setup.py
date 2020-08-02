from distutils.core import Extension, setup
from Cython.Build import cythonize

ext = Extension(name="cooccur", sources=["cooccur.pyx"])
setup(ext_modules=cythonize(ext, language_level="3"))
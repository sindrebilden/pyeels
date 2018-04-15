#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('pyeels.cpyeels', sources = ['./pyeels/_spectrum/spectrum.c'], extra_compile_args=['-std=c99']) ]

try:
      readmefile = open('README.md','r')
      readme = readmefile.read()
except:
      readme = "Python simulation package for EELS volume plasmons"

setup(
      name = 'PyEELS',
      version = '0.2.1',
      author = "Sindre R. Bilden",
      author_email = "s.r.bilden@fys.uio.no",
      url = 'https://github.com/sindrerb/pyeels',
      description = ("(NB! Still in development) Python package for simulating EELS from band structures"),
      long_description = readme,
      install_requires=['hyperspy', 'pythtb', 'spglib'],
      packages=['pyeels'],
      include_dirs = [np.get_include()],
      ext_modules = ext_modules
      )

#readme.close()

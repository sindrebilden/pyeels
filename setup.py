#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('pyeels.cpyeels', sources = ['./pyeels/_spectrum/spectrum.c'], extra_compile_args=['-std=c99']) ]

#readme = open('README.md','r')

setup(
      name = 'PyEELS',
      version = '1.0.9.dev1',
      author = "Sindre R. Bilden",
      author_email = "s.r.bilden@fys.uio.no",
      url = 'https://github.com/sindrerb/pyeels',
      description = ("Python package for simulating EELS from band structures"),
      #long_description = readme.read(),
      install_requires=['hyperspy', 'pythtb', 'spglib'],
      packages=['pyeels'],
      include_dirs = [np.get_include()],
      ext_modules = ext_modules
      )

#readme.close()
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('pyeels.cpyeels', sources = ['./pyeels/_spectrum/spectrum.c']) ]

setup(
      name = 'PyEELS',
      version = '1.0.0.dev1',
      author = "Sindre R. Bilden",
      author_email = "s.r.bilden@fys.uio.no",
      url = 'https://github.com/sindrerb/pyeels',
      description = ("A collection of functions for simulating EELS"),
      install_requires=['hyperspy', 'pythtb', 'spglib'],
      packages=['pyeels'],
      ext_modules = ext_modules
      )


"""
setup(
      name = '_spectrum',
      version = '1.0',
      author = "Sindre R. Bilden",
      author_email = "s.r.bilden@fys.uio.no",
      description = ("A C-extended code for calulationg EEL-spectra"),
      
      )

"""
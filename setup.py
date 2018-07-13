#!/usr/bin/env python

from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()

setup(
    name = 'stisblazefix',
    url = 'http://www.stsci.edu/instruments/stis/',
    version = '1.1',
    description = 'HST/STIS echelle blaze function correction',
    long_description = long_description,
    author = 'M. Baer, C. Proffitt',
    author_email = 'help@stsci.edu',
    maintainer = 'Charles Proffitt',
    maintainer_email = 'help@stsci.edu',
    license = 'BSD-new',
    keywords = ['astronomy'],
    classifiers = ['Programming Language :: Python',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.6',
                   'Development Status :: 3 - Alpha',
                   'Operating System :: OS Independent',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Topic :: Software Development :: Libraries :: Python Modules'],
    packages = ['stisblazefix'],
    install_requires = ['setuptools', 'numpy>=1.13', 'scipy', 'astropy', 'matplotlib', \
        'lmfit', 'datetime'],
    package_data = {'stisblazefix': ['testdata/*.fits']},
    #entry_points={
    #      'console_scripts': ['stisblazefix = stisblazefix.stisblazefix:fluxfix']},
    )

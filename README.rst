STIS Echelle Blaze Function Correction Script
=============================================

Correction for the HST/STIS echelle blaze function.

See documentation at https://stisblazefix.readthedocs.io

.. image:: https://github.com/spacetelescope/stisblazefix/actions/workflows/ci.yml/badge.svg?branch=master
   :alt: Build Status
   :target: https://github.com/spacetelescope/stisblazefix/actions/workflows/ci.yml

.. image:: https://coveralls.io/repos/github/spacetelescope/stisblazefix/badge.svg?branch=master
   :alt: Test Coverage
   :target: https://coveralls.io/github/spacetelescope/stisblazefix?branch=master

.. image:: https://readthedocs.org/projects/stisblazefix/badge/
   :alt: Documentation Status
   :target: https://stisblazefix.readthedocs.io/en/latest/?badge=latest

This module contains a variety of functions to correct the blaze function in HST STIS
echelle modes. It is intended for use with STIS echelle x1d FITS files.
Most users will be interested in the `fluxfix` function.

The `stisblazefix.py` module contains the following routines:

* `fluxfix` takes a list of x1d FITS files and generates corrected x1f files and diagnostic plots.
* `fluxcorrect` takes a shift to the blaze function and recalculates the flux and error.
* `residcalc` takes an echelle spectrum and calculates the flux residuals for the overlapping region.
* `generateplot` takes an old and a corrected spectrum and generates a diagnostic plot.
* `residfunc` is a wrapper for the `lmfit` minimizer.
* `findshift` calculates the shift to the blaze function that best aligns the spectrum.
* `plotblaze` plots the sensitivity curves for an extracted spectrum.
* `cliprange` is used to set the Y range for some of the plots to exclude outliers.
* `mkdqmask` creates a mask based on the bits set in data quality flags.

IDL version by C. Proffitt (Jan 2017)  
Python adaptation by M. Baer (Jun–Aug 2017)  
Additional fixes by C. Proffitt (Aug–Sep 2017)

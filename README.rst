STIS echelle blaze function correction script
=============================================

Correction for the HST/STIS echelle blaze function.

See documentation at https://stisblazefix.readthedocs.io

|build-status| |coverage| |docs| |conda| 

This module contains a variety of functions to correct the blaze function in HST STIS
echelle modes. It is intended for use with STIS echelle x1d fits files.
Most users will be interested in the fluxfix function.

The stisblazefix.py module contains the following routines:
 * fluxfix takes a list of x1d fits files and generates corrected x1f files and diagnostic plots.
 * fluxcorrect takes a shift to the blaze function and recalculates the flux and error.
 * residcalc takes an echelle spectrum and calculates the flux residuals for the overlapping region.
 * generateplot takes an old and a corrected spectrum and generates a diagnostic plot.
 * residfunc is a wrapper for the lmfit minimizer.
 * findshift calculates the shift to the blaze function that best aligns the spectrum.
 * plotblaze plots the sensitivity curves for an extracted spectra.
 * cliprange is used to set the Y range for some of the plots to exclude outliers
 * mkdqmask creates a mask based on the bits set in data quality flags

IDL version by C. Proffitt Jan 2017 |br|
Python adaptation by M. Baer Jun-Aug 2017 |br|
Additional fixes by C. Proffitt Aug-Sep 2017

.. |br| raw:: html

   <br />

.. |build-status| image:: https://travis-ci.org/spacetelescope/stisblazefix.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/spacetelescope/stisblazefix

.. |coverage| image:: https://coveralls.io/repos/github/spacetelescope/stisblazefix/badge.svg?branch=master
    :target: https://coveralls.io/github/spacetelescope/stisblazefix?branch=master

.. |docs| image:: https://readthedocs.org/projects/stisblazefix/badge/
    :alt: Documentation Status
    :scale: 100%
    :target: http://stisblazefix.readthedocs.io/en/latest/?badge=latest

.. |conda| image:: https://anaconda.org/sean-lockwood/stisblazefix/badges/installer/conda.svg
    :alt: conda install
    :target: https://anaconda.org/sean-lockwood/stisblazefix

#!/usr/bin/env python
'''
To run activate astroconda, then enter:
    nosetests -v test_sbf.py
'''

from numpy.testing import assert_allclose
import tempfile
import os
import numpy as np
import shutil
import glob
from astropy.io import fits
import stisblazefix


def determine_x1dlist(tmpdir):
    ''' Currently:
    x1dlist = ['o3zx02070_x1d.fits', 'o4vt02040_x1d.fits', 'ocb6f1020_x1d.fits', 'ocb6i2020_x1d.fits']
    '''
    
    testdata_str = os.path.join(os.path.expanduser('~'), 'spacetelescope', 'stisblazefix', 'stisblazefix', 'testdata', '*.fits')
    testdata = glob.glob(testdata_str)
    if len(testdata) == 0:
        raise FileError('Couldn\'t find test *_x1d.fits files in {}\nor package default in {}'.format(
            ref_dir, os.path.dirname(testdata_str)))
    testdata.sort()
    
    # Copy data to temporary directory:
    outdata = [os.path.join(tmpdir, os.path.basename(f)) for f in testdata]
    for infile, outfile in zip(testdata, outdata):
        shutil.copy(infile, outfile)
        # gunzip file (and modify outdata to remove ".gz")...
    
    x1d = [f for f in outdata if f.rsplit('_',1)[-1][0:3].lower() == 'x1d']
    x1f = [f for f in outdata if f.rsplit('_',1)[-1][0:3].lower() == 'x1f']
    x1f_noiterate_benchmark = [f for f in outdata if f.rsplit('_',1)[-1][0:9].lower() == 'noiterate']
    
    # Only matching pairs of x1d & x1f files and the no-iteration test file:
    assert len(outdata) == len(x1d) + len(x1f) + len(x1f_noiterate_benchmark)
    
    # Rename x1f files to avoid confusion with test output products:
    x1f_benchmark = [f.rsplit('_',1)[0] + '_x1fbenchmark' + f.rsplit('_',1)[1][3:] for f in x1f]
    for x1f_orig, x1f_new in zip(x1f, x1f_benchmark):
        shutil.move(x1f_orig, x1f_new)
    
    # Input and output filenames corresponding to no-iterate benchmark file:
    x1d_noiterate = [f.rsplit('_',2)[0] + '_x1d.fits' for f in x1f_noiterate_benchmark]
    x1f_noiterate = [f.rsplit('_',1)[0] +     '.fits' for f in x1f_noiterate_benchmark]
    
    return x1d, x1f, x1f_benchmark, x1d_noiterate, x1f_noiterate, x1f_noiterate_benchmark


def compare_results(x1fs, x1fs_benchmark):
    for x1f, x1f_benchmark in zip(x1fs, x1fs_benchmark):
        print(x1f)
        with fits.open(x1f) as output, fits.open(x1f_benchmark) as benchmark:
            for col in ['FLUX', 'ERROR']:
                print('    ', col)
                assert_allclose(output[1].data[col], benchmark[1].data[col], rtol=1e-4, atol=1e-18)

class TestFluxFix(object):
    @classmethod
    def setup_class(cls):
        '''This method is run once for each class before any tests are run.'''
        # Ignore expected divide-by-zero warnings:
        np.seterr(divide='ignore', invalid='ignore')
        
        # Create a temporary directory:
        cls.tmpdir = tempfile.mkdtemp(prefix='tmp_stisblazefix_')
        print('Version:  {}'.format(stisblazefix.__version__))
        print('Temporary directory:  {}'.format(cls.tmpdir))
        
        # Copy input and benchmark data into temporary directory:
        cls.x1ds, cls.x1fs, cls.x1fs_benchmark, \
        cls.x1d_noiterate, cls.x1f_noiterate, cls.x1f_noiterate_benchmark \
            = determine_x1dlist(cls.tmpdir)
    
    @classmethod
    def teardown_class(cls):
        '''This method is run once for each class after all tests are run.'''
        # Remove temporary directory and all files contained therein:
        files = glob.glob(os.path.join(cls.tmpdir, '*'))
        for f in files:
            os.remove(f)
        os.rmdir(cls.tmpdir)
    
    def test_standard(self):
        outpdf = os.path.join(self.tmpdir, 'sbftest1.pdf')
        sbf_results = stisblazefix.fluxfix(self.x1ds, outpdf)
        # Check that output files have nearly identical FLUX and ERROR arrays:
        compare_results(self.x1fs, self.x1fs_benchmark)
        # Check that output PDF exists and isn't empty:
        assert os.stat(outpdf).st_size != 0
    
    def test_nxplot(self):
        outpdf = os.path.join(self.tmpdir, 'sbftest2.pdf')
        sbf_results = stisblazefix.fluxfix(self.x1ds, outpdf, nxplot=2)
        # Check that output files have nearly identical FLUX and ERROR arrays:
        compare_results(self.x1fs, self.x1fs_benchmark)
        # Check that output PDF exists and isn't empty:
        assert os.stat(outpdf).st_size != 0
    
    def test_noiterate(self):
        outpdf = os.path.join(self.tmpdir, 'sbftest3.pdf')
        sbf_results = stisblazefix.fluxfix(self.x1d_noiterate, outpdf, guess=(20.0, 0.2), iterate=False)
        # Check that output files have nearly identical FLUX and ERROR arrays:
        compare_results(self.x1f_noiterate, self.x1f_noiterate_benchmark)
        # Check that output PDF exists and isn't empty:
        assert os.stat(outpdf).st_size != 0

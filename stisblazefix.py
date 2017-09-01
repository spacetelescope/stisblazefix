#Created by Malinda Baer on June 20, 2017
#Modified by M. Baer, July/August 2017
#Modified by C. Proffitt, August/September, 2017
#based on IDL routines written by Charles Proffitt in Jan 2017
#
#This module is written for Python 3.6.1,
#although limited testing using Python 2.7.13 shows no problems.
#
#
#This module requires:
#numpy, scipy, astropy, matplotlib, lmfit, time

# LMFIT is not in the standard astroconda distribution,
# but can be added to an anaconda environment by first
# activating the desired environment and then doing
# conda install -c conda-forge lmfit
#
#NOTE: requires at least numpy 1.13, bugs occur with numpy 1.12
#
#Import using the command "import stisblazefix"
#

'''This module contains a variety of functions to correct the blaze function in HST STIS
echelle modes. It is intended for use with x1d fits files. Most users will be interested 
in the fluxfix function.

#This module contains required functions for the following scripts:
fluxcorrect takes a shift to the blaze function and recalculates the flux and error.
residcalc takes an echelle spectrum and calculates the flux residuals for the overlapping region.
generateplot takes an old and corrected spectrum and generates a diagnostic plot.
residfunc is a wrapper for lmfit minimizer.
findshift calculates the shift to the blaze function that best aligns the spectrum.
fluxfix takes a list of x1d fits files and generates corrected x1f files and diagnostic plots.
plotblaze plots the sensitivity curves for an extracted spectra.
'''

from astropy.io import fits
import numpy as np
from numpy import ma
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.text as txt
from lmfit import Parameters, Minimizer, conf_interval, minimize, printfuncs
import time
import ntpath


def fluxcorrect(filedata, pixshift):
    '''Recalculate and return corrected flux and error vectors, based on shifted blaze function.
    
    filedata is the data attribute of an x1d fits file.
    
    pixshift is a 1D vector with length equal to the number of orders in the echelle spectrum.
    It contains the shift of the blaze function in pixels as a function of relative spectral order.
    '''

    origblaze = np.divide(filedata['net'], filedata['flux'])#back out original sensitivity
    shape = np.shape(origblaze)
    newblaze = np.zeros(shape)
    order = 0
    pixnos = np.arange(shape[1])#shape[1] should always be 1024 for STIS
    while order < shape[0]:#number of orders
        oldblaze=origblaze[order]
        isgood=np.where(np.isfinite(oldblaze))
        f = interpolate.interp1d(pixnos[isgood[0]], oldblaze[isgood[0]], fill_value='extrapolate')#
        newblaze[order] = f(pixnos - pixshift[order])#shifts the blaze function
        order += 1
    newflux = np.divide(filedata['net'], newblaze)
    #newerr = np.divide(filedata['net_error'], newblaze)  # need to handle case where net_error column is missing
    newerr = np.divide(filedata['error'],filedata['flux'])*newflux
    isbad = np.where(np.isnan(newerr) | np.isnan(newerr))
    newerr[isbad] = 0.0
    return (newflux, newerr)

def residcalc(filedata, flux=None, err=None, dq=None, ntrim=5):
    '''Calculate and return the flux overlap residuals and weighted error.
    
    Flux residuals are calculated by summing the flux in the overlapping region in adjacent orders,
    taking the ratio, and subtracting from one.
    
    filedata is the data attribute of an x1d fits file.
    
    kwargs:
    flux is a flux vector the same size as the flux in filedata.
    erre is an error vector the same size as the error rin filedata.
    If either flux or error is passed, they will be used to calculate the residuals.
    Otherwise, the flux and error from filedata will be used.
    ntrim is a cut made to the edges of the orders to avoid various edge problems. Defaults to 5.
    '''
    if flux is None:
        flux = filedata['flux']
    if err is None:
        err = filedata['error']
    if dq is None:
        dq = filedata['dq']
    wavelength = filedata['wavelength']
    shape = np.shape(flux)
    resids = np.zeros(shape[0] - 1)#one less residual than orders
    residerr = np.zeros(np.shape(resids))
    order = 0
    while order < (shape[0] - 1):
        #make cut before interpolating
        f = interpolate.interp1d(wavelength[order + 1], flux[order + 1], fill_value='extrapolate')
        g = interpolate.interp1d(wavelength[order + 1], err[order + 1], fill_value='extrapolate')
        q = interpolate.interp1d(wavelength[order + 1], dq[order + 1], kind='nearest',fill_value=4)
        overlap = np.where(wavelength[order] <= wavelength[order + 1][-1])
        #can do some trimming here
        f0 = flux[order][overlap]
        f1 = f(wavelength[order][overlap])#interpolate to the same wavelength bins
        g0 = err[order][overlap]
        g1 = g(wavelength[order][overlap])
        dq0 = dq[order][overlap]
        dq1 = np.round(q(wavelength[order][overlap])).astype(int)
        dqcomb = dq0 | dq1
        dqmask = mkdqmask(dqcomb)
        
        fluxsum0, fluxsum1 = 0., 0.#sum of flux in overlap regions for order and order+1 
        errsum0, errsum1 = 0., 0.#hold error sums
        #trim edges and outliers
        i=ntrim
        while i < np.shape(overlap)[1] - ntrim:
            fluxsum0 += dqmask[i] * f0[i]
            fluxsum1 += dqmask[i] * f1[i]
            errsum0 += dqmask[i] * g0[i] **2
            errsum1 += dqmask[i] * g1[i] **2
            i += 1
        if((fluxsum1 != 0) & (fluxsum0 !=0)):
            resids[order] = 1 - fluxsum0/fluxsum1
            residerr[order] = abs(1 - resids[order]) * np.sqrt(errsum0/(fluxsum0**2) + errsum1/(fluxsum1**2))
        else:
            resids[order] = 0
            residerr[order] = 1
        order += 1
    # next need to identify outliers in the residual plot and up errors to deweight these
    ordnum=np.arange(shape[0]-1)
    pcoff = np.polyfit(ordnum[4:-4],resids[4:-4],1,w=1/residerr[4:-4])
    residoff=pcoff[1]+pcoff[0]*ordnum - resids
    siglim = 3.5 + 0*residerr
    # apply more sensitive threshold to orders near edges of detectors
    siglim[0:4] = 2.0
    siglim[-5:-1] = 2.0 
    ibad = np.where(abs(residoff) > siglim*residerr + 0.02)    
    residerr[ibad] = 1 + residerr[ibad]
    
    return (resids, residerr)

def generateplot(origdata, newflux, newerr, pixshift, oldresids=None, olderr=None, ntrim=5):
    '''Generate a diagnostic plot for a corrected spectrum. 
    
    Plot spectrum and residuals before and after correction to blaze function, 
    and pixshift vs relative spectral order.
    Return a figure object.
    
    origdata is the data attribute of an x1d fits file. This should contain the original flux.
    newflux contains the corrected flux.
    newerr contains the corrected error.
    pixshift is a 1D vector with length equal to the number of orders in the echelle spectrum.
    
    kwargs:
    oldresids contains the residuals for the uncorrected data.
    olderr contains the error in the residuals for the uncorrected data.
    If neither is passed, they will be calculated from origdata.
    ntrim is a cut made to the edges of the orders to avoid various edge problems. Defaults to 5.
    '''
    origflux = origdata['flux']
    wavelength = origdata['wavelength']
    shape = np.shape(origflux)#(no. orders, no. pixels)
    yrange=cliprange(origflux,fraclim=0.02,pad=0.125)
        
    
    if oldresids is None or olderr is None:
        old = residcalc(origdata)
        if oldresids is None:
            oldresids = old[0]
        if olderr is None:
            olderr=old[1]
    newresids, newresiderr = residcalc(origdata, flux=newflux, err=newerr)
    
    fig = plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[5,2])

    oldspec = plt.subplot(gs[0])#spectrum before correction
    order = 0
    while order < shape[0]:
        plt.plot(wavelength[order][ntrim:-(ntrim+1)], origflux[order][ntrim:-(ntrim+1)], lw=0.1, alpha=0.3)
        order += 1
    plt.title('Spectrum Before Correction')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    plt.axis([wavelength.min(),wavelength.max(),yrange[0],yrange[1]])
    
    newspec = plt.subplot(gs[2], sharex=oldspec, sharey=oldspec)#spectrum after correction
    order = 0
    while order < shape[0]:
        plt.plot(wavelength[order][ntrim:-(ntrim+1)], newflux[order][ntrim:-(ntrim+1)], lw=0.1, alpha=0.3)
        order += 1
    plt.title('Spectrum After Correction')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    plt.axis([wavelength.min(),wavelength.max(),yrange[0],yrange[1]])
    
    residplot = plt.subplot(gs[1])#residuals before and after correction vs spectral order
    before = plt.errorbar(np.arange(np.shape(oldresids)[0]), oldresids, yerr=olderr, fmt='o', markersize=3, alpha=0.6)
    after = plt.errorbar(np.arange(np.shape(newresids)[0]), newresids, yerr=newresiderr, fmt='o', markersize=3, alpha=0.6)
    plt.plot(np.arange(np.shape(newresids)[0]), np.zeros(np.shape(newresids)[0]), '-')
    plt.ylabel('1 - Flux Ratio')
    plt.xlabel('Relative Spectral Order')
    plt.title('Flux Overlap')
    plt.legend([before, after], ['Before Correction', 'After Correction'])
    plt.axis([shape[0]+2,-2,min([-3*np.median(np.abs(oldresids)),-0.02]),max([6*np.median(np.abs(oldresids)),0.04])])
    
    pixplot = plt.subplot(gs[3], sharex=residplot)#graph of pixshift v spectral order
    plt.plot(pixshift, 'o', markersize=2)
    plt.title('Pixshift')
    plt.xlabel('Relative Spectral Order')
    plt.ylabel('Shift in Pixels')
    plt.axis([shape[0]+2,-2,min(pixshift)-1,max(pixshift)+1])

    #if params is not None:
    #    txt.text('a + b*x'#check that call

    return fig

def residfunc(pars, x, filedata):#function for lmfit - is there way to add loop for linear/0th/quadratic? - try adjusting number of parameters
    '''Wrapper for lmfit minimizer. 
    
    Return weighted residuals for a given pixshift.
    
    pars is a tuple containing lmfit parameters.
    x is the variable, in this case relative spectral order.
    filedata is the data attribute of an x1d fits file.
    '''
    a = pars['a'].value
    b = pars['b'].value
    pixshift = a + b*x
    new = fluxcorrect(filedata, pixshift)
    newflux = new[0]
    newerr = new[1]
    temp = residcalc(filedata, flux=newflux, err=newerr)#(resid, residerr)
    wresids = np.abs(np.divide(temp[0], temp[1]))#weighted resid = resid/sig
    return wresids
    
def findshift(filedata, guess, iterate=True):
    '''Find pixshift that best aligns spectral orders.
    
    Use lmfit to calculate the pixshift (to apply to the blaze function) as a linear 
    function of relative spectral order that minimizes the flux overlap residuals
    as calculated in residcalc.
    
    Return pixshift, the error in the pixshift, and the final parameters.
    
    filedata is the data attribute of an x1d fits file.
    guess is a tuple containing the starting parameters for the fit.
    iterate is a Boolean that determines whether the function iterates to find the best fit.
    '''
    x = np.arange(np.shape(filedata)[0])
    params = Parameters()
    params.add('a', value=guess[0])
    params.add('b', value=guess[1])
    if not(iterate):
        params['a'].vary, params['b'].vary = False, False
    min = minimize(residfunc, params, args=(x, filedata))
    #include diagnostics and throw warnings
    a = (min.params['a'].value, min.params['a'].stderr)
    b = (min.params['b'].value, min.params['b'].stderr)
    pixshift = a[0] + (b[0])*x
    pixerr = np.sqrt(a[1]**2 + (x * b[1])**2)
    return (pixshift, (a[0], b[0]), (a[1], b[1]))
    

def fluxfix(files, pdfname, guess=None, iterate=True, **kwargs):#add optional arguments for plotting, eg. files is a list of x1d fits spectra
    '''Corrects STIS echelle spectra by aligning the sensitvity function to 
    compensate for shifts in the blaze function.
    
    This routine will iterate over a list of STIS echelle x1d files and for each exposure
    in each file it will find the shifts of the sensitivy curves for the spectral orders 
    that maximize the flux consistency in the overlap between orders. Corrected flux and 
    error vectors are then calculated and saved to new output files where '_x1d.fits' is 
    replaced with '_x1f.fits'.
    
    It assumes that amount by which the sensitivity curve for each spectral order 
    needs to be shifted is a linear function of the order number. The optimal shifts for 
    each exposure to minimize the flux inconsistency in the overlapping regions of the
    spectral orders are found. Each exposure is treated independently.
    
    Args:

        files (list of str): A list of file names corresponding to STIS echelle x1d files.
        All file names should end in '_x1d.fits' and should contain extracted and flux
        calibrated STIS echelle spectra output from CALSTIS.

        pdfname (str): Name for the output pdf file.
        
        guess (tuple of two floats): Contains a starting guess for the linear relation 
        used to shift the blaze function. Should be in the form (a, b), where the blaze 
        shift for each spectral order x will be a + b*x. Defaults to None which then
        substitutes a suitable guess supplied in the code.

        Iterate (logical): When set to True the code will iterate to find the values for
        a & b that minimimize the inconsistency in the flux ovelap regions. When set to 
        False, the code will use the values supplied in Guess without iterating. Defaults
        to True.
    
    Returns:
       An array of tuples, one for each exposure found in the x1d files. Each tuple
       contains:
          - An array containing the final shift value in pixels for each spectral order in
          the file.
          - A two element tuple containing the (a, b) coefficients giving the offset of
          the lowest spectral order and the change in offset per order. The shift array
          above should = a + b*i for each element i in the array.
          - A two element tuple containing the formal errors in a and b found from the 
          minimization routine.
          - A string containing the name of the input file
          - An integer containing the extension number of the input file from which the
          exposure was taken.
    
    Produces:
       A new copy of each x1d file with the fluxes and errors corrected for the new 
       shift of the sensitivity vector. The new files replace 'x1d' in the file names
       with 'x1f'.
       A multipage pdf file with diagnostic plots for each exposure. 
    '''
    if guess is None:
        guess=(1, 0.5)
    pdf = PdfPages(pdfname)
    outline = None
    for filename in files:
        if(filename.find('_x1d.fits') >= 0):
            basename=filename[0:filename.find('_x1d.fits')]
        else:
            basename=filename[0:filename.find('.fit')]
        file = fits.open(filename)
        hdr = file[0].header
        extno = len(file)
        i = 1
        while i < extno:#for i, extension in enumerate(file, start=1):#while i < extno:
            filedata = file[i].data
            hdri = file[i].header
            shift = findshift(filedata, guess, iterate=iterate)
            pixshift, params, paramerr = shift[0], shift[1], shift[2]
            new = fluxcorrect(filedata, pixshift)
            newflux, newerr = new[0], new[1]
            graph = generateplot(filedata, newflux, newerr, pixshift)
            plt.suptitle(filename[:-9] + ' Extension: ' + str(i))
            plt.suptitle(hdr['rootname']+' ext. '+str(i)+', '+hdr['targname']+', '+hdr['opt_elem']+' '+str(hdr['CENWAVE'])+' '+hdr['propaper']+', '+'{:.2f}'.format(hdri['exptime'])+'s, '+hdri['date-obs']+' '+hdri['time-obs'] )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig()
            file[i].data['flux'] = newflux
            file[i].data['error'] = newerr
            outdata=shift + (ntpath.basename(filename),i,)
            if(outline == None):
                outline=[outdata]
            else:
                outline.append(outdata)
            i += 1
        hdr['comment'] = 'Spectra corrected using stisfix on ' + time.strftime('%d/%m/%Y')
        file.writeto(basename + '_x1f.fits',clobber=True)
        file.close()
    pdf.close()
    return outline

def plotblaze(filename, pdfname, ntrim=7):
    '''Plot the blaze function for each order of a STIS echelle spectrum.'''
    pdf = PdfPages(pdfname)
    file = fits.open(filename)
    fig = plt.figure()
    extno = len(file)
    i = 1
    while i < extno:
        filedata = file[i].data
        wavelength = filedata['wavelength']
        flux = filedata['flux']
        net = filedata['net']
        blaze = net/flux
        shape = np.shape(wavelength)

        order = 0
        while order < shape[0]:
            plt.plot(wavelength[order], blaze[order], lw=0.05)
            order += 1
        plt.title('Blaze Function')
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Relative Intensity')
        plt.suptitle(filename[:-9] + ' Extension: ' + str(i))
        pdf.savefig()
        i += 1
    file.close()
    pdf.close()

def cliprange(vector, fraclim=0.02, pad=0.125):
    ''' For a vector first find the limits, v_0 and v_1, on the value that 
    includes from fraclim to 1-fraclim of the pixels.
    Then expand that range on both the lower and upper end by an amount pad*(v_1-v_0)
    '''
    hist=np.histogram(vector,bins='auto')
    cdist=np.cumsum(hist[0])
    cdist=1.0*cdist/cdist.max()
    centers=0.5*(hist[1][1:]+hist[1][:-1])
    intfunc=interpolate.interp1d(cdist,centers,fill_value='extrapolate')
    newrange=intfunc([fraclim,1-fraclim])
    fullrange=newrange[-1]-newrange[0]
    return newrange+pad*np.array([-fullrange,fullrange])
        
def mkdqmask(dq,sdqflags=2604):
    '''Return mask vector set to zero wherever any bit set in the dq vector matches 
    any bit set in sdqflags and to one elsewhere.
    Default sdqflags=4+8+32+512+2048 = 2604
    Detector problem + data masked + large blemish + calibration defect + bad background
    '''
    mask=0*dq+1.0
    mask[np.where((dq & sdqflags) > 0)] = 0
    return mask

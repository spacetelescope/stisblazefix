#!/usr/bin/env python
#simple script to run stisblazefix.fluxfix from the command line
#This will try to run the correction on all x1d files in the working directory
#change the path below to point to your version of the module if it isn't already in your permanent path
#also needs access to all libraries loaded by stisblazefix.py

import os,fnmatch,sys
sys.path.append('/YourPathToModule/stisblazefix/')

import stisblazefix as sbf

x1dlist=fnmatch.filter(os.listdir('.'), '*x1d.fits')

outdiclist=sbf.fluxfix(x1dlist,'allfiles.pdf')

# alternate to process a single input file, but with multi-page before/after plots
#outdicsingle=sbf.fluxfix(x1dlist[0],'singlefile.pdf',nxplot=5)




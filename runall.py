#!/usr/bin/env python
#simple script to run stisblazefix.fluxfix from the command line
#This will try to run the correction on all x1d files in the working directory
#change the path below to point to your version of the script!
#also needs access to all libraries loaded in stisblazefix.py

import os,fnmatch,sys
sys.path.append('/YourPathToModule/stisblazefix/')

import stisblazefix as sbf

x1dlist=fnmatch.filter(os.listdir('.'), '*x1d.fits')

outdiclist=sbf.fluxfix(x1dlist,'all.pdf')




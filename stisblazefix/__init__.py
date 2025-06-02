try:
    from ._version import __version__
except ImportError:
    __version__ = 'unknown'
from .stisblazefix import fluxfix, fluxcorrect, generateplot, genexplot, findshift, \
    residcalc, residfunc, plotblaze, cliprange, mkdqmask

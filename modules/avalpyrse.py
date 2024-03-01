import numpy
import scipy.io
import glob
import warnings
import matplotlib.pyplot as plt
from modules.helper_func_class import structtype
#import os

def lnbin(x,nBins,normalize_by_sum=True):
    x = numpy.asarray(x)
    xx = x[x>0.0]
    if xx.size == 0:
        warnings.warn('WARNING: all x are less than zero, taking absolute value of x')
        x = numpy.abs(x)
    else:
        x = xx
    xmax = numpy.log(numpy.max(x))#numpy.ceil(numpy.log(numpy.max(x)) )
    xmin = numpy.log(numpy.min(x))#numpy.floor(numpy.log(numpy.min(x)))
    # log-scaled bins
    bin_edges = numpy.logspace(xmin, xmax, nBins+1, base=numpy.exp(1))
    bin_edges[0] = 0
    widths = numpy.diff(bin_edges)
    # Calculate histogram
    hist = numpy.histogram(x, bins=bin_edges)
    # normalize by bin width
    hist_norm = (hist[0]/widths)/len(x)
    midpts = bin_edges[:-1] + widths/2.0
    midpts = midpts[hist_norm>0]
    hist_norm = hist_norm[hist_norm>0]
    if normalize_by_sum:
        hist_norm = hist_norm / numpy.sum(hist_norm)
    return midpts,hist_norm

def calc_avalanche_dist(ST,nBins_S=30,nBins_T=None):
    """
    ST -> structtype with fields .S = aval size, and .T = aval duration
          or tuple where ST[0] = aval size, ST[1] = duration
    """
    nBins_T = nBins_S if nBins_T is None else nBins_T
    if type(ST) is structtype:
        s,P_s = lnbin(ST.S,nBins_S,normalize_by_sum=True)
        T,P_T = lnbin(ST.T,nBins_T,normalize_by_sum=True)
    else:
        s,P_s = lnbin(ST[0],nBins_S,normalize_by_sum=True)
        T,P_T = lnbin(ST[1],nBins_T,normalize_by_sum=True)
    return structtype(x=s,P=P_s), structtype(x=T,P=P_T)

def calc_distribution(x,nBins,normalize_by_sum=True):
    P,x_edges = numpy.histogram(x,bins=nBins,density=True)
    if normalize_by_sum:
        P = P / numpy.sum(P)
    return structtype(x=x_edges[:-1],P=P)

def calc_avalanche_avg_size_dur(ST):
    """ returns the average size of an avalanche for each duration
    ST -> struct with fields S and T or tuple where ST[0] = size, ST[1] = duration
    """
    if type(ST) is structtype:
        uT = numpy.unique(ST.T)
        get_T = lambda: ST.T
        get_S = lambda: ST.S
    else:
        uT = numpy.unique(ST[1])
        get_T = lambda: ST[1]
        get_S = lambda: ST[0]
    sAvg = numpy.zeros(uT.shape)
    sStd = numpy.zeros(uT.shape)
    for i,T in enumerate(uT):
        k = numpy.nonzero(get_T()==T)[0]
        sAvg[i] = numpy.nanmean(get_S()[k])
        sStd[i] = numpy.nanstd(get_S()[k])
    return structtype(S_avg=sAvg,S_std=sStd,T=uT)


def calc_avalanche_size_dur(rho,thresh=0.0,dt=1.0,subtract_thresh_from_size=True):
    """
    calculates avalanche sizes and duration according to the PRE by Villegas et al 2019 PRE
    rho    -> # of active sites at each time step
    thresh -> threshold to apply to rho time series to extract avalanche sizes and duration
    dt     -> time scale (if 1 time step = 1 ms, then use dt = 1e-3)
    subtract_thresh_from_size -> if True, subtracts the threshold from the avalanche size (recommended by Villegas et al to keep correct scaling)

    returns
    S,T -> avalanche sizes, avalanche duration (each as numpy.ndarray)
    """
    if not(type(rho) is numpy.ndarray):
        rho = numpy.asarray(rho)
    f = (rho[:-1]-thresh)*(rho[1:]-thresh)
    k_start = numpy.nonzero(numpy.logical_and( f <= 0 , rho[:-1]<=thresh))[0] + 1 # index of the start of an avalanche
    k_end = numpy.nonzero(numpy.logical_and( f <= 0 , rho[1:]<=thresh))[0] + 1 # index of the end of an avalanche
    if k_end[0] <= k_start[0]:
        k_end = k_end[1:]
    if k_start[-1] > k_end[-1]:
        k_start = k_start[:-1]
    if (len(k_start) > 0) and (len(k_end) > 0):
        th = thresh if subtract_thresh_from_size else 0.0
        aval_size = numpy.asarray([ numpy.sum(rho[a:b]-th) for a,b in zip(k_start,k_end) ])
        aval_dur = (k_end - k_start) * dt
        return structtype(S=aval_size[aval_size>=1.0],T=aval_dur[aval_size>=1.0])
    warnings.warn('no avalanches found with given threshold')
    return structtype(S=[],T=[])

def is_scalar(v):
    if not(type(v) is numpy.ndarray):
        v = numpy.asarray(v)
    s = numpy.sum(numpy.asarray(v.shape)==1)
    return s==len(v.shape)

def is_vector(v):
    if not(type(v) is numpy.ndarray):
        v = numpy.asarray(v)
    s = numpy.sum(numpy.asarray(v.shape)==1)
    return (s==len(v.shape)) or (s == (len(v.shape)-1))

def loadmat_ignoreshape(fn):
    """
    calls scipy.io.loadmat
    and reduces all singleton dimensions for each data variable in the file
    """
    d = scipy.io.loadmat(fn)
    for k in d:
        if (k[0:2] == k[-2:]) and (k[0:2] == '__'):
            continue
        if is_scalar(d[k]):
            d[k] = d[k].flatten()[0]
        elif is_vector(d[k]):
            d[k] = d[k].flatten()
    return d

def loadByPattern(filePtrn,sortParamName=None,loadFunc=loadmat_ignoreshape):
    fn = glob.glob(filePtrn)#os.path.join(filePtrn))
    if len(fn) == 0:
        return []
    else:
        d = []
        for f in fn:
            d.append(loadFunc(f))
        if not(sortParamName is None):
            if sortParamName in d[0].keys():
                p = [ s[sortParamName] for s in d ]
                d = [x for _,x in sorted(zip(p,d),key=lambda z:z[0])]
            else:
                warnings.warn('sortParamName does not exist in the loaded files')
        return d
